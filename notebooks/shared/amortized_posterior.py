"""Amortized conditional posterior on the sphere for fast NWJ audit scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .density_common import (
    TrajectoryEncoder,
    physics_consistency_loss,
    vmf_mixture_log_prob,
)
from .sphere_utils import normalize, sample_uniform_sphere, tangent_basis


@dataclass
class PosteriorParams:
    logits: torch.Tensor
    mu: torch.Tensor
    kappa: torch.Tensor


class PosteriorHead(nn.Module):
    """Maps trajectory context to a vMF mixture on the sphere."""

    def __init__(
        self,
        d_model,
        n_components=16,
        dropout=0.03,
        tangent_scale=0.35,
        kappa_floor=20.0,
        kappa_init=50.0,
    ):
        super().__init__()
        self.n_components = n_components
        self.tangent_scale = tangent_scale
        self.kappa_floor = kappa_floor
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + 3),
            nn.Linear(d_model + 3, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, n_components * 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            init_log_kappa = math.log(max(kappa_init - kappa_floor, 1.0))
            for component in range(n_components):
                self.net[-1].bias[component * 4 + 2] = init_log_kappa

    def forward(self, pooled, anchor):
        batch_size = pooled.shape[0]
        raw = self.net(torch.cat([pooled, anchor], dim=-1))
        raw = raw.view(batch_size, self.n_components, 4)
        e1, e2 = tangent_basis(anchor)
        tangent_2d = self.tangent_scale * torch.tanh(raw[..., :2])
        offset = (
            tangent_2d[..., 0:1] * e1[:, None, :]
            + tangent_2d[..., 1:2] * e2[:, None, :]
        )
        mu = normalize(anchor[:, None, :] + offset)
        kappa = F.softplus(raw[..., 2]) + self.kappa_floor
        logits = raw[..., 3]
        return PosteriorParams(logits=logits, mu=mu, kappa=kappa)


class AmortizedPosterior(nn.Module):
    """Encode trajectory once, score endpoints with a closed-form vMF mixture."""

    def __init__(
        self,
        feature_dim,
        feature_mean,
        feature_std,
        d_model=384,
        n_heads=4,
        n_encoder_layers=4,
        n_components=16,
        max_tokens=17,
        dropout=0.03,
        kappa_floor=20.0,
        kappa_init=50.0,
    ):
        super().__init__()
        self.encoder = TrajectoryEncoder(
            feature_dim=feature_dim,
            feature_mean=feature_mean,
            feature_std=feature_std,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            max_tokens=max_tokens,
            dropout=dropout,
        )
        self.head = PosteriorHead(
            d_model=d_model,
            n_components=n_components,
            dropout=dropout,
            kappa_floor=kappa_floor,
            kappa_init=kappa_init,
        )

    def trajectory_anchor(self, x, pad_mask):
        last_idx = (~pad_mask).sum(dim=1).sub(1).clamp_min(0)
        last_b = x[torch.arange(x.shape[0], device=x.device), last_idx, :3]
        return normalize(last_b)

    def encode(self, x, pad_mask):
        _, pooled = self.encoder(x, pad_mask)
        anchor = self.trajectory_anchor(x, pad_mask)
        return self.head(pooled, anchor)

    def mean_direction(self, params):
        weights = F.softmax(params.logits, dim=-1)
        return normalize((weights.unsqueeze(-1) * params.mu).sum(dim=1))

    def mean_direction_from_batch(self, x, pad_mask):
        return self.mean_direction(self.encode(x, pad_mask))

    def log_prob_from_params(self, params, endpoint):
        return vmf_mixture_log_prob(
            endpoint, params.logits, params.mu, params.kappa
        )

    def log_prob_from_params_batched(self, params, endpoints, chunk_size=None):
        """Score many endpoints per trajectory without re-encoding.

        params: PosteriorParams with batch B
        endpoints: [B, N, 3] or [M, 3] with implicit batch mapping via row_index
        """
        if endpoints.dim() == 2:
            return self.log_prob_from_params(params, endpoints)

        batch_size, negative_count, _ = endpoints.shape
        flat_endpoints = endpoints.reshape(-1, 3)
        row_index = torch.arange(batch_size, device=endpoints.device).repeat_interleave(
            negative_count
        )
        if chunk_size is None:
            return self._log_prob_indexed(params, row_index, flat_endpoints)

        chunks = []
        for start in range(0, flat_endpoints.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_endpoints.shape[0])
            chunks.append(
                self._log_prob_indexed(
                    params,
                    row_index[start:stop],
                    flat_endpoints[start:stop],
                )
            )
        return torch.cat(chunks, dim=0)

    def _log_prob_indexed(self, params, row_index, endpoints):
        logits = params.logits.index_select(0, row_index)
        mu = params.mu.index_select(0, row_index)
        kappa = params.kappa.index_select(0, row_index)
        return vmf_mixture_log_prob(endpoints, logits, mu, kappa)

    def log_prob(self, x, pad_mask, endpoint):
        return self.log_prob_from_params(self.encode(x, pad_mask), endpoint)

    def forward(self, x, pad_mask):
        return self.encode(x, pad_mask)


def contrastive_log_probs(
    model,
    x,
    pad_mask,
    endpoint,
    negative_count,
    negative_chunk_size=None,
):
    """Fast NWJ scoring: one encode per batch, cheap vMF queries for negatives."""
    params = model.encode(x, pad_mask)
    positive_logit = model.log_prob_from_params(params, endpoint)
    negatives = sample_uniform_sphere(
        endpoint.shape[0] * negative_count, device=endpoint.device
    ).reshape(endpoint.shape[0], negative_count, 3)
    negative_logit = model.log_prob_from_params_batched(
        params,
        negatives,
        chunk_size=negative_chunk_size,
    )
    return positive_logit, negative_logit.reshape(-1)


def bounded_logits(logits, clamp_abs):
    return logits.clamp(-clamp_abs, clamp_abs)


def nwj_training_nats(positive_logits, negative_logits, clamp_abs, exp_clamp):
    positive_t = bounded_logits(positive_logits, clamp_abs)
    negative_t = bounded_logits(negative_logits, clamp_abs)
    negative_exp = torch.exp((negative_t - 1.0).clamp(max=exp_clamp))
    return positive_t.mean() - negative_exp.mean()


def nwj_training_loss(
    positive_logits,
    negative_logits,
    clamp_abs,
    exp_clamp,
):
    return -nwj_training_nats(positive_logits, negative_logits, clamp_abs, exp_clamp)


def kappa_regularizer(params, kappa_target):
    return F.relu(kappa_target - params.kappa).mean()


def ap_training_loss(
    model,
    x,
    pad_mask,
    endpoint,
    negative_count=32,
    clamp_abs=22.0,
    exp_clamp=8.0,
    nll_weight=1.0,
    nwj_weight=0.5,
    physics_weight=0.1,
    kappa_reg_weight=0.01,
    kappa_target=30.0,
):
    params = model.encode(x, pad_mask)
    nll = -model.log_prob_from_params(params, endpoint).mean()
    mean_endpoint = model.mean_direction(params)
    physics = physics_consistency_loss(mean_endpoint, x, pad_mask).mean()
    kappa_reg = kappa_regularizer(params, kappa_target)

    negatives = sample_uniform_sphere(
        endpoint.shape[0] * negative_count, device=endpoint.device
    ).reshape(endpoint.shape[0], negative_count, 3)
    positive_logit = model.log_prob_from_params(params, endpoint)
    negative_logit = model.log_prob_from_params_batched(params, negatives)
    nwj = nwj_training_loss(
        positive_logit,
        negative_logit.reshape(-1),
        clamp_abs=clamp_abs,
        exp_clamp=exp_clamp,
    )

    objective = (
        nll_weight * nll
        + nwj_weight * nwj
        + physics_weight * physics
        + kappa_reg_weight * kappa_reg
    )
    return objective, dict(
        nll=nll.detach(),
        nwj=nwj.detach(),
        physics=physics.detach(),
        kappa_reg=kappa_reg.detach(),
        mean_kappa=params.kappa.mean().detach(),
        positive_logit=positive_logit.detach(),
        negative_logit=negative_logit.reshape(-1).detach(),
    )
