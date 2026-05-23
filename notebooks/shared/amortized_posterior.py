"""Amortized conditional posterior on the sphere for fast NWJ audit scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .density_common import (
    RefineBlock,
    TrajectoryEncoder,
    path_magnetic_residual,
    physics_consistency_loss,
    vmf_mixture_log_prob,
)
from .sphere_utils import normalize, sample_uniform_sphere, tangent_basis


@dataclass
class PosteriorParams:
    logits: torch.Tensor
    mu: torch.Tensor
    kappa: torch.Tensor
    refined_mu: torch.Tensor
    sharpness: torch.Tensor


class PosteriorHead(nn.Module):
    """Maps refined trajectory context to a vMF mixture on the sphere."""

    def __init__(
        self,
        d_model,
        n_components=16,
        dropout=0.03,
        tangent_scale=0.55,
        kappa_floor=5.0,
        kappa_init=15.0,
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
            nn.Linear(2 * d_model, n_components * 4 + 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            init_log_kappa = math.log(max(kappa_init - kappa_floor, 1.0))
            for component in range(n_components):
                self.net[-1].bias[component * 4 + 2] = init_log_kappa
            self.net[-1].bias[n_components * 4] = math.log(math.exp(8.0) - 1.0)

    def forward(self, pooled, anchor):
        batch_size = pooled.shape[0]
        raw = self.net(torch.cat([pooled, anchor], dim=-1))
        raw_components = raw[:, : self.n_components * 4].view(
            batch_size, self.n_components, 4
        )
        sharpness = F.softplus(raw[:, self.n_components * 4 :]).squeeze(-1) + 1.0
        e1, e2 = tangent_basis(anchor)
        tangent_2d = self.tangent_scale * torch.tanh(raw_components[..., :2])
        offset = (
            tangent_2d[..., 0:1] * e1[:, None, :]
            + tangent_2d[..., 1:2] * e2[:, None, :]
        )
        mu = normalize(anchor[:, None, :] + offset)
        kappa = F.softplus(raw_components[..., 2]) + self.kappa_floor
        logits = raw_components[..., 3]
        return PosteriorParams(
            logits=logits,
            mu=mu,
            kappa=kappa,
            refined_mu=anchor,
            sharpness=sharpness,
        )


class AmortizedPosterior(nn.Module):
    """Encode trajectory once, score endpoints with hybrid vMF + alignment energy."""

    def __init__(
        self,
        feature_dim,
        feature_mean,
        feature_std,
        d_model=384,
        n_heads=4,
        n_encoder_layers=4,
        n_refine_steps=3,
        n_components=32,
        max_tokens=17,
        dropout=0.03,
        kappa_floor=5.0,
        kappa_init=15.0,
    ):
        super().__init__()
        self.n_refine_steps = n_refine_steps
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
        self.refine_blocks = nn.ModuleList(
            [
                RefineBlock(d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_refine_steps)
            ]
        )
        self.head = PosteriorHead(
            d_model=d_model,
            n_components=n_components,
            dropout=dropout,
            kappa_floor=kappa_floor,
            kappa_init=kappa_init,
        )

    def initial_endpoint(self, x, pad_mask):
        last_idx = (~pad_mask).sum(dim=1).sub(1).clamp_min(0)
        last_b = x[torch.arange(x.shape[0], device=x.device), last_idx, :3]
        return normalize(last_b)

    def encode(self, x, pad_mask, return_deltas=False):
        token_states, pooled = self.encoder(x, pad_mask)
        endpoint = self.initial_endpoint(x, pad_mask)
        deltas = []
        for block in self.refine_blocks:
            residual = path_magnetic_residual(endpoint, x, pad_mask)
            endpoint, delta = block(endpoint, token_states, residual, pad_mask)
            deltas.append(delta)
        params = self.head(pooled, endpoint)
        params = PosteriorParams(
            logits=params.logits,
            mu=params.mu,
            kappa=params.kappa,
            refined_mu=endpoint,
            sharpness=params.sharpness,
        )
        if return_deltas:
            return params, deltas
        return params

    def mean_direction(self, params):
        weights = F.softmax(params.logits, dim=-1)
        return normalize((weights.unsqueeze(-1) * params.mu).sum(dim=1))

    def mean_direction_from_batch(self, x, pad_mask):
        return self.mean_direction(self.encode(x, pad_mask))

    def density_log_prob(self, params, endpoint):
        return vmf_mixture_log_prob(
            endpoint, params.logits, params.mu, params.kappa
        )

    def alignment_boost(self, params, endpoint):
        endpoint = normalize(endpoint)
        dot = (endpoint * params.refined_mu).sum(dim=-1).clamp(-1.0, 1.0)
        gate = torch.sigmoid(20.0 * (dot - 0.85))
        return params.sharpness * gate

    def critic_logit(self, params, endpoint, include_boost=True):
        log_p = self.density_log_prob(params, endpoint)
        if not include_boost:
            return log_p
        return log_p + self.alignment_boost(params, endpoint)

    def log_prob_from_params(self, params, endpoint):
        return self.critic_logit(params, endpoint, include_boost=True)

    def log_prob_from_params_batched(self, params, endpoints, chunk_size=None, include_boost=True):
        if endpoints.dim() == 2:
            return self.critic_logit(params, endpoints, include_boost=include_boost)

        batch_size, negative_count, _ = endpoints.shape
        flat_endpoints = endpoints.reshape(-1, 3)
        row_index = torch.arange(batch_size, device=endpoints.device).repeat_interleave(
            negative_count
        )
        if chunk_size is None:
            return self._critic_indexed(
                params, row_index, flat_endpoints, include_boost=include_boost
            )

        chunks = []
        for start in range(0, flat_endpoints.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_endpoints.shape[0])
            chunks.append(
                self._critic_indexed(
                    params,
                    row_index[start:stop],
                    flat_endpoints[start:stop],
                    include_boost=include_boost,
                )
            )
        return torch.cat(chunks, dim=0)

    def _critic_indexed(self, params, row_index, endpoints, include_boost=True):
        indexed = PosteriorParams(
            logits=params.logits.index_select(0, row_index),
            mu=params.mu.index_select(0, row_index),
            kappa=params.kappa.index_select(0, row_index),
            refined_mu=params.refined_mu.index_select(0, row_index),
            sharpness=params.sharpness.index_select(0, row_index),
        )
        return self.critic_logit(indexed, endpoints, include_boost=include_boost)

    def log_prob(self, x, pad_mask, endpoint):
        return self.critic_logit(self.encode(x, pad_mask), endpoint)

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
    params = model.encode(x, pad_mask)
    positive_logit = model.critic_logit(params, endpoint)
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


def direction_loss(params, endpoint):
    mean_dir = F.normalize(
        (F.softmax(params.logits, dim=-1).unsqueeze(-1) * params.mu).sum(dim=1),
        dim=-1,
    )
    dot = (mean_dir * normalize(endpoint)).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - dot).mean()


def ap_training_loss(
    model,
    x,
    pad_mask,
    endpoint,
    epoch=1,
    nll_warmup_epochs=25,
    negative_count=64,
    clamp_abs=22.0,
    exp_clamp=8.0,
    nll_weight=1.0,
    nwj_weight=0.5,
    physics_weight=0.35,
    direction_weight=0.25,
    kappa_reg_weight=0.01,
    kappa_target=80.0,
    delta_weight=0.01,
):
    params, deltas = model.encode(x, pad_mask, return_deltas=True)
    density_nll = -model.density_log_prob(params, endpoint).mean()
    critic_nll = -model.critic_logit(
        params, endpoint, include_boost=boost_scale
    ).mean()
    mean_endpoint = model.mean_direction(params)
    physics = physics_consistency_loss(mean_endpoint, x, pad_mask).mean()
    direction = direction_loss(params, endpoint)
    kappa_reg = kappa_regularizer(params, kappa_target)
    delta_reg = (
        torch.stack(deltas, dim=0).square().mean()
        if deltas
        else torch.zeros((), device=x.device, dtype=x.dtype)
    )

    warmup = epoch <= nll_warmup_epochs
    nwj_scale = 0.0 if warmup else min(1.0, (epoch - nll_warmup_epochs) / 20.0)
    kappa_scale = 0.0 if warmup or epoch <= nll_warmup_epochs + 10 else 1.0
    boost_scale = epoch > nll_warmup_epochs + 15

    nwj = torch.tensor(0.0, device=x.device)
    positive_logit = model.critic_logit(params, endpoint, include_boost=boost_scale)
    negative_logit = positive_logit.detach()
    if nwj_scale > 0.0:
        negatives = sample_uniform_sphere(
            endpoint.shape[0] * negative_count, device=endpoint.device
        ).reshape(endpoint.shape[0], negative_count, 3)
        negative_logit = model.log_prob_from_params_batched(
            params, negatives, include_boost=boost_scale
        )
        nwj = nwj_training_loss(
            positive_logit,
            negative_logit.reshape(-1),
            clamp_abs=clamp_abs,
            exp_clamp=exp_clamp,
        )

    objective = (
        nll_weight * (0.5 * density_nll + 0.5 * critic_nll)
        + nwj_scale * nwj_weight * nwj
        + physics_weight * physics
        + direction_weight * direction
        + kappa_scale * kappa_reg_weight * kappa_reg
        + delta_weight * delta_reg
    )
    return objective, dict(
        nll=density_nll.detach(),
        critic_nll=critic_nll.detach(),
        nwj=nwj.detach(),
        physics=physics.detach(),
        direction=direction.detach(),
        kappa_reg=kappa_reg.detach(),
        mean_kappa=params.kappa.mean().detach(),
        mean_sharpness=params.sharpness.mean().detach(),
        positive_logit=positive_logit.detach(),
        negative_logit=negative_logit.reshape(-1).detach(),
        nwj_scale=float(nwj_scale),
    )
