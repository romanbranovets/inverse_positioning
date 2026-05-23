"""Shared trajectory encoder, vMF density, and physics path utilities."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .magentic_field import B
from .sphere_utils import normalize, tangent_basis
from .trajectory_sampler import EPS_B_STD, exp_map_sphere


def reconstruct_path_from_endpoint(endpoint, observed_tokens, pad_mask):
    """Differentiable forward path reconstruction ending at endpoint."""
    batch_size, token_count, _ = observed_tokens.shape
    positions_rev = [endpoint]
    u = endpoint
    for t in range(token_count - 2, -1, -1):
        active = ~pad_mask[:, t]
        du = observed_tokens[:, t, 6:8]
        e1, e2 = tangent_basis(u)
        back_step = -du[:, 0:1] * e1 - du[:, 1:2] * e2
        previous_u = exp_map_sphere(u, back_step)
        u = torch.where(active[:, None], previous_u, u)
        positions_rev.append(u)
    path = torch.stack(list(reversed(positions_rev)), dim=1)
    valid = (~pad_mask).unsqueeze(-1)
    return path * valid + endpoint.unsqueeze(1) * (~valid)


def path_magnetic_residual(endpoint, observed_tokens, pad_mask):
    path = reconstruct_path_from_endpoint(endpoint, observed_tokens, pad_mask)
    predicted_B = B(path.reshape(-1, 3)).reshape(path.shape)
    residual = (observed_tokens[:, :, :3] - predicted_B) / EPS_B_STD
    valid = (~pad_mask).unsqueeze(-1)
    return residual * valid


def physics_consistency_loss(endpoint, observed_tokens, pad_mask):
    residual = path_magnetic_residual(endpoint, observed_tokens, pad_mask)
    valid = (~pad_mask).unsqueeze(-1)
    return (residual.square() * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)


def vmf_log_prob(x, mu, kappa):
    """Log density of 3D von Mises-Fisher. x, mu: [..., 3]; kappa: [...]."""
    x = normalize(x)
    mu = normalize(mu)
    kappa = kappa.clamp_min(1e-4)
    dot = (x * mu).sum(dim=-1).clamp(-1.0, 1.0)
    log_c = torch.log(kappa) - math.log(4.0 * math.pi) - torch.log(
        torch.sinh(kappa).clamp_min(1e-8)
    )
    return log_c + kappa * dot


def vmf_mixture_log_prob(endpoint, logits, mu, kappa):
    log_w = F.log_softmax(logits, dim=-1)
    log_comp = vmf_log_prob(
        endpoint[:, None, :],
        mu,
        kappa,
    )
    return torch.logsumexp(log_w + log_comp, dim=-1)


class RefineBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.03, step_scale_m=250_000.0):
        super().__init__()
        self.step_scale_m = step_scale_m
        self.endpoint_proj = nn.Linear(3, d_model)
        self.residual_proj = nn.Linear(3, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 2),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, endpoint, token_states, residual, pad_mask):
        endpoint_emb = self.endpoint_proj(normalize(endpoint))
        residual_emb = self.residual_proj(residual)
        token_context = token_states + residual_emb
        query = self.query_norm(endpoint_emb + token_context.mean(dim=1))
        attended, _ = self.cross_attn(
            query[:, None, :],
            token_context,
            token_context,
            key_padding_mask=pad_mask,
        )
        delta_2d = self.step_scale_m * torch.tanh(self.ff(attended.squeeze(1)))
        e1, e2 = tangent_basis(endpoint)
        tangent = delta_2d[:, 0:1] * e1 + delta_2d[:, 1:2] * e2
        return exp_map_sphere(endpoint, tangent), delta_2d


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        feature_mean,
        feature_std,
        d_model=384,
        n_heads=4,
        n_layers=4,
        max_tokens=17,
        dropout=0.03,
    ):
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_std", feature_std.detach().clone())
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, pad_mask):
        scaled = (x - self.feature_mean) / self.feature_std
        tokens = self.input_projection(scaled)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return encoded, pooled
