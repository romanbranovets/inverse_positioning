"""Amortized physics-gated energy critic: encode once, score with lightweight unnormalized energy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .density_common import RefineBlock, TrajectoryEncoder, path_magnetic_residual
from .magentic_field import B, EARTH_RADIUS_M
from .sphere_utils import normalize, sample_uniform_sphere, tangent_basis
from .trajectory_sampler import exp_map_sphere


ENDPOINT_OFFSET_SCALE_M = 2_000_000.0
RESIDUAL_CLIP_SIGMA = 30.0


@dataclass
class TrajectoryContext:
    token_states: torch.Tensor
    pooled: torch.Tensor
    refined_mu: torch.Tensor


def log_map_sphere_2d_meters(base, target):
    dot = (target * base).sum(dim=-1).clamp(-1.0, 1.0)
    tangent = target - dot[:, None] * base
    sin_theta = tangent.norm(dim=-1).clamp_min(1e-12)
    theta = torch.atan2(sin_theta, dot)
    direction = tangent / sin_theta[:, None]
    e1, e2 = tangent_basis(base)
    v_m = EARTH_RADIUS_M * theta[:, None] * direction
    return torch.stack([(v_m * e1).sum(dim=-1), (v_m * e2).sum(dim=-1)], dim=-1)


def initial_endpoint_from_trajectory(x, pad_mask):
    last_idx = (~pad_mask).sum(dim=1).sub(1).clamp_min(0)
    last_b = x[torch.arange(x.shape[0], device=x.device), last_idx, :3]
    return normalize(last_b)


def endpoint_descriptor(refined_mu, endpoint):
    endpoint = normalize(endpoint)
    refined_mu = normalize(refined_mu)
    endpoint_b = B(endpoint)
    offset = log_map_sphere_2d_meters(refined_mu, endpoint) / ENDPOINT_OFFSET_SCALE_M
    dot = (endpoint * refined_mu).sum(dim=-1, keepdim=True)
    return torch.cat([endpoint, endpoint_b, offset, dot], dim=-1)


def slerp_sphere(base, target, t):
    base = normalize(base)
    target = normalize(target)
    dot = (base * target).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-6)
    t = t[..., None] if t.dim() == 1 else t
    weight_base = torch.sin((1.0 - t) * omega) / sin_omega
    weight_target = torch.sin(t * omega) / sin_omega
    return normalize(weight_base * base + weight_target * target)


def perturb_on_sphere(base, std_rad):
    base = normalize(base)
    e1, e2 = tangent_basis(base)
    coeffs = torch.randn(*base.shape[:-1], 2, device=base.device, dtype=base.dtype)
    tangent_m = std_rad * EARTH_RADIUS_M * (
        coeffs[..., 0:1] * e1 + coeffs[..., 1:2] * e2
    )
    return exp_map_sphere(base, tangent_m)


def summarize_path_physics(endpoint, observed_tokens, pad_mask):
    residual = path_magnetic_residual(endpoint, observed_tokens, pad_mask)
    valid = (~pad_mask).float()
    token_energy = torch.log1p(residual.square().sum(dim=-1))
    denom = valid.sum(dim=1).clamp_min(1.0)
    mean_energy = (token_energy * valid).sum(dim=1) / denom
    max_energy = token_energy.masked_fill(pad_mask, -torch.inf).amax(dim=1)
    var_energy = (
        ((token_energy - mean_energy[:, None]).square() * valid).sum(dim=1) / denom
    ).clamp_min(0.0)
    clipped = torch.clamp(residual / RESIDUAL_CLIP_SIGMA, -1.0, 1.0)
    mean_abs = (clipped.abs() * valid.unsqueeze(-1)).sum(dim=(1, 2)) / (
        denom * 3.0
    )
    return torch.stack([mean_energy, max_energy, var_energy.sqrt(), mean_abs], dim=-1)


def sample_hard_endpoints(refined_mu, true_endpoint, count, device, dtype):
    if count <= 0:
        return torch.empty(0, 3, device=device, dtype=dtype)

    refined_mu = normalize(refined_mu.reshape(-1, 3)[0]).expand(count, -1)
    true_endpoint = normalize(true_endpoint.reshape(-1, 3)[0]).expand(count, -1)

    n_near_mu = max(1, count // 3)
    n_slerp = max(1, count // 3)
    n_near_true = max(1, count - n_near_mu - n_slerp)

    near_mu = perturb_on_sphere(refined_mu[:n_near_mu], std_rad=0.08)
    t = 0.35 + 0.55 * torch.rand(n_slerp, device=device, dtype=dtype)
    slerp = slerp_sphere(refined_mu[:n_slerp], true_endpoint[:n_slerp], t)
    near_true = perturb_on_sphere(true_endpoint[:n_near_true], std_rad=0.05)

    points = torch.cat([near_mu, slerp, near_true], dim=0)
    if points.shape[0] < count:
        extra = sample_uniform_sphere(count - points.shape[0], device=device).to(dtype)
        points = torch.cat([points, extra], dim=0)
    return normalize(points[:count])


def sample_mixed_negatives(
    batch_size,
    negative_count,
    refined_mu,
    true_endpoint,
    hard_fraction=0.5,
    in_batch_count=0,
    batch_endpoints=None,
):
    device = true_endpoint.device
    dtype = true_endpoint.dtype
    hard_count = int(round(negative_count * hard_fraction))
    uniform_count = negative_count - hard_count

    pieces = []
    if uniform_count > 0:
        uniform = sample_uniform_sphere(batch_size * uniform_count, device=device).to(
            dtype
        )
        pieces.append(uniform.reshape(batch_size, uniform_count, 3))

    if hard_count > 0:
        hard = []
        for row in range(batch_size):
            hard.append(
                sample_hard_endpoints(
                    refined_mu[row : row + 1],
                    true_endpoint[row : row + 1],
                    hard_count,
                    device,
                    dtype,
                )
            )
        pieces.append(torch.stack(hard, dim=0))

    if in_batch_count > 0 and batch_endpoints is not None:
        perm = torch.randperm(batch_size, device=device)
        shifted = batch_endpoints[perm]
        same = (shifted - true_endpoint).norm(dim=-1) < 1e-5
        if same.any():
            shifted = torch.where(
                same[:, None],
                perturb_on_sphere(true_endpoint, std_rad=0.2),
                shifted,
            )
        pieces.append(
            shifted[:, None, :].expand(-1, in_batch_count, -1).reshape(
                batch_size, in_batch_count, 3
            )
        )

    return normalize(torch.cat(pieces, dim=1))


class FusionBlock(nn.Module):
    def __init__(self, d_model, dropout=0.05):
        super().__init__()
        fused_dim = 4 * d_model
        self.norm = nn.LayerNorm(fused_dim)
        self.gate = nn.Linear(fused_dim, fused_dim)
        self.ff = nn.Sequential(
            nn.Linear(fused_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, traj_h, attn_h, physics_h, endpoint_h):
        packed = torch.cat([traj_h, attn_h, physics_h, endpoint_h], dim=-1)
        gated = self.norm(packed) * torch.sigmoid(self.gate(packed))
        return traj_h + self.ff(gated)


class LightweightEnergyScorer(nn.Module):
    """Physics-gated unnormalized critic without pair-transformer cloning."""

    def __init__(self, d_model, n_heads=4, n_fusion_layers=2, dropout=0.05):
        super().__init__()
        self.endpoint_proj = nn.Sequential(
            nn.Linear(9, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.traj_proj = nn.Sequential(
            nn.LayerNorm(d_model + 3),
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.residual_token_proj = nn.Linear(3, d_model)

        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                for _ in range(2)
            ]
        )
        self.attn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        self.fusion = nn.ModuleList(
            [FusionBlock(d_model, dropout=dropout) for _ in range(n_fusion_layers)]
        )

        self.ctx_physics = nn.Sequential(
            nn.LayerNorm(d_model + 3),
            nn.Linear(d_model + 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.energy_scale = nn.Parameter(torch.tensor(2.5))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.bilinear = nn.Linear(d_model, 3, bias=False)
        self.bilinear_scale = nn.Parameter(torch.tensor(12.0))

    def forward(self, ctx, endpoint, observed_tokens, pad_mask):
        endpoint_h = self.endpoint_proj(endpoint_descriptor(ctx.refined_mu, endpoint))
        traj_h = self.traj_proj(torch.cat([ctx.pooled, ctx.refined_mu], dim=-1))

        residual_tokens = path_magnetic_residual(endpoint, observed_tokens, pad_mask)
        residual_emb = self.residual_token_proj(residual_tokens)

        attn_h = endpoint_h
        for layer_idx, (attn, norm) in enumerate(zip(self.cross_attn, self.attn_norm)):
            memory = ctx.token_states if layer_idx == 0 else residual_emb
            attended, _ = attn(
                attn_h[:, None, :],
                memory,
                memory,
                key_padding_mask=pad_mask,
            )
            attn_h = norm(attended.squeeze(1) + attn_h)

        physics_stats = summarize_path_physics(endpoint, observed_tokens, pad_mask)
        physics_h = self.physics_proj(physics_stats)

        fused = traj_h
        for block in self.fusion:
            fused = block(fused, attn_h, physics_h, endpoint_h)

        mlp_logit = self.head(fused).squeeze(-1)

        dot = (normalize(endpoint) * normalize(ctx.refined_mu)).sum(dim=-1)
        align_gate = torch.sigmoid(12.0 * (dot - 0.86))
        bilinear_logit = align_gate * self.bilinear_scale * (
            self.bilinear(fused) * normalize(endpoint)
        ).sum(dim=-1)

        physics_weight = F.softplus(self.ctx_physics(torch.cat([ctx.pooled, ctx.refined_mu], dim=-1))).squeeze(-1)
        physics_bonus = -F.softplus(self.energy_scale) * physics_weight * physics_stats[:, 0]

        return mlp_logit + bilinear_logit + physics_bonus


class AmortizedEnergyCritic(nn.Module):
    """Encode trajectory once; score endpoints with a lightweight physics-gated energy critic."""

    def __init__(
        self,
        feature_dim,
        feature_mean,
        feature_std,
        d_model=384,
        n_heads=4,
        n_encoder_layers=4,
        n_refine_steps=3,
        n_fusion_layers=2,
        max_tokens=17,
        dropout=0.05,
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
        self.refine_blocks = nn.ModuleList(
            [
                RefineBlock(d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_refine_steps)
            ]
        )
        self.scorer = LightweightEnergyScorer(
            d_model,
            n_heads=n_heads,
            n_fusion_layers=n_fusion_layers,
            dropout=dropout,
        )

    def encode(self, x, pad_mask):
        token_states, pooled = self.encoder(x, pad_mask)
        endpoint = initial_endpoint_from_trajectory(x, pad_mask)
        for block in self.refine_blocks:
            residual = path_magnetic_residual(endpoint, x, pad_mask)
            endpoint, _ = block(endpoint, token_states, residual, pad_mask)
        return TrajectoryContext(
            token_states=token_states,
            pooled=pooled,
            refined_mu=endpoint,
        )

    def critic_logit(self, ctx, endpoint, x, pad_mask):
        return self.scorer(ctx, endpoint, x, pad_mask)

    def forward(self, x, pad_mask, endpoint):
        return self.critic_logit(self.encode(x, pad_mask), endpoint, x, pad_mask)

    def critic_logit_batched(self, ctx, endpoints, x, pad_mask, chunk_size=None):
        if endpoints.dim() == 2:
            return self.critic_logit(ctx, endpoints, x, pad_mask)

        batch_size, negative_count, _ = endpoints.shape
        flat_endpoints = endpoints.reshape(-1, 3)
        row_index = torch.arange(batch_size, device=x.device).repeat_interleave(
            negative_count
        )
        if chunk_size is None:
            return self._critic_indexed(ctx, row_index, flat_endpoints, x, pad_mask)

        chunks = []
        for start in range(0, flat_endpoints.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_endpoints.shape[0])
            chunks.append(
                self._critic_indexed(
                    ctx,
                    row_index[start:stop],
                    flat_endpoints[start:stop],
                    x,
                    pad_mask,
                )
            )
        return torch.cat(chunks, dim=0)

    def _critic_indexed(self, ctx, row_index, endpoints, x, pad_mask):
        indexed = TrajectoryContext(
            token_states=ctx.token_states.index_select(0, row_index),
            pooled=ctx.pooled.index_select(0, row_index),
            refined_mu=ctx.refined_mu.index_select(0, row_index),
        )
        return self.critic_logit(
            indexed,
            endpoints,
            x.index_select(0, row_index),
            pad_mask.index_select(0, row_index),
        )


def contrastive_logits(
    model,
    x,
    pad_mask,
    endpoint,
    negative_count,
    negative_chunk_size=None,
    hard_negative_fraction=0.0,
    in_batch_negatives=0,
):
    ctx = model.encode(x, pad_mask)
    positive_logit = model.critic_logit(ctx, endpoint, x, pad_mask)

    if hard_negative_fraction <= 0.0 and in_batch_negatives <= 0:
        negatives = sample_uniform_sphere(
            endpoint.shape[0] * negative_count, device=endpoint.device
        ).reshape(endpoint.shape[0], negative_count, 3)
    else:
        negatives = sample_mixed_negatives(
            endpoint.shape[0],
            negative_count,
            ctx.refined_mu,
            endpoint,
            hard_fraction=hard_negative_fraction,
            in_batch_count=in_batch_negatives,
            batch_endpoints=endpoint,
        )

    negative_logit = model.critic_logit_batched(
        ctx,
        negatives,
        x,
        pad_mask,
        chunk_size=negative_chunk_size,
    )
    return positive_logit, negative_logit.reshape(-1)


def bounded_logits(logits, clamp_abs):
    return logits.clamp(-clamp_abs, clamp_abs)


def critic_nwj_training_nats(positive_logits, negative_logits, clamp_abs, exp_clamp):
    positive_t = bounded_logits(positive_logits, clamp_abs)
    negative_t = bounded_logits(negative_logits, clamp_abs)
    negative_exp = torch.exp((negative_t - 1.0).clamp(max=exp_clamp))
    return positive_t.mean() - negative_exp.mean()


def critic_nwj_training_loss(
    positive_logits,
    negative_logits,
    clamp_abs,
    exp_clamp,
):
    return -critic_nwj_training_nats(
        positive_logits, negative_logits, clamp_abs, exp_clamp
    )
