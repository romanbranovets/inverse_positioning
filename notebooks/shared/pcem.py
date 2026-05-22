"""Path-Consistent Energy Model (PCEM) for inverse positioning."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .magentic_field import B
from .sphere_utils import DEVICE, DTYPE, normalize, sample_uniform_sphere, tangent_basis
from .trajectory_sampler import (
    EPS_B_STD,
    EPS_U_STD,
    FEATURE_DIM,
    WALK_DIRECTION_RANDOMNESS,
    STEP_METERS,
    exp_map_sphere,
)


def reconstruct_candidate_paths(candidate_endpoints, observed_tokens):
    """Rebuild forward paths from candidate endpoints using observed meter steps."""
    batch_size, token_count, _ = observed_tokens.shape
    positions_rev = []
    u = candidate_endpoints
    positions_rev.append(u)
    for t in range(token_count - 2, -1, -1):
        du = observed_tokens[:, t, 6:8]
        e1, e2 = tangent_basis(u)
        back_step = -du[:, 0:1] * e1 - du[:, 1:2] * e2
        u = exp_map_sphere(u, back_step)
        positions_rev.append(u)
    return torch.stack(list(reversed(positions_rev)), dim=1)


def physics_log_score(candidate_endpoints, observed_tokens, pad_mask):
    """Classic Gaussian magnetic likelihood summed over valid trajectory tokens."""
    paths = reconstruct_candidate_paths(candidate_endpoints, observed_tokens)
    predicted_B = B(paths.reshape(-1, 3)).reshape(
        candidate_endpoints.shape[0], paths.shape[1], 3
    )
    residual = (observed_tokens[:, :, :3] - predicted_B) / EPS_B_STD
    valid = ~pad_mask
    return -0.5 * (residual.square() * valid.unsqueeze(-1)).sum(dim=(1, 2))


@torch.no_grad()
def sample_grid_trajectory(endpoint_grid, step_count, batch_size=1):
    """Sample a trajectory whose endpoint is drawn from a finite grid."""
    endpoint_indices = torch.randint(
        endpoint_grid.shape[0], (batch_size,), device=endpoint_grid.device, dtype=torch.long
    )
    endpoints = endpoint_grid[endpoint_indices]
    max_tokens = step_count + 1
    x = torch.zeros(batch_size, max_tokens, FEATURE_DIM, device=DEVICE, dtype=DTYPE)
    path = torch.zeros(batch_size, max_tokens, 3, device=DEVICE, dtype=DTYPE)

    u = endpoints
    path[:, step_count, :] = u
    heading_2d = torch.randn(batch_size, 2, device=DEVICE, dtype=DTYPE)
    heading_2d = heading_2d / heading_2d.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    for t in range(step_count - 1, -1, -1):
        noisy_direction = heading_2d + WALK_DIRECTION_RANDOMNESS * torch.randn(
            batch_size, 2, device=DEVICE, dtype=DTYPE
        )
        du_2d_meters = (
            STEP_METERS
            * noisy_direction
            / noisy_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        observed_du = du_2d_meters + EPS_U_STD * torch.randn(
            batch_size, 2, device=DEVICE, dtype=DTYPE
        )
        x[:, t, 6:8] = observed_du

        e1, e2 = tangent_basis(u)
        back_step = -observed_du[:, 0:1] * e1 - observed_du[:, 1:2] * e2
        previous_u = exp_map_sphere(u, back_step)
        path[:, t, :] = previous_u
        heading_2d = noisy_direction / noisy_direction.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        u = previous_u

    observed_B = B(path.reshape(-1, 3)).reshape(batch_size, max_tokens, 3)
    observed_B = observed_B + EPS_B_STD * torch.randn_like(observed_B)
    x[:, :, :3] = observed_B
    x[:, 0, 3:6] = 0.0
    x[:, 1:, 3:6] = observed_B[:, 1:, :] - observed_B[:, :-1, :]
    if step_count > 0:
        x[:, :, 8:10] = torch.cat(
            [
                torch.zeros(batch_size, 1, 2, device=DEVICE, dtype=DTYPE),
                torch.cumsum(x[:, :-1, 6:8], dim=1),
            ],
            dim=1,
        )
    token_time = torch.linspace(0.0, 1.0, max_tokens, device=DEVICE, dtype=DTYPE)
    time_phase = 2.0 * math.pi * token_time
    x[:, :, 10] = token_time
    x[:, :, 11] = torch.sin(time_phase)
    x[:, :, 12] = torch.cos(time_phase)
    x[:, -1, 13] = 1.0
    pad_mask = torch.zeros(batch_size, max_tokens, device=DEVICE, dtype=torch.bool)
    return x, pad_mask, endpoints, endpoint_indices, path


def residual_summary_features(candidate_endpoints, observed_tokens, pad_mask):
    """Aggregate path-consistency residuals for each candidate endpoint."""
    paths = reconstruct_candidate_paths(candidate_endpoints, observed_tokens)
    predicted_B = B(paths.reshape(-1, 3)).reshape(
        candidate_endpoints.shape[0], paths.shape[1], 3
    )
    residual_sigma = (observed_tokens[:, :, :3] - predicted_B) / EPS_B_STD
    valid = (~pad_mask).unsqueeze(-1)
    token_energy = torch.log1p(residual_sigma.square().sum(dim=-1, keepdim=True))
    masked_energy = token_energy.masked_fill(~valid, 0.0)
    mean_energy = masked_energy.sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    max_energy = token_energy.masked_fill(~valid, -torch.inf).amax(dim=1)
    max_energy = torch.where(
        valid.any(dim=1),
        max_energy,
        torch.zeros_like(max_energy),
    )
    return torch.cat([mean_energy, max_energy], dim=-1)


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        feature_mean,
        feature_std,
        d_model=256,
        n_heads=4,
        n_layers=3,
        max_tokens=17,
        dropout=0.05,
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
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x, pad_mask):
        scaled = (x - self.feature_mean) / self.feature_std
        tokens = self.input_projection(scaled)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
        attention_logits = self.attention(encoded).squeeze(-1).masked_fill(pad_mask, -torch.inf)
        weights = torch.softmax(attention_logits, dim=-1)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=1)


class PathConsistentEnergyModel(nn.Module):
    """Physics log-likelihood plus a small neural correction on endpoint candidates."""

    def __init__(
        self,
        feature_dim,
        feature_mean,
        feature_std,
        d_model=256,
        n_heads=4,
        n_layers=3,
        max_tokens=17,
        dropout=0.05,
        neural_weight=1.0,
    ):
        super().__init__()
        self.neural_weight = neural_weight
        self.trajectory_encoder = TrajectoryEncoder(
            feature_dim=feature_dim,
            feature_mean=feature_mean,
            feature_std=feature_std,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_tokens=max_tokens,
            dropout=dropout,
        )
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def encode_trajectory(self, x, pad_mask):
        return self.trajectory_encoder(x, pad_mask)

    def _expand_trajectory(self, x, pad_mask, candidate_count):
        batch_size = x.shape[0]
        flat_count = batch_size * candidate_count
        flat_x = x.repeat_interleave(candidate_count, dim=0)
        flat_mask = pad_mask.repeat_interleave(candidate_count, dim=0)
        traj_index = torch.arange(batch_size, device=x.device).repeat_interleave(
            candidate_count
        )
        return flat_x, flat_mask, traj_index, flat_count

    def physics_log_score(self, candidates, x, pad_mask):
        """Score candidates with the exact synthetic-field likelihood."""
        batch_size, candidate_count, _ = candidates.shape
        flat_x, flat_mask, _, _ = self._expand_trajectory(x, pad_mask, candidate_count)
        flat_candidates = candidates.reshape(-1, 3)
        flat_scores = physics_log_score(flat_candidates, flat_x, flat_mask)
        return flat_scores.reshape(batch_size, candidate_count)

    def neural_correction(self, h_traj, candidates, x, pad_mask):
        batch_size, candidate_count, _ = candidates.shape
        flat_x, flat_mask, traj_index, _ = self._expand_trajectory(
            x, pad_mask, candidate_count
        )
        flat_candidates = candidates.reshape(-1, 3)
        flat_h = h_traj.index_select(0, traj_index)
        endpoint_features = self.endpoint_embedding(normalize(flat_candidates))
        residual_features = residual_summary_features(
            flat_candidates, flat_x, flat_mask
        )
        correction_input = torch.cat(
            [flat_h, endpoint_features, residual_features], dim=-1
        )
        return self.correction_head(correction_input).reshape(batch_size, candidate_count)

    def score(self, candidates, x, pad_mask, h_traj=None, physics_only=False):
        if h_traj is None:
            h_traj = self.encode_trajectory(x, pad_mask)
        physics = self.physics_log_score(candidates, x, pad_mask)
        if physics_only:
            return physics
        neural = self.neural_correction(h_traj, candidates, x, pad_mask)
        return physics + self.neural_weight * neural

    def forward(self, candidates, x, pad_mask, physics_only=False):
        h_traj = self.encode_trajectory(x, pad_mask)
        return self.score(
            candidates, x, pad_mask, h_traj=h_traj, physics_only=physics_only
        )


def make_candidate_batch(endpoints, negatives_per_positive, device=DEVICE):
    batch_size = endpoints.shape[0]
    negatives = sample_uniform_sphere(
        batch_size * negatives_per_positive, device=device
    ).reshape(batch_size, negatives_per_positive, 3)
    candidates = torch.cat([endpoints[:, None, :], negatives], dim=1)
    labels = torch.zeros(batch_size, device=device, dtype=torch.long)
    return candidates, labels


def contrastive_nwj_scores(
    model,
    x,
    pad_mask,
    endpoint,
    negative_count,
    negative_chunk_size=None,
    physics_only=False,
):
    positive = model.score(
        endpoint[:, None, :], x, pad_mask, physics_only=physics_only
    ).squeeze(1)
    negatives = sample_uniform_sphere(
        endpoint.shape[0] * negative_count, device=endpoint.device
    ).reshape(endpoint.shape[0], negative_count, 3)
    if negative_chunk_size is None:
        negative_scores = model.score(negatives, x, pad_mask, physics_only=physics_only)
        return positive, negative_scores

    h_traj = None if physics_only else model.encode_trajectory(x, pad_mask)
    flat_negatives = negatives.reshape(-1, 3)
    source_index = torch.arange(x.shape[0], device=x.device).repeat_interleave(
        negative_count
    )
    chunks = []
    for start in range(0, flat_negatives.shape[0], negative_chunk_size):
        stop = min(start + negative_chunk_size, flat_negatives.shape[0])
        row_index = source_index[start:stop]
        chunks.append(
            model.score(
                flat_negatives[start:stop, None, :],
                x.index_select(0, row_index),
                pad_mask.index_select(0, row_index),
                h_traj=h_traj.index_select(0, row_index) if h_traj is not None else None,
                physics_only=physics_only,
            ).squeeze(-1)
        )
    negative_scores = torch.cat(chunks, dim=0).reshape(endpoint.shape[0], negative_count)
    return positive, negative_scores


def nwj_bits_from_scores(positive_scores, negative_scores, clamp_abs=22.0):
    positive_t = positive_scores.clamp(-clamp_abs, clamp_abs)
    negative_t = negative_scores.clamp(-clamp_abs, clamp_abs)
    nwj_nats = positive_t.mean() - torch.exp(negative_t - 1.0).mean()
    return float((nwj_nats / math.log(2.0)).detach().cpu())


@torch.no_grad()
def score_endpoint_grid(model, endpoint_grid, x, pad_mask, chunk_size=512, physics_only=False):
    """Score every grid endpoint for each trajectory in the batch."""
    batch_size = x.shape[0]
    h_traj = None if physics_only else model.encode_trajectory(x, pad_mask)
    score_chunks = []
    for start in range(0, endpoint_grid.shape[0], chunk_size):
        stop = min(start + chunk_size, endpoint_grid.shape[0])
        chunk = endpoint_grid[start:stop]
        candidates = chunk.unsqueeze(0).expand(batch_size, -1, -1)
        chunk_scores = model.score(
            candidates,
            x,
            pad_mask,
            h_traj=h_traj,
            physics_only=physics_only,
        )
        score_chunks.append(chunk_scores)
    grid_scores = torch.cat(score_chunks, dim=1)
    if batch_size == 1:
        return grid_scores.reshape(-1)
    return grid_scores


@torch.no_grad()
def posterior_from_grid_scores(log_scores, prior_count):
    log_prior = -math.log(prior_count)
    log_posterior = log_prior + log_scores - torch.logsumexp(log_prior + log_scores, dim=0)
    posterior = torch.exp(log_posterior)
    entropy_bits = float((-(posterior * log_posterior).sum() / math.log(2.0)).cpu())
    return posterior, entropy_bits
