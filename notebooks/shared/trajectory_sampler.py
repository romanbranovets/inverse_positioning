import math
import torch

from .sphere_utils import *
from .magentic_field import *

EPS_B_STD = 0.003
EPS_U_STD = 0.0001

MIN_STEPS = 1
MAX_STEPS = 16
WALK_DIRECTION_RANDOMNESS = 0.45

STEP_METERS = 100.0
STEP_RAD = STEP_METERS / EARTH_RADIUS_M

# B(3), dB(3), outgoing du meters(2), cumulative du meters(2),
# time/timestep phase(3), final flag(1).
FEATURE_DIM = 14
FEATURE_NORM_SAMPLES = 8192


def exp_map_sphere(u, tangent_step_meters):
    step_meters = tangent_step_meters.norm(dim=-1, keepdim=True)
    theta = step_meters / EARTH_RADIUS_M
    direction = tangent_step_meters / step_meters.clamp_min(1e-8)
    moved = torch.cos(theta) * u + torch.sin(theta) * direction
    return normalize(moved)


def make_trajectories(batch_size, step_counts=None, device=DEVICE, return_path=False):
    if step_counts is None:
        counts = torch.randint(MIN_STEPS, MAX_STEPS + 1, (batch_size,), device=device)
    elif isinstance(step_counts, int):
        counts = torch.full((batch_size,), step_counts, device=device, dtype=torch.long)
    else:
        counts = torch.as_tensor(step_counts, device=device, dtype=torch.long)
        batch_size = int(counts.numel())

    max_steps = int(counts.max().item())
    max_tokens = max_steps + 1
    x = torch.zeros(batch_size, max_tokens, FEATURE_DIM, device=device, dtype=DTYPE)
    pad_mask = torch.ones(batch_size, max_tokens, device=device, dtype=torch.bool)

    u = sample_uniform_sphere(batch_size, device=device)
    path = torch.zeros(batch_size, max_tokens, 3, device=device, dtype=DTYPE)
    path[:, 0, :] = u
    e1, e2 = tangent_basis(u)
    angle = 2.0 * math.pi * torch.rand(batch_size, 1, device=device, dtype=DTYPE)
    heading = torch.cos(angle) * e1 + torch.sin(angle) * e2
    last_observed_B = torch.zeros(batch_size, 3, device=device, dtype=DTYPE)
    cumulative_du = torch.zeros(batch_size, 2, device=device, dtype=DTYPE)

    for t in range(max_steps):
        token_active = t <= counts
        step_active = t < counts
        e1, e2 = tangent_basis(u)
        heading = normalize(heading - (heading * u).sum(dim=-1, keepdim=True) * u)
        heading_2d = torch.stack(
            [(heading * e1).sum(dim=-1), (heading * e2).sum(dim=-1)], dim=-1
        )
        noisy_direction = heading_2d + WALK_DIRECTION_RANDOMNESS * torch.randn(
            batch_size, 2, device=device, dtype=DTYPE
        )
        du_2d_meters = STEP_METERS * normalize(noisy_direction)
        tangent_step_meters = du_2d_meters[:, 0:1] * e1 + du_2d_meters[:, 1:2] * e2

        observed_B = B(u) + EPS_B_STD * torch.randn(
            batch_size, 3, device=device, dtype=DTYPE
        )
        observed_du = du_2d_meters + EPS_U_STD * torch.randn(
            batch_size, 2, device=device, dtype=DTYPE
        )
        field_delta = (
            torch.zeros_like(observed_B) if t == 0 else observed_B - last_observed_B
        )
        token_time = torch.full(
            (batch_size, 1), t / max(max_steps, 1), device=device, dtype=DTYPE
        )
        time_phase = 2.0 * math.pi * token_time
        final_flag = (t == counts).to(DTYPE).unsqueeze(-1)
        token = torch.cat(
            [
                observed_B,
                field_delta,
                torch.where(
                    step_active[:, None], observed_du, torch.zeros_like(observed_du)
                ),
                cumulative_du,
                token_time,
                torch.sin(time_phase),
                torch.cos(time_phase),
                final_flag,
            ],
            dim=-1,
        )
        x[token_active, t, :] = token[token_active]
        pad_mask[token_active, t] = False

        next_u = exp_map_sphere(u, tangent_step_meters)
        transported_heading = normalize(
            tangent_step_meters
            - (tangent_step_meters * next_u).sum(dim=-1, keepdim=True) * next_u
        )
        u = torch.where(step_active[:, None], next_u, u)
        heading = torch.where(step_active[:, None], transported_heading, heading)
        cumulative_du = torch.where(
            step_active[:, None], cumulative_du + observed_du, cumulative_du
        )
        last_observed_B = torch.where(
            token_active[:, None], observed_B, last_observed_B
        )
        path[:, t + 1, :] = u

    final_active = max_steps <= counts
    final_observed_B = B(u) + EPS_B_STD * torch.randn(
        batch_size, 3, device=device, dtype=DTYPE
    )
    final_delta = final_observed_B - last_observed_B
    final_time = torch.ones(batch_size, 1, device=device, dtype=DTYPE)
    final_token = torch.cat(
        [
            final_observed_B,
            final_delta,
            torch.zeros(batch_size, 2, device=device, dtype=DTYPE),
            cumulative_du,
            final_time,
            torch.sin(2.0 * math.pi * final_time),
            torch.cos(2.0 * math.pi * final_time),
            torch.ones(batch_size, 1, device=device, dtype=DTYPE),
        ],
        dim=-1,
    )
    x[final_active, max_steps, :] = final_token[final_active]
    pad_mask[final_active, max_steps] = False

    if return_path:
        return x, pad_mask, u, counts, path
    return x, pad_mask, u, counts


@torch.no_grad()
def estimate_feature_normalization(samples=FEATURE_NORM_SAMPLES, device=DEVICE):
    x, pad_mask, _, _ = make_trajectories(samples, device=device)
    valid = x[~pad_mask]
    feature_mean = valid.mean(dim=0)
    feature_std = valid.std(dim=0).clamp_min(1e-4)
    return feature_mean, feature_std
