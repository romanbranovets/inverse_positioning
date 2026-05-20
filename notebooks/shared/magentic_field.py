import math
import torch

from .sphere_utils import *

EARTH_RADIUS_M = 6_371_000.0


FIELD_CENTERS = normalize(
    torch.tensor(
        [
            [0.22, -0.61, 0.76],
            [-0.78, 0.34, 0.52],
            [0.57, 0.73, -0.38],
            [-0.18, -0.83, -0.53],
        ],
        dtype=DTYPE,
    )
)
FIELD_MOMENTS = normalize(
    torch.tensor(
        [
            [0.88, 0.17, -0.44],
            [-0.31, 0.92, 0.24],
            [0.12, -0.58, 0.81],
            [-0.70, -0.21, -0.68],
        ],
        dtype=DTYPE,
    )
)
FIELD_STRENGTHS = torch.tensor([1.1, -0.8, 0.65, -0.55], dtype=DTYPE)
WAVE_DIRS = normalize(
    torch.tensor(
        [
            [1.0, 0.3, -0.2],
            [-0.4, 1.0, 0.5],
            [0.2, -0.7, 1.0],
            [0.8, -0.1, 0.6],
            [-0.5, -0.6, 0.7],
            [0.3, 0.9, 0.4],
        ],
        dtype=DTYPE,
    )
)
# The field is deliberately not a random fingerprint at a single point.
# A weak coarse component gives about city-scale localization from one
# magnetic sample, while aliased kilometer-scale waves are resolved gradually
# as the trajectory observes how the field changes along the walk.
FIELD_FEATURE_SCALES_M = torch.tensor(
    [
        180_000.0,
        90_000.0,
        45_000.0,
        22_000.0,
        11_000.0,
        5_500.0,
        2_800.0,
        1_400.0,
    ],
    dtype=DTYPE,
)
FIELD_FEATURE_WEIGHTS = torch.tensor(
    [0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0090, 0.0080, 0.0060],
    dtype=DTYPE,
)
COARSE_FIELD_GAIN = 0.038
DIPOLE_FIELD_GAIN = 0.0035
TEXTURE_FIELD_GAIN = 1.0


def B(u):
    """Synthetic magnetic field B(u) -> (Bx, By, Bz) for unit 3D vector u."""
    shape = u.shape
    flat = normalize(u.reshape(-1, 3))
    centers = FIELD_CENTERS.to(flat.device)
    moments = FIELD_MOMENTS.to(flat.device)
    strengths = FIELD_STRENGTHS.to(flat.device)
    dirs = WAVE_DIRS.to(flat.device)
    scales = FIELD_FEATURE_SCALES_M.to(flat.device)
    weights = FIELD_FEATURE_WEIGHTS.to(flat.device)

    r = flat[:, None, :] - 0.55 * centers[None, :, :]
    r2 = (r * r).sum(dim=-1).clamp_min(1e-4)
    mr = (moments[None, :, :] * r).sum(dim=-1)
    dipoles = strengths[None, :, None] * (
        3.0 * r * mr[:, :, None] / r2[:, :, None] ** 2.5
        - moments[None, :, :] / r2[:, :, None] ** 1.5
    )
    dipole_field = dipoles.sum(dim=1)

    projected_m = EARTH_RADIUS_M * (flat @ dirs.T)
    fine = torch.zeros_like(flat)
    for j, (scale, weight) in enumerate(zip(scales, weights, strict=True)):
        p0 = 2.0 * math.pi * projected_m[:, j % dirs.shape[0]] / scale
        p1 = 2.0 * math.pi * projected_m[:, (j + 2) % dirs.shape[0]] / scale
        p2 = 2.0 * math.pi * projected_m[:, (j + 4) % dirs.shape[0]] / scale
        fine[:, 0] = fine[:, 0] + weight * torch.sin(p0 + 0.37 * j)
        fine[:, 1] = fine[:, 1] + weight * torch.cos(p1 + 0.53 * j)
        fine[:, 2] = fine[:, 2] + weight * torch.sin(p2 - 0.29 * j)

    field = (
        COARSE_FIELD_GAIN * flat
        + DIPOLE_FIELD_GAIN * dipole_field
        + TEXTURE_FIELD_GAIN * fine
    )
    return field.reshape(shape)
