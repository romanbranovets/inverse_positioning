import torch
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def sample_uniform_sphere(n, device=DEVICE):
    return normalize(torch.randn(n, 3, device=device, dtype=DTYPE))


def tangent_basis(u):
    flat = u.reshape(-1, 3)
    z = torch.tensor([0.0, 0.0, 1.0], device=flat.device, dtype=flat.dtype).expand_as(
        flat
    )
    x = torch.tensor([1.0, 0.0, 0.0], device=flat.device, dtype=flat.dtype).expand_as(
        flat
    )
    ref = torch.where(flat[:, 2:3].abs() < 0.9, z, x)
    e1 = normalize(torch.cross(ref, flat, dim=-1))
    e2 = torch.cross(flat, e1, dim=-1)
    return e1.reshape_as(u), e2.reshape_as(u)


def fibonacci_sphere(n, device=DEVICE):
    i = torch.arange(n, device=device, dtype=DTYPE) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def equal_area_sphere_grid(n, device=DEVICE):
    """Deterministic equal-area ring grid with exactly n unit-sphere points."""
    if n <= 0:
        return torch.empty(0, 3, device=device, dtype=DTYPE)

    ring_count = max(1, int(round(0.72 * math.sqrt(n))))
    ring_index = torch.arange(ring_count, dtype=torch.float64)
    z = 1.0 - 2.0 * (ring_index + 0.5) / ring_count
    radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    ideal = radius / radius.sum() * n
    counts = torch.floor(ideal).clamp_min(1).to(torch.long)

    diff = int(n - counts.sum().item())
    if diff > 0:
        order = torch.argsort(ideal - counts.to(torch.float64), descending=True)
        for idx in order[:diff]:
            counts[idx] += 1
    elif diff < 0:
        removable = counts > 1
        order = torch.argsort(counts.to(torch.float64) - ideal, descending=True)
        removed = 0
        for idx in order:
            if removable[idx]:
                counts[idx] -= 1
                removed += 1
                if removed == -diff:
                    break

    points = []
    golden_fraction = (math.sqrt(5.0) - 1.0) / 2.0
    for r_idx, count in enumerate(counts.tolist()):
        offset = (r_idx * golden_fraction) % 1.0
        k = torch.arange(count, dtype=DTYPE)
        phi = 2.0 * math.pi * (k + offset) / count
        z_value = torch.full((count,), float(z[r_idx]), dtype=DTYPE)
        radius_value = torch.sqrt((1.0 - z_value * z_value).clamp_min(0.0))
        points.append(
            torch.stack(
                [
                    radius_value * torch.cos(phi),
                    radius_value * torch.sin(phi),
                    z_value,
                ],
                dim=-1,
            )
        )

    return torch.cat(points, dim=0).to(device=device)
