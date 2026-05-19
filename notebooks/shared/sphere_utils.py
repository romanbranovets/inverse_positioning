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
