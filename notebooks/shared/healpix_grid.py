import math

import numpy as np
import torch
from astropy_healpix import healpix_to_xyz, nside_to_npix

from .sphere_utils import DEVICE, DTYPE


def healpix_entropy_bits(nside):
    return math.log2(int(nside_to_npix(nside)))


def healpix_nested_grid(nside, device=DEVICE):
    ipix = np.arange(int(nside_to_npix(nside)), dtype=np.int64)
    x, y, z = healpix_to_xyz(ipix, nside, order="nested")
    points = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    return torch.as_tensor(points, device=device, dtype=DTYPE)


def nested_subset_indices(nside, max_nside):
    if max_nside % nside != 0:
        raise ValueError("max_nside must be an integer multiple of nside")
    ratio = max_nside // nside
    if ratio & (ratio - 1):
        raise ValueError("nested HEALPix reuse expects power-of-two nside ratios")
    count = int(nside_to_npix(nside))
    levels_down = int(math.log2(ratio))
    start_level = int(math.log2(nside))
    indices = torch.arange(count, dtype=torch.long)
    for level_offset in range(levels_down):
        # HEALPix nested children are 4*p + child. A fixed child creates visible
        # polar cross artifacts, so choose a deterministic pseudo-random child
        # per parent while preserving exact nesting across levels.
        current_level = start_level + level_offset
        mixed = torch.sin(indices.to(torch.float64) * 12.9898 + current_level * 78.233)
        unit = torch.remainder(mixed * 43758.5453, 1.0)
        child = torch.floor(unit * 4).to(torch.long)
        indices = indices * 4 + child
    return indices
