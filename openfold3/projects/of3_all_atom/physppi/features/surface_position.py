"""Shared local-coordinate positional basis for PhysPPI surface attention."""

from __future__ import annotations

import math

import torch


def encode_surface_position(xyz: torch.Tensor) -> torch.Tensor:
    """Encode local xyz with 48 Fourier and 16 radial RBF channels."""

    coords = torch.as_tensor(xyz)
    if coords.shape[-1] != 3:
        raise ValueError("surface coordinates must end with an xyz axis")
    if not coords.is_floating_point():
        coords = coords.to(dtype=torch.get_default_dtype())

    bands = torch.pow(
        torch.tensor(2.0, device=coords.device, dtype=coords.dtype),
        torch.arange(8, device=coords.device, dtype=coords.dtype),
    )
    phases = (coords / 32.0).unsqueeze(-1) * (math.pi * bands)
    fourier = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1).flatten(-2)

    radius = torch.linalg.vector_norm(coords, dim=-1, keepdim=True).clamp(0.0, 32.0)
    centers = torch.linspace(
        0.0,
        32.0,
        16,
        device=coords.device,
        dtype=coords.dtype,
    )
    width = 32.0 / 15.0
    rbf = torch.exp(-torch.square((radius - centers) / width))
    return torch.cat((fourier, rbf), dim=-1)


__all__ = ["encode_surface_position"]
