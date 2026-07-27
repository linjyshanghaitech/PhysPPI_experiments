"""Pair-only contact residuals for PhysPPI conditioning."""

from __future__ import annotations

import torch
from torch import nn


class ContactPairEncoder(nn.Module):
    """Project a symmetric binary contact map into pair representation space."""

    def __init__(self, c_z: int = 128) -> None:
        super().__init__()
        self.c_z = int(c_z)
        self.proj = nn.Sequential(
            nn.Linear(1, self.c_z),
            nn.LayerNorm(self.c_z),
            nn.GELU(),
            nn.Linear(self.c_z, self.c_z),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def _pair_mask(self, pair_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.as_tensor(pair_mask, device=self.gate.device).bool()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[-1] != mask.shape[-2]:
            raise ValueError("pair_mask must have shape [B, N, N]")
        mask = mask & mask.transpose(-1, -2)
        diagonal = torch.eye(mask.shape[-1], device=mask.device, dtype=torch.bool)
        return mask & ~diagonal.unsqueeze(0)

    @staticmethod
    def _active_mask(
        active: torch.Tensor | bool | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if active is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        value = torch.as_tensor(active, device=device).bool().reshape(-1)
        if value.numel() == 1:
            return value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError("active must be scalar or have one value per batch item")
        return value

    def _connected_zero(self, pair_mask: torch.Tensor) -> torch.Tensor:
        dtype = self.proj[0].weight.dtype
        reference = pair_mask.to(dtype=dtype).unsqueeze(-1)
        return reference.expand(*pair_mask.shape, self.c_z) * (
            self.gate.to(dtype=dtype) * 0.0
        )

    def forward(
        self,
        *,
        payload: torch.Tensor | None,
        active: torch.Tensor | bool | None,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_pairs = self._pair_mask(pair_mask)
        active_mask = self._active_mask(
            active,
            batch_size=valid_pairs.shape[0],
            device=valid_pairs.device,
        )
        if payload is None or not bool(active_mask.any()):
            return self._connected_zero(valid_pairs)

        values = torch.as_tensor(payload, device=valid_pairs.device).bool()
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.shape != valid_pairs.shape:
            raise ValueError("contact payload must have shape [B, N, N]")
        values = values | values.transpose(-1, -2)
        support = values & valid_pairs & active_mask[:, None, None]
        if not bool(support.any()):
            return self._connected_zero(valid_pairs)

        dtype = self.proj[0].weight.dtype
        delta = self.proj(values.to(dtype=dtype).unsqueeze(-1))
        delta = delta * support.unsqueeze(-1).to(dtype=delta.dtype)
        return delta * torch.tanh(self.gate).to(dtype=delta.dtype)


__all__ = ["ContactPairEncoder"]
