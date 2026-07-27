"""TM-token topology encoder and membrane activation gate."""

from __future__ import annotations

import torch
from torch import nn


class TMTopologyEncoder(nn.Module):
    def __init__(self, c_s: int) -> None:
        super().__init__()
        self.c_s = c_s
        self.proj = nn.Sequential(
            nn.Linear(1, c_s), nn.LayerNorm(c_s), nn.GELU(), nn.Linear(c_s, c_s)
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        tm_token_mask: torch.Tensor | None,
        tm_active: torch.Tensor | bool,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_mask.ndim != 2:
            raise ValueError(
                "token_mask must have shape (batch, tokens), got "
                f"{tuple(token_mask.shape)}"
            )
        token_mask = token_mask.bool()
        if tm_token_mask is None:
            effective = torch.zeros_like(token_mask, dtype=torch.bool)
        else:
            if tm_token_mask.shape != token_mask.shape:
                raise ValueError(
                    "tm_token_mask must match token_mask, got "
                    f"{tuple(tm_token_mask.shape)} and {tuple(token_mask.shape)}"
                )
            effective = tm_token_mask.to(device=token_mask.device).bool() & token_mask
        active = torch.as_tensor(tm_active, device=token_mask.device).bool()
        if active.ndim == 0:
            active = active.expand(token_mask.shape[0])
        else:
            active = active.reshape(-1)
        if active.shape != (token_mask.shape[0],):
            raise ValueError(
                "tm_active must be scalar or contain one value per batch row, got "
                f"{tuple(active.shape)}"
            )
        membrane_on = active & effective.any(dim=-1)
        projection_dtype = next(self.proj.parameters()).dtype
        delta = torch.zeros(
            *effective.shape,
            self.c_s,
            device=token_mask.device,
            dtype=projection_dtype,
        )
        supported = effective & membrane_on.unsqueeze(-1)
        if supported.any():
            projected = self.proj(
                torch.ones(
                    int(supported.sum().item()),
                    1,
                    device=token_mask.device,
                    dtype=projection_dtype,
                )
            ).to(delta)
            delta[supported] = projected
            delta = torch.tanh(self.gate).to(delta.dtype) * delta
        return delta, membrane_on, effective


__all__ = ["TMTopologyEncoder"]
