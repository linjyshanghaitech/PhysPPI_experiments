"""Single-token interface-residue residuals for PhysPPI conditioning."""

from __future__ import annotations

import torch
from torch import nn


class InterfaceResidueEncoder(nn.Module):
    """Project a per-residue interface mask into single representation space.

    Structural copy of ``TMTopologyEncoder``: a shared learned vector is added
    to every interface residue token and gated to zero at initialization.
    """

    def __init__(self, c_s: int) -> None:
        super().__init__()
        self.c_s = int(c_s)
        self.proj = nn.Sequential(
            nn.Linear(1, self.c_s),
            nn.LayerNorm(self.c_s),
            nn.GELU(),
            nn.Linear(self.c_s, self.c_s),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        payload: torch.Tensor | None,
        active: torch.Tensor | bool | None,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_mask.ndim != 2:
            raise ValueError(
                "token_mask must have shape (batch, tokens), got "
                f"{tuple(token_mask.shape)}"
            )
        token_mask = token_mask.bool()
        if payload is None:
            effective = torch.zeros_like(token_mask, dtype=torch.bool)
        else:
            values = torch.as_tensor(payload, device=token_mask.device).bool()
            if values.ndim == 1:
                values = values.unsqueeze(0)
            if values.shape != token_mask.shape:
                raise ValueError(
                    "interface_residue payload must match token_mask, got "
                    f"{tuple(values.shape)} and {tuple(token_mask.shape)}"
                )
            effective = values & token_mask

        active_mask = torch.as_tensor(
            active, device=token_mask.device, dtype=torch.bool
        ).reshape(-1)
        if active_mask.numel() == 1:
            active_mask = active_mask.expand(token_mask.shape[0])
        if active_mask.shape != (token_mask.shape[0],):
            raise ValueError(
                "interface_residue active must be scalar or contain one value "
                "per batch row"
            )

        projection_dtype = next(self.proj.parameters()).dtype
        delta = torch.zeros(
            *effective.shape,
            self.c_s,
            device=token_mask.device,
            dtype=projection_dtype,
        )
        supported = effective & active_mask.unsqueeze(-1)
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
        return torch.tanh(self.gate).to(delta.dtype) * delta


__all__ = ["InterfaceResidueEncoder"]
