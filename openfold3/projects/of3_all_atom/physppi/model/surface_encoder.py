"""Single-query cross-attention from token state to local surface mesh."""

from __future__ import annotations

import torch
from torch import nn

from openfold3.projects.of3_all_atom.physppi.features.surface_position import (
    encode_surface_position,
)


class SurfaceTokenEncoder(nn.Module):
    """Emit a zero-gated single-state residual from same-chain mesh points."""

    def __init__(
        self,
        c_s: int,
        num_surface_features: int = 6,
        no_heads: int = 8,
    ) -> None:
        super().__init__()
        if int(num_surface_features) != 6:
            raise ValueError("PhysPPI surface inputs require exactly 6 attributes")
        if int(c_s) % int(no_heads) != 0:
            raise ValueError("surface width must be divisible by the attention heads")
        self.c_s = int(c_s)
        self.no_heads = int(no_heads)
        self.point_proj = nn.Sequential(
            nn.Linear(64 + 6, self.c_s),
            nn.LayerNorm(self.c_s),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(self.c_s)
        self.query_position = nn.Linear(64, self.c_s)
        self.attention = nn.MultiheadAttention(
            self.c_s,
            self.no_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.c_s)
        self.output_proj = nn.Linear(self.c_s, self.c_s)
        self.gate = nn.Parameter(torch.zeros(()))

    def _connected_zero(self, s_init: torch.Tensor) -> torch.Tensor:
        reference = torch.nan_to_num(s_init) * 0.0
        return reference + self.gate.to(dtype=reference.dtype) * 0.0

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
            raise ValueError(
                "surface_active must be scalar or have one value per batch item"
            )
        return value

    def forward(
        self,
        *,
        s_init: torch.Tensor,
        local_coords: torch.Tensor | None,
        features: torch.Tensor | None,
        surface_mask: torch.Tensor | None,
        surface_active: torch.Tensor | bool | None,
        token_mask: torch.Tensor,
        surface_charge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        single = torch.as_tensor(s_init, device=self.gate.device)
        if single.ndim != 3 or single.shape[-1] != self.c_s:
            raise ValueError("s_init must have shape [B, N, c_s]")
        tokens = torch.as_tensor(token_mask, device=single.device).bool()
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.shape != single.shape[:2]:
            raise ValueError("token_mask must match the first two s_init axes")

        active = self._active_mask(
            surface_active,
            batch_size=single.shape[0],
            device=single.device,
        )
        if (
            local_coords is None
            or features is None
            or surface_mask is None
            or not bool(active.any())
        ):
            return self._connected_zero(single)

        coords = torch.as_tensor(local_coords, device=single.device)
        attrs = torch.as_tensor(features, device=single.device)
        point_mask = torch.as_tensor(surface_mask, device=single.device).bool()
        if coords.ndim != 4 or coords.shape[-1] != 3:
            raise ValueError("surface coordinates must have shape [B, N, K, 3]")
        if attrs.shape != (*coords.shape[:-1], 6):
            raise ValueError("surface features must have shape [B, N, K, 6]")
        if point_mask.shape != coords.shape[:-1]:
            raise ValueError("surface_mask must have shape [B, N, K]")
        if coords.shape[:2] != single.shape[:2]:
            raise ValueError("surface token axes must match s_init")
        if coords.shape[-2] > 128:
            raise ValueError("surface neighborhoods may contain at most 128 points")

        if surface_charge_mask is None:
            charge_mask = torch.ones_like(point_mask)
        else:
            charge_mask = torch.as_tensor(
                surface_charge_mask,
                device=single.device,
            ).bool()
            if charge_mask.shape != point_mask.shape:
                raise ValueError(
                    "surface_charge_mask must match surface feature neighbor shape"
                )
        charge = attrs[..., :1].masked_fill(~charge_mask.unsqueeze(-1), 0.0)
        attrs = torch.cat((charge, attrs[..., 1:]), dim=-1)

        valid_tokens = active[:, None] & tokens & point_mask.any(dim=-1)
        if not bool(valid_tokens.any()):
            return self._connected_zero(single)

        batch_size, n_tokens, neighbors, _ = coords.shape
        flat_valid = valid_tokens.reshape(-1)
        row_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        valid_coords = coords.reshape(-1, neighbors, 3)[flat_valid]
        valid_attrs = attrs.reshape(-1, neighbors, 6)[flat_valid]
        valid_point_mask = point_mask.reshape(-1, neighbors)[flat_valid]

        position = encode_surface_position(valid_coords).to(dtype=valid_attrs.dtype)
        point_input = torch.cat((position, valid_attrs), dim=-1)
        key_value = self.point_proj(point_input)

        valid_single = single.reshape(-1, self.c_s)[flat_valid]
        origin = torch.zeros(
            valid_single.shape[0],
            3,
            device=single.device,
            dtype=single.dtype,
        )
        origin_position = encode_surface_position(origin).to(dtype=single.dtype)
        query = (
            self.query_norm(valid_single) + self.query_position(origin_position)
        ).unsqueeze(1)
        attended, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~valid_point_mask,
            need_weights=False,
        )
        valid_delta = self.output_proj(self.output_norm(attended.squeeze(1)))

        flat_delta = torch.zeros(
            batch_size * n_tokens,
            self.c_s,
            device=single.device,
            dtype=valid_delta.dtype,
        )
        flat_delta = flat_delta.index_copy(0, row_indices, valid_delta)
        delta = flat_delta.reshape(batch_size, n_tokens, self.c_s)
        return delta * torch.tanh(self.gate).to(dtype=delta.dtype)


__all__ = ["SurfaceTokenEncoder"]
