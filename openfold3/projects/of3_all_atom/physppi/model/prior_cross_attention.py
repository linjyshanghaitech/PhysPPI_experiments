"""Shared zero-gated cross-attention block for PhysPPI prior adapters (v5)."""

from __future__ import annotations

import torch
from torch import nn


class PriorCrossAttention(nn.Module):
    """Zero-gated cross-attention with exact-zero masked rows.

    Two entry points:
    - ``forward``: queries [B, Lq, C] attend to one key set [B, Lk, C] per
      batch item (interface / tm / surface->si).
    - ``forward_rows``: each query row is its own attention item
      (queries [R, 1, C], keys [R, Lk, C]) for pair-level priors whose keys
      differ per query row (interaction, surface->zij).

    Rows masked by ``row_mask`` or with no valid key return exact zeros, so a
    dropped or absent prior can never inject a finite value into s/z.

    ``forward`` uses one MultiheadAttention call per batch item (the key set is
    shared by every query row), then zeroes masked rows afterwards; keys are
    never duplicated per query row.
    """

    def __init__(
        self,
        c_q: int,
        *,
        c_in: int | None = None,
        no_heads: int = 8,
        query_position: bool = False,
        query_chunk: int | None = None,
    ) -> None:
        super().__init__()
        if int(c_q) % int(no_heads) != 0:
            raise ValueError("c_q must be divisible by no_heads")
        self.c_q = int(c_q)
        self.c_in = int(c_in) if c_in is not None else int(c_q)
        self.no_heads = int(no_heads)
        self.query_chunk = int(query_chunk) if query_chunk else None
        if self.c_in != self.c_q:
            self.query_in_proj = nn.Linear(self.c_in, self.c_q)
        else:
            self.query_in_proj = None
        self.query_norm = nn.LayerNorm(self.c_q)
        self.query_position = nn.Linear(64, self.c_q) if query_position else None
        self.attention = nn.MultiheadAttention(
            self.c_q, self.no_heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(self.c_q)
        self.output_proj = nn.Linear(self.c_q, self.c_q)
        self.gate = nn.Parameter(torch.zeros(()))

    def _prepare_query(
        self,
        queries: torch.Tensor,
        query_position_input: torch.Tensor | None,
    ) -> torch.Tensor:
        q = queries
        if self.query_in_proj is not None:
            q = self.query_in_proj(q)
        q = self.query_norm(q)
        if self.query_position is not None:
            if query_position_input is None:
                raise ValueError("query_position requires query_position_input")
            q = q + self.query_position(
                query_position_input.to(device=q.device, dtype=q.dtype)
            )
        return q

    def _attend_rows(
        self, queries: torch.Tensor, keys: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query=queries,
            key=keys,
            value=keys,
            key_padding_mask=~key_mask,
            need_weights=False,
        )
        return self.output_proj(self.output_norm(attended))

    def forward_rows(
        self,
        *,
        queries: torch.Tensor,
        keys: torch.Tensor,
        key_mask: torch.Tensor,
        query_position_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = next(self.attention.parameters()).dtype
        q = queries.to(dtype=dtype)
        k = keys.to(dtype=dtype)
        km = key_mask.to(device=q.device, dtype=torch.bool)
        if q.ndim != 3 or q.shape[1] != 1 or k.ndim != 3:
            raise ValueError(
                "forward_rows expects queries [R, 1, C] and keys [R, Lk, C]"
            )
        if km.shape != k.shape[:2] or k.shape[-1] != self.c_q:
            raise ValueError("key_mask must be [R, Lk] and keys must be [R, Lk, c_q]")
        q = self._prepare_query(q, query_position_input)
        has_key = km.any(dim=-1)
        attended = torch.zeros_like(q)
        if bool(has_key.any()):
            attended[has_key] = self._attend_rows(q[has_key], k[has_key], km[has_key])
        return attended * torch.tanh(self.gate).to(dtype=attended.dtype)

    def forward(
        self,
        *,
        queries: torch.Tensor,
        keys: torch.Tensor,
        key_mask: torch.Tensor,
        row_mask: torch.Tensor,
        query_position_input: torch.Tensor | None = None,
        query_chunk: int | None = None,
    ) -> torch.Tensor:
        del query_chunk  # 共享 key 场景走整批 MHA,不再按行复制 key
        q = queries
        k = keys.to(device=q.device)
        km = key_mask.to(device=q.device, dtype=torch.bool)
        rm = row_mask.to(device=q.device, dtype=torch.bool)
        if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]:
            raise ValueError("queries/keys must be [B, L, C] with matching batch")
        if q.shape[-1] != self.c_in or k.shape[-1] != self.c_q:
            raise ValueError(
                f"query dim must be {self.c_in} and key dim {self.c_q}, got "
                f"{tuple(q.shape)} / {tuple(k.shape)}"
            )
        if km.shape != k.shape[:2] or rm.shape != q.shape[:2]:
            raise ValueError("key_mask [B, Lk] and row_mask [B, Lq] required")
        dtype = next(self.attention.parameters()).dtype
        q = q.to(dtype=dtype)
        k = k.to(dtype=dtype)
        q = self._prepare_query(q, query_position_input)
        batch_size, n_rows, _ = q.shape
        delta = torch.zeros(
            batch_size, n_rows, self.c_q, device=q.device, dtype=q.dtype
        )
        for b in range(batch_size):
            if not bool(km[b].any().item()):
                continue
            out = self._attend_rows(
                q[b].unsqueeze(0), k[b].unsqueeze(0), km[b].unsqueeze(0)
            )
            delta[b] = out.squeeze(0) * rm[b].unsqueeze(-1).to(dtype=out.dtype)
        return delta * torch.tanh(self.gate).to(dtype=delta.dtype)


__all__ = ["PriorCrossAttention"]
