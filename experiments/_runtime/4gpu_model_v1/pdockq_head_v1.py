"""PhysPPI pDockQ prediction head v1 — si-only regression, LayerNorm, zero-init.

Instead of 50-bin classification (CE ≥ ln(50) ≈ 3.9), this head predicts a
single scalar via sigmoid and is trained with MSE loss.  Output is strictly
in [0, 1] and the loss starts near 0.01–0.05 (depending on how far the
initial 0.5 prediction is from the true DockQ distribution).

Architecture:
    si → LayerNorm → per-chain mean pool → mean across chains
      → Linear(c_s, 1) → Sigmoid → scalar ∈ [0,1]
"""

from __future__ import annotations

import torch
from torch import nn


class PDockQHead(nn.Module):
    """Predict pDockQ as a scalar ∈ [0,1] from single representation only."""

    def __init__(self, c_s: int, c_z: int = 0, no_bins: int = 50) -> None:
        # c_z, no_bins accepted for API compatibility with original PDockQHead.
        _ = c_z
        _ = no_bins
        super().__init__()
        self.c_s = c_s

        self.layer_norm = nn.LayerNorm(c_s)
        self.linear = nn.Linear(c_s, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        *,
        si: torch.Tensor,
        z: torch.Tensor | None = None,
        asym_id: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw logit (pre-sigmoid) of shape [B, 1] or [B, S, 1].

        ``z`` is accepted for API compatibility but ignored.
        """
        _ = z

        if si.ndim == 3:
            si = si.unsqueeze(1)
            squeeze_sample = True
        elif si.ndim == 4:
            squeeze_sample = False
        else:
            raise ValueError(
                f"si must have 3 or 4 dimensions, got shape {tuple(si.shape)}"
            )

        b, s, n, c_s_in = si.shape
        if c_s_in != self.c_s:
            raise ValueError(
                f"si last dim {c_s_in} does not match head c_s={self.c_s}"
            )

        if asym_id.ndim == 2:
            asym_id = asym_id[:, None, :].expand(b, s, n)
        elif asym_id.ndim == 3 and asym_id.shape == (b, s, n):
            pass
        elif asym_id.ndim == 3 and asym_id.shape[1] == 1:
            asym_id = asym_id.expand(b, s, n)
        else:
            raise ValueError("asym_id must be [B,N], [B,1,N], or [B,S,N]")

        if token_mask.ndim == 2:
            token_mask = token_mask[:, None, :].expand(b, s, n)
        elif token_mask.ndim == 3 and token_mask.shape[1] == 1:
            token_mask = token_mask.expand(b, s, n)
        elif token_mask.ndim == 3 and token_mask.shape == (b, s, n):
            pass
        else:
            raise ValueError("token_mask must be [B,N], [B,1,N], or [B,S,N]")

        token_mask = token_mask.bool()

        si_norm = self.layer_norm(si)

        pooled = si_norm.new_empty(b, s, self.c_s)
        for bi in range(b):
            for si_idx in range(s):
                ids = asym_id[bi, si_idx]
                mask = token_mask[bi, si_idx]
                chain_ids = torch.unique(ids[mask])
                chain_means = []
                for cid in chain_ids:
                    chain_mask = (ids == cid) & mask
                    if chain_mask.any():
                        chain_means.append(
                            si_norm[bi, si_idx, chain_mask].mean(0)
                        )
                if chain_means:
                    pooled[bi, si_idx] = torch.stack(chain_means).mean(0)
                else:
                    pooled[bi, si_idx] = 0.0

        logit = self.linear(pooled)  # [B, S, 1]
        return logit[:, 0] if squeeze_sample else logit

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to get predicted DockQ ∈ [0, 1].

        Args:
            logits: ``[..., 1]`` pre-sigmoid values.

        Returns:
            ``[...]``  predicted DockQ ∈ [0, 1].
        """
        return self.sigmoid(logits).squeeze(-1)


__all__ = ["PDockQHead"]
