"""Candidate-specific, chain-balanced pDockQ prediction head."""

from __future__ import annotations

import torch
from torch import nn


class ChainBalancedInterfacePool(nn.Module):
    """Pool single/pair trunk features without weighting long chains more."""

    def forward(
        self,
        *,
        si: torch.Tensor,
        z: torch.Tensor,
        asym_id: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if si.ndim not in (3, 4) or z.ndim != si.ndim + 1:
            raise ValueError(
                "si must be [B,N,C] or [B,S,N,C] and z must add a pair axis"
            )
        if si.ndim == 3:
            si = si.unsqueeze(1)
            z = z.unsqueeze(1)
            squeeze_sample = True
        else:
            squeeze_sample = False
        b, samples, n, c_s = si.shape
        if z.shape[:3] != (b, samples, n) or z.shape[3] != n:
            raise ValueError("si/z token and pair dimensions do not match")
        if asym_id.ndim == 2:
            asym_id = asym_id[:, None, :].expand(b, samples, n)
        elif asym_id.ndim == 3 and asym_id.shape == (b, samples, n):
            pass
        else:
            raise ValueError("asym_id must be [B,N] or [B,S,N]")
        if token_mask.ndim == 2:
            token_mask = token_mask[:, None, :].expand(b, samples, n)
        elif token_mask.ndim == 3:
            if token_mask.shape == (b, 1, n):
                token_mask = token_mask.expand(b, samples, n)
            elif token_mask.shape != (b, samples, n):
                raise ValueError("token_mask must be [B,N] or [B,S,N]")
        else:
            raise ValueError("token_mask must be [B,N] or [B,S,N]")
        token_mask = token_mask.bool()

        pooled = []
        for batch_index in range(b):
            sample_rows = []
            for sample_index in range(samples):
                ids = asym_id[batch_index, sample_index]
                sample_valid = token_mask[batch_index, sample_index]
                chain_ids = torch.unique(ids[sample_valid])
                chain_single = []
                for chain_id in chain_ids:
                    mask = (ids == chain_id) & sample_valid
                    if mask.any():
                        chain_single.append(si[batch_index, sample_index, mask].mean(0))
                single = (
                    torch.stack(chain_single).mean(0)
                    if chain_single
                    else si.new_zeros(c_s)
                )

                chain_pairs = []
                for left_index, left_chain in enumerate(chain_ids):
                    left = (ids == left_chain) & sample_valid
                    for right_chain in chain_ids[left_index + 1 :]:
                        right = (ids == right_chain) & sample_valid
                        if left.any() and right.any():
                            chain_pairs.append(
                                z[batch_index, sample_index][left][:, right].mean(
                                    (0, 1)
                                )
                            )
                pair = (
                    torch.stack(chain_pairs).mean(0)
                    if chain_pairs
                    else z.new_zeros(z.shape[-1])
                )
                sample_rows.append(torch.cat((single, pair), dim=-1))
            pooled.append(torch.stack(sample_rows))
        result = torch.stack(pooled)
        return result[:, 0] if squeeze_sample else result


class PDockQHead(nn.Module):
    """Predict a 50-bin pDockQ distribution for every rollout candidate."""

    def __init__(self, c_s: int, c_z: int, no_bins: int = 50) -> None:
        super().__init__()
        if no_bins != 50:
            raise ValueError("PhysPPI pDockQ uses exactly 50 bins")
        self.pool = ChainBalancedInterfacePool()
        self.linear = nn.Linear(c_s + c_z, no_bins)
        self.register_buffer(
            "centers", torch.linspace(0.0, 1.0, no_bins), persistent=False
        )

    def forward(
        self,
        *,
        si: torch.Tensor,
        z: torch.Tensor,
        asym_id: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(
            self.pool(
                si=si,
                z=z,
                asym_id=asym_id,
                token_mask=token_mask,
            )
        )

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(device=logits.device, dtype=logits.dtype)
        return (torch.softmax(logits, dim=-1) * centers).sum(-1)


__all__ = ["ChainBalancedInterfacePool", "PDockQHead"]
