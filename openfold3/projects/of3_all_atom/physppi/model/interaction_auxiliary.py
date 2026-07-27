"""Typed interaction distance supervision with a strict empty-batch boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InteractionValidation:
    """Validated interaction records and their boolean supervision mask."""

    valid_mask: torch.Tensor

    @property
    def valid(self) -> torch.Tensor:
        return self.valid_mask


def _record_value(record, *names, default=None):
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
    return default


def _endpoint_atom_name(atom_name_lookup, i, j, endpoint: str):
    candidates = (
        (i, j, endpoint),
        (i, j, 0 if endpoint == "a1" else 1),
        (j, i, endpoint),
        (j, i, 0 if endpoint == "a1" else 1),
    )
    for key in candidates:
        if key in atom_name_lookup:
            return atom_name_lookup[key]
    return None


def validate_interaction_endpoints(
    records: Sequence[object],
    atom_name_lookup: Mapping[object, str],
) -> InteractionValidation:
    """Validate residue endpoints and atom names without silently remapping them."""

    valid = []
    for record in records:
        i = _record_value(record, "i", "r1")
        j = _record_value(record, "j", "r2")
        atom_1 = _record_value(record, "atom_1", "a1")
        atom_2 = _record_value(record, "atom_2", "a2")
        distance = _record_value(record, "distance")
        valid.append(
            i is not None
            and j is not None
            and atom_1 is not None
            and atom_2 is not None
            and distance is not None
            and _endpoint_atom_name(atom_name_lookup, i, j, "a1") == atom_1
            and _endpoint_atom_name(atom_name_lookup, i, j, "a2") == atom_2
        )
    return InteractionValidation(torch.tensor(valid, dtype=torch.bool))


class InteractionDistanceHead(nn.Module):
    """Predict one scalar atom-distance auxiliary value per pair representation."""

    def __init__(self, pair_dim: int):
        super().__init__()
        if pair_dim <= 0:
            raise ValueError("pair_dim must be positive")
        self.projection = nn.Linear(pair_dim, 1)

    def forward(
        self,
        pair_repr: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pair_repr.ndim < 4 or pair_repr.shape[-3] != pair_repr.shape[-2]:
            raise ValueError(
                "pair_repr must have shape (..., tokens, tokens, channels), got "
                f"{tuple(pair_repr.shape)}"
            )
        if pair_repr.shape[-1] != self.projection.in_features:
            raise ValueError(
                "pair_repr channel dimension does not match the head, got "
                f"{pair_repr.shape[-1]} and {self.projection.in_features}"
            )
        n_tokens = pair_repr.shape[-2]
        expected_mask_shape = (*pair_repr.shape[:-3], n_tokens)
        token_mask = token_mask.to(device=pair_repr.device, dtype=torch.bool)
        if token_mask.shape != expected_mask_shape:
            raise ValueError(
                "token_mask must match pair_repr leading/token dimensions, got "
                f"{tuple(token_mask.shape)} and {expected_mask_shape}"
            )
        prediction = self.projection(pair_repr).squeeze(-1)
        prediction = 0.5 * (prediction + prediction.transpose(-1, -2))
        pair_support = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        return prediction * pair_support.to(dtype=prediction.dtype)


def _build_pair_interaction_auxiliary(
    aux_edges: Sequence[Sequence[object]],
    *,
    n_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build symmetric pair targets, averaging duplicate residue-pair edges."""

    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")
    distance_sum = torch.zeros(n_tokens, n_tokens, dtype=torch.float32)
    distance_count = torch.zeros(n_tokens, n_tokens, dtype=torch.int64)
    for edge in aux_edges:
        if len(edge) < 4:
            raise ValueError(
                "interaction auxiliary edge must contain i, j, type, distance"
            )
        left, right, _, distance = edge[:4]
        left, right = int(left), int(right)
        distance = float(distance)
        if (
            left < 0
            or right < 0
            or left >= n_tokens
            or right >= n_tokens
            or left == right
            or not math.isfinite(distance)
        ):
            raise ValueError("interaction auxiliary edge is out of range or non-finite")
        distance_sum[left, right] += distance
        distance_sum[right, left] += distance
        distance_count[left, right] += 1
        distance_count[right, left] += 1
    valid = distance_count > 0
    target = torch.where(
        valid,
        distance_sum / distance_count.clamp_min(1).to(distance_sum.dtype),
        torch.zeros_like(distance_sum),
    )
    return target, valid


def interaction_auxiliary_loss(
    *,
    batch: Mapping[str, torch.Tensor],
    output: Mapping[str, torch.Tensor],
    distance_weight: float = 1.0,
) -> torch.Tensor:
    """Return a connected exact zero when no valid interaction is supervised."""

    prediction = output["interaction_distance_pred"]
    valid = torch.as_tensor(
        batch.get("physppi_interaction_aux_valid", False),
        device=prediction.device,
        dtype=torch.bool,
    )
    target = batch.get("physppi_interaction_distance")
    if target is None:
        return prediction.sum() * 0.0
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if valid.ndim == 0:
        valid = valid.reshape(1).expand_as(prediction)
    if valid.shape != prediction.shape or target.shape != prediction.shape:
        raise ValueError(
            "interaction auxiliary tensors must share the prediction shape: "
            f"pred={tuple(prediction.shape)}, target={tuple(target.shape)}, "
            f"valid={tuple(valid.shape)}"
        )
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    return float(distance_weight) * error.masked_select(valid).mean()
