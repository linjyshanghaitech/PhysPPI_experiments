"""Losses for the label-only PhysPPI membrane plane."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def directed_membrane_plane_loss(
    *,
    batch: dict,
    output: dict,
    normal_weight: float = 1.0,
    offset_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervise a directed plane normal and signed offset when valid.

    The target is label-only: it is read from the aligned complex GT frame and
    is never consumed by the diffusion input path.  Invalid/non-membrane
    examples return an exact zero connected to the prediction graph.  The
    The ordered monomer backbone frame fixes the normal sign.  Therefore
    ``(normal, offset)`` and ``(-normal, -offset)`` are intentionally distinct.
    """

    predicted_normal = output.get("physppi_predicted_membrane_normal")
    predicted_offset = output.get("physppi_predicted_membrane_offset")
    predicted_valid = output.get("physppi_predicted_membrane_valid")
    target_normal = batch.get("physppi_membrane_gt_normal")
    target_point = batch.get("physppi_membrane_gt_point")
    valid = batch.get("physppi_membrane_gt_valid")
    if predicted_normal is None or predicted_offset is None or target_normal is None:
        reference = next(
            (value for value in output.values() if torch.is_tensor(value)), None
        )
        return reference.sum() * 0.0 if reference is not None else torch.zeros(())

    predicted_normal = predicted_normal.float()
    if predicted_normal.shape[-1:] != (3,):
        raise ValueError(
            "physppi_predicted_membrane_normal must have a final dimension of 3, got "
            f"{tuple(predicted_normal.shape)}"
        )
    predicted_normal = predicted_normal.reshape(-1, 3)
    predicted_offset = predicted_offset.float().reshape(-1)
    if predicted_offset.numel() != predicted_normal.shape[0]:
        raise ValueError(
            "physppi_predicted_membrane_offset must contain one value per "
            "predicted normal, "
            f"got {predicted_offset.numel()} and {predicted_normal.shape[0]}"
        )
    target_normal = torch.as_tensor(
        target_normal, device=predicted_normal.device, dtype=predicted_normal.dtype
    )
    if target_normal.shape[-1:] != (3,):
        raise ValueError(
            "physppi_membrane_gt_normal must have a final dimension of 3, got "
            f"{tuple(target_normal.shape)}"
        )
    target_normal = target_normal.reshape(-1, 3)
    if target_normal.shape[0] != predicted_normal.shape[0]:
        raise ValueError(
            "predicted and target membrane normal batch sizes differ: "
            f"{predicted_normal.shape[0]} and {target_normal.shape[0]}"
        )
    if target_point is None:
        target_offset = torch.zeros_like(predicted_offset)
    else:
        target_point = torch.as_tensor(
            target_point,
            device=predicted_normal.device,
            dtype=predicted_normal.dtype,
        )
        if target_point.shape[-1:] != (3,):
            raise ValueError(
                "physppi_membrane_gt_point must have a final dimension of 3, got "
                f"{tuple(target_point.shape)}"
            )
        target_point = target_point.reshape(-1, 3)
        if target_point.shape[0] != predicted_normal.shape[0]:
            raise ValueError(
                "predicted and target membrane point batch sizes differ: "
                f"{predicted_normal.shape[0]} and {target_point.shape[0]}"
            )
        target_offset = (target_normal * target_point).sum(-1).reshape(-1)
    if valid is None:
        valid_mask = torch.ones(
            predicted_offset.shape, device=predicted_offset.device, dtype=torch.bool
        )
    else:
        valid_mask = (
            torch.as_tensor(valid, device=predicted_offset.device).bool().reshape(-1)
        )
        if valid_mask.numel() != predicted_normal.shape[0]:
            raise ValueError(
                "physppi_membrane_gt_valid must contain one value per target normal, "
                f"got {valid_mask.numel()} and {predicted_normal.shape[0]}"
            )
    if predicted_valid is None:
        predicted_valid_mask = torch.ones_like(valid_mask)
    else:
        predicted_valid_mask = (
            torch.as_tensor(predicted_valid, device=predicted_offset.device)
            .bool()
            .reshape(-1)
        )
        if predicted_valid_mask.numel() != predicted_normal.shape[0]:
            raise ValueError(
                "physppi_predicted_membrane_valid must contain one value per "
                f"predicted normal, got {predicted_valid_mask.numel()} and "
                f"{predicted_normal.shape[0]}"
            )
    target_valid = (
        valid_mask
        & predicted_valid_mask
        & torch.isfinite(target_offset)
        & torch.isfinite(target_normal).all(-1)
        & (torch.linalg.vector_norm(target_normal, dim=-1) > eps)
    )
    if not target_valid.any():
        return (
            torch.nan_to_num(predicted_offset).sum() * 0.0
            + torch.nan_to_num(predicted_normal).sum() * 0.0
        )

    # Sanitise only rows excluded from supervision.  A NaN in a valid
    # prediction remains visible and fails the training stability gate.
    safe_predicted_normal = torch.where(
        target_valid.unsqueeze(-1), predicted_normal, torch.zeros_like(predicted_normal)
    )
    safe_predicted_offset = torch.where(
        target_valid, predicted_offset, torch.zeros_like(predicted_offset)
    )
    safe_target_normal = torch.where(
        target_valid.unsqueeze(-1), target_normal, torch.zeros_like(target_normal)
    )
    safe_target_offset = torch.where(
        target_valid, target_offset, torch.zeros_like(target_offset)
    )
    safe_target_normal = F.normalize(safe_target_normal, dim=-1, eps=eps)
    safe_predicted_normal = F.normalize(safe_predicted_normal, dim=-1, eps=eps)

    dot = (safe_predicted_normal * safe_target_normal).sum(-1).clamp(-1.0, 1.0)
    plane_error = normal_weight * (1.0 - dot) + offset_weight * F.smooth_l1_loss(
        safe_predicted_offset, safe_target_offset, reduction="none"
    )
    mask = target_valid.to(plane_error.dtype)
    # Preserve NaNs from a valid prediction so the training stability harness
    # can fail loudly; only invalid GT rows are zeroed before reduction.
    plane_error = torch.where(target_valid, plane_error, torch.zeros_like(plane_error))
    return (plane_error * mask).sum() / mask.sum().clamp_min(1.0)


__all__ = ["directed_membrane_plane_loss"]
