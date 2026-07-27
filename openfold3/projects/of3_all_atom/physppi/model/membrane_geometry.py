"""Directed membrane-plane geometry and the diffusion conditioning bridge.

The plane direction is defined by the ordered N--CA--C backbone frame.  No
global sign canonicalization is applied: reversing the ordered frame therefore
changes the directed normal.  All geometry is represented with row vectors,
matching OpenFold3's rigid-transform convention (``x' = x @ R + t``).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class DirectedPlanePrediction(NamedTuple):
    """Predicted directed plane and per-batch validity mask."""

    normal: torch.Tensor
    offset: torch.Tensor
    signed_depth: torch.Tensor
    normal_error_logits: torch.Tensor
    offset_error_logits: torch.Tensor
    valid: torch.Tensor


def _normalization_valid(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector.float(), dim=-1)
    return torch.isfinite(norm) & (norm > eps)


def normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors, returning an exact zero for degenerate vectors."""

    finite = torch.isfinite(vector).all(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(vector.float(), dim=-1, keepdim=True).to(
        dtype=vector.dtype
    )
    result = vector / norm.clamp_min(eps)
    return torch.where(finite & (norm > eps), result, torch.zeros_like(vector))


def _expand_token_indices(
    indices: torch.Tensor, *, batch_size: int, n_tokens: int, device: torch.device
) -> torch.Tensor:
    indices = indices.to(device=device, dtype=torch.long)
    if indices.shape == (n_tokens,):
        return indices.unsqueeze(0).expand(batch_size, -1)
    if indices.shape == (batch_size, n_tokens):
        return indices
    raise ValueError(
        "token atom indices must have shape "
        f"({n_tokens},) or ({batch_size}, {n_tokens}), got {tuple(indices.shape)}"
    )


def _expand_atom_mask(
    atom_mask: torch.Tensor | None,
    *,
    batch_size: int,
    n_atoms: int,
    device: torch.device,
) -> torch.Tensor:
    if atom_mask is None:
        return torch.ones(batch_size, n_atoms, device=device, dtype=torch.bool)
    atom_mask = atom_mask.to(device=device, dtype=torch.bool)
    if atom_mask.shape == (n_atoms,):
        return atom_mask.unsqueeze(0).expand(batch_size, -1)
    if atom_mask.shape == (batch_size, n_atoms):
        return atom_mask
    raise ValueError(
        "monomer_atom_mask must have shape "
        f"({n_atoms},) or ({batch_size}, {n_atoms}), got {tuple(atom_mask.shape)}"
    )


def _safe_gather(
    atom_positions: torch.Tensor,
    indices: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather finite, present atom coordinates and zero unsupported indices."""

    if atom_positions.ndim != 3 or atom_positions.shape[-1] != 3:
        raise ValueError(
            "atom_positions must have shape (batch, atoms, 3), got "
            f"{tuple(atom_positions.shape)}"
        )
    n_atoms = atom_positions.shape[1]
    if n_atoms == 0:
        shape = (*indices.shape, 3)
        return (
            atom_positions.new_zeros(shape),
            torch.zeros(indices.shape, device=atom_positions.device, dtype=torch.bool),
        )
    valid = (indices >= 0) & (indices < n_atoms)
    safe = indices.clamp(min=0, max=max(n_atoms - 1, 0))
    gathered = torch.gather(
        atom_positions,
        1,
        safe.unsqueeze(-1).expand(*safe.shape, 3),
    )
    expanded_mask = _expand_atom_mask(
        atom_mask,
        batch_size=atom_positions.shape[0],
        n_atoms=n_atoms,
        device=atom_positions.device,
    )
    present = torch.gather(expanded_mask, 1, safe.reshape(atom_positions.shape[0], -1))
    present = present.reshape(safe.shape)
    valid = valid & present & torch.isfinite(gathered).all(dim=-1)
    return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered)), valid


def gather_token_centers(
    atom_positions: torch.Tensor,
    token_center_atom_index: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather token centers; invalid indices produce exact zero coordinates."""

    if token_center_atom_index.ndim == 1:
        n_tokens = token_center_atom_index.shape[0]
    elif token_center_atom_index.ndim == 2:
        n_tokens = token_center_atom_index.shape[1]
    else:
        raise ValueError(
            "token_center_atom_index must have shape (tokens,) or (batch, tokens)"
        )
    index = _expand_token_indices(
        token_center_atom_index,
        batch_size=atom_positions.shape[0],
        n_tokens=n_tokens,
        device=atom_positions.device,
    )
    centers, _ = _safe_gather(atom_positions, index, atom_mask)
    return centers


def _token_center_valid(
    atom_positions: torch.Tensor,
    token_center_atom_index: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    n_tokens = (
        token_center_atom_index.shape[0]
        if token_center_atom_index.ndim == 1
        else token_center_atom_index.shape[1]
    )
    index = _expand_token_indices(
        token_center_atom_index,
        batch_size=atom_positions.shape[0],
        n_tokens=n_tokens,
        device=atom_positions.device,
    )
    return _safe_gather(atom_positions, index, atom_mask)[1]


def gather_backbone_triplets(
    atom_positions: torch.Tensor,
    backbone_atom_indices: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather ordered N, CA, C coordinates, zeroing invalid triplets."""

    if backbone_atom_indices.ndim == 2 and backbone_atom_indices.shape[-1] == 3:
        n_tokens = backbone_atom_indices.shape[0]
    elif backbone_atom_indices.ndim == 3 and backbone_atom_indices.shape[-1] == 3:
        n_tokens = backbone_atom_indices.shape[1]
    else:
        raise ValueError(
            "backbone_atom_indices must have shape (tokens, 3) or (batch, tokens, 3)"
        )
    index = backbone_atom_indices.to(device=atom_positions.device, dtype=torch.long)
    if index.shape == (n_tokens, 3):
        index = index.unsqueeze(0).expand(atom_positions.shape[0], -1, -1)
    elif index.shape != (atom_positions.shape[0], n_tokens, 3):
        raise ValueError(
            "backbone_atom_indices batch dimension does not match atom_positions"
        )
    gathered, valid = _safe_gather(
        atom_positions,
        index.reshape(atom_positions.shape[0], -1),
        atom_mask,
    )
    gathered = gathered.reshape(atom_positions.shape[0], n_tokens, 3, 3)
    valid = valid.reshape(atom_positions.shape[0], n_tokens, 3)
    # A frame is usable only when all ordered N/CA/C atoms are present.  Zero
    # the complete triplet for partial support so callers cannot accidentally
    # consume a mixed real/placeholder frame.
    triplet_valid = valid.all(-1, keepdim=True).unsqueeze(-1)
    gathered = torch.where(triplet_valid, gathered, torch.zeros_like(gathered))
    return gathered[..., 0, :], gathered[..., 1, :], gathered[..., 2, :]


def _backbone_valid(
    atom_positions: torch.Tensor,
    backbone_atom_indices: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    index = backbone_atom_indices.to(device=atom_positions.device, dtype=torch.long)
    n_tokens = index.shape[-2]
    if index.shape == (n_tokens, 3):
        index = index.unsqueeze(0).expand(atom_positions.shape[0], -1, -1)
    if index.shape != (atom_positions.shape[0], n_tokens, 3):
        raise ValueError(
            "backbone_atom_indices batch dimension does not match atom_positions"
        )
    _, valid = _safe_gather(
        atom_positions,
        index.reshape(atom_positions.shape[0], -1),
        atom_mask,
    )
    return valid.reshape(atom_positions.shape[0], n_tokens, 3).all(-1)


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Softmax over a mask, with an exact all-zero result for empty support."""

    mask = mask.to(device=logits.device, dtype=torch.bool)
    safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
    safe_logits = torch.where(
        torch.isfinite(safe_logits), safe_logits, torch.zeros_like(safe_logits)
    )
    very_small = torch.finfo(safe_logits.dtype).min
    masked_logits = safe_logits.masked_fill(~mask, very_small)
    max_logits = masked_logits.max(dim=dim, keepdim=True).values
    max_logits = torch.where(
        mask.any(dim=dim, keepdim=True), max_logits, torch.zeros_like(max_logits)
    )
    weights = torch.exp(masked_logits - max_logits) * mask.to(safe_logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )


class DirectedMembranePlaneEstimator(nn.Module):
    """Pool ordered backbone-frame vectors into a directed membrane plane."""

    def __init__(self, c_s: int, normal_bins: int = 16, offset_bins: int = 31) -> None:
        super().__init__()
        self.vector_coeff = nn.Linear(c_s, 3)
        self.vector_weight = nn.Linear(c_s, 1)
        self.offset_head = nn.Linear(c_s, 1)
        self.normal_error = nn.Linear(c_s, normal_bins)
        self.offset_error = nn.Linear(c_s, offset_bins)

    def forward(
        self,
        *,
        token_repr: torch.Tensor,
        monomer_atom_positions: torch.Tensor,
        monomer_atom_mask: torch.Tensor,
        token_center_atom_index: torch.Tensor,
        backbone_atom_indices: torch.Tensor,
        backbone_frame_mask: torch.Tensor,
        token_mask: torch.Tensor,
        tm_token_mask: torch.Tensor,
        membrane_on: torch.Tensor | bool,
    ) -> DirectedPlanePrediction:
        if token_repr.ndim != 3 or monomer_atom_positions.ndim != 3:
            raise ValueError(
                "token_repr and monomer_atom_positions must be batched tensors"
            )
        b, n, c_s = token_repr.shape
        if (
            monomer_atom_positions.shape[0] != b
            or token_repr.shape[-1] != self.vector_coeff.in_features
        ):
            raise ValueError(
                "token_repr and monomer_atom_positions have incompatible batch shapes"
            )

        monomer_atom_positions = monomer_atom_positions.to(
            device=token_repr.device, dtype=torch.float32
        )
        monomer_atom_mask = monomer_atom_mask.to(
            device=token_repr.device, dtype=torch.bool
        )
        centers = gather_token_centers(
            monomer_atom_positions, token_center_atom_index, monomer_atom_mask
        )
        center_valid = _token_center_valid(
            monomer_atom_positions, token_center_atom_index, monomer_atom_mask
        )
        n_xyz, ca_xyz, c_xyz = gather_backbone_triplets(
            monomer_atom_positions, backbone_atom_indices, monomer_atom_mask
        )
        triplet_valid = _backbone_valid(
            monomer_atom_positions, backbone_atom_indices, monomer_atom_mask
        )

        e1_raw = c_xyz - ca_xyz
        n_dir_raw = n_xyz - ca_xyz
        e1 = normalize(e1_raw)
        n_dir = normalize(n_dir_raw)
        e2_raw = n_dir - (n_dir * e1).sum(-1, keepdim=True) * e1
        e2 = normalize(e2_raw)
        e3 = normalize(torch.cross(e1, e2, dim=-1))
        frame_valid = (
            triplet_valid
            & _normalization_valid(e1_raw)
            & _normalization_valid(n_dir_raw)
            & _normalization_valid(e2_raw)
            & _normalization_valid(e3)
        )

        token_mask = token_mask.to(device=token_repr.device, dtype=torch.bool)
        tm_token_mask = tm_token_mask.to(device=token_repr.device, dtype=torch.bool)
        backbone_frame_mask = backbone_frame_mask.to(
            device=token_repr.device, dtype=torch.bool
        )
        valid = (
            token_mask
            & tm_token_mask
            & backbone_frame_mask
            & center_valid
            & frame_valid
        )
        membrane_on = (
            torch.as_tensor(membrane_on, device=token_repr.device).bool().reshape(-1)
        )
        if membrane_on.shape != (b,):
            raise ValueError(
                f"membrane_on must have shape ({b},), got {tuple(membrane_on.shape)}"
            )

        frame = torch.stack((e1, e2, e3), dim=-2)
        # Keep all coordinate-valued reductions in FP32 even under the
        # surrounding BF16 training autocast.  A tiny low-precision error in
        # weights summing to one otherwise gets multiplied by the absolute
        # coordinate origin and destroys translation equivariance.
        with torch.autocast(device_type=token_repr.device.type, enabled=False):
            geometry_repr = token_repr.float()
            coeff = self.vector_coeff(geometry_repr)
            weight_logits = self.vector_weight(geometry_repr).squeeze(-1)
        # ``frame`` stores basis vectors in axis -2, so contract coefficients
        # over that basis axis (not over xyz components).
        vectors = (coeff.unsqueeze(-1) * frame).sum(-2)
        weights = masked_softmax(weight_logits, valid, dim=-1)
        raw_normal = (weights.unsqueeze(-1) * vectors).sum(-2)
        normal = normalize(raw_normal)
        pooled = (weights.unsqueeze(-1) * geometry_repr).sum(-2)
        point = (weights.unsqueeze(-1) * ca_xyz).sum(-2)
        with torch.autocast(device_type=token_repr.device.type, enabled=False):
            offset_residual = self.offset_head(pooled).squeeze(-1)
            normal_error_logits = self.normal_error(pooled)
            offset_error_logits = self.offset_error(pooled)
        offset = (normal * point).sum(-1) + offset_residual
        signed_depth = (centers * normal.unsqueeze(-2)).sum(-1) - offset.unsqueeze(-1)
        required = token_mask & tm_token_mask
        # A plane is a batch-level object.  If any required TM token lacks a
        # valid center/frame, bypass the whole plane rather than silently
        # fitting a partial subset and leaking a misleading direction.
        active = (
            membrane_on
            & required.any(-1)
            & (valid | ~required).all(-1)
            & torch.isfinite(normal).all(-1)
        )

        normal_mask = active.unsqueeze(-1)
        token_depth_mask = (token_mask & center_valid & active.unsqueeze(-1)).to(
            signed_depth.dtype
        )
        normal = torch.where(normal_mask, normal, torch.zeros_like(normal))
        offset = torch.where(active, offset, torch.zeros_like(offset))
        signed_depth = torch.where(
            token_depth_mask.bool(), signed_depth, torch.zeros_like(signed_depth)
        )
        normal_error_logits = torch.where(
            normal_mask, normal_error_logits, torch.zeros_like(normal_error_logits)
        )
        offset_error_logits = torch.where(
            normal_mask, offset_error_logits, torch.zeros_like(offset_error_logits)
        )
        return DirectedPlanePrediction(
            normal,
            offset,
            signed_depth,
            normal_error_logits,
            offset_error_logits,
            active,
        )


class MembraneDiffusionConditioner(nn.Module):
    """Zero-gated signed-depth residual for diffusion single representations."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        *,
        num_depth_bins: int = 16,
        depth_cutoff: float = 30.0,
    ) -> None:
        super().__init__()
        del c_z  # The bridge intentionally modifies only the single representation.
        self.num_depth_bins = num_depth_bins
        self.depth_cutoff = depth_cutoff
        self.register_buffer(
            "depth_centers",
            torch.linspace(-depth_cutoff, depth_cutoff, num_depth_bins),
            persistent=False,
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(num_depth_bins, c_s),
            nn.LayerNorm(c_s),
            nn.GELU(),
            nn.Linear(c_s, c_s),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def _depth_features(self, depth: torch.Tensor) -> torch.Tensor:
        centers = self.depth_centers.to(device=depth.device, dtype=depth.dtype)
        width = self.depth_cutoff * 2.0 / max(self.num_depth_bins - 1, 1)
        gamma = 0.5 / max(width * width, 1e-6)
        return torch.exp(
            -gamma
            * (
                depth.unsqueeze(-1).clamp(-self.depth_cutoff, self.depth_cutoff)
                - centers
            )
            ** 2
        )

    @staticmethod
    def _prediction_field(prediction, name: str):
        if isinstance(prediction, dict):
            return prediction.get(name)
        return getattr(prediction, name, None)

    def forward(
        self,
        *,
        si: torch.Tensor,
        zij: torch.Tensor,
        batch: dict[str, torch.Tensor | DirectedPlanePrediction],
        token_mask: torch.Tensor,
        atom_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del atom_positions
        membrane_on = batch.get("physppi_membrane_on")
        prediction = batch.get("physppi_predicted_plane")
        if membrane_on is None or prediction is None:
            return si, zij
        signed_depth = self._prediction_field(prediction, "signed_depth")
        prediction_valid = self._prediction_field(prediction, "valid")
        if signed_depth is None:
            return si, zij
        signed_depth = signed_depth.to(device=si.device, dtype=si.dtype)
        target_shape = si.shape[:-1]
        while signed_depth.ndim < len(target_shape):
            signed_depth = signed_depth.unsqueeze(-2)
        try:
            signed_depth = signed_depth.expand(target_shape)
        except RuntimeError as error:
            raise ValueError(
                "predicted signed_depth must broadcast to si's leading/token "
                f"dimensions, got {tuple(signed_depth.shape)} and {target_shape}"
            ) from error

        active_shape = target_shape[:-1]
        active = torch.as_tensor(membrane_on, device=si.device).bool()
        while active.ndim < len(active_shape):
            active = active.unsqueeze(-1)
        try:
            active = active.expand(active_shape)
        except RuntimeError as error:
            raise ValueError(
                "physppi_membrane_on must broadcast to si's leading dimensions, "
                f"got {tuple(active.shape)} and {active_shape}"
            ) from error
        if prediction_valid is not None:
            prediction_valid = torch.as_tensor(
                prediction_valid, device=si.device
            ).bool()
            while prediction_valid.ndim < len(active_shape):
                prediction_valid = prediction_valid.unsqueeze(-1)
            try:
                prediction_valid = prediction_valid.expand(active_shape)
            except RuntimeError as error:
                raise ValueError(
                    "predicted plane validity must broadcast to si's leading "
                    f"dimensions, got {tuple(prediction_valid.shape)} and "
                    f"{active_shape}"
                ) from error
            active = active & prediction_valid
        if not active.any():
            return si, zij
        token_mask = token_mask.to(device=si.device, dtype=torch.bool)
        while token_mask.ndim < len(target_shape):
            token_mask = token_mask.unsqueeze(-2)
        try:
            token_mask = token_mask.expand(target_shape)
        except RuntimeError as error:
            raise ValueError(
                "token_mask must broadcast to si's leading/token dimensions, got "
                f"{tuple(token_mask.shape)} and {target_shape}"
            ) from error
        supported = token_mask & active.unsqueeze(-1)
        if not supported.any():
            return si, zij
        delta = torch.zeros_like(si)
        features = self._depth_features(signed_depth[supported])
        delta[supported] = self.depth_proj(features)
        si = si + torch.tanh(self.gate).to(si.dtype) * delta
        return si, zij


__all__ = [
    "DirectedPlanePrediction",
    "DirectedMembranePlaneEstimator",
    "MembraneDiffusionConditioner",
    "gather_backbone_triplets",
    "gather_token_centers",
    "masked_softmax",
    "normalize",
]
