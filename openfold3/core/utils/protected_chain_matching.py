# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hard compatibility and representative-anchor support for chain matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from openfold3.core.data.resources.residues import (
    DNA_RESTYPE_1TO3,
    DNA_RESTYPE_3TO1,
    PROTEIN_RESTYPE_1TO3,
    PROTEIN_RESTYPE_3TO1,
    RNA_RESTYPE_1TO3,
    RNA_RESTYPE_3TO1,
)
from openfold3.core.data.tools.kalign import run_kalign


class ProtectedChainMatchError(RuntimeError):
    """Base error for protected chain matching."""


class InsufficientRepresentativeAnchors(ProtectedChainMatchError):
    """Raised when a chain lacks finite, non-collinear anchor support."""


class NoLegalSupervisedCorrespondence(ProtectedChainMatchError):
    """Raised when no complete legal source-to-target correspondence exists."""


@dataclass(frozen=True)
class ChainRecord:
    """Sequence and representative-coordinate data for one chain."""

    chain_id: str
    polymer_type: str
    sequence: tuple[str, ...]
    representative_xyz: torch.Tensor
    representative_mask: torch.Tensor
    residue_keys: tuple[tuple[str, int, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate row-wise record invariants before matching."""
        if not isinstance(self.sequence, tuple) or any(
            not isinstance(residue, str) for residue in self.sequence
        ):
            raise ProtectedChainMatchError("chain_record_sequence")
        if (
            not isinstance(self.representative_xyz, torch.Tensor)
            or self.representative_xyz.ndim != 2
            or self.representative_xyz.shape[1] != 3
        ):
            raise ProtectedChainMatchError("chain_record_representative_xyz_shape")
        if (
            not isinstance(self.representative_mask, torch.Tensor)
            or self.representative_mask.ndim != 1
        ):
            raise ProtectedChainMatchError("chain_record_representative_mask_shape")

        row_count = len(self.sequence)
        if (
            self.representative_xyz.shape[0] != row_count
            or self.representative_mask.shape[0] != row_count
        ):
            raise ProtectedChainMatchError("chain_record_length_mismatch")
        if not isinstance(self.residue_keys, tuple):
            raise ProtectedChainMatchError("chain_record_residue_key_shape")
        if self.residue_keys and len(self.residue_keys) != row_count:
            raise ProtectedChainMatchError("chain_record_residue_keys_length")
        if any(
            not isinstance(key, tuple)
            or len(key) != 3
            or not isinstance(key[0], str)
            or not isinstance(key[1], int)
            or not isinstance(key[2], str)
            for key in self.residue_keys
        ):
            raise ProtectedChainMatchError("chain_record_residue_key_shape")


@dataclass(frozen=True)
class PairCompatibility:
    """Hard compatibility result and its exact common anchor support."""

    allowed: bool
    reason: str
    aligned_source_rows: torch.Tensor
    aligned_target_rows: torch.Tensor
    common_representative_count: int


@dataclass(frozen=True)
class ResidueAlignment:
    """Deterministic sequence-aligned row correspondence for two chains."""

    aligned_source_rows: torch.Tensor
    aligned_target_rows: torch.Tensor
    has_conflicting_known_residue: bool

    def common_finite_representative_rows(
        self, pred: ChainRecord, gt: ChainRecord
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned, known rows with finite resolved representatives."""
        source_rows = self.aligned_source_rows.to(pred.representative_xyz.device)
        target_rows = self.aligned_target_rows.to(gt.representative_xyz.device)
        if source_rows.numel() == 0:
            return source_rows, target_rows

        row_pairs = zip(
            source_rows.detach().cpu().tolist(),
            target_rows.detach().cpu().tolist(),
            strict=True,
        )
        known_match_values = [
            _canonical_residue(pred.polymer_type, pred.sequence[source_row])
            == _canonical_residue(gt.polymer_type, gt.sequence[target_row])
            is not None
            for source_row, target_row in row_pairs
        ]
        known_matches = torch.tensor(
            known_match_values,
            dtype=torch.bool,
            device=source_rows.device,
        )
        source_valid = (
            known_matches
            & pred.representative_mask[source_rows.to(pred.representative_mask.device)]
            .bool()
            .to(source_rows.device)
            & torch.isfinite(pred.representative_xyz[source_rows]).all(-1)
        )
        target_valid = gt.representative_mask[
            target_rows.to(gt.representative_mask.device)
        ].bool().to(target_rows.device) & torch.isfinite(
            gt.representative_xyz[target_rows]
        ).all(-1)
        source_valid &= target_valid.to(source_valid.device)
        return source_rows[source_valid], target_rows[
            source_valid.to(target_rows.device)
        ]


@dataclass(frozen=True)
class ProtectedMatchResult:
    """A quality-gated deterministic chain correspondence."""

    mapping: dict[str, str]
    identity_rmsd: float
    candidate_rmsd: float
    matched_coverage: int
    accepted: bool
    reason: str


RigidTransform: TypeAlias = tuple[np.ndarray, np.ndarray]


def chain_record_from_atom_metadata(
    *,
    chain_id: str,
    polymer_type: str,
    sequence: tuple[str, ...],
    atom_xyz: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_names: tuple[str, ...],
    atom_to_token_index: torch.Tensor,
    token_center_atom_mask: torch.Tensor,
    residue_keys: tuple[tuple[str, int, str], ...] = (),
) -> ChainRecord:
    """Build a record using the plan's polymer-specific representatives."""
    atom_count = len(atom_names)
    if not isinstance(atom_names, tuple) or any(
        not isinstance(atom_name, str) for atom_name in atom_names
    ):
        raise ProtectedChainMatchError("chain_record_atom_names")
    if (
        not isinstance(atom_xyz, torch.Tensor)
        or atom_xyz.ndim != 2
        or atom_xyz.shape[1] != 3
        or not isinstance(atom_mask, torch.Tensor)
        or atom_mask.ndim != 1
        or not isinstance(atom_to_token_index, torch.Tensor)
        or atom_to_token_index.ndim != 1
        or not isinstance(token_center_atom_mask, torch.Tensor)
        or token_center_atom_mask.ndim != 1
    ):
        raise ProtectedChainMatchError("chain_record_atom_metadata_shape")
    if any(
        length != atom_count
        for length in (
            atom_xyz.shape[0],
            atom_mask.shape[0],
            atom_to_token_index.shape[0],
            token_center_atom_mask.shape[0],
        )
    ):
        raise ProtectedChainMatchError("chain_record_atom_metadata_length")
    if atom_to_token_index.dtype.is_floating_point or (
        atom_to_token_index.dtype == torch.bool
    ):
        raise ProtectedChainMatchError("chain_record_atom_to_token_index")

    token_rows = atom_to_token_index.detach().cpu().tolist()
    if any(token_row < 0 or token_row >= len(sequence) for token_row in token_rows):
        raise ProtectedChainMatchError("chain_record_atom_to_token_index")
    center_flags = token_center_atom_mask.detach().cpu().bool().tolist()
    atom_mask_values = atom_mask.detach().cpu().bool().tolist()
    representative_atom_name = {
        "protein": "CA",
        "rna": "C4'",
        "dna": "C4'",
    }.get(polymer_type.lower())

    representative_xyz: list[torch.Tensor] = []
    representative_mask: list[bool] = []
    for token_index in range(len(sequence)):
        candidates = [
            atom_index
            for atom_index, (atom_name, token_row, is_token_center) in enumerate(
                zip(atom_names, token_rows, center_flags, strict=True)
            )
            if token_row == token_index
            and (
                atom_name.strip().upper() == representative_atom_name
                if representative_atom_name is not None
                else is_token_center
            )
        ]
        if len(candidates) > 1:
            raise ProtectedChainMatchError("chain_record_representative_atom_count")
        if not candidates:
            representative_xyz.append(atom_xyz.new_zeros(3))
            representative_mask.append(False)
            continue

        atom_index = candidates[0]
        representative_xyz.append(atom_xyz[atom_index])
        representative_mask.append(atom_mask_values[atom_index])

    xyz = (
        torch.stack(representative_xyz)
        if representative_xyz
        else atom_xyz.new_empty((0, 3))
    )
    mask = torch.tensor(
        representative_mask,
        dtype=torch.bool,
        device=atom_xyz.device,
    )
    return ChainRecord(
        chain_id=chain_id,
        polymer_type=polymer_type,
        sequence=sequence,
        representative_xyz=xyz,
        representative_mask=mask,
        residue_keys=residue_keys,
    )


def _canonical_residue(polymer_type: str, residue: str) -> str | None:
    """Return canonical residue identity, or ``None`` for unknown identity."""
    polymer_type = polymer_type.lower()
    residue = residue.strip().upper()

    if polymer_type == "protein":
        if residue in PROTEIN_RESTYPE_3TO1:
            one_letter = PROTEIN_RESTYPE_3TO1[residue]
        elif residue in PROTEIN_RESTYPE_1TO3:
            one_letter = residue
        else:
            return None
        return None if one_letter == "X" else one_letter

    if polymer_type == "dna":
        if residue in DNA_RESTYPE_3TO1:
            one_letter = DNA_RESTYPE_3TO1[residue]
        elif residue in DNA_RESTYPE_1TO3:
            one_letter = residue
        else:
            return None
        return None if one_letter == "N" else one_letter

    if polymer_type == "rna":
        if residue in RNA_RESTYPE_3TO1:
            one_letter = RNA_RESTYPE_3TO1[residue]
        elif residue in RNA_RESTYPE_1TO3:
            one_letter = residue
        else:
            return None
        return None if one_letter == "N" else one_letter

    if residue in {"", "?", "N", "UNK", "X"}:
        return None
    return residue


def _alignment_symbol(polymer_type: str, residue: str) -> str:
    identity = _canonical_residue(polymer_type, residue)
    if identity is None:
        return "X"
    if len(identity) == 1 and identity.isascii() and identity.isalpha():
        return identity
    return "X"


def align_chain_residues(pred: ChainRecord, gt: ChainRecord) -> ResidueAlignment:
    """Align chain sequences using OpenFold3's deterministic Kalign wrapper."""
    empty_source = torch.empty(
        0, dtype=torch.long, device=pred.representative_xyz.device
    )
    empty_target = torch.empty(0, dtype=torch.long, device=gt.representative_xyz.device)
    if not pred.sequence or not gt.sequence:
        return ResidueAlignment(empty_source, empty_target, False)

    if pred.residue_keys and gt.residue_keys:
        target_rows_by_key: dict[tuple[int, str], int] = {}
        for target_row, (_, residue_id, insertion_code) in enumerate(gt.residue_keys):
            key = (residue_id, insertion_code)
            if key in target_rows_by_key:
                raise ProtectedChainMatchError("duplicate_target_residue_key")
            target_rows_by_key[key] = target_row

        source_rows: list[int] = []
        target_rows: list[int] = []
        seen_source_keys: set[tuple[int, str]] = set()
        has_conflict = False
        for source_row, (_, residue_id, insertion_code) in enumerate(pred.residue_keys):
            key = (residue_id, insertion_code)
            if key in seen_source_keys:
                raise ProtectedChainMatchError("duplicate_source_residue_key")
            seen_source_keys.add(key)
            target_row = target_rows_by_key.get(key)
            if target_row is None:
                continue
            pred_identity = _canonical_residue(
                pred.polymer_type, pred.sequence[source_row]
            )
            gt_identity = _canonical_residue(gt.polymer_type, gt.sequence[target_row])
            if (
                pred_identity is not None
                and gt_identity is not None
                and pred_identity != gt_identity
            ):
                has_conflict = True
            source_rows.append(source_row)
            target_rows.append(target_row)
        return ResidueAlignment(
            aligned_source_rows=torch.tensor(
                source_rows,
                dtype=torch.long,
                device=pred.representative_xyz.device,
            ),
            aligned_target_rows=torch.tensor(
                target_rows,
                dtype=torch.long,
                device=gt.representative_xyz.device,
            ),
            has_conflicting_known_residue=has_conflict,
        )

    aligned_sequences = run_kalign(
        [
            "".join(
                _alignment_symbol(pred.polymer_type, residue)
                for residue in pred.sequence
            ),
            "".join(
                _alignment_symbol(gt.polymer_type, residue) for residue in gt.sequence
            ),
        ]
    )
    if len(aligned_sequences) != 2:
        raise ProtectedChainMatchError("sequence_alignment_count")

    aligned_pred, aligned_gt = aligned_sequences
    if len(aligned_pred) != len(aligned_gt):
        raise ProtectedChainMatchError("sequence_alignment_length")

    source_rows: list[int] = []
    target_rows: list[int] = []
    source_row = 0
    target_row = 0
    has_conflict = False

    for pred_symbol, gt_symbol in zip(aligned_pred, aligned_gt, strict=True):
        pred_row = None if pred_symbol == "-" else source_row
        gt_row = None if gt_symbol == "-" else target_row
        source_row += pred_row is not None
        target_row += gt_row is not None

        if pred_row is None or gt_row is None:
            continue

        pred_identity = _canonical_residue(pred.polymer_type, pred.sequence[pred_row])
        gt_identity = _canonical_residue(gt.polymer_type, gt.sequence[gt_row])
        if (
            pred_identity is not None
            and gt_identity is not None
            and pred_identity != gt_identity
        ):
            has_conflict = True
        source_rows.append(pred_row)
        target_rows.append(gt_row)

    if source_row != len(pred.sequence) or target_row != len(gt.sequence):
        raise ProtectedChainMatchError("sequence_alignment_rows")

    return ResidueAlignment(
        aligned_source_rows=torch.tensor(
            source_rows,
            dtype=torch.long,
            device=pred.representative_xyz.device,
        ),
        aligned_target_rows=torch.tensor(
            target_rows,
            dtype=torch.long,
            device=gt.representative_xyz.device,
        ),
        has_conflicting_known_residue=has_conflict,
    )


def _has_nondegenerate_geometry(xyz: torch.Tensor) -> bool:
    if xyz.shape[0] < 4:
        return False
    centered = xyz - xyz.mean(0, keepdim=True)
    return bool((torch.linalg.matrix_rank(centered.float()) >= 2).item())


def _has_nondegenerate_correspondence(
    source_xyz: torch.Tensor, target_xyz: torch.Tensor
) -> bool:
    if source_xyz.shape != target_xyz.shape or source_xyz.shape[0] < 4:
        return False
    source_centered = source_xyz.float() - source_xyz.float().mean(0, keepdim=True)
    target_xyz = target_xyz.to(source_xyz.device).float()
    target_centered = target_xyz - target_xyz.mean(0, keepdim=True)
    covariance = source_centered.transpose(0, 1) @ target_centered
    return bool((torch.linalg.matrix_rank(covariance) >= 2).item())


def chains_compatible(pred: ChainRecord, gt: ChainRecord) -> PairCompatibility:
    """Apply hard polymer, sequence, and common-representative constraints."""
    empty_source = torch.empty(
        0, dtype=torch.long, device=pred.representative_xyz.device
    )
    empty_target = torch.empty(0, dtype=torch.long, device=gt.representative_xyz.device)
    if pred.polymer_type != gt.polymer_type:
        return PairCompatibility(
            False,
            "polymer_type_mismatch",
            empty_source,
            empty_target,
            0,
        )

    alignment = align_chain_residues(pred, gt)
    if alignment.has_conflicting_known_residue:
        return PairCompatibility(
            False,
            "sequence_mismatch",
            empty_source,
            empty_target,
            0,
        )

    source_rows, target_rows = alignment.common_finite_representative_rows(pred, gt)
    count = int(source_rows.numel())
    if count < 4:
        return PairCompatibility(
            False,
            "insufficient_common_representatives",
            source_rows,
            target_rows,
            count,
        )
    source_xyz = pred.representative_xyz[source_rows.to(pred.representative_xyz.device)]
    target_xyz = gt.representative_xyz[target_rows.to(gt.representative_xyz.device)]
    if not _has_nondegenerate_correspondence(source_xyz, target_xyz):
        return PairCompatibility(
            False,
            "degenerate_common_representatives",
            source_rows,
            target_rows,
            count,
        )
    return PairCompatibility(True, "allowed", source_rows, target_rows, count)


def representative_anchor_rows(record: ChainRecord) -> torch.Tensor:
    """Return finite resolved anchors after enforcing Kabsch support geometry."""
    mask = record.representative_mask.bool().to(
        record.representative_xyz.device
    ) & torch.isfinite(record.representative_xyz).all(-1)
    xyz = record.representative_xyz[mask]
    if not _has_nondegenerate_geometry(xyz):
        raise InsufficientRepresentativeAnchors(record.chain_id)
    return xyz


def _assignment_total(cost: np.ndarray) -> float:
    """Return the finite physical optimum for one square cost matrix."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ProtectedChainMatchError("assignment_cost_shape")
    if cost.shape[0] == 0:
        return 0.0
    if any(not np.isfinite(cost[row]).any() for row in range(cost.shape[0])):
        raise NoLegalSupervisedCorrespondence("source_chain_without_legal_target")
    try:
        rows, columns = linear_sum_assignment(cost)
    except ValueError as error:
        raise NoLegalSupervisedCorrespondence("no_complete_legal_assignment") from error
    selected = cost[rows, columns]
    if rows.size != cost.shape[0] or not np.isfinite(selected).all():
        raise NoLegalSupervisedCorrespondence("no_complete_legal_assignment")
    return float(selected.sum())


def lexicographic_linear_sum_assignment(
    cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the physical optimum with deterministic lexicographic tie breaks."""
    cost = np.asarray(cost, dtype=np.float64)
    optimum = _assignment_total(cost)
    tolerance = max(1e-12, 8 * float(np.spacing(abs(optimum))))

    remaining_rows = list(range(cost.shape[0]))
    remaining_columns = list(range(cost.shape[1]))
    selected_rows: list[int] = []
    selected_columns: list[int] = []
    fixed_cost = 0.0

    while remaining_rows:
        row = remaining_rows[0]
        chosen_column = None
        for column in remaining_columns:
            edge_cost = cost[row, column]
            if not np.isfinite(edge_cost):
                continue
            tail_rows = remaining_rows[1:]
            tail_columns = [
                candidate for candidate in remaining_columns if candidate != column
            ]
            if tail_rows:
                tail_cost = cost[np.ix_(tail_rows, tail_columns)]
                try:
                    constrained_total = (
                        fixed_cost + float(edge_cost) + _assignment_total(tail_cost)
                    )
                except NoLegalSupervisedCorrespondence:
                    continue
            else:
                constrained_total = fixed_cost + float(edge_cost)
            if abs(constrained_total - optimum) <= tolerance:
                chosen_column = column
                break
        if chosen_column is None:
            raise NoLegalSupervisedCorrespondence(
                "lexicographic_optimum_has_no_legal_edge"
            )
        selected_rows.append(row)
        selected_columns.append(chosen_column)
        fixed_cost += float(cost[row, chosen_column])
        remaining_rows.pop(0)
        remaining_columns.remove(chosen_column)

    return (
        np.asarray(selected_rows, dtype=np.int64),
        np.asarray(selected_columns, dtype=np.int64),
    )


def _common_xyz(
    pred: ChainRecord,
    gt: ChainRecord,
    compatibility: PairCompatibility,
) -> tuple[np.ndarray, np.ndarray]:
    source_rows = compatibility.aligned_source_rows.to(pred.representative_xyz.device)
    target_rows = compatibility.aligned_target_rows.to(gt.representative_xyz.device)
    source = (
        pred.representative_xyz[source_rows]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    target = (
        gt.representative_xyz[target_rows]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    return source, target


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> RigidTransform:
    """Fit a proper Kabsch transform mapping row-vector source onto target."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ProtectedChainMatchError("rigid_fit_shape")
    if source.shape[0] < 4:
        raise InsufficientRepresentativeAnchors("combined_correspondence")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    if np.linalg.matrix_rank(source_centered) < 2:
        raise InsufficientRepresentativeAnchors("combined_source_geometry")
    if np.linalg.matrix_rank(target_centered) < 2:
        raise InsufficientRepresentativeAnchors("combined_target_geometry")

    left, _, right_transpose = np.linalg.svd(source_centered.T @ target_centered)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _apply_rigid(xyz: np.ndarray, transform: RigidTransform) -> np.ndarray:
    rotation, translation = transform
    return xyz @ rotation.T + translation


def _mapping_support(
    mapping: dict[str, str],
    pred_by_id: dict[str, ChainRecord],
    gt_by_id: dict[str, ChainRecord],
    compatibility: dict[tuple[str, str], PairCompatibility],
) -> tuple[np.ndarray, np.ndarray]:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for source_id in sorted(mapping):
        target_id = mapping[source_id]
        pair = compatibility[(source_id, target_id)]
        if not pair.allowed:
            raise NoLegalSupervisedCorrespondence("mapping_contains_forbidden_edge")
        source, target = _common_xyz(
            pred_by_id[source_id],
            gt_by_id[target_id],
            pair,
        )
        source_parts.append(source)
        target_parts.append(target)
    if not source_parts:
        raise NoLegalSupervisedCorrespondence("empty_chain_mapping")
    return np.concatenate(source_parts), np.concatenate(target_parts)


def _mapping_metrics(
    mapping: dict[str, str],
    pred_by_id: dict[str, ChainRecord],
    gt_by_id: dict[str, ChainRecord],
    compatibility: dict[tuple[str, str], PairCompatibility],
) -> tuple[float, int]:
    source, target = _mapping_support(
        mapping,
        pred_by_id,
        gt_by_id,
        compatibility,
    )
    transform = _fit_rigid(source, target)
    residual = _apply_rigid(source, transform) - target
    rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))
    return rmsd, int(source.shape[0])


def _chain_center(record: ChainRecord) -> np.ndarray:
    return (
        representative_anchor_rows(record)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .mean(axis=0)
    )


def _fingerprint_cost(
    pred_group: list[ChainRecord],
    gt_group: list[ChainRecord],
    compatibility: dict[tuple[str, str], PairCompatibility],
) -> np.ndarray:
    source_centers = np.stack([_chain_center(chain) for chain in pred_group])
    target_centers = np.stack([_chain_center(chain) for chain in gt_group])
    source_fingerprints = np.sort(
        np.linalg.norm(source_centers[:, None] - source_centers[None, :], axis=-1),
        axis=-1,
    )
    target_fingerprints = np.sort(
        np.linalg.norm(target_centers[:, None] - target_centers[None, :], axis=-1),
        axis=-1,
    )
    cost = np.full(
        (len(pred_group), len(gt_group)),
        np.inf,
        dtype=np.float64,
    )
    for source_index, source in enumerate(pred_group):
        for target_index, target in enumerate(gt_group):
            if compatibility[(source.chain_id, target.chain_id)].allowed:
                delta = (
                    source_fingerprints[source_index]
                    - target_fingerprints[target_index]
                )
                cost[source_index, target_index] = float(np.mean(delta * delta))
    return cost


def _pair_cost(
    pred: ChainRecord,
    gt: ChainRecord,
    compatibility: PairCompatibility,
    transform: RigidTransform,
) -> float:
    if not compatibility.allowed:
        return np.inf
    source, target = _common_xyz(pred, gt, compatibility)
    transformed = _apply_rigid(source, transform)
    return float(np.mean(np.sum((transformed - target) ** 2, axis=-1)))


def _identity_mapping(
    pred: list[ChainRecord],
    gt: list[ChainRecord],
    compatibility: dict[tuple[str, str], PairCompatibility],
) -> dict[str, str]:
    source_ids = sorted(chain.chain_id for chain in pred)
    target_ids = sorted(chain.chain_id for chain in gt)
    cost = np.asarray(
        [
            [
                0.0 if compatibility[(source_id, target_id)].allowed else np.inf
                for target_id in target_ids
            ]
            for source_id in source_ids
        ],
        dtype=np.float64,
    )
    rows, columns = lexicographic_linear_sum_assignment(cost)
    return {
        source_ids[row]: target_ids[column]
        for row, column in zip(rows, columns, strict=True)
    }


def _compatibility_groups(
    pred: list[ChainRecord],
    gt: list[ChainRecord],
    compatibility: dict[tuple[str, str], PairCompatibility],
) -> list[tuple[list[ChainRecord], list[ChainRecord]]]:
    pred_by_id = {chain.chain_id: chain for chain in pred}
    gt_by_id = {chain.chain_id: chain for chain in gt}
    unseen_source_ids = set(pred_by_id)
    groups: list[tuple[list[ChainRecord], list[ChainRecord]]] = []

    while unseen_source_ids:
        source_frontier = [min(unseen_source_ids)]
        source_ids: set[str] = set()
        target_ids: set[str] = set()
        while source_frontier:
            source_id = source_frontier.pop()
            if source_id in source_ids:
                continue
            source_ids.add(source_id)
            unseen_source_ids.discard(source_id)
            newly_reached_targets = {
                target.chain_id
                for target in gt
                if compatibility[(source_id, target.chain_id)].allowed
            }
            target_ids.update(newly_reached_targets)
            newly_reached_sources = {
                source.chain_id
                for source in pred
                if any(
                    compatibility[(source.chain_id, target_id)].allowed
                    for target_id in newly_reached_targets
                )
            }
            source_frontier.extend(sorted(newly_reached_sources - source_ids))
        if not target_ids:
            raise NoLegalSupervisedCorrespondence(
                f"source_chain_without_legal_target:{min(source_ids)}"
            )
        groups.append(
            (
                [pred_by_id[source_id] for source_id in sorted(source_ids)],
                [gt_by_id[target_id] for target_id in sorted(target_ids)],
            )
        )
    return sorted(
        groups,
        key=lambda group: (
            tuple(chain.chain_id for chain in group[0]),
            tuple(chain.chain_id for chain in group[1]),
        ),
    )


def _initial_transforms(
    groups: list[tuple[list[ChainRecord], list[ChainRecord]]],
    compatibility: dict[tuple[str, str], PairCompatibility],
) -> list[RigidTransform]:
    unique_mapping: dict[str, str] = {}
    pred_by_id: dict[str, ChainRecord] = {}
    gt_by_id: dict[str, ChainRecord] = {}
    for source_group, target_group in groups:
        pred_by_id.update((chain.chain_id, chain) for chain in source_group)
        gt_by_id.update((chain.chain_id, chain) for chain in target_group)
        legal_edges = [
            (source.chain_id, target.chain_id)
            for source in source_group
            for target in target_group
            if compatibility[(source.chain_id, target.chain_id)].allowed
        ]
        if len(source_group) == len(target_group) == len(legal_edges) == 1:
            unique_mapping[legal_edges[0][0]] = legal_edges[0][1]

    if unique_mapping:
        source, target = _mapping_support(
            unique_mapping,
            pred_by_id,
            gt_by_id,
            compatibility,
        )
        try:
            return [_fit_rigid(source, target)]
        except InsufficientRepresentativeAnchors:
            pass

    seed_mapping: dict[str, str] = {}
    for source_group, target_group in groups:
        if len(source_group) != len(target_group):
            raise NoLegalSupervisedCorrespondence("compatibility_group_size_mismatch")
        rows, columns = lexicographic_linear_sum_assignment(
            _fingerprint_cost(source_group, target_group, compatibility)
        )
        seed_mapping.update(
            {
                source_group[row].chain_id: target_group[column].chain_id
                for row, column in zip(rows, columns, strict=True)
            }
        )
    source, target = _mapping_support(
        seed_mapping,
        pred_by_id,
        gt_by_id,
        compatibility,
    )
    try:
        fingerprint_transform = _fit_rigid(source, target)
    except InsufficientRepresentativeAnchors:
        fingerprint_transform = (
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
    identity_transform = (
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    if all(
        np.allclose(left, right, atol=1e-12, rtol=0.0)
        for left, right in zip(
            fingerprint_transform,
            identity_transform,
            strict=True,
        )
    ):
        return [identity_transform]
    return [fingerprint_transform, identity_transform]


def _candidate_mapping(
    groups: list[tuple[list[ChainRecord], list[ChainRecord]]],
    compatibility: dict[tuple[str, str], PairCompatibility],
    transform: RigidTransform,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pred_group, gt_group in groups:
        if len(pred_group) != len(gt_group):
            raise NoLegalSupervisedCorrespondence("compatibility_group_size_mismatch")
        cost = np.full(
            (len(pred_group), len(gt_group)),
            np.inf,
            dtype=np.float64,
        )
        for source_index, source in enumerate(pred_group):
            for target_index, target in enumerate(gt_group):
                cost[source_index, target_index] = _pair_cost(
                    source,
                    target,
                    compatibility[(source.chain_id, target.chain_id)],
                    transform,
                )
        rows, columns = lexicographic_linear_sum_assignment(cost)
        mapping.update(
            {
                pred_group[row].chain_id: gt_group[column].chain_id
                for row, column in zip(rows, columns, strict=True)
            }
        )
    return mapping


def _improves_rmsd(identity_rmsd: float, candidate_rmsd: float) -> bool:
    """Apply the strict physical acceptance boundary in Å."""
    return identity_rmsd - candidate_rmsd > 1e-3


def protected_chain_match(
    pred_chains: tuple[ChainRecord, ...],
    gt_chains: tuple[ChainRecord, ...],
) -> ProtectedMatchResult:
    """Solve compatible chain assignment and protect the identity baseline."""
    pred = sorted(pred_chains, key=lambda chain: chain.chain_id)
    gt = sorted(gt_chains, key=lambda chain: chain.chain_id)
    if not pred or len(pred) != len(gt):
        raise NoLegalSupervisedCorrespondence("chain_count_mismatch")
    if len({chain.chain_id for chain in pred}) != len(pred) or len(
        {chain.chain_id for chain in gt}
    ) != len(gt):
        raise ProtectedChainMatchError("duplicate_chain_id")

    compatibility = {
        (source.chain_id, target.chain_id): chains_compatible(source, target)
        for source in pred
        for target in gt
    }
    groups = _compatibility_groups(pred, gt, compatibility)
    identity_mapping = _identity_mapping(pred, gt, compatibility)
    pred_by_id = {chain.chain_id: chain for chain in pred}
    gt_by_id = {chain.chain_id: chain for chain in gt}
    identity_rmsd, identity_coverage = _mapping_metrics(
        identity_mapping,
        pred_by_id,
        gt_by_id,
        compatibility,
    )

    if any(
        len(source_group) > 20 or len(target_group) > 20
        for source_group, target_group in groups
    ):
        return ProtectedMatchResult(
            mapping=identity_mapping,
            identity_rmsd=identity_rmsd,
            candidate_rmsd=identity_rmsd,
            matched_coverage=identity_coverage,
            accepted=False,
            reason="oversized_compatibility_group",
        )

    candidate_options: list[
        tuple[float, tuple[tuple[str, str], ...], int, dict[str, str]]
    ] = []
    seen_candidates: set[tuple[tuple[str, str], ...]] = set()
    for transform in _initial_transforms(groups, compatibility):
        mapping = _candidate_mapping(groups, compatibility, transform)
        mapping_key = tuple(sorted(mapping.items()))
        if mapping_key in seen_candidates:
            continue
        seen_candidates.add(mapping_key)
        rmsd, coverage = _mapping_metrics(
            mapping,
            pred_by_id,
            gt_by_id,
            compatibility,
        )
        candidate_options.append((rmsd, mapping_key, coverage, mapping))
    if not candidate_options:
        raise NoLegalSupervisedCorrespondence("no_candidate_mapping")
    candidate_rmsd, _, candidate_coverage, candidate_mapping = min(
        candidate_options,
        key=lambda option: (option[0], option[1]),
    )
    if candidate_coverage < identity_coverage:
        return ProtectedMatchResult(
            mapping=identity_mapping,
            identity_rmsd=identity_rmsd,
            candidate_rmsd=candidate_rmsd,
            matched_coverage=identity_coverage,
            accepted=False,
            reason="insufficient_coverage",
        )
    if not _improves_rmsd(identity_rmsd, candidate_rmsd):
        return ProtectedMatchResult(
            mapping=identity_mapping,
            identity_rmsd=identity_rmsd,
            candidate_rmsd=candidate_rmsd,
            matched_coverage=identity_coverage,
            accepted=False,
            reason="insufficient_rmsd_improvement",
        )
    return ProtectedMatchResult(
        mapping=candidate_mapping,
        identity_rmsd=identity_rmsd,
        candidate_rmsd=candidate_rmsd,
        matched_coverage=candidate_coverage,
        accepted=True,
        reason="accepted",
    )


__all__ = [
    "ChainRecord",
    "InsufficientRepresentativeAnchors",
    "NoLegalSupervisedCorrespondence",
    "PairCompatibility",
    "ProtectedChainMatchError",
    "ProtectedMatchResult",
    "ResidueAlignment",
    "_improves_rmsd",
    "align_chain_residues",
    "chain_record_from_atom_metadata",
    "chains_compatible",
    "lexicographic_linear_sum_assignment",
    "protected_chain_match",
    "representative_anchor_rows",
]
