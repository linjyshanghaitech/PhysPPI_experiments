"""Deterministic audit-view coverage for production PhysPPI screening."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from openfold3.projects.of3_all_atom.physppi.data.priors.occurrence import (
    build_occurrence_rng,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    AuditView,
    DatasetName,
)

AUDIT_DATA_SEED: Final = 42
AUDIT_EPOCH: Final = 0
AUDIT_VIEW_COUNT: Final = 7


class DeterministicAuditView(AuditView):
    """One requested audit view bound to its deterministic occurrence seed."""

    audit_epoch: Literal[0] = AUDIT_EPOCH
    global_slot: int = Field(ge=0)
    occurrence_seed: int = Field(ge=0, le=(1 << 64) - 1)


_REQUESTED_MATRIX: Final = (
    {
        "crop_strategy": "spatial",
        "requested_level": 0,
        "interface_cutoff": 8.0,
        "view_index": 0,
    },
    {
        "crop_strategy": "spatial_interface",
        "requested_level": 1,
        "interface_cutoff": 18.0,
        "view_index": 1,
    },
    {
        "crop_strategy": "spatial",
        "requested_level": 2,
        "interface_cutoff": 13.0,
        "view_index": 2,
    },
    {
        "crop_strategy": "spatial_interface",
        "requested_level": 3,
        "interface_cutoff": 8.0,
        "view_index": 3,
    },
    {
        "crop_strategy": "spatial",
        "requested_level": 4,
        "interface_cutoff": 18.0,
        "view_index": 4,
    },
    {
        "crop_strategy": "spatial_interface",
        "requested_level": 4,
        "interface_cutoff": 13.0,
        "view_index": 5,
        "multichain_probe": True,
    },
    {
        "crop_strategy": "spatial",
        "requested_level": 4,
        "interface_cutoff": 13.0,
        "view_index": 6,
        "multichain_probe": True,
    },
)


def audit_global_slot(source_ordinal: int, view_index: int) -> int:
    """Map one source row and view index to a stable audit sampler slot."""
    if source_ordinal < 0:
        raise ValueError("source_ordinal must be nonnegative")
    if not 0 <= view_index < AUDIT_VIEW_COUNT:
        raise ValueError(f"view_index must be in [0, {AUDIT_VIEW_COUNT})")
    return source_ordinal * AUDIT_VIEW_COUNT + view_index


def audit_views(
    dataset: DatasetName | str,
    sample_id: str,
    *,
    source_ordinal: int = 0,
    data_seed: int = AUDIT_DATA_SEED,
) -> tuple[DeterministicAuditView, ...]:
    """Return the fixed requested matrix; availability is resolved downstream."""
    dataset_name = DatasetName(dataset)
    if not sample_id:
        raise ValueError("sample_id must be nonempty")

    views = []
    for requested in _REQUESTED_MATRIX:
        global_slot = audit_global_slot(source_ordinal, requested["view_index"])
        occurrence = build_occurrence_rng(
            data_seed,
            AUDIT_EPOCH,
            global_slot,
            dataset_name.value,
            sample_id,
        )
        views.append(
            DeterministicAuditView(
                **requested,
                global_slot=global_slot,
                occurrence_seed=occurrence.seed,
            )
        )
    return tuple(views)
