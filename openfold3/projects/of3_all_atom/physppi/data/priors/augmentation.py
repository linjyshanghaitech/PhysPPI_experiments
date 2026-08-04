"""Sampling and light augmentation for optional Q-BioLiP inputs.

Optional inputs have three independent pieces of state.  ``available`` says
that the source supplied a value, ``selected`` says that the value was picked
for this example, and ``valid`` is a final data-quality gate.  Keeping these
states separate avoids treating an unavailable value as a selected zero
tensor.

This module intentionally depends only on the standard library and NumPy so
that it can also be used by lightweight data-processing and test workers.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np


class OptionalSlot(str, Enum):
    """Optional feature groups that may be independently sampled."""

    INTERFACE_RESIDUE = "interface_residue"
    INTERACTION = "interaction"
    SURFACE = "surface"
    TM = "tm"


PRIOR_OPTIONAL_SLOTS = (
    OptionalSlot.INTERFACE_RESIDUE,
    OptionalSlot.INTERACTION,
    OptionalSlot.SURFACE,
    OptionalSlot.TM,
)


def build_paired_msa_chain_mask(available: object) -> np.ndarray:
    """Return a boolean availability mask for paired MSA chains."""

    values = np.asarray(available, dtype=bool)
    if values.ndim != 1:
        raise ValueError("paired MSA availability must be one-dimensional")
    return values.copy()


def available_optional_slots(sample: object) -> dict[OptionalSlot, bool]:
    """Return the canonical source-availability state for optional priors."""
    res_loc = sample.opm.get("res_loc") if isinstance(sample.opm, dict) else None
    available = {
        OptionalSlot.INTERFACE_RESIDUE: bool(sample.dmap),
        OptionalSlot.INTERACTION: bool(sample.interactions),
        OptionalSlot.SURFACE: bool(sample.mesh),
        OptionalSlot.TM: bool(res_loc),
    }
    return {slot: available[slot] for slot in PRIOR_OPTIONAL_SLOTS}


@dataclass(frozen=True)
class SlotState:
    """Immutable availability, sampling, and validation state for one slot."""

    available: bool
    selected: bool
    valid: bool
    provenance: str

    @property
    def active(self) -> bool:
        """Whether this slot can be consumed by a downstream feature builder."""

        return self.available and self.selected and self.valid


@dataclass(frozen=True)
class PriorLevelSelection:
    """Requested and available cumulative optional-prior levels."""

    requested_level: int
    effective_level: int
    states: dict[OptionalSlot, SlotState]


def select_prior_level(
    available: Mapping[OptionalSlot, bool],
    requested_level: int,
) -> PriorLevelSelection:
    """Resolve a requested L0-L4 prefix against source availability."""
    if requested_level not in range(5):
        raise ValueError("requested_level must be in [0, 4]")

    effective_level = 0
    for level, slot in enumerate(PRIOR_OPTIONAL_SLOTS, start=1):
        if level > requested_level or not bool(available.get(slot, False)):
            break
        effective_level = level

    states = {
        slot: SlotState(
            available=bool(available.get(slot, False)),
            selected=index <= effective_level,
            valid=bool(available.get(slot, False)),
            provenance="source" if available.get(slot, False) else "unavailable",
        )
        for index, slot in enumerate(PRIOR_OPTIONAL_SLOTS, start=1)
    }
    return PriorLevelSelection(requested_level, effective_level, states)


def sample_prior_level(
    available: Mapping[OptionalSlot, bool], rng: np.random.Generator
) -> PriorLevelSelection:
    """Sample a uniformly distributed requested L0-L4 cumulative prefix."""
    return select_prior_level(available, int(rng.integers(0, 5)))


def subsample_interactions_by_type(
    edges: np.ndarray,
    type_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample interaction edges while preserving every interaction type.

    At least one edge is retained for each distinct value in ``type_ids``;
    half-sampling is used for larger groups.  Returned edge/type arrays remain
    aligned and inputs are never modified.
    """

    edge_array = np.asarray(edges)
    type_array = np.asarray(type_ids)
    if edge_array.ndim == 0:
        raise ValueError("edges must have a leading edge dimension")
    if type_array.ndim != 1:
        raise ValueError("type_ids must be one-dimensional")
    if edge_array.shape[0] != type_array.shape[0]:
        raise ValueError("edges and type_ids must have matching lengths")
    if not type_array.size:
        return edge_array[:0].copy(), type_array[:0].copy()

    kept_edges: list[np.ndarray] = []
    kept_types: list[np.ndarray] = []
    for type_id in np.unique(type_array):
        indices = np.flatnonzero(type_array == type_id)
        count = max(1, int(np.ceil(indices.size * 0.5)))
        chosen = np.asarray(
            rng.choice(indices, size=count, replace=False), dtype=np.intp
        )
        kept_edges.append(edge_array[chosen])
        kept_types.append(type_array[chosen])
    return np.concatenate(kept_edges), np.concatenate(kept_types)


def mask_surface_patches(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly hide surface points while preserving mask shape and input."""

    result = np.asarray(mask).copy()
    if result.size:
        result[rng.random(result.shape) < 0.25] = False
    return result


def mask_tm_segments(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly hide TM positions while preserving mask shape and input."""

    result = np.asarray(mask).copy()
    if result.size:
        result[rng.random(result.shape) < 0.15] = False
    return result


__all__ = [
    "OptionalSlot",
    "PRIOR_OPTIONAL_SLOTS",
    "PriorLevelSelection",
    "SlotState",
    "build_paired_msa_chain_mask",
    "mask_surface_patches",
    "mask_tm_segments",
    "sample_prior_level",
    "select_prior_level",
    "subsample_interactions_by_type",
]
