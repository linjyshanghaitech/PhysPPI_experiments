"""Deterministic, production-feature preflight for PhysPPI screening."""

from __future__ import annotations

import copy
import hashlib
import os
import random
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.core.data.framework.stochastic_sampler_dataset import SampleOccurrence
from openfold3.projects.of3_all_atom.physppi.screening.config import (
    _source_index,
    _source_root,
    build_preflight_dataset,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Finding,
    ScreeningOptions,
    Severity,
)
from openfold3.projects.of3_all_atom.physppi.screening.views import (
    DeterministicAuditView,
)

_WORKER_OPTIONS: ScreeningOptions | None = None
_WORKER_DATASETS: dict[tuple[DatasetName, str], Any] = {}


def _finding(
    code: str,
    message: str,
    *,
    stage: str = "preflight",
    **metrics: int | float | str | bool,
) -> Finding:
    return Finding(
        code=code,
        severity=Severity.LATEST_PREFLIGHT,
        message=message,
        metrics={"stage": stage, **metrics},
    )


def _shape(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def _integer_values(value: object) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or value.is_floating_point():
            return None
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
        return value
    return None


def _has_nonfinite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_floating_point() and not bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.floating) and not bool(
            np.isfinite(value).all()
        )
    if isinstance(value, Mapping):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(item) for item in value)
    return isinstance(value, float) and not np.isfinite(value)


def _token_count(features: Mapping[str, object]) -> int | None:
    token_mask_shape = _shape(features.get("token_mask"))
    if token_mask_shape is None or len(token_mask_shape) != 1:
        return None
    return token_mask_shape[0]


def _atom_count(features: Mapping[str, object]) -> int | None:
    atom_mask_shape = _shape(features.get("atom_mask"))
    if atom_mask_shape is None or len(atom_mask_shape) != 1:
        return None
    return atom_mask_shape[0]


def _template_aligned(features: Mapping[str, object], n_tokens: int) -> bool:
    for name, value in features.items():
        if not name.startswith("template_"):
            continue
        shape = _shape(value)
        if shape is None or len(shape) < 2 or shape[1] != n_tokens:
            return False
        if name in {"template_distogram", "template_unit_vector"} and (
            len(shape) < 3 or shape[2] != n_tokens
        ):
            return False
    return True


def _scalar_int(value: object) -> int | None:
    if isinstance(value, torch.Tensor):
        if (
            value.numel() != 1
            or value.dtype == torch.bool
            or value.is_floating_point()
            or value.is_complex()
        ):
            return None
        return int(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
            return None
        return int(value.item())
    if isinstance(value, Integral) and not isinstance(value, bool):
        return value
    return None


def _scalar_bool(value: object) -> bool | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype != torch.bool:
            return None
        return bool(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1 or value.dtype != np.bool_:
            return None
        return bool(value.item())
    if isinstance(value, bool):
        return value
    return None


def _valid_retained_view(provenance: Mapping[str, object]) -> bool:
    required = {
        "parent_sample_id",
        "original_chain_ids",
        "source_chain_ids",
        "component_chain_ids",
        "seed_edge",
        "trigger",
        "target_chain_count",
        "dmap_cutoff",
        "occurrence_seed",
        "matcher_result",
    }
    if set(provenance) != required:
        return False
    original = provenance["original_chain_ids"]
    source = provenance["source_chain_ids"]
    component = provenance["component_chain_ids"]
    if any(
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        for value in (original, source, component)
    ):
        return False
    original_ids = tuple(str(value) for value in original)
    source_ids = tuple(str(value) for value in source)
    component_ids = tuple(str(value) for value in component)
    target_count = _scalar_int(provenance["target_chain_count"])
    occurrence_seed = _scalar_int(provenance["occurrence_seed"])
    seed_edge = provenance["seed_edge"]
    if seed_edge is not None and (
        not isinstance(seed_edge, Sequence)
        or isinstance(seed_edge, (str, bytes))
        or len(seed_edge) != 2
    ):
        return False
    seed_edge_ids = (
        None if seed_edge is None else tuple(str(value) for value in seed_edge)
    )
    try:
        dmap_cutoff = float(provenance["dmap_cutoff"])
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        isinstance(provenance["parent_sample_id"], str)
        and isinstance(provenance["trigger"], str)
        and isinstance(provenance["matcher_result"], str)
        and original_ids == tuple(sorted(set(original_ids)))
        and source_ids == tuple(sorted(set(source_ids)))
        and component_ids == tuple(sorted(set(component_ids)))
        and 2 <= len(source_ids) <= 4
        and set(source_ids) <= set(component_ids) <= set(original_ids)
        and (
            seed_edge_ids is None
            or len(set(seed_edge_ids)) == 2
            and set(seed_edge_ids) <= set(source_ids)
        )
        and target_count == len(source_ids)
        and dmap_cutoff == 18.0
        and occurrence_seed is not None
        and occurrence_seed >= 0
    )


def _valid_matcher_diagnostics(diagnostics: Mapping[str, object]) -> bool:
    required = {
        "accepted",
        "fallback",
        "reason",
        "identity_rmsd",
        "candidate_rmsd",
        "matched_coverage",
    }
    if set(diagnostics) != required:
        return False
    accepted = _scalar_bool(diagnostics["accepted"])
    coverage = _scalar_int(diagnostics["matched_coverage"])
    try:
        identity_rmsd = float(diagnostics["identity_rmsd"])
        candidate_rmsd = float(diagnostics["candidate_rmsd"])
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        accepted is not None
        and diagnostics["fallback"] in {"none", "identity"}
        and isinstance(diagnostics["reason"], str)
        and np.isfinite(identity_rmsd)
        and np.isfinite(candidate_rmsd)
        and coverage is not None
        and coverage >= 0
    )


def validate_physppi_features(
    features,
    *,
    retained_view: Mapping[str, object] | None = None,
    matcher_diagnostics: Mapping[str, object] | None = None,
) -> list[Finding]:
    """Return stable contract findings for one uncollated production feature dict."""
    if not isinstance(features, Mapping):
        return [_finding("feature_mapping_invalid", "features must be a mapping")]
    legacy_fields = {
        "physppi_contact_token_mask",
        "physppi_surface_interface_mask",
    } & features.keys()
    if legacy_fields:
        return [
            _finding(
                "legacy_prior_field",
                "legacy prior fields are forbidden",
                count=len(legacy_fields),
            )
        ]
    n_tokens = _token_count(features)
    n_atoms = _atom_count(features)
    if n_tokens is None or n_atoms is None:
        return [
            _finding(
                "token_atom_cardinality", "token_mask and atom_mask must be rank one"
            )
        ]
    if "entity_id" not in features:
        return [_finding("missing_entity_id", "entity_id is required")]
    entity_shape = _shape(features["entity_id"])
    asym_shape = _shape(features.get("asym_id"))
    if entity_shape not in {(n_tokens,), (n_atoms,)} or asym_shape not in {
        (n_tokens,),
        (n_atoms,),
    }:
        return [
            _finding(
                "entity_chain_cardinality", "entity_id/asym_id cardinality is invalid"
            )
        ]

    atom_to_token = _integer_values(features.get("atom_to_token_index"))
    if (
        atom_to_token is None
        or atom_to_token.shape != (n_atoms,)
        or (
            atom_to_token.size
            and (atom_to_token.min() < 0 or atom_to_token.max() >= n_tokens)
        )
    ):
        return [
            _finding(
                "label_index_out_of_bounds", "atom-to-token labels exceed token bounds"
            )
        ]

    msa = _shape(features.get("msa"))
    msa_fields = ("msa_mask", "has_deletion", "deletion_value")
    if (
        msa is None
        or len(msa) < 2
        or msa[1] != n_tokens
        or any(
            (shape := _shape(features.get(name))) is None
            or len(shape) < 2
            or shape[1] != n_tokens
            for name in msa_fields
        )
        or _shape(features.get("deletion_mean")) != (n_tokens,)
        or (profile := _shape(features.get("profile"))) is None
        or len(profile) < 1
        or profile[0] != n_tokens
    ):
        return [
            _finding("msa_width_mismatch", "MSA features must align to token width")
        ]

    asym_values = _integer_values(features.get("asym_id"))
    token_mask = features["token_mask"]
    if isinstance(token_mask, torch.Tensor):
        valid_tokens = token_mask.detach().cpu().numpy().astype(bool, copy=False)
    else:
        valid_tokens = np.asarray(token_mask, dtype=bool)
    if asym_values is not None and asym_values.shape == (n_tokens,):
        asym_values = asym_values[valid_tokens]
    if asym_values is not None:
        n_chains = int(np.unique(asym_values[asym_values != 0]).size)
    else:
        n_chains = 0
    paired_shape = _shape(features.get("physppi_paired_msa_chain_mask"))
    if paired_shape != (n_chains,):
        return [_finding("paired_msa_shape", "paired-MSA mask must be [N_chains]")]

    ref_uids = _integer_values(features.get("ref_space_uid"))
    permutations = features.get("ref_space_uid_to_perm")
    if (
        ref_uids is None
        or ref_uids.shape != (n_atoms,)
        or not isinstance(permutations, Mapping)
    ):
        return [
            _finding(
                "canonical_mapping_invalid", "canonical reference mapping is invalid"
            )
        ]
    for raw_uid, permutation in permutations.items():
        try:
            uid = int(raw_uid)
        except (TypeError, ValueError, OverflowError):
            return [
                _finding("canonical_mapping_invalid", "reference-space key is invalid")
            ]
        local_count = int(np.count_nonzero(ref_uids == uid))
        values = _integer_values(permutation)
        if (
            local_count == 0
            or values is None
            or values.ndim != 2
            or values.size == 0
            or values.min() < 0
            or values.max() >= local_count
        ):
            return [
                _finding(
                    "canonical_mapping_invalid",
                    "reference-space permutations exceed bounds",
                )
            ]

    if not _template_aligned(features, n_tokens):
        return [
            _finding(
                "template_alignment_mismatch", "template features must align to tokens"
            )
        ]

    centers = _integer_values(features.get("physppi_token_center_atom_index"))
    if centers is None or centers.shape != (n_tokens,):
        return [
            _finding("token_center_out_of_bounds", "token centers exceed atom bounds")
        ]
    valid_centers = centers[valid_tokens]
    if valid_centers.size and (
        valid_centers.min() < 0 or valid_centers.max() >= n_atoms
    ):
        return [
            _finding("token_center_out_of_bounds", "token centers exceed atom bounds")
        ]

    backbone = _integer_values(features.get("physppi_backbone_atom_indices"))
    frame_mask_shape = _shape(features.get("physppi_backbone_frame_mask"))
    if (
        backbone is None
        or backbone.shape != (n_tokens, 3)
        or frame_mask_shape != (n_tokens,)
    ):
        return [
            _finding(
                "frame_atom_out_of_bounds", "backbone frame atoms exceed atom bounds"
            )
        ]
    frame_mask = np.asarray(
        features["physppi_backbone_frame_mask"].detach().cpu()
        if isinstance(features["physppi_backbone_frame_mask"], torch.Tensor)
        else features["physppi_backbone_frame_mask"],
        dtype=bool,
    )
    active_backbone = backbone[valid_tokens & frame_mask]
    inactive_backbone = backbone[valid_tokens & ~frame_mask]
    if (
        active_backbone.size
        and (active_backbone.min() < 0 or active_backbone.max() >= n_atoms)
    ) or (
        inactive_backbone.size
        and (inactive_backbone.min() < -1 or inactive_backbone.max() >= n_atoms)
    ):
        return [
            _finding(
                "frame_atom_out_of_bounds", "backbone frame atoms exceed atom bounds"
            )
        ]

    contact = features.get("physppi_contact_pair_mask")
    contact_shape = _shape(contact)
    interaction_shape = _shape(features.get("physppi_interaction_type_mask"))
    interaction_distance_shape = _shape(features.get("physppi_interaction_distance"))
    interaction_valid_shape = _shape(features.get("physppi_interaction_aux_valid"))
    if (
        contact_shape != (n_tokens, n_tokens)
        or interaction_shape != (n_tokens, n_tokens, 7)
        or interaction_distance_shape != (n_tokens, n_tokens)
        or interaction_valid_shape != (n_tokens, n_tokens)
    ):
        return [
            _finding(
                "interaction_shape",
                "pair contact and interaction auxiliary labels must align to tokens",
            )
        ]
    contact_values = (
        contact.detach().cpu().numpy()
        if isinstance(contact, torch.Tensor)
        else np.asarray(contact)
    )
    if not np.array_equal(contact_values, contact_values.T) or np.any(
        np.diag(contact_values)
    ):
        return [
            _finding(
                "interaction_shape",
                "contact pair mask must be symmetric with a false diagonal",
            )
        ]

    if (
        _shape(features.get("physppi_surface_local_coords")) != (n_tokens, 128, 3)
        or _shape(features.get("physppi_surface_features")) != (n_tokens, 128, 6)
        or _shape(features.get("physppi_surface_mask")) != (n_tokens, 128)
        or _shape(features.get("physppi_surface_charge_mask")) != (n_tokens, 128)
    ):
        return [_finding("surface_shape", "surface features must be [N, 128, ...]")]

    requested_level = _scalar_int(features.get("physppi_prior_requested_level"))
    effective_level = _scalar_int(features.get("physppi_prior_effective_level"))
    occurrence_seed = _scalar_int(features.get("physppi_occurrence_seed"))
    active_values = [
        _scalar_bool(features.get(f"physppi_{slot}_active"))
        for slot in ("contact", "interaction", "surface", "tm")
    ]
    state_values = [
        _scalar_bool(features.get(f"physppi_{slot}_{state}"))
        for slot in ("contact", "interaction", "surface", "tm")
        for state in ("available", "selected", "valid", "provenance")
    ]
    expected_active = (
        [index < effective_level for index in range(4)]
        if effective_level is not None
        else None
    )
    if (
        requested_level is None
        or effective_level is None
        or requested_level not in range(5)
        or effective_level not in range(requested_level + 1)
        or occurrence_seed is None
        or occurrence_seed < 0
        or None in active_values
        or None in state_values
        or active_values != expected_active
    ):
        return [
            _finding(
                "prior_level_contract",
                "prior levels and slot states must form one cumulative L0-L4 prefix",
            )
        ]
    for slot_index, slot in enumerate(("contact", "interaction", "surface", "tm")):
        available = _scalar_bool(features[f"physppi_{slot}_available"])
        selected = _scalar_bool(features[f"physppi_{slot}_selected"])
        valid = _scalar_bool(features[f"physppi_{slot}_valid"])
        if active_values[slot_index] != bool(available and selected and valid):
            return [
                _finding(
                    "prior_level_contract",
                    "active slot state must equal available and selected and valid",
                )
            ]

    if retained_view is not None and (
        not isinstance(retained_view, Mapping)
        or not _valid_retained_view(retained_view)
    ):
        return [
            _finding(
                "retained_view_provenance",
                "retained-chain view provenance is incomplete or inconsistent",
            )
        ]
    if matcher_diagnostics is not None:
        if not isinstance(matcher_diagnostics, Mapping) or not (
            _valid_matcher_diagnostics(matcher_diagnostics)
        ):
            return [
                _finding(
                    "chain_compatibility",
                    "protected matcher diagnostics are incomplete or nonfinite",
                )
            ]
        accepted = _scalar_bool(matcher_diagnostics["accepted"])
        if not accepted and matcher_diagnostics["fallback"] != "identity":
            return [
                _finding(
                    "chain_compatibility",
                    "protected matcher found no legal supervised correspondence",
                    reason=str(matcher_diagnostics["reason"]),
                )
            ]

    if _has_nonfinite(features):
        return [_finding("nonfinite_feature", "floating-point features must be finite")]
    return []


def _seed(dataset: DatasetName, sample_id: str, crop_strategy: str) -> None:
    digest = hashlib.sha256(
        f"{dataset.value}:{sample_id}:{crop_strategy}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)


def initialize_worker(options: ScreeningOptions) -> None:
    """Set the immutable options used by subsequent worker-local preflight calls."""
    global _WORKER_OPTIONS
    _WORKER_OPTIONS = options
    _WORKER_DATASETS.clear()


def _worker_cache_dir(
    options: ScreeningOptions, dataset: DatasetName, crop: str
) -> Path:
    return (
        options.output_root
        / "worker-cache"
        / str(os.getpid())
        / f"{dataset.value}-{crop}"
    )


def _infrastructure_finding(sample_id: str, crop: str, exc: BaseException) -> Finding:
    return Finding(
        code="preflight_infrastructure",
        severity=Severity.INFRASTRUCTURE,
        message=f"{sample_id} ({crop}): {exc}",
        exception_type=type(exc).__name__,
        metrics={"sample_id": sample_id, "crop_strategy": crop},
    )


def _sample_error(sample_id: str, crop: str, stage: str, exc: BaseException) -> Finding:
    return Finding(
        code=f"{stage}_error",
        severity=Severity.LATEST_PREFLIGHT,
        message=f"{sample_id} ({crop}): {exc}",
        exception_type=type(exc).__name__,
        metrics={"sample_id": sample_id, "crop_strategy": crop},
    )


def _validate_collated(batch: Mapping[str, object], n_tokens: int) -> list[Finding]:
    if _shape(batch.get("physppi_surface_local_coords")) != (1, n_tokens, 128, 3):
        return [
            _finding(
                "collated_shape",
                "collated surface coordinates are invalid",
                stage="collator",
            )
        ]
    if _shape(batch.get("physppi_surface_features")) != (1, n_tokens, 128, 6):
        return [
            _finding(
                "collated_shape",
                "collated surface features are invalid",
                stage="collator",
            )
        ]
    if _shape(batch.get("physppi_surface_mask")) != (1, n_tokens, 128):
        return [
            _finding(
                "collated_shape",
                "collated surface mask is invalid",
                stage="collator",
            )
        ]
    if _shape(batch.get("physppi_surface_charge_mask")) != (1, n_tokens, 128):
        return [
            _finding(
                "collated_shape",
                "collated surface charge mask is invalid",
                stage="collator",
            )
        ]
    paired = _shape(batch.get("physppi_paired_msa_chain_mask"))
    if paired is None or len(paired) != 2 or paired[0] != 1:
        return [
            _finding(
                "collated_shape",
                "collated paired-MSA mask is invalid",
                stage="collator",
            )
        ]
    expected_pair_shapes = {
        "physppi_contact_pair_mask": (1, n_tokens, n_tokens),
        "physppi_interaction_type_mask": (1, n_tokens, n_tokens, 7),
        "physppi_interaction_distance": (1, n_tokens, n_tokens),
        "physppi_interaction_aux_valid": (1, n_tokens, n_tokens),
    }
    if any(
        _shape(batch.get(name)) != shape for name, shape in expected_pair_shapes.items()
    ):
        return [
            _finding(
                "collated_shape",
                "collated pair features are invalid",
                stage="collator",
            )
        ]
    return []


def _preflight_sample(
    dataset: DatasetName,
    index: int,
    sample_id: str,
    crop_strategy: str,
    *,
    global_slot: int,
    audit_view: DeterministicAuditView | None = None,
) -> list[Finding]:
    """Fetch, validate, and collate one exact dataset record deterministically."""
    if _WORKER_OPTIONS is None:
        raise RuntimeError(
            "initialize_worker(options) must run before preflight_sample"
        )
    options = _WORKER_OPTIONS
    try:
        _seed(dataset, sample_id, crop_strategy)
        if (
            not _source_root(dataset, options).is_dir()
            or not _source_index(dataset, options).is_file()
            or not options.components_cif.is_file()
            or not options.reference_molecule_directory.is_dir()
        ):
            raise FileNotFoundError("required shared PhysPPI resource is unavailable")
        key = (dataset, crop_strategy)
        current_dataset = _WORKER_DATASETS.get(key)
        if current_dataset is None:
            current_dataset = build_preflight_dataset(
                dataset, options, _worker_cache_dir(options, dataset, crop_strategy)
            )
            _WORKER_DATASETS[key] = current_dataset
        rows = current_dataset.datapoint_cache.index[
            current_dataset.datapoint_cache["pdb_id"] == sample_id
        ].tolist()
        if rows != [index]:
            raise ValueError(
                f"exact sample ID is not at requested index {index}: {rows}"
            )
        current_dataset.set_current_datapoint_index(index)
        current_dataset.set_current_occurrence(
            SampleOccurrence(
                dataset_idx=0,
                datapoint_idx=index,
                epoch=0,
                global_slot=global_slot,
            )
        )
        if audit_view is None:
            features = current_dataset[index]
        else:
            with current_dataset.audit_prior_view(
                requested_level=audit_view.requested_level,
                contact_cutoff=audit_view.contact_cutoff,
            ):
                features = current_dataset[index]
    except (MemoryError, TimeoutError, KeyboardInterrupt, SystemExit) as exc:
        return [_infrastructure_finding(sample_id, crop_strategy, exc)]
    except FileNotFoundError as exc:
        return [_infrastructure_finding(sample_id, crop_strategy, exc)]
    except Exception as exc:
        return [_sample_error(sample_id, crop_strategy, "dataset", exc)]

    findings = validate_physppi_features(features)
    if findings:
        return findings
    try:
        batch = openfold_batch_collator([copy.deepcopy(features)])
        return _validate_collated(batch, int(features["token_mask"].shape[0]))
    except (MemoryError, TimeoutError, KeyboardInterrupt, SystemExit) as exc:
        return [_infrastructure_finding(sample_id, crop_strategy, exc)]
    except Exception as exc:
        return [_sample_error(sample_id, crop_strategy, "collator", exc)]


def preflight_sample(
    dataset: DatasetName, index: int, sample_id: str, crop_strategy: str
) -> list[Finding]:
    """Backward-compatible deterministic preflight for one crop strategy."""
    return _preflight_sample(
        dataset,
        index,
        sample_id,
        crop_strategy,
        global_slot=0,
        audit_view=None,
    )


def preflight_audit_view(
    dataset: DatasetName,
    index: int,
    sample_id: str,
    view: DeterministicAuditView,
) -> list[Finding]:
    """Run one production feature pass at the view's deterministic occurrence."""
    return _preflight_sample(
        dataset,
        index,
        sample_id,
        view.crop_strategy,
        global_slot=view.global_slot,
        audit_view=view,
    )
