# =============================================================================
# Modified by Jinyu Lin (linjyshanghaitech), ShanghaiTech University
# Date: 2026-07-28
# Project: PhysPPI — https://github.com/linjyshanghaitech/PhysPPI_experiments
#
# Modification summary:
#   Integrated mesh_prior module: replaces build_surface_features() KNN with
#   pre-computed KDTree-based per-residue mesh tensors ({case_id}_mesh.pt).
#   Falls back to original KNN when .pt is missing.
#
#   See: experiments/mesh_prior/README.md for full context.
# =============================================================================

"""Shared OF3 runtime adapter for canonical PhysPPI prior-complex data."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from openfold3.core.data.framework.single_datasets.base_of3 import BaseOF3Dataset
from openfold3.core.data.framework.single_datasets.dataset_utils import (
    check_invalid_feature_dict,
)
from openfold3.core.data.io.sequence.msa import parse_msas_preparsed
from openfold3.core.data.io.structure.mol import (
    read_single_annotated_sdf,
    write_annotated_sdf,
)
from openfold3.core.data.io.structure.pdb import parse_pdb_af2
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.pipelines.featurization.loss_weights import set_loss_weights
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_of3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_of3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    get_reference_conformer_data_of3,
)
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_atom_array_of3,
)
from openfold3.core.data.primitives.caches.format import PreprocessingChainData
from openfold3.core.data.primitives.permutation.mol_labels import (
    separate_cropped_and_gt,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    MsaArrayCollection,
    calculate_profile,
)
from openfold3.core.data.primitives.structure.cleanup import (
    canonicalize_atom_order,
    convert_MSE_to_MET,
    remove_hydrogens,
    remove_non_CCD_atoms,
    remove_std_residue_terminal_atoms,
)
from openfold3.core.data.primitives.structure.component import mol_from_ccd_entry
from openfold3.core.data.primitives.structure.conformer import (
    resolve_and_format_fallback_conformer,
)
from openfold3.core.data.primitives.structure.template import (
    TemplateSlice,
    TemplateSliceCollection,
)
from openfold3.core.data.primitives.structure.tokenization import add_token_positions
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.physppi.data.adapters.chain_views import (
    CHAIN_VIEW_COMPATIBILITY_TRIGGER,
    CHAIN_VIEW_TOKEN_TRIGGER,
    RetainedChainView,
    build_chain_graph,
    needs_resource_view,
    sample_retained_chain_view,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    OptionalSlot,
    PriorLevelSelection,
    SlotState,
    available_optional_slots,
    build_paired_msa_chain_mask,
    select_prior_level,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.occurrence import (
    build_occurrence_rng,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    PriorComplexSample,
    PriorSamplePaths,
)
from openfold3.projects.of3_all_atom.physppi.features import (
    PhysPPIDataError,
    build_interaction_features,
    build_interface_residue_features,
    build_membrane_target,
    build_surface_features,
    build_tm_features,
    restrict_tm_payload_to_chains,
)

# Mesh Prior (KDTree-based, Jinyu Lin, ShanghaiTech)
# Replaces build_surface_features with pre-computed per-residue mesh tensors.
# See: experiments/mesh_prior/README.md
import torch as _torch

from experiments.mesh_prior.mesh_adapter import build_surface_from_precomputed
# PDB cache (Jinyu Lin, ShanghaiTech)
# Replaces on-the-fly PDB parsing + atom canonicalization with
# torch.load() from precomputed AtomArray .pt files.
from experiments.pdb_cache.pdb_adapter import (
    load_atomarray_from_pt as _load_atomarray_from_pt,
)
from experiments.residue_prior.residue_adapter import (
    build_interface_residue_from_precomputed as _build_interface_residue_from_pt,
    build_interaction_from_precomputed as _build_interaction_from_pt,
    build_membrane_target_from_precomputed as _build_membrane_target_from_pt,
    build_tm_from_precomputed as _build_tm_from_pt,
)


def _using_precomputed_root(adapter) -> bool:
    try:
        root = Path(adapter.physppi.root)
    except Exception:
        return False
    return root.parent.name == "cli_v3" and root.parent.parent.name == "precomputed_mesh"

# Precomputed .pt directory layout (Jinyu Lin, ShanghaiTech):
#   {physppi.root} may be either the old source tree
#     .../Prior_out/training/{dataset}/
#   or the strict precomputed tree
#     .../Prior_out/precomputed_mesh/cli_v3/{dataset}/
#   Precomputed dir = .../precomputed_mesh/cli_v3/{dataset}/{chunk}/


def _precomputed_dir(
    adapter,
    sample: PriorComplexSample,
) -> Path | None:
    """Return the precomputed .pt directory for *sample*.

    PhysPPI modification (Jinyu Lin, ShanghaiTech):
    Derived from ``physppi.root`` config and GT PDB path — NOT from the
    legacy ``prior_pkl`` path.  All precomputed data (mesh, residue, MSA,
    PDB cache) live under a single directory tree keyed by dataset name
    and chunk name.
    """
    try:
        root = Path(adapter.physppi.root)
        dataset_dir = root.name  # e.g. "Q-BioLiP-10k-new"
        if root.parent.name == "cli_v3" and root.parent.parent.name == "precomputed_mesh":
            precomputed_root = root
            chunk_name = sample.paths.gt_pdb.parent.name
        else:
            precomputed_root = root.parent.parent / "precomputed_mesh" / "cli_v3" / dataset_dir
            chunk_name = sample.paths.gt_pdb.parent.parent.name
        return precomputed_root / chunk_name
    except Exception:
        return None


def _derive_pt_path(
    adapter,
    sample: PriorComplexSample,
    pt_name: str,
) -> Path | None:
    """Derive a precomputed .pt file path from config, not from prior_pkl."""
    pt_dir = _precomputed_dir(adapter, sample)
    if pt_dir is None:
        return None
    return pt_dir / f"{sample.sample_id}_{pt_name}"


def _source_training_gt_pdb_path(sample: PriorComplexSample) -> Path | None:
    """Return the original training GT PDB for validation artifact export."""

    sample_paths = getattr(sample, "paths", None)
    gt_pdb = getattr(sample_paths, "gt_pdb", None)
    if gt_pdb is None:
        return None

    gt_path = Path(gt_pdb)
    if gt_path.suffix.lower() == ".pdb" and gt_path.is_file():
        return gt_path

    if gt_path.suffix.lower() != ".pt" or gt_path.parent.parent.parent.name != "cli_v3":
        return None

    dataset_dir = gt_path.parent.parent.name
    chunk_name = gt_path.parent.name
    prior_out_root = gt_path.parents[4]
    training_roots = []
    env_root = os.environ.get("PHYSPPI_TRAINING_ROOT")
    if env_root:
        training_roots.append(Path(env_root))
    training_roots.append(prior_out_root / "training")

    for training_root in training_roots:
        source_path = (
            training_root
            / dataset_dir
            / "chunk"
            / chunk_name
            / "gt"
            / f"{sample.sample_id}.pdb"
        )
        if source_path.is_file():
            return source_path
    return None


def _build_surface_mesh_prior(
    adapter,
    sample,
    token_index,
    transforms,
    input_atom_array,
):
    """Build surface features from pre-computed mesh .pt, falling back to KNN.

    PhysPPI modification (Jinyu Lin, ShanghaiTech):
    Attempts to load a pre-computed KDTree-based mesh tensor from the
    cli_v3 precomputed_mesh directory.  When the .pt is unavailable
    (e.g. during screening / missing chunk), falls back silently to the
    original build_surface_features().
    """
    import logging
    logger = logging.getLogger(__name__)
    pt_dir = _precomputed_dir(adapter, sample)
    if pt_dir is not None:
        try:
            mesh_pt = pt_dir / f"{sample.sample_id}_mesh.pt"
            if mesh_pt.exists():
                return build_surface_from_precomputed(
                    mesh_pt_path=mesh_pt,
                    token_index=token_index,
                    atom_array=input_atom_array,
                    monomer_transforms=transforms,
                    k=64,
                )
        except Exception as exc:
            if _using_precomputed_root(adapter):
                raise PhysPPIDataError(
                    "malformed_precomputed_mesh_pt",
                    f"failed to load required precomputed mesh: {mesh_pt}",
                ) from exc
            logger.debug(
                "mesh_prior unavailable for %s; using KNN fallback",
                sample.sample_id, exc_info=True,
            )
    # Fallback to original KNN method
    k = int(adapter.physppi.num_surface_neighbors)
    return build_surface_features(
        sample.mesh, token_index, transforms, k,
        atom_array=input_atom_array,
    )


def _resolve_physppi_sampling_weights(
    sample_ids: list[str],
    physppi,
) -> list[float]:
    """Load raw identity-based sampling weights for the selected sample IDs.

    PhysPPI modification (Jinyu Lin, ShanghaiTech):
      When a manifest is active the sibling ``sampling_weights.pt`` is
      loaded and per-sample weights are mapped to the filtered ID list.
      Raw weights are preserved so multiple datasets can be concatenated into
      one global categorical distribution. Validation never opens the weight
      file.
    """
    manifest = getattr(physppi, "manifest", None)
    if (
        manifest is None
        or getattr(physppi, "manifest_partition", "train_ids") != "train_ids"
    ):
        return [1.0] * len(sample_ids)
    configured_weights = getattr(physppi, "sampling_weights", None)
    weights_path = (
        Path(str(configured_weights))
        if configured_weights is not None
        else Path(str(manifest)).parent / "sampling_weights.pt"
    )
    try:
        raw = _torch.load(weights_path, map_location="cpu", weights_only=False)
    except (FileNotFoundError, RuntimeError) as exc:
        if configured_weights is not None:
            raise ValueError(
                f"could not load configured PhysPPI sampling weights: {weights_path}"
            ) from exc
        return [1.0] * len(sample_ids)
    raw_weights = raw.get("sampling_weights", {}) if isinstance(raw, dict) else {}
    if not raw_weights:
        raise ValueError(
            f"PhysPPI sampling weights are empty or malformed: {weights_path}"
        )
    missing = [sample_id for sample_id in sample_ids if sample_id not in raw_weights]
    if missing:
        raise ValueError(
            "PhysPPI sampling weights do not cover selected training IDs; "
            f"first missing IDs: {missing[:10]}"
        )
    probs = np.array(
        [float(raw_weights[sid]) for sid in sample_ids],
        dtype=np.float64,
    )
    invalid = ~np.isfinite(probs) | (probs <= 0)
    if invalid.any():
        invalid_ids = [
            sample_ids[index] for index in np.flatnonzero(invalid)[:10]
        ]
        raise ValueError(
            "PhysPPI sampling weights must be finite and positive; "
            f"invalid IDs: {invalid_ids}"
        )
    return probs.tolist()


def _has_valid_interface_dmap(sample: PriorComplexSample) -> bool:
    """Return whether the sample has at least one usable dmap chain edge."""

    try:
        graph = build_chain_graph(sample.chain_ids, sample.dmap)
    except Exception:
        return False
    return any(graph[left] for left in graph)


def _coerce_precomputed_dmap(
    raw_dmap: object,
) -> dict[str, tuple[tuple[int, int, float], ...]]:
    """Parse precomputed interface_residues.pt dmap rows without loading a sample."""

    if raw_dmap is None:
        return {}
    if not isinstance(raw_dmap, Mapping):
        raise ValueError("precomputed dmap must be a mapping")
    parsed: dict[str, tuple[tuple[int, int, float], ...]] = {}
    for chain_pair, rows in raw_dmap.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            raise ValueError("precomputed dmap key must use chain-pair syntax")
        row_array = rows.detach().cpu().numpy() if torch.is_tensor(rows) else rows
        parsed_rows: list[tuple[int, int, float]] = []
        for row in row_array:
            if len(row) != 3:
                raise ValueError("precomputed dmap row must have 3 columns")
            parsed_rows.append((int(row[0]), int(row[1]), float(row[2])))
        parsed[chain_pair] = tuple(parsed_rows)
    return parsed


def _has_precomputed_valid_interface_dmap(reader, sample_id: str) -> bool | None:
    """Use only interface_residues.pt for eager dmap filtering when available.

    Returning ``None`` means the reader is not a precomputed-only reader, so the
    caller should fall back to the full sample parser.
    """

    if not getattr(reader, "precomputed_only", False):
        return None
    paths = reader.locate(sample_id)
    gt_pdb = getattr(paths, "gt_pdb", None)
    monomer_pdbs = getattr(paths, "monomer_pdbs", None)
    if gt_pdb is None or not isinstance(monomer_pdbs, Mapping):
        return None
    interface_path = Path(gt_pdb).parent / f"{sample_id}_interface_residues.pt"
    if not interface_path.is_file():
        return False
    raw = _torch.load(interface_path, map_location="cpu", weights_only=False)
    raw_dmap = raw.get("dmap", {}) if isinstance(raw, Mapping) else {}
    graph = build_chain_graph(
        tuple(sorted(str(chain_id) for chain_id in monomer_pdbs)),
        _coerce_precomputed_dmap(raw_dmap),
    )
    return any(graph[left] for left in graph)


def _filter_sample_ids_with_valid_dmap(reader, sample_ids: list[str]) -> list[str]:
    retained: list[str] = []
    excluded: list[str] = []
    for sample_id in sample_ids:
        precomputed_status = _has_precomputed_valid_interface_dmap(reader, sample_id)
        if precomputed_status is None:
            sample = reader.read(sample_id)
            is_valid = _has_valid_interface_dmap(sample)
        else:
            is_valid = precomputed_status
        if is_valid:
            retained.append(sample_id)
        else:
            excluded.append(sample_id)
    if excluded:
        import logging

        logging.getLogger(__name__).warning(
            "excluded %d PhysPPI samples without valid dmap; first 10: %s",
            len(excluded),
            excluded[:10],
        )
    return retained


def _physppi_crop_attempt_seed(base_seed: int, attempt: int) -> int:
    payload = f"{int(base_seed)}:{int(attempt)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _retryable_physppi_crop_error(exc: PhysPPIDataError) -> bool:
    if not exc.args:
        return False
    reason = str(exc.args[0])
    return reason in {
        "invalid_physppi_crop",
        "missing_mandatory_monomer_geometry",
        "interface_preserving_crop_budget_exhausted",
        "interface_preserving_crop_too_few_chain_residues",
        "physppi_crop_residue_floor_violation",
    }


def _cropped_protein_residue_counts(atom_array: AtomArray) -> dict[str, int]:
    """Count cropped protein residues by chain in a tokenized AtomArray."""

    chain_ids = np.asarray(atom_array.chain_id).astype(str)
    res_ids = np.asarray(atom_array.res_id).astype(int)
    moltypes = (
        np.asarray(atom_array.molecule_type_id)
        if "molecule_type_id" in atom_array.get_annotation_categories()
        else np.full(len(atom_array), int(MoleculeType.PROTEIN), dtype=np.int64)
    )
    residues_by_chain: dict[str, set[int]] = {}
    for chain_id, residue_id, moltype in zip(
        chain_ids,
        res_ids,
        moltypes,
        strict=True,
    ):
        if int(moltype) != int(MoleculeType.PROTEIN):
            continue
        chain_id = str(chain_id)
        residues_by_chain.setdefault(chain_id, set()).add(int(residue_id))
    return {
        chain_id: len(residue_ids)
        for chain_id, residue_ids in sorted(residues_by_chain.items())
    }


def _validate_cropped_protein_residue_floor(
    atom_array: AtomArray,
    min_residues_per_chain: int,
    expected_chain_ids: tuple[str, ...] | None = None,
) -> None:
    counts = _cropped_protein_residue_counts(atom_array)
    chain_ids = (
        tuple(sorted(expected_chain_ids))
        if expected_chain_ids
        else tuple(counts)
    )
    short = {
        chain_id: counts.get(chain_id, 0)
        for chain_id in chain_ids
        if counts.get(chain_id, 0) < min_residues_per_chain
    }
    if short:
        raise PhysPPIDataError(
            "physppi_crop_residue_floor_violation",
            "cropped retained protein chains below residue floor "
            f"{min_residues_per_chain}: {short}",
        )


def _dmap_residue_counts(
    dmap: Mapping[str, tuple[tuple[int, int, float], ...]],
) -> dict[str, int]:
    residues_by_chain: dict[str, set[int]] = {}
    for chain_pair, rows in dmap.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        left, right = chain_pair.split("|", 1)
        for row in rows:
            residues_by_chain.setdefault(left, set()).add(int(row[0]))
            residues_by_chain.setdefault(right, set()).add(int(row[1]))
    return {
        chain_id: len(residue_ids)
        for chain_id, residue_ids in sorted(residues_by_chain.items())
    }


def _monomer_geometry_status(
    atom_array: AtomArray,
    monomer_atom_arrays: Mapping[str, AtomArray],
) -> dict[str, dict[str, int | bool]]:
    """Return the strict geometry stats used by mandatory monomer validation."""

    result: dict[str, dict[str, int | bool]] = {}
    retained_chains = sorted(str(chain) for chain in np.unique(atom_array.chain_id))
    for chain_id in retained_chains:
        monomer = monomer_atom_arrays.get(chain_id)
        if monomer is None:
            result[chain_id] = {
                "has_monomer": False,
                "matched_atom_count": 0,
                "representative_count": 0,
                "rank": 0,
            }
            continue

        lookup: dict[tuple[str, int, str], np.ndarray] = {}
        for index in range(len(monomer)):
            coordinate = np.asarray(monomer.coord[index], dtype=np.float32)
            if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
                continue
            key = (
                str(monomer.chain_id[index]),
                int(monomer.res_id[index]),
                str(monomer.atom_name[index]),
            )
            lookup.setdefault(key, coordinate)

        retained_indices = np.flatnonzero(
            np.asarray(atom_array.chain_id).astype(str) == chain_id
        )
        matched_atom_count = 0
        representative_positions: list[np.ndarray] = []
        residue_ids = np.asarray(atom_array.res_id[retained_indices], dtype=int)
        for residue_id in dict.fromkeys(residue_ids.tolist()):
            residue_indices = retained_indices[residue_ids == residue_id]
            names = np.asarray(atom_array.atom_name[residue_indices]).astype(str)
            ca_indices = residue_indices[names == "CA"]
            candidates = ca_indices if ca_indices.size else residue_indices
            for candidate in candidates:
                key = (
                    chain_id,
                    int(atom_array.res_id[candidate]),
                    str(atom_array.atom_name[candidate]),
                )
                if key in lookup:
                    matched_atom_count += 1
                    representative_positions.append(lookup[key])
                    break

        representatives = np.asarray(representative_positions, dtype=np.float32)
        rank = (
            int(np.linalg.matrix_rank(representatives - representatives[0]))
            if representatives.size
            else 0
        )
        result[chain_id] = {
            "has_monomer": True,
            "matched_atom_count": matched_atom_count,
            "representative_count": int(representatives.shape[0]),
            "rank": rank,
        }
    return result


def _physppi_crop_diagnostics(
    sample: PriorComplexSample,
    sample_data: dict | None,
) -> str:
    parts = [
        f"dmap_residue_counts={_dmap_residue_counts(getattr(sample, 'dmap', {}))}"
    ]
    if sample_data is None:
        return "; ".join(parts)
    atom_array = sample_data.get("atom_array_cropped")
    if atom_array is not None:
        residue_counts = _cropped_protein_residue_counts(atom_array)
        expected_chain_ids = tuple(
            sorted(
                str(chain_id)
                for chain_id in getattr(sample, "chain_sequences", {})
            )
        )
        if expected_chain_ids:
            residue_counts = {
                chain_id: residue_counts.get(chain_id, 0)
                for chain_id in expected_chain_ids
            }
        parts.append(
            "cropped_residue_counts="
            f"{residue_counts}"
        )
    monomer_inputs = sample_data.get("monomer_inputs")
    if atom_array is not None and monomer_inputs is not None:
        parts.append(
            "monomer_geometry="
            f"{_monomer_geometry_status(atom_array, monomer_inputs)}"
        )
    return "; ".join(parts)


# MSA .pt loading (Jinyu Lin, ShanghaiTech)
# Replaces _read_msa_npz() with torch.load() from precomputed _msa.pt
# files.  Each _msa.pt contains per-chain main/paired MSA arrays,
# deletion matrices, profiles, and deletion means — all precomputed
# from the original .npz files.  Falls back silently to .npz when the
# .pt is unavailable.

def _load_msa_from_pt(
    adapter,
    sample: PriorComplexSample,
    chain_ids: list[str],
) -> dict | None:
    """Load precomputed MSA arrays from ``_msa.pt`` for every chain.

    Returns a dict with ``main``, ``paired``, ``profile``, and
    ``deletion_mean`` keys, each mapping ``chain_id`` → MsaArray or
    numpy array.  Returns ``None`` when the .pt is missing or fails.
    """
    pt_path = _derive_pt_path(adapter, sample, "msa.pt")
    if pt_path is None or not pt_path.is_file():
        return None
    import logging
    logger = logging.getLogger(__name__)
    try:
        pt = _torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        if _using_precomputed_root(adapter):
            raise PhysPPIDataError(
                "malformed_precomputed_msa_pt",
                f"failed to load required precomputed MSA: {pt_path}",
            ) from exc
        logger.debug(
            "msa.pt load failed for %s; using .npz fallback",
            sample.sample_id, exc_info=True,
        )
        return None

    main = {}
    paired = {}
    profile = {}
    deletion_mean = {}

    pt_main = pt.get("main_msa", {})
    pt_main_del = pt.get("main_deletion", {})
    pt_paired = pt.get("paired_msa", {})
    pt_paired_del = pt.get("paired_deletion", {})
    pt_profile = pt.get("profile", {})
    pt_del_mean = pt.get("deletion_mean", {})

    for chain_id in chain_ids:
        msa_arr = pt_main.get(chain_id)
        if msa_arr is not None:
            del_arr = pt_main_del.get(chain_id)
            if del_arr is None:
                del_arr = np.zeros(msa_arr.shape, dtype=np.int32)
            else:
                del_arr = np.asarray(del_arr)
            main[chain_id] = MsaArray(
                msa=np.asarray(msa_arr),
                deletion_matrix=del_arr,
                metadata=pd.DataFrame(),
            )

        paired_arr = pt_paired.get(chain_id)
        if paired_arr is not None:
            paired_del = pt_paired_del.get(chain_id)
            if paired_del is None:
                paired_del = np.zeros(paired_arr.shape, dtype=np.int32)
            else:
                paired_del = np.asarray(paired_del)
            paired[chain_id] = MsaArray(
                msa=np.asarray(paired_arr),
                deletion_matrix=paired_del,
                metadata=pd.DataFrame(),
            )

        prof = pt_profile.get(chain_id)
        if prof is not None:
            profile[chain_id] = np.asarray(prof, dtype=np.float32)

        dm = pt_del_mean.get(chain_id)
        if dm is not None:
            deletion_mean[chain_id] = np.asarray(dm, dtype=np.float32)

    if not main:
        return None
    return {
        "main": main,
        "paired": paired,
        "profile": profile,
        "deletion_mean": deletion_mean,
    }


def _build_residue_priors(
    adapter,
    sample: PriorComplexSample,
    token_index: dict[tuple[str, int], int],
    cutoff: float,
    atom_array,
):
    """Build interface/interaction/membrane features from precomputed .pt files.

    PhysPPI modification (Jinyu Lin, ShanghaiTech):
    Each prior is independently loaded from its .pt.  If a .pt is missing
    or fails to load, that prior silently returns None so the caller can
    fall back to the original builder.
    """
    import logging
    logger = logging.getLogger(__name__)

    interface = None
    interaction = None
    tm_mask = None
    membrane = None

    # --- interface residues ---
    pt_path = _derive_pt_path(adapter, sample, "interface_residues.pt")
    if pt_path is not None and pt_path.exists():
        try:
            interface = _build_interface_residue_from_pt(
                pt_path, token_index, cutoff
            )
        except Exception as exc:
            if _using_precomputed_root(adapter):
                raise PhysPPIDataError(
                    "malformed_precomputed_interface_pt",
                    f"failed to load precomputed interface residues: {pt_path}",
                ) from exc
            logger.debug(
                "interface_residues.pt failed for %s; falling back", sample.sample_id,
                exc_info=True,
            )

    # --- interaction pairs ---
    pt_path = _derive_pt_path(adapter, sample, "interaction_pairs.pt")
    if pt_path is not None and pt_path.exists():
        try:
            interaction = _build_interaction_from_pt(
                pt_path, token_index, atom_array=atom_array,
            )
        except Exception as exc:
            if _using_precomputed_root(adapter):
                raise PhysPPIDataError(
                    "malformed_precomputed_interaction_pt",
                    f"failed to load precomputed interaction pairs: {pt_path}",
                ) from exc
            logger.debug(
                "interaction_pairs.pt failed for %s; falling back", sample.sample_id,
                exc_info=True,
            )

    # --- membrane relative location ---
    pt_path = _derive_pt_path(adapter, sample, "membrane_relative_location.pt")
    if pt_path is not None and pt_path.exists():
        try:
            tm_mask = _build_tm_from_pt(pt_path, token_index)
            membrane = _build_membrane_target_from_pt(pt_path)
        except Exception as exc:
            if _using_precomputed_root(adapter):
                raise PhysPPIDataError(
                    "malformed_precomputed_membrane_pt",
                    f"failed to load precomputed membrane relative location: {pt_path}",
                ) from exc
            logger.debug(
                "membrane_relative_location.pt failed for %s; falling back",
                sample.sample_id, exc_info=True,
            )

    return interface, interaction, tm_mask, membrane


from openfold3.projects.of3_all_atom.physppi.model.interaction_auxiliary import (
    _build_pair_interaction_auxiliary,
)

_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

_CHAIN_ID_ENCODING_WIDTH = 16
_CHAIN_VIEW_TRIGGER_CODES = {
    "none": 0,
    "token_budget": 1,
    "compatibility_group": 2,
    "token_budget+compatibility_group": 3,
}
_CHAIN_VIEW_MATCHER_CODES = {
    "pending": 0,
    "identity": 1,
    "accepted": 2,
    "fallback": 3,
}


def _filter_chain_keyed_paths(
    paths: Mapping[str, Path],
    original_chain_ids: set[str],
    retained_chain_ids: set[str],
) -> dict[str, Path]:
    """Drop chain-owned paths while preserving sample-level shared files."""

    return {
        key: path
        for key, path in paths.items()
        if key not in original_chain_ids or key in retained_chain_ids
    }


def restrict_prior_sample_to_chains(
    sample: PriorComplexSample,
    chain_ids: tuple[str, ...],
) -> PriorComplexSample:
    """Return an immutable prior sample restricted to one retained chain view."""

    retained = set(chain_ids)
    original = set(sample.chain_ids)
    if not retained:
        raise ValueError("retained-chain view cannot be empty")
    if not retained <= original:
        missing = sorted(retained - original)
        raise ValueError(f"retained-chain view contains unknown chains: {missing}")

    def pair_is_retained(chain_pair: str) -> bool:
        return set(chain_pair.split("|")) <= retained

    paths = PriorSamplePaths(
        sample_id=sample.paths.sample_id,
        gt_pdb=sample.paths.gt_pdb,
        monomer_pdbs=_filter_chain_keyed_paths(
            sample.paths.monomer_pdbs, original, retained
        ),
        main_msa_npzs=_filter_chain_keyed_paths(
            sample.paths.main_msa_npzs, original, retained
        ),
        paired_msa_npzs=_filter_chain_keyed_paths(
            sample.paths.paired_msa_npzs, original, retained
        ),
        prior_pkl=sample.paths.prior_pkl,
    )
    if set(paths.monomer_pdbs) != retained:
        missing = sorted(retained - set(paths.monomer_pdbs))
        stale = sorted(set(paths.monomer_pdbs) - retained)
        raise PhysPPIDataError(
            "retained_chain_monomer_mismatch",
            f"missing retained monomers={missing}; stale monomers={stale}",
        )
    opm = dict(sample.opm)
    res_loc = opm.get("res_loc")
    if res_loc is not None:
        opm["res_loc"] = restrict_tm_payload_to_chains(res_loc, chain_ids)

    restricted = replace(
        sample,
        dmap={
            chain_pair: rows
            for chain_pair, rows in sample.dmap.items()
            if pair_is_retained(chain_pair)
        },
        interactions=tuple(
            interaction
            for interaction in sample.interactions
            if pair_is_retained(interaction.chain_pair)
        ),
        mesh={
            chain_id: rows
            for chain_id, rows in sample.mesh.items()
            if chain_id in retained
        },
        opm=opm,
        paths=paths,
        chain_sequences={
            chain_id: sequence
            for chain_id, sequence in sample.chain_sequences.items()
            if chain_id in retained
        },
    )
    if set(restricted.chain_ids) != retained:
        raise ValueError("retained-chain restriction left stale or missing chain state")
    return restricted


def _restrict_gt_atom_array_to_sample(
    atom_array: AtomArray,
    sample: PriorComplexSample,
) -> AtomArray:
    """Filter GT atom rows to the already-restricted sample inventory."""

    retained_ids = getattr(sample, "chain_ids", None)
    if retained_ids is None:
        retained_ids = tuple(getattr(sample, "chain_sequences", {}))
    retained = set(retained_ids)
    chain_ids = np.asarray(atom_array.chain_id).astype(str)
    filtered = atom_array[np.isin(chain_ids, sorted(retained))]
    observed = set(np.asarray(filtered.chain_id).astype(str))
    if observed != retained:
        missing = sorted(retained - observed)
        stale = sorted(observed - retained)
        raise PhysPPIDataError(
            "retained_chain_gt_mismatch",
            f"missing retained GT chains={missing}; stale GT chains={stale}",
        )
    return filtered


def _uncropped_token_count(sample: PriorComplexSample) -> int:
    chain_ids = sample.chain_ids
    sequences = sample.chain_sequences
    if all(sequences.get(chain_id) for chain_id in chain_ids):
        return sum(len(sequences[chain_id]) for chain_id in chain_ids)

    atom_array = _restrict_gt_atom_array_to_sample(
        _parse_pdb_atom_array(sample.paths.gt_pdb), sample
    )
    return int(len(struc.get_residue_starts(atom_array)))


def _sequences_are_hard_compatible(left: str, right: str) -> bool:
    """Conservative protein compatibility for the pre-featurization trigger."""

    return len(left) == len(right) and all(
        a == b or a in {"X", "?"} or b in {"X", "?"}
        for a, b in zip(left, right, strict=True)
    )


def _max_compatibility_group(sample: PriorComplexSample) -> int:
    sequences = {
        chain_id: sample.chain_sequences.get(chain_id, "")
        for chain_id in sample.chain_ids
    }
    if not sequences:
        return 0
    joined_sequences = "".join(sequences.values())
    if all(sequences.values()) and not {"X", "?"} & set(joined_sequences):
        return max(Counter(sequences.values()).values())

    remaining = set(sequences)
    largest = 1
    while remaining:
        start = min(remaining)
        component = {start}
        frontier = [start]
        remaining.remove(start)
        while frontier:
            current = frontier.pop()
            compatible = sorted(
                candidate
                for candidate in remaining
                if _sequences_are_hard_compatible(
                    sequences[current], sequences[candidate]
                )
            )
            for candidate in compatible:
                remaining.remove(candidate)
                component.add(candidate)
                frontier.append(candidate)
        largest = max(largest, len(component))
    return largest


def select_retained_chain_view(
    sample: PriorComplexSample,
    rng: np.random.Generator,
    occurrence_seed: int,
    candidate_cap: int = 1024,
) -> RetainedChainView:
    """Select the sole retained view for this immutable sample occurrence."""

    chain_ids = sample.chain_ids
    token_count = _uncropped_token_count(sample)
    max_compatibility_group = _max_compatibility_group(sample)
    token_triggered = token_count > CHAIN_VIEW_TOKEN_TRIGGER
    compatibility_triggered = max_compatibility_group > CHAIN_VIEW_COMPATIBILITY_TRIGGER
    if not needs_resource_view(
        len(chain_ids),
        token_count,
        max_compatibility_group,
    ):
        return RetainedChainView(
            source_chain_ids=chain_ids,
            trigger="none",
            component_chain_ids=chain_ids,
            seed_edge=None,
            target_chain_count=len(chain_ids),
            parent_sample_id=sample.sample_id,
            original_chain_ids=chain_ids,
            occurrence_seed=occurrence_seed,
            matcher_result="pending",
        )

    trigger = (
        "token_budget+compatibility_group"
        if token_triggered and compatibility_triggered
        else "token_budget"
        if token_triggered
        else "compatibility_group"
    )
    return sample_retained_chain_view(
        build_chain_graph(chain_ids, sample.dmap),
        rng,
        trigger,
        chain_lengths={
            chain_id: len(sample.chain_sequences.get(chain_id, ""))
            for chain_id in chain_ids
        },
        candidate_cap=candidate_cap,
        parent_sample_id=sample.sample_id,
        occurrence_seed=occurrence_seed,
    )


def _encode_chain_ids(chain_ids: tuple[str, ...]) -> torch.Tensor:
    encoded = torch.zeros((len(chain_ids), _CHAIN_ID_ENCODING_WIDTH), dtype=torch.uint8)
    for row, chain_id in enumerate(chain_ids):
        value = chain_id.encode("utf-8")
        if len(value) > _CHAIN_ID_ENCODING_WIDTH:
            raise ValueError(
                f"chain id {chain_id!r} exceeds {_CHAIN_ID_ENCODING_WIDTH} UTF-8 bytes"
            )
        if value:
            encoded[row, : len(value)] = torch.tensor(tuple(value), dtype=torch.uint8)
    return encoded


def encode_retained_chain_view(
    view: RetainedChainView,
) -> dict[str, torch.Tensor]:
    """Encode complete replayable view provenance in batch-safe tensors."""

    try:
        trigger_code = _CHAIN_VIEW_TRIGGER_CODES[view.trigger]
        matcher_code = _CHAIN_VIEW_MATCHER_CODES[view.matcher_result]
    except KeyError as exc:
        raise ValueError(
            f"unknown retained-chain provenance value: {exc.args[0]}"
        ) from exc
    seed_edge = view.seed_edge if view.seed_edge is not None else ("", "")
    return {
        "physppi_original_chain_ids": _encode_chain_ids(view.original_chain_ids),
        "physppi_retained_chain_ids": _encode_chain_ids(view.source_chain_ids),
        "physppi_retained_component_chain_ids": _encode_chain_ids(
            view.component_chain_ids
        ),
        "physppi_retained_seed_edge": _encode_chain_ids(seed_edge),
        "physppi_retained_view_trigger": torch.tensor(trigger_code, dtype=torch.int64),
        "physppi_retained_target_chain_count": torch.tensor(
            view.target_chain_count, dtype=torch.int64
        ),
        "physppi_retained_dmap_cutoff": torch.tensor(
            view.dmap_cutoff, dtype=torch.float32
        ),
        "physppi_retained_occurrence_seed": torch.tensor(
            view.occurrence_seed, dtype=torch.uint64
        ),
        "physppi_retained_matcher_result": torch.tensor(
            matcher_code, dtype=torch.int64
        ),
    }


@cache
def _canonical_protein_atom_names(
    reference_molecule_directory: Path, residue_name: str
) -> frozenset[str]:
    """Return the atom-name vocabulary used by a canonical residue reference."""
    molecule = read_single_annotated_sdf(
        reference_molecule_directory / f"{residue_name}.sdf"
    )
    return frozenset(atom.GetProp("annot_atom_name") for atom in molecule.GetAtoms())


def _ensure_ccd_reference_molecule(
    residue_name: str, reference_cache_directory: Path, *, ccd
) -> Path:
    """Materialize an OF3-compatible annotated reference SDF from the CCD.

    PhysPPI supplies protein-only PDBs, but a protein chain can still contain a
    noncanonical peptide residue such as N-formylmethionine (FME).  Those residues
    are not part of the small static canonical-protein SDF bundle.  OF3 normally
    creates their annotated SDFs during preprocessing from ``components.cif``;
    create the same artifact in a writable runtime cache here instead of mutating
    the checked-in canonical reference directory.
    """

    reference_cache_directory.mkdir(parents=True, exist_ok=True)
    output_path = reference_cache_directory / f"{residue_name}.sdf"
    if output_path.is_file():
        return output_path

    molecule = mol_from_ccd_entry(residue_name, ccd)
    molecule, _ = resolve_and_format_fallback_conformer(molecule)
    temporary_path = reference_cache_directory / (f".{residue_name}.{os.getpid()}.sdf")
    try:
        write_annotated_sdf(molecule, temporary_path)
        # Atomic replacement makes concurrent DataLoader workers converge on one
        # cache entry without exposing a partially-written SDF.
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _token_starts(atom_array: AtomArray) -> np.ndarray:
    token_ids = np.asarray(atom_array.token_id)
    _, starts = np.unique(token_ids, return_index=True)
    return np.sort(starts)


def _token_center_for_indices(atom_array: AtomArray, indices: np.ndarray) -> int:
    if "token_center_atom" in atom_array.get_annotation_categories():
        center_indices = indices[np.asarray(atom_array.token_center_atom[indices]).astype(bool)]
        if center_indices.size:
            return int(center_indices[0])
    names = np.asarray(atom_array.atom_name[indices]).astype(str)
    ca_indices = indices[names == "CA"]
    return int(ca_indices[0] if ca_indices.size else indices[0])


def _stable_jittered_order(
    items,
    rng: np.random.Generator,
    key,
) -> list:
    ordered = sorted(items, key=key)
    jitters = rng.random(len(ordered)) if ordered else np.asarray([], dtype=float)
    return [
        item
        for _, item in sorted(
            zip(jitters, ordered, strict=True),
            key=lambda pair: (key(pair[1]), float(pair[0])),
        )
    ]


def _physppi_interface_preserving_crop_mask(
    atom_array: AtomArray,
    dmap: Mapping[str, tuple[tuple[int, int, float], ...]],
    *,
    token_budget: int,
    min_residues_per_chain: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a PhysPPI crop mask that prioritizes chain coverage and dmap."""

    starts = _token_starts(atom_array)
    token_ids = np.asarray(atom_array.token_id)
    chain_ids = np.asarray(atom_array.chain_id).astype(str)
    res_ids = np.asarray(atom_array.res_id).astype(int)
    moltypes = (
        np.asarray(atom_array.molecule_type_id)
        if "molecule_type_id" in atom_array.get_annotation_categories()
        else np.full(len(atom_array), int(MoleculeType.PROTEIN), dtype=np.int64)
    )

    residue_to_tokens: dict[tuple[str, int], set[int]] = {}
    residue_coord: dict[tuple[str, int], np.ndarray] = {}
    protein_residues_by_chain: dict[str, set[tuple[str, int]]] = {}
    token_to_indices: dict[int, np.ndarray] = {}
    for token_id in np.unique(token_ids):
        indices = np.flatnonzero(token_ids == token_id)
        token_to_indices[int(token_id)] = indices
    for start in starts:
        token_id = int(token_ids[start])
        chain_id = str(chain_ids[start])
        residue_id = int(res_ids[start])
        key = (chain_id, residue_id)
        residue_to_tokens.setdefault(key, set()).add(token_id)
        center_index = _token_center_for_indices(atom_array, token_to_indices[token_id])
        residue_coord.setdefault(
            key,
            np.asarray(atom_array.coord[center_index], dtype=np.float32),
        )
        if int(moltypes[start]) == int(MoleculeType.PROTEIN):
            protein_residues_by_chain.setdefault(chain_id, set()).add(key)

    dmap_priority: dict[tuple[str, int], float] = {}
    for chain_pair, rows in dmap.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        left, right = chain_pair.split("|", 1)
        for row in rows:
            left_key = (left, int(row[0]))
            right_key = (right, int(row[1]))
            distance = float(row[2])
            for key in (left_key, right_key):
                if key not in residue_to_tokens:
                    continue
                dmap_priority[key] = min(distance, dmap_priority.get(key, distance))

    if not dmap_priority:
        raise PhysPPIDataError(
            "missing_interface_dmap_for_retained_view",
            "retained view has no dmap residues that map to tokens",
        )

    selected_residues: set[tuple[str, int]] = set()
    selected_tokens: set[int] = set()

    def add_residue(key: tuple[str, int], *, required: bool) -> bool:
        tokens = residue_to_tokens.get(key, set())
        new_tokens = tokens - selected_tokens
        if not new_tokens:
            selected_residues.add(key)
            return True
        if len(selected_tokens) + len(new_tokens) > token_budget:
            if required:
                raise PhysPPIDataError(
                    "interface_preserving_crop_budget_exhausted",
                    "cannot satisfy required per-chain residue floor within token budget",
                )
            return False
        selected_residues.add(key)
        selected_tokens.update(new_tokens)
        return True

    def min_distance_to(keys: set[tuple[str, int]], coord: np.ndarray) -> float:
        coords = [
            residue_coord[key]
            for key in keys
            if key in residue_coord and np.isfinite(residue_coord[key]).all()
        ]
        if not coords:
            return float("inf")
        stacked = np.stack(coords, axis=0)
        return float(np.linalg.norm(stacked - coord.reshape(1, 3), axis=1).min())

    for chain_id in sorted(protein_residues_by_chain):
        chain_residues = protein_residues_by_chain[chain_id]
        chain_dmap = [
            key for key in chain_residues if key in dmap_priority
        ]
        chain_dmap = _stable_jittered_order(
            chain_dmap,
            rng,
            key=lambda item: (dmap_priority[item], item[0], item[1]),
        )
        anchors = list(chain_dmap[:min_residues_per_chain])
        if len(anchors) < min_residues_per_chain:
            anchor_set = set(chain_dmap)
            context = [
                key for key in chain_residues if key not in anchor_set
            ]
            context = _stable_jittered_order(
                context,
                rng,
                key=lambda item: (
                    min_distance_to(anchor_set, residue_coord[item]),
                    item[0],
                    item[1],
                ),
            )
            anchors.extend(context[: min_residues_per_chain - len(anchors)])
        if len(anchors) < min_residues_per_chain:
            raise PhysPPIDataError(
                "interface_preserving_crop_too_few_chain_residues",
                f"chain {chain_id!r} has fewer than {min_residues_per_chain} residues",
            )
        for key in anchors:
            add_residue(key, required=True)

    dmap_residues = _stable_jittered_order(
        dmap_priority,
        rng,
        key=lambda item: (dmap_priority[item], item[0], item[1]),
    )
    for key in dmap_residues:
        add_residue(key, required=False)

    retained_interface = set(selected_residues) & set(dmap_priority)
    if not retained_interface:
        retained_interface = set(selected_residues)
    context_residues = [
        key for key in residue_to_tokens if key not in selected_residues
    ]
    context_residues = _stable_jittered_order(
        context_residues,
        rng,
        key=lambda item: (
            min_distance_to(retained_interface, residue_coord[item]),
            item[0],
            item[1],
        ),
    )
    for key in context_residues:
        if len(selected_tokens) >= token_budget:
            break
        add_residue(key, required=False)

    mask = np.isin(token_ids, np.asarray(sorted(selected_tokens), dtype=token_ids.dtype))
    return np.asarray(mask, dtype=bool)


def _crop_monomer_template_to_query_tokens(
    template_atom_array: AtomArray, query_token_atoms: AtomArray
) -> TemplateSlice | None:
    """Align template tokens to the query without duplicating atomized residues."""

    has_ins_code = (
        "ins_code" in template_atom_array.get_annotation_categories()
        and "ins_code" in query_token_atoms.get_annotation_categories()
    )
    has_is_atomized = (
        "is_atomized" in template_atom_array.get_annotation_categories()
        and "is_atomized" in query_token_atoms.get_annotation_categories()
    )

    def token_key(atom) -> tuple[int, str, str | None]:
        ins_code = str(atom.ins_code) if has_ins_code else ""
        atom_name = (
            str(atom.atom_name) if has_is_atomized and bool(atom.is_atomized) else None
        )
        return int(atom.res_id), ins_code, atom_name

    template_starts = struc.get_residue_starts(template_atom_array)
    template_stops = np.append(template_starts[1:], len(template_atom_array))
    residue_atom_indices = {
        int(start): np.arange(start, stop)
        for start, stop in zip(template_starts, template_stops, strict=True)
    }
    template_token_indices: dict[tuple[int, str, str | None], np.ndarray | None] = {}
    for template_start in _token_starts(template_atom_array):
        template_atom = template_atom_array[template_start]
        indices = (
            np.asarray([template_start])
            if has_is_atomized and bool(template_atom.is_atomized)
            else residue_atom_indices[int(template_start)]
        )
        key = token_key(template_atom)
        template_token_indices[key] = (
            indices if key not in template_token_indices else None
        )

    selected_residue_indices: dict[int, list[int]] = {}
    selected_query_positions: dict[int, list[int]] = {}
    used_template_indices: set[int] = set()
    for query_atom in query_token_atoms:
        indices = template_token_indices.get(token_key(query_atom))
        if indices is None or any(
            int(index) in used_template_indices for index in indices
        ):
            continue
        residue_index = np.searchsorted(template_starts, indices[0], side="right") - 1
        residue_start = int(template_starts[residue_index])
        selected_residue_indices.setdefault(residue_start, []).extend(map(int, indices))
        selected_query_positions.setdefault(residue_start, []).append(
            int(query_atom.token_position)
        )
        used_template_indices.update(map(int, indices))

    if not selected_residue_indices:
        return None
    selected_residue_starts = sorted(selected_residue_indices)
    atom_indices = [
        np.asarray(selected_residue_indices[start], dtype=int)
        for start in selected_residue_starts
    ]
    query_positions = [
        position
        for start in selected_residue_starts
        for position in selected_query_positions[start]
    ]
    residue_repeats = [
        len(selected_query_positions[start]) for start in selected_residue_starts
    ]
    return TemplateSlice(
        atom_array=template_atom_array[np.concatenate(atom_indices)],
        query_token_positions=np.asarray(query_positions, dtype=int),
        template_residue_repeats=np.asarray(residue_repeats, dtype=int),
    )


def _chain_metadata(atom_array: AtomArray) -> dict[str, PreprocessingChainData]:
    metadata: dict[str, PreprocessingChainData] = {}
    for chain_id in sorted(str(chain) for chain in np.unique(atom_array.chain_id)):
        chain = atom_array[atom_array.chain_id == chain_id]
        categories = chain.get_annotation_categories()
        if "entity_id" in categories:
            entity_id = int(chain.entity_id[0])
        elif "mol_entity_id" in categories:
            entity_id = int(chain.mol_entity_id[0])
        else:
            entity_id = 1
        metadata[chain_id] = PreprocessingChainData(
            label_asym_id=chain_id,
            auth_asym_id=chain_id,
            entity_id=entity_id,
            molecule_type=MoleculeType.PROTEIN,
            reference_mol_id=None,
        )
    return metadata


def _parse_pdb_atom_array(path: Path) -> AtomArray:
    parsed = parse_pdb_af2(path, allow_malformed_charge=True)
    if not hasattr(parsed, "atom_array"):
        raise ValueError(f"failed to parse canonical PDB: {path}")
    return parsed.atom_array


def _normalize_fme_carbonyl_oxygen(atom_array: AtomArray) -> AtomArray:
    """Disambiguate legacy PDB FME oxygen names for the official CCD.

    The CCD names FME's formyl oxygen ``O1`` and its backbone carbonyl oxygen
    ``O``.  Q-BioLiP PDBs can label both atoms ``O``.  The formyl oxygen is the
    one geometrically bonded to (and therefore nearest) the formyl carbon ``CN``.
    """

    atom_array = atom_array.copy()
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for residue_start, residue_stop in zip(
        residue_starts[:-1], residue_starts[1:], strict=True
    ):
        residue = atom_array[residue_start:residue_stop]
        if residue.res_name[0] != "FME" or "O1" in residue.atom_name:
            continue
        cn_indices = np.flatnonzero(residue.atom_name == "CN")
        oxygen_indices = np.flatnonzero(residue.atom_name == "O")
        if len(cn_indices) != 1 or len(oxygen_indices) < 2:
            continue
        distances = np.linalg.norm(
            residue.coord[oxygen_indices] - residue.coord[cn_indices[0]], axis=1
        )
        formyl_oxygen_index = residue_start + oxygen_indices[np.argmin(distances)]
        atom_array.atom_name[formyl_oxygen_index] = "O1"
    return atom_array


def _canonical_ppi_heavy_atom_array(
    atom_array: AtomArray,
    reference_molecule_directory: Path | None = None,
    ccd=None,
) -> AtomArray:
    """Normalize PPI atoms to the canonical protein reference vocabulary."""

    if reference_molecule_directory is None:
        reference_molecule_directory = (
            Path(__file__).resolve().parents[5]
            / "core/data/resources/canonical_protein_data/reference_mols"
        )
    # The canonical reference set represents selenomethionine as MET and does
    # not provide an MSE SDF. Normalize before looking up residue vocabularies
    # or constructing reference conformers.
    atom_array = atom_array.copy()
    convert_MSE_to_MET(atom_array)
    atom_array = atom_array[atom_array.molecule_type_id == MoleculeType.PROTEIN]
    if len(atom_array) == 0:
        raise ValueError("canonical PPI structure contains no protein heavy atoms")
    atom_array = _normalize_fme_carbonyl_oxygen(atom_array)
    atom_array = remove_hydrogens(atom_array)
    atom_array = remove_std_residue_terminal_atoms(atom_array)
    keep_mask = np.ones(len(atom_array), dtype=bool)
    needs_ccd = False
    noncanonical_residue_names: set[str] = set()
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for residue_start, residue_stop in zip(
        residue_starts[:-1], residue_starts[1:], strict=True
    ):
        residue = atom_array[residue_start:residue_stop]
        reference_path = reference_molecule_directory / f"{residue.res_name[0]}.sdf"
        if not reference_path.is_file():
            needs_ccd = True
            noncanonical_residue_names.add(str(residue.res_name[0]))
            continue
        allowed_names = _canonical_protein_atom_names(
            reference_molecule_directory, str(residue.res_name[0])
        )
        keep_mask[residue_start:residue_stop] = np.isin(
            residue.atom_name, list(allowed_names)
        )
    atom_array = atom_array[keep_mask]
    if needs_ccd:
        if ccd is None:
            raise ValueError(
                "noncanonical protein residues require dataset_paths.ccd_file: "
                + ", ".join(sorted(noncanonical_residue_names))
            )
        atom_array = remove_non_CCD_atoms(atom_array, ccd)
        atom_array = canonicalize_atom_order(atom_array, ccd)
    if len(atom_array) == 0:
        raise ValueError("canonical PPI structure contains no heavy atoms")
    return atom_array


def _annotate_canonical_mol_entity_ids(
    atom_array: AtomArray, sample: PriorComplexSample | None = None
) -> AtomArray:
    categories = atom_array.get_annotation_categories()
    has_entity_id = "entity_id" in categories
    has_mol_entity_id = "mol_entity_id" in categories
    if has_entity_id and has_mol_entity_id:
        if not np.array_equal(atom_array.entity_id, atom_array.mol_entity_id):
            raise ValueError("canonical entity_id and mol_entity_id disagree")
        return atom_array
    if has_entity_id:
        atom_array.set_annotation("mol_entity_id", atom_array.entity_id.copy())
        return atom_array
    if has_mol_entity_id:
        atom_array.set_annotation("entity_id", atom_array.mol_entity_id.copy())
        return atom_array

    chain_ids = sorted(str(chain) for chain in np.unique(atom_array.chain_id))
    sample_sequences = (
        getattr(sample, "chain_sequences", {}) if sample is not None else {}
    )
    sequence_to_entity_id: dict[str, int] = {}
    chain_to_entity_id: dict[str, int] = {}
    for chain_id in chain_ids:
        sequence = str(sample_sequences.get(chain_id, "")) or _sequence_for_chain(
            atom_array, chain_id
        )
        if not sequence:
            raise ValueError(
                "cannot infer mol_entity_id for canonical PhysPPI chain "
                f"{chain_id!r}: missing chain sequence"
            )
        entity_id = sequence_to_entity_id.setdefault(
            sequence, len(sequence_to_entity_id) + 1
        )
        chain_to_entity_id[chain_id] = entity_id

    entity_ids = np.asarray(
        [chain_to_entity_id[str(chain_id)] for chain_id in atom_array.chain_id],
        dtype=int,
    )
    atom_array.set_annotation("entity_id", entity_ids)
    atom_array.set_annotation("mol_entity_id", entity_ids)
    return atom_array


def _sequence_for_chain(atom_array: AtomArray, chain_id: str) -> str:
    chain = atom_array[atom_array.chain_id == chain_id]
    starts = struc.get_residue_starts(chain)
    residues = [str(res) for res in chain[starts].res_name]
    return "".join(_AA3_TO_1.get(residue, "X") for residue in residues)


def _query_only_msa(sequence: str) -> MsaArray:
    msa = np.asarray([list(sequence)], dtype="<U1")
    return MsaArray(
        msa=msa,
        deletion_matrix=np.zeros(msa.shape, dtype=int),
        metadata=pd.DataFrame(),
    )


def _read_msa_npz(path: Path) -> MsaArray:
    parsed = parse_msas_preparsed([path])
    if not parsed:
        raise ValueError(f"MSA npz does not contain any arrays: {path}")
    arrays = list(parsed.values())
    if len(arrays) == 1:
        return arrays[0]
    return MsaArray.multi_concatenate(arrays, axis=0)


def _empty_msa_like(reference: MsaArray) -> MsaArray:
    return MsaArray(
        msa=reference.msa[:0, :],
        deletion_matrix=reference.deletion_matrix[:0, :],
        metadata=pd.DataFrame(),
    )


def _msa_query_sequence(msa: MsaArray) -> str:
    if len(msa) == 0:
        return ""
    return "".join(np.asarray(msa.msa[0]).astype(str).tolist())


def _normalized_msa_query_sequence(msa: MsaArray) -> str:
    """Return the query row without alignment-gap columns."""

    return _msa_query_sequence(msa).replace("-", "")


def _project_msa_to_query_tokens(msa: MsaArray, query_msa: MsaArray) -> MsaArray | None:
    """Project an MSA onto the residue tokens observed in the GT structure.

    Local MSA files are indexed by the full deposited sequence, whereas the
    structure can omit unresolved residues. The MSA query must therefore be a
    supersequence of the GT query after alignment gaps are removed. A greedy
    subsequence match is deterministic and succeeds whenever such a projection
    exists; a GT ``X`` is treated as an unknown residue.
    """

    source = np.asarray(msa.msa)
    target = _normalized_msa_query_sequence(query_msa)
    if source.ndim != 2 or source.shape[0] == 0 or not target:
        return None
    if np.asarray(msa.deletion_matrix).shape != source.shape:
        return None

    source_columns = np.flatnonzero(source[0].astype(str) != "-")
    source_query = "".join(source[0, source_columns].astype(str).tolist())
    if not source_query:
        return None
    if source_query == target and source.shape[1] == len(target):
        return msa

    def _select_columns(allow_unknown: bool) -> list[int] | None:
        selected_columns: list[int] = []
        target_index = 0
        for column in source_columns:
            if target_index == len(target):
                break
            residue = str(source[0, column])
            target_residue = target[target_index]
            if residue == target_residue or (allow_unknown and target_residue == "X"):
                selected_columns.append(int(column))
                target_index += 1
        return selected_columns if target_index == len(target) else None

    selected_columns = _select_columns(allow_unknown=False)
    if selected_columns is None:
        selected_columns = _select_columns(allow_unknown=True)
    if selected_columns is None:
        return None
    selected = np.asarray(selected_columns, dtype=int)
    return MsaArray(
        msa=source[:, selected],
        deletion_matrix=np.asarray(msa.deletion_matrix)[:, selected],
        metadata=msa.metadata,
    )


def _afdb_query_matches_expected(actual: str, expected: str) -> bool:
    """Allow only AFDB's observed trailing unknown-residue padding."""

    if not actual.startswith(expected):
        return False
    return set(actual[len(expected) :]) <= {"X"}


def _is_afdb_sample(sample: PriorComplexSample) -> bool:
    dataset_name = str(getattr(sample, "dataset_name", "")).lower().replace("-", "")
    return dataset_name in {"afdb", "afdb10k"}


def _afdb_homodimer_shared_main_msa(
    sample: PriorComplexSample,
    chain_ids: list[str],
    query_msas: dict[str, MsaArray],
    main_by_key: dict[str, MsaArray],
) -> MsaArray | None:
    if not _is_afdb_sample(sample) or len(chain_ids) != 2 or len(main_by_key) != 1:
        return None
    chain_sequences = getattr(sample, "chain_sequences", {}) or {}
    expected_sequences = [
        str(chain_sequences.get(chain_id, "")) for chain_id in chain_ids
    ]
    if any(not sequence for sequence in expected_sequences):
        return None
    if len(set(expected_sequences)) != 1:
        return None
    query_sequences = [
        _normalized_msa_query_sequence(query_msas[chain_id]) for chain_id in chain_ids
    ]
    if any(
        not _afdb_query_matches_expected(actual, expected)
        for actual, expected in zip(query_sequences, expected_sequences)
    ):
        return None
    source_msa = next(iter(main_by_key.values()))
    if not _afdb_query_matches_expected(
        _normalized_msa_query_sequence(source_msa), expected_sequences[0]
    ):
        return None
    return source_msa


def _build_main_msas(
    sample: PriorComplexSample,
    chain_ids: list[str],
    query_msas: dict[str, MsaArray],
) -> tuple[dict[str, MsaArray], dict[str, str]]:
    """Build per-chain main MSAs with explicit source provenance.

    The model always receives a main-MSA tensor.  A searched MSA is preferred;
    AFDB/afdb10k homodimers can share one validated representative, while a
    missing searched file degrades to a query-only row and is audited as a
    warning rather than dropping the sample.
    """

    main_by_key = {
        key: _read_msa_npz(path) for key, path in sample.paths.main_msa_npzs.items()
    }
    exact = {
        chain_id: main_by_key[chain_id]
        for chain_id in chain_ids
        if chain_id in main_by_key
    }
    shared_msa = _afdb_homodimer_shared_main_msa(
        sample,
        chain_ids,
        query_msas,
        main_by_key,
    )
    main_msas: dict[str, MsaArray] = {}
    provenance: dict[str, str] = {}
    for chain_id in chain_ids:
        source_msa: MsaArray | None = None
        source_provenance: str | None = None
        if chain_id in exact:
            source_msa = exact[chain_id]
            source_provenance = "chain_file"
        elif shared_msa is not None:
            source_msa = shared_msa
            source_provenance = "shared_homodimer_file"

        if source_msa is not None:
            projected = _project_msa_to_query_tokens(source_msa, query_msas[chain_id])
            if projected is not None:
                main_msas[chain_id] = projected
                provenance[chain_id] = source_provenance
                continue

        if chain_id in query_msas:
            # A query-only row is still a mandatory MSA tensor at the model
            # boundary. A malformed or incompatible searched MSA falls back
            # here rather than creating an invalid feature batch.
            main_msas[chain_id] = query_msas[chain_id]
            provenance[chain_id] = "query_only"
        else:
            raise AssertionError("unreachable missing main MSA path")
    return main_msas, provenance


def deterministic_source_mixture_indices(
    labels: list[str] | tuple[str, ...],
    epoch_len: int,
    fractions: dict[str, float] | None = None,
) -> list[int]:
    """Return deterministic source-balanced indices using per-sample labels."""

    if epoch_len < 0:
        raise ValueError("epoch_len must be non-negative")
    label_array = [str(label) for label in labels]
    if not label_array or epoch_len == 0:
        return []

    fractions = fractions or {"qbiolip10k": 0.8, "afdb10k": 0.2}
    by_label = {
        label: [index for index, value in enumerate(label_array) if value == label]
        for label in dict.fromkeys(label_array)
    }
    label_order = [label for label in fractions if label in by_label]
    label_order.extend(label for label in by_label if label not in fractions)
    weights = {
        label: float(fractions.get(label, 1.0))
        for label in label_order
        if by_label[label] and float(fractions.get(label, 1.0)) > 0.0
    }
    if not weights:
        raise ValueError("at least one label must have a positive sampling fraction")

    total_weight = sum(weights.values())
    raw_counts = {
        label: epoch_len * weight / total_weight for label, weight in weights.items()
    }
    counts = {label: int(np.floor(count)) for label, count in raw_counts.items()}
    remaining = epoch_len - sum(counts.values())
    remainder_order = sorted(
        weights,
        key=lambda label: (
            raw_counts[label] - counts[label],
            -label_order.index(label),
        ),
        reverse=True,
    )
    for label in remainder_order[:remaining]:
        counts[label] += 1

    result: list[int] = []
    for label in label_order:
        if label not in counts:
            continue
        source_indices = by_label[label]
        for offset in range(counts[label]):
            result.append(source_indices[offset % len(source_indices)])
    return result


def _token_index(atom_array: AtomArray, features: dict) -> dict[tuple[str, int], int]:
    starts = _token_starts(atom_array)
    token_ids = features["token_index"].detach().cpu().numpy()
    token_mask = features["token_mask"].detach().cpu().numpy().astype(bool)
    valid_token_ids = token_ids[token_mask]
    if len(starts) != len(valid_token_ids):
        raise ValueError(
            "canonical atom tokens and valid feature tokens must have matching lengths"
        )
    return {
        (str(atom_array.chain_id[start]), int(atom_array.res_id[start])): dense_index
        for dense_index, start in enumerate(starts)
    }


def _pad_prior_payloads_to_token_count(
    payloads: dict[str, np.ndarray], n_tokens: int
) -> dict[str, np.ndarray]:
    """Pad cropped optional-prior payloads to the model token budget.

    The crop contains only real tokens, whereas OF3 structure features are padded to
    the configured token budget.  Optional inputs must share that padded token space;
    their newly added rows and columns are inactive zeros.
    """
    padded: dict[str, np.ndarray] = {}
    for name, payload in payloads.items():
        value = np.asarray(payload)
        if value.ndim < 1:
            raise ValueError(f"{name} must have a token axis")
        actual_n_tokens = value.shape[0]
        if actual_n_tokens > n_tokens:
            raise ValueError(
                f"{name} has {actual_n_tokens} tokens, exceeding model budget "
                f"{n_tokens}"
            )
        is_pair_payload = name in {"interaction"} and value.ndim >= 2
        if is_pair_payload:
            if value.shape[1] != actual_n_tokens:
                raise ValueError(
                    f"{name} pair axes must have matching token counts, got "
                    f"{value.shape[:2]}"
                )
            output = np.zeros((n_tokens, n_tokens, *value.shape[2:]), dtype=value.dtype)
            output[:actual_n_tokens, :actual_n_tokens] = value
        else:
            output = np.zeros((n_tokens, *value.shape[1:]), dtype=value.dtype)
            output[:actual_n_tokens] = value
        padded[name] = output
    return padded


def _geometry_features(
    atom_array: AtomArray,
    token_mask: torch.Tensor,
    monomer_positions: np.ndarray,
    monomer_mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    coords = np.asarray(monomer_positions, dtype=np.float32)
    atom_mask = np.asarray(monomer_mask, dtype=bool)
    if coords.shape != (len(atom_array), 3) or atom_mask.shape != (len(atom_array),):
        raise ValueError("monomer geometry must match the cropped model atom axis")
    token_ids = np.asarray(atom_array.token_id)
    starts = _token_starts(atom_array)
    n_tokens = int(token_mask.shape[0])
    center = np.full(n_tokens, -1, dtype=np.int64)
    backbone = np.full((n_tokens, 3), -1, dtype=np.int64)
    frame_mask = np.zeros(n_tokens, dtype=bool)
    for token, (_, token_id) in enumerate(zip(starts, token_ids[starts], strict=True)):
        if token >= n_tokens:
            break
        atom_indices = np.flatnonzero(token_ids == token_id)
        if atom_indices.size == 0:
            continue
        names = np.asarray(atom_array.atom_name[atom_indices]).astype(str)
        center_candidates = atom_indices[names == "CA"]
        center[token] = int(
            center_candidates[0] if center_candidates.size else atom_indices[0]
        )
        triplet = []
        for name in ("N", "CA", "C"):
            candidates = atom_indices[names == name]
            triplet.append(int(candidates[0]) if candidates.size else -1)
        backbone[token] = triplet
        frame_mask[token] = (
            all(index >= 0 for index in triplet)
            and atom_mask[triplet].all()
            and np.isfinite(coords[triplet]).all()
        )
    return {
        "physppi_token_center_atom_index": torch.from_numpy(center),
        "physppi_backbone_atom_indices": torch.from_numpy(backbone),
        "physppi_backbone_frame_mask": torch.from_numpy(frame_mask),
    }


def build_monomer_input_geometry(
    atom_array: AtomArray,
    monomer_atom_arrays: dict[str, AtomArray],
) -> dict[str, np.ndarray]:
    """Map chain-local monomer coordinates onto the cropped model atom axis.

    The cropped complex contributes only atom identity and ordering. Coordinates
    are copied exclusively from mandatory per-chain monomer/template inputs.
    """

    monomer_positions = np.zeros((len(atom_array), 3), dtype=np.float32)
    monomer_mask = np.zeros((len(atom_array),), dtype=bool)
    retained_chains = sorted(str(chain) for chain in np.unique(atom_array.chain_id))
    for chain_id in retained_chains:
        monomer = monomer_atom_arrays.get(chain_id)
        if monomer is None:
            raise PhysPPIDataError(
                "missing_mandatory_monomer_geometry",
                f"retained chain {chain_id!r} has no monomer input",
            )

        lookup: dict[tuple[str, int, str], np.ndarray] = {}
        for index in range(len(monomer)):
            coordinate = np.asarray(monomer.coord[index], dtype=np.float32)
            if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
                continue
            key = (
                str(monomer.chain_id[index]),
                int(monomer.res_id[index]),
                str(monomer.atom_name[index]),
            )
            lookup.setdefault(key, coordinate)

        retained_indices = np.flatnonzero(
            np.asarray(atom_array.chain_id).astype(str) == chain_id
        )
        for index in retained_indices:
            key = (
                chain_id,
                int(atom_array.res_id[index]),
                str(atom_array.atom_name[index]),
            )
            coordinate = lookup.get(key)
            if coordinate is None:
                continue
            monomer_positions[index] = coordinate
            monomer_mask[index] = True

        representative_positions: list[np.ndarray] = []
        residue_ids = np.asarray(atom_array.res_id[retained_indices], dtype=int)
        for residue_id in dict.fromkeys(residue_ids.tolist()):
            residue_indices = retained_indices[residue_ids == residue_id]
            names = np.asarray(atom_array.atom_name[residue_indices]).astype(str)
            ca_indices = residue_indices[names == "CA"]
            candidates = ca_indices if ca_indices.size else residue_indices
            matched = candidates[monomer_mask[candidates]]
            if matched.size:
                representative_positions.append(monomer_positions[matched[0]])

        representatives = np.asarray(representative_positions, dtype=np.float32)
        has_four = representatives.shape[0] >= 4
        non_collinear = has_four and (
            np.linalg.matrix_rank(representatives - representatives[0]) >= 2
        )
        if not non_collinear:
            raise PhysPPIDataError(
                "missing_mandatory_monomer_geometry",
                f"retained chain {chain_id!r} lacks four finite, "
                "non-collinear representative atoms",
            )

    return {
        "physppi_monomer_atom_positions": monomer_positions.astype(np.float32),
        "physppi_monomer_atom_mask": monomer_mask.astype(bool),
    }


def _slot_state_features(
    states: dict[OptionalSlot, SlotState],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for slot, state in states.items():
        prefix = f"physppi_{slot.value}"
        result.update(
            {
                f"{prefix}_available": torch.tensor(state.available),
                f"{prefix}_selected": torch.tensor(state.selected),
                f"{prefix}_valid": torch.tensor(state.valid),
                f"{prefix}_provenance": torch.tensor(state.provenance == "source"),
                f"{prefix}_active": torch.tensor(state.active),
            }
        )
    return result


def _resolve_post_build_prior_states(states: dict[OptionalSlot, SlotState]) -> None:
    """Truncate an initial cumulative prefix after post-view validity checks."""
    requested_level = sum(state.selected for state in states.values())
    selectable = {
        slot: state.available and state.valid for slot, state in states.items()
    }
    resolved = select_prior_level(selectable, requested_level)
    states.update(
        {
            slot: replace(
                resolved.states[slot],
                available=state.available,
                valid=state.valid,
                provenance=state.provenance,
            )
            for slot, state in states.items()
        }
    )


@dataclass(frozen=True)
class _AuditPriorViewOverride:
    """Screening-only deterministic optional-prior view."""

    requested_level: int
    interface_cutoff: float


class PhysPPIPriorComplexDatasetAdapter(BaseOF3Dataset):
    """Shared canonical runtime for Q-BioLiP and AFDB prior-complex datasets."""

    reader_name = "PhysPPI"

    def __init__(self, dataset_config, world_size: int | None = None) -> None:
        super().__init__(dataset_config)
        ccd_file = dataset_config.dataset_paths.ccd_file
        self._physppi_ccd_file = Path(ccd_file) if ccd_file is not None else None
        self._physppi_ccd = None
        self._physppi_reference_cache_directory = (
            Path(self._dataset_cache_file).parent / "physppi_ccd_reference_mols"
        )
        self._physppi_data_seed = int(getattr(dataset_config, "data_seed", 42))
        self._physppi_audit_view_override: _AuditPriorViewOverride | None = None

    @contextmanager
    def audit_prior_view(
        self, *, requested_level: int, interface_cutoff: float
    ) -> Iterator[None]:
        """Temporarily force one screening-only optional-prior audit view."""
        if getattr(self, "_physppi_audit_view_override", None) is not None:
            raise RuntimeError("PhysPPI audit prior view is already active")
        if requested_level not in range(5):
            raise ValueError("requested_level must be in [0, 4]")
        if not np.isfinite(interface_cutoff) or not 8.0 <= interface_cutoff <= 18.0:
            raise ValueError("interface_cutoff must be finite and in [8.0, 18.0]")
        self._physppi_audit_view_override = _AuditPriorViewOverride(
            requested_level=requested_level,
            interface_cutoff=float(interface_cutoff),
        )
        try:
            yield
        finally:
            self._physppi_audit_view_override = None

    def _select_prior_level(
        self,
        available: Mapping[OptionalSlot, bool],
        rng: np.random.Generator,
    ) -> PriorLevelSelection:
        """Select a production level, with an audit-only deterministic override."""
        sampled_level = int(rng.integers(0, 5))
        override = getattr(self, "_physppi_audit_view_override", None)
        requested_level = (
            override.requested_level if override is not None else sampled_level
        )
        return select_prior_level(available, requested_level)

    def _select_interface_cutoff(self, rng: np.random.Generator) -> float:
        """Select a production cutoff, with an audit-only deterministic override."""
        sampled_cutoff = float(rng.uniform(8.0, 18.0))
        override = getattr(self, "_physppi_audit_view_override", None)
        return override.interface_cutoff if override is not None else sampled_cutoff

    def _ccd_for_atom_array(self, atom_array: AtomArray):
        """Load the official CCD only for samples needing a nonstatic residue."""

        protein_atom_array = atom_array[
            atom_array.molecule_type_id == MoleculeType.PROTEIN
        ]
        residue_names = set(map(str, protein_atom_array.res_name))
        residue_names.discard("MSE")  # normalized to MET before CCD processing
        needs_ccd = any(
            not (self.reference_molecule_directory / f"{name}.sdf").is_file()
            for name in residue_names
        )
        if not needs_ccd:
            return None
        if self._physppi_ccd_file is None:
            missing = sorted(
                name
                for name in residue_names
                if not (self.reference_molecule_directory / f"{name}.sdf").is_file()
            )
            raise ValueError(
                "noncanonical PhysPPI protein residues require "
                "dataset_paths.ccd_file; missing reference SDFs for "
                + ", ".join(missing)
            )
        if self._physppi_ccd is None:
            self._physppi_ccd = pdbx.CIFFile.read(self._physppi_ccd_file)
        return self._physppi_ccd

    def _ensure_ccd_reference_molecules(self, atom_array: AtomArray, ccd) -> None:
        for residue_name in sorted(set(map(str, atom_array.res_name))):
            if (self.reference_molecule_directory / f"{residue_name}.sdf").is_file():
                continue
            _ensure_ccd_reference_molecule(
                residue_name,
                self._physppi_reference_cache_directory,
                ccd=ccd,
            )

    def create_datapoint_cache_from_reader(self, reader, physppi) -> None:
        sample_ids = list(reader.sample_ids())
        manifest_bound = physppi.manifest is not None
        if manifest_bound:
            if physppi.sample_ids:
                raise ValueError("manifest and sample_ids are mutually exclusive")
            try:
                manifest = json.loads(Path(physppi.manifest).read_text())
                manifest_ids = manifest["datasets"][physppi.manifest_dataset_key][
                    physppi.manifest_partition
                ]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "invalid PhysPPI manifest: expected "
                    "datasets.<dataset key>.<partition>"
                ) from exc
            if not isinstance(manifest_ids, list) or not all(
                isinstance(sample_id, str) for sample_id in manifest_ids
            ):
                raise ValueError("PhysPPI manifest partition must be a list of IDs")
            reader_ids = set(sample_ids)
            selected_ids = set(manifest_ids)
            missing_ids = selected_ids - reader_ids
            if missing_ids:
                import logging
                _logger = logging.getLogger(__name__)
                _logger.warning(
                    "PhysPPI manifest contains %d IDs absent from the reader "
                    "(e.g. still in .chunk/ staging); they will be excluded "
                    "from this run. First 10: %s",
                    len(missing_ids),
                    sorted(missing_ids)[:10],
                )
            sample_ids = [
                sample_id for sample_id in sample_ids if sample_id in selected_ids
            ]
        elif physppi.sample_ids:
            allowed = set(physppi.sample_ids)
            sample_ids = [sample_id for sample_id in sample_ids if sample_id in allowed]
        if physppi.max_samples is not None:
            sample_ids = sample_ids[: int(physppi.max_samples)]

        if not manifest_bound:
            sample_ids = _filter_sample_ids_with_valid_dmap(reader, sample_ids)

        # PhysPPI modification (Jinyu Lin, ShanghaiTech):
        #   Load identity-based sampling weights when a manifest is active.
        #   Raw weights are preserved for the global datapoint sampler.
        datapoint_probabilities = _resolve_physppi_sampling_weights(
            sample_ids, physppi
        )

        self.datapoint_cache = pd.DataFrame(
            {
                "pdb_id": sample_ids,
                "preferred_chain_or_interface": [None] * len(sample_ids),
                "datapoint_probabilities": datapoint_probabilities,
                "n_clust": [1] * len(sample_ids),
            }
        )

    @staticmethod
    def _available_slots(sample: PriorComplexSample) -> dict[OptionalSlot, bool]:
        return available_optional_slots(sample)

    def create_canonical_structure_features(
        self,
        sample: PriorComplexSample,
        preferred_chain_or_interface: str | list[str, str] | None,
        return_full_atom_array: bool,
        return_crop_strategy: bool,
        crop_attempt_seed: int | None = None,
    ) -> dict:
        # PhysPPI modification (Jinyu Lin, ShanghaiTech):
        #   Prefer precomputed GT AtomArray from _gt.pt.  The .pt already
        #   contains the canonized heavy-atom structure so we skip PDB
        #   parsing + _canonical_ppi_heavy_atom_array.  Falls back to the
        #   original PDB path when the .pt is missing.
        gt_pt = _derive_pt_path(self, sample, "gt.pt")
        gt_from_cache = None
        if gt_pt is not None and gt_pt.is_file():
            try:
                gt_from_cache = _load_atomarray_from_pt(gt_pt)
            except Exception as exc:
                if _using_precomputed_root(self):
                    raise PhysPPIDataError(
                        "malformed_precomputed_gt_pt",
                        f"failed to load required precomputed GT: {gt_pt}",
                    ) from exc
        elif _using_precomputed_root(self):
            raise PhysPPIDataError(
                "missing_precomputed_gt_pt",
                f"required precomputed GT is missing: {gt_pt}",
            )

        if gt_from_cache is not None:
            raw_atom_array = _restrict_gt_atom_array_to_sample(
                gt_from_cache, sample
            )
            atom_array = raw_atom_array  # already canonized in .pt
            ccd = self._ccd_for_atom_array(atom_array)
            self._ensure_ccd_reference_molecules(atom_array, ccd)
            atom_array = _annotate_canonical_mol_entity_ids(atom_array, sample)
        else:
            raw_atom_array = _restrict_gt_atom_array_to_sample(
                _parse_pdb_atom_array(sample.paths.gt_pdb), sample
            )
            ccd = self._ccd_for_atom_array(raw_atom_array)
            atom_array = _canonical_ppi_heavy_atom_array(
                raw_atom_array, self.reference_molecule_directory, ccd=ccd
            )
            self._ensure_ccd_reference_molecules(atom_array, ccd)
            atom_array = _annotate_canonical_mol_entity_ids(atom_array, sample)

        per_chain_metadata = _chain_metadata(atom_array)
        token_crop_settings = self.crop["token_crop"]
        use_physppi_crop = (
            bool(getattr(self.physppi, "interface_preserving_crop", True))
            and bool(token_crop_settings["enabled"])
        )
        if use_physppi_crop:
            processed = process_target_atom_array_of3(
                atom_array=atom_array,
                pdb_id=sample.sample_id,
                crop_config={
                    "chain_crop": {"enabled": False},
                    "token_crop": {"enabled": False},
                },
                preferred_chain_or_interface=preferred_chain_or_interface,
                per_chain_metadata=per_chain_metadata,
            )
            atom_array_gt = processed.atom_array_gt
            token_budget = int(token_crop_settings["token_budget"])
            crop_mask = _physppi_interface_preserving_crop_mask(
                atom_array_gt,
                sample.dmap,
                token_budget=token_budget,
                min_residues_per_chain=int(
                    getattr(
                        self.physppi,
                        "min_residues_per_retained_protein_chain",
                        15,
                    )
                ),
                rng=np.random.default_rng(
                    0 if crop_attempt_seed is None else int(crop_attempt_seed)
                ),
            )
            atom_array_gt.set_annotation("crop_mask", crop_mask)
            crop_strategy = "spatial_interface"
            self.n_tokens = token_budget
        else:
            processed = process_target_atom_array_of3(
                atom_array=atom_array,
                pdb_id=sample.sample_id,
                crop_config=self.crop,
                preferred_chain_or_interface=preferred_chain_or_interface,
                per_chain_metadata=per_chain_metadata,
            )
            atom_array_gt = processed.atom_array_gt
            crop_strategy = processed.crop_strategy
            self.n_tokens = processed.n_tokens
        processed_reference_molecules = get_reference_conformer_data_of3(
            atom_array=atom_array_gt,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=self.dataset_cache.reference_molecule_data,
            reference_mol_dir=self.reference_molecule_directory,
            reference_mol_fallback_dir=self._physppi_reference_cache_directory,
        )
        atom_array_full = atom_array_gt.copy() if return_full_atom_array else None
        atom_array_cropped, atom_array_gt = separate_cropped_and_gt(
            atom_array_gt=atom_array_gt,
            crop_strategy=crop_strategy,
            processed_ref_mol_list=processed_reference_molecules,
        )
        add_token_positions(atom_array_cropped)
        target_structure_features = featurize_target_gt_structure_of3(
            atom_array=atom_array_cropped,
            atom_array_gt=atom_array_gt,
            n_tokens=self.n_tokens,
        )
        reference_conformer_features = featurize_reference_conformers_of3(
            processed_ref_mol_list=processed_reference_molecules
        )
        target_structure_data = {
            "atom_array_gt": atom_array_gt,
            "atom_array_cropped": atom_array_cropped,
            "target_structure_features": target_structure_features,
            "reference_conformer_features": reference_conformer_features,
        }
        if return_full_atom_array:
            target_structure_data["atom_array"] = atom_array_full
        if return_crop_strategy:
            target_structure_data["crop_strategy"] = crop_strategy
        return target_structure_data

    def create_canonical_msa_features(
        self, sample: PriorComplexSample, atom_array: AtomArray
    ) -> dict:
        chain_ids = [str(chain) for chain in np.unique(atom_array.chain_id)]
        chain_to_rep = {chain_id: chain_id for chain_id in chain_ids}
        rep_to_chain = dict(chain_to_rep)
        chain_to_mol = {chain_id: MoleculeType.PROTEIN for chain_id in chain_ids}
        rep_to_mol = dict(chain_to_mol)
        query_msas = {
            chain_id: _query_only_msa(_sequence_for_chain(atom_array, chain_id))
            for chain_id in chain_ids
        }

        # PhysPPI modification (Jinyu Lin, ShanghaiTech):
        #   Prefer precomputed _msa.pt over reading individual .npz files.
        #   The .pt stores per-chain MsaArrays (main + paired) that replace
        #   _build_main_msas / _read_msa_npz.  Projection, truncation, and
        #   profile computation still run on top of the loaded arrays.
        pt_msa = _load_msa_from_pt(self, sample, chain_ids)

        if pt_msa is not None:
            # PhysPPI modification (Jinyu Lin, ShanghaiTech):
            #   Precomputed MSAs have columns for ALL GT residues (different
            #   per chain).  Project each chain to its query tokens so that
            #   column counts match across chains for vstack_pad_msa_arrays.
            main_msas = {}
            for chain_id in chain_ids:
                src = pt_msa["main"].get(chain_id)
                if src is not None:
                    projected = _project_msa_to_query_tokens(src, query_msas[chain_id])
                    if projected is not None:
                        main_msas[chain_id] = projected
                        continue
                # Fall back to query-only (same as _build_main_msas)
                main_msas[chain_id] = query_msas[chain_id]
            paired_by_path = pt_msa["paired"]
            pt_profiles = pt_msa.get("profile", {})
            pt_deletion_mean = pt_msa.get("deletion_mean", {})
        else:
            main_msas, _ = _build_main_msas(sample, chain_ids, query_msas)
            paired_by_path = {
                key: _read_msa_npz(path)
                for key, path in sample.paths.paired_msa_npzs.items()
            }
            pt_profiles = {}
            pt_deletion_mean = {}

        paired_msas: dict[str, MsaArray] = {}
        paired_available: list[bool] = []
        for chain_id in chain_ids:
            source_msa = paired_by_path.get(chain_id)
            if source_msa is None:
                source_msa = paired_by_path.get(sample.sample_id)
            projected = (
                _project_msa_to_query_tokens(source_msa, query_msas[chain_id])
                if source_msa is not None
                else None
            )
            if projected is None:
                paired_available.append(False)
            else:
                paired_msas[chain_id] = projected
                paired_available.append(True)
        msa_settings = getattr(self, "msa_settings", None)
        max_rows = getattr(msa_settings, "max_rows", 16384)
        max_rows_paired = getattr(msa_settings, "max_rows_paired", 8192)
        max_paired_rows = min(
            max((len(msa) for msa in paired_msas.values()), default=0),
            max_rows_paired,
            max_rows - 1,
        )
        paired_msas = {
            chain_id: msa.truncate(max_paired_rows)
            for chain_id, msa in paired_msas.items()
        }
        if max_paired_rows:
            for chain_id in chain_ids:
                paired_msas.setdefault(chain_id, _empty_msa_like(query_msas[chain_id]))
        max_main_rows = max_rows - max_paired_rows - 1
        main_msas = {
            chain_id: msa.truncate(max_main_rows) for chain_id, msa in main_msas.items()
        }
        paired_chain_mask = build_paired_msa_chain_mask(paired_available)
        collection = MsaArrayCollection(
            chain_id_to_rep_id=chain_to_rep,
            chain_id_to_mol_type=chain_to_mol,
            rep_id_to_chain_id=rep_to_chain,
            rep_id_to_mol_type=rep_to_mol,
        )
        # PhysPPI modification (Jinyu Lin, ShanghaiTech):
        #   Use precomputed profile / deletion_mean from _msa.pt when the
        #   MSA was not projected (column count unchanged).  Otherwise fall
        #   back to computing them from the projected MSA.
        profiles: dict[str, np.ndarray] = {}
        deletion_mean: dict[str, np.ndarray] = {}
        for chain_id in chain_ids:
            msa = main_msas[chain_id]
            pt_prof = pt_profiles.get(chain_id)
            pt_dm = pt_deletion_mean.get(chain_id)
            if (
                pt_prof is not None
                and pt_dm is not None
                and pt_prof.shape[0] == msa.msa.shape[1]
            ):
                profiles[chain_id] = pt_prof
                deletion_mean[chain_id] = pt_dm
            else:
                profiles[chain_id] = calculate_profile(
                    msa_array=msa.msa,
                    molecule_type=MoleculeType.PROTEIN,
                    chunk_size=1000,
                )
                deletion_mean[chain_id] = np.mean(
                    msa.deletion_matrix, axis=0
                )
        max_main_rows = max(len(msa) for msa in main_msas.values()) if main_msas else 0
        collection.set_state_processed(
            chain_id_to_query_seq=query_msas,
            chain_id_to_paired_msa=paired_msas,
            chain_id_to_main_msa=main_msas,
            chain_id_to_profile=profiles,
            chain_id_to_deletion_mean=deletion_mean,
        )
        collection.set_row_counts(
            n_rows_total=1 + max_paired_rows + max_main_rows,
            n_rows_paired_subsampled=max_paired_rows,
            n_rows_main_subsampled={
                chain_id: len(msa) for chain_id, msa in main_msas.items()
            },
        )
        features = self.msa_featurizer_of3(
            atom_array=atom_array,
            msa_array_collection=collection,
            n_tokens=self.n_tokens,
        )
        features["physppi_paired_msa_chain_mask"] = torch.from_numpy(paired_chain_mask)
        return features

    def _process_monomer_template(self, path: Path) -> AtomArray:
        raw_atom_array = _parse_pdb_atom_array(path)
        ccd = self._ccd_for_atom_array(raw_atom_array)
        atom_array = _canonical_ppi_heavy_atom_array(
            raw_atom_array, self.reference_molecule_directory, ccd=ccd
        )
        atom_array = _annotate_canonical_mol_entity_ids(atom_array)
        processed = process_target_atom_array_of3(
            atom_array=atom_array,
            pdb_id=path.stem,
            crop_config={
                "chain_crop": {"enabled": False},
                "token_crop": {"enabled": False},
            },
            preferred_chain_or_interface=None,
            per_chain_metadata=_chain_metadata(atom_array),
        )
        add_token_positions(processed.atom_array_gt)
        return processed.atom_array_gt

    def _load_mandatory_monomer_inputs(
        self, sample: PriorComplexSample, atom_array: AtomArray
    ) -> dict[str, AtomArray]:
        # PhysPPI modification (Jinyu Lin, ShanghaiTech):
        #   Prefer precomputed monomer AtomArrays from _{chain}_monomer.pt.
        #   Each .pt already has the fully-processed structure (parsed,
        #   canonized, no-crop processed, token-positioned).  Falls back
        #   to PDB parsing for chains whose .pt is missing.
        cache_dir = _precomputed_dir(self, sample)
        monomer_inputs: dict[str, AtomArray] = {}
        for chain_id in sorted(str(chain) for chain in np.unique(atom_array.chain_id)):
            if cache_dir is not None:
                pt_path = cache_dir / f"{sample.sample_id}_{chain_id}_monomer.pt"
                try:
                    if pt_path.is_file():
                        monomer_inputs[chain_id] = _load_atomarray_from_pt(pt_path)
                        continue
                except Exception as exc:
                    if _using_precomputed_root(self):
                        raise PhysPPIDataError(
                            "malformed_precomputed_monomer_pt",
                            f"failed to load required precomputed monomer: {pt_path}",
                        ) from exc
                    pass  # fall through to PDB path
                if _using_precomputed_root(self):
                    raise PhysPPIDataError(
                        "missing_precomputed_monomer_pt",
                        f"required precomputed monomer is missing: {pt_path}",
                    )
            path = sample.paths.monomer_pdbs.get(chain_id)
            if path is None:
                raise PhysPPIDataError(
                    "missing_mandatory_monomer_geometry",
                    f"retained chain {chain_id!r} has no monomer PDB",
                )
            monomer_inputs[chain_id] = self._process_monomer_template(path)
        return monomer_inputs

    def create_canonical_template_features(
        self,
        sample: PriorComplexSample,
        atom_array: AtomArray,
        monomer_inputs: dict[str, AtomArray] | None = None,
    ) -> dict:
        if monomer_inputs is None:
            monomer_inputs = self._load_mandatory_monomer_inputs(sample, atom_array)
        if self.template.n_templates == 0:
            collection = TemplateSliceCollection(template_slices={})
        else:
            template_slices: dict[str, list[TemplateSlice]] = {}
            for chain_id, template_atom_array in sorted(monomer_inputs.items()):
                query_chain = atom_array[atom_array.chain_id == chain_id]
                if len(query_chain) == 0:
                    continue
                query_token_atoms = query_chain[_token_starts(query_chain)]
                template_slice = _crop_monomer_template_to_query_tokens(
                    template_atom_array, query_token_atoms
                )
                if template_slice is not None:
                    template_slices[chain_id] = [template_slice]
            collection = TemplateSliceCollection(template_slices=template_slices)
        return featurize_template_structures_of3(
            atom_array=atom_array,
            template_slice_collection=collection,
            n_templates=self.template.n_templates,
            n_tokens=self.n_tokens,
            min_bin=self.template.distogram.min_bin,
            max_bin=self.template.distogram.max_bin,
            n_bins=self.template.distogram.n_bins,
        )

    def create_loss_features(self, pdb_id: str) -> dict:
        return {
            "loss_weights": set_loss_weights(self.loss, resolution=None),
            "physppi_loss_weight": _torch.tensor(
                float(self.physppi.loss_weight),
                dtype=_torch.float32,
            ),
        }

    def build_prior_features(
        self,
        sample: PriorComplexSample,
        atom_array: AtomArray,
        token_index: dict[tuple[str, int], int],
        states: dict[OptionalSlot, SlotState],
        rng: np.random.Generator,
        n_tokens: int | None = None,
        *,
        input_atom_array: AtomArray | None = None,
    ) -> dict[str, torch.Tensor]:
        if n_tokens is None:
            n_tokens = len(token_index)
        # PhysPPI residue priors (Jinyu Lin, ShanghaiTech):
        #   Prefer pre-computed .pt files for interface/interaction/membrane.
        #   Each prior independently falls back to the original builder.
        interface_cutoff = self._select_interface_cutoff(rng)
        pt_interface, pt_interaction, pt_tm, pt_membrane = _build_residue_priors(
            self, sample, token_index, interface_cutoff, atom_array,
        )
        if pt_interface is not None:
            interface = pt_interface
        else:
            interface = build_interface_residue_features(
                sample.dmap, token_index, cutoff=interface_cutoff,
            )
        if pt_interaction is not None:
            interaction = pt_interaction
        else:
            interaction = build_interaction_features(
                sample.interactions, token_index, atom_array=atom_array,
            )
        chains = set(str(chain) for chain in np.asarray(atom_array.chain_id))
        transforms = {
            chain: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            for chain in chains
        }
        # PhysPPI mesh prior (Jinyu Lin, ShanghaiTech):
        #   Prefer pre-computed KDTree-based mesh-to-residue tensors.
        #   Fall back to KNN-based build_surface_features when .pt is missing.
        surface = _build_surface_mesh_prior(
            self, sample, token_index, transforms, input_atom_array
        )
        if pt_tm is not None:
            tm_mask = pt_tm
        else:
            res_loc = sample.opm.get("res_loc", {}) if isinstance(sample.opm, dict) else {}
            tm_mask = build_tm_features(res_loc, token_index)
        # PhysPPI modification (Jinyu Lin, ShanghaiTech):
        #   Precomputed mesh adapter returns physppi_surface_local_coords /
        #   physppi_surface_features / ...; the legacy KNN fallback returns
        #   coords / features / ...  Accept either convention.
        if "physppi_surface_local_coords" in surface:
            surf_coords = surface["physppi_surface_local_coords"]
            surf_feats = surface["physppi_surface_features"]
            surf_mask = surface["physppi_surface_mask"]
            surf_charge = surface["physppi_surface_charge_mask"]
        else:
            surf_coords = surface["coords"]
            surf_feats = surface["features"]
            surf_mask = surface["mask"]
            surf_charge = surface["charge_mask"]

        padded_payloads = _pad_prior_payloads_to_token_count(
            {
                "interface_residue": np.asarray(
                    interface["interface_mask"], dtype=bool
                ),
                "interaction": np.asarray(interaction["type_mask"], dtype=bool),
                "surface_coords": np.asarray(surf_coords, dtype=np.float32),
                "surface_features": np.asarray(surf_feats, dtype=np.float32),
                "surface_mask": np.asarray(surf_mask, dtype=bool),
                "surface_charge_mask": np.asarray(surf_charge, dtype=bool),
                "tm": np.asarray(tm_mask, dtype=bool),
            },
            n_tokens,
        )
        membrane = (
            pt_membrane
            if pt_membrane is not None
            else build_membrane_target(sample.opm, bool(tm_mask.any()))
        )
        states[OptionalSlot.INTERFACE_RESIDUE] = replace(
            states[OptionalSlot.INTERFACE_RESIDUE],
            valid=bool(interface["valid"]),
        )
        states[OptionalSlot.INTERACTION] = replace(
            states[OptionalSlot.INTERACTION], valid=bool(interaction["valid"])
        )
        states[OptionalSlot.SURFACE] = replace(
            states[OptionalSlot.SURFACE], valid=bool(surface["valid"])
        )
        states[OptionalSlot.TM] = replace(
            states[OptionalSlot.TM], valid=bool(tm_mask.any())
        )
        _resolve_post_build_prior_states(states)
        result = _slot_state_features(states)
        result.update(
            {
                "physppi_interface_residue_mask": torch.from_numpy(
                    padded_payloads["interface_residue"]
                ),
                "physppi_interaction_type_mask": torch.from_numpy(
                    padded_payloads["interaction"]
                ),
                "physppi_surface_local_coords": torch.from_numpy(
                    padded_payloads["surface_coords"]
                ),
                "physppi_surface_features": torch.from_numpy(
                    padded_payloads["surface_features"]
                ),
                "physppi_surface_mask": torch.from_numpy(
                    padded_payloads["surface_mask"]
                ),
                "physppi_surface_charge_mask": torch.from_numpy(
                    padded_payloads["surface_charge_mask"]
                ),
                "physppi_tm_token_mask": torch.from_numpy(padded_payloads["tm"]),
                "physppi_membrane_gt_normal": torch.from_numpy(
                    np.asarray(membrane["normal"], dtype=np.float32)
                ),
                "physppi_membrane_gt_point": torch.from_numpy(
                    np.asarray(membrane["point"], dtype=np.float32)
                ),
                "physppi_membrane_gt_valid": torch.tensor(membrane["valid"]),
            }
        )
        aux_edges = list(interaction["aux"])
        (
            result["physppi_interaction_distance"],
            result["physppi_interaction_aux_valid"],
        ) = _build_pair_interaction_auxiliary(aux_edges, n_tokens=n_tokens)
        return result

    def create_canonical_all_features(
        self,
        sample: PriorComplexSample,
        preferred_chain_or_interface: str | list[str, str] | None,
        return_atom_arrays: bool,
        return_crop_strategy: bool,
        crop_attempt_seed: int | None = None,
    ) -> dict:
        sample_data = {"features": {}}
        structure_data = self.create_canonical_structure_features(
            sample,
            preferred_chain_or_interface,
            return_atom_arrays,
            return_crop_strategy,
            crop_attempt_seed,
        )
        sample_data["features"].update(structure_data["target_structure_features"])
        sample_data["features"].update(structure_data["reference_conformer_features"])
        atom_array = structure_data["atom_array_cropped"]
        sample_data["features"].update(
            self.create_canonical_msa_features(sample, atom_array)
        )
        monomer_inputs = self._load_mandatory_monomer_inputs(sample, atom_array)
        sample_data["features"].update(
            self.create_canonical_template_features(
                sample, atom_array, monomer_inputs=monomer_inputs
            )
        )
        sample_data["monomer_inputs"] = monomer_inputs
        sample_data["features"].update(self.create_loss_features(sample.sample_id))
        if return_atom_arrays:
            sample_data["atom_array"] = structure_data["atom_array"]
            sample_data["atom_array_gt"] = structure_data["atom_array_gt"]
            sample_data["atom_array_cropped"] = structure_data["atom_array_cropped"]
        if return_crop_strategy:
            sample_data["crop_strategy"] = structure_data["crop_strategy"]
        return sample_data

    def finish_physppi_features(
        self, sample: PriorComplexSample, sample_data: dict
    ) -> dict[str, torch.Tensor]:
        features = dict(sample_data["features"])
        atom_array = sample_data["atom_array_cropped"]
        monomer_geometry = build_monomer_input_geometry(
            atom_array, sample_data["monomer_inputs"]
        )
        input_atom_array = atom_array.copy()
        input_coordinates = monomer_geometry["physppi_monomer_atom_positions"].copy()
        input_coordinates[~monomer_geometry["physppi_monomer_atom_mask"]] = np.nan
        input_atom_array.coord = input_coordinates
        token_index = _token_index(atom_array, features)
        occurrence = self.occurrence
        occurrence_rng = build_occurrence_rng(
            self._physppi_data_seed,
            occurrence.epoch if occurrence is not None else 0,
            occurrence.global_slot if occurrence is not None else 0,
            self.name,
            sample.sample_id,
        )
        selection = self._select_prior_level(
            self._available_slots(sample), occurrence_rng.slots
        )
        priors = self.build_prior_features(
            sample,
            atom_array,
            token_index,
            selection.states,
            occurrence_rng.interface_residue,
            int(features["token_mask"].shape[0]),
            input_atom_array=input_atom_array,
        )
        priors.update(
            {
                "physppi_prior_requested_level": torch.tensor(
                    selection.requested_level, dtype=torch.int64
                ),
                "physppi_prior_effective_level": torch.tensor(
                    sum(state.active for state in selection.states.values()),
                    dtype=torch.int64,
                ),
                "physppi_occurrence_seed": torch.tensor(
                    occurrence_rng.seed, dtype=torch.uint64
                ),
            }
        )
        priors.update(
            _geometry_features(
                atom_array,
                features["token_mask"],
                monomer_geometry["physppi_monomer_atom_positions"],
                monomer_geometry["physppi_monomer_atom_mask"],
            )
        )
        priors.update(
            {key: torch.from_numpy(value) for key, value in monomer_geometry.items()}
        )
        features.update(priors)
        return features

    def build_sample_features(self, sample: PriorComplexSample, datapoint) -> dict:
        occurrence = self.occurrence
        physppi = getattr(self, "physppi", None)
        occurrence_rng = build_occurrence_rng(
            self._physppi_data_seed,
            occurrence.epoch if occurrence is not None else 0,
            occurrence.global_slot if occurrence is not None else 0,
            self.name,
            sample.sample_id,
        )
        view = select_retained_chain_view(
            sample,
            occurrence_rng.chain_view,
            occurrence_rng.seed,
            candidate_cap=int(
                getattr(physppi, "retained_view_candidate_cap", 1024)
            ),
        )
        restricted_sample = restrict_prior_sample_to_chains(
            sample, view.source_chain_ids
        )
        retry_attempts = int(
            getattr(physppi, "geometry_crop_retry_attempts", 16)
        )
        residue_floor = getattr(
            physppi,
            "min_residues_per_retained_protein_chain",
            None,
        )
        last_crop_error: PhysPPIDataError | None = None
        last_crop_details = ""
        for attempt in range(retry_attempts):
            crop_attempt_seed = _physppi_crop_attempt_seed(
                occurrence_rng.seed,
                attempt,
            )
            sample_data = None
            try:
                sample_data = self.create_canonical_all_features(
                    sample=restricted_sample,
                    preferred_chain_or_interface=datapoint[
                        "preferred_chain_or_interface"
                    ],
                    return_atom_arrays=True,
                    return_crop_strategy=False,
                    crop_attempt_seed=crop_attempt_seed,
                )
                if residue_floor is not None:
                    _validate_cropped_protein_residue_floor(
                        sample_data["atom_array_cropped"],
                        int(residue_floor),
                        tuple(
                            sorted(
                                str(chain_id)
                                for chain_id in getattr(
                                    restricted_sample,
                                    "chain_sequences",
                                    {},
                                )
                            )
                        )
                        or None,
                    )
                features = self.finish_physppi_features(
                    restricted_sample,
                    sample_data,
                )
                break
            except PhysPPIDataError as exc:
                if not _retryable_physppi_crop_error(exc):
                    raise
                last_crop_error = exc
                last_crop_details = _physppi_crop_diagnostics(
                    restricted_sample,
                    sample_data,
                )
        else:
            detail = (
                f"sample {sample.sample_id!r} retained chains "
                f"{view.source_chain_ids!r} failed {retry_attempts} crop attempts"
            )
            if last_crop_details:
                detail = f"{detail}; last_attempt={last_crop_details}"
            if last_crop_error is not None:
                detail = f"{detail}; last_error={last_crop_error}"
            raise PhysPPIDataError(
                "physppi_crop_retry_exhausted",
                detail,
            ) from last_crop_error
        features.update(encode_retained_chain_view(view))
        if self.debug_mode:
            check_invalid_feature_dict(features)
        features["pdb_id"] = sample.sample_id
        source_gt_pdb = _source_training_gt_pdb_path(sample)
        if source_gt_pdb is not None:
            features["physppi_gt_pdb_path"] = str(source_gt_pdb)
        features["preferred_chain_or_interface"] = "None"
        atom_array_cropped = sample_data["atom_array_cropped"]
        atom_count = len(atom_array_cropped)
        feature_atom_count = int(features["atom_mask"].shape[-1])
        if atom_count != feature_atom_count:
            raise ValueError(
                "cropped AtomArray atom order contract failed for "
                f"{sample.sample_id}: atom_array={atom_count}, "
                f"atom_mask={feature_atom_count}"
            )
        features["atom_array"] = atom_array_cropped
        return features
