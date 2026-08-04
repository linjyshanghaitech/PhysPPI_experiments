"""Isolated one-GPU finite-gradient smoke test for canonical PhysPPI priors.

This script deliberately lives outside the normal training entry point. It reads
the normal runner YAML only for dataset/model dimensions, filters the canonical
Q-BioLiP and AFDB readers with an audit JSON, and runs every remaining sample
through the active PhysPPI optional-prior conditioner. Each sample gets one
optimizer step and is checked for finite features, loss, gradients, and
parameters. No checkpoint, trainer, YAML, or persistent training state is
modified.

The default ``prior_only`` mode exercises canonical prior parsing and the real
PhysPPI conditioner without constructing the heavyweight OF3 structure features.
The optional ``of3_features`` mode additionally runs the full dataset adapter and
input embedder for a small representative probe; it is not intended for every
prior because that path is substantially more expensive.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pickle
import random
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
import pandas as pd
import torch

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.core.data.framework.single_datasets.afdb10k import AFDB10kDataset
from openfold3.core.data.framework.single_datasets.qbiolip10k import QBioLiP10kDataset
from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    MsaArrayCollection,
    calculate_profile,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.model.feature_embedders.input_embedders import (
    InputEmbedderAllAtom,
)
from openfold3.entry_points.validator import TrainingExperimentConfig
from openfold3.projects.of3_all_atom.config.dataset_configs import TrainingDatasetSpec
from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    PRIOR_OPTIONAL_SLOTS,
    SlotState,
    build_paired_msa_chain_mask,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    parse_prior_payload,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    QBioLiP10kReader,
)
from openfold3.projects.of3_all_atom.physppi.features import (
    build_interaction_features,
    build_interface_residue_features,
    build_membrane_target,
    build_surface_features,
    build_tm_features,
)
from openfold3.projects.of3_all_atom.physppi.model.conditioning import (
    OptionalPriorConditioner,
)

LOGGER = logging.getLogger("physppi_all_prior_finite_smoke")


def _sample_seed(sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _empty_audit_state() -> dict[str, Any]:
    return {
        "fatal_ids": {"qbiolip": set(), "afdb": set()},
        "data_errors": {"qbiolip": [], "afdb": []},
        "allowed_states": {
            "qbiolip": {"opm_none": [], "mesh_charge_nan": []},
            "afdb": {"opm_none": [], "mesh_charge_nan": []},
        },
    }


def _load_audit(paths: list[Path]) -> dict[str, Any]:
    audit = _empty_audit_state()
    for path in paths:
        with path.open() as handle:
            report = json.load(handle)
        if report.get("schema_version") != 1:
            raise ValueError(f"unsupported audit schema in {path}")
        for dataset_name, dataset_report in report.get("datasets", {}).items():
            if dataset_name not in audit["fatal_ids"]:
                continue
            fatal = dataset_report.get("fatal", {})
            fatal_ids = [str(sample_id) for sample_id in fatal.get("ids", [])]
            audit["fatal_ids"][dataset_name].update(fatal_ids)
            reasons = fatal.get("reasons", {})
            for sample_id in fatal_ids:
                audit["data_errors"][dataset_name].append(
                    {
                        "sample_id": sample_id,
                        "reasons": [
                            str(reason) for reason in reasons.get(sample_id, [])
                        ],
                    }
                )
            allowed = dataset_report.get("allowed", {})
            for state_name in audit["allowed_states"][dataset_name]:
                audit["allowed_states"][dataset_name][state_name].extend(
                    str(sample_id) for sample_id in allowed.get(state_name, [])
                )
    for dataset_name in audit["data_errors"]:
        audit["data_errors"][dataset_name].sort(key=lambda item: item["sample_id"])
        for state_ids in audit["allowed_states"][dataset_name].values():
            state_ids[:] = sorted(set(state_ids))
    return audit


def _finite_tree(value: Any, path: str = "") -> tuple[bool, str]:
    if torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            return False, path or "<tensor>"
        return True, ""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fciu" and not np.isfinite(value).all():
            return False, path or "<numpy>"
        return True, ""
    if isinstance(value, Mapping):
        for key, child in value.items():
            ok, bad = _finite_tree(child, f"{path}.{key}" if path else str(key))
            if not ok:
                return ok, bad
        return True, ""
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            ok, bad = _finite_tree(child, f"{path}[{index}]")
            if not ok:
                return ok, bad
    return True, ""


def _move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_move_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(child, device) for child in value)
    return value


def _write_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def _resize_msa(msa: MsaArray, target_width: int) -> MsaArray:
    """Make a query/paired array compatible with its representative MSA width."""

    current_width = msa.msa.shape[1]
    if current_width == target_width:
        return msa
    if current_width > target_width:
        return MsaArray(
            msa=msa.msa[:, :target_width],
            deletion_matrix=msa.deletion_matrix[:, :target_width],
            metadata=pd.DataFrame(),
        )
    pad_width = target_width - current_width
    pad_value = "-" if msa.msa.dtype.kind in "OUS" else 0
    return MsaArray(
        msa=np.pad(msa.msa, ((0, 0), (0, pad_width)), constant_values=pad_value),
        deletion_matrix=np.pad(
            msa.deletion_matrix, ((0, 0), (0, pad_width)), constant_values=0
        ),
        metadata=pd.DataFrame(),
    )


def _safe_create_canonical_msa_features(self, sample, atom_array):
    """Smoke-only MSA boundary that preserves canonical full-chain widths.

    The training crop contains only a residue subset, while the canonical MSA
    files are aligned to the full chain. The production adapter currently builds
    a cropped query row before stacking it with the full-width MSA, which raises
    before the model sees the sample. For this isolated finite check, use the
    representative main-MSA row as the query-width carrier and let OF3's token
    mapper select the cropped residue columns.
    """

    from openfold3.projects.of3_all_atom.physppi.data.adapters import (
        _runtime as structure_adapter,
    )

    chain_ids = [str(chain) for chain in np.unique(atom_array.chain_id)]
    chain_to_rep = {chain_id: chain_id for chain_id in chain_ids}
    rep_to_chain = dict(chain_to_rep)
    chain_to_mol = {chain_id: MoleculeType.PROTEIN for chain_id in chain_ids}
    rep_to_mol = dict(chain_to_mol)
    query_msas = {}
    main_by_key = {
        key: structure_adapter._read_msa_npz(path)
        for key, path in sample.paths.main_msa_npzs.items()
    }
    for chain_id in chain_ids:
        main = main_by_key.get(chain_id)
        if (
            main is None
            and len(main_by_key) == 1
            and str(getattr(sample, "dataset_name", "")).lower().replace("-", "")
            in {"afdb", "afdb10k"}
        ):
            main = next(iter(main_by_key.values()))
        if main is None:
            query_msas[chain_id] = structure_adapter._query_only_msa(
                structure_adapter._sequence_for_chain(atom_array, chain_id)
            )
        else:
            query_msas[chain_id] = MsaArray(
                msa=main.msa[:1].copy(),
                deletion_matrix=main.deletion_matrix[:1].copy(),
                metadata=pd.DataFrame(),
            )
    main_msas, _ = structure_adapter._build_main_msas(sample, chain_ids, query_msas)

    paired_by_path = {
        key: structure_adapter._read_msa_npz(path)
        for key, path in sample.paths.paired_msa_npzs.items()
    }
    paired_msas = {}
    for chain_id in chain_ids:
        paired = paired_by_path.get(chain_id) or paired_by_path.get(sample.sample_id)
        if paired is not None:
            paired_msas[chain_id] = _resize_msa(
                paired, main_msas[chain_id].msa.shape[1]
            )
    max_paired_rows = max((len(msa) for msa in paired_msas.values()), default=0)
    if max_paired_rows:
        for chain_id in chain_ids:
            paired_msas.setdefault(
                chain_id,
                structure_adapter._empty_msa_like(query_msas[chain_id]),
            )
    paired_chain_mask = build_paired_msa_chain_mask(
        [
            chain_id in paired_by_path or sample.sample_id in paired_by_path
            for chain_id in chain_ids
        ]
    )
    collection = MsaArrayCollection(
        chain_id_to_rep_id=chain_to_rep,
        chain_id_to_mol_type=chain_to_mol,
        rep_id_to_chain_id=rep_to_chain,
        rep_id_to_mol_type=rep_to_mol,
    )
    profiles = {
        chain_id: calculate_profile(
            msa_array=main_msas[chain_id].msa,
            molecule_type=MoleculeType.PROTEIN,
            chunk_size=1000,
        )
        for chain_id in chain_ids
    }
    deletion_mean = {
        chain_id: np.mean(main_msas[chain_id].deletion_matrix, axis=0)
        for chain_id in chain_ids
    }
    max_main_rows = max(len(msa) for msa in main_msas.values())
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


def _build_prior_readers(
    runner_yaml: Path,
    audit_ids: dict[str, set[str]],
    max_samples: int | None,
):
    """Index canonical readers without constructing the heavyweight OF3 dataset."""

    runner_dict = config_utils.load_yaml(runner_yaml)
    specs = {
        "qbiolip": (
            Path(
                runner_dict["dataset_configs"]["train"]["qbiolip10k"]["config"][
                    "physppi"
                ]["root"]
            ),
            QBioLiP10kReader,
        ),
        "afdb": (
            Path(
                runner_dict["dataset_configs"]["train"]["afdb10k"]["config"]["physppi"][
                    "root"
                ]
            ),
            AFDB10kReader,
        ),
    }
    readers = []
    for name, (root, reader_class) in specs.items():
        # A bounded probe only needs enough sorted entries to survive audit
        # exclusions. Full validation deliberately leaves this unset so every
        # canonical prior is indexed and checked.
        reader_limit = (
            None if max_samples is None else max_samples + len(audit_ids[name])
        )
        reader = reader_class(root, max_samples=reader_limit)
        sample_ids = [
            sample_id
            for sample_id in reader.sample_ids()
            if sample_id not in audit_ids[name]
        ]
        if max_samples is not None:
            sample_ids = sample_ids[:max_samples]
        readers.append((name, reader, sample_ids))
    return readers


def _build_prior_only_probe(device: torch.device):
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    cfg = OF3ProjectEntry().get_model_config_with_presets()
    conditioner = (
        OptionalPriorConditioner(
            c_s=int(cfg.architecture.shared.c_s),
            c_z=int(cfg.architecture.shared.c_z),
            num_interaction_types=int(
                cfg.architecture.physppi.optional_prior_conditioner.num_interaction_types
            ),
        )
        .to(device=device)
        .train()
    )
    optimizer = torch.optim.AdamW(conditioner.parameters(), lr=1.0e-4)
    return (
        conditioner,
        optimizer,
        int(cfg.architecture.shared.c_s),
        int(cfg.architecture.shared.c_z),
    )


def _read_smoke_sample(reader: Any, sample_id: str):
    """Read a canonical sample, allowing absent membrane labels as empty."""

    try:
        return reader.read(sample_id)
    except ValueError as exc:
        if "opm must be a mapping" not in str(exc):
            raise
    paths = reader.locate(sample_id)
    with paths.prior_pkl.open("rb") as handle:
        raw = pickle.load(handle)
    payload = raw[sample_id]
    if isinstance(payload, Mapping):
        payload = dict(payload)
        payload["opm"] = {}
    else:
        payload = copy.copy(payload)
        payload.opm = {}
    raw[sample_id] = payload
    dataset_name = (
        "AFDB" if reader.__class__.__name__.startswith("AFDB") else "Q-BioLiP"
    )
    chain_sequences = getattr(reader, "_chain_sequences", {}).get(sample_id)
    return parse_prior_payload(
        raw, sample_id, paths, dataset_name, chain_sequences=chain_sequences
    )


def _build_prior_only_batch(
    *,
    sample,
    sample_id: str,
    c_s: int,
    c_z: int,
    device: torch.device,
    max_tokens: int,
    mode: str,
) -> tuple[dict[str, torch.Tensor], int, bool]:
    """Build the real canonical-prior batch used by isolated train probes."""

    from openfold3.projects.of3_all_atom.physppi.data.adapters import (
        _runtime as structure_adapter,
    )

    atom_array = structure_adapter._parse_pdb_atom_array(sample.paths.gt_pdb)
    if "entity_id" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("entity_id", np.ones(len(atom_array), dtype=np.int32))
    if "token_id" not in atom_array.get_annotation_categories():
        residue_starts = np.asarray(struc.get_residue_starts(atom_array))
        token_ids = (
            np.searchsorted(residue_starts, np.arange(len(atom_array)), side="right")
            - 1
        )
        atom_array.set_annotation("token_id", token_ids.astype(np.int64))
    starts = structure_adapter._token_starts(atom_array)
    token_keys = [
        (str(atom_array.chain_id[start]), int(atom_array.res_id[start]))
        for start in starts
    ][:max_tokens]
    token_index = {key: index for index, key in enumerate(token_keys)}
    n_tokens = len(token_index)
    if n_tokens == 0:
        raise ValueError("no residue tokens after canonical PDB parsing")
    rng = np.random.default_rng(_sample_seed(sample_id))
    interface = build_interface_residue_features(
        sample.dmap, token_index, cutoff=float(rng.uniform(8.0, 18.0))
    )
    interaction = build_interaction_features(
        sample.interactions, token_index, atom_array=atom_array
    )
    chains = set(str(chain) for chain in np.asarray(atom_array.chain_id))
    transforms = {
        chain: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        for chain in chains
    }
    surface = build_surface_features(
        sample.mesh,
        token_index,
        transforms,
        128,
        atom_array=atom_array,
    )
    res_loc = sample.opm.get("res_loc", {}) if isinstance(sample.opm, dict) else {}
    tm_mask = build_tm_features(res_loc, token_index)
    membrane = build_membrane_target(sample.opm, bool(tm_mask.any()))
    available = {
        PRIOR_OPTIONAL_SLOTS[0]: bool(sample.dmap),
        PRIOR_OPTIONAL_SLOTS[1]: bool(sample.interactions),
        PRIOR_OPTIONAL_SLOTS[2]: bool(sample.mesh),
        PRIOR_OPTIONAL_SLOTS[3]: bool(res_loc),
    }
    if mode not in {"mandatory_only", "all_available"}:
        raise ValueError(f"unknown prior-only mode: {mode}")
    states = {
        slot: SlotState(
            available=is_available,
            selected=is_available and mode == "all_available",
            valid=is_available,
            provenance=(f"finite_smoke:{mode}" if is_available else "unavailable"),
        )
        for slot, is_available in available.items()
    }
    batch = {
        "token_mask": torch.ones(1, n_tokens, dtype=torch.bool, device=device),
        "physppi_interface_residue_mask": torch.from_numpy(
            interface["interface_mask"]
        )[None].to(device),
        "physppi_interface_residue_active": torch.tensor(
            [states[PRIOR_OPTIONAL_SLOTS[0]].active], device=device
        ),
        "physppi_interaction_type_mask": torch.from_numpy(interaction["type_mask"])[
            None
        ].to(device),
        "physppi_interaction_active": torch.tensor(
            [states[PRIOR_OPTIONAL_SLOTS[1]].active], device=device
        ),
        "physppi_surface_local_coords": torch.from_numpy(surface["coords"])[None].to(
            device
        ),
        "physppi_surface_features": torch.from_numpy(surface["features"])[None].to(
            device
        ),
        "physppi_surface_mask": torch.from_numpy(surface["mask"])[None].to(device),
        "physppi_surface_charge_mask": torch.from_numpy(surface["charge_mask"])[
            None
        ].to(device),
        "physppi_surface_active": torch.tensor(
            [states[PRIOR_OPTIONAL_SLOTS[2]].active], device=device
        ),
        "physppi_tm_token_mask": torch.from_numpy(tm_mask)[None].to(device),
        "physppi_tm_active": torch.tensor(
            [states[PRIOR_OPTIONAL_SLOTS[3]].active], device=device
        ),
    }
    ok, bad = _finite_tree(batch)
    if not ok:
        raise FloatingPointError(f"non-finite canonical prior feature: {bad}")
    return batch, n_tokens, bool(membrane["valid"])


def _run_prior_only_one(
    *,
    sample,
    sample_id: str,
    conditioner: OptionalPriorConditioner,
    optimizer: torch.optim.Optimizer,
    c_s: int,
    c_z: int,
    device: torch.device,
    max_tokens: int,
    mode: str = "all_available",
) -> dict[str, Any]:
    batch, n_tokens, membrane_valid = _build_prior_only_batch(
        sample=sample,
        sample_id=sample_id,
        c_s=c_s,
        c_z=c_z,
        device=device,
        max_tokens=max_tokens,
        mode=mode,
    )
    # Use a tiny deterministic non-zero seed so the smoke check still creates
    # a meaningful finite objective when augmentation samples no optional slot.
    torch_generator = torch.Generator(device=device).manual_seed(
        _sample_seed(sample_id) % (2**63 - 1)
    )
    s_init = (
        torch.randn((1, n_tokens, c_s), device=device, generator=torch_generator)
        * 1.0e-2
    )
    z_init = (
        torch.randn(
            (1, n_tokens, n_tokens, c_z), device=device, generator=torch_generator
        )
        * 1.0e-2
    )
    token_mask = batch["token_mask"].to(dtype=s_init.dtype)
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    optimizer.zero_grad(set_to_none=True)
    conditioned = conditioner(
        s_init=s_init,
        z_init=z_init,
        batch=batch,
        token_mask=token_mask,
        pair_mask=pair_mask,
    )
    loss = conditioned.s_init.square().mean() + conditioned.z_init.square().mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("non-finite prior-only smoke loss")
    loss.backward()
    max_grad = 0.0
    for name, parameter in conditioner.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"non-finite gradient: {name}")
        max_grad = max(max_grad, float(parameter.grad.detach().abs().max().cpu()))
    optimizer.step()
    for name, parameter in conditioner.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"non-finite parameter after step: {name}")
    return {
        "sample_id": sample_id,
        "mode": mode,
        "loss": float(loss.detach().cpu()),
        "max_grad": max_grad,
        "n_tokens": n_tokens,
        "membrane_valid": membrane_valid,
    }


def _run_prior_only(args, audit: dict[str, Any], device: torch.device) -> int:
    audit_ids = audit["fatal_ids"]
    readers = _build_prior_readers(args.runner_yaml, audit_ids, args.max_samples)
    conditioner, optimizer, c_s, c_z = _build_prior_only_probe(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "records.jsonl"
    summary_path = args.out_dir / "summary.json"
    summary: dict[str, Any] = {
        "mode": "prior_only",
        "device": str(device),
        "runner_yaml": str(args.runner_yaml),
        "audit_json": [str(path) for path in args.audit_json],
        "max_tokens": args.max_tokens,
        "excluded": {key: len(value) for key, value in audit_ids.items()},
        "data_errors": audit["data_errors"],
        "finite_failures": [],
        "allowed_states": audit["allowed_states"],
        "checked": {"qbiolip": [], "afdb": []},
        "datasets": {},
        "passed": 0,
        "failed": 0,
    }
    with records_path.open("w") as handle:
        for dataset_name, reader, sample_ids in readers:
            dataset_summary = {
                "total_after_filter": len(sample_ids),
                "passed": 0,
                "failed": 0,
            }
            summary["datasets"][dataset_name] = dataset_summary
            LOGGER.info(
                "%s: %d samples after audit filtering", dataset_name, len(sample_ids)
            )
            for index, sample_id in enumerate(sample_ids):
                try:
                    record = _run_prior_only_one(
                        sample=_read_smoke_sample(reader, sample_id),
                        sample_id=sample_id,
                        conditioner=conditioner,
                        optimizer=optimizer,
                        c_s=c_s,
                        c_z=c_z,
                        device=device,
                        max_tokens=args.max_tokens,
                        mode="all_available",
                    )
                    record.update({"dataset": dataset_name, "status": "passed"})
                    dataset_summary["passed"] += 1
                    summary["passed"] += 1
                except Exception as exc:
                    record = {
                        "dataset": dataset_name,
                        "sample_id": sample_id,
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                    dataset_summary["failed"] += 1
                    summary["failed"] += 1
                    summary["finite_failures"].append(
                        {
                            "dataset": dataset_name,
                            "sample_id": sample_id,
                            "error": repr(exc),
                        }
                    )
                    LOGGER.error("%s/%s failed: %s", dataset_name, sample_id, exc)
                    if args.fail_fast:
                        _write_record(handle, record)
                        raise
                summary["checked"][dataset_name].append(sample_id)
                _write_record(handle, record)
                if index % 50 == 0:
                    LOGGER.info(
                        "%s: %d/%d (passed=%d failed=%d)",
                        dataset_name,
                        index + 1,
                        len(sample_ids),
                        dataset_summary["passed"],
                        dataset_summary["failed"],
                    )
                if device.type == "cuda" and index % 25 == 0:
                    torch.cuda.empty_cache()
    summary["ok"] = (
        not any(summary["data_errors"].values()) and not summary["finite_failures"]
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    LOGGER.info("summary=%s", summary_path)
    return 0 if summary["ok"] else 1


def _build_datasets(
    runner_yaml: Path,
    audit_ids: dict[str, set[str]],
    max_samples: int | None,
    cache_path: Path,
) -> list[tuple[str, Any]]:
    runner_dict = config_utils.load_yaml(runner_yaml)
    repo_root = Path(__file__).resolve().parents[2]
    resource_root = repo_root / "openfold3/core/data/resources/canonical_protein_data"
    reference_mols = resource_root / "reference_mols"
    for dataset_path in runner_dict["dataset_paths"].values():
        # Canonical PhysPPI readers own GT/MSA/template paths. The OF3 base
        # dataset still validates these shared fields, so point them at the
        # repository's protein resources for this isolated probe.
        dataset_path["dataset_cache_file"] = str(cache_path)
        dataset_path["alignment_array_directory"] = str(resource_root)
        dataset_path["alignment_db_directory"] = None
        dataset_path["alignments_directory"] = None
        dataset_path["target_structures_directory"] = str(resource_root)
        dataset_path["target_structure_file_format"] = "pdb"
        dataset_path["reference_molecule_directory"] = str(reference_mols)
        dataset_path["template_cache_directory"] = None
        dataset_path["template_structures_directory"] = None
        dataset_path["template_structure_array_directory"] = None
        dataset_path["template_file_format"] = None
    experiment = TrainingExperimentConfig.model_validate(runner_dict)
    result = []
    dataset_classes = {
        "qbiolip10k": ("qbiolip", QBioLiP10kDataset),
        "afdb10k": ("afdb", AFDB10kDataset),
    }
    for name, (audit_name, dataset_class) in dataset_classes.items():
        raw_spec = copy.deepcopy(experiment.dataset_configs["train"][name])
        raw_spec["name"] = name
        raw_spec["mode"] = "train"
        raw_spec["config"]["dataset_paths"] = experiment.dataset_paths[name]
        cfg = TrainingDatasetSpec.model_validate(raw_spec).config
        cfg.template.n_templates = 0
        # Keep bounded debug runs cheap by limiting the canonical reader index.
        # The default full smoke run leaves this as None and indexes every prior.
        cfg.physppi.max_samples = max_samples
        dataset = dataset_class(cfg)
        # Start from canonical reader IDs, then remove the full audit union.
        valid_ids = [
            sample_id
            for sample_id in dataset.reader.sample_ids()
            if sample_id not in audit_ids[audit_name]
        ]
        if max_samples is not None:
            valid_ids = valid_ids[:max_samples]
        dataset.datapoint_cache = dataset.datapoint_cache[
            dataset.datapoint_cache["pdb_id"].isin(valid_ids)
        ].reset_index(drop=True)
        result.append((audit_name, dataset))
    return result


def _build_probe(runner_yaml: Path, device: torch.device, cache_path: Path):
    runner_dict = config_utils.load_yaml(runner_yaml)
    repo_root = Path(__file__).resolve().parents[2]
    resource_root = repo_root / "openfold3/core/data/resources/canonical_protein_data"
    for dataset_path in runner_dict["dataset_paths"].values():
        dataset_path["dataset_cache_file"] = str(cache_path)
        dataset_path["alignment_array_directory"] = str(resource_root)
        dataset_path["alignment_db_directory"] = None
        dataset_path["alignments_directory"] = None
        dataset_path["target_structures_directory"] = str(resource_root)
        dataset_path["reference_molecule_directory"] = str(
            resource_root / "reference_mols"
        )
        dataset_path["template_cache_directory"] = None
        dataset_path["template_structures_directory"] = None
        dataset_path["template_structure_array_directory"] = None
        dataset_path["template_file_format"] = None
    experiment = TrainingExperimentConfig.model_validate(runner_dict)
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    cfg = OF3ProjectEntry().get_model_config_with_update(experiment.model_update)
    input_embedder = InputEmbedderAllAtom(**cfg.architecture.input_embedder)
    conditioner = OptionalPriorConditioner(
        c_s=int(cfg.architecture.shared.c_s),
        c_z=int(cfg.architecture.shared.c_z),
        num_interaction_types=int(
            cfg.architecture.physppi.optional_prior_conditioner.num_interaction_types
        ),
    )
    input_embedder.to(device=device).train()
    conditioner.to(device=device).train()
    optimizer = torch.optim.AdamW(
        list(input_embedder.parameters()) + list(conditioner.parameters()),
        lr=1.0e-4,
    )
    return input_embedder, conditioner, optimizer


def _run_one(
    *,
    sample: dict[str, Any],
    sample_id: str,
    input_embedder: InputEmbedderAllAtom,
    conditioner: OptionalPriorConditioner,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    ok, bad = _finite_tree(sample)
    if not ok:
        raise FloatingPointError(f"non-finite raw feature: {bad}")
    batch = openfold_batch_collator([sample])
    batch = _move_tree(batch, device)
    ok, bad = _finite_tree(batch)
    if not ok:
        raise FloatingPointError(f"non-finite collated feature: {bad}")

    optimizer.zero_grad(set_to_none=True)
    s_input, s_init, z_init = input_embedder(
        batch=batch,
        inplace_safe=False,
        use_high_precision_attention=True,
    )
    token_mask = batch["token_mask"].to(dtype=s_init.dtype)
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    conditioned = conditioner(
        s_init=s_init,
        z_init=z_init,
        batch=batch,
        token_mask=token_mask,
        pair_mask=pair_mask,
    )
    # A connected, scale-stable training probe over both active residual paths.
    loss = (
        conditioned.s_init.square().mean()
        + conditioned.z_init.square().mean()
        + 0.1 * s_input.square().mean()
    )
    if not torch.isfinite(loss).item():
        raise FloatingPointError("non-finite smoke loss")
    loss.backward()
    max_grad = 0.0
    for module in (input_embedder, conditioner):
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"non-finite gradient: {name}")
            max_grad = max(max_grad, float(parameter.grad.detach().abs().max().cpu()))
    optimizer.step()
    for module in (input_embedder, conditioner):
        for name, parameter in module.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError(f"non-finite parameter after step: {name}")
    return {
        "sample_id": sample_id,
        "loss": float(loss.detach().cpu()),
        "max_grad": max_grad,
        "n_tokens": int(batch["token_mask"].sum().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner-yaml",
        type=Path,
        default=Path("examples/training_yamls/physppi_training.yml"),
    )
    parser.add_argument("--audit-json", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("prior_only", "of3_features"),
        default="prior_only",
        help=(
            "prior_only checks all canonical priors; "
            "of3_features runs the full adapter/input embedder"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a visible CUDA device")
    device = torch.device(args.device)
    if device.index not in (0, None):
        raise ValueError("The isolated smoke test is intentionally pinned to cuda:0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_float32_matmul_precision("high")

    # The canonical PDBs do not carry the legacy entity annotation expected by
    # the OF3 permutation utility. Keep this compatibility shim local to the
    # smoke process; the production adapter and training configuration remain
    # untouched.
    from openfold3.projects.of3_all_atom.physppi.data.adapters import (
        _runtime as structure_adapter,
    )

    parse_pdb_atom_array = structure_adapter._parse_pdb_atom_array

    def _parse_smoke_pdb(path: Path):
        atom_array = parse_pdb_atom_array(path)
        if "entity_id" not in atom_array.get_annotation_categories():
            atom_array.set_annotation(
                "entity_id", np.ones(len(atom_array), dtype=np.int32)
            )
        return atom_array

    structure_adapter._parse_pdb_atom_array = _parse_smoke_pdb

    def _dense_smoke_token_index(atom_array, features):
        starts = structure_adapter._token_starts(atom_array)
        return {
            (str(atom_array.chain_id[start]), int(atom_array.res_id[start])): token
            for token, start in enumerate(starts)
        }

    structure_adapter._token_index = _dense_smoke_token_index
    adapter_class = structure_adapter.PhysPPIPriorComplexDatasetAdapter
    adapter_class.create_canonical_msa_features = _safe_create_canonical_msa_features

    audit = _load_audit(args.audit_json)
    audit_ids = audit["fatal_ids"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prior_only":
        return _run_prior_only(args, audit, device)

    cache_path = args.out_dir / "smoke_dataset_cache.json"
    if not cache_path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        reference_metadata = json.loads(
            (
                repo_root
                / "openfold3/core/data/resources/canonical_protein_data"
                / "reference_molecule_data.json"
            ).read_text()
        )
        cache_path.write_text(
            json.dumps(
                {
                    "_type": "ProteinMonomerDatasetCache",
                    "name": "physppi-smoke",
                    "structure_data": {},
                    "reference_molecule_data": reference_metadata,
                },
                indent=2,
            )
            + "\n"
        )
    datasets = _build_datasets(
        args.runner_yaml, audit_ids, args.max_samples, cache_path
    )
    records_path = args.out_dir / "records.jsonl"
    summary_path = args.out_dir / "summary.json"
    input_embedder, conditioner, optimizer = _build_probe(
        args.runner_yaml, device, cache_path
    )

    summary: dict[str, Any] = {
        "device": str(device),
        "runner_yaml": str(args.runner_yaml),
        "audit_json": [str(path) for path in args.audit_json],
        "excluded": {key: len(value) for key, value in audit_ids.items()},
        "data_errors": audit["data_errors"],
        "finite_failures": [],
        "allowed_states": audit["allowed_states"],
        "checked": {"qbiolip": [], "afdb": []},
        "datasets": {},
        "passed": 0,
        "failed": 0,
    }
    with records_path.open("w") as handle:
        for dataset_name, dataset in datasets:
            total = len(dataset)
            dataset_summary = {"total_after_filter": total, "passed": 0, "failed": 0}
            summary["datasets"][dataset_name] = dataset_summary
            LOGGER.info("%s: %d samples after audit filtering", dataset_name, total)
            for index in range(total):
                sample_id = str(dataset.datapoint_cache.iloc[index]["pdb_id"])
                try:
                    record = _run_one(
                        sample=dataset[index],
                        sample_id=sample_id,
                        input_embedder=input_embedder,
                        conditioner=conditioner,
                        optimizer=optimizer,
                        device=device,
                    )
                    record.update({"dataset": dataset_name, "status": "passed"})
                    dataset_summary["passed"] += 1
                    summary["passed"] += 1
                except Exception as exc:  # keep auditing all IDs by default
                    record = {
                        "dataset": dataset_name,
                        "sample_id": sample_id,
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                    dataset_summary["failed"] += 1
                    summary["failed"] += 1
                    summary["finite_failures"].append(
                        {
                            "dataset": dataset_name,
                            "sample_id": sample_id,
                            "error": repr(exc),
                        }
                    )
                    LOGGER.error("%s/%s failed: %s", dataset_name, sample_id, exc)
                    if args.fail_fast:
                        _write_record(handle, record)
                        raise
                summary["checked"][dataset_name].append(sample_id)
                _write_record(handle, record)
                if index % 50 == 0:
                    LOGGER.info(
                        "%s: %d/%d (passed=%d failed=%d)",
                        dataset_name,
                        index + 1,
                        total,
                        dataset_summary["passed"],
                        dataset_summary["failed"],
                    )
                if device.type == "cuda" and index % 25 == 0:
                    torch.cuda.empty_cache()

    summary["ok"] = (
        not any(summary["data_errors"].values()) and not summary["finite_failures"]
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    LOGGER.info("summary=%s", summary_path)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())
