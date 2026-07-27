"""Read-only audit harness for protected PhysPPI chain matching."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openfold3.core.data.framework.stochastic_sampler_dataset import SampleOccurrence
from openfold3.core.utils.permutation_alignment import (
    find_protected_mol_permutation,
    find_sequence_compatible_identity_mol_permutation,
)
from openfold3.core.utils.protected_chain_matching import ProtectedChainMatchError
from openfold3.projects.of3_all_atom.physppi.data.adapters import (
    restrict_prior_sample_to_chains,
    select_retained_chain_view,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.occurrence import (
    build_occurrence_rng,
)
from openfold3.projects.of3_all_atom.physppi.screening.config import (
    build_preflight_dataset,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    ScreeningOptions,
)
from openfold3.projects.of3_all_atom.physppi.screening.preflight import _seed

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_QBIOLIP_ROOT = Path("/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new")
_AFDB_ROOT = Path("/101045/jinyu/PPI/Prior_out/training/AFDB-10k-1")
_FORMAL_ROOT = Path("/101045/jinyu/PhysPPI/runs/physppi_formal_8gpu")
_INDEX_CACHE_ROOT = _FORMAL_ROOT / "reader-index-cache"
_MANIFEST_ROOT = _FORMAL_ROOT / "manifests"
_COMPONENTS_CIF = Path(
    "/101045/jinyu/PhysPPI/openfold-3/output/physppi_msa_cache/components.cif"
)
_REFERENCE_MOLECULES = (
    _REPOSITORY_ROOT
    / "openfold3/core/data/resources/canonical_protein_data/reference_mols"
)
_PRODUCTION_WORK_ROOT: tempfile.TemporaryDirectory[str] | None = None
_PRODUCTION_DATASETS: dict[DatasetName, object] = {}


class ProtectedMatchAuditRecord(BaseModel):
    """One terminal protected-matcher diagnostic for one exact source ID."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    dataset: DatasetName
    sample_id: str = Field(min_length=1)
    status: Literal["accepted", "identity_fallback"]
    reason: str = Field(min_length=1)
    identity_rmsd: float = Field(ge=0)
    candidate_rmsd: float = Field(ge=0)
    matched_coverage: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_status_reason(self) -> ProtectedMatchAuditRecord:
        if self.status == "accepted" and self.reason != "accepted":
            raise ValueError("accepted audit records require reason=accepted")
        return self


class MatchEvaluator(Protocol):
    def __call__(
        self, dataset: DatasetName, sample_id: str
    ) -> Mapping[str, Any] | object: ...


def load_runtime_ids(path: Path) -> tuple[tuple[DatasetName, str], ...]:
    """Load exact ledger IDs once, rejecting malformed or duplicate entries."""
    ids: list[tuple[DatasetName, str]] = []
    seen: set[tuple[DatasetName, str]] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                key = (
                    DatasetName(str(payload["dataset"]).lower()),
                    str(payload["sample_id"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid runtime ledger record at line {line_number}"
                ) from exc
            if not key[1]:
                raise ValueError(f"empty sample ID at line {line_number}")
            if key in seen:
                raise ValueError(f"duplicate runtime ID: {key[0].value}:{key[1]}")
            seen.add(key)
            ids.append(key)
    return tuple(ids)


def _diagnostics(value: Mapping[str, Any] | object) -> dict[str, object]:
    if isinstance(value, Mapping):
        source = value
        get: Callable[[str], object] = source.__getitem__
    else:

        def get(name: str) -> object:
            return getattr(value, name)

    accepted = bool(get("accepted"))
    identity_rmsd = float(get("identity_rmsd"))
    candidate_rmsd = float(get("candidate_rmsd"))
    if not math.isfinite(identity_rmsd) or not math.isfinite(candidate_rmsd):
        raise ValueError("matcher RMSD diagnostics must be finite")
    return {
        "status": "accepted" if accepted else "identity_fallback",
        "reason": str(get("reason")),
        "identity_rmsd": identity_rmsd,
        "candidate_rmsd": candidate_rmsd,
        "matched_coverage": int(get("matched_coverage")),
    }


def _production_options(output_root: Path) -> ScreeningOptions:
    return ScreeningOptions(
        qbiolip_root=_QBIOLIP_ROOT,
        afdb_root=_AFDB_ROOT,
        qbiolip_index_cache=_INDEX_CACHE_ROOT / "qbiolip10k-v1.pkl",
        afdb_index_cache=_INDEX_CACHE_ROOT / "afdb10k-v1.pkl",
        qbiolip_canonical_report=_MANIFEST_ROOT / "qbiolip-audit.json",
        afdb_canonical_report=_MANIFEST_ROOT / "afdb-audit.json",
        history_runs_root=_FORMAL_ROOT,
        manifests_dir=_MANIFEST_ROOT,
        runtime_ledger=_MANIFEST_ROOT / "runtime-failure-ledger.jsonl",
        prior_audit=(
            _FORMAL_ROOT
            / "audits/v1-v32-full-20k-data-errors/data-error-ids.json"
        ),
        components_cif=_COMPONENTS_CIF,
        reference_molecule_directory=_REFERENCE_MOLECULES,
        output_root=output_root,
        workers=1,
    )


def _dataset_for(dataset: DatasetName):
    global _PRODUCTION_WORK_ROOT
    if dataset in _PRODUCTION_DATASETS:
        return _PRODUCTION_DATASETS[dataset]
    if _PRODUCTION_WORK_ROOT is None:
        _PRODUCTION_WORK_ROOT = tempfile.TemporaryDirectory(
            prefix="physppi-protected-match-audit-"
        )
    root = Path(_PRODUCTION_WORK_ROOT.name)
    worker_cache = (
        root / "worker-cache" / str(os.getpid()) / f"{dataset.value}-spatial"
    )
    source = build_preflight_dataset(
        dataset,
        _production_options(root),
        worker_cache,
    )
    _PRODUCTION_DATASETS[dataset] = source
    return source


def _build_production_match_inputs(
    dataset: DatasetName,
    sample_id: str,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    """Build the canonical route once, before optional-prior materialization."""
    source = _dataset_for(dataset)
    _seed(dataset, sample_id, "spatial")
    rows = source.datapoint_cache.index[
        source.datapoint_cache["pdb_id"] == sample_id
    ].tolist()
    if len(rows) != 1:
        raise ValueError(
            f"expected one source row for {dataset.value}:{sample_id}, got {rows}"
        )
    index = int(rows[0])
    datapoint = source.datapoint_cache.iloc[index]
    source.set_current_datapoint_index(index)
    source.set_current_occurrence(
        SampleOccurrence(
            dataset_idx=0,
            datapoint_idx=index,
            epoch=0,
            global_slot=index,
        )
    )
    sample = source.reader.read(sample_id)
    occurrence_rng = build_occurrence_rng(
        source._physppi_data_seed,
        0,
        index,
        source.name,
        sample_id,
    )
    view = select_retained_chain_view(
        sample,
        occurrence_rng.chain_view,
        occurrence_rng.seed,
    )
    restricted = restrict_prior_sample_to_chains(sample, view.source_chain_ids)
    sample_data = source.create_canonical_all_features(
        sample=restricted,
        preferred_chain_or_interface=datapoint["preferred_chain_or_interface"],
        return_atom_arrays=True,
        return_crop_strategy=False,
    )
    pred_features = sample_data["features"]
    gt_features = pred_features["ground_truth"]
    atom_positions = gt_features["atom_positions"]
    atom_mask = gt_features["atom_resolved_mask"].bool()
    return pred_features, gt_features, atom_positions, atom_mask


def _run_matcher_with_identity_fallback(
    *,
    pred_features: dict,
    gt_features: dict,
    atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    protected_matcher: Callable[..., tuple[dict, Mapping[str, Any]]] = (
        find_protected_mol_permutation
    ),
    identity_matcher: Callable[..., tuple[dict, Mapping[str, Any]]] = (
        find_sequence_compatible_identity_mol_permutation
    ),
) -> Mapping[str, Any]:
    matcher_kwargs = {
        "pred_features": pred_features,
        "pred_atom_positions": atom_positions,
        "pred_atom_mask": atom_mask,
        "gt_features": gt_features,
        "gt_atom_positions": atom_positions,
        "gt_atom_mask": atom_mask,
    }
    try:
        _, diagnostics = protected_matcher(**matcher_kwargs)
        return diagnostics
    except ProtectedChainMatchError as exc:
        _, diagnostics = identity_matcher(
            **matcher_kwargs,
            reason=f"{type(exc).__name__}:{exc}",
        )
        return diagnostics


def production_match_evaluator(
    dataset: DatasetName,
    sample_id: str,
) -> Mapping[str, Any]:
    """Evaluate one exact ID through canonical features and production matching."""
    pred_features, gt_features, atom_positions, atom_mask = (
        _build_production_match_inputs(dataset, sample_id)
    )
    return _run_matcher_with_identity_fallback(
        pred_features=pred_features,
        gt_features=gt_features,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
    )


def run_protected_match_audit(
    runtime_ledger: Path,
    output: Path,
    evaluator: MatchEvaluator,
) -> tuple[ProtectedMatchAuditRecord, ...]:
    """Evaluate every exact ledger ID and atomically publish terminal JSONL."""
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    ids = load_runtime_ids(runtime_ledger)
    records = tuple(
        ProtectedMatchAuditRecord(
            dataset=dataset,
            sample_id=sample_id,
            **_diagnostics(evaluator(dataset, sample_id)),
        )
        for dataset, sample_id in ids
    )
    if len(records) != len(ids):
        raise RuntimeError("audit did not produce exactly one terminal row per ID")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(record.model_dump_json())
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        reread = tuple(
            ProtectedMatchAuditRecord.model_validate_json(line)
            for line in temporary.read_text().splitlines()
            if line
        )
        if len(reread) != len(ids):
            raise RuntimeError("serialized audit terminal-row count mismatch")
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return records


def _load_evaluator(specification: str) -> MatchEvaluator:
    try:
        module_name, function_name = specification.split(":", 1)
        evaluator = getattr(importlib.import_module(module_name), function_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError("evaluator must be an importable module:function") from exc
    if not callable(evaluator):
        raise TypeError("configured evaluator is not callable")
    return evaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact historical IDs through a caller-supplied read-only "
            "production matcher adapter."
        )
    )
    parser.add_argument("--runtime-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluator",
        help=(
            "Optional import path module:function. By default, use the canonical "
            "read-only PhysPPI feature/matcher adapter."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluator = (
        production_match_evaluator
        if args.evaluator is None
        else _load_evaluator(args.evaluator)
    )
    run_protected_match_audit(
        args.runtime_ledger,
        args.output,
        evaluator,
    )


if __name__ == "__main__":
    main()
