"""Normalize bounded historical PhysPPI failure evidence into seed records."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    HistoricalSeed,
)

SIGNATURE_PATTERNS = {
    "reference_molecule": (
        "reference molecule",
        "reference_mols",
        "failed to parse mse",
    ),
    "entity_id": ("entity_id",),
    "msa_width": ("msa", "width mismatch"),
    "malformed_charge": ("invalid literal for int() with base 10",),
    "canonical_mapping": (
        "canonical residue mapping",
        "canonical mapping fatal",
    ),
    "template_broadcast": ("could not be broadcast", "template"),
    "atom_id_order": ("atom ids are not strictly increasing",),
    "label_index": ("scattergatherkernel", "index out of bounds"),
    "paired_msa_contract": ("physppi_paired_msa_chain_mask must have shape",),
    "token_center": ("token center atom index is out of bounds",),
    "frame_start": ("invalid-frame-start",),
    "surface_shape": ("physppi_surface_local_coords must have trailing shape",),
    "nonfinite_feature": ("nan", "non-finite"),
    "missing_resource": ("no such file", "mandatory", "missing"),
}

_CATEGORY_SIGNATURES = {
    "canonical_residue_mapping_fatal": "canonical_mapping",
    "historical_log_label_index_failures": "label_index",
    "historical_log_paired_msa_contract_failures": "paired_msa_contract",
    "historical_log_surface_shape_failures": "surface_shape",
    "historical_runtime_label_index_failures": "label_index",
    "historical_runtime_surface_shape_failures": "surface_shape",
    "malformed_gt_pdb_charge": "malformed_charge",
    "missing_gt_pdb": "missing_resource",
    "nonincreasing_gt_atom_ids": "atom_id_order",
    "nonincreasing_template_atom_ids": "atom_id_order",
    "surface_neighbor_axis_collision": "surface_shape",
}
_SAMPLE_PATTERN = re.compile(r"Train step failed with pdb id (\S+)")
_VERSION_PATTERN = re.compile(r"(?:^|-)v(\d+)(?:-|$)")
_ERROR_BLOCK_MAX_LINES = 32


def _normalized_text(value: object) -> str:
    return re.sub(r"[-_]+", " ", str(value).lower())


def _matching_signatures(value: object) -> list[str]:
    text = str(value).lower()
    normalized = _normalized_text(value)
    return [
        signature
        for signature, patterns in SIGNATURE_PATTERNS.items()
        if (
            all(
                pattern in text or _normalized_text(pattern) in normalized
                for pattern in patterns
            )
            if signature in {"msa_width", "template_broadcast"}
            else any(
                pattern in text or _normalized_text(pattern) in normalized
                for pattern in patterns
            )
        )
    ]


def _dataset_from_sample_id(sample_id: str) -> DatasetName:
    if sample_id.startswith(("AF-", "a-")):
        return DatasetName.AFDB
    return DatasetName.QBIOLIP


def _versioned_log_paths(runs_root: Path) -> Iterable[tuple[Path, int, Path]]:
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        version_match = _VERSION_PATTERN.search(run_dir.name)
        if version_match is None:
            continue
        version = int(version_match.group(1))
        if not 1 <= version <= 32:
            continue
        log_path = run_dir / "train.log"
        if log_path.is_file():
            yield run_dir, version, log_path


def _versioned_log_evidence(
    runs_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_dir, version, log_path in _versioned_log_paths(runs_root):
        pending_signatures: dict[str, int] = {}
        latest_sample_id: str | None = None
        latest_sample_line: int | None = None
        found_sample_id = False
        run_signatures: set[str] = set()
        run_name, run_version, source = run_dir.name, version, str(log_path)
        lines = log_path.read_text(errors="replace").splitlines()

        def append_records(
            sample_ids: Iterable[str],
            signatures: Iterable[str],
            context: tuple[str, int, str] = (run_name, run_version, source),
        ) -> None:
            for sample_id in sample_ids:
                for signature in signatures:
                    records.append(
                        {
                            "sample_id": sample_id,
                            "run": context[0],
                            "version": context[1],
                            "signature": signature,
                            "source": context[2],
                        }
                    )

        for line_number, line in enumerate(lines):
            pending_signatures = {
                signature: end_line
                for signature, end_line in pending_signatures.items()
                if end_line >= line_number
            }
            sample_ids = _SAMPLE_PATTERN.findall(line)
            if sample_ids:
                found_sample_id = True
                append_records(sample_ids[:1], pending_signatures)
                pending_signatures.clear()
                latest_sample_id = sample_ids[-1]
                latest_sample_line = line_number
            line_signatures = _matching_signatures(line)
            if not line_signatures:
                continue
            run_signatures.update(line_signatures)
            next_block_has_id = any(
                _SAMPLE_PATTERN.search(candidate)
                for candidate in lines[
                    line_number + 1 : line_number + _ERROR_BLOCK_MAX_LINES + 1
                ]
            )
            if (
                latest_sample_id is not None
                and latest_sample_line is not None
                and line_number - latest_sample_line <= _ERROR_BLOCK_MAX_LINES
                and not next_block_has_id
            ):
                append_records([latest_sample_id], line_signatures)
            elif next_block_has_id:
                for signature in line_signatures:
                    pending_signatures[signature] = line_number + _ERROR_BLOCK_MAX_LINES
        if run_signatures and not found_sample_id:
            summaries.append(
                {
                    "run": run_name,
                    "version": version,
                    "signatures": sorted(run_signatures),
                    "source": source,
                }
            )
    return (
        sorted(
            records,
            key=lambda item: (item["run"], item["sample_id"], item["signature"]),
        ),
        sorted(summaries, key=lambda item: (item["run"], item["source"])),
    )


def load_versioned_log_sample_ids(runs_root: Path) -> list[dict[str, Any]]:
    """Extract exact sample IDs from data-triggered v1-v32 training logs."""
    records, _ = _versioned_log_evidence(runs_root)
    return [
        {key: record[key] for key in ("sample_id", "run", "version", "signature")}
        for record in records
    ]


def load_versioned_log_run_summaries(runs_root: Path) -> list[dict[str, Any]]:
    """Keep bounded error evidence that a historical log cannot tie to an ID."""
    _, summaries = _versioned_log_evidence(runs_root)
    return summaries


def _load_runtime_ledger(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid runtime ledger JSON at line {line_number}"
                ) from exc
    return entries


def load_manifest_runtime_skips(manifests_dir: Path) -> list[dict[str, Any]]:
    """Collect the first record for every exact dataset/ID skip."""
    by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(manifests_dir.glob("formal-manifest*.json")):
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for entry in manifest.get("runtime_skips", []):
            dataset = str(entry.get("dataset", "")).lower()
            sample_id = entry.get("sample_id")
            if dataset not in {"qbiolip", "afdb"} or not sample_id:
                continue
            key = (dataset, str(sample_id))
            if key not in by_id:
                record = dict(entry)
                record["dataset"] = dataset
                record["sample_id"] = str(sample_id)
                record["first_seen_manifest"] = str(path)
                by_id[key] = record
    return [by_id[key] for key in sorted(by_id)]


def _versioned_manifest_runtime_skips(
    manifests_dir: Path,
) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(manifests_dir.glob("formal-manifest*.json")):
        version_match = _VERSION_PATTERN.search(path.stem)
        if version_match is None or not 1 <= int(version_match.group(1)) <= 32:
            continue
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for entry in manifest.get("runtime_skips", []):
            if isinstance(entry, dict):
                yield path, entry


def _runtime_category(
    entries: list[dict[str, Any]], terms: tuple[str, ...]
) -> dict[str, list[str]]:
    result = {"qbiolip": [], "afdb": []}
    for entry in entries:
        dataset = str(entry.get("dataset", "")).lower()
        sample_id = entry.get("sample_id")
        reason = " ".join(
            str(entry.get(key, ""))
            for key in ("reason", "exception_message", "failure_category")
        ).lower()
        if dataset in result and sample_id and any(term in reason for term in terms):
            result[dataset].append(str(sample_id))
    return result


def _entry_signatures(entry: dict[str, Any]) -> list[str]:
    signatures = [
        signature
        for field in (
            "reason",
            "exception_message",
            "failure_category",
            "message",
            "error_message",
            "error",
            "category",
        )
        for signature in _matching_signatures(entry.get(field, ""))
    ]
    category = str(entry.get("failure_category", "")).replace("-", "_")
    if category == "surface_local_coordinates_shape":
        signatures.append("surface_shape")
    return sorted(set(signatures))


def _dataset_name(value: object) -> DatasetName | None:
    try:
        return DatasetName(str(value).lower())
    except ValueError:
        return None


def load_historical_seeds(
    runs_root: Path,
    manifests_dir: Path,
    runtime_ledger: Path,
    prior_audit: Path,
) -> dict[tuple[DatasetName, str], HistoricalSeed]:
    """Union exact bounded historical evidence without turning it into exclusion."""
    evidence: dict[tuple[DatasetName, str], dict[str, set[str]]] = {}

    def add(
        dataset: DatasetName | None,
        sample_id: object,
        signatures: Iterable[str],
        source: Path | str,
    ) -> None:
        if dataset is None or not sample_id:
            return
        key = (dataset, str(sample_id))
        bucket = evidence.setdefault(key, {"signatures": set(), "sources": set()})
        bucket["signatures"].update(signatures)
        bucket["sources"].add(str(source))

    log_records, _ = _versioned_log_evidence(runs_root)
    for record in log_records:
        add(
            _dataset_from_sample_id(record["sample_id"]),
            record["sample_id"],
            [record["signature"]],
            record["source"],
        )
    for manifest_path, entry in _versioned_manifest_runtime_skips(manifests_dir):
        add(
            _dataset_name(entry.get("dataset")),
            entry.get("sample_id"),
            _entry_signatures(entry),
            manifest_path,
        )
    for entry in _load_runtime_ledger(runtime_ledger):
        add(
            _dataset_name(entry.get("dataset")),
            entry.get("sample_id"),
            _entry_signatures(entry),
            runtime_ledger,
        )
    audit = json.loads(prior_audit.read_text())
    for category, payload in audit.get("categories", {}).items():
        signature = _CATEGORY_SIGNATURES.get(category)
        if signature is None:
            signature = next(iter(_matching_signatures(category)), None)
        for dataset, field in (
            (DatasetName.QBIOLIP, "qbiolip_ids"),
            (DatasetName.AFDB, "afdb_ids"),
        ):
            for sample_id in payload.get(field, []):
                add(
                    dataset,
                    sample_id,
                    [] if signature is None else [signature],
                    prior_audit,
                )
    return {
        key: HistoricalSeed(
            dataset=key[0],
            sample_id=key[1],
            signatures=sorted(bucket["signatures"]),
            sources=sorted(bucket["sources"]),
        )
        for key, bucket in sorted(
            evidence.items(), key=lambda item: (item[0][0].value, item[0][1])
        )
    }


def historical_ids_by_dataset(
    seeds: Mapping[tuple[DatasetName, str], HistoricalSeed],
) -> dict[DatasetName, tuple[str, ...]]:
    """Return the exact historical quarantine union in deterministic order."""
    result: dict[DatasetName, list[str]] = {
        DatasetName.QBIOLIP: [],
        DatasetName.AFDB: [],
    }
    for (dataset, sample_id), seed in sorted(
        seeds.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        if seed.dataset is not dataset or seed.sample_id != sample_id:
            raise ValueError("historical seed key does not match its payload")
        result[dataset].append(sample_id)
    return {dataset: tuple(sample_ids) for dataset, sample_ids in result.items()}
