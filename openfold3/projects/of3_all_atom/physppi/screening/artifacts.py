"""Finalize complete PhysPPI screening outputs into immutable artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    SampleRecord,
    SampleStatus,
    ScanMetadata,
    ScanSummary,
)

EXPECTED_SOURCE_COUNT = 10_000
EXPECTED_TOTAL_COUNT = 20_000
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HASHED_ARTIFACTS = (
    "screen-results.jsonl",
    "data-error-ids.json",
    "historical-quarantine-ids.json",
    "qbiolip-eligible-manifest.json",
    "afdb-eligible-manifest.json",
    "screened-training-manifest.json",
    "summary.json",
    "metadata.json",
    "evidence.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: object) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_bytes_atomic(path, serialized.encode())


def _dataset_mapping(
    values: Mapping[DatasetName | str, Sequence[str]],
) -> dict[DatasetName, tuple[str, ...]]:
    result: dict[DatasetName, tuple[str, ...]] = {}
    for raw_dataset, raw_ids in values.items():
        dataset = (
            raw_dataset
            if isinstance(raw_dataset, DatasetName)
            else DatasetName(raw_dataset)
        )
        ids = tuple(raw_ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {dataset.value} source IDs")
        result[dataset] = ids
    if set(result) != {DatasetName.QBIOLIP, DatasetName.AFDB}:
        raise ValueError("source mappings must contain qbiolip and afdb")
    return result


def verify_sha256s(path: Path) -> bool:
    """Verify every relative file listed in a GNU-style SHA256SUMS file."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False
    if not lines:
        return False
    for line in lines:
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            return False
        target = path.parent / relative
        if not _HEX_64.fullmatch(expected) or not target.is_file():
            return False
        if _sha256_file(target) != expected:
            return False
    return True


def _validate_inputs(
    records: Sequence[SampleRecord],
    source_ids: Mapping[DatasetName, tuple[str, ...]],
    historical_ids: Mapping[DatasetName, tuple[str, ...]],
) -> dict[tuple[DatasetName, str], SampleRecord]:
    if len(records) != EXPECTED_TOTAL_COUNT:
        raise ValueError("finalization requires exactly 20,000 terminal records")
    by_key: dict[tuple[DatasetName, str], SampleRecord] = {}
    for record in records:
        key = (record.dataset, record.sample_id)
        if key in by_key:
            raise ValueError(
                f"duplicate terminal record: {record.dataset.value}:{record.sample_id}"
            )
        if record.status is SampleStatus.INFRASTRUCTURE_ERROR:
            raise ValueError("infrastructure records cannot be finalized")
        by_key[key] = record
    for dataset in DatasetName:
        inventory = source_ids[dataset]
        if len(inventory) != EXPECTED_SOURCE_COUNT:
            raise ValueError(
                f"{dataset.value} inventory must contain exactly 10,000 IDs"
            )
        expected = {(dataset, sample_id) for sample_id in inventory}
        actual = {key for key in by_key if key[0] is dataset}
        if actual != expected:
            missing = len(expected - actual)
            unknown = len(actual - expected)
            raise ValueError(
                f"{dataset.value} record partition mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        unknown_historical = set(historical_ids[dataset]) - set(inventory)
        if unknown_historical:
            raise ValueError(f"{dataset.value} historical IDs outside source inventory")
    return by_key


def _category_payload(
    records: Sequence[SampleRecord],
    source_ids: Mapping[DatasetName, tuple[str, ...]],
    excluded_ids: Mapping[DatasetName, set[str]],
) -> dict[str, Any]:
    category_sets: dict[str, dict[DatasetName, set[str]]] = {}
    for record in records:
        for finding in record.findings:
            category_sets.setdefault(
                finding.code,
                {
                    DatasetName.QBIOLIP: set(),
                    DatasetName.AFDB: set(),
                },
            )[record.dataset].add(record.sample_id)
    categories: dict[str, dict[str, list[str]]] = {}
    for code, per_dataset in sorted(category_sets.items()):
        categories[code] = {
            dataset.value: [
                sample_id
                for sample_id in source_ids[dataset]
                if sample_id in per_dataset[dataset]
            ]
            for dataset in DatasetName
        }
    return {
        "schema_version": 2,
        "categories": categories,
        "datasets": {
            dataset.value: {
                "ids": [
                    sample_id
                    for sample_id in source_ids[dataset]
                    if sample_id in excluded_ids[dataset]
                ],
                "count": len(excluded_ids[dataset]),
            }
            for dataset in DatasetName
        },
    }


def _reread_validate(root: Path) -> None:
    records = [
        SampleRecord.model_validate_json(line)
        for line in (root / "screen-results.jsonl").read_text().splitlines()
        if line
    ]
    if len(records) != EXPECTED_TOTAL_COUNT:
        raise ValueError("serialized record count is not exactly 20,000")
    ScanMetadata.model_validate_json((root / "metadata.json").read_text())
    ScanSummary.model_validate_json((root / "summary.json").read_text())
    for filename in _HASHED_ARTIFACTS:
        if filename.endswith(".json"):
            json.loads((root / filename).read_text())


def finalize_artifacts(
    *,
    records: Iterable[SampleRecord],
    historical_ids: Mapping[DatasetName | str, Sequence[str]],
    source_ids: Mapping[DatasetName | str, Sequence[str]],
    metadata: ScanMetadata,
    output: Path,
    implementation_git_head: str,
    screen_config_sha256: str,
    evidence: Mapping[str, object],
) -> dict[str, str]:
    """Write the complete production artifact set and publish COMPLETE last."""
    if not _HEX_40.fullmatch(implementation_git_head):
        raise ValueError("implementation_git_head must be 40 lowercase hex digits")
    if not _HEX_64.fullmatch(screen_config_sha256):
        raise ValueError("screen_config_sha256 must be 64 lowercase hex digits")
    if output.exists():
        raise FileExistsError(f"artifact output already exists: {output}")

    ordered_records = tuple(records)
    inventories = _dataset_mapping(source_ids)
    quarantine = _dataset_mapping(historical_ids)
    by_key = _validate_inputs(ordered_records, inventories, quarantine)
    output.mkdir(parents=True)

    terminal_excluded = {
        dataset: {
            record.sample_id
            for record in ordered_records
            if record.dataset is dataset and record.status is SampleStatus.EXCLUDED
        }
        for dataset in DatasetName
    }
    exclusion_union = {
        dataset: terminal_excluded[dataset] | set(quarantine[dataset])
        for dataset in DatasetName
    }
    diagnostic = {
        dataset: {
            record.sample_id
            for record in ordered_records
            if record.dataset is dataset
            and (record.findings or record.historical_signatures)
        }
        | set(quarantine[dataset])
        for dataset in DatasetName
    }
    train_ids = {
        dataset: [
            sample_id
            for sample_id in inventories[dataset]
            if sample_id not in exclusion_union[dataset]
        ]
        for dataset in DatasetName
    }
    diagnostic_ids = {
        dataset: [
            sample_id
            for sample_id in inventories[dataset]
            if sample_id in diagnostic[dataset]
        ]
        for dataset in DatasetName
    }

    results = "".join(
        by_key[(dataset, sample_id)].model_dump_json() + "\n"
        for dataset in DatasetName
        for sample_id in inventories[dataset]
    )
    _write_bytes_atomic(output / "screen-results.jsonl", results.encode())

    data_errors = _category_payload(
        ordered_records,
        inventories,
        exclusion_union,
    )
    _write_json_atomic(output / "data-error-ids.json", data_errors)
    _write_json_atomic(
        output / "historical-quarantine-ids.json",
        {
            "schema_version": 2,
            "datasets": {
                dataset.value: {
                    "ids": list(quarantine[dataset]),
                    "count": len(quarantine[dataset]),
                }
                for dataset in DatasetName
            },
        },
    )

    for dataset in DatasetName:
        _write_json_atomic(
            output / f"{dataset.value}-eligible-manifest.json",
            {
                "schema_version": 2,
                "dataset": dataset.value,
                "source_inventory": len(inventories[dataset]),
                "train_ids": train_ids[dataset],
                "diagnostic_ids": diagnostic_ids[dataset],
            },
        )

    error_ids_sha256 = _sha256_file(output / "data-error-ids.json")
    combined_manifest = {
        "schema_version": 2,
        "source_inventory": {
            dataset.value: len(inventories[dataset]) for dataset in DatasetName
        },
        "datasets": {
            dataset.value: {
                "train_ids": train_ids[dataset],
                "diagnostic_ids": diagnostic_ids[dataset],
            }
            for dataset in DatasetName
        },
        "error_ids_sha256": error_ids_sha256,
        "implementation_git_head": implementation_git_head,
        "screen_config_sha256": screen_config_sha256,
    }
    _write_json_atomic(
        output / "screened-training-manifest.json",
        combined_manifest,
    )

    status_counts = Counter(record.status for record in ordered_records)
    category_counts = Counter(
        finding.code for record in ordered_records for finding in record.findings
    )
    summary = ScanSummary(
        complete=True,
        source_counts={dataset: len(inventories[dataset]) for dataset in DatasetName},
        terminal_record_count=len(ordered_records),
        duplicate_terminal_record_ids=[],
        status_counts={status: status_counts.get(status, 0) for status in SampleStatus},
        category_counts=dict(sorted(category_counts.items())),
        unresolved_infrastructure_ids=[],
        artifact_sha256={},
    )
    _write_json_atomic(output / "summary.json", summary.model_dump(mode="json"))
    _write_json_atomic(output / "metadata.json", metadata.model_dump(mode="json"))
    _write_json_atomic(output / "evidence.json", dict(evidence))

    _reread_validate(output)
    hashes = {
        filename: _sha256_file(output / filename) for filename in _HASHED_ARTIFACTS
    }
    sums = "".join(
        f"{digest}  {filename}\n" for filename, digest in sorted(hashes.items())
    )
    _write_bytes_atomic(output / "SHA256SUMS", sums.encode())
    if not verify_sha256s(output / "SHA256SUMS"):
        raise ValueError("artifact SHA256 verification failed")
    _write_bytes_atomic(
        output / "COMPLETE",
        (_sha256_file(output / "SHA256SUMS") + "\n").encode(),
    )
    return hashes


__all__ = ["finalize_artifacts", "verify_sha256s"]
