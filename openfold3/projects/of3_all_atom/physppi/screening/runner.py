"""Restart-safe parent coordinator for production PhysPPI screening."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from openfold3.projects.of3_all_atom.physppi.screening.dataset import (
    PreflightFn,
    ScreeningDataset,
    ScreeningItem,
    build_screening_items,
    identity_collate,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Finding,
    HistoricalSeed,
    SampleRecord,
    SampleStatus,
    ScanMetadata,
    ScanSummary,
    ScreeningOptions,
    Severity,
)

SCANNER_VERSION = "3"
PRODUCTION_RECORD_COUNT = 20_000
REPORT_FILES = {
    "records": "records.jsonl",
    "all_findings": "data-error-ids.json",
    "excluded": "excluded-ids.json",
    "recovered": "recovered-historical-ids.json",
    "warnings": "warning-ids.json",
}
COMPLETE_MARKER = "COMPLETE"

LoaderFactory = Callable[[ScreeningDataset], Iterable[list[SampleRecord]]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _current_git_head() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("biotite", "gemmi", "pydantic", "scipy", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def _cli_options(options: ScreeningOptions) -> dict[str, str | int | bool | list[str]]:
    payload: dict[str, str | int | bool | list[str]] = {}
    for name, value in options.model_dump().items():
        if name == "resume":
            payload[name] = False
        elif isinstance(value, Path):
            payload[name] = str(value)
        elif isinstance(value, tuple):
            payload[name] = [str(item) for item in value]
        else:
            payload[name] = value
    return payload


def build_scan_metadata(options: ScreeningOptions) -> tuple[ScanMetadata, str]:
    metadata = ScanMetadata(
        scanner_version=SCANNER_VERSION,
        git_head=_current_git_head(),
        source_roots={
            DatasetName.QBIOLIP: options.qbiolip_root,
            DatasetName.AFDB: options.afdb_root,
        },
        reader_cache_sha256={
            DatasetName.QBIOLIP: _sha256_file(options.qbiolip_index_cache),
            DatasetName.AFDB: _sha256_file(options.afdb_index_cache),
        },
        canonical_report_sha256={
            DatasetName.QBIOLIP: _sha256_file(options.qbiolip_canonical_report),
            DatasetName.AFDB: _sha256_file(options.afdb_canonical_report),
        },
        dependency_versions=_dependency_versions(),
        worker_count=options.workers,
        sample_timeout_seconds=options.sample_timeout_seconds,
        max_attempts=options.max_attempts,
        crop_strategies=options.crop_strategies,
        cli_options=_cli_options(options),
    )
    canonical = _canonical_json(metadata.model_dump(mode="json"))
    return metadata, hashlib.sha256(canonical.encode()).hexdigest()


def build_screening_dataloader(dataset: ScreeningDataset) -> DataLoader:
    """Construct the one approved production DataLoader contract."""
    if dataset.options.workers != 8:
        raise ValueError("production screening requires exactly 8 DataLoader workers")
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
        shuffle=False,
        multiprocessing_context="forkserver",
        collate_fn=identity_collate,
    )


def _prepare_output(
    options: ScreeningOptions, metadata: ScanMetadata, metadata_hash: str
) -> None:
    root = options.output_root
    metadata_path = root / "metadata.json"
    if not options.resume:
        if root.exists():
            raise FileExistsError(
                f"output root already exists; pass --resume after validation: {root}"
            )
        root.mkdir(parents=True)
        _write_json_atomic(
            metadata_path,
            {
                "metadata": metadata.model_dump(mode="json"),
                "metadata_hash": metadata_hash,
            },
        )
        return
    if not root.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError("valid metadata.json is required for --resume")
    stored = json.loads(metadata_path.read_text())
    stored_metadata = stored.get("metadata")
    stored_hash = stored.get("metadata_hash")
    actual_stored_hash = hashlib.sha256(
        _canonical_json(stored_metadata).encode()
    ).hexdigest()
    if (
        stored_hash != actual_stored_hash
        or stored_hash != metadata_hash
        or stored_metadata != metadata.model_dump(mode="json")
    ):
        raise ValueError("metadata hash mismatch; refusing resume")


def _load_records(path: Path) -> dict[tuple[DatasetName, str], SampleRecord]:
    records: dict[tuple[DatasetName, str], SampleRecord] = {}
    if not path.exists():
        return records
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = SampleRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid terminal record at line {line_number}"
                ) from exc
            key = (record.dataset, record.sample_id)
            if key in records:
                raise ValueError(
                    "duplicate terminal record: "
                    f"{record.dataset.value}:{record.sample_id}"
                )
            records[key] = record
    return records


def _append_record(path: Path, record: SampleRecord) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parent_infrastructure_record(
    item: ScreeningItem, exc: BaseException
) -> SampleRecord:
    return SampleRecord.classify(
        ordinal=item.ordinal,
        dataset=item.dataset,
        sample_id=item.sample_id,
        findings=[
            Finding(
                code="dataloader_generation",
                severity=Severity.INFRASTRUCTURE,
                message=f"{item.dataset.value}:{item.sample_id}: {exc}",
                exception_type=type(exc).__name__,
            )
        ],
        audit_views=[],
        historical_signatures=(
            [] if item.historical_seed is None else item.historical_seed.signatures
        ),
        elapsed_seconds=0.0,
        source_hashes={},
        validator_versions={},
    )


def _run_generation(
    dataset: ScreeningDataset,
    loader_factory: LoaderFactory,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    loader = loader_factory(dataset)
    iterator = iter(loader)
    try:
        for position, batch in enumerate(iterator):
            if not isinstance(batch, list) or len(batch) != 1:
                raise RuntimeError("identity collate must return one-record lists")
            record = batch[0]
            if not isinstance(record, SampleRecord):
                raise TypeError("DataLoader returned a non-SampleRecord payload")
            expected = dataset.items[position]
            if (
                record.dataset is not expected.dataset
                or record.sample_id != expected.sample_id
                or record.ordinal != expected.ordinal
            ):
                raise ValueError(
                    "DataLoader record identity does not match its source reader"
                )
            records.append(record)
    except Exception as exc:
        for item in dataset.items[len(records) :]:
            records.append(_parent_infrastructure_record(item, exc))
    finally:
        del iterator
        del loader
        gc.collect()
    return records


def _finding_report(records: Sequence[SampleRecord]) -> dict[str, Any]:
    categories: dict[str, dict[str, list[str]]] = {}
    for record in sorted(records, key=lambda item: item.ordinal):
        for finding in record.findings:
            bucket = categories.setdefault(
                finding.code, {"qbiolip_ids": [], "afdb_ids": []}
            )
            bucket[f"{record.dataset.value}_ids"].append(record.sample_id)
    return {"categories": categories}


def _status_ids(
    records: Sequence[SampleRecord], status: SampleStatus
) -> dict[str, list[str]]:
    result = {"qbiolip_ids": [], "afdb_ids": []}
    for record in sorted(records, key=lambda item: item.ordinal):
        if record.status is status:
            result[f"{record.dataset.value}_ids"].append(record.sample_id)
    return result


def _write_reports(
    root: Path,
    records: Sequence[SampleRecord],
) -> dict[str, str]:
    payloads = {
        "all_findings": _finding_report(records),
        "excluded": _status_ids(records, SampleStatus.EXCLUDED),
        "recovered": _status_ids(records, SampleStatus.RECOVERED),
        "warnings": _status_ids(records, SampleStatus.WARNING),
    }
    for name, payload in payloads.items():
        _write_json_atomic(root / REPORT_FILES[name], payload)
    hashes = {
        name: _sha256_file(root / filename) for name, filename in REPORT_FILES.items()
    }
    return hashes


def _build_summary(
    source_ids: Mapping[DatasetName, Sequence[str]],
    records: Sequence[SampleRecord],
    artifact_sha256: Mapping[str, str],
    expected_count: int,
) -> ScanSummary:
    status_counts = Counter(record.status for record in records)
    category_counts = Counter(
        finding.code for record in records for finding in record.findings
    )
    unresolved = sorted(
        f"{record.dataset.value}:{record.sample_id}"
        for record in records
        if record.status is SampleStatus.INFRASTRUCTURE_ERROR
    )
    source_counts = {
        DatasetName.QBIOLIP: len(source_ids.get(DatasetName.QBIOLIP, ())),
        DatasetName.AFDB: len(source_ids.get(DatasetName.AFDB, ())),
    }
    complete = (
        expected_count == PRODUCTION_RECORD_COUNT
        and source_counts[DatasetName.QBIOLIP] == 10_000
        and source_counts[DatasetName.AFDB] == 10_000
        and len(records) == PRODUCTION_RECORD_COUNT
        and not unresolved
    )
    return ScanSummary(
        complete=complete,
        source_counts=source_counts,
        terminal_record_count=len(records),
        duplicate_terminal_record_ids=[],
        status_counts={status: status_counts.get(status, 0) for status in SampleStatus},
        category_counts=dict(sorted(category_counts.items())),
        unresolved_infrastructure_ids=unresolved,
        artifact_sha256=artifact_sha256,
    )


def run_screening(
    options: ScreeningOptions,
    *,
    source_ids: Mapping[DatasetName, Sequence[str]],
    historical_seeds: Mapping[tuple[DatasetName, str], HistoricalSeed],
    preflight: PreflightFn,
    loader_factory: LoaderFactory = build_screening_dataloader,
    expected_count: int = PRODUCTION_RECORD_COUNT,
) -> ScanSummary:
    """Run one sequential generation plus one infrastructure-only retry."""
    if options.workers != 8:
        raise ValueError("production screening requires exactly 8 workers")
    if options.max_attempts != 2:
        raise ValueError("production screening requires exactly two attempts")
    metadata, metadata_hash = build_scan_metadata(options)
    _prepare_output(options, metadata, metadata_hash)
    complete_path = options.output_root / COMPLETE_MARKER
    if complete_path.exists():
        complete_path.unlink()

    all_items = build_screening_items(source_ids, historical_seeds)
    allowed = {(item.dataset, item.sample_id): item for item in all_items}
    records_path = options.output_root / REPORT_FILES["records"]
    records_by_key = _load_records(records_path)
    unknown = set(records_by_key) - set(allowed)
    if unknown:
        rendered = ", ".join(f"{d.value}:{i}" for d, i in sorted(unknown))
        raise ValueError(
            f"resume contains IDs outside current source inventory: {rendered}"
        )

    pending = [
        item
        for item in all_items
        if (item.dataset, item.sample_id) not in records_by_key
    ]
    first_dataset = ScreeningDataset(pending, options, preflight=preflight)
    first_records = _run_generation(first_dataset, loader_factory)
    retry_items: list[ScreeningItem] = []
    for item, record in zip(pending, first_records, strict=True):
        if record.status is SampleStatus.INFRASTRUCTURE_ERROR:
            retry_items.append(item)
            continue
        key = (record.dataset, record.sample_id)
        records_by_key[key] = record
        _append_record(records_path, record)

    if retry_items:
        retry_dataset = ScreeningDataset(retry_items, options, preflight=preflight)
        retry_records = _run_generation(retry_dataset, loader_factory)
        for record in retry_records:
            key = (record.dataset, record.sample_id)
            records_by_key[key] = record
            _append_record(records_path, record)

    ordered_records = sorted(records_by_key.values(), key=lambda record: record.ordinal)
    hashes = _write_reports(options.output_root, ordered_records)
    summary = _build_summary(
        source_ids,
        ordered_records,
        hashes,
        expected_count,
    )
    _write_json_atomic(
        options.output_root / "summary.json",
        {
            **summary.model_dump(mode="json"),
            "metadata_hash": metadata_hash,
            "wall_time_unix": time.time(),
        },
    )
    if summary.complete:
        complete_path.write_text(
            _sha256_file(options.output_root / "summary.json") + "\n"
        )
    return summary


__all__ = [
    "PRODUCTION_RECORD_COUNT",
    "build_scan_metadata",
    "build_screening_dataloader",
    "run_screening",
]
