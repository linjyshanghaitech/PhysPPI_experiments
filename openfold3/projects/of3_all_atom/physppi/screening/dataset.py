"""Sequential production dataset used by the PhysPPI screening DataLoader."""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Finding,
    HistoricalSeed,
    SampleRecord,
    ScreeningOptions,
    Severity,
)
from openfold3.projects.of3_all_atom.physppi.screening.preflight import (
    initialize_worker,
    preflight_audit_view,
)
from openfold3.projects.of3_all_atom.physppi.screening.views import (
    DeterministicAuditView,
    audit_views,
)

PreflightFn = Callable[
    [DatasetName, int, str, DeterministicAuditView], Sequence[Finding]
]


@dataclass(frozen=True, slots=True)
class ScreeningItem:
    """Stable source identity independent of execution order."""

    ordinal: int
    dataset: DatasetName
    source_index: int
    sample_id: str
    historical_seed: HistoricalSeed | None = None


def identity_collate(batch: list[SampleRecord]) -> list[SampleRecord]:
    """Keep validated records intact; the production batch size is exactly one."""
    return batch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("biotite", "gemmi", "pydantic", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def _source_hashes(options: ScreeningOptions) -> dict[str, str]:
    return {
        "reader_index": _sha256_file(options.qbiolip_index_cache),
        "afdb_reader_index": _sha256_file(options.afdb_index_cache),
        "canonical_report": _sha256_file(options.qbiolip_canonical_report),
        "afdb_canonical_report": _sha256_file(options.afdb_canonical_report),
    }


def _tag_finding(finding: Finding, view: DeterministicAuditView) -> Finding:
    return finding.model_copy(
        update={
            "metrics": {
                **finding.metrics,
                "view_index": view.view_index,
                "crop_strategy": view.crop_strategy,
                "requested_level": view.requested_level,
                "interface_cutoff": view.interface_cutoff,
                "multichain_probe": view.multichain_probe,
            }
        }
    )


def _protocol_finding(
    item: ScreeningItem, view: DeterministicAuditView, exc: BaseException
) -> Finding:
    return Finding(
        code="screening_worker_protocol",
        severity=Severity.INFRASTRUCTURE,
        message=f"{item.dataset.value}:{item.sample_id}: {exc}",
        exception_type=type(exc).__name__,
        metrics={
            "view_index": view.view_index,
            "crop_strategy": view.crop_strategy,
            "requested_level": view.requested_level,
            "interface_cutoff": view.interface_cutoff,
            "multichain_probe": view.multichain_probe,
        },
    )


def build_screening_items(
    source_ids: Mapping[DatasetName, Sequence[str]],
    historical_seeds: Mapping[tuple[DatasetName, str], HistoricalSeed],
) -> tuple[ScreeningItem, ...]:
    """Front-load in-source historical IDs once, then retain canonical order."""
    canonical: list[ScreeningItem] = []
    seen: set[tuple[DatasetName, str]] = set()
    ordinal = 0
    for dataset in (DatasetName.QBIOLIP, DatasetName.AFDB):
        ids = tuple(source_ids.get(dataset, ()))
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {dataset.value} source IDs")
        for source_index, sample_id in enumerate(ids):
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"invalid {dataset.value} source ID")
            key = (dataset, sample_id)
            if key in seen:
                raise ValueError(
                    f"duplicate source identity: {dataset.value}:{sample_id}"
                )
            seen.add(key)
            canonical.append(
                ScreeningItem(
                    ordinal=ordinal,
                    dataset=dataset,
                    source_index=source_index,
                    sample_id=sample_id,
                    historical_seed=historical_seeds.get(key),
                )
            )
            ordinal += 1
    historical = [item for item in canonical if item.historical_seed is not None]
    ordinary = [item for item in canonical if item.historical_seed is None]
    return tuple(historical + ordinary)


class ScreeningDataset(Dataset[SampleRecord]):
    """Run all deterministic views for one source ID inside one worker call."""

    def __init__(
        self,
        items: Sequence[ScreeningItem],
        options: ScreeningOptions,
        *,
        preflight: PreflightFn = preflight_audit_view,
    ) -> None:
        self.items = tuple(items)
        self.options = options
        self.preflight = preflight
        self.source_hashes = _source_hashes(options)
        self.validator_versions = _validation_versions()
        self._initialized_pid: int | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> SampleRecord:
        if self._initialized_pid != os.getpid():
            initialize_worker(self.options)
            self._initialized_pid = os.getpid()
        item = self.items[index]
        started = time.monotonic()
        findings: list[Finding] = []
        attempted_views: list[DeterministicAuditView] = []
        for view in audit_views(
            item.dataset,
            item.sample_id,
            source_ordinal=item.ordinal,
        ):
            attempted_views.append(view)
            try:
                view_findings = self.preflight(
                    item.dataset,
                    item.source_index,
                    item.sample_id,
                    view,
                )
            except Exception as exc:
                findings.append(_protocol_finding(item, view, exc))
                break
            if any(
                finding.severity is Severity.INFRASTRUCTURE for finding in view_findings
            ):
                findings.extend(
                    _tag_finding(finding, view) for finding in view_findings
                )
                break
            findings.extend(_tag_finding(finding, view) for finding in view_findings)

        historical_signatures = (
            () if item.historical_seed is None else item.historical_seed.signatures
        )
        return SampleRecord.classify(
            ordinal=item.ordinal,
            dataset=item.dataset,
            sample_id=item.sample_id,
            findings=findings,
            audit_views=attempted_views,
            historical_signatures=historical_signatures,
            elapsed_seconds=time.monotonic() - started,
            source_hashes=self.source_hashes,
            validator_versions=self.validator_versions,
        )


def load_cached_source_ids(dataset: DatasetName, cache_path: Path) -> tuple[str, ...]:
    """Load and validate the immutable reader-index payload without scanning roots."""
    with cache_path.open("rb") as handle:
        payload: Any = pickle.load(handle)
    if not isinstance(payload, Mapping) or "index" not in payload:
        raise ValueError(f"invalid {dataset.value} reader-index cache payload")
    index = payload["index"]
    paths: object
    if dataset is DatasetName.QBIOLIP:
        paths = index
    elif isinstance(index, tuple) and len(index) == 2:
        paths = index[0]
    else:
        raise ValueError("AFDB reader-index cache must contain paths and sequences")
    if not isinstance(paths, Mapping):
        raise ValueError(f"{dataset.value} reader-index paths must be a mapping")
    ids = tuple(paths)
    if (
        len(ids) != 10_000
        or len(set(ids)) != 10_000
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
    ):
        raise ValueError(
            f"{dataset.value} cache must contain exactly 10,000 unique IDs"
        )
    return ids


__all__ = [
    "PreflightFn",
    "ScreeningDataset",
    "ScreeningItem",
    "build_screening_items",
    "identity_collate",
    "load_cached_source_ids",
]
