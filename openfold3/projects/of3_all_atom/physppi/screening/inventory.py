"""Fingerprint and freeze canonical PhysPPI reader inventories."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader
from openfold3.projects.of3_all_atom.physppi.data.readers.index_cache import (
    load_or_create_reader_index,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    QBioLiP10kReader,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    SCHEMA_VERSION,
    DatasetName,
)

_EXPECTED_SOURCE_COUNT = 10_000
_READ_ONLY_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


@dataclass(frozen=True, slots=True)
class SourceInventory:
    """One exact reader inventory backed by an immutable shared index."""

    dataset: DatasetName
    root: Path
    ids: tuple[str, ...]
    cache_path: Path
    fingerprint: str
    cache_sha256: str


def _git_head() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _reader_class(dataset: DatasetName) -> type[QBioLiP10kReader] | type[AFDB10kReader]:
    if dataset is DatasetName.QBIOLIP:
        return QBioLiP10kReader
    return AFDB10kReader


def _reader_metadata(dataset: DatasetName, root: Path) -> dict[str, object]:
    return {
        "reader": ("qbiolip10k" if dataset is DatasetName.QBIOLIP else "afdb10k"),
        "root": str(root),
        "max_samples": None,
    }


def _default_builder(dataset: DatasetName, root: Path) -> Callable[[], object]:
    def build_qbiolip() -> object:
        reader = QBioLiP10kReader(root)
        return reader._paths

    def build_afdb() -> object:
        reader = AFDB10kReader(root)
        return reader._paths, reader._chain_sequences

    return build_qbiolip if dataset is DatasetName.QBIOLIP else build_afdb


def _ordered_ids(dataset: DatasetName, index: object) -> tuple[str, ...]:
    paths: object
    if dataset is DatasetName.QBIOLIP:
        paths = index
    elif isinstance(index, tuple) and len(index) == 2:
        paths = index[0]
    else:
        raise ValueError("AFDB reader index must contain paths and chain sequences")
    if not isinstance(paths, Mapping):
        raise ValueError(f"{dataset.value} reader index paths must be a mapping")

    ids = tuple(paths)
    if (
        len(ids) != _EXPECTED_SOURCE_COUNT
        or len(set(ids)) != _EXPECTED_SOURCE_COUNT
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
    ):
        raise ValueError(
            f"{dataset.value} inventory must contain exactly 10,000 unique ordered IDs"
        )
    return ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_source_inventory(
    dataset: DatasetName | str,
    root: str | Path,
    cache_root: str | Path,
    *,
    builder: Callable[[], object] | None = None,
    implementation_git_head: str | None = None,
) -> SourceInventory:
    """Build one canonical reader index, then reuse its fingerprinted cache."""

    dataset = DatasetName(dataset)
    root = Path(root).resolve()
    cache_root = Path(cache_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing {dataset.value} source root: {root}")

    root_stat = root.stat()
    reader_class = _reader_class(dataset)
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset.value,
        "root": str(root),
        "root_stat": {"device": root_stat.st_dev, "inode": root_stat.st_ino},
        "reader_class": reader_class.__qualname__,
        "implementation_git_head": implementation_git_head or _git_head(),
        "max_samples": None,
    }
    canonical_payload = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    fingerprint = hashlib.sha256(canonical_payload).hexdigest()
    inventory_dir = cache_root / dataset.value / fingerprint
    inventory_dir.mkdir(parents=True, exist_ok=True)
    cache_path = inventory_dir / "reader-index.pkl"
    inventory_lock_path = inventory_dir / "inventory.lock"

    with inventory_lock_path.open("a+b") as inventory_lock:
        fcntl.flock(inventory_lock.fileno(), fcntl.LOCK_EX)

        def validated_builder() -> object:
            index = (builder or _default_builder(dataset, root))()
            _ordered_ids(dataset, index)
            return index

        index = load_or_create_reader_index(
            cache_path,
            _reader_metadata(dataset, root),
            validated_builder,
        )
        ids = _ordered_ids(dataset, index)
        cache_sha256 = _sha256(cache_path)
        cache_path.chmod(_READ_ONLY_MODE)
        metadata_path = inventory_dir / "metadata.json"
        _write_json_atomic(
            metadata_path,
            {
                **fingerprint_payload,
                "fingerprint": fingerprint,
                "cache_file": cache_path.name,
                "cache_sha256": cache_sha256,
                "id_count": len(ids),
            },
        )
        metadata_path.chmod(_READ_ONLY_MODE)

    return SourceInventory(
        dataset=dataset,
        root=root,
        ids=ids,
        cache_path=cache_path,
        fingerprint=fingerprint,
        cache_sha256=cache_sha256,
    )


__all__ = ["SourceInventory", "build_source_inventory"]
