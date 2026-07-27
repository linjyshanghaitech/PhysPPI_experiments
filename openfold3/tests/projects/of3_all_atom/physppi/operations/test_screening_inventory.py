from __future__ import annotations

import hashlib
import json
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openfold3.projects.of3_all_atom.physppi.screening.inventory import (
    build_source_inventory,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import DatasetName


def _fake_qbiolip_index() -> dict[str, str]:
    return {f"qbiolip-{index:05d}": f"path-{index}" for index in range(10_000)}


def _fake_afdb_index() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    paths = {f"afdb-{index:05d}": f"path-{index}" for index in range(10_000)}
    sequences = {sample_id: {"A": "AA", "B": "AA"} for sample_id in paths}
    return paths, sequences


def test_inventory_reuses_one_read_only_cache_and_preserves_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "qbiolip"
    root.mkdir()
    calls = 0

    def builder() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return _fake_qbiolip_index()

    first = build_source_inventory(
        DatasetName.QBIOLIP,
        root,
        tmp_path / "cache",
        builder=builder,
        implementation_git_head="test-head",
    )
    second = build_source_inventory(
        DatasetName.QBIOLIP,
        root,
        tmp_path / "cache",
        builder=builder,
        implementation_git_head="test-head",
    )

    assert first == second
    assert first.ids == tuple(_fake_qbiolip_index())
    assert len(first.ids) == len(set(first.ids)) == 10_000
    assert first.cache_path.name == "reader-index.pkl"
    assert first.cache_path.parent.name == first.fingerprint
    assert (
        first.cache_sha256 == hashlib.sha256(first.cache_path.read_bytes()).hexdigest()
    )
    assert stat.S_IMODE(first.cache_path.stat().st_mode) == 0o444
    assert calls == 1

    metadata = json.loads((first.cache_path.parent / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["fingerprint"] == first.fingerprint
    assert metadata["cache_sha256"] == first.cache_sha256
    assert metadata["id_count"] == 10_000


def test_inventory_fingerprint_changes_with_implementation_head(
    tmp_path: Path,
) -> None:
    root = tmp_path / "qbiolip"
    root.mkdir()

    first = build_source_inventory(
        "qbiolip",
        root,
        tmp_path / "cache",
        builder=_fake_qbiolip_index,
        implementation_git_head="head-a",
    )
    second = build_source_inventory(
        "qbiolip",
        root,
        tmp_path / "cache",
        builder=_fake_qbiolip_index,
        implementation_git_head="head-b",
    )

    assert first.fingerprint != second.fingerprint
    assert first.cache_path != second.cache_path


def test_afdb_inventory_preserves_reader_payload_order(tmp_path: Path) -> None:
    root = tmp_path / "afdb"
    root.mkdir()

    inventory = build_source_inventory(
        DatasetName.AFDB,
        root,
        tmp_path / "cache",
        builder=_fake_afdb_index,
        implementation_git_head="test-head",
    )

    assert inventory.ids == tuple(_fake_afdb_index()[0])
    assert len(inventory.ids) == len(set(inventory.ids)) == 10_000


@pytest.mark.parametrize("count", [9_999, 10_001])
def test_inventory_rejects_noncanonical_source_count(
    tmp_path: Path, count: int
) -> None:
    root = tmp_path / "qbiolip"
    root.mkdir()

    cache_root = tmp_path / "cache"
    with pytest.raises(ValueError, match="exactly 10,000 unique ordered IDs"):
        build_source_inventory(
            "qbiolip",
            root,
            cache_root,
            builder=lambda: {
                f"qbiolip-{index:05d}": f"path-{index}" for index in range(count)
            },
            implementation_git_head="test-head",
        )
    assert not list(cache_root.rglob("reader-index.pkl"))


def test_inventory_lock_allows_only_one_concurrent_build(tmp_path: Path) -> None:
    root = tmp_path / "qbiolip"
    root.mkdir()
    calls = 0
    calls_lock = threading.Lock()

    def builder() -> dict[str, str]:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return _fake_qbiolip_index()

    def build():
        return build_source_inventory(
            "qbiolip",
            root,
            tmp_path / "cache",
            builder=builder,
            implementation_git_head="test-head",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        inventories = list(executor.map(lambda _: build(), range(4)))

    assert calls == 1
    assert {inventory.cache_sha256 for inventory in inventories} == {
        inventories[0].cache_sha256
    }
    assert {inventory.cache_path for inventory in inventories} == {
        inventories[0].cache_path
    }
