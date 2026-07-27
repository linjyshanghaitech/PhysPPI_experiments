"""Create deterministic fatal-exclusion manifests for formal PhysPPI training."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

_DATASET_READERS = {
    "QBioLiP10kDataset": (
        "qbiolip",
        "openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k",
        "QBioLiP10kReader",
    ),
    "AFDB10kDataset": (
        "afdb",
        "openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k",
        "AFDB10kReader",
    ),
}


def _fatal_ids(audit_paths: Iterable[Path], dataset_name: str) -> list[str]:
    fatal: set[str] = set()
    dataset_found = False
    for audit_path in audit_paths:
        payload = json.loads(audit_path.read_text())
        datasets = payload.get("datasets") if isinstance(payload, Mapping) else None
        if not isinstance(datasets, Mapping):
            raise ValueError(f"datasets must be a mapping in {audit_path}")
        if dataset_name not in datasets:
            continue
        dataset_found = True
        dataset = datasets[dataset_name]
        if not isinstance(dataset, Mapping):
            raise ValueError(f"invalid {dataset_name} audit in {audit_path}")
        fatal_entry = dataset.get("fatal")
        if not isinstance(fatal_entry, Mapping):
            raise ValueError(f"invalid fatal audit for {dataset_name} in {audit_path}")
        ids = fatal_entry.get("ids")
        if not isinstance(ids, list) or not all(
            isinstance(value, str) for value in ids
        ):
            raise ValueError(f"invalid fatal IDs for {dataset_name} in {audit_path}")
        fatal.update(ids)
    if not dataset_found:
        raise ValueError(f"{dataset_name} not found in any audit JSON")
    return sorted(fatal)


def build_manifests(
    *,
    audit_paths: Iterable[Path],
    ids: Mapping[str, Iterable[str]],
    diagnostic_counts: Mapping[str, int],
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    """Return fatal-only training and diagnostic ID partitions.

    Diagnostics are a deterministic subset of the trainable population, rather
    than a held-out partition, so they remain in ``train_ids`` by design.
    """

    audit_paths = tuple(Path(path) for path in audit_paths)
    result: dict[str, dict[str, list[str]]] = {}
    for dataset_name in sorted(ids):
        all_ids = sorted(set(ids[dataset_name]))
        if not all(isinstance(sample_id, str) for sample_id in all_ids):
            raise ValueError(f"reader IDs for {dataset_name} must be strings")
        diagnostic_count = diagnostic_counts.get(dataset_name, 0)
        if diagnostic_count < 0:
            raise ValueError("diagnostic counts must be non-negative")
        fatal_ids = _fatal_ids(audit_paths, dataset_name)
        train_ids = sorted(set(all_ids) - set(fatal_ids))
        if diagnostic_count > len(train_ids):
            raise ValueError(
                f"requested {diagnostic_count} diagnostics from {dataset_name}, "
                f"but only {len(train_ids)} trainable IDs exist"
            )
        diagnostic_ids = sorted(
            random.Random(f"{seed}:{dataset_name}").sample(train_ids, diagnostic_count)
        )
        result[dataset_name] = {
            "fatal_ids": fatal_ids,
            "train_ids": train_ids,
            "diagnostic_ids": diagnostic_ids,
        }
    return result


def _discover_reader_ids(runner_yaml: Path) -> dict[str, list[str]]:
    config = yaml.safe_load(runner_yaml.read_text())
    entries = config.get("dataset_configs", {})
    discovered: dict[str, list[str]] = {}
    for mode_entries in entries.values():
        if not isinstance(mode_entries, Mapping):
            continue
        for entry in mode_entries.values():
            if not isinstance(entry, Mapping):
                continue
            reader_spec = _DATASET_READERS.get(entry.get("dataset_class"))
            if reader_spec is None:
                continue
            dataset_name, module_name, reader_name = reader_spec
            root = entry.get("config", {}).get("physppi", {}).get("root")
            if root is None:
                raise ValueError(f"{dataset_name} entry has no physppi.root")
            module = __import__(module_name, fromlist=[reader_name])
            reader = getattr(module, reader_name)(Path(root))
            sample_ids = list(reader.sample_ids())
            prior = discovered.setdefault(dataset_name, sample_ids)
            if prior != sample_ids:
                raise ValueError(
                    f"runner YAML has inconsistent {dataset_name} readers across modes"
                )
    missing = sorted({"qbiolip", "afdb"} - set(discovered))
    if missing:
        raise ValueError(f"runner YAML is missing canonical datasets: {missing}")
    return discovered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", action="append", required=True, type=Path)
    parser.add_argument("--runner-yaml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qbiolip-diagnostic-count", type=int, default=205)
    parser.add_argument("--afdb-diagnostic-count", type=int, default=51)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    datasets = build_manifests(
        audit_paths=args.audit_json,
        ids=_discover_reader_ids(args.runner_yaml),
        diagnostic_counts={
            "qbiolip": args.qbiolip_diagnostic_count,
            "afdb": args.afdb_diagnostic_count,
        },
        seed=args.seed,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "seed": args.seed,
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
