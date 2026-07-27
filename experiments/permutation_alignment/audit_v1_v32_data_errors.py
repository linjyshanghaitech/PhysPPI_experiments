#!/usr/bin/env python3
"""Aggregate v1-v32 PhysPPI data failures and scan both complete 10k sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biotite.structure.io.pdb.hybrid36 import decode_hybrid36

from openfold3.projects.of3_all_atom.physppi.screening.history import (
    _load_runtime_ledger,
    _runtime_category,
    load_manifest_runtime_skips,
    load_versioned_log_sample_ids,
)

__all__ = [
    "load_manifest_runtime_skips",
    "load_versioned_log_sample_ids",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _charge_is_invalid(raw: str) -> bool:
    if not raw.strip():
        return False
    normalized = raw if raw[0] in "+-" else raw[::-1]
    try:
        int(normalized)
    except ValueError:
        return True
    return False


def scan_pdb(path: Path, *, surface_neighbors: int) -> dict[str, Any]:
    """Scan fixed-width records without invoking a structure parser."""

    atom_records = 0
    invalid_charge_fields = 0
    nonincreasing_atom_ids = 0
    previous_atom_id: int | None = None
    residue_names: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_records += 1
            raw_charge = line.rstrip("\r\n")[78:80]
            invalid_charge_fields += int(_charge_is_invalid(raw_charge))

            try:
                atom_id = decode_hybrid36(line[6:11])
            except ValueError:
                atom_id = None
            if atom_id is not None:
                if previous_atom_id is not None and atom_id <= previous_atom_id:
                    nonincreasing_atom_ids += 1
                previous_atom_id = atom_id

            residue_key = (line[21:22], line[22:26], line[26:27])
            residue_names[residue_key].add(line[17:20].strip())

    token_count = len(residue_names)
    microheterogeneous = sum(len(names) > 1 for names in residue_names.values())
    return {
        "atom_record_count": atom_records,
        "token_count": token_count,
        "invalid_charge_field_count": invalid_charge_fields,
        "nonincreasing_atom_id_count": nonincreasing_atom_ids,
        "microheterogeneous_residue_count": microheterogeneous,
        "surface_neighbor_axis_collision": token_count == surface_neighbors,
    }


def category_payload(
    *,
    qbiolip_ids: Iterable[str],
    afdb_ids: Iterable[str],
    source: str,
    severity: str,
) -> dict[str, Any]:
    qbiolip = sorted(set(qbiolip_ids))
    afdb = sorted(set(afdb_ids))
    return {
        "source": source,
        "severity": severity,
        "qbiolip_ids": qbiolip,
        "afdb_ids": afdb,
        "counts": {
            "qbiolip": len(qbiolip),
            "afdb": len(afdb),
            "total": len(qbiolip) + len(afdb),
        },
    }


def _canonical_fatal_ids(path: Path, dataset: str) -> list[str]:
    report = json.loads(path.read_text())
    return list(report["datasets"][dataset]["fatal"]["ids"])


def _load_manifest_ids(path: Path) -> dict[str, set[str]]:
    manifest = json.loads(path.read_text())
    return {
        dataset: set(manifest["datasets"][dataset]["train_ids"])
        for dataset in ("qbiolip", "afdb")
    }


def _template_paths(root: Path, selected_ids: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((root / "chunk").glob("chunk*/bb_input/*.pdb")):
        sample_id = path.stem.rsplit("_", 1)[0]
        if sample_id in selected_ids:
            result[sample_id].append(path)
    return result


def _add_manifest_eligible_counts(
    categories: dict[str, dict[str, Any]],
    manifest_ids: dict[str, set[str]],
) -> None:
    for payload in categories.values():
        counts = {}
        for dataset in ("qbiolip", "afdb"):
            ids = payload[f"{dataset}_ids"]
            counts[dataset] = sum(
                sample_id in manifest_ids[dataset] for sample_id in ids
            )
        counts["total"] = counts["qbiolip"] + counts["afdb"]
        payload["manifest_eligible_counts"] = counts


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import (
        AFDB10kReader,
    )
    from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
        QBioLiP10kReader,
    )

    readers = {
        "qbiolip": QBioLiP10kReader(
            args.qbiolip_root, index_cache_file=args.qbiolip_index_cache
        ),
        "afdb": AFDB10kReader(args.afdb_root, index_cache_file=args.afdb_index_cache),
    }
    ids = {dataset: list(reader.sample_ids()) for dataset, reader in readers.items()}
    if {dataset: len(values) for dataset, values in ids.items()} != {
        "qbiolip": 10000,
        "afdb": 10000,
    }:
        raise ValueError("complete scan requires exactly 10,000 IDs per source")

    malformed_charge = {"qbiolip": [], "afdb": []}
    nonincreasing_gt = {"qbiolip": [], "afdb": []}
    surface_collision = {"qbiolip": [], "afdb": []}
    microheterogeneous = {"qbiolip": [], "afdb": []}
    missing_gt = {"qbiolip": [], "afdb": []}
    details: dict[str, dict[str, dict[str, int]]] = {
        "qbiolip": {},
        "afdb": {},
    }

    for dataset, reader in readers.items():
        for index, sample_id in enumerate(ids[dataset], 1):
            gt_pdb = reader.locate(sample_id).gt_pdb
            if not gt_pdb.is_file():
                missing_gt[dataset].append(sample_id)
                continue
            stats = scan_pdb(gt_pdb, surface_neighbors=args.surface_neighbors)
            anomaly_stats = {
                key: int(value)
                for key, value in stats.items()
                if key.endswith("_count") and int(value) > 0
            }
            if stats["invalid_charge_field_count"]:
                malformed_charge[dataset].append(sample_id)
            if stats["nonincreasing_atom_id_count"]:
                nonincreasing_gt[dataset].append(sample_id)
            if stats["surface_neighbor_axis_collision"]:
                surface_collision[dataset].append(sample_id)
                anomaly_stats["token_count"] = stats["token_count"]
            if stats["microheterogeneous_residue_count"]:
                microheterogeneous[dataset].append(sample_id)
            if anomaly_stats:
                details[dataset][sample_id] = anomaly_stats
            if index % 1000 == 0:
                print(f"{dataset}: scanned {index}/10000 GT PDBs", file=sys.stderr)

    nonincreasing_template: list[str] = []
    template_event_counts: dict[str, int] = {}
    template_paths = _template_paths(args.qbiolip_root, set(ids["qbiolip"]))
    for index, sample_id in enumerate(ids["qbiolip"], 1):
        event_count = 0
        for path in template_paths.get(sample_id, []):
            event_count += scan_pdb(path, surface_neighbors=args.surface_neighbors)[
                "nonincreasing_atom_id_count"
            ]
        if event_count:
            nonincreasing_template.append(sample_id)
            template_event_counts[sample_id] = event_count
        if index % 1000 == 0:
            print(
                f"qbiolip: scanned templates for {index}/10000 IDs",
                file=sys.stderr,
            )

    runtime_entries = _load_runtime_ledger(args.runtime_ledger)
    runtime_label = _runtime_category(
        runtime_entries,
        (
            "scatter-gather",
            "token-center",
            "frame-start",
            "label",
        ),
    )
    runtime_surface = _runtime_category(
        runtime_entries, ("surface-local-coordinates-shape",)
    )

    categories = {
        "canonical_residue_mapping_fatal": category_payload(
            qbiolip_ids=_canonical_fatal_ids(args.qbiolip_canonical_report, "qbiolip"),
            afdb_ids=_canonical_fatal_ids(args.afdb_canonical_report, "afdb"),
            source="canonical_prior_full_10k_audits",
            severity="fatal_data",
        ),
        "malformed_gt_pdb_charge": category_payload(
            qbiolip_ids=malformed_charge["qbiolip"],
            afdb_ids=malformed_charge["afdb"],
            source="full_20k_fixed_width_gt_pdb_scan",
            severity="recoverable_data",
        ),
        "nonincreasing_gt_atom_ids": category_payload(
            qbiolip_ids=nonincreasing_gt["qbiolip"],
            afdb_ids=nonincreasing_gt["afdb"],
            source="full_20k_gt_pdb_scan",
            severity="fatal_data",
        ),
        "nonincreasing_template_atom_ids": category_payload(
            qbiolip_ids=nonincreasing_template,
            afdb_ids=[],
            source="full_qbiolip_10k_template_scan",
            severity="fatal_data",
        ),
        "surface_neighbor_axis_collision": category_payload(
            qbiolip_ids=surface_collision["qbiolip"],
            afdb_ids=surface_collision["afdb"],
            source="full_20k_gt_token_count_scan",
            severity="data_triggered_code_boundary",
        ),
        "microheterogeneous_gt_residues": category_payload(
            qbiolip_ids=microheterogeneous["qbiolip"],
            afdb_ids=microheterogeneous["afdb"],
            source="full_20k_gt_pdb_scan",
            severity="warning_data",
        ),
        "missing_gt_pdb": category_payload(
            qbiolip_ids=missing_gt["qbiolip"],
            afdb_ids=missing_gt["afdb"],
            source="full_20k_reader_index_scan",
            severity="fatal_data",
        ),
        "historical_runtime_label_index_failures": category_payload(
            qbiolip_ids=runtime_label["qbiolip"],
            afdb_ids=runtime_label["afdb"],
            source="v1_v32_runtime_failure_ledger",
            severity="data_triggered_code_boundary",
        ),
        "historical_runtime_surface_shape_failures": category_payload(
            qbiolip_ids=runtime_surface["qbiolip"],
            afdb_ids=runtime_surface["afdb"],
            source="v1_v32_runtime_failure_ledger",
            severity="data_triggered_code_boundary",
        ),
    }
    manifest_ids = _load_manifest_ids(args.training_manifest)
    _add_manifest_eligible_counts(categories, manifest_ids)

    union_by_dataset = {"qbiolip": set(), "afdb": set()}
    for name, payload in categories.items():
        if name == "microheterogeneous_gt_residues":
            continue
        for dataset in union_by_dataset:
            union_by_dataset[dataset].update(payload[f"{dataset}_ids"])

    sources = {
        "training_manifest": args.training_manifest,
        "qbiolip_canonical_report": args.qbiolip_canonical_report,
        "afdb_canonical_report": args.afdb_canonical_report,
        "runtime_ledger": args.runtime_ledger,
        "qbiolip_index_cache": args.qbiolip_index_cache,
        "afdb_index_cache": args.afdb_index_cache,
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scan_scope": {
            "qbiolip": len(ids["qbiolip"]),
            "afdb": len(ids["afdb"]),
            "total": len(ids["qbiolip"]) + len(ids["afdb"]),
            "training_manifest_eligible": {
                dataset: len(values) for dataset, values in manifest_ids.items()
            },
            "surface_neighbors": args.surface_neighbors,
        },
        "source_evidence": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "categories": categories,
        "error_union_excluding_microheterogeneity_warning": category_payload(
            qbiolip_ids=union_by_dataset["qbiolip"],
            afdb_ids=union_by_dataset["afdb"],
            source="deduplicated_category_union",
            severity="mixed",
        ),
        "anomaly_metrics_by_id": details,
        "template_nonincreasing_event_counts": dict(
            sorted(template_event_counts.items())
        ),
        "runtime_failure_ledger_entries": runtime_entries,
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qbiolip-root", type=Path, required=True)
    parser.add_argument("--afdb-root", type=Path, required=True)
    parser.add_argument("--qbiolip-index-cache", type=Path, required=True)
    parser.add_argument("--afdb-index-cache", type=Path, required=True)
    parser.add_argument("--qbiolip-canonical-report", type=Path, required=True)
    parser.add_argument("--afdb-canonical-report", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--runtime-ledger", type=Path, required=True)
    parser.add_argument("--surface-neighbors", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_scan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
