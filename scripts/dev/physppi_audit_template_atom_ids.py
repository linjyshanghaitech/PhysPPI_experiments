#!/usr/bin/env python3
"""Read-only manifest-scoped audit for nonincreasing template PDB atom IDs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from biotite.structure.io.pdb.hybrid36 import decode_hybrid36


def _nonincreasing_atom_ids(path: Path) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    previous_atom_id: int | None = None
    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_id = decode_hybrid36(line[6:11])
            if previous_atom_id is not None and atom_id <= previous_atom_id:
                records.append(
                    {
                        "line": line_number,
                        "previous_atom_id": previous_atom_id,
                        "atom_id": atom_id,
                    }
                )
            previous_atom_id = atom_id
    return records


def audit_dataset(root: Path, sample_ids: list[str]) -> dict[str, object]:
    selected = set(sample_ids)
    found: set[str] = set()
    invalid_atom_ids: list[dict[str, object]] = []
    skipped_sample_ids: set[str] = set()
    for chunk in sorted((root / "chunk").glob("chunk*")):
        found.update(
            prior_path.stem
            for prior_path in (chunk / "prior_pkl").glob("*.pkl")
            if prior_path.stem in selected
        )
        for template_path in sorted((chunk / "bb_input").glob("*.pdb")):
            sample_id = template_path.stem.rsplit("_", 1)[0]
            if sample_id not in selected:
                continue
            for record in _nonincreasing_atom_ids(template_path):
                invalid_atom_ids.append(
                    {
                        "sample_id": sample_id,
                        "path": str(template_path),
                        **record,
                    }
                )
                skipped_sample_ids.add(sample_id)
    return {
        "selected": len(selected),
        "found": len(found),
        "missing_ids": sorted(selected - found),
        "skipped_sample_ids": sorted(skipped_sample_ids),
        "invalid_atom_ids": invalid_atom_ids,
    }


def build_filtered_manifest(
    manifest: dict[str, object], skipped_sample_ids: list[str]
) -> dict[str, object]:
    filtered = copy.deepcopy(manifest)
    qbiolip = filtered["datasets"]["qbiolip"]
    skipped = set(skipped_sample_ids)
    qbiolip["train_ids"] = [
        sample_id for sample_id in qbiolip["train_ids"] if sample_id not in skipped
    ]
    qbiolip["fatal_ids"] = sorted(set(qbiolip["fatal_ids"]) | skipped)
    runtime_skips = filtered.setdefault("runtime_skips", [])
    recorded = {
        (entry.get("dataset"), entry.get("sample_id")) for entry in runtime_skips
    }
    for sample_id in sorted(skipped):
        key = ("qbiolip", sample_id)
        if key in recorded:
            continue
        runtime_skips.append(
            {
                "dataset": "qbiolip",
                "sample_id": sample_id,
                "reason": "template PDB atom IDs are not strictly increasing",
                "source_run": "one-time-template-atom-id-audit",
            }
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qbiolip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filtered-manifest-output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    report = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "qbiolip": audit_dataset(
            args.qbiolip_root,
            manifest["datasets"]["qbiolip"]["train_ids"],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    filtered_manifest = build_filtered_manifest(
        manifest, report["qbiolip"]["skipped_sample_ids"]
    )
    args.filtered_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    filtered_temporary = args.filtered_manifest_output.with_suffix(
        args.filtered_manifest_output.suffix + ".tmp"
    )
    filtered_temporary.write_text(
        json.dumps(filtered_manifest, indent=2, sort_keys=True) + "\n"
    )
    filtered_temporary.replace(args.filtered_manifest_output)


if __name__ == "__main__":
    main()
