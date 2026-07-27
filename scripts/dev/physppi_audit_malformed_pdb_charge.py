#!/usr/bin/env python3
"""Read-only manifest-scoped audit for malformed PDB charge columns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _invalid_charge_fields(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            raw = line.rstrip("\r\n")[78:80]
            if raw == "  ":
                continue
            normalized = raw if raw and raw[0] in "+-" else raw[::-1]
            try:
                int(normalized)
            except ValueError:
                records.append(
                    {
                        "line": line_number,
                        "raw_charge": raw,
                        "normalized_charge": normalized,
                    }
                )
    return records


def audit_dataset(root: Path, sample_ids: list[str]) -> dict[str, object]:
    selected = set(sample_ids)
    records: list[dict[str, object]] = []
    found: set[str] = set()
    for prior_path in sorted((root / "chunk").glob("chunk*/prior_pkl/*.pkl")):
        sample_id = prior_path.stem
        if sample_id not in selected:
            continue
        found.add(sample_id)
        gt_path = prior_path.parents[1] / "gt" / f"{sample_id}.pdb"
        for record in _invalid_charge_fields(gt_path):
            records.append(
                {"sample_id": sample_id, "path": str(gt_path), **record}
            )
    return {
        "selected": len(selected),
        "found": len(found),
        "missing_ids": sorted(selected - found),
        "invalid_charge_fields": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qbiolip-root", type=Path, required=True)
    parser.add_argument("--afdb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    datasets = manifest["datasets"]
    report = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "datasets": {
            "qbiolip": audit_dataset(
                args.qbiolip_root, datasets["qbiolip"]["train_ids"]
            ),
            "afdb": audit_dataset(
                args.afdb_root, datasets["afdb"]["train_ids"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
