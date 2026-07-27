from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path


def _load_case(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if len(payload) != 1:
        raise ValueError(f"{path} must contain one case, got {len(payload)}")
    sample_id = next(iter(payload))
    return sample_id, payload[sample_id]


def _count_ones(matrix) -> int:
    total = 0
    for row in matrix:
        total += sum(1 for value in row if int(value) != 0)
    return total


def audit(pkl_dir: Path) -> dict:
    record_attrs: Counter[str] = Counter()
    residue_keys: Counter[str] = Counter()
    cmap_keys: Counter[str] = Counter()
    mesh_widths: Counter[int] = Counter()
    samples_with_cmap_positive = 0
    total_cmap_pairs = 0
    total_cmap_positive = 0
    examples = []

    pkl_paths = sorted(pkl_dir.glob("*.pkl"))
    for path in pkl_paths:
        sample_id, record = _load_case(path)
        for key in getattr(record, "__dict__", {}):
            record_attrs[str(key)] += 1

        sample_positive = 0
        mesh = getattr(record, "mesh", {})
        for vertices in mesh.values():
            if vertices:
                mesh_widths[len(vertices[0])] += 1

        residue = getattr(record, "residue", {})
        for chain_id, chain_payload in residue.items():
            if not isinstance(chain_payload, dict):
                continue
            for key in chain_payload:
                residue_keys[str(key)] += 1
            cmap = chain_payload.get("cmap", {})
            if not isinstance(cmap, dict):
                continue
            for partner_chain_id, pair_payload in cmap.items():
                if not isinstance(pair_payload, dict):
                    continue
                for key in pair_payload:
                    cmap_keys[str(key)] += 1
                contact = pair_payload.get("contact")
                if contact is None:
                    continue
                positive = _count_ones(contact)
                total_cmap_pairs += 1
                total_cmap_positive += positive
                sample_positive += positive
                if positive and len(examples) < 20:
                    examples.append(
                        {
                            "sample_id": sample_id,
                            "chain_id": str(chain_id),
                            "partner_chain_id": str(partner_chain_id),
                            "positive_contacts": positive,
                        }
                    )
        if sample_positive:
            samples_with_cmap_positive += 1

    return {
        "pkl_dir": str(pkl_dir),
        "sample_count": len(pkl_paths),
        "record_attrs": dict(record_attrs),
        "residue_keys": dict(residue_keys),
        "cmap_keys": dict(cmap_keys),
        "mesh_widths": dict(mesh_widths),
        "samples_with_cmap_positive": samples_with_cmap_positive,
        "total_cmap_pairs": total_cmap_pairs,
        "total_cmap_positive": total_cmap_positive,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = audit(args.pkl_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    if result["sample_count"] == 0:
        raise SystemExit("DATA_NOT_READY: no PKLs found")
    if result["samples_with_cmap_positive"] == 0:
        raise SystemExit("DATA_NOT_READY: no positive cmap contacts found")
    mesh_widths = {int(key) for key in result["mesh_widths"]}
    if mesh_widths != {10}:
        print("WARN: mesh widths are not uniformly 10; inspect output JSON")


if __name__ == "__main__":
    main()
