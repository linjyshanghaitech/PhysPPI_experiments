from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

SPECIFIC_CONTACT_TYPES = (
    "hbond",
    "saltbridge",
    "hydrophobic",
    "pistacking",
    "pication",
    "waterbridge",
    "halogen",
)


def _load_case(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if len(payload) != 1:
        raise ValueError(f"{path} must contain one case, got {len(payload)}")
    sample_id = next(iter(payload))
    return sample_id, payload[sample_id]


def census(pkl_dir: Path) -> dict:
    pkl_paths = sorted(pkl_dir.glob("*.pkl"))
    sample_count = 0
    samples_with_specific = 0
    chain_count = 0
    chains_with_specific = 0
    type_counts: Counter[str] = Counter()
    examples: list[dict] = []

    for path in pkl_paths:
        sample_id, record = _load_case(path)
        sample_count += 1
        sample_has_specific = False
        residue = getattr(record, "residue", {})
        for chain_id, chain_payload in residue.items():
            chain_count += 1
            specific = chain_payload.get("specific_contacts")
            if not isinstance(specific, dict) or not specific:
                continue
            chains_with_specific += 1
            sample_has_specific = True
            for partner_chain_id, rows in specific.items():
                if not rows:
                    continue
                if len(examples) < 20:
                    examples.append(
                        {
                            "sample_id": sample_id,
                            "chain_id": str(chain_id),
                            "partner_chain_id": str(partner_chain_id),
                            "first": rows[0],
                        }
                    )
                for row in rows:
                    contact_type = str(row.get("t", ""))
                    type_counts[contact_type] += 1
        if sample_has_specific:
            samples_with_specific += 1

    unknown_types = sorted(
        key for key in type_counts if key not in SPECIFIC_CONTACT_TYPES
    )
    return {
        "pkl_dir": str(pkl_dir),
        "sample_count": sample_count,
        "samples_with_specific": samples_with_specific,
        "chain_count": chain_count,
        "chains_with_specific": chains_with_specific,
        "type_counts": dict(type_counts),
        "unknown_types": unknown_types,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = census(args.pkl_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    if result["sample_count"] == 0:
        raise SystemExit("DATA_NOT_READY: no PKLs found")
    if result["samples_with_specific"] == 0:
        raise SystemExit("DATA_NOT_READY: zero PKLs contain specific_contacts")
    if result["unknown_types"]:
        raise SystemExit(
            "DATA_SCHEMA_FAIL: unknown specific contact types "
            f"{result['unknown_types']}"
        )


if __name__ == "__main__":
    main()
