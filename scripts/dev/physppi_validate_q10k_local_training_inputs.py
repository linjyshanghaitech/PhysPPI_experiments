from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_split_ids(split_dir: Path) -> list[str]:
    ids: list[str] = []
    for name in ["train_ids.json", "val_ids.json", "test_ids.json"]:
        path = split_dir / name
        if path.is_file():
            ids.extend(str(item) for item in _load_json(path))
    return ids


def _chain_map(frozen: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for case in frozen["cases"]:
        case_id = str(case["case_id"])
        for chain in case["chains"]:
            rows[(case_id, str(chain["chain_id"]))] = {
                "sequence": str(chain.get("sequence") or ""),
                "main_msa_path": str(chain["main_msa_path"]),
                "paired_msa_path": str(chain["paired_msa_path"]),
            }
    return rows


def validate_training_inputs(
    *,
    frozen_json: Path,
    query_jsons: list[Path],
    split_dir: Path,
    require_existing_files: bool,
) -> dict[str, Any]:
    frozen = _load_json(frozen_json)
    frozen_ids = set(str(item) for item in frozen["case_ids"])
    split_ids = _load_split_ids(split_dir)
    query_ids: list[str] = []
    errors: list[str] = []
    if len(split_ids) != len(set(split_ids)):
        errors.append("duplicate_split_ids")
    for case_id in split_ids:
        if case_id not in frozen_ids:
            errors.append(f"split_id_not_in_frozen:{case_id}")

    chains = _chain_map(frozen)
    for query_json in query_jsons:
        payload = _load_json(query_json)
        for case_id, query in payload.get("queries", {}).items():
            case_id = str(case_id)
            query_ids.append(case_id)
            if case_id not in frozen_ids:
                errors.append(f"query_id_not_in_frozen:{case_id}")
            for chain in query.get("chains", []):
                chain_ids = chain.get("chain_ids", [])
                if len(chain_ids) != 1:
                    errors.append(f"query_chain_id_count:{case_id}")
                    continue
                chain_id = str(chain_ids[0])
                expected = chains.get((case_id, chain_id))
                if expected is None:
                    errors.append(f"query_chain_not_in_frozen:{case_id}:{chain_id}")
                    continue
                sequence = str(chain.get("sequence") or "")
                if not sequence:
                    errors.append(f"query_missing_sequence:{case_id}:{chain_id}")
                elif sequence != expected["sequence"]:
                    errors.append(f"query_sequence_mismatch:{case_id}:{chain_id}")
                if chain.get("main_msa_file_paths", []) != [expected["main_msa_path"]]:
                    errors.append(f"query_main_msa_mismatch:{case_id}:{chain_id}")
                if chain.get("paired_msa_file_paths", []) != [
                    expected["paired_msa_path"]
                ]:
                    errors.append(f"query_paired_msa_mismatch:{case_id}:{chain_id}")

    if require_existing_files:
        for case in frozen["cases"]:
            for field in ["total_pkl", "gt_multimer"]:
                if not Path(case[field]).is_file():
                    errors.append(f"missing_frozen_file:{case['case_id']}:{field}")
            for chain in case["chains"]:
                if not chain.get("sequence"):
                    errors.append(
                        f"missing_frozen_sequence:{case['case_id']}:{chain['chain_id']}"
                    )
                for field in ["main_msa_path", "paired_msa_path"]:
                    if not Path(chain[field]).is_file():
                        errors.append(
                            f"missing_frozen_file:{case['case_id']}:{chain['chain_id']}:{field}"
                        )

    return {
        "status": "ok" if not errors else "failed",
        "frozen_json": str(frozen_json),
        "split_dir": str(split_dir),
        "query_jsons": [str(path) for path in query_jsons],
        "frozen_case_count": len(frozen_ids),
        "split_case_count": len(set(split_ids)),
        "query_case_count": len(set(query_ids)),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-json", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, action="append", required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--allow-missing-files", action="store_true")
    args = parser.parse_args()
    report = validate_training_inputs(
        frozen_json=args.frozen_json,
        query_jsons=args.query_json,
        split_dir=args.split_dir,
        require_existing_files=not args.allow_missing_files,
    )
    _write_json(args.output_report, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
