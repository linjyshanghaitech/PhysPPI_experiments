from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SAMPLE_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}_[0-9]+$")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_id_list(path: Path, *, label: str) -> list[str]:
    value = _load_json(path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"INPUT_FAIL: {label} must be a JSON list of strings: {path}")
    return value


def _iter_case_ids(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if SAMPLE_ID_RE.match(value):
            yield value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and SAMPLE_ID_RE.match(key):
                yield key
            yield from _iter_case_ids(child)
        return
    if isinstance(value, list):
        for child in value:
            yield from _iter_case_ids(child)


def _extract_cache_ids(cache: dict[str, Any]) -> tuple[set[str], str]:
    structure_data = cache.get("structure_data")
    if isinstance(structure_data, dict):
        ids = {key for key in structure_data if SAMPLE_ID_RE.match(str(key))}
        if ids:
            return ids, "structure_data.keys"

    ids = set(_iter_case_ids(cache))
    if ids:
        return ids, "recursive_case_id_scan"
    raise SystemExit("CACHE_FAIL: no sample IDs found in training cache")


def _load_all_cases(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    malformed = [sample_id for sample_id in ids if not SAMPLE_ID_RE.match(sample_id)]
    if malformed:
        raise SystemExit(f"INPUT_FAIL: malformed case IDs in {path}: {malformed[:20]}")
    return ids


def _dedupe(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-cases", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--fixed-test32", type=Path, required=True)
    parser.add_argument("--fixed-train32", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    all_ids = _dedupe(_load_all_cases(args.all_cases))
    cache_ids, cache_id_source = _extract_cache_ids(_load_json(args.training_cache))
    fixed_test32 = _load_id_list(args.fixed_test32, label="fixed-test32")
    fixed_train32 = _load_id_list(args.fixed_train32, label="fixed-train32")

    duplicate_all = len(all_ids) != len(set(all_ids))
    if duplicate_all:
        raise SystemExit("INPUT_FAIL: all-cases contains duplicate sample IDs")

    fixed_overlap = sorted(set(fixed_test32) & set(fixed_train32))
    if fixed_overlap:
        raise SystemExit(
            f"LEAKAGE_FAIL: fixed train/test overlap: {fixed_overlap[:20]}"
        )

    fixed_test_leakage = sorted(set(fixed_test32) & cache_ids)
    if fixed_test_leakage:
        raise SystemExit(
            "LEAKAGE_FAIL: fixed test IDs in training cache: "
            f"{fixed_test_leakage[:20]}"
        )

    all_id_set = set(all_ids)
    unknown_cache_ids = sorted(cache_ids - all_id_set)
    if unknown_cache_ids:
        raise SystemExit(
            "CACHE_FAIL: training cache IDs absent from all-cases: "
            f"{unknown_cache_ids[:20]}"
        )

    fixed_test32_missing_from_all = sorted(set(fixed_test32) - all_id_set)
    fixed_train32_missing_from_all = sorted(set(fixed_train32) - all_id_set)
    fixed_test32_available = [
        sample_id for sample_id in fixed_test32 if sample_id in all_id_set
    ]
    fixed_train32_available = [
        sample_id for sample_id in fixed_train32 if sample_id in all_id_set
    ]

    heldout = [sample_id for sample_id in all_ids if sample_id not in cache_ids]
    rng = random.Random(args.seed)
    rng.shuffle(heldout)

    test_ids = _dedupe(fixed_test32_available + heldout)[: args.test_count]
    train_ids = _dedupe(fixed_train32_available + sorted(cache_ids))[: args.train_count]

    if len(test_ids) < args.test_count:
        raise SystemExit(
            f"EVAL_SET_NOT_READY: only {len(test_ids)} heldout test IDs available"
        )
    if len(train_ids) < args.train_count:
        raise SystemExit(
            f"EVAL_SET_NOT_READY: only {len(train_ids)} train IDs available"
        )

    test_training_overlap = sorted(set(test_ids) & cache_ids)
    if test_training_overlap:
        raise SystemExit(
            "LEAKAGE_FAIL: test set intersects training cache: "
            f"{test_training_overlap[:20]}"
        )
    train_not_in_cache = sorted(set(train_ids) - cache_ids)
    if train_not_in_cache:
        raise SystemExit(
            "LEAKAGE_FAIL: train set contains IDs outside training cache: "
            f"{train_not_in_cache[:20]}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.out_dir / f"test{args.test_count}_sample_ids.json"
    train_path = args.out_dir / f"train{args.train_count}_sample_ids.json"
    report_path = args.out_dir / "eval_set_report.json"

    test_path.write_text(json.dumps(test_ids, indent=2) + "\n")
    train_path.write_text(json.dumps(train_ids, indent=2) + "\n")

    report = {
        "seed": args.seed,
        "cache_id_source": cache_id_source,
        "all_count": len(all_ids),
        "training_cache_id_count": len(cache_ids),
        "heldout_count": len(heldout),
        "fixed_test32_count": len(fixed_test32),
        "fixed_train32_count": len(fixed_train32),
        "fixed_test32_available_in_all_cases_count": len(fixed_test32_available),
        "fixed_train32_available_in_all_cases_count": len(fixed_train32_available),
        "fixed_test32_missing_from_all_cases": fixed_test32_missing_from_all,
        "fixed_train32_missing_from_all_cases": fixed_train32_missing_from_all,
        "fixed_test32_training_overlap": fixed_test_leakage,
        "test_count": len(test_ids),
        "train_count": len(train_ids),
        "test_training_overlap_count": len(test_training_overlap),
        "train_not_in_cache_count": len(train_not_in_cache),
        "test_ids_path": str(test_path),
        "train_ids_path": str(train_path),
        "test_first10": test_ids[:10],
        "train_first10": train_ids[:10],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
