#!/usr/bin/env python3
"""Build deterministic train/test evaluation sets for PhysPPI epoch14 sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_IDS = Path(
    "output/physppi_finetune_validation/phase6/full2107/sample_ids_full2107.json"
)
DEFAULT_TRAIN_QUERY = Path("output/physppi_full_msa_cache/query_full_source1.json")
DEFAULT_TEST_QUERY = Path("output/physppi_openfold3_baseline/query_test.json")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _limit_value(raw: str) -> int | None:
    if raw.lower() == "all":
        return None
    value = int(raw)
    if value < 1:
        raise ValueError("limits must be positive integers or 'all'")
    return value


def _query_chains(query: dict[str, Any]) -> list[dict[str, Any]]:
    chains = query.get("chains", [])
    if not isinstance(chains, list):
        return []
    return [chain for chain in chains if chain.get("molecule_type") == "protein"]


def _total_sequence_length(query: dict[str, Any]) -> int:
    total = 0
    for chain in _query_chains(query):
        seq = chain.get("sequence") or ""
        chain_ids = chain.get("chain_ids") or []
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        total += len(seq) * max(len(chain_ids), 1)
    return total


def _is_protein_complex(query: dict[str, Any]) -> bool:
    chain_count = 0
    for chain in _query_chains(query):
        chain_ids = chain.get("chain_ids") or []
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        if chain.get("sequence"):
            chain_count += len(chain_ids)
    return chain_count >= 2


def _evenly_spaced(
    items: list[tuple[str, dict[str, Any]]], limit: int | None
) -> list[tuple[str, dict[str, Any]]]:
    if limit is None or limit >= len(items):
        return items
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    idxs = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    selected = [items[i] for i in idxs]
    seen = {sample_id for sample_id, _ in selected}
    for item in items:
        if len(selected) >= limit:
            break
        if item[0] not in seen:
            selected.append(item)
            seen.add(item[0])
    return sorted(selected, key=lambda x: (_total_sequence_length(x[1]), x[0]))


def _cap_total_length(
    items: list[tuple[str, dict[str, Any]]], max_total_length: int | None
) -> list[tuple[str, dict[str, Any]]]:
    if max_total_length is None:
        return items
    return [
        item for item in items if _total_sequence_length(item[1]) <= max_total_length
    ]


def _select_items(
    items: list[tuple[str, dict[str, Any]]],
    limit: int | None,
    max_total_length: int | None,
    selection_mode: str,
) -> list[tuple[str, dict[str, Any]]]:
    eligible = _cap_total_length(items, max_total_length)
    if selection_mode == "shortest":
        return eligible if limit is None else eligible[:limit]
    if selection_mode == "even":
        return _evenly_spaced(eligible, limit)
    raise ValueError(f"Unknown selection mode: {selection_mode}")


def _select_train(
    train_ids_path: Path,
    train_query_path: Path,
    limit: int | None,
    max_total_length: int | None,
    selection_mode: str,
) -> tuple[list[tuple[str, dict[str, Any]]], list[str], list[str], int]:
    ids = [str(x) for x in _load_json(train_ids_path)]
    queries = _load_json(train_query_path).get("queries", {})
    items = []
    missing_query = []
    not_complex = []
    for sample_id in ids:
        query = queries.get(sample_id)
        if not isinstance(query, dict):
            missing_query.append(sample_id)
            continue
        if not _is_protein_complex(query):
            not_complex.append(sample_id)
            continue
        items.append((sample_id, query))
    items.sort(key=lambda x: (_total_sequence_length(x[1]), x[0]))
    selected = _select_items(items, limit, max_total_length, selection_mode)
    return selected, missing_query, not_complex, len(items)


def _select_test(
    test_query_path: Path,
    limit: int | None,
    max_total_length: int | None,
    selection_mode: str,
) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    queries = _load_json(test_query_path).get("queries", {})
    items = [
        (str(sample_id), query)
        for sample_id, query in queries.items()
        if isinstance(query, dict) and _is_protein_complex(query)
    ]
    items.sort(key=lambda x: (_total_sequence_length(x[1]), x[0]))
    return _select_items(items, limit, max_total_length, selection_mode), len(items)


def _write_query(path: Path, items: list[tuple[str, dict[str, Any]]]) -> None:
    payload = {
        "seeds": [42],
        "queries": {sample_id: query for sample_id, query in items},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_ids(path: Path, items: list[tuple[str, dict[str, Any]]]) -> None:
    path.write_text(json.dumps([sample_id for sample_id, _ in items], indent=2) + "\n")


def _length_min_max(items: list[tuple[str, dict[str, Any]]]) -> list[int]:
    lengths = [_total_sequence_length(query) for _, query in items]
    return [min(lengths), max(lengths)] if lengths else [0, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ids", type=Path, default=DEFAULT_TRAIN_IDS)
    parser.add_argument("--train-query", type=Path, default=DEFAULT_TRAIN_QUERY)
    parser.add_argument("--test-query", type=Path, default=DEFAULT_TEST_QUERY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", default="64")
    parser.add_argument("--test-limit", default="64")
    parser.add_argument("--max-total-length", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["even", "shortest"],
        default="even",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    train_limit = _limit_value(str(args.train_limit))
    test_limit = _limit_value(str(args.test_limit))

    train_items, missing_train_query, train_not_complex, train_eligible_before_cap = (
        _select_train(
            args.train_ids,
            args.train_query,
            train_limit,
            args.max_total_length,
            args.selection_mode,
        )
    )
    test_items, test_eligible_before_cap = _select_test(
        args.test_query,
        test_limit,
        args.max_total_length,
        args.selection_mode,
    )

    _write_ids(out / "train_sample_ids.json", train_items)
    _write_ids(out / "test_sample_ids.json", test_items)
    _write_query(out / "query_train.json", train_items)
    _write_query(out / "query_test.json", test_items)

    summary = {
        "train_requested_limit": args.train_limit,
        "test_requested_limit": args.test_limit,
        "selection_mode": args.selection_mode,
        "max_total_length": args.max_total_length,
        "train_eligible_before_length_cap": train_eligible_before_cap,
        "test_eligible_before_length_cap": test_eligible_before_cap,
        "train_selected": len(train_items),
        "test_selected": len(test_items),
        "train_missing_query_count": len(missing_train_query),
        "train_not_complex_count": len(train_not_complex),
        "train_total_length_min_max": _length_min_max(train_items),
        "test_total_length_min_max": _length_min_max(test_items),
        "outputs": {
            "train_sample_ids": str(out / "train_sample_ids.json"),
            "test_sample_ids": str(out / "test_sample_ids.json"),
            "query_train": str(out / "query_train.json"),
            "query_test": str(out / "query_test.json"),
        },
    }
    (out / "eval_set_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
