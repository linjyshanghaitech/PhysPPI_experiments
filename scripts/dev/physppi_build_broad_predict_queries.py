#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path("/101045/jinyu/PhysPPI/openfold-3")
DEFAULT_TRAIN_CACHE = (
    REPO
    / "output/physppi_finetune_validation/phase6/full2107"
    / "training_cache_full2107.json"
)
DEFAULT_OUTPUT_DIR = REPO / "output/physppi_cmap_only_research/broad_eval_inputs"
QUERY_SOURCES = [
    REPO / "output/physppi_full_msa_cache/colabfold_msas/chunks",
    REPO / "output/openfold3_official_docker_migration/comparison_inputs",
    REPO / "output/physppi_cmap_only_research/smoke",
    REPO
    / "output/physppi_cmap_only_research/broad_eval_inputs"
    / "missing_test128_msa/chunks",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def training_ids_from_cache(path: Path) -> set[str]:
    payload = load_json(path)
    if isinstance(payload.get("samples"), dict):
        return {str(x) for x in payload["samples"]}
    if isinstance(payload.get("structure_data"), dict):
        return {str(x) for x in payload["structure_data"]}
    if isinstance(payload.get("sample_ids"), list):
        return {str(x) for x in payload["sample_ids"]}
    raise ValueError(f"cannot infer training IDs from {path}")


def validate_no_test_leakage(train_cache: Path, test_ids_path: Path) -> list[str]:
    train_ids = training_ids_from_cache(train_cache)
    test_ids = {str(x) for x in load_json(test_ids_path)}
    return sorted(train_ids & test_ids)


def _candidate_query_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.glob("**/query_msa.json"))
            files.extend(path.glob("**/*with_msa.json"))
            files.extend(path.glob("**/*physppi_query.json"))
    return sorted(set(files))


def _path_exists(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_file() or (REPO / candidate).is_file()


def _query_component_status(query: dict[str, Any]) -> dict[str, bool]:
    chains = query.get("chains") or []
    has_sequence = bool(chains) and all(chain.get("sequence") for chain in chains)
    has_main_msa = bool(chains) and all(
        all(_path_exists(str(path)) for path in chain.get("main_msa_file_paths") or [])
        and bool(chain.get("main_msa_file_paths"))
        for chain in chains
    )
    has_template_alignment = bool(chains) and all(
        bool(chain.get("template_alignment_file_path"))
        and _path_exists(str(chain["template_alignment_file_path"]))
        for chain in chains
    )
    # Homomeric ColabFold outputs often have no explicit paired MSA path. The
    # query can still be predict-compatible when use_paired_msas is false/absent
    # or when per-chain paired paths exist and resolve.
    paired_paths = [
        path for chain in chains for path in (chain.get("paired_msa_file_paths") or [])
    ]
    has_paired_msa = (
        all(_path_exists(str(path)) for path in paired_paths) if paired_paths else True
    )
    return {
        "sequence": has_sequence,
        "main_msa": has_main_msa,
        "paired_msa": has_paired_msa,
        "template_alignment": has_template_alignment,
    }


def build_query_index(query_files: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for query_file in query_files:
        try:
            payload = load_json(query_file)
        except (json.JSONDecodeError, OSError):
            continue
        queries = payload.get("queries")
        if not isinstance(queries, dict):
            continue
        for sample_id, query in queries.items():
            sample_id = str(sample_id)
            if sample_id not in index:
                index[sample_id] = query
    return index


def materialize_split(
    *,
    split_name: str,
    sample_ids: list[str],
    query_index: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    queries: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    counts = {
        f"{split_name}_total": len(sample_ids),
        f"{split_name}_with_sequence": 0,
        f"{split_name}_with_main_msa": 0,
        f"{split_name}_with_paired_msa": 0,
        f"{split_name}_with_template_alignment": 0,
    }
    for sample_id in sample_ids:
        query = query_index.get(sample_id)
        if query is None:
            missing.append({"sample_id": sample_id, "reason": "missing_query"})
            continue
        status = _query_component_status(query)
        for key, ok in status.items():
            if ok:
                counts[f"{split_name}_with_{key}"] += 1
        if all(status.values()):
            queries[sample_id] = query
        else:
            missing.append(
                {
                    "sample_id": sample_id,
                    "reason": "incomplete_query_components",
                    "components": status,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"seeds": [42], "queries": queries}, indent=2) + "\n"
    )
    return {"counts": counts, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-source", action="append", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_ids = [str(x) for x in load_json(args.train_ids)]
    test_ids = [str(x) for x in load_json(args.test_ids)]
    leakage = validate_no_test_leakage(args.training_cache, args.test_ids)
    query_sources = args.query_source or QUERY_SOURCES
    query_files = _candidate_query_files(query_sources)
    query_index = build_query_index(query_files)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_result = materialize_split(
        split_name="train64",
        sample_ids=train_ids,
        query_index=query_index,
        output_path=args.output_dir / "train64_with_msa.json",
    )
    test_result = materialize_split(
        split_name="test128",
        sample_ids=test_ids,
        query_index=query_index,
        output_path=args.output_dir / "test128_with_msa.json",
    )
    missing = train_result["missing"] + test_result["missing"]
    report = {
        "train_ids": str(args.train_ids),
        "test_ids": str(args.test_ids),
        "training_cache": str(args.training_cache),
        "query_sources": [str(path) for path in query_sources],
        "query_files_scanned": len(query_files),
        "query_index_size": len(query_index),
        **train_result["counts"],
        **test_result["counts"],
        "test128_training_leakage": leakage,
        "missing": missing,
    }
    report["status"] = (
        "READY_FOR_BROAD_EVAL"
        if not leakage
        and not missing
        and report["train64_with_sequence"] == 64
        and report["test128_with_sequence"] == 128
        else "BLOCKED_QUERY_COVERAGE"
    )
    (args.output_dir / "query_coverage_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "broad_query_build_preflight.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return
    if report["status"] != "READY_FOR_BROAD_EVAL":
        raise SystemExit("broad query coverage is incomplete; not ready for GPU eval")


if __name__ == "__main__":
    main()
