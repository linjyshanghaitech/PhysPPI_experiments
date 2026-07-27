#!/usr/bin/env python3
"""Create a versioned manifest that records and excludes one failed sample."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

SAMPLE_FAILURE_CATEGORIES = frozenset({"parse", "feature", "label"})


def failure_fingerprint(exception_class: str, message: str) -> str:
    normalized = re.sub(r"worker\s+\d+", "worker <n>", message, flags=re.I)
    normalized = re.sub(r"0x[0-9a-f]+", "0x<addr>", normalized, flags=re.I)
    payload = f"{exception_class}:{normalized.strip()}".encode()
    return hashlib.sha256(payload).hexdigest()


def validate_sample_failure_category(category: str) -> None:
    if category not in SAMPLE_FAILURE_CATEGORIES:
        raise ValueError(f"failure category {category!r} is not sample-specific")


def append_failure_record(ledger_path: Path, record: dict[str, object]) -> bool:
    key = (
        record["dataset"],
        record["sample_id"],
        record["failure_fingerprint"],
    )
    existing = []
    if ledger_path.exists():
        existing = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    if any(
        (
            row["dataset"],
            row["sample_id"],
            row.get("failure_fingerprint", row.get("fingerprint")),
        )
        == key
        for row in existing
    ):
        return False

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return True


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_runtime_skip(
    manifest: dict[str, object],
    *,
    dataset: str,
    sample_id: str,
    reason: str,
    source_run: str,
    source_global_step: int | None = None,
    source_micro_batch: int | None = None,
) -> dict[str, object]:
    updated = copy.deepcopy(manifest)
    dataset_manifest = updated["datasets"][dataset]
    dataset_manifest["train_ids"] = [
        value for value in dataset_manifest["train_ids"] if value != sample_id
    ]
    dataset_manifest["fatal_ids"] = sorted(
        set(dataset_manifest["fatal_ids"]) | {sample_id}
    )
    runtime_skips = updated.setdefault("runtime_skips", [])
    if not any(
        entry.get("dataset") == dataset
        and entry.get("sample_id") == sample_id
        for entry in runtime_skips
    ):
        record = {
            "dataset": dataset,
            "sample_id": sample_id,
            "reason": reason,
            "source_run": source_run,
        }
        if source_global_step is not None:
            record["source_global_step"] = source_global_step
        if source_micro_batch is not None:
            record["source_micro_batch"] = source_micro_batch
        runtime_skips.append(record)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--failure-category", required=True)
    parser.add_argument("--exception-class", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--optimizer-step", "--source-global-step", type=int)
    parser.add_argument("--micro-step", "--source-micro-batch", type=int)
    parser.add_argument("--git-head")
    parser.add_argument("--yaml", type=Path)
    parser.add_argument("--source-log", type=Path)
    args = parser.parse_args()

    validate_sample_failure_category(args.failure_category)
    manifest = json.loads(args.manifest.read_text())
    fingerprint = failure_fingerprint(args.exception_class, args.reason)
    ledger_record: dict[str, object] = {
        "dataset": args.dataset,
        "sample_id": args.sample_id,
        "failure_category": args.failure_category,
        "exception_class": args.exception_class,
        "reason": args.reason,
        "failure_fingerprint": fingerprint,
        "source_run": args.source_run,
        "rank": args.rank,
        "epoch": args.epoch,
        "optimizer_step": args.optimizer_step,
        "micro_step": args.micro_step,
        "git_head": args.git_head,
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "yaml_path": str(args.yaml) if args.yaml is not None else None,
        "yaml_sha256": sha256_file(args.yaml),
        "source_log_path": str(args.source_log)
        if args.source_log is not None
        else None,
        "source_log_sha256": sha256_file(args.source_log),
    }
    append_failure_record(args.ledger, ledger_record)
    updated = record_runtime_skip(
        manifest,
        dataset=args.dataset,
        sample_id=args.sample_id,
        reason=args.reason,
        source_run=args.source_run,
        source_global_step=args.optimizer_step,
        source_micro_batch=args.micro_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
