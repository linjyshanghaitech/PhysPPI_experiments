import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[6]
        / "scripts/dev/physppi_record_runtime_skip.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_skip", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_records_failed_sample_without_mutating_source_manifest() -> None:
    manifest = {
        "datasets": {
            "qbiolip": {
                "train_ids": ["good", "failed"],
                "fatal_ids": ["old"],
            },
            "afdb": {"train_ids": ["af"], "fatal_ids": []},
        },
        "runtime_skips": [],
    }

    updated = _load_module().record_runtime_skip(
        manifest,
        dataset="qbiolip",
        sample_id="failed",
        reason="index out of bounds",
        source_run="smoke-v1",
        source_global_step=4,
        source_micro_batch=19,
    )

    assert updated["datasets"]["qbiolip"] == {
        "train_ids": ["good"],
        "fatal_ids": ["failed", "old"],
    }
    assert updated["datasets"]["afdb"] == manifest["datasets"]["afdb"]
    assert updated["runtime_skips"] == [
        {
            "dataset": "qbiolip",
            "sample_id": "failed",
            "reason": "index out of bounds",
            "source_run": "smoke-v1",
            "source_global_step": 4,
            "source_micro_batch": 19,
        }
    ]
    assert manifest["datasets"]["qbiolip"]["train_ids"] == [
        "good",
        "failed",
    ]


def test_does_not_duplicate_an_existing_runtime_skip() -> None:
    manifest = {
        "datasets": {
            "qbiolip": {"train_ids": [], "fatal_ids": ["failed"]}
        },
        "runtime_skips": [
            {
                "dataset": "qbiolip",
                "sample_id": "failed",
                "reason": "original",
                "source_run": "smoke-v1",
            }
        ],
    }

    updated = _load_module().record_runtime_skip(
        manifest,
        dataset="qbiolip",
        sample_id="failed",
        reason="duplicate",
        source_run="smoke-v2",
    )

    assert updated["runtime_skips"] == manifest["runtime_skips"]


def test_failure_fingerprint_normalizes_worker_and_address_noise() -> None:
    module = _load_module()

    first = module.failure_fingerprint(
        "ValueError",
        "worker 3 failed at 0x7f123: Atom IDs are not strictly increasing",
    )
    second = module.failure_fingerprint(
        "ValueError",
        "worker 7 failed at 0x9abc: Atom IDs are not strictly increasing",
    )

    assert first == second


def test_append_failure_record_is_idempotent(tmp_path: Path) -> None:
    module = _load_module()
    ledger = tmp_path / "failures.jsonl"
    record = {
        "dataset": "qbiolip",
        "sample_id": "3b62_2",
        "failure_fingerprint": "abc",
    }

    assert module.append_failure_record(ledger, record) is True
    assert module.append_failure_record(ledger, record) is False
    assert len(ledger.read_text().splitlines()) == 1


def test_append_failure_record_accepts_legacy_fingerprint_rows(
    tmp_path: Path,
) -> None:
    module = _load_module()
    ledger = tmp_path / "failures.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "dataset": "qbiolip",
                "sample_id": "legacy",
                "fingerprint": "old-schema-fingerprint",
            }
        )
        + "\n"
    )

    assert module.append_failure_record(
        ledger,
        {
            "dataset": "qbiolip",
            "sample_id": "fresh",
            "failure_fingerprint": "new-schema-fingerprint",
        },
    ) is True
    assert len(ledger.read_text().splitlines()) == 2


def test_non_sample_failure_category_cannot_exclude_id() -> None:
    with pytest.raises(ValueError, match="not sample-specific"):
        _load_module().validate_sample_failure_category("nccl")


def test_cli_records_versioned_sample_exclusion_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    manifest = {
        "datasets": {
            "qbiolip": {
                "train_ids": ["healthy", "failed"],
                "fatal_ids": [],
            },
            "afdb": {"train_ids": ["afdb"], "fatal_ids": []},
        },
        "runtime_skips": [],
    }
    manifest_path = tmp_path / "formal-manifest-v7.json"
    yaml_path = tmp_path / "stage1-v12.yml"
    log_path = tmp_path / "train.log"
    output_path = tmp_path / "formal-manifest-v8-parse-failed.json"
    ledger_path = tmp_path / "sample-failures.jsonl"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    yaml_path.write_text("data_module_args: {}\n")
    log_path.write_text("ValueError: Atom IDs are not strictly increasing\n")
    argv = [
        "physppi_record_runtime_skip.py",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
        "--ledger",
        str(ledger_path),
        "--dataset",
        "qbiolip",
        "--sample-id",
        "failed",
        "--reason",
        "worker 4 failed at 0x7f123: Atom IDs are not strictly increasing",
        "--failure-category",
        "parse",
        "--exception-class",
        "ValueError",
        "--source-run",
        "stage1-full-v12-smoke32",
        "--rank",
        "4",
        "--epoch",
        "1",
        "--optimizer-step",
        "3",
        "--micro-step",
        "7",
        "--git-head",
        "deadbeef",
        "--yaml",
        str(yaml_path),
        "--source-log",
        str(log_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()
    module.main()

    updated = json.loads(output_path.read_text())
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    record = records[0]

    assert updated["datasets"]["qbiolip"] == {
        "train_ids": ["healthy"],
        "fatal_ids": ["failed"],
    }
    assert len(records) == 1
    assert record["failure_category"] == "parse"
    assert record["exception_class"] == "ValueError"
    assert record["rank"] == 4
    assert record["epoch"] == 1
    assert record["optimizer_step"] == 3
    assert record["micro_step"] == 7
    assert record["git_head"] == "deadbeef"
    assert record["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert record["yaml_sha256"] == hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    assert record["source_log_sha256"] == hashlib.sha256(
        log_path.read_bytes()
    ).hexdigest()
