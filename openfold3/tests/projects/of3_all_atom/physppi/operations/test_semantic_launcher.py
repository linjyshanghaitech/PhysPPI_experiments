from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from openfold3.projects.of3_all_atom.runner import sampler_trace_row
from scripts.physppi.launch_semantic_training import (
    GPUProcess,
    launch_semantic_training,
    materialize_sampler_prefix,
)


def test_trace_row_contains_parity_fields() -> None:
    row = sampler_trace_row(
        {
            "pdb_id": ["AF-fixture"],
            "physppi_occurrence_seed": torch.tensor([17], dtype=torch.uint64),
            "physppi_prior_requested_level": torch.tensor([4]),
            "physppi_prior_effective_level": torch.tensor([3]),
        },
        rank=3,
        epoch=0,
        micro_step=7,
        optimizer_step=2,
    )

    assert set(row) >= {
        "rank",
        "epoch",
        "micro_step",
        "dataset",
        "sample_id",
        "occurrence_seed",
        "requested_level",
        "effective_level",
        "contact_cutoff",
        "optimizer_step",
    }
    assert row["dataset"] == "afdb"
    assert 8.0 <= row["contact_cutoff"] <= 18.0


def _prefix_config(manifest: Path, epoch_len: int) -> dict:
    return {
        "data_module_args": {
            "data_seed": 42,
            "epoch_len": epoch_len,
            "distinct_samples_per_rank_group": True,
        },
        "dataset_configs": {
            "train": {
                "qbiolip10k": {
                    "weight": 0.8,
                    "config": {
                        "physppi": {
                            "manifest": str(manifest),
                            "manifest_dataset_key": "qbiolip",
                            "manifest_partition": "train_ids",
                        }
                    },
                },
                "afdb10k": {
                    "weight": 0.2,
                    "config": {
                        "physppi": {
                            "manifest": str(manifest),
                            "manifest_dataset_key": "afdb",
                            "manifest_partition": "train_ids",
                        }
                    },
                },
            }
        },
    }


def test_formal_sampler_prefix_exactly_matches_smoke_prefix(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "datasets": {
                    "qbiolip": {
                        "train_ids": [f"q-{index}" for index in range(997)],
                        "diagnostic_ids": [],
                    },
                    "afdb": {
                        "train_ids": [f"AF-{index}" for index in range(991)],
                        "diagnostic_ids": [],
                    },
                },
            }
        )
    )

    smoke = materialize_sampler_prefix(
        _prefix_config(manifest, 512),
        prefix_micro_steps=64,
    )
    formal = materialize_sampler_prefix(
        _prefix_config(manifest, 4096),
        prefix_micro_steps=64,
    )

    assert formal == smoke
    assert {len(rows) for rows in formal.values()} == {64}
    for micro_step in range(64):
        ids = {formal[rank][micro_step]["sample_id"] for rank in range(8)}
        assert len(ids) == 8


def test_launcher_refuses_existing_output(tmp_path: Path) -> None:
    existing = tmp_path / "run"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        launch_semantic_training(tmp_path / "config.yml", existing, "a" * 64)


def test_launcher_refuses_busy_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.physppi.launch_semantic_training.gpu_processes",
        lambda: (
            GPUProcess(gpu=0, pid=123, process_name="python", used_memory_mib=100),
        ),
    )

    with pytest.raises(RuntimeError, match="GPU"):
        launch_semantic_training(
            tmp_path / "config.yml",
            tmp_path / "fresh",
            "a" * 64,
        )


def test_mocked_launcher_records_one_owned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yml"
    config.write_text(
        "\n".join(
            [
                "experiment_settings:",
                f"  output_dir: {tmp_path / 'run'}",
                "dataset_configs:",
                "  train:",
                "    qbiolip10k:",
                "      config:",
                "        physppi:",
                f"          manifest: {manifest}",
            ]
        )
        + "\n"
    )
    output = tmp_path / "run"
    monkeypatch.setattr(
        "scripts.physppi.launch_semantic_training.gpu_processes",
        lambda: (),
    )
    monkeypatch.setattr(
        "scripts.physppi.launch_semantic_training._validate_launch_config",
        lambda *_args, **_kwargs: (manifest, checkpoint),
    )
    monkeypatch.setattr(
        "scripts.physppi.launch_semantic_training._capture_static_evidence",
        lambda *_args, **_kwargs: None,
    )

    class Process:
        pid = 4321

    captured_environment: dict[str, str] = {}

    def fake_popen(*_args, **kwargs):
        if not kwargs.get("start_new_session"):
            pytest.fail("launcher must use start_new_session")
        captured_environment.update(kwargs["env"])
        return Process()

    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    monkeypatch.setattr(
        "scripts.physppi.launch_semantic_training.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "scripts.physppi.launch_semantic_training.os.getpgid",
        lambda pid: pid,
    )

    result = launch_semantic_training(config, output, "a" * 64)

    evidence = json.loads((output / "evidence" / "launcher-process.json").read_text())
    assert result.pgid == 4321
    assert evidence["child_pid"] == 4321
    assert evidence["pgid"] == 4321
    repository = Path(__file__).resolve().parents[6]
    assert captured_environment["PYTHONPATH"].split(os.pathsep) == [
        str(repository),
        "/existing/path",
    ]
