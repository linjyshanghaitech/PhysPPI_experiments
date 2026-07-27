from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.physppi.monitor_semantic_training import (
    _sampler_progress,
    strict_log_scan,
)
from scripts.physppi.validate_semantic_checkpoint import validate_checkpoint


@pytest.mark.parametrize(
    "text",
    [
        "NaN",
        "non-finite",
        "Traceback",
        "CUDA out of memory",
        "NCCL timeout",
        "DataLoader worker",
        "template skip",
        "broadcast error",
        "scattergather",
        "invalid token-center",
        "physppi_surface_local_coords",
    ],
)
def test_strict_scan_rejects_error_signatures(text: str) -> None:
    result = strict_log_scan(text)

    assert result.clean is False
    assert result.matches


def test_strict_scan_ignores_only_known_nonfatal_warnings() -> None:
    text = "\n".join(
        [
            "DeprecationWarning: torch.jit.script is not supported in Python 3.14+",
            "AccumulateGrad stream warning",
        ]
    )

    assert strict_log_scan(text).clean is True


def test_sampler_progress_reads_lightning_log_evidence_dir(tmp_path: Path) -> None:
    sampler_dir = tmp_path / "logs" / "evidence" / "sampler"
    sampler_dir.mkdir(parents=True)
    sampler_dir.joinpath("rank-0.jsonl").write_text(
        '{"rank": 0, "micro_step": 2, "sample_id": "qbiolip:1abc"}\n',
        encoding="utf-8",
    )
    sampler_dir.joinpath("rank-1.jsonl").write_text(
        '{"rank": 1, "micro_step": 2, "sample_id": "afdb:AF-1"}\n',
        encoding="utf-8",
    )

    result = _sampler_progress(tmp_path)

    assert result["rows_by_rank"] == {"rank-0": 1, "rank-1": 1}
    assert result["latest_rank_zero"] == {
        "rank": 0,
        "micro_step": 2,
        "sample_id": "qbiolip:1abc",
    }


def test_checkpoint_validator_requires_full_training_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "epoch=00-step=00256.ckpt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 256,
            "state_dict": {"weight": torch.ones(2)},
            "optimizer_states": [{"state": {}, "param_groups": [{}]}],
            "lr_schedulers": [{"last_epoch": 256}],
            "ema": {
                "decay": 0.999,
                "params": {"weight": torch.ones(2)},
            },
            "loops": {"fit_loop": {"epoch_progress": 1}},
            "DataModule": {
                "manifest_sha256_by_dataset": {
                    "qbiolip10k": "a" * 64,
                    "afdb10k": "a" * 64,
                }
            },
        },
        checkpoint,
    )

    report = validate_checkpoint(
        checkpoint,
        expected_manifest_sha256="a" * 64,
        expected_epoch=0,
        expected_global_step=256,
    )

    assert report.valid
    assert report.has_model and report.has_optimizer and report.has_scheduler
    assert report.has_ema and report.has_loops and report.has_datamodule
    assert report.epoch_only_boundary


def test_checkpoint_validator_rejects_nonfinite_tensor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.ckpt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 256,
            "state_dict": {"weight": torch.tensor([float("nan")])},
            "optimizer_states": [{}],
            "lr_schedulers": [{"last_epoch": 256}],
            "ema": {"decay": 0.999, "params": {"weight": torch.ones(1)}},
            "loops": {"fit_loop": {}},
            "DataModule": {"manifest_sha256_by_dataset": {"qbiolip10k": "a" * 64}},
        },
        checkpoint,
    )

    report = validate_checkpoint(
        checkpoint,
        expected_manifest_sha256="a" * 64,
    )

    assert not report.valid
    assert "nonfinite_tensor" in report.errors
