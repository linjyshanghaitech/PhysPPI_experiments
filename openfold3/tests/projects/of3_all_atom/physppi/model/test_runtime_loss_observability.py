from __future__ import annotations

import inspect
import json

import pytest
import torch

import openfold3.projects.of3_all_atom.runner as runner_module


def _batch(sample_id: str, loss_weight: float) -> dict:
    return {
        "pdb_id": [sample_id],
        "physppi_loss_weight": torch.tensor([[loss_weight]]),
        "physppi_membrane_gt_valid": torch.tensor([[True]]),
        "physppi_tm_token_mask": torch.tensor([[[True, False, True]]]),
    }


def _output() -> dict:
    return {
        "physppi_predicted_membrane_valid": torch.tensor([[True]]),
        "physppi_membrane_valid_frame_count": torch.tensor([[1]]),
    }


def test_all_ranks_append_case_specific_loss_rows(tmp_path) -> None:
    path = tmp_path / "physppi_route_c_train_metrics.jsonl"

    runner_module.write_physppi_route_c_train_metrics(
        path=path,
        batch=_batch("q-case", 1.0),
        output=_output(),
        loss_breakdown={
            "loss": torch.tensor(7.0),
            "pdockq_loss": torch.tensor(3.0),
            "membrane_plane_loss": torch.tensor(2.0),
        },
        rank=0,
        global_step=50,
        epoch=0,
    )
    runner_module.write_physppi_route_c_train_metrics(
        path=path,
        batch=_batch("AF-case", 0.8),
        output=_output(),
        loss_breakdown={
            "loss": torch.tensor(5.6),
            "pdockq_loss": torch.tensor(2.4),
            "membrane_plane_loss": torch.tensor(1.6),
        },
        rank=3,
        global_step=50,
        epoch=0,
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["rank"] for row in rows] == [0, 3]
    assert [row["sample_id"] for row in rows] == ["q-case", "AF-case"]
    assert [row["dataset"] for row in rows] == ["qbiolip", "afdb"]
    assert [row["dataset_loss_weight"] for row in rows] == [1.0, 0.8]
    assert all(row["global_step"] == 50 for row in rows)
    assert rows[1]["train/loss"] == pytest.approx(5.6)
    assert rows[1]["train/pdockq_loss"] == pytest.approx(2.4)
    assert rows[1]["train/membrane_plane_loss"] == pytest.approx(1.6)
    assert rows[1]["membrane_gt_valid"] is True
    assert rows[1]["membrane_tm_token_count"] == 2
    assert rows[1]["membrane_valid_frame_count"] == 1
    assert rows[1]["membrane_prediction_valid"] is True


def test_route_c_metrics_are_written_for_every_train_step() -> None:
    source = inspect.getsource(runner_module.OpenFold3AllAtom._log)

    assert "if train:\n            self._write_physppi_route_c_train_metrics" in source
    assert (
        "if train and log_train_step_metrics:\n"
        "            self._write_physppi_route_c_train_metrics"
        not in source
    )
