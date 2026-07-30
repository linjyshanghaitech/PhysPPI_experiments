from __future__ import annotations

import csv
import json
import types
from types import SimpleNamespace

import numpy as np
import torch
from biotite import structure

from openfold3.core.runners.writer import OF3OutputWriter
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom


def _atom_array(n_atoms: int = 3) -> structure.AtomArray:
    atoms = [structure.Atom([float(i), 0.0, 0.0], chain_id="A") for i in range(n_atoms)]
    atom_array = structure.array(atoms)
    atom_array.entity_id = np.array(["A"] * n_atoms)
    atom_array.molecule_type_id = np.array(["0"] * n_atoms)
    atom_array.pdbx_formal_charge = np.array(["0"] * n_atoms)
    return atom_array


def test_validation_artifact_writer_exports_24_pdbs_and_candidate_metrics(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_write_structure_prediction(
        atom_array,
        predicted_coords,
        plddt,
        output_file,
        make_ost_compatible=True,
    ):
        calls.append(
            {
                "output_file": output_file,
                "coords_shape": tuple(predicted_coords.shape),
                "plddt_shape": tuple(plddt.shape),
            }
        )
        output_file.write_text("MODEL\nEND\n")

    monkeypatch.setattr(
        OF3OutputWriter,
        "write_structure_prediction",
        staticmethod(fake_write_structure_prediction),
    )

    runner = SimpleNamespace(
        config=SimpleNamespace(
            settings=SimpleNamespace(
                validation_artifacts={
                    "enabled": True,
                    "output_dir": str(tmp_path / "validation_artifacts"),
                    "structure_format": "pdb",
                    "expected_samples": 24,
                    "fail_on_sample_count_mismatch": True,
                    "profile_timing": True,
                    "synchronize_cuda_timing": False,
                }
            )
        )
    )
    runner._validation_artifact_config = types.MethodType(
        OpenFold3AllAtom._validation_artifact_config,
        runner,
    )
    runner.log_dir = tmp_path / "logs"
    runner.global_rank = 3
    runner.global_step = 50
    gt_pdb = tmp_path / "caseA_source_gt.pdb"
    gt_pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")
    batch = {
        "pdb_id": ["caseA"],
        "atom_array": [_atom_array()],
        "atom_mask": torch.ones(1, 3),
        "token_mask": torch.ones(1, 1, 3),
        "physppi_gt_pdb_path": [str(gt_pdb)],
    }
    outputs = {
        "atom_positions_predicted": torch.randn(1, 24, 3, 3),
    }
    metrics_per_sample = {
        "rmsd": torch.arange(24, dtype=torch.float32).reshape(1, 24),
        "dockq_protein_protein_w": torch.linspace(0.0, 1.0, 24).reshape(1, 24),
    }
    confidence_scores = {
        "plddt": torch.ones(1, 24, 3) * 75.0,
        "pdockq": torch.linspace(0.1, 0.9, 24).reshape(1, 24),
        "sample_ranking_score": torch.linspace(1.0, 2.0, 24).reshape(1, 24),
    }
    loss_breakdown = {
        "loss": torch.tensor(1.5),
        "pdockq_loss": torch.tensor(0.25),
    }

    OpenFold3AllAtom._write_validation_artifacts(
        runner,
        batch=batch,
        outputs=outputs,
        metrics_per_sample=metrics_per_sample,
        confidence_scores=confidence_scores,
        loss_breakdown=loss_breakdown,
        batch_idx=2,
        dataloader_idx=1,
        timing={"forward_seconds": 0.12},
    )

    case_dir = tmp_path / "validation_artifacts" / "dataloader_1" / "caseA"
    pdb_files = sorted(case_dir.glob("caseA_candidate_*.pdb"))
    assert len(pdb_files) == 24
    assert (case_dir / "caseA_gt.pdb").read_text() == gt_pdb.read_text()
    assert len(calls) == 24
    assert calls[0]["coords_shape"] == (3, 3)
    assert calls[0]["plddt_shape"] == (3,)

    with (case_dir / "candidate_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert rows[0]["candidate_index"] == "1"
    assert rows[0]["structure_file"] == "caseA_candidate_01.pdb"
    assert "rmsd" in rows[0]
    assert "dockq_protein_protein_w" in rows[0]
    assert "confidence/pdockq" in rows[0]
    assert "confidence/sample_ranking_score" in rows[0]

    summary = json.loads((case_dir / "case_summary.json").read_text())
    assert summary["case_id"] == "caseA"
    assert summary["gt_structure_file"] == "caseA_gt.pdb"
    assert summary["rank"] == 3
    assert summary["num_candidates"] == 24
    assert summary["loss_breakdown"]["loss"] == 1.5
    assert summary["loss_breakdown"]["pdockq_loss"] == 0.25
    assert summary["timing"]["forward_seconds"] == 0.12

    timing_path = (
        tmp_path / "logs" / "evidence" / "validation_artifacts" / "rank-3.jsonl"
    )
    timing_rows = [json.loads(line) for line in timing_path.read_text().splitlines()]
    assert timing_rows[0]["case_id"] == "caseA"
    assert timing_rows[0]["num_candidates"] == 24


def test_validation_artifact_writer_skips_nested_confidence_dicts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        OF3OutputWriter,
        "write_structure_prediction",
        staticmethod(
            lambda atom_array, predicted_coords, plddt, output_file, **_: output_file.write_text(
                "MODEL\nEND\n"
            )
        ),
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(
            settings=SimpleNamespace(
                validation_artifacts={
                    "enabled": True,
                    "output_dir": str(tmp_path / "validation_artifacts"),
                    "expected_samples": 1,
                }
            )
        ),
        log_dir=tmp_path / "logs",
        global_rank=0,
        global_step=50,
    )
    runner._validation_artifact_config = types.MethodType(
        OpenFold3AllAtom._validation_artifact_config,
        runner,
    )

    OpenFold3AllAtom._write_validation_artifacts(
        runner,
        batch={"pdb_id": ["caseB"], "atom_array": [_atom_array()]},
        outputs={"atom_positions_predicted": torch.randn(1, 1, 3, 3)},
        metrics_per_sample={},
        confidence_scores={
            "plddt": torch.ones(1, 1, 3),
            "chain_ptm": {"1": torch.ones(1, 1)},
        },
        loss_breakdown={"loss": torch.tensor(1.0)},
        batch_idx=0,
        dataloader_idx=0,
        timing={},
    )

    with (
        tmp_path
        / "validation_artifacts"
        / "dataloader_0"
        / "caseB"
        / "candidate_metrics.csv"
    ).open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert "confidence/chain_ptm" not in row


def test_eval_artifacts_use_lightweight_confidence_not_full_ranking():
    outputs = {
        "atom_positions_predicted": torch.randn(1, 1, 3, 3),
        "plddt_logits": torch.randn(1, 1, 3, 50),
        "pdockq_logits": torch.randn(1, 1, 50),
    }
    seen = {}

    class FakeRunner:
        global_rank = 0
        global_step = 50

        def __call__(self, batch):
            return batch, outputs

        def _validation_artifacts_enabled(self):
            return True

        def _synchronize_validation_timing(self):
            return None

        def _populate_physppi_pdockq_targets(self, batch, model_outputs):
            return None

        def loss(self, batch, model_outputs, _return_breakdown):
            return torch.tensor(0.0), {"loss": torch.tensor(0.0)}

        def _get_validation_metrics_detail(self, batch, model_outputs):
            return {}, {}

        def _log(self, *args, **kwargs):
            return None

        def _compute_confidence_scores(self, batch, model_outputs):
            raise AssertionError("full confidence ranking should not run for artifacts")

        def _compute_validation_artifact_confidence_scores(self, batch, model_outputs):
            return {"plddt": torch.ones(1, 1, 3), "pdockq": torch.ones(1, 1)}

        def _write_validation_artifacts(self, **kwargs):
            seen["confidence_scores"] = kwargs["confidence_scores"]

    OpenFold3AllAtom.eval_step(
        FakeRunner(),
        {"pdb_id": ["caseC"]},
        batch_idx=0,
        dataloader_idx=0,
    )

    assert set(seen["confidence_scores"]) == {"plddt", "pdockq"}


def test_validation_epoch_start_can_skip_ema_weight_swap():
    class FakeModel:
        def __init__(self):
            self.loaded = []

        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

        def load_state_dict(self, state_dict, strict=True):
            self.loaded.append((state_dict, strict))

    runner = SimpleNamespace(
        cached_weights=None,
        model=FakeModel(),
        ema=SimpleNamespace(
            state_dict=lambda: {"params": {"weight": torch.tensor([2.0])}}
        ),
        config=SimpleNamespace(
            settings=SimpleNamespace(validation_use_ema_weights=False)
        ),
    )

    OpenFold3AllAtom.on_validation_epoch_start(runner)

    assert runner.cached_weights is None
    assert runner.model.loaded == []


def test_validation_epoch_end_skips_restore_when_ema_swap_was_disabled():
    class FakeMetric:
        def compute(self):
            return {}

        def reset(self):
            return None

    class FakeStrategy:
        def barrier(self):
            return None

    runner = SimpleNamespace(
        cached_weights=None,
        model=SimpleNamespace(load_state_dict=lambda state_dict: None),
        val_losses=FakeMetric(),
        val_metrics=FakeMetric(),
        trainer=SimpleNamespace(
            current_epoch=0,
            global_step=0,
            strategy=FakeStrategy(),
        ),
        global_rank=0,
    )
    runner._log_epoch_metrics = lambda *args, **kwargs: None

    OpenFold3AllAtom.on_validation_epoch_end(runner)

    assert runner.cached_weights is None
