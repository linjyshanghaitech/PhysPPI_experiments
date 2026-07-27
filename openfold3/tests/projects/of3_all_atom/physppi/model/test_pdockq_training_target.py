from __future__ import annotations

import inspect
import types
from types import SimpleNamespace

import pytest
import torch

from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.constants import TRAIN_LOSSES
from openfold3.projects.of3_all_atom.model import OpenFold3
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

PHYSPPI_SCOPE = "physppi_lora_assessment"


def test_pdockq_loss_is_registered_for_epoch_logging():
    assert "pdockq_loss" in TRAIN_LOSSES


def _runner(scope: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            settings=SimpleNamespace(training_scope=scope),
        )
    )


def _minimal_target_batch() -> dict:
    return {
        "ground_truth": {"atom_positions": torch.zeros(1, 1, 4, 3)},
        "atom_mask": torch.ones(1, 1, 4, dtype=torch.bool),
        "asym_id_atomized": torch.tensor([[[0, 0, 1, 1]]]),
        "res_id_atomized": torch.arange(4).reshape(1, 1, 4),
        "ref_atom_name_chars_atomized": torch.zeros(1, 1, 4, 4, 64),
        "inter_filter_atomized": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        "is_protein_atomized": torch.ones(1, 1, 4, dtype=torch.bool),
        "is_rna_atomized": torch.zeros(1, 1, 4, dtype=torch.bool),
        "is_dna_atomized": torch.zeros(1, 1, 4, dtype=torch.bool),
    }


def test_runner_populates_candidate_specific_pdockq_targets(monkeypatch):
    import openfold3.projects.of3_all_atom.runner as module

    seen = {}

    def fake_per_candidate_complex_dockq(*, prediction, gt, batch):
        seen["prediction"] = prediction
        seen["gt"] = gt
        seen["batch"] = batch
        return torch.tensor([[0.25, 0.75]]), torch.tensor([[True, True]])

    monkeypatch.setattr(
        module, "per_candidate_complex_dockq", fake_per_candidate_complex_dockq
    )
    batch = _minimal_target_batch()
    outputs = {
        "atom_positions_diffusion": torch.randn(1, 2, 4, 3),
        "pdockq_logits": torch.randn(1, 2, 50),
    }

    OpenFold3AllAtom._populate_physppi_pdockq_targets(
        _runner(PHYSPPI_SCOPE), batch, outputs
    )

    assert seen["prediction"] is outputs["atom_positions_diffusion"]
    assert seen["gt"] is batch["ground_truth"]["atom_positions"]
    assert batch["physppi_pdockq_target"].shape == (1, 2, 50)
    assert batch["physppi_pdockq_target_score"].shape == (1, 2)
    assert batch["physppi_pdockq_valid"].shape == (1, 2)
    assert torch.equal(batch["physppi_pdockq_target"].sum(-1), torch.ones(1, 2))


def test_baseline_scope_does_not_build_or_attach_pdockq_targets(monkeypatch):
    import openfold3.projects.of3_all_atom.runner as module

    monkeypatch.setattr(
        module,
        "per_candidate_complex_dockq",
        lambda **_: pytest.fail("baseline route must not construct DockQ targets"),
    )
    batch = _minimal_target_batch()
    outputs = {
        "atom_positions_diffusion": torch.randn(1, 48, 4, 3),
        "pdockq_logits": torch.randn(1, 48, 50),
    }

    OpenFold3AllAtom._populate_physppi_pdockq_targets(
        _runner("baseline"), batch, outputs
    )

    assert not any(key.startswith("physppi_pdockq_") for key in batch)


@pytest.mark.parametrize(
    ("missing_from", "key"),
    [
        ("output", "atom_positions_diffusion"),
        ("output", "pdockq_logits"),
        ("ground_truth", "atom_positions"),
    ],
)
def test_physppi_scope_requires_every_pdockq_training_input(missing_from, key):
    batch = _minimal_target_batch()
    outputs = {
        "atom_positions_diffusion": torch.randn(1, 48, 4, 3),
        "pdockq_logits": torch.randn(1, 48, 50),
    }
    if missing_from == "output":
        outputs.pop(key)
    else:
        batch["ground_truth"].pop(key)

    with pytest.raises(KeyError, match=key):
        OpenFold3AllAtom._populate_physppi_pdockq_targets(
            _runner(PHYSPPI_SCOPE), batch, outputs
        )


def test_assessment_head_preserves_all_48_training_diffusion_candidates(
    monkeypatch,
):
    import openfold3.core.model.heads.head_modules as module

    batch_size, candidates, tokens, atoms = 1, 48, 3, 4
    c_s, c_z = 5, 7
    candidate_positions = torch.randn(batch_size, candidates, atoms, 3)
    representative = torch.randn(batch_size, candidates, tokens, 3)
    representative_mask = torch.ones(batch_size, 1, tokens, dtype=torch.bool)
    monkeypatch.setattr(
        module,
        "get_token_representative_atoms",
        lambda **_: (representative, representative_mask),
    )
    chunk_sizes = []

    class FakeEmbedding:
        def __call__(self, *, si, zij, x_pred, **kwargs):
            chunk_sizes.append(x_pred.shape[1])
            assert x_pred.shape[1] <= 2
            assert kwargs["single_mask"].shape[1] == x_pred.shape[1]
            return (
                si.expand(batch_size, x_pred.shape[1], tokens, c_s),
                zij.expand(batch_size, x_pred.shape[1], tokens, tokens, c_z),
            )

    class FakePDockQ:
        def __call__(self, *, si, z, asym_id, token_mask):
            chunk = si.shape[1]
            assert z.shape[1] == chunk
            assert asym_id.shape == (batch_size, chunk, tokens)
            assert token_mask.shape == (batch_size, chunk, tokens)
            return torch.randn(batch_size, chunk, 50)

        @staticmethod
        def expectation(logits):
            return logits.softmax(-1).mean(-1)

    heads = SimpleNamespace(
        pairformer_embedding=FakeEmbedding(),
        pdockq=FakePDockQ(),
        pdockq_candidate_chunk_size=2,
        training=False,
    )
    batch = {
        "atom_mask": torch.ones(batch_size, 1, atoms, dtype=torch.bool),
        "token_mask": torch.ones(batch_size, 1, tokens, dtype=torch.bool),
        "asym_id": torch.tensor([[[0, 0, 1]]]),
    }
    output = {
        "si_trunk": torch.randn(batch_size, 1, tokens, c_s),
        "zij_trunk": torch.randn(batch_size, 1, tokens, tokens, c_z),
    }

    result = AuxiliaryHeadsAllAtom.forward_pdockq_candidates(
        heads,
        batch=batch,
        si_input=torch.randn(batch_size, 1, tokens, 9),
        output=output,
        candidate_positions=candidate_positions,
        use_zij_trunk_embedding=True,
    )

    assert result["pdockq_logits"].shape == (1, 48, 50)
    assert result["pdockq_expectation"].shape == (1, 48)
    assert chunk_sizes == [2] * 24


def test_resolved_training_config_assesses_the_48_diffusion_replica_axis():
    diffusion = model_config.architecture.shared.diffusion
    assert diffusion.no_samples == 48
    assert diffusion.no_mini_rollout_samples == 1
    assert model_config.architecture.heads.pdockq.candidate_chunk_size == 1

    source = inspect.getsource(OpenFold3.forward)
    diffusion_at = source.index("output.update(diffusion_output)")
    assessment_at = source.index("forward_pdockq_candidates")
    candidate_at = source.index('"atom_positions_diffusion"', assessment_at)
    assert diffusion_at < assessment_at < candidate_at


def test_both_training_paths_populate_targets_after_model_before_loss():
    automatic = inspect.getsource(OpenFold3AllAtom._training_step)
    manual = inspect.getsource(OpenFold3AllAtom._training_step_manual_clip)
    for source in (automatic, manual):
        model_at = source.index("self.model(batch)")
        target_at = source.index("self._populate_physppi_pdockq_targets")
        loss_at = source.index("self.loss(")
        assert model_at < target_at < loss_at


def test_automatic_training_step_passes_runtime_targets_to_loss(monkeypatch):
    import openfold3.projects.of3_all_atom.runner as module

    monkeypatch.setattr(
        module,
        "per_candidate_complex_dockq",
        lambda **_: (
            torch.tensor([[0.3, 0.7]]),
            torch.tensor([[True, True]]),
        ),
    )
    batch = _minimal_target_batch()
    batch.update(
        {
            "token_mask": torch.ones(1, 1, 4),
            "pdb_id": ["case"],
            "preferred_chain_or_interface": ["A", "B"],
        }
    )
    outputs = {
        "atom_positions_diffusion": torch.randn(1, 2, 4, 3),
        "pdockq_logits": torch.randn(1, 2, 50, requires_grad=True),
    }
    seen = {}

    class FakeEMA:
        device = torch.device("cpu")

        def to(self, device):
            self.device = device

    class FakeLoss:
        def __call__(self, processed, model_outputs, _return_breakdown):
            seen["target"] = processed["physppi_pdockq_target"]
            seen["score"] = processed["physppi_pdockq_target_score"]
            seen["valid"] = processed["physppi_pdockq_valid"]
            return model_outputs["pdockq_logits"].sum() * 0.0, {}

    runner = SimpleNamespace(
        config=SimpleNamespace(settings=SimpleNamespace(training_scope=PHYSPPI_SCOPE)),
        ema=FakeEMA(),
        model=lambda value: (value, outputs),
        loss=FakeLoss(),
        global_rank=0,
        global_step=0,
        _log=lambda *args, **kwargs: None,
    )
    runner._populate_physppi_pdockq_targets = types.MethodType(
        OpenFold3AllAtom._populate_physppi_pdockq_targets, runner
    )

    OpenFold3AllAtom._training_step(runner, batch)

    assert seen["target"].shape == (1, 2, 50)
    torch.testing.assert_close(seen["score"], torch.tensor([[0.3, 0.7]]))
    assert seen["valid"].tolist() == [[True, True]]
