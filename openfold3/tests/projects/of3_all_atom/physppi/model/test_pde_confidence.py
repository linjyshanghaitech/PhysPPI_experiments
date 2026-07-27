"""Focused regression tests for the active PhysPPI PDE confidence route."""

import torch

from openfold3.core.loss.confidence import confidence_loss, pde_loss
from openfold3.core.metrics.aggregate_confidence_ranking import _get_confidence_scores
from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests import test_confidence_loss as _confidence_loss_tests


def test_pde_is_configured_and_experimentally_resolved_is_not_constructed():
    config = OF3ProjectEntry().get_model_config_with_presets()
    heads = AuxiliaryHeadsAllAtom(config.architecture.heads)

    assert config.architecture.heads.pde.c_out == 64
    assert hasattr(heads, "pde")
    assert not hasattr(heads, "experimentally_resolved")
    assert "experimentally_resolved" not in config.architecture.loss_module.confidence


def test_pde_loss_is_finite_and_differentiable_for_valid_representatives():
    batch = _confidence_loss_tests.TestConfidenceLoss().setup_features()
    n_token = batch["token_mask"].shape[-1]
    logits = torch.randn(
        (1, n_token, n_token, 64),
        requires_grad=True,
    )

    loss = pde_loss(
        batch=batch,
        x=torch.randn_like(batch["ground_truth"]["atom_positions"]),
        logits=logits,
        no_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        eps=1e-8,
    )

    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_pde_loss_with_no_resolved_representatives_is_finite_zero():
    batch = _confidence_loss_tests.TestConfidenceLoss().setup_features()
    batch["ground_truth"]["atom_resolved_mask"].zero_()
    n_token = batch["token_mask"].shape[-1]
    logits = torch.randn(
        (1, n_token, n_token, 64),
        requires_grad=True,
    )

    loss = pde_loss(
        batch=batch,
        x=torch.randn_like(batch["ground_truth"]["atom_positions"]),
        logits=logits,
        no_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        eps=1e-8,
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.sum().backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_confidence_loss_requires_active_pde_logits():
    batch = _confidence_loss_tests.TestConfidenceLoss().setup_features()
    config = OF3ProjectEntry().get_model_config_with_presets()
    n_atom = batch["ground_truth"]["atom_positions"].shape[-2]
    n_token = batch["token_mask"].shape[-1]
    output = {
        "atom_positions_predicted": torch.randn_like(
            batch["ground_truth"]["atom_positions"]
        ),
        "plddt_logits": torch.randn(
            1, n_atom, config.architecture.loss_module.confidence.plddt.no_bins
        ),
        "pae_logits": torch.randn(
            1,
            n_token,
            n_token,
            config.architecture.loss_module.confidence.pae.no_bins,
        ),
    }

    try:
        confidence_loss(
            batch=batch,
            output=output,
            **config.architecture.loss_module.confidence,
        )
    except KeyError as exc:
        assert exc.args[0] == "pde_logits"
    else:
        raise AssertionError("active PDE loss must require pde_logits")


def test_pde_metric_exposes_expected_error_and_optional_probabilities():
    batch = _confidence_loss_tests.TestConfidenceLoss().setup_features()
    config = OF3ProjectEntry().get_model_config_with_presets()
    config.architecture.heads.pae.enabled = False
    config.confidence.pde.return_probs = True
    n_atom = batch["ground_truth"]["atom_positions"].shape[-2]
    n_token = batch["token_mask"].shape[-1]
    outputs = {
        "plddt_logits": torch.randn(1, n_atom, config.confidence.plddt.no_bins),
        "pde_logits": torch.randn(1, n_token, n_token, config.confidence.pde.no_bins),
    }

    scores = _get_confidence_scores(batch=batch, outputs=outputs, config=config)

    assert scores["pde"].shape == (1, n_token, n_token)
    assert scores["pde_probs"].shape == (
        1,
        n_token,
        n_token,
        config.confidence.pde.no_bins,
    )
    assert torch.isfinite(scores["pde"]).all()
