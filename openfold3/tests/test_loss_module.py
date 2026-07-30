from __future__ import annotations

import ml_collections as mlc
import torch

from openfold3.core.loss import loss_module as loss_module_lib
from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.projects.of3_all_atom.config.model_config import model_config


def _auxiliary_loss_inputs():
    base = torch.tensor(1.0, requires_grad=True)
    membrane_pred = torch.tensor(2.0, requires_grad=True)
    interaction_pred = torch.tensor(3.0, requires_grad=True)
    grad_enabled = {"membrane": None, "interaction": None}

    def fake_confidence_loss(**_kwargs):
        return base * 0.0, {}

    def fake_membrane_loss(**_kwargs):
        grad_enabled["membrane"] = torch.is_grad_enabled()
        return membrane_pred * 5.0

    def fake_interaction_loss(**_kwargs):
        grad_enabled["interaction"] = torch.is_grad_enabled()
        return interaction_pred * 7.0

    batch = {
        "physppi_membrane_gt_normal": torch.zeros(1),
        "physppi_interaction_aux_valid": torch.ones(1, dtype=torch.bool),
    }
    output = {
        "physppi_predicted_membrane_normal": torch.zeros(1),
        "physppi_predicted_membrane_offset": torch.zeros(1),
        "interaction_distance_pred": torch.zeros(1),
    }
    return {
        "base": base,
        "membrane_pred": membrane_pred,
        "interaction_pred": interaction_pred,
        "grad_enabled": grad_enabled,
        "fake_confidence_loss": fake_confidence_loss,
        "fake_membrane_loss": fake_membrane_loss,
        "fake_interaction_loss": fake_interaction_loss,
        "batch": batch,
        "output": output,
    }


def _patch_auxiliary_losses(monkeypatch, inputs) -> None:
    monkeypatch.setattr(
        loss_module_lib,
        "confidence_loss",
        inputs["fake_confidence_loss"],
    )
    monkeypatch.setattr(
        loss_module_lib,
        "directed_membrane_plane_loss",
        inputs["fake_membrane_loss"],
    )
    monkeypatch.setattr(
        loss_module_lib,
        "interaction_auxiliary_loss",
        inputs["fake_interaction_loss"],
    )


def test_auxiliary_losses_log_only_do_not_contribute_gradients(monkeypatch):
    inputs = _auxiliary_loss_inputs()
    _patch_auxiliary_losses(monkeypatch, inputs)
    config = mlc.ConfigDict(
        {
            "train_confidence_only": True,
            "confidence": {},
            "membrane": {"include_in_total_loss": False},
            "interaction_auxiliary": {"include_in_total_loss": False},
        }
    )

    loss, breakdown = OpenFold3Loss(config).loss(
        inputs["batch"],
        inputs["output"],
    )
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.0))
    assert torch.isclose(breakdown["loss"], torch.tensor(0.0))
    assert torch.isclose(breakdown["membrane_plane_loss"], torch.tensor(10.0))
    assert torch.isclose(
        breakdown["interaction_auxiliary_loss"],
        torch.tensor(21.0),
    )
    assert inputs["grad_enabled"] == {"membrane": False, "interaction": False}
    assert inputs["membrane_pred"].grad is None
    assert inputs["interaction_pred"].grad is None


def test_auxiliary_losses_default_include_in_total_contributes_gradients(
    monkeypatch,
):
    inputs = _auxiliary_loss_inputs()
    _patch_auxiliary_losses(monkeypatch, inputs)
    config = mlc.ConfigDict(
        {
            "train_confidence_only": True,
            "confidence": {},
            "membrane": {},
            "interaction_auxiliary": {},
        }
    )

    loss, breakdown = OpenFold3Loss(config).loss(
        inputs["batch"],
        inputs["output"],
    )
    loss.backward()

    assert torch.isclose(loss, torch.tensor(31.0))
    assert torch.isclose(breakdown["loss"], torch.tensor(31.0))
    assert inputs["grad_enabled"] == {"membrane": True, "interaction": True}
    assert torch.isclose(inputs["membrane_pred"].grad, torch.tensor(5.0))
    assert torch.isclose(inputs["interaction_pred"].grad, torch.tensor(7.0))


def test_auxiliary_losses_are_enabled_in_default_model_config():
    loss_config = model_config.architecture.loss_module

    assert loss_config.membrane.include_in_total_loss is True
    assert loss_config.interaction_auxiliary.include_in_total_loss is True
