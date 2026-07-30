import pytest
import torch

from openfold3.projects.of3_all_atom.constants import VAL_LOSSES
from openfold3.projects.of3_all_atom.runner import apply_physppi_loss_weight


def test_validation_tracks_interaction_auxiliary_loss():
    assert "interaction_auxiliary_loss" in VAL_LOSSES


def test_physppi_loss_weight_scales_total_and_every_breakdown_component():
    loss = torch.tensor(10.0, requires_grad=True)
    breakdown = {
        "confidence_loss": torch.tensor(2.0),
        "diffusion_loss": torch.tensor(3.0),
        "pdockq_loss": torch.tensor(4.0),
        "membrane_plane_loss": torch.tensor(5.0),
        "interaction_auxiliary_loss": torch.tensor(6.0),
        "loss": torch.tensor(10.0),
    }

    weighted_loss, weighted_breakdown = apply_physppi_loss_weight(
        {"physppi_loss_weight": torch.tensor([0.8])},
        loss,
        breakdown,
    )

    assert weighted_loss.item() == pytest.approx(8.0)
    assert {
        name: value.item() for name, value in weighted_breakdown.items()
    } == pytest.approx(
        {
            "confidence_loss": 1.6,
            "diffusion_loss": 2.4,
            "pdockq_loss": 3.2,
            "membrane_plane_loss": 4.0,
            "interaction_auxiliary_loss": 4.8,
            "loss": 8.0,
        }
    )
    weighted_loss.backward()
    assert loss.grad.item() == pytest.approx(0.8)


def test_non_physppi_loss_is_unchanged():
    loss = torch.tensor(3.0)
    breakdown = {"loss": torch.tensor(3.0)}

    result_loss, result_breakdown = apply_physppi_loss_weight(
        {},
        loss,
        breakdown,
    )

    assert result_loss is loss
    assert result_breakdown is breakdown
