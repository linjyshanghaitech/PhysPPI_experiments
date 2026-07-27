from types import SimpleNamespace

import torch

from openfold3.projects.of3_all_atom.constants import TRAIN_LOSSES
from openfold3.projects.of3_all_atom.physppi.model.interaction_auxiliary import (
    InteractionDistanceHead,
    _build_pair_interaction_auxiliary,
    interaction_auxiliary_loss,
    validate_interaction_endpoints,
)


def test_interaction_auxiliary_loss_is_registered_for_epoch_logging():
    assert "interaction_auxiliary_loss" in TRAIN_LOSSES


def test_endpoint_validation_rejects_wrong_atom_names():
    records = [
        SimpleNamespace(i=1, j=2, atom_1="ND1", atom_2="O", distance=3.0),
        SimpleNamespace(i=1, j=2, atom_1="CA", atom_2="O", distance=3.0),
    ]
    lookup = {(1, 2, "a1"): "ND1", (1, 2, "a2"): "O"}
    result = validate_interaction_endpoints(records, lookup)
    assert result.valid_mask.tolist() == [True, False]


def test_empty_interaction_batch_has_exact_zero_auxiliary_loss():
    prediction = torch.randn(1, 3, 3, requires_grad=True)
    loss = interaction_auxiliary_loss(
        batch={
            "physppi_interaction_aux_valid": torch.zeros(
                1, 3, 3, dtype=torch.bool
            ),
            "physppi_interaction_distance": torch.zeros(1, 3, 3),
        },
        output={"interaction_distance_pred": prediction},
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_valid_interaction_distance_is_supervised():
    prediction = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True
    )
    loss = interaction_auxiliary_loss(
        batch={
            "physppi_interaction_aux_valid": torch.tensor(
                [[[False, True], [True, False]]]
            ),
            "physppi_interaction_distance": torch.tensor(
                [[[0.0, 2.0], [2.0, 0.0]]]
            ),
        },
        output={"interaction_distance_pred": prediction},
        distance_weight=2.0,
    )
    assert torch.isclose(loss, torch.tensor(1.0))


def test_interaction_distance_head_reads_and_symmetrizes_final_pair_state():
    head = InteractionDistanceHead(pair_dim=4)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        head.projection.bias.zero_()
    final_pair = torch.zeros(1, 3, 3, 4)
    final_pair[0, 0, 1, 0] = 2.0
    final_pair[0, 1, 0, 0] = 4.0
    final_pair[0, 0, 2, 0] = 100.0

    prediction = head(
        final_pair,
        token_mask=torch.tensor([[True, True, False]]),
    )

    assert prediction.shape == (1, 3, 3)
    assert torch.equal(prediction, prediction.transpose(-1, -2))
    assert torch.equal(prediction[0, :2, :2], torch.tensor([[0.0, 3.0], [3.0, 0.0]]))
    assert torch.count_nonzero(prediction[0, 2]) == 0
    assert torch.count_nonzero(prediction[0, :, 2]) == 0


def test_pair_interaction_targets_are_symmetric_and_average_duplicate_edges():
    target, valid = _build_pair_interaction_auxiliary(
        (
            (0, 2, 0, 3.0, ("N",), ("O",)),
            (0, 2, 1, 5.0, ("NZ",), ("OE1",)),
            (1, 3, 2, 7.0, ("C",), ("C",)),
        ),
        n_tokens=5,
    )

    assert target.shape == (5, 5)
    assert valid.shape == (5, 5)
    assert torch.equal(target, target.transpose(-1, -2))
    assert torch.equal(valid, valid.transpose(-1, -2))
    assert target[0, 2].item() == 4.0
    assert target[1, 3].item() == 7.0
    assert valid.sum().item() == 4
    assert not valid.diagonal().any()
