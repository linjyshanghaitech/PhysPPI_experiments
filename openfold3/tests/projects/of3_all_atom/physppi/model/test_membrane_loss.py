import torch

from openfold3.core.loss.membrane import directed_membrane_plane_loss
from openfold3.projects.of3_all_atom.constants import TRAIN_LOSSES, VAL_LOSSES


def test_membrane_plane_loss_is_registered_for_epoch_logging():
    assert "membrane_plane_loss" in TRAIN_LOSSES
    assert "membrane_plane_loss" in VAL_LOSSES


def test_invalid_membrane_label_is_exact_zero():
    normal = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    offset = torch.tensor([2.0], requires_grad=True)
    loss = directed_membrane_plane_loss(
        batch={
            "physppi_membrane_gt_normal": torch.tensor([[1.0, 0.0, 0.0]]),
            "physppi_membrane_gt_point": torch.zeros(1, 3),
            "physppi_membrane_gt_valid": torch.tensor([False]),
        },
        output={
            "physppi_predicted_membrane_normal": normal,
            "physppi_predicted_membrane_offset": offset,
            "physppi_predicted_membrane_valid": torch.tensor([False]),
        },
    )
    assert loss.item() == 0.0
    loss.backward()
    assert normal.grad is not None and offset.grad is not None


def test_directed_plane_loss_uses_normal_direction_and_offset():
    loss = directed_membrane_plane_loss(
        batch={
            "physppi_membrane_gt_normal": torch.tensor([[0.0, 0.0, 1.0]]),
            "physppi_membrane_gt_point": torch.tensor([[0.0, 0.0, 2.0]]),
            "physppi_membrane_gt_valid": torch.tensor([True]),
        },
        output={
            "physppi_predicted_membrane_normal": torch.tensor([[0.0, 0.0, 1.0]]),
            "physppi_predicted_membrane_offset": torch.tensor([0.0]),
            "physppi_predicted_membrane_valid": torch.tensor([True]),
        },
    )
    assert loss.item() > 0.0


def test_directed_loss_penalizes_the_geometrically_equivalent_flipped_plane():
    batch = {
        "physppi_membrane_gt_normal": torch.tensor([[0.0, 0.0, 1.0]]),
        "physppi_membrane_gt_point": torch.tensor([[0.0, 0.0, 2.0]]),
        "physppi_membrane_gt_valid": torch.tensor([True]),
    }
    aligned = directed_membrane_plane_loss(
        batch=batch,
        output={
            "physppi_predicted_membrane_normal": torch.tensor([[0.0, 0.0, 1.0]]),
            "physppi_predicted_membrane_offset": torch.tensor([2.0]),
            "physppi_predicted_membrane_valid": torch.tensor([True]),
        },
    )
    flipped = directed_membrane_plane_loss(
        batch=batch,
        output={
            "physppi_predicted_membrane_normal": torch.tensor([[0.0, 0.0, -1.0]]),
            "physppi_predicted_membrane_offset": torch.tensor([-2.0]),
            "physppi_predicted_membrane_valid": torch.tensor([True]),
        },
    )

    assert torch.isfinite(aligned)
    assert torch.equal(aligned, torch.zeros_like(aligned))
    assert flipped > aligned


def test_invalid_predicted_plane_is_connected_exact_zero():
    normal = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    offset = torch.tensor([2.0], requires_grad=True)
    loss = directed_membrane_plane_loss(
        batch={
            "physppi_membrane_gt_normal": torch.tensor([[0.0, 0.0, 1.0]]),
            "physppi_membrane_gt_point": torch.tensor([[0.0, 0.0, 2.0]]),
            "physppi_membrane_gt_valid": torch.tensor([True]),
        },
        output={
            "physppi_predicted_membrane_normal": normal,
            "physppi_predicted_membrane_offset": offset,
            "physppi_predicted_membrane_valid": torch.tensor([False]),
        },
    )
    assert loss.item() == 0.0
    loss.backward()
    assert normal.grad is not None
    assert offset.grad is not None


def test_invalid_nan_prediction_is_connected_exact_zero():
    normal = torch.tensor([[float("nan"), 0.0, 1.0]], requires_grad=True)
    offset = torch.tensor([float("nan")], requires_grad=True)
    loss = directed_membrane_plane_loss(
        batch={
            "physppi_membrane_gt_normal": torch.tensor([[0.0, 0.0, 1.0]]),
            "physppi_membrane_gt_point": torch.tensor([[0.0, 0.0, 2.0]]),
            "physppi_membrane_gt_valid": torch.tensor([True]),
        },
        output={
            "physppi_predicted_membrane_normal": normal,
            "physppi_predicted_membrane_offset": offset,
            "physppi_predicted_membrane_valid": torch.tensor([False]),
        },
    )
    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert normal.grad is not None
    assert offset.grad is not None


def test_invalid_nan_row_does_not_poison_valid_row_or_gradients():
    normal = torch.tensor(
        [[0.0, 0.0, 1.0], [float("nan"), 0.0, 1.0]], requires_grad=True
    )
    offset = torch.tensor([2.0, float("nan")], requires_grad=True)
    loss = directed_membrane_plane_loss(
        batch={
            "physppi_membrane_gt_normal": torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            "physppi_membrane_gt_point": torch.tensor(
                [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
            ),
            "physppi_membrane_gt_valid": torch.tensor([True, True]),
        },
        output={
            "physppi_predicted_membrane_normal": normal,
            "physppi_predicted_membrane_offset": offset,
            "physppi_predicted_membrane_valid": torch.tensor([True, False]),
        },
    )
    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert torch.isfinite(normal.grad).all()
    assert torch.isfinite(offset.grad).all()
