from __future__ import annotations

from types import SimpleNamespace

import torch

from openfold3.projects.of3_all_atom.model import OpenFold3
from openfold3.projects.of3_all_atom.physppi.model.membrane_geometry import (
    DirectedMembranePlaneEstimator,
    MembraneDiffusionConditioner,
    gather_backbone_triplets,
    gather_token_centers,
)
from openfold3.projects.of3_all_atom.physppi.model.tm_topology import TMTopologyEncoder


def _geometry_case(*, backbone_order_reversed: bool = False) -> dict[str, torch.Tensor]:
    # Each residue has N, CA and C atoms.  The ordered triplets define a frame
    # without any global sign canonicalization.
    atom_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
        ]
    ).unsqueeze(0)
    backbone = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    if backbone_order_reversed:
        backbone = backbone[:, [2, 1, 0]]
    return {
        "token_repr": torch.arange(3 * 16, dtype=torch.float32).reshape(1, 3, 16)
        / 16.0,
        "monomer_atom_positions": atom_positions,
        "monomer_atom_mask": torch.ones(1, atom_positions.shape[1], dtype=torch.bool),
        "token_center_atom_index": torch.tensor([[1, 4, 7]]),
        "backbone_atom_indices": backbone.unsqueeze(0),
        "backbone_frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "token_mask": torch.ones(1, 3, dtype=torch.bool),
        "tm_token_mask": torch.ones(1, 3, dtype=torch.bool),
        "membrane_on": torch.tensor([True]),
    }


def _fixed_estimator() -> DirectedMembranePlaneEstimator:
    estimator = DirectedMembranePlaneEstimator(c_s=16)
    with torch.no_grad():
        estimator.vector_coeff.weight.zero_()
        estimator.vector_coeff.bias[:] = torch.tensor([0.0, 0.0, 1.0])
        estimator.vector_weight.weight.zero_()
        estimator.vector_weight.bias.zero_()
        estimator.offset_head.weight.zero_()
        estimator.offset_head.bias.zero_()
        estimator.normal_error.weight.zero_()
        estimator.normal_error.bias.copy_(torch.arange(16, dtype=torch.float32))
        estimator.offset_error.weight.zero_()
        estimator.offset_error.bias.copy_(torch.arange(31, dtype=torch.float32))
    return estimator


def _rotation() -> torch.Tensor:
    q, _ = torch.linalg.qr(
        torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    )
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_directed_plane_estimator_is_se3_equivariant():
    estimator = _fixed_estimator()
    base = _geometry_case()
    rotation = _rotation()
    translation = torch.tensor([3.0, -2.0, 4.0])
    transformed = dict(base)
    transformed["monomer_atom_positions"] = (
        base["monomer_atom_positions"] @ rotation + translation
    )

    first = estimator(**base)
    second = estimator(**transformed)

    assert torch.allclose(second.normal, first.normal @ rotation, atol=1e-5)
    assert torch.allclose(
        second.offset,
        first.offset + (first.normal @ rotation * translation).sum(-1),
        atol=1e-5,
    )
    assert torch.allclose(second.signed_depth, first.signed_depth, atol=1e-5)
    assert torch.allclose(
        second.normal_error_logits, first.normal_error_logits, atol=1e-5
    )
    assert torch.allclose(
        second.offset_error_logits, first.offset_error_logits, atol=1e-5
    )
    assert estimator.normal_error.out_features == 16


def test_directed_plane_estimator_preserves_large_translation_under_bf16_autocast():
    estimator = _fixed_estimator()
    base = _geometry_case()
    translated = dict(base)
    translated["monomer_atom_positions"] = base[
        "monomer_atom_positions"
    ] + torch.tensor([10000.0, -7000.0, 4000.0])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        first = estimator(**base)
        second = estimator(**translated)

    torch.testing.assert_close(second.normal, first.normal, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        second.signed_depth, first.signed_depth, atol=2e-3, rtol=2e-3
    )


def test_ordered_backbone_frame_fixes_normal_sign():
    estimator = _fixed_estimator()
    result = estimator(**_geometry_case())
    flipped = estimator(**_geometry_case(backbone_order_reversed=True))
    assert torch.allclose(result.normal, -flipped.normal, atol=1e-5)


def test_invalid_atom_indices_hard_bypass_plane_prediction():
    estimator = _fixed_estimator()
    case = _geometry_case()
    case["token_center_atom_index"] = torch.full((1, 3), -1)
    case["backbone_atom_indices"] = torch.full((1, 3, 3), -1)
    output = estimator(**case)

    assert torch.equal(output.valid, torch.tensor([False]))
    assert torch.count_nonzero(output.normal) == 0
    assert torch.count_nonzero(output.offset) == 0
    assert torch.count_nonzero(output.signed_depth) == 0
    assert torch.count_nonzero(output.normal_error_logits) == 0
    assert torch.count_nonzero(output.offset_error_logits) == 0


def test_monomer_atom_mask_hard_bypasses_plane_prediction():
    estimator = _fixed_estimator()
    case = _geometry_case()
    case["monomer_atom_mask"][:, 1] = False
    output = estimator(**case)
    assert torch.equal(output.valid, torch.tensor([False]))


def test_gather_helpers_return_zero_for_invalid_indices():
    positions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    centers = gather_token_centers(positions, torch.tensor([[0, -1, 4]]))
    center_valid = torch.tensor([[True, False, False]])
    assert torch.equal(centers[0, 1:], torch.zeros(2, 3))

    triplets = gather_backbone_triplets(
        positions, torch.tensor([[[0, 1, 2], [0, -1, 1]]])
    )
    assert torch.equal(
        torch.stack([part[0, 1] for part in triplets]), torch.zeros(3, 3)
    )
    assert torch.equal(center_valid, torch.tensor([[True, False, False]]))


def test_membrane_conditioner_is_identity_for_non_membrane_batch():
    bridge = MembraneDiffusionConditioner(c_s=16, c_z=8)
    si = torch.randn(1, 3, 16)
    zij = torch.randn(1, 3, 3, 8)
    out_si, out_z = bridge(
        si=si,
        zij=zij,
        batch={"physppi_membrane_on": torch.tensor([False])},
        token_mask=torch.ones(1, 3),
        atom_positions=torch.zeros(1, 3, 3),
    )
    assert out_si is si
    assert out_z is zij


def test_membrane_conditioner_bypasses_invalid_predicted_plane():
    bridge = MembraneDiffusionConditioner(c_s=16, c_z=8)
    si = torch.randn(1, 3, 16)
    zij = torch.randn(1, 3, 3, 8)
    case = _geometry_case()
    prediction = _fixed_estimator()(**case)._replace(valid=torch.tensor([False]))
    batch = {
        "physppi_membrane_on": torch.tensor([True]),
        "physppi_predicted_plane": prediction,
    }

    out_si, out_z = bridge(
        si=si,
        zij=zij,
        batch=batch,
        token_mask=case["token_mask"],
        atom_positions=torch.randn(1, 9, 3),
    )

    assert out_si is si
    assert out_z is zij


def test_membrane_conditioner_bypasses_empty_opm_style_batch_without_plane():
    bridge = MembraneDiffusionConditioner(c_s=16, c_z=8)
    si = torch.randn(1, 3, 16)
    zij = torch.randn(1, 3, 3, 8)

    out_si, out_z = bridge(
        si=si,
        zij=zij,
        batch={},
        token_mask=torch.ones(1, 3, dtype=torch.bool),
        atom_positions=torch.zeros(1, 3, 3),
    )

    assert out_si is si
    assert out_z is zij


def test_membrane_conditioner_never_builds_a_plane_from_gt_or_noisy_coordinates():
    bridge = MembraneDiffusionConditioner(c_s=16, c_z=8)
    si = torch.randn(1, 3, 16)
    zij = torch.randn(1, 3, 3, 8)
    batch = {
        "physppi_membrane_on": torch.tensor([True]),
        "physppi_membrane_gt_normal": torch.tensor([[0.0, 0.0, 1.0]]),
        "physppi_membrane_gt_point": torch.zeros(1, 3),
        "physppi_membrane_gt_valid": torch.tensor([True]),
    }
    out_si, out_z = bridge(
        si=si,
        zij=zij,
        batch=batch,
        token_mask=torch.ones(1, 3, dtype=torch.bool),
        atom_positions=torch.randn(1, 9, 3),
    )
    assert out_si is si
    assert out_z is zij


def test_membrane_conditioner_uses_signed_depth_only_when_gate_is_open():
    bridge = MembraneDiffusionConditioner(c_s=16, c_z=8)
    bridge.gate.data.fill_(0.5)
    si = torch.zeros(1, 3, 16)
    zij = torch.zeros(1, 3, 3, 8)
    prediction = _fixed_estimator()(**_geometry_case())
    out_si, out_z = bridge(
        si=si,
        zij=zij,
        batch={
            "physppi_membrane_on": torch.tensor([True]),
            "physppi_predicted_plane": prediction,
        },
        token_mask=torch.ones(1, 3),
        atom_positions=torch.zeros(1, 3, 3),
    )
    assert torch.count_nonzero(out_si) > 0
    assert out_z is zij


def test_membrane_conditioner_is_exact_zero_initialized_with_gate_gradient():
    bridge = MembraneDiffusionConditioner(c_s=16, c_z=8)
    with torch.no_grad():
        bridge.depth_proj[-1].weight.zero_()
        bridge.depth_proj[-1].bias.fill_(1.0)
    si = torch.zeros(1, 3, 16)
    prediction = _fixed_estimator()(**_geometry_case())
    out_si, _ = bridge(
        si=si,
        zij=torch.zeros(1, 3, 3, 8),
        batch={
            "physppi_membrane_on": torch.tensor([True]),
            "physppi_predicted_plane": prediction,
        },
        token_mask=torch.ones(1, 3, dtype=torch.bool),
        atom_positions=torch.randn(1, 9, 3),
    )
    assert torch.equal(out_si, si)
    out_si.sum().backward()
    assert bridge.gate.grad is not None
    assert torch.count_nonzero(bridge.gate.grad) == 1
    assert bridge.depth_cutoff == 30.0
    assert bridge.num_depth_bins == 16


def test_membrane_conditioner_broadcasts_one_plane_over_48_diffusion_samples():
    bridge = MembraneDiffusionConditioner(c_s=16, c_z=8)
    bridge.gate.data.fill_(0.5)
    si = torch.zeros(2, 48, 3, 16)
    zij = torch.randn(2, 1, 3, 3, 8)
    base_prediction = _fixed_estimator()(**_geometry_case())
    prediction = base_prediction._replace(
        normal=base_prediction.normal.expand(2, -1).unsqueeze(1),
        offset=base_prediction.offset.expand(2).unsqueeze(1),
        signed_depth=base_prediction.signed_depth.expand(2, -1).unsqueeze(1),
        normal_error_logits=base_prediction.normal_error_logits.expand(2, -1).unsqueeze(
            1
        ),
        offset_error_logits=base_prediction.offset_error_logits.expand(2, -1).unsqueeze(
            1
        ),
        valid=torch.tensor([[True], [False]]),
    )

    out_si, out_z = bridge(
        si=si,
        zij=zij,
        batch={
            "physppi_membrane_on": torch.tensor([[True], [True]]),
            "physppi_predicted_plane": prediction,
        },
        token_mask=torch.ones(2, 1, 3, dtype=torch.bool),
        atom_positions=torch.randn(2, 48, 9, 3),
    )

    assert torch.count_nonzero(out_si[0]) > 0
    assert torch.equal(out_si[1], si[1])
    assert out_z is zij


def test_tm_encoder_is_exact_zero_initialized_and_skips_inactive_projection():
    encoder = TMTopologyEncoder(c_s=16)
    with torch.no_grad():
        encoder.proj[-1].weight.zero_()
        encoder.proj[-1].bias.fill_(1.0)
    delta, membrane_on, _ = encoder(
        tm_token_mask=torch.tensor([[True, False, True]]),
        tm_active=torch.tensor([True]),
        token_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(membrane_on, torch.tensor([True]))
    delta.sum().backward()
    assert encoder.gate.grad is not None
    assert torch.count_nonzero(encoder.gate.grad) == 1

    calls = []
    handle = encoder.proj.register_forward_hook(
        lambda module, args, output: calls.append((module, args, output))
    )
    inactive, membrane_on, _ = encoder(
        tm_token_mask=torch.tensor([[True, False, True]]),
        tm_active=torch.tensor([False]),
        token_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    handle.remove()
    assert calls == []
    assert torch.equal(inactive, torch.zeros_like(inactive))
    assert torch.equal(membrane_on, torch.tensor([False]))


def test_model_plane_wiring_passes_only_monomer_input_geometry():
    captured = {}

    class RecordingEstimator:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return None

    model = SimpleNamespace(membrane_plane_estimator=RecordingEstimator())
    case = _geometry_case()
    batch = {
        "physppi_monomer_atom_positions": case["monomer_atom_positions"],
        "physppi_monomer_atom_mask": case["monomer_atom_mask"],
        "physppi_token_center_atom_index": case["token_center_atom_index"],
        "physppi_backbone_atom_indices": case["backbone_atom_indices"],
        "physppi_backbone_frame_mask": case["backbone_frame_mask"],
        "physppi_tm_token_mask": case["tm_token_mask"],
        "physppi_membrane_on": case["membrane_on"],
        "ground_truth": {"atom_positions": torch.randn(1, 9, 3)},
    }
    OpenFold3._predict_physppi_membrane_plane(
        model,
        token_repr=case["token_repr"],
        batch=batch,
        token_mask=case["token_mask"],
    )
    assert captured["monomer_atom_positions"] is batch["physppi_monomer_atom_positions"]
    assert captured["monomer_atom_mask"] is batch["physppi_monomer_atom_mask"]
    assert "atom_positions" not in captured
