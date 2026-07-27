import importlib

import pytest
import torch
from torch import nn


def test_physppi_package_exports_only_semantic_pair_adapters():
    package = importlib.import_module("openfold3.projects.of3_all_atom.physppi")
    assert hasattr(package, "ContactPairEncoder")
    assert hasattr(package, "InteractionPairEncoder")
    assert not hasattr(package, "SparsePairConstraintEncoder")
    assert not hasattr(package, "ContactTokenEncoder")
    assert not hasattr(package, "TokenContactEncoder")


def _interaction_payload(size: int, *edges: tuple[int, int, int]) -> torch.Tensor:
    payload = torch.zeros(1, size, size, 7, dtype=torch.bool)
    for left, right, channel in edges:
        payload[0, left, right, channel] = True
    return payload


def _contact_encoder_class():
    try:
        module = importlib.import_module(
            "openfold3.projects.of3_all_atom.physppi.model.contact_encoder"
        )
    except ModuleNotFoundError:
        pytest.fail("contact_encoder module is missing")
    assert hasattr(module, "ContactPairEncoder")
    return module.ContactPairEncoder


def _interaction_encoder_class():
    module = importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.model.pair_constraints"
    )
    assert hasattr(module, "InteractionPairEncoder")
    return module.InteractionPairEncoder


@pytest.mark.parametrize(
    ("encoder_class", "width"),
    [(_contact_encoder_class, 1), (_interaction_encoder_class, 7)],
)
def test_pair_adapter_has_exact_projection_and_zero_gate(encoder_class, width):
    encoder = encoder_class()(c_z=8)
    assert len(encoder.proj) == 4
    assert isinstance(encoder.proj[0], nn.Linear)
    assert encoder.proj[0].in_features == width
    assert encoder.proj[0].out_features == 8
    assert isinstance(encoder.proj[1], nn.LayerNorm)
    assert isinstance(encoder.proj[2], nn.GELU)
    assert isinstance(encoder.proj[3], nn.Linear)
    assert encoder.proj[3].in_features == 8
    assert encoder.proj[3].out_features == 8
    assert encoder.gate.shape == torch.Size([])
    assert torch.equal(encoder.gate, torch.zeros_like(encoder.gate))


@pytest.mark.parametrize(
    ("encoder_class", "payload"),
    [
        (
            _contact_encoder_class,
            torch.tensor(
                [[[False, True, False], [False, False, False], [False, False, False]]]
            ),
        ),
        (
            _interaction_encoder_class,
            _interaction_payload(3, (0, 1, 4)),
        ),
    ],
)
def test_pair_adapter_is_bitwise_zero_at_initialization_with_gate_gradient(
    encoder_class, payload
):
    encoder = encoder_class()(c_z=8)
    with torch.no_grad():
        encoder.proj[-1].weight.zero_()
        encoder.proj[-1].bias.fill_(1.0)
    pair_mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    delta = encoder(
        payload=payload,
        active=torch.tensor([True]),
        pair_mask=pair_mask,
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None
    assert torch.count_nonzero(encoder.gate.grad) == 1


@pytest.mark.parametrize(
    ("encoder_class", "payload"),
    [
        (
            _contact_encoder_class,
            torch.tensor(
                [[[False, True, False], [False, False, True], [False, False, False]]]
            ),
        ),
        (
            _interaction_encoder_class,
            _interaction_payload(3, (0, 1, 4), (1, 2, 2)),
        ),
    ],
)
def test_open_pair_adapter_is_symmetric_and_masks_diagonal_and_padding(
    encoder_class, payload
):
    encoder = encoder_class()(c_z=8)
    with torch.no_grad():
        encoder.proj[-1].weight.zero_()
        encoder.proj[-1].bias.fill_(1.0)
        encoder.gate.fill_(0.5)
    pair_mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    delta = encoder(
        payload=payload,
        active=torch.tensor([True]),
        pair_mask=pair_mask,
    )
    torch.testing.assert_close(delta, delta.transpose(-3, -2))
    assert torch.count_nonzero(delta[0, 0, 1]) > 0
    assert torch.count_nonzero(torch.diagonal(delta, dim1=-3, dim2=-2)) == 0
    assert torch.count_nonzero(delta[0, 2]) == 0
    assert torch.count_nonzero(delta[0, :, 2]) == 0


@pytest.mark.parametrize(
    "encoder_class", [_contact_encoder_class, _interaction_encoder_class]
)
@pytest.mark.parametrize("payload", [None])
def test_absent_pair_payload_is_connected_exact_zero(encoder_class, payload):
    encoder = encoder_class()(c_z=8)
    delta = encoder(
        payload=payload,
        active=torch.tensor([True]),
        pair_mask=torch.ones(1, 3, 3, dtype=torch.bool),
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None


@pytest.mark.parametrize(
    ("encoder_class", "payload"),
    [
        (
            _contact_encoder_class,
            torch.tensor([[[False, True], [True, False]]]),
        ),
        (
            _interaction_encoder_class,
            _interaction_payload(2, (0, 1, 3), (1, 0, 3)),
        ),
    ],
)
def test_inactive_pair_adapter_does_not_execute_dense_projection(
    encoder_class, payload
):
    encoder = encoder_class()(c_z=8)
    calls = []
    handle = encoder.proj.register_forward_hook(
        lambda module, args, output: calls.append((module, args, output))
    )
    delta = encoder(
        payload=payload,
        active=torch.tensor([False]),
        pair_mask=torch.ones(1, 2, 2, dtype=torch.bool),
    )
    handle.remove()
    assert calls == []
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None


def test_interaction_adapter_preserves_seven_channel_multihot_payload():
    encoder = _interaction_encoder_class()(c_z=8)
    recorded = {}

    def record_input(module, args):
        del module
        recorded["payload"] = args[0].detach().clone()

    handle = encoder.proj[0].register_forward_pre_hook(record_input)
    payload = torch.zeros(1, 2, 2, 7, dtype=torch.bool)
    payload[0, 0, 1, 0] = True
    payload[0, 0, 1, 4] = True
    with torch.no_grad():
        encoder.gate.fill_(0.5)
    encoder(
        payload=payload,
        active=torch.tensor([True]),
        pair_mask=torch.ones(1, 2, 2, dtype=torch.bool),
    )
    handle.remove()
    assert recorded["payload"].shape[-1] == 7
    assert recorded["payload"][0, 0, 1, 0] == 1
    assert recorded["payload"][0, 0, 1, 4] == 1
    assert recorded["payload"][0, 1, 0, 0] == 1
    assert recorded["payload"][0, 1, 0, 4] == 1
