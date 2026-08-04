import importlib

import pytest
import torch
from torch import nn


def test_physppi_package_exports_interface_residue_encoder():
    package = importlib.import_module("openfold3.projects.of3_all_atom.physppi")
    assert hasattr(package, "InterfaceResidueEncoder")
    assert hasattr(package, "InteractionPairEncoder")
    assert not hasattr(package, "ContactPairEncoder")
    assert not hasattr(package, "ContactTokenEncoder")
    assert not hasattr(package, "TokenContactEncoder")


def _interface_encoder_class():
    try:
        module = importlib.import_module(
            "openfold3.projects.of3_all_atom.physppi.model."
            "interface_residue_encoder"
        )
    except ModuleNotFoundError:
        pytest.fail("interface_residue_encoder module is missing")
    assert hasattr(module, "InterfaceResidueEncoder")
    return module.InterfaceResidueEncoder


def test_interface_residue_encoder_matches_tm_topology_structure():
    tm_module = importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.model.tm_topology"
    )
    encoder = _interface_encoder_class()(c_s=8)
    tm_encoder = tm_module.TMTopologyEncoder(c_s=8)

    assert len(encoder.proj) == 4
    assert isinstance(encoder.proj[0], nn.Linear)
    assert encoder.proj[0].in_features == 1
    assert encoder.proj[0].out_features == 8
    assert isinstance(encoder.proj[1], nn.LayerNorm)
    assert isinstance(encoder.proj[2], nn.GELU)
    assert isinstance(encoder.proj[3], nn.Linear)
    assert encoder.proj[3].in_features == 8
    assert encoder.proj[3].out_features == 8
    assert encoder.gate.shape == torch.Size([])
    assert torch.equal(encoder.gate, torch.zeros_like(encoder.gate))
    assert (
        sum(parameter.numel() for parameter in encoder.parameters())
        == sum(parameter.numel() for parameter in tm_encoder.parameters())
    )


def test_interface_residue_encoder_is_bitwise_zero_at_initialization():
    encoder = _interface_encoder_class()(c_s=8)
    with torch.no_grad():
        encoder.proj[-1].weight.zero_()
        encoder.proj[-1].bias.fill_(1.0)
    delta = encoder(
        payload=torch.tensor([[True, True, False]]),
        active=torch.tensor([True]),
        token_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None
    assert torch.count_nonzero(encoder.gate.grad) == 1


def test_interface_residue_encoder_marks_only_interface_tokens_and_masks_padding():
    encoder = _interface_encoder_class()(c_s=8)
    with torch.no_grad():
        encoder.proj[-1].weight.zero_()
        encoder.proj[-1].bias.fill_(1.0)
        encoder.gate.fill_(0.5)
    token_mask = torch.tensor([[True, True, False]])
    delta = encoder(
        payload=torch.tensor([[True, False, True]]),
        active=torch.tensor([True]),
        token_mask=token_mask,
    )
    assert torch.count_nonzero(delta[0, 0]) > 0
    assert torch.count_nonzero(delta[0, 1]) == 0
    # Padding token stays zero even though the payload marks it.
    assert torch.count_nonzero(delta[0, 2]) == 0


@pytest.mark.parametrize("payload", [None, torch.tensor([[True, False]])])
def test_absent_or_all_padding_payload_is_exact_zero(payload):
    encoder = _interface_encoder_class()(c_s=8)
    token_mask = torch.tensor([[True, False]])
    delta = encoder(
        payload=payload,
        active=torch.tensor([True]),
        token_mask=token_mask,
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None


def test_inactive_interface_residue_encoder_does_not_execute_dense_projection():
    encoder = _interface_encoder_class()(c_s=8)
    calls = []
    handle = encoder.proj.register_forward_hook(
        lambda module, args, output: calls.append((module, args, output))
    )
    delta = encoder(
        payload=torch.tensor([[True, True]]),
        active=torch.tensor([False]),
        token_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    handle.remove()
    assert calls == []
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None
