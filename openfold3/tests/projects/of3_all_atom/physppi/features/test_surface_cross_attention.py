import importlib

import pytest
import torch
from torch import nn


def _surface_module():
    return importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.model.surface_encoder"
    )


def _position_module():
    try:
        return importlib.import_module(
            "openfold3.projects.of3_all_atom.physppi.features.surface_position"
        )
    except ModuleNotFoundError:
        pytest.fail("surface_position module is missing")


def test_shared_surface_position_encoding_is_exactly_64_dimensional():
    encode = _position_module().encode_surface_position
    coords = torch.tensor([[[[32.0 / 15.0, 0.0, 0.0]]]])
    encoded = encode(coords)
    assert encoded.shape == (1, 1, 1, 64)
    assert torch.isclose(encoded[..., 48], torch.exp(torch.tensor(-1.0))).all()


def test_surface_position_clips_only_rbf_radius_above_32_angstrom():
    encode = _position_module().encode_surface_position
    at_cutoff = encode(torch.tensor([[32.0, 0.0, 0.0]]))
    above_cutoff = encode(torch.tensor([[64.0, 0.0, 0.0]]))
    torch.testing.assert_close(above_cutoff[..., 48:], at_cutoff[..., 48:])
    assert not torch.equal(above_cutoff[..., :48], at_cutoff[..., :48])


def test_surface_encoder_has_exact_single_attention_architecture():
    encoder = _surface_module().SurfaceTokenEncoder(c_s=384)
    assert len(encoder.point_proj) == 3
    assert isinstance(encoder.point_proj[0], nn.Linear)
    assert encoder.point_proj[0].in_features == 70
    assert encoder.point_proj[0].out_features == 384
    assert isinstance(encoder.point_proj[1], nn.LayerNorm)
    assert isinstance(encoder.point_proj[2], nn.GELU)
    assert isinstance(encoder.query_norm, nn.LayerNorm)
    assert isinstance(encoder.query_position, nn.Linear)
    assert encoder.query_position.in_features == 64
    assert encoder.query_position.out_features == 384
    assert isinstance(encoder.attention, nn.MultiheadAttention)
    assert encoder.attention.embed_dim == 384
    assert encoder.attention.num_heads == 8
    assert encoder.attention.dropout == 0.0
    assert encoder.attention.batch_first is True
    assert isinstance(encoder.output_norm, nn.LayerNorm)
    assert isinstance(encoder.output_proj, nn.Linear)
    assert encoder.gate.shape == torch.Size([])
    assert torch.equal(encoder.gate, torch.zeros_like(encoder.gate))


def test_surface_encoder_uses_one_token_query_and_mesh_points_as_key_value():
    encoder = _surface_module().SurfaceTokenEncoder(c_s=16)
    captured = {}

    def capture(module, args, kwargs):
        del module, args
        captured["query"] = kwargs["query"].detach().clone()
        captured["key"] = kwargs["key"].detach().clone()
        captured["value"] = kwargs["value"].detach().clone()
        captured["key_padding_mask"] = kwargs["key_padding_mask"].detach().clone()

    handle = encoder.attention.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.no_grad():
        encoder.gate.fill_(0.5)
    encoder(
        s_init=torch.randn(1, 2, 16),
        local_coords=torch.randn(1, 2, 5, 3),
        features=torch.randn(1, 2, 5, 6),
        surface_mask=torch.ones(1, 2, 5, dtype=torch.bool),
        surface_charge_mask=torch.ones(1, 2, 5, dtype=torch.bool),
        surface_active=torch.tensor([True]),
        token_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    handle.remove()
    assert captured["query"].shape == (2, 1, 16)
    assert captured["key"].shape == (2, 5, 16)
    assert captured["value"].shape == (2, 5, 16)
    assert captured["key_padding_mask"].shape == (2, 5)


def test_surface_encoder_is_pair_independent_and_returns_only_single_delta():
    encoder = _surface_module().SurfaceTokenEncoder(c_s=16)
    with torch.no_grad():
        encoder.gate.fill_(0.5)
    delta = encoder(
        s_init=torch.randn(1, 2, 16),
        local_coords=torch.randn(1, 2, 3, 3),
        features=torch.randn(1, 2, 3, 6),
        surface_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        surface_charge_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        surface_active=torch.tensor([True]),
        token_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert delta.shape == (1, 2, 16)
