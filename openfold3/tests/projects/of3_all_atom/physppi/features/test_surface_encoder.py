import importlib

import pytest
import torch


def _encoder_class():
    module = importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.model.surface_encoder"
    )
    return module.SurfaceTokenEncoder


def _inputs(*, active: bool = True, any_points: bool = True):
    return {
        "s_init": torch.randn(1, 2, 16),
        "local_coords": torch.randn(1, 2, 3, 3),
        "features": torch.randn(1, 2, 3, 6),
        "surface_mask": torch.full((1, 2, 3), any_points, dtype=torch.bool),
        "surface_charge_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "surface_active": torch.tensor([active]),
        "token_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def test_surface_initial_output_is_bitwise_zero_with_connected_gate_gradient():
    encoder = _encoder_class()(c_s=16)
    with torch.no_grad():
        encoder.output_proj.weight.zero_()
        encoder.output_proj.bias.fill_(1.0)
    delta = encoder(**_inputs())
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None
    assert torch.count_nonzero(encoder.gate.grad) == 1


@pytest.mark.parametrize(
    ("active", "any_points"),
    [(False, True), (True, False)],
)
def test_surface_inactive_or_empty_skips_attention_and_is_connected_zero(
    active, any_points
):
    encoder = _encoder_class()(c_s=16)
    calls = []
    handle = encoder.attention.register_forward_hook(
        lambda module, args, output: calls.append((module, args, output))
    )
    delta = encoder(**_inputs(active=active, any_points=any_points))
    handle.remove()
    assert calls == []
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert encoder.gate.grad is not None


def test_surface_attention_never_executes_an_all_masked_token_row():
    encoder = _encoder_class()(c_s=16)
    captured = {}

    def capture(module, args, kwargs):
        del module, args
        captured["query_rows"] = kwargs["query"].shape[0]
        captured["masked"] = kwargs["key_padding_mask"].detach().clone()

    handle = encoder.attention.register_forward_pre_hook(capture, with_kwargs=True)
    inputs = _inputs()
    inputs["surface_mask"][0, 1] = False
    with torch.no_grad():
        encoder.gate.fill_(0.5)
    delta = encoder(**inputs)
    handle.remove()
    assert captured["query_rows"] == 1
    assert not captured["masked"].all(dim=-1).any()
    assert torch.count_nonzero(delta[0, 1]) == 0


def test_surface_charge_mask_zeroes_only_charge_before_point_projection():
    encoder = _encoder_class()(c_s=16)
    captured = {}

    def capture(module, args):
        del module
        captured["point_input"] = args[0].detach().clone()

    handle = encoder.point_proj[0].register_forward_pre_hook(capture)
    inputs = _inputs()
    inputs["features"] = torch.tensor(
        [
            [
                [[7.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3,
                [[8.0, 9.0, 10.0, 11.0, 12.0, 13.0]] * 3,
            ]
        ]
    )
    inputs["surface_charge_mask"][0, 0, 0] = False
    with torch.no_grad():
        encoder.gate.fill_(0.5)
    encoder(**inputs)
    handle.remove()
    attrs = captured["point_input"][..., 64:]
    assert attrs[0, 0, 0] == 0
    assert torch.equal(attrs[0, 0, 1:], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    assert attrs[0, 1, 0] == 7.0


def test_surface_open_gate_backpropagates_into_point_encoder():
    encoder = _encoder_class()(c_s=16)
    with torch.no_grad():
        encoder.gate.fill_(0.5)
    loss = encoder(**_inputs()).square().sum()
    loss.backward()
    grad = encoder.point_proj[0].weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad) > 0
