import torch
from torch import nn

from openfold3.projects.of3_all_atom.physppi.model.conditioning import (
    OptionalPriorConditioner,
)
from openfold3.projects.of3_all_atom.physppi.model.surface_encoder import (
    SurfaceTokenEncoder,
)
from openfold3.projects.of3_all_atom.physppi.model.tm_topology import TMTopologyEncoder


def _conditioner_batch(*, active: bool) -> dict[str, torch.Tensor]:
    interaction = torch.zeros(1, 3, 3, 7, dtype=torch.bool)
    interaction[0, 1, 2, 2] = True
    interaction[0, 2, 1, 2] = True
    return {
        "physppi_contact_pair_mask": torch.tensor(
            [
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ]
            ]
        ),
        "physppi_contact_active": torch.tensor([active]),
        "physppi_interaction_type_mask": interaction,
        "physppi_interaction_active": torch.tensor([active]),
        "physppi_surface_local_coords": torch.zeros(1, 3, 2, 3),
        "physppi_surface_features": torch.zeros(1, 3, 2, 6),
        "physppi_surface_mask": torch.ones(1, 3, 2, dtype=torch.bool),
        "physppi_surface_charge_mask": torch.ones(1, 3, 2, dtype=torch.bool),
        "physppi_surface_active": torch.tensor([active]),
        "physppi_tm_token_mask": torch.tensor([[True, False, False]]),
        "physppi_tm_active": torch.tensor([active]),
    }


def test_conditioner_initial_pair_adapters_are_exact_zero():
    conditioner = OptionalPriorConditioner(c_s=16, c_z=8)
    s = torch.zeros(1, 3, 16)
    z = torch.zeros(1, 3, 3, 8)
    batch = _conditioner_batch(active=True)
    batch["physppi_surface_active"] = torch.tensor([False])
    batch["physppi_tm_active"] = torch.tensor([False])
    out = conditioner(
        s_init=s,
        z_init=z,
        batch=batch,
        token_mask=torch.ones(1, 3),
        pair_mask=torch.ones(1, 3, 3),
    )
    assert torch.equal(out.s_init, s)
    assert torch.equal(out.z_init, z)
    assert torch.equal(
        out.contact_pair_delta, torch.zeros_like(out.contact_pair_delta)
    )
    assert torch.equal(
        out.interaction_pair_delta, torch.zeros_like(out.interaction_pair_delta)
    )


def test_open_contact_and_interaction_change_only_pair_representation():
    conditioner = OptionalPriorConditioner(c_s=16, c_z=8)
    with torch.no_grad():
        conditioner.contact.gate.fill_(0.5)
        conditioner.interaction.gate.fill_(0.5)
    s = torch.randn(1, 3, 16)
    z = torch.randn(1, 3, 3, 8)
    batch = _conditioner_batch(active=True)
    batch["physppi_surface_active"] = torch.tensor([False])
    batch["physppi_tm_active"] = torch.tensor([False])
    out = conditioner(
        s_init=s,
        z_init=z,
        batch=batch,
        token_mask=torch.ones(1, 3),
        pair_mask=torch.ones(1, 3, 3),
    )
    assert torch.equal(out.s_init, s)
    assert not torch.equal(out.z_init, z)
    torch.testing.assert_close(
        out.z_init - z,
        out.contact_pair_delta + out.interaction_pair_delta,
    )
    torch.testing.assert_close(
        out.contact_pair_delta,
        out.contact_pair_delta.transpose(-3, -2),
    )
    torch.testing.assert_close(
        out.interaction_pair_delta,
        out.interaction_pair_delta.transpose(-3, -2),
    )


def test_inactive_optional_slots_are_exact_noops():
    conditioner = OptionalPriorConditioner(c_s=16, c_z=8)
    s = torch.randn(1, 3, 16)
    z = torch.randn(1, 3, 3, 8)
    out = conditioner(
        s_init=s,
        z_init=z,
        batch=_conditioner_batch(active=False),
        token_mask=torch.ones(1, 3),
        pair_mask=torch.ones(1, 3, 3),
    )
    assert torch.equal(out.s_init, s)
    assert torch.equal(out.z_init, z)
    assert torch.equal(
        out.contact_pair_delta, torch.zeros_like(out.contact_pair_delta)
    )
    assert torch.equal(
        out.interaction_pair_delta, torch.zeros_like(out.interaction_pair_delta)
    )
    assert torch.equal(
        out.surface_single_delta, torch.zeros_like(out.surface_single_delta)
    )
    assert torch.equal(out.tm_single_delta, torch.zeros_like(out.tm_single_delta))
    assert torch.equal(out.membrane_on, torch.tensor([False]))


def test_surface_encoder_returns_single_delta_only():
    encoder = SurfaceTokenEncoder(c_s=16, num_surface_features=6)
    result = encoder(
        s_init=torch.zeros(1, 2, 16),
        local_coords=torch.zeros(1, 2, 3, 3),
        features=torch.zeros(1, 2, 3, 6),
        surface_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        surface_active=torch.tensor([True]),
        token_mask=torch.ones(1, 2),
    )
    assert result.shape == (1, 2, 16)


def test_surface_encoder_zeroes_masked_charge_before_projection():
    class RecordingProjection(nn.Module):
        out_features = 16

        def __init__(self) -> None:
            super().__init__()
            self.recorded = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.recorded = value.detach().clone()
            return torch.zeros(*value.shape[:-1], self.out_features)

    encoder = SurfaceTokenEncoder(c_s=16, num_surface_features=6)
    recorder = RecordingProjection()
    encoder.point_proj = nn.Sequential(recorder)

    encoder(
        s_init=torch.zeros(1, 1, 16),
        local_coords=torch.zeros(1, 1, 1, 3),
        features=torch.tensor([[[[7.0, 2.0, 0.0, 0.0, 1.0, 3.0]]]]),
        surface_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        surface_charge_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        surface_active=torch.tensor([True]),
        token_mask=torch.ones(1, 1),
    )

    assert recorder.recorded is not None
    assert torch.equal(recorder.recorded[..., 64], torch.zeros(1, 1))
    assert torch.equal(
        recorder.recorded[..., 65:],
        torch.tensor([[[2.0, 0.0, 0.0, 1.0, 3.0]]]),
    )


def test_conditioner_threads_surface_charge_mask_to_encoder():
    class RecordingSurface(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.charge_mask = None

        def forward(
            self,
            *,
            s_init,
            local_coords,
            features,
            surface_mask,
            surface_charge_mask,
            surface_active,
            token_mask,
        ):
            del s_init, local_coords, features, surface_mask, surface_active
            self.charge_mask = surface_charge_mask
            return torch.zeros(*token_mask.shape, 16)

    conditioner = OptionalPriorConditioner(c_s=16, c_z=8)
    recorder = RecordingSurface()
    conditioner.surface = recorder
    batch = _conditioner_batch(active=True)
    batch["physppi_surface_charge_mask"] = torch.tensor(
        [[[False, True], [True, True], [True, True]]]
    )

    conditioner(
        s_init=torch.zeros(1, 3, 16),
        z_init=torch.zeros(1, 3, 3, 8),
        batch=batch,
        token_mask=torch.ones(1, 3),
        pair_mask=torch.ones(1, 3, 3),
    )

    assert torch.equal(recorder.charge_mask, batch["physppi_surface_charge_mask"])


def test_tm_topology_is_exact_noop_without_active_tm_tokens():
    encoder = TMTopologyEncoder(c_s=16)
    delta, membrane_on, effective = encoder(
        tm_token_mask=torch.zeros(1, 3, dtype=torch.bool),
        tm_active=torch.tensor([True]),
        token_mask=torch.ones(1, 3),
    )
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(membrane_on, torch.tensor([False]))
    assert torch.count_nonzero(effective) == 0


def test_tm_topology_zeroes_non_tm_tokens_and_accepts_bool_token_mask():
    encoder = TMTopologyEncoder(c_s=16)
    delta, membrane_on, effective = encoder(
        tm_token_mask=torch.tensor([[True, False, True]]),
        tm_active=torch.tensor([True]),
        token_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.equal(membrane_on, torch.tensor([True]))
    assert torch.equal(effective, torch.tensor([[True, False, True]]))
    assert torch.count_nonzero(delta[:, 1]) == 0


def test_tm_topology_accepts_active_tokens_under_bfloat16_autocast():
    encoder = TMTopologyEncoder(c_s=16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        delta, membrane_on, effective = encoder(
            tm_token_mask=torch.tensor([[True, False, True]]),
            tm_active=torch.tensor([True]),
            token_mask=torch.ones(1, 3),
        )

    assert delta.dtype == next(encoder.proj.parameters()).dtype
    assert torch.equal(membrane_on, torch.tensor([True]))
    assert torch.equal(effective, torch.tensor([[True, False, True]]))
