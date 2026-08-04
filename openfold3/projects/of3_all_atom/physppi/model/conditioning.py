"""Single-injection optional-prior conditioning for canonical PhysPPI inputs."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from openfold3.projects.of3_all_atom.physppi.model.interface_residue_encoder import (
    InterfaceResidueEncoder,
)
from openfold3.projects.of3_all_atom.physppi.model.pair_constraints import (
    InteractionPairEncoder,
)
from openfold3.projects.of3_all_atom.physppi.model.surface_encoder import (
    SurfaceTokenEncoder,
)
from openfold3.projects.of3_all_atom.physppi.model.tm_topology import TMTopologyEncoder


class OptionalPriorOutput(NamedTuple):
    """Static optional-prior residuals and effective membrane activation."""

    s_init: torch.Tensor
    z_init: torch.Tensor
    membrane_on: torch.Tensor
    tm_token_mask: torch.Tensor
    interface_single_delta: torch.Tensor
    interaction_pair_delta: torch.Tensor
    surface_single_delta: torch.Tensor
    tm_single_delta: torch.Tensor


class OptionalPriorConditioner(nn.Module):
    """Apply canonical optional priors exactly once at input embedding time."""

    def __init__(self, c_s: int, c_z: int, num_interaction_types: int = 5) -> None:
        super().__init__()
        self.interface_residue = InterfaceResidueEncoder(c_s=c_s)
        self.interaction = InteractionPairEncoder(
            c_z=c_z,
            num_interaction_types=num_interaction_types,
        )
        self.surface = SurfaceTokenEncoder(c_s=c_s)
        self.tm = TMTopologyEncoder(c_s=c_s)

    def forward(
        self,
        *,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        batch: dict[str, torch.Tensor],
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> OptionalPriorOutput:
        interface_single_delta = self.interface_residue(
            payload=batch.get("physppi_interface_residue_mask"),
            active=batch.get("physppi_interface_residue_active", False),
            token_mask=token_mask,
        )
        interaction_pair_delta = self.interaction(
            payload=batch.get("physppi_interaction_type_mask"),
            active=batch.get("physppi_interaction_active", False),
            pair_mask=pair_mask,
        )
        surface_single_delta = self.surface(
            s_init=s_init,
            local_coords=batch.get("physppi_surface_local_coords"),
            features=batch.get("physppi_surface_features"),
            surface_mask=batch.get("physppi_surface_mask"),
            surface_charge_mask=batch.get("physppi_surface_charge_mask"),
            surface_active=batch.get("physppi_surface_active", False),
            token_mask=token_mask,
        )
        tm_single_delta, membrane_on, effective_tm_mask = self.tm(
            tm_token_mask=batch.get("physppi_tm_token_mask"),
            tm_active=batch.get("physppi_tm_active", False),
            token_mask=token_mask,
        )

        surface_single_delta = surface_single_delta.to(
            device=s_init.device,
            dtype=s_init.dtype,
        )
        tm_single_delta = tm_single_delta.to(
            device=s_init.device,
            dtype=s_init.dtype,
        )
        interface_single_delta = interface_single_delta.to(
            device=s_init.device,
            dtype=s_init.dtype,
        )
        interaction_pair_delta = interaction_pair_delta.to(
            device=z_init.device,
            dtype=z_init.dtype,
        )
        return OptionalPriorOutput(
            s_init=(
                s_init
                + interface_single_delta
                + surface_single_delta
                + tm_single_delta
            ),
            z_init=z_init + interaction_pair_delta,
            membrane_on=membrane_on,
            tm_token_mask=effective_tm_mask,
            interface_single_delta=interface_single_delta,
            interaction_pair_delta=interaction_pair_delta,
            surface_single_delta=surface_single_delta,
            tm_single_delta=tm_single_delta,
        )


__all__ = ["OptionalPriorConditioner", "OptionalPriorOutput"]
