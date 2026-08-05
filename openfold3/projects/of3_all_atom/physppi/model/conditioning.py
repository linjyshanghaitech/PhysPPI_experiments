"""Optional-prior conditioning with legacy and v5 (cross-attention) modes."""

from __future__ import annotations

import importlib.util
import sys
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


def _load_encoder_class(
    path: str | None,
    fallback: type,
    *,
    attr: str | None = None,
) -> type:
    """Load an experiment-local encoder module (head_module precedent)."""

    if not path:
        return fallback
    if fallback is None and not attr:
        raise ValueError("fallback class or attr is required to name the module")
    name = f"_physppi_{attr or fallback.__name__}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load encoder module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, attr or fallback.__name__)


def _sample_prior_drops(
    *,
    batch_size: int,
    device: torch.device,
    prob: float,
    training: bool,
) -> dict[str, torch.Tensor]:
    names = ("interface_residue", "interaction", "surface", "tm")
    if training and prob > 0.0:
        rand = torch.rand(batch_size, 4, device=device)
        return {name: rand[:, i] < prob for i, name in enumerate(names)}
    return {
        name: torch.zeros(batch_size, device=device, dtype=torch.bool)
        for name in names
    }


def _inter_pair_mask(
    batch: dict[str, torch.Tensor],
    tokens: torch.Tensor,
) -> torch.Tensor:
    asym = torch.as_tensor(batch["asym_id"], device=tokens.device).long()
    if asym.ndim == 1:
        asym = asym.unsqueeze(0)
    n = tokens.shape[1]
    diag = torch.eye(n, device=tokens.device, dtype=torch.bool).unsqueeze(0)
    different = asym[:, :, None] != asym[:, None, :]
    return different & tokens[:, :, None] & tokens[:, None, :] & ~diag


def _build_token_atom_bag(
    batch: dict[str, torch.Tensor],
    *,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    element = torch.as_tensor(batch["ref_element"], device=device).float()
    atom_to_token = torch.as_tensor(
        batch["atom_to_token_index"], device=device
    ).long()
    atom_mask = torch.as_tensor(batch["atom_mask"], device=device).bool()
    if element.ndim == 2:
        element = element.unsqueeze(0)
        atom_to_token = atom_to_token.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
    b, n_atoms, c = element.shape
    valid = atom_mask & (atom_to_token >= 0) & (atom_to_token < num_tokens)
    element = element * valid.unsqueeze(-1)
    idx = atom_to_token.clamp(0, num_tokens - 1)
    bag = torch.zeros(b, num_tokens, c, device=device, dtype=element.dtype)
    bag.scatter_reduce_(
        1,
        idx.unsqueeze(-1).expand(-1, -1, c),
        element,
        reduce="amax",
        include_self=True,
    )
    return bag


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
    surface_pair_delta: torch.Tensor


class OptionalPriorConditioner(nn.Module):
    """Apply canonical optional priors once at input embedding time.

    Legacy mode (no ``encoder_modules``) keeps the original four adapter
    classes and parameter names, so old checkpoints load unchanged.  v5 mode
    loads experiment-local encoder modules from config file paths and enables
    the cross-attention prior design plus model-side prior dropout.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_interaction_types: int = 5,
        *,
        no_heads_single: int = 8,
        no_heads_pair: int = 8,
        prior_drop_prob: float = 0.25,
        pair_attn_chunk: int = 1024,
        pair_prior_inter_only: bool = True,
        use_restype: bool = True,
        encoder_modules: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        modules = {
            name: value
            for name, value in (encoder_modules or {}).items()
            if value
        }
        self.v5 = bool(modules)
        if self.v5 and not all(modules.values()):
            raise ValueError(
                "v5 prior conditioning requires all four encoder_module paths"
            )
        self.prior_drop_prob = float(prior_drop_prob)
        self.pair_attn_chunk = int(pair_attn_chunk)
        self.pair_prior_inter_only = bool(pair_prior_inter_only)
        self.use_restype = bool(use_restype)
        interface_cls = _load_encoder_class(
            modules.get("interface_residue"), InterfaceResidueEncoder
        )
        interaction_cls = _load_encoder_class(
            modules.get("interaction"), InteractionPairEncoder
        )
        surface_cls = _load_encoder_class(
            modules.get("surface"), SurfaceTokenEncoder
        )
        tm_cls = _load_encoder_class(modules.get("tm"), TMTopologyEncoder)
        if self.v5:
            try:
                from openfold3.projects.of3_all_atom.physppi.model.surface_encoder import (
                    SurfacePairEncoder as _FallbackPairEncoder,
                )
            except ImportError:
                _FallbackPairEncoder = None
            surface_pair_cls = _load_encoder_class(
                modules.get("surface"),
                _FallbackPairEncoder,
                attr="SurfacePairEncoder",
            )
            self.interface_residue = interface_cls(
                c_s=c_s, no_heads=no_heads_single
            )
            self.interaction = interaction_cls(
                c_z=c_z,
                c_q=c_s,
                num_interaction_types=num_interaction_types,
                no_heads=no_heads_pair,
                query_chunk=pair_attn_chunk,
            )
            self.surface = surface_cls(c_s=c_s, no_heads=no_heads_single)
            self.surface_pair = surface_pair_cls(
                c_z=c_z,
                c_q=c_s,
                no_heads=no_heads_pair,
                point_proj=self.surface.point_proj,
                query_chunk=pair_attn_chunk,
            )
            self.tm = tm_cls(c_s=c_s, no_heads=no_heads_single)
        else:
            self.interface_residue = interface_cls(c_s=c_s)
            self.interaction = interaction_cls(
                c_z=c_z, num_interaction_types=num_interaction_types
            )
            self.surface = surface_cls(c_s=c_s)
            self.tm = tm_cls(c_s=c_s)
            self.surface_pair = None

    def forward(
        self,
        *,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        batch: dict[str, torch.Tensor],
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> OptionalPriorOutput:
        if not self.v5:
            return self._forward_legacy(
                s_init=s_init,
                z_init=z_init,
                batch=batch,
                token_mask=token_mask,
                pair_mask=pair_mask,
            )
        return self._forward_v5(
            s_init=s_init,
            z_init=z_init,
            batch=batch,
            token_mask=token_mask,
            pair_mask=pair_mask,
        )

    def _forward_legacy(
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
            surface_pair_delta=torch.zeros_like(z_init),
        )

    def _forward_v5(
        self,
        *,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        batch: dict[str, torch.Tensor],
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> OptionalPriorOutput:
        tokens = torch.as_tensor(token_mask, device=s_init.device).bool()
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        b = tokens.shape[0]
        if not self.use_restype:
            raise ValueError("v5 prior conditioning requires use_restype=True")
        restype = batch.get("restype")
        if restype is None:
            raise ValueError("v5 prior conditioning requires batch['restype']")
        drops = _sample_prior_drops(
            batch_size=b,
            device=s_init.device,
            prob=self.prior_drop_prob,
            training=self.training,
        )

        def flag(name: str) -> torch.Tensor:
            return torch.as_tensor(
                batch.get(f"physppi_{name}_active", False),
                device=s_init.device,
                dtype=torch.bool,
            ).reshape(-1)

        interface_active = flag("interface_residue") & ~drops["interface_residue"]
        interaction_active = flag("interaction") & ~drops["interaction"]
        surface_active = flag("surface") & ~drops["surface"]
        inter_mask = _inter_pair_mask(batch, tokens)
        if not self.pair_prior_inter_only:
            inter_mask = tokens[:, :, None] & tokens[:, None, :]

        # TM side 回退:数据契约未含 side 时,由现有 tm_token_mask 推导
        # (side=0 & valid=tm_mask),等价旧二值行为;重新生成数据后自动获得 ±1。
        tm_side = batch.get("physppi_tm_side")
        tm_side_valid = batch.get("physppi_tm_side_valid")
        if tm_side is None or tm_side_valid is None:
            tm_mask = batch.get("physppi_tm_token_mask")
            if tm_mask is not None:
                tm_mask_b = torch.as_tensor(tm_mask, device=tokens.device).bool()
                tm_side = torch.zeros_like(tm_mask_b, dtype=torch.long)
                tm_side_valid = tm_mask_b

        interface_single_delta = self.interface_residue(
            s_init=s_init,
            payload=batch.get("physppi_interface_residue_mask"),
            active=interface_active,
            token_mask=tokens,
            restype=restype,
        )
        tm_single_delta, membrane_on, effective_tm_mask = self.tm(
            s_init=s_init,
            tm_side=tm_side,
            tm_side_valid=tm_side_valid,
            tm_active=flag("tm"),
            tm_dropped=drops["tm"],
            token_mask=tokens,
            restype=restype,
        )
        surface_single_delta = self.surface(
            s_init=s_init,
            local_coords=batch.get("physppi_surface_local_coords"),
            features=batch.get("physppi_surface_features"),
            surface_mask=batch.get("physppi_surface_mask"),
            surface_charge_mask=batch.get("physppi_surface_charge_mask"),
            surface_active=surface_active,
            token_mask=tokens,
        )
        atom_bag = _build_token_atom_bag(
            batch, num_tokens=tokens.shape[1], device=s_init.device
        )
        pair_mask_bool = torch.as_tensor(pair_mask, device=z_init.device).bool()
        interaction_pair_delta = self.interaction(
            z_init=z_init,
            payload=batch.get("physppi_interaction_type_mask"),
            active=interaction_active,
            pair_mask=pair_mask_bool,
            inter_pair_mask=inter_mask,
            restype=restype,
            atom_bag=atom_bag,
        )
        surface_pair_delta = self.surface_pair(
            z_init=z_init,
            local_coords=batch.get("physppi_surface_local_coords"),
            features=batch.get("physppi_surface_features"),
            surface_mask=batch.get("physppi_surface_mask"),
            surface_charge_mask=batch.get("physppi_surface_charge_mask"),
            surface_active=surface_active,
            pair_mask=pair_mask_bool,
            inter_pair_mask=inter_mask,
            token_mask=tokens,
            monomer_atom_positions=batch.get("physppi_monomer_atom_positions"),
            monomer_atom_mask=batch.get("physppi_monomer_atom_mask"),
            backbone_atom_indices=batch.get("physppi_backbone_atom_indices"),
            backbone_frame_mask=batch.get("physppi_backbone_frame_mask"),
        )
        deltas = (
            interface_single_delta.to(device=s_init.device, dtype=s_init.dtype),
            surface_single_delta.to(device=s_init.device, dtype=s_init.dtype),
            tm_single_delta.to(device=s_init.device, dtype=s_init.dtype),
        )
        interaction_pair_delta = interaction_pair_delta.to(
            device=z_init.device, dtype=z_init.dtype
        )
        surface_pair_delta = surface_pair_delta.to(
            device=z_init.device, dtype=z_init.dtype
        )
        return OptionalPriorOutput(
            s_init=s_init + deltas[0] + deltas[1] + deltas[2],
            z_init=z_init + interaction_pair_delta + surface_pair_delta,
            membrane_on=membrane_on,
            tm_token_mask=effective_tm_mask,
            interface_single_delta=deltas[0],
            interaction_pair_delta=interaction_pair_delta,
            surface_single_delta=deltas[1],
            tm_single_delta=deltas[2],
            surface_pair_delta=surface_pair_delta,
        )


__all__ = ["OptionalPriorConditioner", "OptionalPriorOutput"]
