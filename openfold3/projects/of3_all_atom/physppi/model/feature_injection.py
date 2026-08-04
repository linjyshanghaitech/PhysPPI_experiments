"""Validation and transport for the canonical Q-BioLiP feature contract."""

from __future__ import annotations

import warnings

import torch

ESM_FORBIDDEN_KEYS = frozenset(
    {"esm_embedding", "esm2_embedding", "esm_prior_feat", "physppi_sequence_embedding"}
)

LEGACY_PHYS_PPI_MODEL_INPUT_KEYS = frozenset(
    {
        "physppi_contact_token_mask",
        "physppi_surface_interface_mask",
        "physppi_template_atom_positions",
        "physppi_membrane_plane_normal",
        "physppi_membrane_plane_point",
        "physppi_membrane_plane_valid",
    }
)

AUDIT_ONLY_PHYS_PPI_KEYS = frozenset({"physppi_interface_residue_cutoff"})

PHYS_PPI_KEYS = frozenset(
    {
        "physppi_interface_residue_available",
        "physppi_interface_residue_selected",
        "physppi_interface_residue_valid",
        "physppi_interface_residue_provenance",
        "physppi_interface_residue_active",
        "physppi_interface_residue_mask",
        "physppi_paired_msa_chain_mask",
        "physppi_interaction_available",
        "physppi_interaction_selected",
        "physppi_interaction_valid",
        "physppi_interaction_provenance",
        "physppi_interaction_active",
        "physppi_interaction_type_mask",
        "physppi_surface_available",
        "physppi_surface_selected",
        "physppi_surface_valid",
        "physppi_surface_provenance",
        "physppi_surface_active",
        "physppi_surface_local_coords",
        "physppi_surface_features",
        "physppi_surface_mask",
        "physppi_surface_charge_mask",
        "physppi_tm_available",
        "physppi_tm_selected",
        "physppi_tm_valid",
        "physppi_tm_provenance",
        "physppi_tm_active",
        "physppi_tm_token_mask",
        "physppi_token_center_atom_index",
        "physppi_backbone_atom_indices",
        "physppi_backbone_frame_mask",
        "physppi_monomer_atom_positions",
        "physppi_monomer_atom_mask",
        "physppi_prior_requested_level",
        "physppi_prior_effective_level",
        "physppi_occurrence_seed",
        "physppi_membrane_gt_normal",
        "physppi_membrane_gt_point",
        "physppi_membrane_gt_valid",
        "physppi_interaction_aux_valid",
        "physppi_interaction_distance",
    }
)

_SLOTS = {
    "interface_residue": (
        "physppi_interface_residue_available",
        "physppi_interface_residue_selected",
        "physppi_interface_residue_valid",
        "physppi_interface_residue_active",
    ),
    "interaction": (
        "physppi_interaction_available",
        "physppi_interaction_selected",
        "physppi_interaction_valid",
        "physppi_interaction_active",
    ),
    "surface": (
        "physppi_surface_available",
        "physppi_surface_selected",
        "physppi_surface_valid",
        "physppi_surface_active",
    ),
    "tm": (
        "physppi_tm_available",
        "physppi_tm_selected",
        "physppi_tm_valid",
        "physppi_tm_active",
    ),
}


def assert_no_esm_features(batch: dict) -> None:
    forbidden = sorted(set(batch).intersection(ESM_FORBIDDEN_KEYS))
    if forbidden:
        raise ValueError(
            "ESM features are outside the PhysPPI contract: " + ", ".join(forbidden)
        )


def _issue(message: str, mode: str) -> None:
    if mode == "train":
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def _token_mask(batch: dict, mode: str) -> torch.Tensor | None:
    value = batch.get("token_mask")
    if not torch.is_tensor(value):
        _issue("token_mask is required for PhysPPI validation", mode)
        return None
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        _issue("token_mask must have shape [B, N]", mode)
        return None
    return value.bool()


def _flags(
    batch: dict, names: tuple[str, ...], batch_size: int, device
) -> torch.Tensor:
    values = []
    for name in names:
        values.append(
            torch.as_tensor(batch.get(name, False), device=device, dtype=torch.bool)
        )
    normalized = []
    for value, name in zip(values, names, strict=True):
        if value.ndim == 0:
            value = value.reshape(1).expand(batch_size)
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must be scalar or shape [{batch_size}]")
        normalized.append(value)
    return torch.stack(normalized, dim=0)


def _payload(batch: dict, key: str, batch_size: int, mode: str) -> torch.Tensor | None:
    value = batch.get(key)
    if not torch.is_tensor(value):
        _issue(f"{key} must be a torch.Tensor", mode)
        return None
    if value.ndim == 0 or value.shape[0] not in (batch_size,):
        _issue(f"{key} must have leading batch dimension {batch_size}", mode)
        return None
    if value.dtype.is_floating_point and not torch.isfinite(value).all():
        _issue(f"{key} must contain finite values", mode)
    return value


def _shape_check(
    value: torch.Tensor, key: str, expected: tuple[int, ...], mode: str
) -> None:
    if tuple(value.shape[1:]) != expected:
        _issue(
            f"{key} must have trailing shape {expected}, got {tuple(value.shape)}", mode
        )


def validate_optional_prior_features(
    batch: dict,
    *,
    token_mask: torch.Tensor | None = None,
    mode: str = "train",
) -> None:
    """Validate independent availability/selection/validity slot state."""

    assert_no_esm_features(batch)
    audit_only_keys = sorted(set(batch).intersection(AUDIT_ONLY_PHYS_PPI_KEYS))
    if audit_only_keys:
        _issue(
            "audit-only PhysPPI metadata is forbidden at the model boundary: "
            + ", ".join(audit_only_keys),
            mode,
        )
    legacy_keys = sorted(set(batch).intersection(LEGACY_PHYS_PPI_MODEL_INPUT_KEYS))
    if legacy_keys:
        _issue(
            "legacy PhysPPI model input keys are forbidden: " + ", ".join(legacy_keys),
            mode,
        )
    token_mask = token_mask if token_mask is not None else _token_mask(batch, mode)
    if token_mask is None:
        return
    if token_mask.ndim == 1:
        token_mask = token_mask.unsqueeze(0)
    batch_size, n_tokens = token_mask.shape
    device = token_mask.device

    paired_chain_mask = batch.get("physppi_paired_msa_chain_mask")
    if paired_chain_mask is not None:
        value = _payload(batch, "physppi_paired_msa_chain_mask", batch_size, mode)
        if value is not None and value.ndim != 2:
            _issue(
                "physppi_paired_msa_chain_mask must have shape [B, N_chains]",
                mode,
            )

    for slot, names in _SLOTS.items():
        states = _flags(batch, names[:3], batch_size, device)
        expected_active = states.all(dim=0)
        active = torch.as_tensor(
            batch.get(names[3], expected_active), device=device, dtype=torch.bool
        )
        if active.ndim == 0:
            active = active.reshape(1).expand(batch_size)
        if active.shape != (batch_size,):
            _issue(f"{names[3]} must be scalar or shape [{batch_size}]", mode)
            continue
        if bool(active.any()):
            payload_keys = {
                "interface_residue": ("physppi_interface_residue_mask",),
                "interaction": ("physppi_interaction_type_mask",),
                "surface": (
                    "physppi_surface_local_coords",
                    "physppi_surface_features",
                    "physppi_surface_mask",
                    "physppi_surface_charge_mask",
                ),
                "tm": ("physppi_tm_token_mask",),
            }[slot]
            missing = [key for key in payload_keys if key not in batch]
            if missing:
                _issue(
                    f"active optional {slot} slot is missing: {', '.join(missing)}",
                    mode,
                )
                continue
        if not torch.equal(active, expected_active):
            _issue(f"{names[3]} must equal available & selected & valid", mode)
        if not bool(active.any()):
            continue

        if slot == "interface_residue":
            value = _payload(
                batch, "physppi_interface_residue_mask", batch_size, mode
            )
            if value is not None:
                if value.dtype != torch.bool:
                    _issue("physppi_interface_residue_mask must be boolean", mode)
                _shape_check(
                    value,
                    "physppi_interface_residue_mask",
                    (n_tokens,),
                    mode,
                )
        elif slot == "interaction":
            value = _payload(batch, "physppi_interaction_type_mask", batch_size, mode)
            if value is not None:
                _shape_check(
                    value,
                    "physppi_interaction_type_mask",
                    (n_tokens, n_tokens, 5),
                    mode,
                )
                if not torch.equal(value, value.transpose(-3, -2)):
                    _issue("physppi_interaction_type_mask must be symmetric", mode)
        elif slot == "surface":
            coords = _payload(batch, "physppi_surface_local_coords", batch_size, mode)
            feats = _payload(batch, "physppi_surface_features", batch_size, mode)
            mask = _payload(batch, "physppi_surface_mask", batch_size, mode)
            charge_mask = _payload(
                batch, "physppi_surface_charge_mask", batch_size, mode
            )
            if coords is not None:
                _shape_check(
                    coords,
                    "physppi_surface_local_coords",
                    (n_tokens, 64, 3),
                    mode,
                )
            if feats is not None:
                _shape_check(
                    feats,
                    "physppi_surface_features",
                    (n_tokens, 64, 6),
                    mode,
                )
            if mask is not None:
                _shape_check(mask, "physppi_surface_mask", (n_tokens, 64), mode)
            if charge_mask is not None:
                _shape_check(
                    charge_mask, "physppi_surface_charge_mask", (n_tokens, 64), mode
                )
        elif slot == "tm":
            value = _payload(batch, "physppi_tm_token_mask", batch_size, mode)
            if value is not None:
                _shape_check(value, "physppi_tm_token_mask", (n_tokens,), mode)


def validate_required_physppi_features(
    batch: dict,
    enabled: bool = True,
    mode: str = "train",
    config=None,
    require_deployable: bool = False,
) -> None:
    """Compatibility entry point with the canonical optional-slot semantics."""

    del config, require_deployable
    if enabled:
        validate_optional_prior_features(batch, mode=mode)


def inject_physppi_features(
    batch: dict[str, torch.Tensor],
    physppi_features: dict[str, torch.Tensor],
    *,
    allow_ground_truth_interface_input: bool = False,
) -> dict[str, torch.Tensor]:
    """Copy only canonical Q-BioLiP tensors and validate their slot states."""

    del allow_ground_truth_interface_input
    audit_only_keys = sorted(
        set(physppi_features).intersection(AUDIT_ONLY_PHYS_PPI_KEYS)
    )
    if audit_only_keys:
        raise ValueError(
            "audit-only PhysPPI metadata is forbidden at the model boundary: "
            + ", ".join(audit_only_keys)
        )
    legacy_keys = sorted(
        set(physppi_features).intersection(LEGACY_PHYS_PPI_MODEL_INPUT_KEYS)
    )
    if legacy_keys:
        raise ValueError(
            "legacy PhysPPI model input keys are forbidden: " + ", ".join(legacy_keys)
        )
    out = dict(batch)
    for key, value in physppi_features.items():
        if key in PHYS_PPI_KEYS:
            if not torch.is_tensor(value):
                raise TypeError(f"{key} must be a torch.Tensor")
            out[key] = value.to(device=batch["token_mask"].device)
    validate_optional_prior_features(out, mode="train")
    return out
