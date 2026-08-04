import pytest
import torch

from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
    PHYS_PPI_KEYS,
    inject_physppi_features,
    validate_optional_prior_features,
)


def _active_surface_batch() -> dict[str, torch.Tensor]:
    return {
        "token_mask": torch.ones(1, 2, dtype=torch.bool),
        "physppi_surface_available": torch.tensor([True]),
        "physppi_surface_selected": torch.tensor([True]),
        "physppi_surface_valid": torch.tensor([True]),
        "physppi_surface_active": torch.tensor([True]),
        "physppi_surface_local_coords": torch.zeros(1, 2, 64, 3),
        "physppi_surface_features": torch.zeros(1, 2, 64, 6),
        "physppi_surface_mask": torch.ones(1, 2, 64, dtype=torch.bool),
        "physppi_surface_charge_mask": torch.ones(1, 2, 64, dtype=torch.bool),
    }


def test_surface_charge_mask_is_required_and_validated_for_active_surface_slot():
    batch = _active_surface_batch()
    validate_optional_prior_features(batch)

    missing = dict(batch)
    missing.pop("physppi_surface_charge_mask")

    try:
        validate_optional_prior_features(missing)
    except ValueError as exc:
        assert "physppi_surface_charge_mask" in str(exc)
    else:
        raise AssertionError("missing charge mask should fail validation")


def test_injection_copies_surface_charge_mask():
    base = {"token_mask": torch.ones(1, 2, dtype=torch.bool)}
    out = inject_physppi_features(base, _active_surface_batch())

    assert "physppi_surface_charge_mask" in out
    assert torch.equal(
        out["physppi_surface_charge_mask"],
        torch.ones(1, 2, 64, dtype=torch.bool),
    )


def test_injection_copies_tm_source_provenance():
    base = {"token_mask": torch.ones(1, 2, dtype=torch.bool)}
    out = inject_physppi_features(base, {"physppi_tm_provenance": torch.tensor([True])})

    assert torch.equal(out["physppi_tm_provenance"], torch.tensor([True]))


def test_active_interface_residue_requires_per_token_boolean_mask():
    batch = {
        "token_mask": torch.ones(1, 2, dtype=torch.bool),
        "physppi_interface_residue_available": torch.tensor([True]),
        "physppi_interface_residue_selected": torch.tensor([True]),
        "physppi_interface_residue_valid": torch.tensor([True]),
        "physppi_interface_residue_active": torch.tensor([True]),
        "physppi_interface_residue_mask": torch.tensor([[True, False]]),
    }

    validate_optional_prior_features(batch)

    non_boolean = dict(batch)
    non_boolean["physppi_interface_residue_mask"] = batch[
        "physppi_interface_residue_mask"
    ].float()
    with pytest.raises(ValueError, match="boolean"):
        validate_optional_prior_features(non_boolean)

    wrong_shape = dict(batch)
    wrong_shape["physppi_interface_residue_mask"] = torch.tensor(
        [[[True, False], [False, True]]]
    )
    with pytest.raises(ValueError, match="trailing shape"):
        validate_optional_prior_features(wrong_shape)


def test_injection_rejects_legacy_model_input_key():
    base = {"token_mask": torch.ones(1, 2, dtype=torch.bool)}

    with pytest.raises(ValueError, match="legacy PhysPPI model input"):
        inject_physppi_features(
            base,
            {"physppi_template_atom_positions": torch.zeros(1, 4, 3)},
        )


def test_interface_residue_cutoff_audit_metadata_is_forbidden_at_model_boundary():
    base = {"token_mask": torch.ones(1, 2, dtype=torch.bool)}
    cutoff = {"physppi_interface_residue_cutoff": torch.tensor([12.0])}

    assert "physppi_interface_residue_cutoff" not in PHYS_PPI_KEYS
    with pytest.raises(
        ValueError, match="audit-only.*physppi_interface_residue_cutoff"
    ):
        validate_optional_prior_features({**base, **cutoff})
    with pytest.raises(
        ValueError, match="audit-only.*physppi_interface_residue_cutoff"
    ):
        inject_physppi_features(base, cutoff)
