from types import SimpleNamespace

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from openfold3.projects.of3_all_atom.config.features import (
    NUM_ATOMS,
    NUM_TOKENS,
    feature_dict,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    PRIOR_OPTIONAL_SLOTS,
    select_prior_level,
)

LEGACY_MODEL_INPUT_KEYS = {
    "physppi_contact_token_mask",
    "physppi_surface_interface_mask",
    "physppi_template_atom_positions",
    "physppi_membrane_plane_normal",
    "physppi_membrane_plane_point",
    "physppi_membrane_plane_valid",
}


def _single_residue_atom_array() -> AtomArray:
    atoms = AtomArray(3)
    atoms.chain_id = np.asarray(["A", "A", "A"])
    atoms.res_id = np.asarray([1, 1, 1])
    atoms.res_name = np.asarray(["ALA", "ALA", "ALA"])
    atoms.atom_name = np.asarray(["N", "CA", "C"])
    atoms.coord = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return atoms


def test_prior_builder_exposes_pair_contacts_and_directed_membrane_labels_only():
    import openfold3.core.data.framework.single_datasets.afdb10k  # noqa: F401
    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module

    adapter = object.__new__(module.PhysPPIPriorComplexDatasetAdapter)
    adapter.physppi = SimpleNamespace(num_surface_neighbors=128)
    sample = SimpleNamespace(
        dmap={"A|B": ((1, 1, 7.0),)},
        interactions=(),
        mesh={},
        opm={
            "res_loc": {"A": {1: 0}},
            "outer_normal": [0.0, 0.0, 2.0],
            "outer_centroid": [0.0, 0.0, 4.0],
        },
    )
    states = select_prior_level(
        {slot: False for slot in PRIOR_OPTIONAL_SLOTS}, requested_level=0
    ).states

    result = adapter.build_prior_features(
        sample,
        _single_residue_atom_array(),
        {("A", 1): 0},
        states,
        np.random.default_rng(0),
        n_tokens=2,
    )

    assert result["physppi_contact_pair_mask"].shape == (2, 2)
    assert result["physppi_contact_pair_mask"].dtype == torch.bool
    assert "physppi_contact_cutoff" not in result
    assert LEGACY_MODEL_INPUT_KEYS.isdisjoint(result)
    assert result["physppi_membrane_gt_valid"]
    assert np.allclose(result["physppi_membrane_gt_normal"], [0.0, 0.0, 1.0])
    assert np.allclose(result["physppi_membrane_gt_point"], [0.0, 0.0, 4.0])


def test_physppi_feature_shapes_declare_only_strict_boundary_fields():
    shapes = feature_dict.feat

    assert shapes["physppi_contact_pair_mask"] == [NUM_TOKENS, NUM_TOKENS]
    assert "physppi_contact_cutoff" not in shapes
    assert shapes["physppi_prior_requested_level"] == []
    assert shapes["physppi_prior_effective_level"] == []
    assert shapes["physppi_occurrence_seed"] == []
    assert shapes["physppi_monomer_atom_positions"] == [NUM_ATOMS, 3]
    assert shapes["physppi_monomer_atom_mask"] == [NUM_ATOMS]
    assert LEGACY_MODEL_INPUT_KEYS.isdisjoint(shapes)


@pytest.mark.parametrize("legacy_key", sorted(LEGACY_MODEL_INPUT_KEYS))
def test_training_validation_rejects_legacy_model_input_keys(legacy_key):
    from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
        validate_optional_prior_features,
    )

    with pytest.raises(ValueError, match=f"legacy PhysPPI model input.*{legacy_key}"):
        validate_optional_prior_features(
            {
                "token_mask": torch.ones((1, 2), dtype=torch.bool),
                legacy_key: torch.zeros((1,), dtype=torch.float32),
            },
            mode="train",
        )
