from pathlib import Path

import pytest
import torch

from openfold3.core.data.framework.single_datasets.abstract_single import (
    DATASET_REGISTRY,
)


def test_legacy_dataset_registration_is_retired():
    assert "PhysPPIOF3Dataset" not in DATASET_REGISTRY


def test_legacy_reader_and_builder_modules_are_retired():
    root = Path(__file__).resolve().parents[5]
    assert not (root / "core/data/framework/single_datasets/physppi.py").exists()
    assert not (root / "projects/of3_all_atom/physppi/data/pkl_reader.py").exists()
    assert not (root / "projects/of3_all_atom/physppi/data/prior_builder.py").exists()


def test_active_runtime_has_no_legacy_symbols():
    root = Path(__file__).resolve().parents[5]
    source_files = [
        path
        for path in root.rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ]
    source = "\n".join(path.read_text() for path in source_files)
    for symbol in (
        "PhysPPI" + "PklReader",
        "build_" + "physppi_priors",
        "interface_pair_" + "source",
        "use_ground_truth_" + "as_input",
    ):
        assert symbol not in source


def test_latest_contract_rejects_stale_feature_shapes():
    from openfold3.projects.of3_all_atom.config.features import (
        NUM_TOKENS,
        feature_dict,
    )

    features = feature_dict.feat

    assert features["physppi_contact_pair_mask"] == [NUM_TOKENS, NUM_TOKENS]
    assert features["physppi_surface_features"] == [NUM_TOKENS, 128, 6]
    assert "physppi_contact_token_mask" not in features
    assert "physppi_surface_interface_mask" not in features
    assert "physppi_contact_mask" not in features
    assert "physppi_paired_msa_selected" not in features


def test_feature_injection_boundary_uses_latest_contract_keys():
    from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
        PHYS_PPI_KEYS,
        validate_optional_prior_features,
    )

    assert "physppi_contact_pair_mask" in PHYS_PPI_KEYS
    assert "physppi_contact_token_mask" not in PHYS_PPI_KEYS
    assert "physppi_surface_interface_mask" not in PHYS_PPI_KEYS
    assert "physppi_contact_mask" not in PHYS_PPI_KEYS
    assert "physppi_paired_msa_selected" not in PHYS_PPI_KEYS
    assert "physppi_tm_provenance" in PHYS_PPI_KEYS

    batch = {
        "token_mask": torch.ones(1, 2, dtype=torch.bool),
        "physppi_contact_available": torch.tensor([True]),
        "physppi_contact_selected": torch.tensor([True]),
        "physppi_contact_valid": torch.tensor([True]),
        "physppi_contact_active": torch.tensor([True]),
        "physppi_contact_pair_mask": torch.tensor([[[False, True], [True, False]]]),
        "physppi_surface_available": torch.tensor([True]),
        "physppi_surface_selected": torch.tensor([True]),
        "physppi_surface_valid": torch.tensor([True]),
        "physppi_surface_active": torch.tensor([True]),
        "physppi_surface_local_coords": torch.zeros(1, 2, 128, 3),
        "physppi_surface_features": torch.zeros(1, 2, 128, 6),
        "physppi_surface_mask": torch.ones(1, 2, 128, dtype=torch.bool),
        "physppi_surface_charge_mask": torch.ones(1, 2, 128, dtype=torch.bool),
        "physppi_tm_available": torch.tensor([True]),
        "physppi_tm_selected": torch.tensor([True]),
        "physppi_tm_valid": torch.tensor([True]),
        "physppi_tm_active": torch.tensor([True]),
        "physppi_tm_token_mask": torch.tensor([[True, False]]),
    }
    validate_optional_prior_features(batch)


def _active_surface_batch() -> dict[str, torch.Tensor]:
    return {
        "token_mask": torch.ones(1, 2, dtype=torch.bool),
        "physppi_surface_available": torch.tensor([True]),
        "physppi_surface_selected": torch.tensor([True]),
        "physppi_surface_valid": torch.tensor([True]),
        "physppi_surface_active": torch.tensor([True]),
        "physppi_surface_local_coords": torch.zeros(1, 2, 128, 3),
        "physppi_surface_features": torch.zeros(1, 2, 128, 6),
        "physppi_surface_mask": torch.ones(1, 2, 128, dtype=torch.bool),
        "physppi_surface_charge_mask": torch.ones(1, 2, 128, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("physppi_surface_local_coords", torch.zeros(1, 2, 3)),
        ("physppi_surface_local_coords", torch.zeros(1, 2, 127, 3)),
        ("physppi_surface_features", torch.zeros(1, 2, 6)),
        ("physppi_surface_features", torch.zeros(1, 2, 127, 6)),
        ("physppi_surface_mask", torch.ones(1, 2, dtype=torch.bool)),
        ("physppi_surface_mask", torch.ones(1, 2, 127, dtype=torch.bool)),
    ],
)
def test_feature_injection_requires_exact_surface_payload_shapes_in_train(
    key: str, value: torch.Tensor
):
    from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
        validate_optional_prior_features,
    )

    batch = _active_surface_batch()
    batch[key] = value

    with pytest.raises(ValueError, match=key):
        validate_optional_prior_features(batch, mode="train")
