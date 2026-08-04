from pathlib import Path

import pytest
import torch
import yaml

from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    PHYSPPI_SURFACE_NEIGHBORS,
)
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    AFDB10kConfig,
    AFDB10kSettings,
    QBioLiP10kConfig,
    QBioLiP10kSettings,
    TrainingDatasetSpec,
)


def _dataset_paths(tmp_path: Path) -> dict:
    cache = tmp_path / "cache.json"
    cache.write_text("{}")
    return {
        "alignments_directory": None,
        "alignment_db_directory": None,
        "alignment_array_directory": tmp_path,
        "dataset_cache_file": cache,
        "target_structures_directory": tmp_path,
        "target_structure_file_format": "npz",
        "reference_molecule_directory": tmp_path,
        "template_cache_directory": tmp_path,
        "template_structures_directory": None,
        "template_structure_array_directory": tmp_path,
        "template_file_format": "npz",
        "ccd_file": None,
    }


def _spec(dataset_class: str, root: Path, weight: float, tmp_path: Path):
    return TrainingDatasetSpec.model_validate(
        {
            "name": dataset_class.lower(),
            "dataset_class": dataset_class,
            "mode": "train",
            "weight": weight,
            "config": {
                "physppi": {"root": root},
                "dataset_paths": _dataset_paths(tmp_path),
            },
        }
    )


def test_physppi_configs_use_canonical_roots_and_reject_legacy_fields(tmp_path):
    q_root = tmp_path / "Q-BioLiP-10k-new"
    af_root = tmp_path / "AFDB-10k-1"
    q_root.mkdir()
    af_root.mkdir()

    q_spec = _spec("QBioLiP10kDataset", q_root, 0.8, tmp_path)
    af_spec = _spec("AFDB10kDataset", af_root, 0.2, tmp_path)

    assert q_spec.config.physppi.root == q_root
    assert af_spec.config.physppi.root == af_root
    assert isinstance(q_spec.config.physppi, QBioLiP10kSettings)
    assert isinstance(af_spec.config.physppi, AFDB10kSettings)
    assert q_spec.config.physppi.num_surface_neighbors == PHYSPPI_SURFACE_NEIGHBORS
    assert af_spec.config.physppi.num_surface_neighbors == 64

    for config_class in (QBioLiP10kConfig, AFDB10kConfig):
        with pytest.raises(ValueError, match="extra|pkl_directory"):
            config_class.model_validate(
                {
                    "name": "legacy",
                    "physppi": {"root": q_root, "pkl_directory": q_root},
                    "dataset_paths": _dataset_paths(tmp_path),
                }
            )
        with pytest.raises(ValueError, match="canonical.*128"):
            config_class.model_validate(
                {
                    "name": "wrong-surface-size",
                    "physppi": {"root": q_root, "num_surface_neighbors": 64},
                    "dataset_paths": _dataset_paths(tmp_path),
                }
            )


def test_physppi_defaults_are_spatial_only_with_active_pde():
    crop = QBioLiP10kConfig.model_fields["crop"].default
    loss = QBioLiP10kConfig.model_fields["loss"].default
    weights = crop.token_crop.crop_weights

    assert weights.contiguous == 0.0
    assert weights.spatial == 0.5
    assert weights.spatial_interface == 0.5
    assert loss.confidence_loss_names == ["plddt", "pde", "pae"]
    assert loss.loss_weights.pde > 0.0
    assert loss.loss_weights.experimentally_resolved == 0.0
    assert not hasattr(loss.loss_weights, "penergy")
    assert not hasattr(loss.loss_weights, "delta_g")


def draw_dataset_indices(weights: list[float], draws: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.multinomial(
        torch.tensor(weights), draws, replacement=True, generator=generator
    )


def test_physppi_mixture_is_long_run_80_20():
    indices = draw_dataset_indices(weights=[0.8, 0.2], draws=10_000, seed=7)
    counts = torch.bincount(indices, minlength=2)
    assert abs(counts[0].item() / counts.sum().item() - 0.8) < 0.02
    assert abs(counts[1].item() / counts.sum().item() - 0.2) < 0.02


def test_physppi_training_yaml_declares_canonical_mixture_and_loss():
    repo_root = Path(__file__).resolve().parents[6]
    data = yaml.safe_load(
        (repo_root / "examples/training_yamls/physppi_training.yml").read_text()
    )
    entries = data["dataset_configs"]["train"]
    assert [entries["qbiolip10k"]["weight"], entries["afdb10k"]["weight"]] == [0.8, 0.2]
    assert entries["qbiolip10k"]["config"]["physppi"]["root"].endswith(
        "/Q-BioLiP-10k-new"
    )
    assert entries["afdb10k"]["config"]["physppi"]["root"].endswith("/AFDB-10k-1")
    for entry in entries.values():
        crop_weights = entry["config"]["crop"]["token_crop"]["crop_weights"]
        assert crop_weights == {
            "contiguous": 0.0,
            "spatial": 0.5,
            "spatial_interface": 0.5,
        }
        assert entry["config"]["physppi"]["num_surface_neighbors"] == 64
        loss = entry["config"]["loss"]
        assert loss["confidence_loss_names"] == ["plddt", "pde", "pae"]
        assert loss["loss_weights"]["experimentally_resolved"] == 0.0
        assert loss["loss_weights"]["pde"] > 0.0
