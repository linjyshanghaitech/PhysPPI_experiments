import copy
import importlib.util
from pathlib import Path

import pytest

COMPONENTS_CIF = (
    "/101045/jinyu/PhysPPI/openfold-3/output/physppi_msa_cache/components.cif"
)


def _load_module():
    path = (
        Path(__file__).resolve().parents[6]
        / "scripts/physppi/prepare_distinct_rank_stage1.py"
    )
    spec = importlib.util.spec_from_file_location("distinct_rank_stage1", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline() -> dict[str, object]:
    dataset_config = {
        "template": {"n_templates": 4, "take_top_k": False},
        "crop": {"token_crop": {"enabled": True, "token_budget": 768}},
        "physppi": {"root": "/data/root", "num_surface_neighbors": 128},
        "loss": {
            "confidence_loss_names": ["plddt", "pde", "pae"],
            "loss_weights": {"experimentally_resolved": 0.0, "pde": 1.0e-4},
        },
    }
    return {
        "experiment_settings": {"output_dir": "/runs/stage1-full-v11"},
        "data_module_args": {"batch_size": 1, "epoch_len": 16384},
        "pl_trainer_args": {
            "devices": 8,
            "precision": "bf16-mixed",
            "max_epochs": 35,
            "accumulate_grad_batches": 4,
        },
        "checkpoint_config": {
            "every_n_epochs": 1,
            "every_n_train_steps": None,
            "save_on_train_epoch_end": True,
        },
        "model_update": {
            "custom": {
                "settings": {"training_scope": "physppi_lora_assessment"},
                "architecture": {
                    "physppi": {
                        "enabled": True,
                        "optional_prior_conditioner": {"enabled": True},
                        "membrane_geometry": {"enabled": True},
                    }
                },
            }
        },
        "dataset_configs": {
            "train": {
                "qbiolip10k": {"weight": 0.8, "config": copy.deepcopy(dataset_config)},
                "afdb10k": {"weight": 0.2, "config": copy.deepcopy(dataset_config)},
            }
        },
        "dataset_paths": {
            "qbiolip10k": {"ccd_file": COMPONENTS_CIF},
            "afdb10k": {"ccd_file": COMPONENTS_CIF},
        },
    }


def test_builder_preserves_contract_and_exact_optimizer_arithmetic() -> None:
    module = _load_module()
    baseline = _baseline()
    manifest = "/runs/manifests/formal-manifest-v7.json"

    smoke, formal = module.build_stage1_configs(baseline, manifest, "v12")

    assert formal["data_module_args"] == {
        **formal["data_module_args"],
        "batch_size": 1,
        "epoch_len": 4096,
        "distinct_samples_per_rank_group": True,
    }
    assert formal["pl_trainer_args"]["devices"] == 8
    assert formal["pl_trainer_args"]["precision"] == "bf16-mixed"
    assert formal["pl_trainer_args"]["max_epochs"] == 50
    assert formal["pl_trainer_args"]["accumulate_grad_batches"] == 2
    assert formal["pl_trainer_args"]["max_steps"] == 12_800
    assert (
        formal["model_update"]["custom"]["architecture"]["shared"]["diffusion"][
            "no_samples"
        ]
        == 48
    )
    assert formal["checkpoint_config"]["every_n_epochs"] == 1
    assert formal["checkpoint_config"]["every_n_train_steps"] is None
    assert formal["checkpoint_config"]["save_on_train_epoch_end"] is True
    assert smoke["data_module_args"]["epoch_len"] == 512
    assert smoke["pl_trainer_args"]["max_epochs"] == 1
    assert smoke["pl_trainer_args"]["max_steps"] == 32
    assert smoke["pl_trainer_args"]["accumulate_grad_batches"] == 2
    assert (
        smoke["experiment_settings"]["output_dir"]
        != formal["experiment_settings"]["output_dir"]
    )
    assert formal["dataset_paths"] == baseline["dataset_paths"]
    assert (
        formal["dataset_configs"]["train"]["qbiolip10k"]["config"]["physppi"][
            "manifest"
        ]
        == manifest
    )
    assert (
        formal["dataset_configs"]["train"]["afdb10k"]["config"]["physppi"]["manifest"]
        == manifest
    )


def test_validator_rejects_wrong_update_count() -> None:
    module = _load_module()
    _smoke, formal = module.build_stage1_configs(
        _baseline(), "/runs/manifests/formal-manifest-v7.json", "v12"
    )
    formal["data_module_args"]["epoch_len"] = 2048

    with pytest.raises(ValueError, match="expected 256 optimizer updates"):
        module.validate_stage1_config(formal, smoke=False)


def test_legacy_builder_delegates_to_semantic_50epoch_validator() -> None:
    module = _load_module()
    smoke, formal = module.build_stage1_configs(
        _baseline(), "/runs/manifests/formal-manifest-v7.json", "v12"
    )

    assert smoke["pl_trainer_args"]["max_steps"] == 32
    assert formal["pl_trainer_args"]["max_epochs"] == 50
    assert formal["pl_trainer_args"]["max_steps"] == 12_800
    formal["pl_trainer_args"]["max_epochs"] = 35
    with pytest.raises(ValueError, match="max_epochs"):
        module.validate_stage1_config(formal, smoke=False)
