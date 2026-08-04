import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openfold3.core.data.framework.data_module import DataModuleConfig
from openfold3.entry_points.validator import DataModuleArgs
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate, OF3ProjectEntry

LORA_GROUPS = [
    {
        "patterns": [r"^template_embedder\.linear_t$"],
        "rank": 2,
        "alpha": 4,
        "dropout": 0.0,
        "max_trainable_lora_params": 131_968,
    },
    {
        "patterns": [
            r"^pairformer_stack\.blocks\.(4[0-7])\.pair_stack\.tri_att_(start|end)\.mha\.linear_(q|k|v|o)$",
            r"^pairformer_stack\.blocks\.(4[0-7])\.pair_stack\.pair_transition\.(swiglu\.linear_(a|b)|linear_out)$",
            r"^pairformer_stack\.blocks\.(4[0-7])\.attn_pair_bias\.linear_z$",
        ],
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
        "max_trainable_lora_params": 131_968,
    },
]


def _canonical_checkpoint() -> str:
    return os.environ["OPENFOLD3_CHECKPOINT"]


def _components_cif() -> str:
    return os.environ["PHYSPPI_COMPONENTS_CIF"]


@pytest.fixture(autouse=True)
def _portable_production_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENFOLD3_CHECKPOINT", str(tmp_path / "canonical.pt"))
    monkeypatch.setenv("PHYSPPI_COMPONENTS_CIF", str(tmp_path / "components.cif"))


def _load_module():
    path = (
        Path(__file__).resolve().parents[6]
        / "scripts/physppi/prepare_semantic_50epoch.py"
    )
    spec = importlib.util.spec_from_file_location("semantic_50epoch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_config(root: str, key: str) -> dict[str, object]:
    return {
        "debug_mode": False,
        "template": {"n_templates": 1, "take_top_k": True},
        "crop": {
            "token_crop": {
                "enabled": True,
                "token_budget": 256,
                "crop_weights": {
                    "contiguous": 1.0,
                    "spatial": 0.0,
                    "spatial_interface": 0.0,
                },
            },
            "chain_crop": {"enabled": True},
        },
        "physppi": {
            "root": root,
            "manifest_dataset_key": key,
            "manifest_partition": "train_ids",
            "num_surface_neighbors": 32,
        },
        "loss": {
            "confidence_loss_names": ["plddt"],
            "loss_weights": {"experimentally_resolved": 1.0, "pde": 0.0},
        },
    }


def _baseline() -> dict[str, object]:
    return {
        "experiment_settings": {
            "mode": "train",
            "output_dir": "/runs/stale",
            "seed": 7,
            "restart_checkpoint_path": "/tmp/stale.ckpt",
            "preemption_safe_resume": True,
            "ckpt_load_settings": {
                "manual_checkpoint_loading": False,
                "init_from_ema_weights": False,
                "restore_lr_scheduler": True,
                "restore_time_step": True,
                "strict_loading": True,
            },
        },
        "data_module_args": {
            "batch_size": 4,
            "data_seed": 7,
            "num_workers": 2,
            "prefetch_factor": 8,
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "pin_memory": True,
            "epoch_len": 16_384,
        },
        "pl_trainer_args": {
            "devices": 4,
            "num_nodes": 2,
            "precision": "32-true",
            "max_epochs": 35,
            "max_steps": 256,
            "accumulate_grad_batches": 4,
            "log_every_n_steps": 1,
        },
        "checkpoint_config": {
            "every_n_epochs": 2,
            "every_n_train_steps": 10,
            "save_on_train_epoch_end": False,
            "save_top_k": 1,
            "save_last": False,
            "filename": "stale",
            "auto_insert_metric_name": True,
        },
        "model_update": {
            "presets": ["train"],
            "custom": {
                "settings": {
                    "training_scope": "baseline",
                    "optimizer": {
                        "learning_rate": 1.0e-3,
                        "beta1": 0.8,
                        "beta2": 0.9,
                        "eps": 1.0e-6,
                        "weight_decay": 1.0e-2,
                    },
                    "lr_scheduler": {
                        "base_lr": 1.0e-5,
                        "warmup_no_steps": 10,
                        "start_decay_after_n_steps": 20,
                        "decay_every_n_steps": 20,
                        "decay_factor": 0.5,
                    },
                    "ema": {"decay": 0.9, "submodules_to_update": ["x"]},
                    "gradient_clipping": {
                        "per_sample_clipping": False,
                        "clip_val": 1.0,
                    },
                },
                "architecture": {
                    "shared": {"diffusion": {"no_samples": 1}},
                    "physppi": {
                        "enabled": False,
                        "training_scope": "baseline",
                        "optional_prior_conditioner": {"enabled": False},
                        "membrane_geometry": {"enabled": False},
                    },
                    "lora": {
                        "enabled": False,
                        "target_preset": "other",
                        "target_patterns": ["anything"],
                        "groups": [],
                        "exclude_patterns": ["anything"],
                        "rank": 1,
                        "alpha": 1,
                        "dropout": 0.0,
                        "train_bias": True,
                        "init": "other",
                        "max_trainable_lora_params": 1,
                    },
                    "heads": {"pdockq": {"candidate_chunk_size": 48}},
                    "loss_module": {
                        "confidence_loss_names": ["plddt"],
                        "membrane": {"normal_weight": 0.0, "offset_weight": 0.0},
                        "interaction_auxiliary": {"distance_weight": 0.0},
                    },
                },
            },
        },
        "dataset_configs": {
            "train": {
                "qbiolip10k": {
                    "dataset_class": "QBioLiP10kDataset",
                    "weight": 0.5,
                    "config": _dataset_config("/qbiolip", "qbiolip"),
                },
                "afdb10k": {
                    "dataset_class": "AFDB10kDataset",
                    "weight": 0.5,
                    "config": _dataset_config("/afdb", "afdb"),
                },
            }
        },
        "dataset_paths": {
            "qbiolip10k": {"ccd_file": _components_cif()},
            "afdb10k": {"ccd_file": _components_cif()},
        },
    }


@pytest.fixture
def manifest(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "screened-manifest.json"
    path.write_text('{"schema_version": 1, "datasets": {}}')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_common(config: dict[str, object], manifest: Path) -> None:
    assert config["data_module_args"] == {
        "batch_size": 1,
        "data_seed": 42,
        "num_workers": 16,
        "prefetch_factor": 2,
        "multiprocessing_context": "forkserver",
        "persistent_workers": False,
        "pin_memory": False,
        "epoch_len": config["data_module_args"]["epoch_len"],
        "distinct_samples_per_rank_group": True,
    }
    assert config["pl_trainer_args"] == {
        **config["pl_trainer_args"],
        "devices": 8,
        "num_nodes": 1,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 2,
        "log_every_n_steps": 10,
    }
    settings = config["model_update"]["custom"]["settings"]
    assert settings["training_scope"] == "physppi_priors_only"
    assert settings["optimizer"] == {
        "learning_rate": 5.0e-5,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1.0e-8,
    }
    assert settings["lr_scheduler"] == {
        "base_lr": 0.0,
        "warmup_no_steps": 512,
        "start_decay_after_n_steps": 17_920,
        "decay_every_n_steps": 17_920,
        "decay_factor": 1.0,
    }
    assert settings["ema"] == {"decay": 0.999, "submodules_to_update": None}
    assert settings["gradient_clipping"] == {
        "per_sample_clipping": True,
        "clip_val": 10.0,
    }
    architecture = config["model_update"]["custom"]["architecture"]
    assert architecture["shared"]["diffusion"]["no_samples"] == 48
    assert architecture["physppi"]["enabled"] is True
    assert architecture["physppi"]["optional_prior_conditioner"]["enabled"] is True
    assert architecture["physppi"]["membrane_geometry"]["enabled"] is True
    assert architecture["heads"]["pdockq"]["candidate_chunk_size"] == 1
    assert architecture["loss_module"]["confidence_loss_names"] == [
        "plddt",
        "pde",
        "pae",
    ]
    assert architecture["loss_module"]["membrane"] == {
        "normal_weight": 1.0,
        "offset_weight": 1.0,
    }
    assert architecture["loss_module"]["interaction_auxiliary"] == {
        "distance_weight": 1.0
    }
    lora = architecture["lora"]
    assert lora["enabled"] is True
    assert lora["target_preset"] == "physppi_always_on"
    assert lora["target_patterns"] == []
    assert lora["groups"] == LORA_GROUPS
    assert lora["exclude_patterns"] == []
    assert lora["rank"] == 8
    assert lora["alpha"] == 16
    assert lora["dropout"] == 0.05
    assert lora["train_bias"] is False
    assert lora["init"] == "b_zero"
    assert lora["max_trainable_lora_params"] == 131_968
    for name, weight, key in (
        ("qbiolip10k", 0.8, "qbiolip"),
        ("afdb10k", 0.2, "afdb"),
    ):
        dataset = config["dataset_configs"]["train"][name]
        assert dataset["weight"] == weight
        dataset_config = dataset["config"]
        assert dataset_config["template"] == {
            "n_templates": 4,
            "take_top_k": False,
        }
        assert dataset_config["crop"]["token_crop"] == {
            "enabled": True,
            "token_budget": 768,
            "crop_weights": {
                "contiguous": 0.0,
                "spatial": 0.5,
                "spatial_interface": 0.5,
            },
        }
        assert dataset_config["physppi"]["manifest"] == str(manifest)
        assert dataset_config["physppi"]["manifest_dataset_key"] == key
        assert dataset_config["physppi"]["manifest_partition"] == "train_ids"
        assert dataset_config["physppi"]["num_surface_neighbors"] == 64
        assert dataset_config["loss"] == {
            "confidence_loss_names": ["plddt", "pde", "pae"],
            "loss_weights": {"experimentally_resolved": 0.0, "pde": 1.0e-4},
        }
        assert config["dataset_paths"][name]["ccd_file"] == _components_cif()
    assert config["checkpoint_config"] == {
        "every_n_epochs": 1,
        "every_n_train_steps": None,
        "save_on_train_epoch_end": True,
        "save_top_k": -1,
        "save_last": True,
        "filename": "epoch={epoch:02d}-step={step:05d}",
        "auto_insert_metric_name": False,
        "train_time_interval": None,
    }


def test_builder_produces_exact_independent_smoke_and_formal(
    manifest: tuple[Path, str],
) -> None:
    module = _load_module()
    manifest_path, digest = manifest
    smoke, formal = module.build_semantic_configs(
        _baseline(),
        str(manifest_path),
        "v1",
        manifest_sha256=digest,
    )

    _assert_common(smoke, manifest_path)
    _assert_common(formal, manifest_path)
    assert smoke["data_module_args"]["epoch_len"] == 512
    assert smoke["pl_trainer_args"]["max_epochs"] == 1
    assert smoke["pl_trainer_args"]["max_steps"] == 32
    assert formal["data_module_args"]["epoch_len"] == 4096
    assert formal["pl_trainer_args"]["max_epochs"] == 50
    assert formal["pl_trainer_args"]["max_steps"] == 12_800
    assert (
        smoke["experiment_settings"]["output_dir"]
        != formal["experiment_settings"]["output_dir"]
    )
    assert smoke["experiment_settings"] == {
        **smoke["experiment_settings"],
        "seed": 42,
        "restart_checkpoint_path": _canonical_checkpoint(),
        "preemption_safe_resume": False,
        "ckpt_load_settings": {
            "manual_checkpoint_loading": True,
            "init_from_ema_weights": True,
            "restore_lr_scheduler": False,
            "restore_time_step": False,
            "strict_loading": False,
        },
    }
    assert formal["experiment_settings"]["restart_checkpoint_path"] == (
        _canonical_checkpoint()
    )


def test_builder_resolves_production_paths_at_call_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: tuple[Path, str],
) -> None:
    module = _load_module()
    checkpoint = tmp_path / "runtime-canonical.pt"
    components = tmp_path / "runtime-components.cif"
    monkeypatch.setenv("OPENFOLD3_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("PHYSPPI_COMPONENTS_CIF", str(components))
    manifest_path, digest = manifest

    smoke, formal = module.build_semantic_configs(
        _baseline(),
        str(manifest_path),
        "v1",
        manifest_sha256=digest,
    )

    for config in (smoke, formal):
        assert config["experiment_settings"]["restart_checkpoint_path"] == str(
            checkpoint
        )
        assert config["dataset_paths"] == {
            "qbiolip10k": {"ccd_file": str(components)},
            "afdb10k": {"ccd_file": str(components)},
        }


def test_builder_produces_native_recovery_not_manual_loading(
    manifest: tuple[Path, str],
) -> None:
    module = _load_module()
    manifest_path, digest = manifest
    _smoke, formal = module.build_semantic_configs(
        _baseline(), str(manifest_path), "v1", manifest_sha256=digest
    )
    recovery = module.build_recovery_config(
        formal,
        "/runs/semantic-v1/checkpoints/last.ckpt",
        manifest_sha256=digest,
    )

    assert recovery["experiment_settings"]["restart_checkpoint_path"].endswith(
        "last.ckpt"
    )
    assert recovery["experiment_settings"]["ckpt_load_settings"] == {
        "manual_checkpoint_loading": False,
        "init_from_ema_weights": False,
        "restore_lr_scheduler": False,
        "restore_time_step": False,
        "strict_loading": True,
    }
    assert recovery["pl_trainer_args"]["max_epochs"] == 50
    assert recovery["pl_trainer_args"]["max_steps"] == 12_800
    module.validate_semantic_config(
        recovery,
        smoke=False,
        manifest_sha256=digest,
        recovery=True,
    )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("pl_trainer_args", "devices"), 4, "devices"),
        (("pl_trainer_args", "max_steps"), 256, "max_steps"),
        (("data_module_args", "pin_memory"), True, "pin_memory"),
        (
            ("model_update", "custom", "settings", "ema", "decay"),
            0.99,
            "ema",
        ),
        (
            (
                "dataset_configs",
                "train",
                "qbiolip10k",
                "config",
                "crop",
                "token_crop",
                "token_budget",
            ),
            512,
            "token_budget",
        ),
        (
            (
                "model_update",
                "custom",
                "architecture",
                "lora",
                "groups",
            ),
            [],
            "LoRA groups",
        ),
        (
            (
                "model_update",
                "custom",
                "architecture",
                "physppi",
                "training_scope",
            ),
            "baseline",
            "training_scope",
        ),
    ],
)
def test_validator_rejects_contract_drift(
    manifest: tuple[Path, str],
    path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    module = _load_module()
    manifest_path, digest = manifest
    _smoke, formal = module.build_semantic_configs(
        _baseline(), str(manifest_path), "v1", manifest_sha256=digest
    )
    cursor = formal
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        module.validate_semantic_config(formal, smoke=False, manifest_sha256=digest)


def test_validator_rejects_extra_training_source(
    manifest: tuple[Path, str],
) -> None:
    module = _load_module()
    manifest_path, digest = manifest
    _smoke, formal = module.build_semantic_configs(
        _baseline(), str(manifest_path), "v1", manifest_sha256=digest
    )
    formal["dataset_configs"]["train"]["unexpected"] = {
        "weight": 0.1,
        "config": {},
    }

    with pytest.raises(ValueError, match="training datasets"):
        module.validate_semantic_config(formal, smoke=False, manifest_sha256=digest)


def test_validator_rejects_wrong_manifest_hash(
    manifest: tuple[Path, str],
) -> None:
    module = _load_module()
    manifest_path, _digest = manifest
    with pytest.raises(ValueError, match="SHA256"):
        module.build_semantic_configs(
            _baseline(),
            str(manifest_path),
            "v1",
            manifest_sha256="0" * 64,
        )


def test_yaml_roundtrip_and_refuses_reuse(
    tmp_path: Path, manifest: tuple[Path, str]
) -> None:
    module = _load_module()
    manifest_path, digest = manifest
    smoke, _formal = module.build_semantic_configs(
        _baseline(), str(manifest_path), "v1", manifest_sha256=digest
    )
    output = tmp_path / "smoke.yml"
    module.write_yaml(output, smoke)
    assert yaml.safe_load(output.read_text()) == smoke
    with pytest.raises(FileExistsError, match="historical YAML"):
        module.write_yaml(output, smoke)


def test_pin_memory_is_strict_and_available_to_runtime_config() -> None:
    assert DataModuleArgs(pin_memory=False).pin_memory is False
    with pytest.raises(ValidationError):
        DataModuleArgs(pin_memory="false")
    assert "pin_memory" in DataModuleConfig.model_fields
    assert DataModuleConfig.model_fields["pin_memory"].default is False


def test_actual_locked_model_config_accepts_semantic_loss_overrides(
    manifest: tuple[Path, str],
) -> None:
    module = _load_module()
    manifest_path, digest = manifest
    smoke, _formal = module.build_semantic_configs(
        _baseline(), str(manifest_path), "v1", manifest_sha256=digest
    )

    config = OF3ProjectEntry().get_model_config_with_update(
        ModelUpdate.model_validate(smoke["model_update"])
    )

    assert config.architecture.loss_module.interaction_auxiliary.distance_weight == 1.0
