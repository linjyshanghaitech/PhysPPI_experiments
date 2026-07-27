#!/usr/bin/env python3
# ruff: noqa: E402
"""Build immutable PhysPPI smoke, formal, and native-recovery YAMLs."""

from __future__ import annotations

if __package__:
    from scripts.physppi._bootstrap import ensure_repository_root
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import ensure_repository_root

ensure_repository_root()

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CANONICAL_CHECKPOINT = "/101045/jinyu/.openfold3/of3-p2-155k.pt"
COMPONENTS_CIF = (
    "/101045/jinyu/PhysPPI/openfold-3/output/physppi_msa_cache/components.cif"
)
TRAINING_SCOPE = "physppi_lora_assessment"
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

_DATA_MODULE_COMMON = {
    "batch_size": 1,
    "data_seed": 42,
    "num_workers": 16,
    "prefetch_factor": 2,
    "multiprocessing_context": "forkserver",
    "persistent_workers": False,
    "pin_memory": False,
    "distinct_samples_per_rank_group": True,
}
_TRAINER_COMMON = {
    "devices": 8,
    "num_nodes": 1,
    "precision": "bf16-mixed",
    "accumulate_grad_batches": 2,
    "log_every_n_steps": 10,
}
_OPTIMIZER = {
    "learning_rate": 5.0e-5,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1.0e-8,
}
_LR_SCHEDULER = {
    "base_lr": 0.0,
    "warmup_no_steps": 512,
    "start_decay_after_n_steps": 17_920,
    "decay_every_n_steps": 17_920,
    "decay_factor": 1.0,
}
_CHECKPOINT = {
    "every_n_epochs": 1,
    "every_n_train_steps": None,
    "save_on_train_epoch_end": True,
    "save_top_k": -1,
    "save_last": True,
    "filename": "epoch={epoch:02d}-step={step:05d}",
    "auto_insert_metric_name": False,
    "train_time_interval": None,
}
_COLD_LOAD_SETTINGS = {
    "manual_checkpoint_loading": True,
    "init_from_ema_weights": True,
    "restore_lr_scheduler": False,
    "restore_time_step": False,
    "strict_loading": False,
}
_RECOVERY_LOAD_SETTINGS = {
    "manual_checkpoint_loading": False,
    "init_from_ema_weights": False,
    "restore_lr_scheduler": False,
    "restore_time_step": False,
    "strict_loading": True,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _output_dir(base: dict[str, Any], version: str, *, smoke: bool) -> str:
    parent = Path(base["experiment_settings"]["output_dir"]).parent
    suffix = (
        f"stage1-semantic-{version}-smoke32"
        if smoke
        else (f"stage1-semantic-{version}")
    )
    return str(parent / suffix)


def _configure_dataset(
    dataset: dict[str, Any],
    *,
    manifest: str,
    manifest_dataset_key: str,
    weight: float,
) -> None:
    dataset["weight"] = weight
    config = dataset["config"]
    config["template"] = {"n_templates": 4, "take_top_k": False}
    crop = config.setdefault("crop", {})
    crop["token_crop"] = {
        "enabled": True,
        "token_budget": 768,
        "crop_weights": {
            "contiguous": 0.0,
            "spatial": 0.5,
            "spatial_interface": 0.5,
        },
    }
    physppi = config.setdefault("physppi", {})
    physppi.update(
        {
            "manifest": manifest,
            "manifest_dataset_key": manifest_dataset_key,
            "manifest_partition": "train_ids",
            "num_surface_neighbors": 128,
        }
    )
    config["loss"] = {
        "confidence_loss_names": ["plddt", "pde", "pae"],
        "loss_weights": {"experimentally_resolved": 0.0, "pde": 1.0e-4},
    }


def _apply_contract(
    config: dict[str, Any],
    *,
    manifest: str,
    version: str,
    smoke: bool,
) -> None:
    config["experiment_settings"] = {
        **config.get("experiment_settings", {}),
        "mode": "train",
        "output_dir": _output_dir(config, version, smoke=smoke),
        "seed": 42,
        "restart_checkpoint_path": CANONICAL_CHECKPOINT,
        "preemption_safe_resume": False,
        "ckpt_load_settings": copy.deepcopy(_COLD_LOAD_SETTINGS),
    }
    config["data_module_args"] = {
        **_DATA_MODULE_COMMON,
        "epoch_len": 512 if smoke else 4096,
    }
    config["pl_trainer_args"] = {
        **_TRAINER_COMMON,
        "max_epochs": 1 if smoke else 50,
        "max_steps": 32 if smoke else 12_800,
    }
    config["checkpoint_config"] = copy.deepcopy(_CHECKPOINT)

    custom = config.setdefault("model_update", {}).setdefault("custom", {})
    settings = custom.setdefault("settings", {})
    settings.update(
        {
            "training_scope": TRAINING_SCOPE,
            "optimizer": copy.deepcopy(_OPTIMIZER),
            "lr_scheduler": copy.deepcopy(_LR_SCHEDULER),
            "ema": {"decay": 0.999, "submodules_to_update": None},
            "gradient_clipping": {
                "per_sample_clipping": True,
                "clip_val": 10.0,
            },
        }
    )
    architecture = custom.setdefault("architecture", {})
    architecture.setdefault("shared", {}).setdefault("diffusion", {})["no_samples"] = 48
    physppi = architecture.setdefault("physppi", {})
    physppi.update({"enabled": True, "training_scope": TRAINING_SCOPE})
    physppi.setdefault("optional_prior_conditioner", {})["enabled"] = True
    physppi.setdefault("membrane_geometry", {})["enabled"] = True
    architecture["lora"] = {
        "enabled": True,
        "target_preset": "physppi_always_on",
        "target_patterns": [],
        "groups": copy.deepcopy(LORA_GROUPS),
        "exclude_patterns": [],
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "train_bias": False,
        "init": "b_zero",
        "max_trainable_lora_params": 131_968,
    }
    architecture.setdefault("heads", {}).setdefault("pdockq", {})[
        "candidate_chunk_size"
    ] = 1
    loss_module = architecture.setdefault("loss_module", {})
    loss_module["confidence_loss_names"] = ["plddt", "pde", "pae"]
    loss_module["membrane"] = {"normal_weight": 1.0, "offset_weight": 1.0}
    loss_module["interaction_auxiliary"] = {"distance_weight": 1.0}

    train = config["dataset_configs"]["train"]
    _configure_dataset(
        train["qbiolip10k"],
        manifest=manifest,
        manifest_dataset_key="qbiolip",
        weight=0.8,
    )
    _configure_dataset(
        train["afdb10k"],
        manifest=manifest,
        manifest_dataset_key="afdb",
        weight=0.2,
    )
    for name in ("qbiolip10k", "afdb10k"):
        config["dataset_paths"][name]["ccd_file"] = COMPONENTS_CIF


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")


def _validate_manifest(config: dict[str, Any], manifest_sha256: str | None) -> str:
    train = config["dataset_configs"]["train"]
    paths = {
        train[name]["config"]["physppi"]["manifest"]
        for name in ("qbiolip10k", "afdb10k")
    }
    if len(paths) != 1:
        raise ValueError("both datasets must use the same manifest path")
    manifest = paths.pop()
    if manifest_sha256 is not None:
        actual = _sha256(Path(manifest))
        if actual != manifest_sha256:
            raise ValueError(
                f"manifest SHA256 mismatch: expected {manifest_sha256}, got {actual}"
            )
    return manifest


def _validate_no_forbidden_losses(config: dict[str, Any]) -> None:
    serialized = json.dumps(config, sort_keys=True).lower()
    normalized = serialized.replace("_", "-")
    if "penergy" in serialized or "delta-g" in normalized:
        raise ValueError("pEnergy and Delta-G must remain absent")


def validate_semantic_config(
    config: dict[str, object],
    *,
    smoke: bool,
    manifest_sha256: str | None,
    recovery: bool = False,
) -> None:
    typed: dict[str, Any] = config
    expected_epoch_len = 512 if smoke else 4096
    expected_epochs = 1 if smoke else 50
    expected_steps = 32 if smoke else 12_800
    _require_equal(
        typed["data_module_args"],
        {**_DATA_MODULE_COMMON, "epoch_len": expected_epoch_len},
        "data_module_args (including pin_memory)",
    )
    _require_equal(
        typed["pl_trainer_args"],
        {
            **_TRAINER_COMMON,
            "max_epochs": expected_epochs,
            "max_steps": expected_steps,
        },
        "pl_trainer_args max_epochs/max_steps/devices",
    )
    trainer = typed["pl_trainer_args"]
    updates_per_epoch = (
        typed["data_module_args"]["epoch_len"]
        // trainer["devices"]
        // trainer["accumulate_grad_batches"]
    )
    _require_equal(
        updates_per_epoch,
        32 if smoke else 256,
        "optimizer updates per epoch",
    )
    _require_equal(typed["checkpoint_config"], _CHECKPOINT, "checkpoint_config")

    settings = typed["model_update"]["custom"]["settings"]
    _require_equal(settings["training_scope"], TRAINING_SCOPE, "training_scope")
    _require_equal(settings["optimizer"], _OPTIMIZER, "optimizer")
    _require_equal(settings["lr_scheduler"], _LR_SCHEDULER, "lr_scheduler")
    _require_equal(
        settings["ema"],
        {"decay": 0.999, "submodules_to_update": None},
        "ema",
    )
    _require_equal(
        settings["gradient_clipping"],
        {"per_sample_clipping": True, "clip_val": 10.0},
        "gradient_clipping",
    )

    architecture = typed["model_update"]["custom"]["architecture"]
    _require_equal(
        architecture["shared"]["diffusion"]["no_samples"],
        48,
        "diffusion no_samples",
    )
    _require_equal(architecture["physppi"]["enabled"], True, "PhysPPI enabled")
    _require_equal(
        architecture["physppi"]["training_scope"],
        TRAINING_SCOPE,
        "PhysPPI training_scope",
    )
    _require_equal(
        architecture["physppi"]["optional_prior_conditioner"]["enabled"],
        True,
        "mandatory prior conditioner",
    )
    _require_equal(
        architecture["physppi"]["membrane_geometry"]["enabled"],
        True,
        "directed membrane geometry",
    )
    _require_equal(
        architecture["heads"]["pdockq"]["candidate_chunk_size"],
        1,
        "pDockQ candidate chunk size",
    )
    _require_equal(
        architecture["loss_module"]["confidence_loss_names"],
        ["plddt", "pde", "pae"],
        "confidence losses",
    )
    _require_equal(
        architecture["loss_module"]["membrane"],
        {"normal_weight": 1.0, "offset_weight": 1.0},
        "membrane loss",
    )
    _require_equal(
        architecture["loss_module"]["interaction_auxiliary"],
        {"distance_weight": 1.0},
        "interaction auxiliary",
    )
    expected_lora = {
        "enabled": True,
        "target_preset": "physppi_always_on",
        "target_patterns": [],
        "groups": LORA_GROUPS,
        "exclude_patterns": [],
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "train_bias": False,
        "init": "b_zero",
        "max_trainable_lora_params": 131_968,
    }
    if architecture["lora"].get("groups") != LORA_GROUPS:
        raise ValueError("LoRA groups must match the exact production contract")
    _require_equal(architecture["lora"], expected_lora, "LoRA contract")

    manifest = _validate_manifest(typed, manifest_sha256)
    train = typed["dataset_configs"]["train"]
    _require_equal(
        set(train),
        {"qbiolip10k", "afdb10k"},
        "training datasets",
    )
    for name, weight, key in (
        ("qbiolip10k", 0.8, "qbiolip"),
        ("afdb10k", 0.2, "afdb"),
    ):
        dataset = train[name]
        _require_equal(dataset["weight"], weight, f"{name} weight")
        dataset_config = dataset["config"]
        _require_equal(
            dataset_config["template"],
            {"n_templates": 4, "take_top_k": False},
            f"{name} template",
        )
        _require_equal(
            dataset_config["crop"]["token_crop"],
            {
                "enabled": True,
                "token_budget": 768,
                "crop_weights": {
                    "contiguous": 0.0,
                    "spatial": 0.5,
                    "spatial_interface": 0.5,
                },
            },
            f"{name} token_budget/crop weights",
        )
        physppi = dataset_config["physppi"]
        _require_equal(physppi["manifest"], manifest, f"{name} manifest")
        _require_equal(
            physppi["manifest_dataset_key"], key, f"{name} manifest dataset key"
        )
        _require_equal(
            physppi["manifest_partition"], "train_ids", f"{name} manifest partition"
        )
        _require_equal(
            physppi["num_surface_neighbors"], 128, f"{name} surface neighbors"
        )
        _require_equal(
            dataset_config["loss"],
            {
                "confidence_loss_names": ["plddt", "pde", "pae"],
                "loss_weights": {
                    "experimentally_resolved": 0.0,
                    "pde": 1.0e-4,
                },
            },
            f"{name} losses",
        )
        _require_equal(
            typed["dataset_paths"][name]["ccd_file"],
            COMPONENTS_CIF,
            f"{name} ccd_file",
        )

    experiment = typed["experiment_settings"]
    _require_equal(experiment["seed"], 42, "seed")
    _require_equal(
        experiment["preemption_safe_resume"], False, "preemption_safe_resume"
    )
    expected_load = _RECOVERY_LOAD_SETTINGS if recovery else _COLD_LOAD_SETTINGS
    _require_equal(
        experiment["ckpt_load_settings"], expected_load, "checkpoint load settings"
    )
    if recovery:
        if experiment["restart_checkpoint_path"] == CANONICAL_CHECKPOINT:
            raise ValueError("recovery requires a complete PhysPPI checkpoint")
    else:
        _require_equal(
            experiment["restart_checkpoint_path"],
            CANONICAL_CHECKPOINT,
            "cold-start checkpoint",
        )
    _validate_no_forbidden_losses(typed)


def build_semantic_configs(
    base: dict[str, object],
    manifest: str,
    version: str,
    *,
    manifest_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    smoke = copy.deepcopy(base)
    formal = copy.deepcopy(base)
    _apply_contract(smoke, manifest=manifest, version=version, smoke=True)
    _apply_contract(formal, manifest=manifest, version=version, smoke=False)
    if (
        smoke["experiment_settings"]["output_dir"]
        == formal["experiment_settings"]["output_dir"]
    ):
        raise ValueError("smoke and formal output directories must differ")
    validate_semantic_config(smoke, smoke=True, manifest_sha256=manifest_sha256)
    validate_semantic_config(formal, smoke=False, manifest_sha256=manifest_sha256)
    return smoke, formal


def build_recovery_config(
    formal: dict[str, object],
    checkpoint: str,
    *,
    manifest_sha256: str | None,
) -> dict[str, object]:
    recovery = copy.deepcopy(formal)
    experiment = recovery["experiment_settings"]
    experiment["restart_checkpoint_path"] = checkpoint
    experiment["preemption_safe_resume"] = False
    experiment["ckpt_load_settings"] = copy.deepcopy(_RECOVERY_LOAD_SETTINGS)
    validate_semantic_config(
        recovery,
        smoke=False,
        manifest_sha256=manifest_sha256,
        recovery=True,
    )
    return recovery


def write_yaml(path: Path, config: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to reuse historical YAML: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--version", required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    parser.add_argument("--formal-output", type=Path, required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_sha256 = args.manifest_sha256 or _sha256(args.manifest)
    baseline = yaml.safe_load(args.baseline.read_text())
    smoke, formal = build_semantic_configs(
        baseline,
        str(args.manifest),
        args.version,
        manifest_sha256=manifest_sha256,
    )
    write_yaml(args.smoke_output, smoke)
    write_yaml(args.formal_output, formal)
    if (args.resume_checkpoint is None) != (args.resume_output is None):
        raise ValueError(
            "--resume-checkpoint and --resume-output must be provided together"
        )
    if args.resume_checkpoint is not None:
        recovery = build_recovery_config(
            formal,
            args.resume_checkpoint,
            manifest_sha256=manifest_sha256,
        )
        write_yaml(args.resume_output, recovery)


if __name__ == "__main__":
    main()
