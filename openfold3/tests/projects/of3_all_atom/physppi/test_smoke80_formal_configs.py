from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[5]
SMOKE_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "_runtime"
    / "8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80.yaml"
)
FORMAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "_runtime"
    / "8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730.yaml"
)
VALIDATION10_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "_runtime"
    / "8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80"
    / "validation10_eval.yaml"
)
VALIDATION16_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "_runtime"
    / "8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80"
    / "validation16_eval.yaml"
)
BASE_CHECKPOINT = "/101045/jinyu/.openfold3/of3-p2-155k.pt"


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _assert_fresh_log_only_config(config: dict, output_dir: str) -> None:
    experiment = config["experiment_settings"]
    checkpoint = experiment["ckpt_load_settings"]
    loss_config = config["model_update"]["custom"]["architecture"]["loss_module"]

    assert experiment["output_dir"] == output_dir
    assert experiment["restart_checkpoint_path"] == BASE_CHECKPOINT
    assert experiment["seed"] == 42
    assert checkpoint["manual_checkpoint_loading"] is True
    assert checkpoint["init_from_ema_weights"] is True
    assert config["data_module_args"]["data_seed"] == 42
    assert config["checkpoint_config"]["dirpath"] == f"{output_dir}/checkpoint"
    assert loss_config["membrane"]["include_in_total_loss"] is False
    assert (
        loss_config["interaction_auxiliary"]["include_in_total_loss"] is False
    )


def test_smoke_config_is_fresh_isolated_and_stops_after_50_train_steps():
    config = _load_config(SMOKE_CONFIG_PATH)
    output_dir = (
        "experiments/_runtime/"
        "8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80"
    )

    _assert_fresh_log_only_config(config, output_dir)
    assert config["pl_trainer_args"]["max_steps"] == 50
    assert config["pl_trainer_args"]["val_check_interval"] == 50
    assert config["pl_trainer_args"]["limit_val_batches"] == 0


def test_validation10_config_uses_step50_checkpoint_and_exactly_10_cases():
    config = _load_config(VALIDATION10_CONFIG_PATH)
    experiment = config["experiment_settings"]
    checkpoint = experiment["ckpt_load_settings"]
    trainer = config["pl_trainer_args"]
    validation = config["dataset_configs"]["validation"]
    loss_config = config["model_update"]["custom"]["architecture"]["loss_module"]

    assert experiment["mode"] == "eval"
    assert experiment["output_dir"].endswith("/validation10_eval")
    assert experiment["restart_checkpoint_path"].endswith(
        "/pre_val_cap_20260730/checkpoint/step-000050.ckpt"
    )
    assert checkpoint == {
        "manual_checkpoint_loading": True,
        "init_from_ema_weights": False,
        "restore_lr_scheduler": False,
        "restore_time_step": False,
        "strict_loading": True,
    }
    assert trainer["devices"] == 1
    assert trainer["limit_val_batches"] == 1.0
    assert trainer["num_sanity_val_steps"] == 0
    assert loss_config["membrane"]["include_in_total_loss"] is False
    assert (
        loss_config["interaction_auxiliary"]["include_in_total_loss"] is False
    )

    for spec in config["dataset_configs"]["train"].values():
        physppi = spec["config"]["physppi"]
        assert "manifest" not in physppi
        assert "sampling_weights" not in physppi
        assert len(physppi["sample_ids"]) == 1

    selected_ids = []
    for spec in validation.values():
        physppi = spec["config"]["physppi"]
        assert "manifest" not in physppi
        assert len(physppi["sample_ids"]) == 5
        selected_ids.extend(physppi["sample_ids"])
    assert len(selected_ids) == 10
    assert len(set(selected_ids)) == 10


def test_validation16_config_uses_8gpu_24_rollouts_and_artifact_gate():
    config = _load_config(VALIDATION16_CONFIG_PATH)
    experiment = config["experiment_settings"]
    trainer = config["pl_trainer_args"]
    custom = config["model_update"]["custom"]
    diffusion = custom["architecture"]["shared"]["diffusion"]
    artifacts = custom["settings"]["validation_artifacts"]
    validation = config["dataset_configs"]["validation"]

    assert experiment["mode"] == "eval"
    assert experiment["output_dir"].endswith("/validation16_eval")
    assert experiment["restart_checkpoint_path"].endswith(
        "/pre_val_cap_20260730/checkpoint/step-000050.ckpt"
    )
    assert trainer["devices"] == 8
    assert trainer["limit_val_batches"] == 1.0
    assert trainer["num_sanity_val_steps"] == 0
    assert diffusion["no_samples"] == 24
    assert diffusion["no_full_rollout_samples"] == 24
    assert artifacts == {
        "enabled": True,
        "output_dir": (
            "experiments/_runtime/"
            "8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/"
            "validation16_eval/validation_artifacts"
        ),
        "structure_format": "pdb",
        "expected_samples": 24,
        "fail_on_sample_count_mismatch": True,
        "profile_timing": True,
        "synchronize_cuda_timing": True,
    }

    selected_ids = []
    for spec in validation.values():
        physppi = spec["config"]["physppi"]
        assert len(physppi["sample_ids"]) == 8
        selected_ids.extend(physppi["sample_ids"])
    assert len(selected_ids) == 16
    assert len(set(selected_ids)) == 16


def test_formal_config_can_reach_10000_steps_after_smoke_gate():
    config = _load_config(FORMAL_CONFIG_PATH)
    output_dir = (
        "experiments/_runtime/"
        "8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730"
    )

    _assert_fresh_log_only_config(config, output_dir)
    assert config["pl_trainer_args"]["max_steps"] == 10000
    assert config["pl_trainer_args"]["max_epochs"] == -1
    assert config["pl_trainer_args"]["val_check_interval"] == 512
    assert config["pl_trainer_args"]["num_sanity_val_steps"] == 0
    assert "limit_val_batches" not in config["pl_trainer_args"]
