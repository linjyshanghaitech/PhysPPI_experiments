from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH = REPO_ROOT / "experiments" / "_runtime" / "4gpu_train.yaml"
MANIFEST_PATH = (
    REPO_ROOT / "experiments" / "split_training_val" / "split_manifest.json"
)
SAMPLING_WEIGHTS_PATH = Path(
    "/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/sampling_weights.pt"
)


def _runtime_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_runtime_config_uses_requested_split_manifest():
    config = _runtime_config()
    dataset_configs = config["dataset_configs"]

    manifests = {
        dataset["config"]["physppi"]["manifest"]
        for mode in ("train", "validation")
        for dataset in dataset_configs[mode].values()
    }

    assert manifests == {str(MANIFEST_PATH)}


def test_runtime_config_uses_sampling_weights_only_for_training():
    config = _runtime_config()
    train = config["dataset_configs"]["train"]
    validation = config["dataset_configs"]["validation"]

    assert {
        dataset["config"]["physppi"]["sampling_weights"]
        for dataset in train.values()
    } == {str(SAMPLING_WEIGHTS_PATH)}
    assert all(
        "sampling_weights" not in dataset["config"]["physppi"]
        for dataset in validation.values()
    )


def test_runtime_config_uses_one_global_training_case_distribution():
    config = _runtime_config()
    train = config["dataset_configs"]["train"]

    assert config["data_module_args"]["global_datapoint_sampling"] is True
    assert [dataset["weight"] for dataset in train.values()] == [0.5, 0.5]


def test_runtime_config_scales_only_afdb_training_total_loss():
    config = _runtime_config()
    train = config["dataset_configs"]["train"]
    validation = config["dataset_configs"]["validation"]

    assert train["qbiolip_train"]["config"]["physppi"]["loss_weight"] == 1.0
    assert train["afdb_train"]["config"]["physppi"]["loss_weight"] == 0.8
    assert all(
        "loss_weight" not in dataset["config"]["physppi"]
        for dataset in validation.values()
    )


def test_runtime_config_runs_complete_qbiolip_and_afdb_validation():
    config = _runtime_config()
    validation = config["dataset_configs"]["validation"]

    assert validation["qbiolip_val"]["dataset_class"] == "QBioLiP10kDataset"
    assert validation["qbiolip_val"]["config"]["physppi"]["manifest_dataset_key"] == (
        "qbiolip"
    )
    assert validation["afdb_val"]["dataset_class"] == "AFDB10kDataset"
    assert validation["afdb_val"]["config"]["physppi"]["manifest_dataset_key"] == "afdb"
    assert all(
        "max_samples" not in dataset["config"]["physppi"]
        for dataset in validation.values()
    )
    assert Path(
        config["dataset_paths"]["afdb_val"]["alignments_directory"]
    ).name == "AFDB-10k-1"
