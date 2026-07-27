# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    TrainingDatasetSpec,
)

CANONICAL_PHYSPPI_BASE_CHECKPOINT = Path("/101045/jinyu/.openfold3/of3-p2-155k.pt")


def _manifest(path: Path, sample_ids: list[str]) -> Path:
    path.write_text("\n".join(sample_ids) + "\n")
    return path


def _dataset_spec(
    tmp_path: Path,
    *,
    name: str,
    dataset_class: str,
    manifest: Path,
    manifest_dataset_key: str,
    weight: float,
    sample_in_order: bool = False,
) -> TrainingDatasetSpec:
    cache_file = tmp_path / f"{name}-cache.json"
    cache_file.write_text("{}")
    root = tmp_path / f"{name}-root"
    root.mkdir(exist_ok=True)
    return TrainingDatasetSpec.model_validate(
        {
            "name": name,
            "dataset_class": dataset_class,
            "mode": "train",
            "weight": weight,
            "config": {
                "sample_in_order": sample_in_order,
                "dataset_paths": {
                    "alignments_directory": tmp_path,
                    "dataset_cache_file": cache_file,
                    "target_structures_directory": tmp_path,
                    "target_structure_file_format": "npz",
                    "reference_molecule_directory": tmp_path,
                },
                "physppi": {
                    "root": root,
                    "manifest": manifest,
                    "manifest_dataset_key": manifest_dataset_key,
                },
            },
        }
    )


def _data_module(
    tmp_path: Path,
    qbiolip_manifest: Path,
    afdb_manifest: Path | None = None,
    *,
    sample_in_order: bool = False,
) -> DataModule:
    specs = [
        _dataset_spec(
            tmp_path,
            name="qbiolip",
            dataset_class="QBioLiP10kDataset",
            manifest=qbiolip_manifest,
            manifest_dataset_key="qbiolip",
            weight=1.0 if afdb_manifest is None else 0.8,
            sample_in_order=sample_in_order,
        )
    ]
    if afdb_manifest is not None:
        specs.append(
            _dataset_spec(
                tmp_path,
                name="afdb",
                dataset_class="AFDB10kDataset",
                manifest=afdb_manifest,
                manifest_dataset_key="afdb",
                weight=0.2,
            )
        )
    return DataModule(DataModuleConfig(datasets=specs, data_seed=42))


def test_state_contains_sha256_for_each_training_manifest(tmp_path: Path) -> None:
    qbiolip_manifest = _manifest(tmp_path / "qbiolip.json", ["q1", "q2"])
    afdb_manifest = _manifest(tmp_path / "afdb.json", ["a1"])

    state = _data_module(tmp_path, qbiolip_manifest, afdb_manifest).state_dict()

    assert state["manifest_sha256_by_dataset"] == {
        "qbiolip": hashlib.sha256(qbiolip_manifest.read_bytes()).hexdigest(),
        "afdb": hashlib.sha256(afdb_manifest.read_bytes()).hexdigest(),
    }


def test_same_manifest_restores_native_datamodule_state(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json", ["a", "b"])
    original = _data_module(tmp_path, manifest, sample_in_order=True)
    state = original.state_dict() | {
        "next_dataset_indices": {"qbiolip": 17},
        "next_epoch": 7,
    }
    resumed = _data_module(tmp_path, manifest, sample_in_order=True)
    resumed.generators["train"] = torch.Generator().manual_seed(999)

    resumed.load_state_dict(state)

    assert resumed.next_epoch == 7
    assert resumed.next_data_seed == 49
    assert resumed.next_dataset_indices == {"qbiolip": 17}
    assert resumed.generators == {}
    assert resumed.manifest_changed_on_resume is False


def test_changed_manifest_preserves_epoch_but_resets_sampling_state(
    tmp_path: Path,
) -> None:
    old_manifest = _manifest(tmp_path / "old.json", ["a", "b"])
    state = _data_module(tmp_path, old_manifest).state_dict() | {
        "next_dataset_indices": {"qbiolip": 17},
        "next_epoch": 7,
    }
    new_manifest = _manifest(tmp_path / "new.json", ["a"])
    resumed = _data_module(tmp_path, new_manifest)
    resumed.generators["train"] = torch.Generator().manual_seed(999)

    resumed.load_state_dict(state)

    assert resumed.next_epoch == 7
    assert resumed.next_data_seed == 49
    assert resumed.next_dataset_indices == {}
    assert resumed.generators == {}
    assert resumed.manifest_changed_on_resume is True


def _policy_runner(
    *,
    checkpoint: str,
    manual: bool,
    init_from_ema: bool,
) -> TrainingExperimentRunner:
    runner = object.__new__(TrainingExperimentRunner)
    runner.dataset_configs = {
        "train": {
            "qbiolip": {
                "config": {
                    "physppi": {
                        "manifest": "/tmp/screened-manifest.json",
                    }
                }
            }
        }
    }
    runner.restart_checkpoint_path = checkpoint
    runner.preemption_safe_resume = False
    runner.ckpt_load_settings = SimpleNamespace(
        manual_checkpoint_loading=manual,
        init_from_ema_weights=init_from_ema,
    )
    return runner


def test_physppi_cold_start_requires_canonical_ema_manual_loading() -> None:
    runner = _policy_runner(
        checkpoint=str(CANONICAL_PHYSPPI_BASE_CHECKPOINT),
        manual=True,
        init_from_ema=True,
    )

    runner._validate_physppi_checkpoint_loading_policy()
    assert runner.ckpt_path is None

    runner.ckpt_load_settings.init_from_ema_weights = False
    with pytest.raises(ValueError, match="EMA"):
        runner._validate_physppi_checkpoint_loading_policy()


def test_physppi_recovery_requires_native_lightning_resume(tmp_path: Path) -> None:
    recovery = tmp_path / "epoch.ckpt"
    recovery.write_text("checkpoint")
    runner = _policy_runner(
        checkpoint=str(recovery),
        manual=False,
        init_from_ema=False,
    )

    runner._validate_physppi_checkpoint_loading_policy()
    assert runner.ckpt_path == str(recovery)

    runner.ckpt_load_settings.manual_checkpoint_loading = True
    with pytest.raises(ValueError, match="native Lightning"):
        runner._validate_physppi_checkpoint_loading_policy()
