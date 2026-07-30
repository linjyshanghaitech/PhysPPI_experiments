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

import csv
import gc
import importlib
import json
import logging
import os
import shutil
import time
import traceback
import warnings
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from torchmetrics import MeanMetric, MetricCollection, PearsonCorrCoef

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.loss.pdockq import (
    dockq_to_soft_bins,
    per_candidate_complex_dockq,
)
from openfold3.core.metrics.confidence import compute_plddt
from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores
from openfold3.core.metrics.model_selection import (
    compute_final_model_selection_metric,
    compute_valid_model_selection_metrics,
)
from openfold3.core.metrics.quality import (
    get_metrics,
    get_metrics_chunked,
)
from openfold3.core.model.lora import (
    is_lora_parameter_name,
    validate_lora_checkpoint_keys,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.runners.writer import OF3OutputWriter
from openfold3.core.utils.grad_manager import PerSampleGradManager, compute_global_norm
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.core.utils.timing import PerformanceTimer
from openfold3.projects.of3_all_atom.config.model_config import (
    PHYSPPI_TRAINABLE_PREFIXES,
    PHYSPPI_TRAINING_SCOPE,
    model_selection_metric_weights_config,
    validate_physppi_semantic_config,
)
from openfold3.projects.of3_all_atom.constants import (
    CORRELATION_METRICS,
    METRIC_DENOMINATOR_ATTRS,
    TRAIN_LOGGED_METRICS,
    TRAIN_LOSSES,
    VAL_LOGGED_METRICS,
    VAL_LOSSES,
)
from openfold3.projects.of3_all_atom.model import OpenFold3

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

logger = logging.getLogger(__name__)
# We define extra metrics that will cause this warning depending on the training stage
# Only metrics with values present are logged, so we can ignore this error
warnings.filterwarnings(
    "ignore",
    message=(
        r"The ``compute`` method of metric .* was called before the ``update`` method"
    ),
    category=UserWarning,
    module="torchmetrics",
)

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


def _trace_scalar(value, name: str) -> int:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must contain exactly one value per rank")
    return int(tensor.item())


def sampler_trace_row(
    batch: dict,
    *,
    rank: int,
    epoch: int,
    micro_step: int,
    optimizer_step: int | None = None,
) -> dict[str, int | float | str]:
    """Build one rank-local semantic sampler trace row before model forward."""
    sample_ids = batch.get("pdb_id")
    if not isinstance(sample_ids, (list, tuple)) or len(sample_ids) != 1:
        raise ValueError("semantic sampler trace requires one pdb_id per rank")
    sample_id = str(sample_ids[0])
    occurrence_seed = _trace_scalar(
        batch["physppi_occurrence_seed"], "physppi_occurrence_seed"
    )
    contact_child = np.random.SeedSequence(occurrence_seed).spawn(6)[1]
    contact_cutoff = float(np.random.default_rng(contact_child).uniform(8.0, 18.0))
    row: dict[str, int | float | str] = {
        "rank": int(rank),
        "epoch": int(epoch),
        "micro_step": int(micro_step),
        "dataset": "afdb" if sample_id.startswith("AF-") else "qbiolip",
        "sample_id": sample_id,
        "occurrence_seed": occurrence_seed,
        "requested_level": _trace_scalar(
            batch["physppi_prior_requested_level"],
            "physppi_prior_requested_level",
        ),
        "effective_level": _trace_scalar(
            batch["physppi_prior_effective_level"],
            "physppi_prior_effective_level",
        ),
        "contact_cutoff": contact_cutoff,
    }
    if optimizer_step is not None:
        row["optimizer_step"] = int(optimizer_step)
    return row


def apply_physppi_loss_weight(
    batch: dict,
    loss: torch.Tensor,
    loss_breakdown: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scale the complete rank-local PhysPPI loss after aggregation."""
    configured_weight = batch.get("physppi_loss_weight")
    if configured_weight is None:
        return loss, loss_breakdown

    weight = torch.as_tensor(
        configured_weight,
        device=loss.device,
        dtype=loss.dtype,
    )
    if weight.numel() != 1:
        raise ValueError(
            "physppi_loss_weight must contain one scalar per rank-local sample"
        )
    weight = weight.reshape(())
    if not torch.isfinite(weight) or weight <= 0:
        raise ValueError("physppi_loss_weight must be finite and positive")

    weighted_breakdown = {
        name: value * weight.to(device=value.device, dtype=value.dtype)
        for name, value in loss_breakdown.items()
    }
    return loss * weight, weighted_breakdown


def _append_sampler_trace(
    path: Path, row: dict[str, int | float | str | bool | None]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite_mean(value) -> float | None:
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    scalar = value.detach().float().mean()
    if not torch.isfinite(scalar):
        return None
    return float(scalar.cpu())


def _single_bool(value) -> bool | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().bool().reshape(-1)
    if tensor.numel() != 1:
        return None
    return bool(tensor.item())


def _single_float(value) -> float | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().float().reshape(-1)
    if tensor.numel() != 1 or not torch.isfinite(tensor).all():
        return None
    return round(float(tensor.item()), 6)


def _config_value(config, name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    if hasattr(config, "get"):
        try:
            return config.get(name, default)
        except TypeError:
            pass
    return getattr(config, name, default)


def _safe_path_component(value: str) -> str:
    safe = [
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ]
    return "".join(safe) or "case"


def _copy_validation_gt_pdb(
    *,
    batch: dict,
    batch_index: int,
    case_dir: Path,
    safe_case_id: str,
) -> str | None:
    gt_paths = batch.get("physppi_gt_pdb_path")
    if gt_paths is None:
        return None
    if not isinstance(gt_paths, (list, tuple)):
        raise ValueError("physppi_gt_pdb_path must be a list when provided")
    if batch_index >= len(gt_paths):
        raise ValueError(
            "physppi_gt_pdb_path length is smaller than validation batch size"
        )
    source_value = gt_paths[batch_index]
    if source_value is None:
        return None
    source_path = Path(str(source_value))
    if not source_path.is_file():
        raise FileNotFoundError(
            f"validation GT PDB does not exist for {safe_case_id}: {source_path}"
        )
    file_name = f"{safe_case_id}_gt.pdb"
    shutil.copyfile(source_path, case_dir / file_name)
    return file_name


def _candidate_scalar(value, batch_index: int, sample_index: int) -> float | int | bool | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return None
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value)
        except (RuntimeError, TypeError, ValueError):
            return None
    tensor = value.detach()
    if tensor.ndim >= 2:
        if tensor.shape[0] <= batch_index or tensor.shape[1] <= sample_index:
            return None
        tensor = tensor[batch_index, sample_index]
    elif tensor.ndim == 1:
        if tensor.shape[0] <= sample_index:
            return None
        tensor = tensor[sample_index]
    if tensor.numel() != 1:
        return None
    if tensor.dtype == torch.bool:
        return bool(tensor.cpu().item())
    scalar = tensor.float().cpu().reshape(())
    if not torch.isfinite(scalar):
        return None
    return float(scalar.item())


def _scalar_summary(value) -> float | int | bool | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError):
            return None
    tensor = value.detach()
    if tensor.numel() == 0:
        return None
    if tensor.dtype == torch.bool and tensor.numel() == 1:
        return bool(tensor.cpu().item())
    tensor = tensor.float()
    finite = torch.isfinite(tensor)
    if not finite.any():
        return None
    return float(tensor[finite].mean().cpu().item())


def write_physppi_route_c_train_metrics(
    *,
    path: Path,
    batch: dict,
    output: dict,
    loss_breakdown: dict,
    rank: int,
    global_step: int,
    epoch: int,
) -> None:
    """Append one rank-local case and its weighted loss breakdown atomically."""

    sample_ids = batch.get("pdb_id")
    if not isinstance(sample_ids, (list, tuple)) or len(sample_ids) != 1:
        raise ValueError("PhysPPI loss logging requires one pdb_id per rank")
    sample_id = str(sample_ids[0])
    record: dict[str, int | float | str | bool | None] = {
        "rank": int(rank),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "sample_id": sample_id,
        "dataset": "afdb" if sample_id.startswith("AF-") else "qbiolip",
        "dataset_loss_weight": _single_float(batch.get("physppi_loss_weight")),
    }
    for loss_name, loss_value in loss_breakdown.items():
        metric_value = _finite_mean(loss_value)
        if metric_value is not None:
            record[f"train/{loss_name}"] = metric_value
        elif loss_name == "loss":
            record[f"train/{loss_name}"] = None
    if "train/loss" not in record:
        return

    gt_valid = _single_bool(batch.get("physppi_membrane_gt_valid"))
    prediction_valid = _single_bool(
        output.get("physppi_predicted_membrane_valid")
    )
    tm_mask = batch.get("physppi_tm_token_mask")
    valid_frame_count = output.get("physppi_membrane_valid_frame_count")
    if gt_valid is not None:
        record["membrane_gt_valid"] = gt_valid
    if prediction_valid is not None:
        record["membrane_prediction_valid"] = prediction_valid
    if torch.is_tensor(tm_mask):
        record["membrane_tm_token_count"] = int(
            tm_mask.detach().bool().sum().cpu().item()
        )
    if torch.is_tensor(valid_frame_count) and valid_frame_count.numel() == 1:
        record["membrane_valid_frame_count"] = int(
            valid_frame_count.detach().reshape(()).cpu().item()
        )

    pdockq_target_mean = batch.get("physppi_pdockq_diag_target_mean")
    pdockq_target_std = batch.get("physppi_pdockq_diag_target_std")
    pdockq_valid_count = batch.get("physppi_pdockq_diag_valid_count")
    pdockq_pred_mean = batch.get("physppi_pdockq_diag_pred_mean")
    if pdockq_target_mean is not None:
        record["pdockq_target_mean"] = float(pdockq_target_mean)
    if pdockq_target_std is not None:
        record["pdockq_target_std"] = float(pdockq_target_std)
    if pdockq_valid_count is not None:
        record["pdockq_valid_candidates"] = int(pdockq_valid_count)
    if pdockq_pred_mean is not None:
        record["pdockq_pred_mean"] = float(pdockq_pred_mean)

    _append_sampler_trace(path, record)


class OpenFold3AllAtom(ModelRunner):
    def __init__(self, model_config, log_dir: Path = None):
        validate_physppi_semantic_config(model_config)
        super().__init__(model_class=OpenFold3, config=model_config)

        self.log_dir = log_dir

        self.loss = OpenFold3Loss(config=model_config.architecture.loss_module)

        self.model_selection_weights = model_selection_metric_weights_config[
            self.config.settings.model_selection_weight_scheme
        ]

        # Settings for per-sample gradient clipping
        self.per_sample_grad_clipping = (
            model_config.settings.gradient_clipping.per_sample_clipping
        )
        self.grad_manager = None
        if self.per_sample_grad_clipping:
            self.grad_manager = PerSampleGradManager(
                gradient_clip_val=model_config.settings.gradient_clipping.clip_val,
                accumulate_grad_batches=model_config.settings.manual_optimization.accumulate_grad_batches,
                log_grad_norm=model_config.settings.debug.log_grad_norm,
            )
            self.automatic_optimization = False
            self.log_lr = model_config.settings.manual_optimization.log_lr

    @property
    def version(self):
        v = self.model.version_tensor.long().tolist()
        return f"{v[0]}.{v[1]}.{v[2]}"

    def setup(self, stage: str):
        # Setup metrics
        self._setup_train_metrics()
        self._setup_val_metrics()

        training_scope = self.config.settings.get("training_scope", "baseline")

        # Keep grads enabled for confidence head parameters only.
        if stage == "fit" and self.config.settings.train_confidence_only:
            exempt_submodule = [
                self.model.aux_heads.pairformer_embedding,
                self.model.aux_heads.plddt,
                self.model.aux_heads.pae,
                self.model.aux_heads.pdockq,
                self.model.aux_heads.pde,
            ]
            self._freeze_model_params(exempt_submodule=exempt_submodule)

        if stage == "fit" and training_scope == "physppi_lora_assessment":
            if self.config.settings.train_confidence_only:
                raise ValueError(
                    "settings.training_scope and train_confidence_only "
                    "cannot both be true"
                )
            self._freeze_model_params_for_physppi_and_lora()

        # Initialize the gradient manager if doing per-sample grad clipping
        if self.per_sample_grad_clipping:
            self.grad_manager.setup(
                model=self.model, trainer=self.trainer, logger=self.logger
            )
            self._identify_confidence_params()

    def _get_physppi_only_exempt_submodules(self) -> list[torch.nn.Module]:
        """Return the explicit PhysPPI/assessment trainable whitelist."""
        physppi_conditioning = getattr(self.model, "physppi_conditioning", None)
        if physppi_conditioning is None:
            raise ValueError(
                "settings.training_scope requires architecture.physppi.enabled=True"
            )
        modules = [physppi_conditioning]
        for name in ("membrane_plane_estimator", "interaction_distance_head"):
            module = getattr(self.model, name, None)
            if module is not None:
                modules.append(module)
        diffusion_bridge = getattr(
            self.model.diffusion_module, "extra_conditioner", None
        )
        if diffusion_bridge is not None:
            modules.append(diffusion_bridge)
        for name in ("pairformer_embedding", "plddt", "pae", "pdockq", "pde"):
            module = getattr(self.model.aux_heads, name, None)
            if module is not None:
                modules.append(module)
        return modules

    def _freeze_model_params(self, exempt_submodule: list[torch.nn.Module]):
        """Freeze all model parameters excluding those specified in exempt_submodule."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the exempt parameters
        for layer in exempt_submodule:
            for param in layer.parameters():
                param.requires_grad = True

    def _freeze_model_params_for_physppi_and_lora(self):
        """Freeze OF3, then unfreeze PhysPPI/assessment modules and LoRA A/B."""
        if getattr(self.config.architecture, "lora", None) is None:
            raise ValueError("physppi_lora_assessment requires architecture.lora")
        if not self.config.architecture.lora.enabled:
            raise ValueError(
                "physppi_lora_assessment requires architecture.lora.enabled"
            )

        for param in self.model.parameters():
            param.requires_grad = False

        for module in self._get_physppi_only_exempt_submodules():
            for param in module.parameters():
                param.requires_grad = True

        lora_param_count = 0
        for name, param in self.model.named_parameters():
            if is_lora_parameter_name(name):
                param.requires_grad = True
                lora_param_count += param.numel()

        if lora_param_count == 0:
            raise ValueError("physppi_lora_assessment found zero LoRA parameters")

        unexpected = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and not (
                name.startswith(PHYSPPI_TRAINABLE_PREFIXES)
                or is_lora_parameter_name(name)
            )
        ]
        if unexpected:
            raise ValueError(
                "unexpected trainable OpenFold3 parameters: "
                + ", ".join(unexpected[:10])
            )

    def _route_c_metric_value(self, value) -> float | None:
        return _finite_mean(value)

    def _write_physppi_route_c_train_metrics(
        self,
        batch,
        output,
        loss_breakdown,
    ) -> None:
        if (
            self.config.settings.get("training_scope", "baseline")
            != "physppi_lora_assessment"
        ):
            return
        if self.log_dir is None or self.trainer is None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        write_physppi_route_c_train_metrics(
            path=self.log_dir / "physppi_route_c_train_metrics.jsonl",
            batch=batch,
            output=output,
            loss_breakdown=loss_breakdown,
            rank=self.global_rank,
            global_step=self.trainer.global_step,
            epoch=self.trainer.current_epoch,
        )

    def _identify_confidence_params(self):
        """
        Identifies which parameters belong to confidence heads.
        """
        confidence_modules_prefixes = [
            "aux_heads.pairformer_embedding",
            "aux_heads.plddt",
            "aux_heads.pae",
            "aux_heads.pdockq",
            "aux_heads.pde",
        ]

        self.confidence_param_names = set()

        for name, _ in self.model.named_parameters():
            # Check if this param belongs to confidence module
            is_confidence = any(
                name.startswith(f"{prefix}.") for prefix in confidence_modules_prefixes
            )
            if is_confidence:
                self.confidence_param_names.add(name)

    def reseed(self, seed):
        pl.seed_everything(seed)

    def _setup_train_metrics(self):
        """Set up training loss and metric collection objects."""

        # TODO: Forcing naming convention to be compatible with older runs
        #  Make consistent later
        # Initialize all training epoch metric objects
        train_losses = {
            loss_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for loss_name in TRAIN_LOSSES
        }
        self.train_losses = MetricCollection(
            train_losses, prefix="train/", postfix="_epoch"
        )

        train_metrics = {
            metric_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for metric_name in TRAIN_LOGGED_METRICS
        }

        self.train_metrics = MetricCollection(train_metrics, prefix="train/")

    def _setup_val_metrics(self):
        """Set up validation loss and metric collection objects."""

        # Initialize all validation epoch metric objects
        val_losses = {
            loss_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for loss_name in VAL_LOSSES
        }
        self.val_losses = MetricCollection(val_losses, prefix="val/")

        val_metrics = {
            metric_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for metric_name in VAL_LOGGED_METRICS
        }
        val_metrics.update(
            {
                metric_name: PearsonCorrCoef(num_outputs=1, sync_on_compute=False)
                for metric_name in CORRELATION_METRICS
            }
        )
        self.val_metrics = MetricCollection(val_metrics, prefix="val/")

    def _update_epoch_metric(
        self,
        phase: str,
        metric_log_name: str,
        metric_value: [torch.Tensor, tuple],
        metric_collection: MetricCollection,
    ):
        """Update metrics for the epoch logging.

        Args:
            phase:
                Phase of training, accepts "train" or "val"
            metric_log_name:
                Name of the metric in the log, including prefix or postfix
            metric_value:
                Value of the metric to update
            metric_collection:
                MetricCollection object containing the metric to update
        """

        if metric_log_name not in metric_collection.keys():  # noqa: SIM118
            raise ValueError(
                f"Metric {metric_log_name} is not being tracked and will "
                f"not appear in epoch metrics. Please add it to "
                f"the {phase.upper()}_LOSSES or METRICS constants."
            )

        metric_obj = metric_collection[metric_log_name]
        metric_value = (
            (metric_value,) if type(metric_value) is not tuple else metric_value
        )

        metric_obj.update(*metric_value)

    def _get_validation_metrics_detail(self, batch, outputs) -> tuple[dict, dict]:
        num_samples = self.config.architecture.shared.diffusion.no_full_rollout_samples
        num_atoms = outputs["atom_positions_predicted"].shape[-2]
        chunk_metrics_computation = (
            num_samples > 1
            and self.config.settings.memory.eval.per_sample_atom_cutoff is not None
            and num_atoms > self.config.settings.memory.eval.per_sample_atom_cutoff
        )

        if chunk_metrics_computation:
            metrics_per_sample = get_metrics_chunked(
                batch,
                outputs,
                compute_extra_val_metrics=True,
            )
        else:
            metrics_per_sample = get_metrics(
                batch,
                outputs,
                compute_extra_val_metrics=True,
            )

        metrics = compute_valid_model_selection_metrics(
            confidence_config=self.config.confidence,
            outputs=outputs,
            metrics=metrics_per_sample,
        )

        for metric_name in CORRELATION_METRICS:
            molecule_type = metric_name.split("_")[-1]
            plddt_key = f"plddt_{molecule_type}"
            lddt_key = f"lddt_intra_{molecule_type}"

            plddt = metrics_per_sample.get(plddt_key)
            lddt = metrics_per_sample.get(lddt_key)

            if plddt is not None and lddt is not None:
                plddt = plddt.reshape((-1, 1))
                lddt = lddt.reshape((-1, 1))
                metrics[metric_name] = (lddt, plddt)

        return metrics, metrics_per_sample

    def _get_metrics(self, batch, outputs, train=True) -> dict:
        with torch.no_grad():
            if train:
                return get_metrics(
                    batch,
                    outputs,
                    compute_lig_diffusion_metrics=True,
                    compute_extra_val_metrics=False,
                )

            metrics, _ = self._get_validation_metrics_detail(batch, outputs)
            return metrics

    def _validation_artifact_config(self):
        return _config_value(self.config.settings, "validation_artifacts", {})

    def _validation_artifacts_enabled(self) -> bool:
        return bool(_config_value(self._validation_artifact_config(), "enabled", False))

    def _synchronize_validation_timing(self) -> None:
        artifact_config = self._validation_artifact_config()
        should_sync = bool(
            _config_value(artifact_config, "synchronize_cuda_timing", False)
        )
        if should_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _compute_validation_artifact_confidence_scores(
        self,
        batch: dict,
        outputs: dict,
    ) -> dict[str, torch.Tensor]:
        confidence_scores: dict[str, torch.Tensor] = {}
        if "plddt_logits" in outputs:
            confidence_scores["plddt"] = compute_plddt(outputs["plddt_logits"]) * 100.0

        pdockq_logits = outputs.get("pdockq_logits")
        if pdockq_logits is not None:
            if "pdockq_expectation" in outputs:
                confidence_scores["pdockq"] = outputs["pdockq_expectation"]
            elif pdockq_logits.shape[-1] == 1:
                confidence_scores["pdockq"] = pdockq_logits.squeeze(-1)
            else:
                probs = torch.softmax(pdockq_logits, dim=-1)
                bins = torch.linspace(
                    0.0,
                    1.0,
                    probs.shape[-1],
                    device=probs.device,
                    dtype=probs.dtype,
                )
                confidence_scores["pdockq"] = torch.sum(probs * bins, dim=-1)

        return confidence_scores

    def _write_validation_artifacts(
        self,
        *,
        batch: dict,
        outputs: dict,
        metrics_per_sample: dict,
        confidence_scores: dict,
        loss_breakdown: dict,
        batch_idx: int,
        dataloader_idx: int,
        timing: dict[str, float] | None = None,
    ) -> None:
        artifact_config = self._validation_artifact_config()
        if not bool(_config_value(artifact_config, "enabled", False)):
            return

        start = time.perf_counter()
        output_dir = _config_value(artifact_config, "output_dir", None)
        if output_dir is None:
            if self.log_dir is None:
                raise RuntimeError("validation artifact export requires log_dir")
            output_root = Path(self.log_dir).parent / "validation_artifacts"
        else:
            output_root = Path(output_dir)

        structure_format = str(
            _config_value(artifact_config, "structure_format", "pdb")
        )
        positions = outputs["atom_positions_predicted"]
        batch_size = int(positions.shape[0])
        num_candidates = int(positions.shape[1])
        expected_samples = _config_value(artifact_config, "expected_samples", None)
        if expected_samples is not None and num_candidates != int(expected_samples):
            message = (
                "validation artifact sample count mismatch: "
                f"expected {int(expected_samples)}, got {num_candidates}"
            )
            if bool(
                _config_value(
                    artifact_config,
                    "fail_on_sample_count_mismatch",
                    True,
                )
            ):
                raise ValueError(message)
            logger.warning(message)

        sample_ids = batch.get("pdb_id") or batch.get("query_id")
        if not isinstance(sample_ids, (list, tuple)):
            raise ValueError("validation artifact export requires pdb_id/query_id list")
        atom_arrays = batch.get("atom_array")
        if not isinstance(atom_arrays, (list, tuple)):
            raise ValueError("validation artifact export requires atom_array list")

        for batch_index in range(batch_size):
            case_id = str(sample_ids[batch_index])
            safe_case_id = _safe_path_component(case_id)
            case_dir = output_root / f"dataloader_{dataloader_idx}" / safe_case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            gt_structure_file = _copy_validation_gt_pdb(
                batch=batch,
                batch_index=batch_index,
                case_dir=case_dir,
                safe_case_id=safe_case_id,
            )

            atom_array = atom_arrays[batch_index]
            candidate_rows: list[dict[str, object]] = []
            for sample_index in range(num_candidates):
                file_name = (
                    f"{safe_case_id}_candidate_{sample_index + 1:02d}."
                    f"{structure_format}"
                )
                output_file = case_dir / file_name
                predicted_coords = (
                    positions[batch_index, sample_index]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                plddt = confidence_scores.get("plddt")
                if plddt is None:
                    plddt_np = np.zeros(predicted_coords.shape[0], dtype=np.float32)
                else:
                    plddt_np = (
                        plddt[batch_index, sample_index]
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )

                OF3OutputWriter.write_structure_prediction(
                    atom_array=atom_array.copy(),
                    predicted_coords=predicted_coords,
                    plddt=plddt_np,
                    output_file=output_file,
                )

                row: dict[str, object] = {
                    "case_id": case_id,
                    "dataloader_idx": int(dataloader_idx),
                    "batch_idx": int(batch_idx),
                    "rank": int(self.global_rank),
                    "candidate_index": int(sample_index + 1),
                    "structure_file": file_name,
                }
                for name, value in metrics_per_sample.items():
                    scalar = _candidate_scalar(value, batch_index, sample_index)
                    if scalar is not None:
                        row[name] = scalar
                for name, value in confidence_scores.items():
                    scalar = _candidate_scalar(value, batch_index, sample_index)
                    if scalar is not None:
                        row[f"confidence/{name}"] = scalar
                target_score = _candidate_scalar(
                    batch.get("physppi_pdockq_target_score"),
                    batch_index,
                    sample_index,
                )
                target_valid = _candidate_scalar(
                    batch.get("physppi_pdockq_valid"),
                    batch_index,
                    sample_index,
                )
                if target_score is not None:
                    row["target/pdockq"] = target_score
                if target_valid is not None:
                    row["target/pdockq_valid"] = target_valid
                candidate_rows.append(row)

            base_fields = [
                "case_id",
                "dataloader_idx",
                "batch_idx",
                "rank",
                "candidate_index",
                "structure_file",
            ]
            dynamic_fields = sorted(
                {
                    key
                    for row in candidate_rows
                    for key in row.keys()
                    if key not in base_fields
                }
            )
            with (case_dir / "candidate_metrics.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=[*base_fields, *dynamic_fields]
                )
                writer.writeheader()
                writer.writerows(candidate_rows)

            case_timing = dict(timing or {})
            case_timing["artifact_write_seconds"] = time.perf_counter() - start
            summary = {
                "case_id": case_id,
                "dataloader_idx": int(dataloader_idx),
                "batch_idx": int(batch_idx),
                "rank": int(self.global_rank),
                "global_step": int(self.global_step),
                "num_candidates": num_candidates,
                "num_atoms": int(positions.shape[-2]),
                "loss_breakdown": {
                    name: _scalar_summary(value)
                    for name, value in loss_breakdown.items()
                },
                "timing": case_timing,
            }
            if gt_structure_file is not None:
                summary["gt_structure_file"] = gt_structure_file
            (case_dir / "case_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )

            if bool(_config_value(artifact_config, "profile_timing", False)):
                timing_path = (
                    Path(self.log_dir)
                    / "evidence"
                    / "validation_artifacts"
                    / f"rank-{self.global_rank}.jsonl"
                )
                _append_sampler_trace(
                    timing_path,
                    {
                        "case_id": case_id,
                        "dataloader_idx": int(dataloader_idx),
                        "batch_idx": int(batch_idx),
                        "rank": int(self.global_rank),
                        "global_step": int(self.global_step),
                        "num_candidates": num_candidates,
                        "num_atoms": int(positions.shape[-2]),
                        **case_timing,
                    },
                )

    def _log(
        self,
        loss_breakdown,
        batch,
        outputs,
        train=True,
        log_train_step_metrics=True,
        metrics_override=None,
    ):
        phase = "train" if train else "val"

        metrics = (
            metrics_override
            if metrics_override is not None
            else self._get_metrics(batch, outputs, train=train)
        )

        loss_collection = self.train_losses if phase == "train" else self.val_losses
        for loss_name, indiv_loss in loss_breakdown.items():
            metric_log_name = f"{phase}/{loss_name}"
            metric_epoch_name = f"{metric_log_name}_epoch" if train else metric_log_name

            # Update mean metrics for epoch logging
            self._update_epoch_metric(
                phase=phase,
                metric_log_name=metric_epoch_name,
                metric_value=indiv_loss,
                metric_collection=loss_collection,
            )

            # Only log steps for training
            if train and log_train_step_metrics:
                self.log(
                    metric_log_name,
                    indiv_loss,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

        metric_collection = self.train_metrics if phase == "train" else self.val_metrics
        for metric_name, metric_value in metrics.items():
            metric_log_name = f"{phase}/{metric_name}"

            # Update mean metrics for epoch logging
            self._update_epoch_metric(
                phase=phase,
                metric_log_name=metric_log_name,
                metric_value=metric_value,
                metric_collection=metric_collection,
            )

            # Only log steps for training
            if train and log_train_step_metrics:
                self.log(
                    f"{metric_log_name}_step",
                    metric_value,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

        if train:
            self._write_physppi_route_c_train_metrics(batch, outputs, loss_breakdown)

    def _is_opt_step_ready(self, batch_idx: int) -> bool:
        """
        Checks if the optimizer step should be performed.
        Used in manual mode.
        """
        if self.per_sample_grad_clipping:
            accum_steps = self.grad_manager.accumulate_grad_batches
        else:
            accum_steps = self.trainer.accumulate_grad_batches

        is_last_step_of_cycle = (batch_idx + 1) % accum_steps == 0
        return is_last_step_of_cycle or self.trainer.is_last_batch

    def _get_sample_disabled_param_names(self, loss_weights: dict) -> set | None:
        """
        Returns a list of confidence head parameters that should be disabled
        when counting grads across ranks, else None.
        """
        confidence_loss_name = (
            self.config.architecture.loss_module.confidence_loss_names
        )

        total_conf_weight = sum(
            loss_weights[name].item() for name in confidence_loss_name
        )

        is_valid_confidence_sample = total_conf_weight > 0

        # Confidence losses valid are not valid for distillation samples or
        # samples with resolution out-of-bounds
        if not is_valid_confidence_sample:
            # Return the pre-computed list of confidence param names
            return self.confidence_param_names

        # If no params are disabled, return None
        return None

    def _training_step_manual_clip(self, batch, batch_idx):
        assert len(batch["pdb_id"]) == 1, (
            "Currently only local batch size of 1 per GPU is supported."
        )

        if self.trainer.world_size > 1:
            assert isinstance(self.trainer.strategy, DDPStrategy), (
                "Per-sample gradient clipping is only supported with DDPStrategy."
            )

        example_feat = batch["token_mask"]
        if self.ema.device != example_feat.device:
            self.ema.to(example_feat.device)

        if self.grad_manager.device != example_feat.device:
            self.grad_manager.to(example_feat.device)

        pdb_id = ", ".join(batch["pdb_id"])
        preferred_chain_or_interface = batch["preferred_chain_or_interface"]
        logging_info = {
            "pdb_id": pdb_id,
            "preferred_chain_or_interface": preferred_chain_or_interface,
        }

        logger.debug(
            f"Started model forward pass for {pdb_id} with preferred chain or "
            f"interface {preferred_chain_or_interface} on rank {self.global_rank} "
            f"step {self.global_step}"
        )

        opt = self.optimizers()

        # zero_grad() must be called on every micro-batch to ensure
        # self.manual_backward() sets p.grad instead of adding to it
        # when doing gradient accumulation.
        # It's probably overkill to handle the clipping this exactly
        # instead of just using the averaged microbatch, but I'll revisit
        # that later if needed.
        opt.zero_grad()

        try:
            # Only required when running in distributed mode
            sync_context = (
                self.trainer.model.no_sync()
                if self.trainer.world_size > 1
                else nullcontext()
            )

            # When using DDP, this disables the automatic sync that would happen on
            # manual_backward and break the per-sample grad clipping
            with sync_context:
                # Run the model
                batch, outputs = self.model(batch)
                self._populate_physppi_pdockq_targets(batch, outputs)

                # Compute loss
                loss, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)
                loss, loss_breakdown = apply_physppi_loss_weight(
                    batch,
                    loss,
                    loss_breakdown,
                )

                self.manual_backward(loss)

                disabled_params = self._get_sample_disabled_param_names(
                    loss_weights=batch["loss_weights"]
                )
                self.grad_manager.clip_and_accumulate(
                    logging_info=logging_info, disabled_params=disabled_params
                )

            if self._is_opt_step_ready(batch_idx):
                # Average and sync grads
                self.grad_manager.sync_and_average_grads()

                self.grad_manager.log_average_grad_norm()

                opt.step()
                self.lr_schedulers().step()

                # Zero the grad accumulator
                self.grad_manager.reset_accumulator()

                # Log LR and step metrics only after the optimizer step
                # to mimic logging behavior when using automatic optimization
                if self.log_lr:
                    self.log(
                        "AlphaFoldLRScheduler",
                        opt.param_groups[0]["lr"],
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        sync_dist=False,
                    )

                self._log(
                    loss_breakdown,
                    batch,
                    outputs,
                    train=True,
                    log_train_step_metrics=True,
                )

                # Workaround for PL step logging issues. Avoids using
                # `self.trainer.fit_loop.epoch_loop._batches_that_stepped` if this
                # metric exists.
                self.log(
                    "step",
                    self.global_step,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

            else:
                # Always update epoch metrics
                self._log(
                    loss_breakdown,
                    batch,
                    outputs,
                    train=True,
                    log_train_step_metrics=False,
                )

        except Exception:
            logger.exception(
                f"Train step failed with pdb id {pdb_id} with "
                f"preferred chain or interface {preferred_chain_or_interface}"
            )

            # Clear grad accumulator on error
            # Only really necessary if trainer is not reinitialized after exception
            self.grad_manager.reset_accumulator()

            raise

        return loss

    def _training_step(self, batch):
        example_feat = batch["token_mask"]
        if self.ema.device != example_feat.device:
            self.ema.to(example_feat.device)

        pdb_id = ", ".join(batch["pdb_id"])
        preferred_chain_or_interface = batch["preferred_chain_or_interface"]
        logger.debug(
            f"Started model forward pass for {pdb_id} with preferred chain or "
            f"interface {preferred_chain_or_interface} on rank {self.global_rank} "
            f"step {self.global_step}"
        )

        try:
            # Run the model
            batch, outputs = self.model(batch)
            self._populate_physppi_pdockq_targets(batch, outputs)

            # Compute loss
            loss, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)
            loss, loss_breakdown = apply_physppi_loss_weight(
                batch,
                loss,
                loss_breakdown,
            )

            self._log(loss_breakdown, batch, outputs)

        except Exception:
            logger.exception(
                f"Train step failed with pdb id {pdb_id} with "
                f"preferred chain or interface {preferred_chain_or_interface}"
            )
            raise

        return loss

    def _populate_physppi_pdockq_targets(
        self,
        batch: dict,
        outputs: dict,
    ) -> None:
        """Build label-only candidate DockQ targets after model forward."""

        if (
            getattr(self.config.settings, "training_scope", "baseline")
            != PHYSPPI_TRAINING_SCOPE
        ):
            return
        prediction_key = (
            "atom_positions_diffusion"
            if "atom_positions_diffusion" in outputs
            else "atom_positions_predicted"
        )
        if prediction_key not in outputs:
            raise KeyError(
                "atom_positions_diffusion or atom_positions_predicted is required "
                "for PhysPPI pDockQ supervision"
            )
        if "pdockq_logits" not in outputs:
            raise KeyError("pdockq_logits is required for PhysPPI pDockQ supervision")
        ground_truth = batch.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise KeyError("ground_truth is required for PhysPPI pDockQ supervision")
        if "atom_positions" not in ground_truth:
            raise KeyError("atom_positions is required for PhysPPI pDockQ supervision")
        prediction = outputs[prediction_key]
        logits = outputs["pdockq_logits"]
        num_bins = logits.shape[-1]
        if logits.shape[:-1] != prediction.shape[:2]:
            raise ValueError(
                "pDockQ logits batch/candidate dims must match coordinates, "
                f"got {tuple(logits.shape)} vs {tuple(prediction.shape)}"
            )
        if num_bins not in (1, 50):
            raise ValueError(
                f"pDockQ logits last dim must be 1 (regression) or 50 "
                f"(classification), got {num_bins}"
            )
        with torch.no_grad():
            scores, valid = per_candidate_complex_dockq(
                prediction=prediction,
                gt=ground_truth["atom_positions"],
                batch=batch,
            )
            if num_bins == 50:
                target = dockq_to_soft_bins(scores, no_bins=50)
                target = target * valid.unsqueeze(-1).to(dtype=target.dtype)
            else:
                target = scores.unsqueeze(-1)
        batch["physppi_pdockq_target_score"] = scores
        batch["physppi_pdockq_valid"] = valid
        batch["physppi_pdockq_target"] = target
        batch["physppi_pdockq_diag_target_mean"] = (
            scores[valid].mean().item() if valid.any() else float("nan")
        )
        batch["physppi_pdockq_diag_target_std"] = (
            scores[valid].std().item() if valid.any() else float("nan")
        )
        batch["physppi_pdockq_diag_valid_count"] = int(valid.sum().item())
        expectation = outputs.get("pdockq_expectation")
        if expectation is not None and valid.any():
            batch["physppi_pdockq_diag_pred_mean"] = (
                expectation[valid].mean().item()
            )

    def training_step(self, batch, batch_idx):
        if (
            getattr(self.config.settings, "training_scope", "baseline")
            == PHYSPPI_TRAINING_SCOPE
        ):
            if self.log_dir is None:
                raise RuntimeError("PhysPPI semantic training requires a log directory")
            row = sampler_trace_row(
                batch,
                rank=self.global_rank,
                epoch=self.trainer.current_epoch,
                micro_step=batch_idx,
                optimizer_step=(
                    self.trainer.global_step if self.global_rank == 0 else None
                ),
            )
            _append_sampler_trace(
                Path(self.log_dir)
                / "evidence"
                / "sampler"
                / f"rank-{self.global_rank}.jsonl",
                row,
            )
        if self.per_sample_grad_clipping:
            return self._training_step_manual_clip(batch=batch, batch_idx=batch_idx)

        return self._training_step(batch=batch)

    def eval_step(self, batch, batch_idx, dataloader_idx=0):
        pdb_id = batch["pdb_id"]
        is_repeated_sample = batch.get("repeated_sample")
        if is_repeated_sample:
            logger.debug(
                f"Skipping repeated sample {', '.join(pdb_id)} on rank "
                f"{self.global_rank}"
            )
            return

        logger.debug(
            f"Started validation for {', '.join(pdb_id)} on rank {self.global_rank} "
            f"step {self.global_step}"
        )

        try:
            timing: dict[str, float] = {}
            profile_timing = self._validation_artifacts_enabled()

            # Run the model
            self._synchronize_validation_timing()
            section_start = time.perf_counter()
            batch, outputs = self(batch)
            self._synchronize_validation_timing()
            if profile_timing:
                timing["forward_seconds"] = time.perf_counter() - section_start

            self._synchronize_validation_timing()
            section_start = time.perf_counter()
            self._populate_physppi_pdockq_targets(batch, outputs)
            self._synchronize_validation_timing()
            if profile_timing:
                timing["pdockq_target_seconds"] = time.perf_counter() - section_start

            # Compute loss and other metrics
            self._synchronize_validation_timing()
            section_start = time.perf_counter()
            _, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)
            self._synchronize_validation_timing()
            if profile_timing:
                timing["loss_seconds"] = time.perf_counter() - section_start

            self._synchronize_validation_timing()
            section_start = time.perf_counter()
            metrics, metrics_per_sample = self._get_validation_metrics_detail(
                batch, outputs
            )
            self._synchronize_validation_timing()
            if profile_timing:
                timing["metrics_seconds"] = time.perf_counter() - section_start

            self._log(
                loss_breakdown,
                batch,
                outputs,
                train=False,
                metrics_override=metrics,
            )

            if self._validation_artifacts_enabled():
                self._synchronize_validation_timing()
                section_start = time.perf_counter()
                confidence_scores = (
                    self._compute_validation_artifact_confidence_scores(
                        batch,
                        outputs,
                    )
                )
                self._synchronize_validation_timing()
                timing["confidence_seconds"] = time.perf_counter() - section_start
                self._write_validation_artifacts(
                    batch=batch,
                    outputs=outputs,
                    metrics_per_sample=metrics_per_sample,
                    confidence_scores=confidence_scores,
                    loss_breakdown=loss_breakdown,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                    timing=timing,
                )

        except Exception:
            logger.exception(f"Validation step failed with pdb id {', '.join(pdb_id)}")
            raise

    def on_validation_epoch_start(self):
        # At the start of validation, load the EMA weights
        if not bool(
            _config_value(self.config.settings, "validation_use_ema_weights", True)
        ):
            return

        assert self.cached_weights is None

        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        self.cached_weights = tensor_tree_map(
            lambda t: t.detach().clone(), self.model.state_dict()
        )

        self.model.load_state_dict(self.ema.state_dict()["params"], strict=False)

    def on_before_optimizer_step(self, *args, **kwargs):
        """
        Logs unclipped grad norm and gradients for the single-transition
        linear_out layers. This logging can be enabled in config.settings.debug.

        These gradients can be associated with instabilities, so we're logging them on
        every single step (bypassing log_every_n_steps) for more accurate monitoring.
        """
        debug_settings = self.config.settings.debug

        # Transition layers included in this logging are frozen when
        # training confidence only
        should_log_extra_metrics = (
            False
            if self.config.settings.train_confidence_only
            else debug_settings.log_extra_grad_metrics
        )
        should_log_grad_norm = debug_settings.log_grad_norm

        if not should_log_extra_metrics and not should_log_grad_norm:
            return

        extra_grad_metrics = {}

        # Only rank zero will actually log the gradients
        log_grad_metrics = self.trainer.is_global_zero and self.logger is not None

        # Only log 4 representative blocks to reduce overhead
        block_idxs = [0, 16, 32, 47]

        # To see if this slows down training, we additionally log runtimes from the
        # global_zero process
        # TODO: Set this to log-level INFO and configure per-module log-levels in a more
        # principled way
        timing_context = partial(PerformanceTimer, logger=logger, level=logging.WARNING)
        log_timing = log_grad_metrics and debug_settings.profile_grad_logging

        context = (
            timing_context("Extra-gradient fetching and calculation")
            if log_timing
            else nullcontext()
        )

        with context:
            if should_log_extra_metrics:
                for idx in block_idxs:
                    block = self.model.pairformer_stack.blocks[idx]
                    param = block.single_transition.linear_out.weight

                    if isinstance(self.trainer.strategy, DeepSpeedStrategy):
                        # Needs to be called on every rank to avoid hanging
                        # https://github.com/deepspeedai/DeepSpeed/issues/7117#issuecomment-2717974187
                        grad = deepspeed.utils.safe_get_full_grad(param)
                    else:
                        grad = param.grad

                    if grad is None:
                        continue

                    assert not grad.requires_grad

                    if log_grad_metrics:
                        tag = (
                            f"extra_gradients/model.pairformer_stack.blocks.{idx}."
                            "single_transition.linear_out.weight"
                        )

                        extra_grad_metrics[f"{tag}_norm"] = grad.norm().item()
                        extra_grad_metrics[f"{tag}_max"] = grad.abs().max().item()

            if not self.per_sample_grad_clipping and should_log_grad_norm:
                # Compute global grad norm for per-batch grad clipping
                # Per sample clipping handles this logging in the grad manager
                global_norm, _ = compute_global_norm(parameters=self.model.parameters())
                extra_grad_metrics["extra_gradients/avg_unclipped_grad_norm"] = (
                    global_norm.item()
                )

        if log_grad_metrics:
            context = (
                timing_context("Extra-gradient logging")
                if log_timing
                else nullcontext()
            )
            with context:
                # NOTE: This out-of-schedule logging might interact a bit weirdly with
                # the WandB Step, so always plot against trainer/global_step
                self.logger.log_metrics(extra_grad_metrics, step=self.global_step)

    def _log_epoch_metrics(
        self, metrics: MetricCollection, compute_model_selection: bool = False
    ):
        """Log aggregated epoch metrics for training or validation.

        Args:
            metrics: MetricCollection object containing the metrics to log
        """
        if not self.trainer.sanity_checking:
            # Sync and reduce metrics across ranks
            # Done separately from compute() to get the sample counts
            # so that only enabled metrics are logged
            for metric in metrics.values():
                metric.sync()
                metric._should_unsync = False

            metrics_output = metrics.compute()

            # Only log metrics that have been updated
            enabled_metrics = {}
            for name, result in metrics_output.items():
                metric_obj = metrics[name]
                metric_type = type(metric_obj)

                # Get the sample count attribute name (e.g., 'weight')
                attr_name = METRIC_DENOMINATOR_ATTRS.get(metric_type)

                if attr_name is None:
                    raise NotImplementedError(
                        f"Failed to get sample count for metric type "
                        f"'{metric_type.__name__}'. Please add this metric "
                        f"to the METRIC_DENOMINATOR_ATTRS constant."
                    )

                n_samples = getattr(metric_obj, attr_name).sum().item()
                if n_samples > 0:
                    enabled_metrics[name] = result

            if self.per_sample_grad_clipping and self.logger is not None:
                self.logger.log_metrics(enabled_metrics, step=self.global_step)
            else:
                for name, result in enabled_metrics.items():
                    self.log(
                        name,
                        result,
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        sync_dist=False,  # Already synced
                    )

            if compute_model_selection:
                model_selection = compute_final_model_selection_metric(
                    metrics=metrics_output,
                    model_selection_weights=self.model_selection_weights,
                )

                if self.per_sample_grad_clipping and self.logger is not None:
                    self.logger.log_metrics(
                        {"val/model_selection": model_selection}, step=self.global_step
                    )
                else:
                    self.log(
                        "val/model_selection",
                        model_selection,
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        sync_dist=False,
                    )

        # Reset metrics for next epoch
        metrics.reset()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after optimizer.step(). Gradients are present and clipped."""

        # Skip grad accumulation steps
        if not self._is_opt_step_ready(batch_idx):
            return

        # EMA weight update
        self.ema.update(self.model)

        # Log the clipped step norm when not using per-sample gradient clipping
        # In order to match the logging step of per-sample grad clipping,
        # the step is shifted by 1
        should_log_grad_norm = (
            self.config.settings.debug.log_grad_norm and self.logger is not None
        )
        if (
            not self.per_sample_grad_clipping
            and should_log_grad_norm
            and self.trainer.global_step > 0
        ):
            global_norm, _ = compute_global_norm(parameters=self.model.parameters())
            self.logger.log_metrics(
                {"extra_gradients/avg_clipped_grad_norm": global_norm.item()},
                step=self.global_step - 1,
            )

    def _get_train_sampler(self):
        dl = self.trainer.train_dataloader
        # If PL uses multiple loaders, dl can be CombinedLoader etc.
        # Handle the simple/common case first:
        sampler = getattr(dl, "sampler", None)

        # If it's a BatchSampler, the underlying sampler is sampler.sampler
        if (
            sampler is not None
            and hasattr(sampler, "sampler")
            and (hasattr(sampler, "batch_size") and hasattr(sampler, "drop_last"))
        ):
            # only unwrap if it looks like a BatchSampler wrapper
            # (BatchSampler has .sampler and .batch_size, .drop_last)
            sampler = sampler.sampler
        return sampler

    def on_train_epoch_start(self):
        sampler = self._get_train_sampler()

        logger.info(
            f"Rank - {self.global_rank} starting epoch {self.trainer.current_epoch} "
            f"sampler epoch: {sampler.epoch} "
            f"global_step={self.trainer.global_step} "
            f"next_dataset_indices={self.trainer.datamodule.next_dataset_indices}"
        )

    def on_train_epoch_end(self):
        """Log aggregated epoch metrics for training."""
        self._log_epoch_metrics(metrics=self.train_losses)
        self._log_epoch_metrics(metrics=self.train_metrics)
        sampler = self._get_train_sampler()
        logger.info(
            f"Rank - {self.global_rank} finished epoch {self.trainer.current_epoch} "
            f"sampler epoch: {sampler.epoch} "
            f"global_step={self.trainer.global_step} "
            f"next_dataset_indices={self.trainer.datamodule.next_dataset_indices}"
        )

    def on_validation_epoch_end(self):
        """Log aggregated epoch metrics for validation."""
        self._log_epoch_metrics(metrics=self.val_losses)
        self._log_epoch_metrics(metrics=self.val_metrics, compute_model_selection=True)

        # Restore the model weights to normal
        if self.cached_weights is not None:
            self.model.load_state_dict(self.cached_weights)
            self.cached_weights = None

        # Temp fix for val dataloader worker seg fault issues
        # TODO: Figure out why this is not being cleaned up properly
        gc.collect()
        torch.cuda.empty_cache()
        self.trainer.strategy.barrier()

    def configure_optimizers(self) -> dict:
        optimizer_config = self.config.settings.optimizer
        parameters = [param for param in self.model.parameters() if param.requires_grad]

        optimizer = torch.optim.Adam(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
        )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = optimizer_config.learning_rate

        lr_sched_config = self.config.settings.lr_scheduler
        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=self.last_lr_step,
            base_lr=lr_sched_config.base_lr,
            max_lr=optimizer_config.learning_rate,
            warmup_no_steps=lr_sched_config.warmup_no_steps,
            start_decay_after_n_steps=lr_sched_config.start_decay_after_n_steps,
            decay_factor=lr_sched_config.decay_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            },
        }

    def _audit_physppi_checkpoint(self, params: dict) -> None:
        """Audit active PhysPPI checkpoint keys before loading the EMA.

        LoRA-only migration keys may be absent from an older OF3 checkpoint,
        but the active PDE head is part of the canonical assessment contract
        and may not silently disappear from a PhysPPI checkpoint.
        """
        if (
            self.config.settings.get("training_scope", "baseline")
            != "physppi_lora_assessment"
        ):
            return

        expected = set(self.model.state_dict())
        checkpoint_keys = set(params)
        missing = sorted(expected - checkpoint_keys)
        pde_missing = [key for key in missing if key.startswith("aux_heads.pde.")]
        if pde_missing:
            raise ValueError(
                "PhysPPI checkpoint is missing active PDE parameter(s): "
                + ", ".join(pde_missing[:10])
            )

        unexpected = sorted(checkpoint_keys - expected)
        lora_config = getattr(self.config.architecture, "lora", None)
        validate_lora_checkpoint_keys(
            missing,
            unexpected,
            lora_enabled=bool(lora_config is not None and lora_config.enabled),
        )

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        self._audit_physppi_checkpoint(ema["params"])
        self.ema.load_state_dict(ema)

    def _compute_confidence_scores(self, batch: dict, outputs: dict) -> dict:
        """Compute confidence metrics. This function is called during inference.

        Args:
            batch (dict):
                Input feature dictionary
            outputs (dict:
                Output dictionary containing the predicted trunk embeddings,
                all-atom positions, and distogram head logits

        Returns:
            confidence_scores (dict):
                Dict containing the following confidence measures:
                pLDDT, PDE, PAE, pTM, iPTM, weighted pTM
        """
        num_samples = self.config.architecture.shared.diffusion.no_full_rollout_samples
        num_atoms = outputs["atom_positions_predicted"].shape[-2]
        compute_per_sample = (
            num_samples > 1
            and self.config.settings.memory.eval.per_sample_atom_cutoff is not None
            and num_atoms > self.config.settings.memory.eval.per_sample_atom_cutoff
        )

        confidence_scores = get_confidence_scores(
            batch=batch,
            outputs=outputs,
            config=self.config,
            compute_per_sample=compute_per_sample,
        )

        return confidence_scores

    def predict_step(self, batch, batch_idx):
        # Skip if dataloader fails -> returns empty batch
        is_repeated_sample = batch.get("repeated_sample")
        valid_sample = batch.get("valid_sample")
        if not valid_sample or is_repeated_sample:
            return

        query_id = batch["query_id"]

        # Convert seeds back to list
        seed = batch["seed"].cpu().tolist()
        batch["seed"] = seed

        self.reseed(seed[0])  # TODO: assuming we have bs = 1 for now

        # Probably need to change the logic
        logger.debug(
            f"Started inference for {', '.join(query_id)} on rank {self.global_rank} "
            f"step {self.global_step}"
        )
        try:
            batch, outputs = self(batch)

            # Generate confidence scores
            confidence_scores = self._compute_confidence_scores(batch, outputs)
            outputs["confidence_scores"] = confidence_scores

            return batch, outputs

        except torch.OutOfMemoryError as e:
            logger.error(
                f"OOM for query_id(s) {', '.join(query_id)}. "
                f"See {self.log_dir}/predict_err_rank{self.global_rank}.log "
                f"for details."
            )

            self._log_predict_exception(e, query_id)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(
                f"Failed for query_id(s) {', '.join(query_id)}: {e}. "
                f"See {self.log_dir}/predict_err_rank{self.global_rank}.log "
                f"for details."
            )

            self._log_predict_exception(e, query_id)

    def _log_predict_exception(self, e, query_id):
        """Formats and appends exceptions to a rank-specific error log."""

        # Output dir is not specified
        if self.log_dir is None:
            return

        log_file = self.log_dir / f"predict_err_rank{self.global_rank}.log"

        # Get traceback and format message
        error_traceback = traceback.format_exc()

        lines = [
            "==================================================",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Query ID(s): {', '.join(query_id)}",
            f"Error Type: {type(e).__name__}",
            f"Error Message: {e}",
            "--------------------------------------------------",
            f"Traceback:{error_traceback}",
            "==================================================",
        ]
        log_entry = "\n".join(lines)

        # Append the entry to the log file
        with open(log_file, "a") as f:
            f.write(log_entry)
