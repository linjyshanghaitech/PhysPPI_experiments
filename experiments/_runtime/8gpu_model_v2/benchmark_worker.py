#!/usr/bin/env python3
"""Run a PhysPPI benchmark YAML with per-step CUDA timing.

This intentionally reuses the normal TrainingExperimentRunner so the benchmark
covers the real model forward, backward, optimizer step, EMA update, and data
path used by training.
"""

import argparse
import json
import sys
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _tensor_sum(value: Any) -> int | None:
    if value is None or not hasattr(value, "sum"):
        return None
    return int(value.detach().sum().item())


def _token_pad(batch: dict[str, Any]) -> int | None:
    token_mask = batch.get("token_mask")
    if token_mask is None or not hasattr(token_mask, "shape"):
        return None
    return int(token_mask.shape[-1])


def _batch_size(batch: dict[str, Any]) -> int | None:
    pdb_ids = batch.get("pdb_id")
    if isinstance(pdb_ids, (list, tuple)):
        return len(pdb_ids)
    token_mask = batch.get("token_mask")
    if token_mask is not None and hasattr(token_mask, "ndim") and token_mask.ndim > 1:
        return int(token_mask.shape[0])
    return None


def _atom_count(batch: dict[str, Any]) -> int | None:
    atom_count = _tensor_sum(batch.get("atom_mask"))
    if atom_count is not None:
        return atom_count
    ground_truth = batch.get("ground_truth")
    if isinstance(ground_truth, dict):
        return _tensor_sum(ground_truth.get("atom_resolved_mask"))
    return None


def _pdb_ids(batch: dict[str, Any]) -> list[str]:
    pdb_ids = batch.get("pdb_id")
    if isinstance(pdb_ids, (list, tuple)):
        return [str(pdb_id) for pdb_id in pdb_ids]
    if pdb_ids is None:
        return []
    return [str(pdb_ids)]


def _pad_and_stack_tensors_for_benchmark(values: list[torch.Tensor]) -> torch.Tensor:
    """Right-pad tensors across every non-batch sample dimension before stacking."""
    if values[0].ndim == 0:
        return torch.stack(values, dim=0)
    shapes = [tuple(value.shape) for value in values]
    if len(set(shapes)) == 1:
        stacked = torch.stack(values, dim=0)
    else:
        if len({len(shape) for shape in shapes}) != 1:
            raise RuntimeError(f"cannot collate tensors with different ranks: {shapes}")
        max_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(len(shapes[0])))
        padded = []
        for value in values:
            pad_spec: list[int] = []
            for dim in reversed(range(value.ndim)):
                pad_spec.extend([0, max_shape[dim] - value.shape[dim]])
            padded.append(torch.nn.functional.pad(value, pad_spec, value=0))
        stacked = torch.stack(padded, dim=0)
    if values[0].ndim == 1:
        return stacked.squeeze(-1)
    return stacked


def _benchmark_batch_collator(samples: list[dict[str, torch.Tensor]]):
    """Benchmark-only collator that pads PhysPPI tail dimensions before stack."""
    from openfold3.core.utils.tensor_utils import dict_multimap

    for sample in samples:
        interaction_mask = sample.get("physppi_interaction_type_mask")
        if not torch.is_tensor(interaction_mask) or interaction_mask.ndim < 1:
            continue
        if int(interaction_mask.shape[-1]) == 5:
            continue
        active = torch.as_tensor(
            sample.get("physppi_interaction_active", False), dtype=torch.bool
        ).reshape(-1)
        if bool(active.any()):
            raise ValueError(
                "active physppi_interaction_type_mask must have 5 channels, "
                f"got {tuple(interaction_mask.shape)}"
            )
        if int(interaction_mask.shape[-1]) > 5:
            sample["physppi_interaction_type_mask"] = interaction_mask[..., :5]

    has_pdb_id = "pdb_id" in samples[0]
    has_physppi_paired_msa_chain_mask = "physppi_paired_msa_chain_mask" in samples[0]

    if has_pdb_id:
        pdb_ids = [sample.pop("pdb_id") for sample in samples]

    if has_physppi_paired_msa_chain_mask:
        physppi_paired_msa_chain_masks = [
            sample.pop("physppi_paired_msa_chain_mask") for sample in samples
        ]

    has_ref_space_uid_to_perm = "ref_space_uid_to_perm" in samples[0]
    if has_ref_space_uid_to_perm:
        ref_space_uid_to_perm_dicts = [
            sample.pop("ref_space_uid_to_perm") for sample in samples
        ]

    has_atom_array = "atom_array" in samples[0]
    if has_atom_array:
        atom_arrays = [sample.pop("atom_array") for sample in samples]

    batch = dict_multimap(_pad_and_stack_tensors_for_benchmark, samples)

    if has_ref_space_uid_to_perm:
        batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm_dicts
    if has_atom_array:
        batch["atom_array"] = atom_arrays
    if has_pdb_id:
        batch["pdb_id"] = pdb_ids
    if has_physppi_paired_msa_chain_mask:
        batch["physppi_paired_msa_chain_mask"] = (
            _pad_and_stack_tensors_for_benchmark(physppi_paired_msa_chain_masks)
        )
    return batch


def _patch_openfold_batch_collator_for_benchmark() -> None:
    import openfold3.core.data.framework.data_module as data_module

    data_module.openfold_batch_collator = _benchmark_batch_collator


def _mean_positive_loss_weight(
    configured_weight: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if configured_weight is None:
        return None
    weight = torch.as_tensor(configured_weight, device=device, dtype=dtype).reshape(-1)
    if weight.numel() == 0:
        raise ValueError("physppi_loss_weight must not be empty")
    if not torch.isfinite(weight).all() or torch.any(weight <= 0):
        raise ValueError("physppi_loss_weight must be finite and positive")
    return weight.mean()


def _patch_physppi_batch_logging_for_benchmark() -> None:
    import openfold3.projects.of3_all_atom.runner as runner_module

    def sampler_trace_row(
        batch: dict[str, Any],
        *,
        rank: int,
        epoch: int,
        micro_step: int,
        optimizer_step: int | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "rank": int(rank),
            "epoch": int(epoch),
            "micro_step": int(micro_step),
            "sample_ids": _pdb_ids(batch),
            "sample_count": len(_pdb_ids(batch)),
            "benchmark_note": "batch logging aggregated by benchmark_worker",
        }
        if optimizer_step is not None:
            row["optimizer_step"] = int(optimizer_step)
        return row

    def write_physppi_route_c_train_metrics(**kwargs) -> None:
        return None

    def apply_physppi_loss_weight(
        batch: dict[str, Any],
        loss: torch.Tensor,
        loss_breakdown: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weight = _mean_positive_loss_weight(
            batch.get("physppi_loss_weight"),
            device=loss.device,
            dtype=loss.dtype,
        )
        if weight is None:
            return loss, loss_breakdown
        weighted_breakdown = {
            name: value * weight.to(device=value.device, dtype=value.dtype)
            for name, value in loss_breakdown.items()
        }
        return loss * weight, weighted_breakdown

    def populate_physppi_pdockq_targets(self, batch, outputs) -> None:
        return None

    def log_step_metrics(self, loss_breakdown, batch, outputs, train: bool = True):
        return None

    runner_module.sampler_trace_row = sampler_trace_row
    runner_module.write_physppi_route_c_train_metrics = (
        write_physppi_route_c_train_metrics
    )
    runner_module.apply_physppi_loss_weight = apply_physppi_loss_weight
    runner_module.OpenFold3AllAtom._populate_physppi_pdockq_targets = (
        populate_physppi_pdockq_targets
    )
    runner_module.OpenFold3AllAtom._log = log_step_metrics


class BenchmarkTimingCallback(pl.Callback):
    def __init__(self, timing_dir: Path, warmup_steps: int, measured_steps: int):
        super().__init__()
        self.timing_dir = Path(timing_dir)
        self.warmup_steps = int(warmup_steps)
        self.measured_steps = int(measured_steps)
        self._start: float | None = None

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._sync_cuda()
        self._start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync_cuda()
        if self._start is None:
            return
        elapsed = time.perf_counter() - self._start
        rank = int(getattr(trainer, "global_rank", 0))
        row = {
            "rank": rank,
            "local_rank": int(getattr(trainer, "local_rank", 0)),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "batch_idx": int(batch_idx),
            "phase": (
                "warmup"
                if int(batch_idx) < self.warmup_steps
                else "measured"
            ),
            "elapsed_seconds": float(elapsed),
            "n_pad": _token_pad(batch),
            "batch_size": _batch_size(batch),
            "token_count": _tensor_sum(batch.get("token_mask")),
            "atom_count": _atom_count(batch),
            "pdb_id": _pdb_ids(batch),
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else None
            ),
            "peak_reserved_gb": (
                torch.cuda.max_memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else None
            ),
        }
        timing_path = self.timing_dir / f"benchmark_step_rank-{rank}.jsonl"
        _append_jsonl(timing_path, row)


def _benchmark_runner_class():
    from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
    from openfold3.entry_points.validator import TrainingExperimentConfig

    class BenchmarkTrainingExperimentRunner(TrainingExperimentRunner):
        def __init__(
            self,
            experiment_config: TrainingExperimentConfig,
            *,
            timing_dir: Path,
            warmup_steps: int,
            measured_steps: int,
        ):
            super().__init__(experiment_config)
            self.benchmark_timing_dir = Path(timing_dir)
            self.benchmark_warmup_steps = int(warmup_steps)
            self.benchmark_measured_steps = int(measured_steps)

        @cached_property
        def callbacks(self):
            return [
                *super().callbacks,
                BenchmarkTimingCallback(
                    timing_dir=self.benchmark_timing_dir,
                    warmup_steps=self.benchmark_warmup_steps,
                    measured_steps=self.benchmark_measured_steps,
                ),
            ]

        @cached_property
        def strategy(self):
            if self.deepspeed_config_path is not None or not self.is_distributed:
                return super().strategy
            from pytorch_lightning.strategies import DDPStrategy

            return DDPStrategy(
                find_unused_parameters=True,
                cluster_environment=self.cluster_environment,
                timeout=self.pl_trainer_args.distributed_timeout,
            )

    return BenchmarkTrainingExperimentRunner


def run_benchmark(args: argparse.Namespace) -> None:
    from openfold3.core.config import config_utils
    from openfold3.entry_points.import_utils import _torch_gpu_setup
    from openfold3.entry_points.validator import TrainingExperimentConfig

    _torch_gpu_setup()
    _patch_physppi_batch_logging_for_benchmark()
    _patch_openfold_batch_collator_for_benchmark()
    runner_dict = config_utils.load_yaml(Path(args.runner_yaml))
    if args.seed is not None:
        runner_dict["experiment_settings"]["seed"] = int(args.seed)
    if args.data_seed is not None:
        runner_dict["data_module_args"]["data_seed"] = int(args.data_seed)

    expt_config = TrainingExperimentConfig.model_validate(runner_dict)
    BenchmarkTrainingExperimentRunner = _benchmark_runner_class()
    runner = BenchmarkTrainingExperimentRunner(
        expt_config,
        timing_dir=Path(args.timing_dir),
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(args.measured_steps),
    )
    runner.setup()
    runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-yaml", required=True)
    parser.add_argument("--timing-dir", required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--measured-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
