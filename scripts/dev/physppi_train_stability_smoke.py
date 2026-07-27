"""Isolated finite train-step smoke for PhysPPI optional priors.

The smoke intentionally avoids the production trainer, checkpoint paths, and
YAML mutation.  It builds a tiny normal PyTorch module with the production
PhysPPI optional-prior conditioner, freezes the base projection, keeps an
always-on LoRA-style adapter trainable, and records JSONL results for
``mandatory_only`` and ``all_available`` prior modes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SMOKE_MODES = ("mandatory_only", "all_available")


@dataclass(frozen=True)
class TrainStabilitySmokeConfig:
    """Configuration for the isolated smoke run."""

    output: Path
    runner_yaml: Path = Path("examples/training_yamls/physppi_training.yml")
    modes: tuple[str, ...] = SMOKE_MODES
    audit_json: tuple[Path, ...] = ()
    max_steps: int = 1
    seed: int = 20260720
    device: str = "cpu"
    c_s: int = 16
    c_z: int = 8
    n_tokens: int = 4
    max_tokens: int = 128
    n_surface_neighbors: int = 128
    learning_rate: float = 1e-3
    production_freeze: Mapping[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "frozen_modules": ("base_projection",),
        }
    )
    lora: Mapping[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "always_on": True,
            "target_modules": ("lora_adapter",),
        }
    )


def production_stability_config(output: str | Path) -> TrainStabilitySmokeConfig:
    """Return the production-like freeze plus always-on-LoRA smoke config."""

    return TrainStabilitySmokeConfig(output=Path(output))


def _record_sample_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("sample_id", record.get("pdb_id", record.get("id")))
    return None if value is None else str(value)


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        yield dict(value)
        return
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload.get("records"), list):
        for record in payload["records"]:
            if isinstance(record, Mapping):
                yield dict(record)
        return
    for dataset, report in payload.get("datasets", {}).items():
        if not isinstance(report, Mapping):
            continue
        fatal = set(str(value) for value in report.get("fatal", {}).get("ids", []))
        valid_ids = report.get("valid_ids") or report.get("ids") or ()
        for sample_id in valid_ids:
            sample_id = str(sample_id)
            if sample_id not in fatal:
                yield {"dataset": str(dataset), "sample_id": sample_id}
        for state, sample_ids in report.get("allowed", {}).items():
            for sample_id in sample_ids:
                sample_id = str(sample_id)
                if sample_id not in fatal:
                    yield {
                        "dataset": str(dataset),
                        "sample_id": sample_id,
                        "audit_state": str(state),
                    }


def iter_prior_records(paths: Iterable[Path], max_steps: int) -> list[dict[str, Any]]:
    """Return valid audit records to exercise, bounded by ``max_steps``."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for record in _iter_json_records(path):
            status = str(record.get("status", "passed")).lower()
            if status in {"failed", "fatal", "error"}:
                continue
            sample_id = _record_sample_id(record)
            if sample_id is None:
                continue
            dataset = str(record.get("dataset", "unknown"))
            key = (dataset, sample_id)
            if key in seen:
                continue
            seen.add(key)
            records.append({"dataset": dataset, "sample_id": sample_id})
            if len(records) >= max_steps:
                return records
    if not records:
        records.append({"dataset": "synthetic", "sample_id": "synthetic_0"})
    return records[:max_steps]


def _set_deterministic_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch_module.manual_seed(seed)
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module, "use_deterministic_algorithms"):
        torch_module.use_deterministic_algorithms(True, warn_only=True)


def _slot_flags(torch_module, active: bool, device) -> dict[str, Any]:
    value = torch_module.tensor([active], dtype=torch_module.bool, device=device)
    return {
        "available": value.clone(),
        "selected": value.clone(),
        "valid": value.clone(),
        "active": value.clone(),
    }


def build_prior_batch(
    torch_module,
    *,
    mode: str,
    device,
    n_tokens: int,
    n_surface_neighbors: int,
) -> dict[str, Any]:
    """Build a deterministic canonical-prior batch for one smoke mode."""

    if mode not in SMOKE_MODES:
        raise ValueError(f"unknown smoke mode: {mode}")
    active = mode == "all_available"
    token_mask = torch_module.ones(1, n_tokens, dtype=torch_module.bool, device=device)
    batch: dict[str, Any] = {
        "token_mask": token_mask,
        "physppi_contact_token_mask": torch_module.zeros(
            1, n_tokens, dtype=torch_module.bool, device=device
        ),
        "physppi_interaction_type_mask": torch_module.zeros(
            1, n_tokens, n_tokens, 7, dtype=torch_module.bool, device=device
        ),
        "physppi_surface_local_coords": torch_module.zeros(
            1, n_tokens, n_surface_neighbors, 3, device=device
        ),
        "physppi_surface_features": torch_module.zeros(
            1, n_tokens, n_surface_neighbors, 6, device=device
        ),
        "physppi_surface_mask": torch_module.zeros(
            1, n_tokens, n_surface_neighbors, dtype=torch_module.bool, device=device
        ),
        "physppi_surface_charge_mask": torch_module.zeros(
            1, n_tokens, n_surface_neighbors, dtype=torch_module.bool, device=device
        ),
        "physppi_tm_token_mask": torch_module.zeros(
            1, n_tokens, dtype=torch_module.bool, device=device
        ),
    }
    if active:
        batch["physppi_contact_token_mask"][:, :2] = True
        batch["physppi_interaction_type_mask"][:, 0, 1, 0] = True
        batch["physppi_interaction_type_mask"][:, 1, 0, 0] = True
        batch["physppi_surface_mask"][:, :, :2] = True
        batch["physppi_surface_charge_mask"][:, :, :1] = True
        batch["physppi_surface_features"][:, :, :2, :] = 0.25
        batch["physppi_surface_local_coords"][:, :, :2, 0] = 0.5
        batch["physppi_tm_token_mask"][:, 0] = True
    for slot in ("contact", "interaction", "surface", "tm"):
        for name, value in _slot_flags(torch_module, active, device).items():
            batch[f"physppi_{slot}_{name}"] = value
    return batch


class TrainStabilityProbe:
    """Factory wrapper that lazily constructs the real PyTorch module."""

    @staticmethod
    def build(config: TrainStabilitySmokeConfig):
        import torch
        from torch import nn

        from openfold3.projects.of3_all_atom.physppi.model.conditioning import (
            OptionalPriorConditioner,
        )

        class _Probe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_projection = nn.Linear(config.c_s, config.c_s)
                self.lora_adapter = nn.Linear(config.c_s, config.c_s, bias=False)
                self.conditioner = OptionalPriorConditioner(
                    c_s=config.c_s, c_z=config.c_z
                )
                for parameter in self.base_projection.parameters():
                    parameter.requires_grad_(False)

            def forward(self, batch):
                token_mask = batch["token_mask"]
                batch_size, n_tokens = token_mask.shape
                x = torch.arange(
                    batch_size * n_tokens * config.c_s,
                    device=token_mask.device,
                    dtype=torch.float32,
                ).reshape(batch_size, n_tokens, config.c_s)
                x = x / max(float(config.c_s * n_tokens), 1.0)
                s_init = self.base_projection(x) + self.lora_adapter(x)
                z_init = torch.zeros(
                    batch_size,
                    n_tokens,
                    n_tokens,
                    config.c_z,
                    device=token_mask.device,
                )
                out = self.conditioner(
                    s_init=s_init,
                    z_init=z_init,
                    batch=batch,
                    token_mask=token_mask,
                    pair_mask=torch.ones(
                        batch_size,
                        n_tokens,
                        n_tokens,
                        device=token_mask.device,
                    ),
                )
                return out.s_init.square().mean() + out.z_init.square().mean()

        return _Probe()


def _finite_named_parameters(model) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not parameter.isfinite().all().item():
            bad.append(name)
    return not bad, bad


def _finite_gradients(model) -> tuple[bool, list[str], int]:
    bad: list[str] = []
    nonzero = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            bad.append(f"{name}:missing")
            continue
        if not parameter.grad.isfinite().all().item():
            bad.append(name)
        if parameter.grad.abs().sum().item() > 0.0:
            nonzero += 1
    return not bad and nonzero > 0, bad, nonzero


def validate_step_record(record: Mapping[str, Any]) -> None:
    """Validate finite loss/grad/parameter assertions from a smoke step."""

    for key in ("dataset", "sample_id", "step", "status"):
        if key not in record:
            raise RuntimeError(f"missing smoke record field: {key}")
    if record["status"] != "passed":
        raise RuntimeError(f"smoke step failed for {record.get('sample_id')}")
    loss = float(record["loss"])
    if not math.isfinite(loss):
        raise RuntimeError(f"nonfinite loss for {record.get('mode')}: {loss}")
    if not bool(record.get("finite_gradients")):
        raise RuntimeError(f"nonfinite gradients for {record.get('mode')}")
    if not bool(record.get("finite_parameters")):
        raise RuntimeError(f"nonfinite parameters for {record.get('mode')}")
    if (
        record.get("require_nonzero_gradients", True)
        and int(record.get("nonzero_gradient_tensors", 0)) <= 0
    ):
        raise RuntimeError(f"no nonzero gradients for {record.get('mode')}")


def run_training_step(
    config: TrainStabilitySmokeConfig,
    *,
    mode: str,
    prior_record: Mapping[str, Any],
    step: int,
    model_factory: Callable[[TrainStabilitySmokeConfig], Any] | None = None,
    batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one isolated finite train step and return a JSON-serializable record.

    ``batch`` is optional for the synthetic unit probe.  The real-data path
    passes a batch built from the canonical prior reader so the same frozen
    base/always-on-LoRA check exercises actual dataset features.
    """

    import torch

    _set_deterministic_seed(config.seed, torch)
    device = torch.device(config.device)
    model = (model_factory or TrainStabilityProbe.build)(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    if batch is None:
        batch = build_prior_batch(
            torch,
            mode=mode,
            device=device,
            n_tokens=config.n_tokens,
            n_surface_neighbors=config.n_surface_neighbors,
        )
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"nonfinite loss before backward for {mode}")
    loss.backward()
    finite_gradients, bad_gradients, nonzero_gradients = _finite_gradients(model)
    if not finite_gradients:
        raise RuntimeError(f"bad gradients for {mode}: {bad_gradients}")
    optimizer.step()
    finite_parameters, bad_parameters = _finite_named_parameters(model)
    if not finite_parameters:
        raise RuntimeError(f"bad parameters for {mode}: {bad_parameters}")
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    record = {
        "event": "train_stability_step",
        "dataset": str(prior_record["dataset"]),
        "sample_id": str(prior_record["sample_id"]),
        "step": int(step),
        "status": "passed",
        "mode": mode,
        "seed": config.seed,
        "loss": float(loss.detach().cpu()),
        "finite_gradients": True,
        "finite_parameters": True,
        "nonzero_gradient_tensors": int(nonzero_gradients),
        "trainable_parameter_count": int(trainable_count),
        "production_freeze": dict(config.production_freeze),
        "lora": dict(config.lora),
    }
    validate_step_record(record)
    return record


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_smoke(
    config: TrainStabilitySmokeConfig,
    *,
    step_runner: Callable[
        [TrainStabilitySmokeConfig, str, Mapping[str, Any], int], Mapping[str, Any]
    ]
    | None = None,
) -> list[dict[str, Any]]:
    """Run requested modes without mutating the caller's config object."""

    records: list[dict[str, Any]] = []
    runner = step_runner
    prior_records = iter_prior_records(config.audit_json, config.max_steps)
    failed = 0
    for mode in config.modes:
        if mode not in SMOKE_MODES:
            raise ValueError(f"unknown smoke mode: {mode}")
        for step, prior_record in enumerate(prior_records):
            mode_config = copy.deepcopy(config)
            try:
                record = dict(
                    runner(mode_config, mode, prior_record, step)
                    if runner is not None
                    else run_training_step(
                        mode_config,
                        mode=mode,
                        prior_record=prior_record,
                        step=step,
                    )
                )
                validate_step_record(record)
            except Exception as exc:
                failed += 1
                record = {
                    "event": "train_stability_step",
                    "dataset": str(prior_record.get("dataset", "unknown")),
                    "sample_id": str(prior_record.get("sample_id", "unknown")),
                    "step": int(step),
                    "status": "failed",
                    "mode": mode,
                    "seed": config.seed,
                    "error": repr(exc),
                }
            records.append(record)
    records.append(
        {
            "event": "train_stability_summary",
            "modes": list(config.modes),
            "audit_json": [str(path) for path in config.audit_json],
            "max_steps": config.max_steps,
            "seed": config.seed,
            "records": len(records),
            "failed": failed,
        }
    )
    _write_jsonl(config.output, records)
    if failed:
        raise RuntimeError(f"{failed} train stability smoke step(s) failed")
    return records


def run_real_dataset_smoke(config: TrainStabilitySmokeConfig) -> list[dict[str, Any]]:
    """Run bounded stability checks against the canonical reader IDs.

    Both modes use the same canonical feature construction as the prior-only
    finite smoke.  The mode only changes which available optional slots are
    active, while the frozen base and always-on LoRA probe is exercised on the
    resulting real batch.
    """

    import torch

    from scripts.dev import physppi_all_prior_finite_smoke as prior_smoke

    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the real-data stability smoke")
    device = torch.device(config.device)
    audit = prior_smoke._load_audit(list(config.audit_json))
    readers = prior_smoke._build_prior_readers(
        config.runner_yaml, audit["fatal_ids"], config.max_steps
    )
    prior_records: list[tuple[str, Any, str]] = []
    for dataset, reader, sample_ids in readers:
        for sample_id in sample_ids:
            prior_records.append((dataset, reader, sample_id))

    records: list[dict[str, Any]] = []
    failed = 0
    for mode in config.modes:
        for step, (dataset, reader, sample_id) in enumerate(prior_records):
            prior_record = {"dataset": dataset, "sample_id": sample_id}
            try:
                sample = prior_smoke._read_smoke_sample(reader, sample_id)
                real_batch, n_tokens, membrane_valid = (
                    prior_smoke._build_prior_only_batch(
                        sample=sample,
                        sample_id=sample_id,
                        c_s=config.c_s,
                        c_z=config.c_z,
                        device=device,
                        max_tokens=config.max_tokens,
                        mode=mode,
                    )
                )
                result = run_training_step(
                    config,
                    mode=mode,
                    prior_record=prior_record,
                    step=step,
                    batch=real_batch,
                )
                result.update(
                    {
                        "real_canonical_prior": True,
                        "n_tokens": n_tokens,
                        "membrane_valid": membrane_valid,
                    }
                )
                validate_step_record(result)
            except Exception as exc:
                failed += 1
                result = {
                    "event": "train_stability_step",
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "step": step,
                    "status": "failed",
                    "mode": mode,
                    "seed": config.seed,
                    "error": repr(exc),
                }
            records.append(result)
    records.append(
        {
            "event": "train_stability_summary",
            "modes": list(config.modes),
            "audit_json": [str(path) for path in config.audit_json],
            "runner_yaml": str(config.runner_yaml),
            "max_steps": config.max_steps,
            "seed": config.seed,
            "records": len(records),
            "failed": failed,
            "data_errors": audit["data_errors"],
        }
    )
    _write_jsonl(config.output, records)
    if failed:
        raise RuntimeError(f"{failed} real-data stability step(s) failed")
    return records


def _parse_modes(value: str) -> tuple[str, ...]:
    if value == "both":
        return SMOKE_MODES
    if value not in SMOKE_MODES:
        raise argparse.ArgumentTypeError(
            f"mode must be one of {', '.join((*SMOKE_MODES, 'both'))}"
        )
    return (value,)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runner-yaml",
        type=Path,
        default=Path("examples/training_yamls/physppi_training.yml"),
    )
    parser.add_argument("--audit-json", type=Path, action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--mode", type=_parse_modes, default=SMOKE_MODES)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = TrainStabilitySmokeConfig(
        output=args.output,
        runner_yaml=args.runner_yaml,
        modes=args.mode,
        audit_json=tuple(args.audit_json),
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        seed=args.seed,
        device=args.device,
    )
    if config.audit_json:
        run_real_dataset_smoke(config)
    else:
        run_smoke(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
