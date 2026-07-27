"""Run one real OpenFold3-ready sample with PhysPPI enabled on one GPU.

The sample path must point to a torch-saved dictionary containing OpenFold3-ready
features, or to a dictionary with `batch` and optional `physppi_features` keys.
This launcher intentionally does not build ESM features.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
    inject_physppi_features,
    validate_required_physppi_features,
)
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom


def load_sample(path: Path) -> dict:
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "batch" in loaded:
        batch = loaded["batch"]
        physppi = loaded.get("physppi_features", {})
        return inject_physppi_features(batch=batch, physppi_features=physppi)
    if isinstance(loaded, dict):
        physppi = {k: v for k, v in loaded.items() if k.startswith("physppi_")}
        batch = {k: v for k, v in loaded.items() if not k.startswith("physppi_")}
        return inject_physppi_features(batch=batch, physppi_features=physppi)
    raise TypeError(f"Unsupported sample object type: {type(loaded)!r}")


def build_model(device: torch.device) -> tuple[OpenFold3AllAtom, OpenFold3Loss]:
    project_entry = OF3ProjectEntry()
    config = project_entry.get_model_config_with_presets()
    config.architecture.physppi.enabled = True
    config.architecture.shared.num_recycles = 0
    config.architecture.shared.diffusion.use_conditioning_prob = 1.0
    config.architecture.shared.use_confidence_emb_prob = 1.0
    config.settings.memory.train.use_deepspeed_evo_attention = False
    config.settings.memory.eval.use_deepspeed_evo_attention = False
    config.settings.gradient_clipping.per_sample_clipping = False
    model = OpenFold3AllAtom(config).to(device=device)
    loss_fn = OpenFold3Loss(config=config.architecture.loss_module)
    return model, loss_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    random.seed(11)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = load_sample(args.sample)
    validate_required_physppi_features(batch, enabled=True, mode="train")
    batch = OF3DeepSpeedPrecision(precision="32-true").convert_input(batch)
    batch = tensor_tree_map(lambda t: t.to(device=device), batch)

    model, loss_fn = build_model(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "loss", "step_time_s", "max_mem_gb"],
        )
        writer.writeheader()
        for step in range(args.steps):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _, output = model(batch)
            loss, _ = loss_fn(batch=batch, output=output, _return_breakdown=True)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                max_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            else:
                max_mem_gb = 0.0
            step_time = time.perf_counter() - t0
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "step_time_s": step_time,
                "max_mem_gb": max_mem_gb,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"step={step} loss={row['loss']:.6g} "
                f"step_time_s={step_time:.3f} max_mem_gb={max_mem_gb:.2f}",
                flush=True,
            )
    print(f"metrics_csv={metrics_path}")


if __name__ == "__main__":
    main()
