"""Summarize PhysPPI training metrics and GPU utilization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_metrics(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    losses = [float(row["loss"]) for row in rows if row.get("loss")]
    step_times = [float(row["step_time_s"]) for row in rows if row.get("step_time_s")]
    mem = [float(row["max_mem_gb"]) for row in rows if row.get("max_mem_gb")]
    return {
        "steps": len(rows),
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last": losses[-1] if losses else float("nan"),
        "loss_min": min(losses) if losses else float("nan"),
        "avg_step_time_excluding_step0": mean(step_times[1:])
        if len(step_times) > 1
        else (step_times[0] if step_times else float("nan")),
        "max_memory_gb": max(mem) if mem else float("nan"),
    }


def read_gpu(path: Path) -> dict[str, float]:
    util = []
    mem = []
    power = []
    if not path.exists():
        return {"gpu_samples": 0}
    for line in path.read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            util.append(float(parts[2]))
            mem.append(float(parts[3]))
            power.append(float(parts[4]))
        except ValueError:
            continue
    return {
        "gpu_samples": len(util),
        "gpu_util_avg": mean(util) if util else float("nan"),
        "gpu_util_max": max(util) if util else float("nan"),
        "gpu_memory_mib_max": max(mem) if mem else float("nan"),
        "gpu_power_w_max": max(power) if power else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = {"run_dir": str(args.run_dir)}
    metrics = args.run_dir / "metrics.csv"
    if metrics.exists():
        summary.update(read_metrics(metrics))
    summary.update(read_gpu(args.run_dir / "gpu.csv"))
    out = args.run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
