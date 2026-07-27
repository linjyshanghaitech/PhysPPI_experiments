#!/usr/bin/env python3
"""Launch Q10k official_predict shards across multiple GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path("/101045/jinyu/PhysPPI/openfold-3")
DEFAULT_INPUT_DIR = REPO / "output/physppi_q10k_local_msa_training/inputs"
DEFAULT_OUTPUT_BASE = REPO / "output/physppi_q10k_local_msa_training"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def split_ids_round_robin(ids: list[str], shard_count: int) -> list[list[str]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for index, sample_id in enumerate(ids):
        shards[index % shard_count].append(sample_id)
    return shards


def _launch_shard(
    *,
    shard_index: int,
    gpu: str,
    run_root: Path,
    input_dir: Path,
    shard_train_ids: Path,
    shard_test_ids: Path,
    dry_run: bool,
) -> subprocess.Popen | subprocess.CompletedProcess:
    cmd = [
        "python3",
        "scripts/dev/physppi_broad_predict_compatible_eval.py",
        "--variant",
        "official_predict",
        "--gpu",
        gpu,
        "--train-query",
        str(input_dir / "queries" / "train_with_msa.json"),
        "--test-query",
        str(input_dir / "queries" / "test_with_msa.json"),
        "--train-ids",
        str(shard_train_ids),
        "--test-ids",
        str(shard_test_ids),
        "--pkl-dir",
        "/101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k/total_pkl",
        "--native-gt-dir",
        "/101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k/gt_multimer",
        "--run-root",
        str(run_root / f"shard_{shard_index}"),
    ]
    if dry_run:
        return subprocess.run(cmd + ["--dry-run"], cwd=REPO, check=True)
    log_path = run_root / "logs" / f"shard_{shard_index}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id")
    run_root = args.run_root or DEFAULT_OUTPUT_BASE / (
        "official_shards_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=False)

    train_ids = [
        str(x) for x in load_json(args.input_dir / "splits" / "train_ids.json")
    ]
    test_ids = [str(x) for x in load_json(args.input_dir / "splits" / "test_ids.json")]
    train_shards = split_ids_round_robin(train_ids, len(gpus))
    test_shards = split_ids_round_robin(test_ids, len(gpus))

    shard_records = []
    processes: list[tuple[int, subprocess.Popen]] = []
    for index, gpu in enumerate(gpus):
        shard_dir = run_root / "inputs" / f"shard_{index}"
        shard_train_ids = shard_dir / "train_ids.json"
        shard_test_ids = shard_dir / "test_ids.json"
        write_json(shard_train_ids, train_shards[index])
        write_json(shard_test_ids, test_shards[index])
        shard_records.append(
            {
                "shard": index,
                "gpu": gpu,
                "train_count": len(train_shards[index]),
                "test_count": len(test_shards[index]),
                "run_root": str(run_root / f"shard_{index}"),
                "train_ids": str(shard_train_ids),
                "test_ids": str(shard_test_ids),
            }
        )
        proc = _launch_shard(
            shard_index=index,
            gpu=gpu,
            run_root=run_root,
            input_dir=args.input_dir,
            shard_train_ids=shard_train_ids,
            shard_test_ids=shard_test_ids,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            processes.append((index, proc))

    manifest = {
        "run_root": str(run_root),
        "gpus": gpus,
        "dry_run": args.dry_run,
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "shards": shard_records,
    }
    write_json(run_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))

    if args.dry_run:
        return
    write_json(run_root / "pids.json", {str(i): p.pid for i, p in processes})
    failures = []
    for index, proc in processes:
        code = proc.wait()
        if code != 0:
            failures.append({"shard": index, "returncode": code})
    if failures:
        write_json(run_root / "failures.json", failures)
        raise SystemExit(f"shard failures: {failures}")
    (run_root / "status.done").write_text("done\n")


if __name__ == "__main__":
    main()
