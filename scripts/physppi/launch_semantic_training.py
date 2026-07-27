# ruff: noqa: E402
"""Validate and launch one immutable eight-GPU PhysPPI semantic run."""

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
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from openfold3.core.data.framework.stochastic_sampler_dataset import (
    OF3DistributedSampler,
    SamplerDataset,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.occurrence import (
    build_occurrence_rng,
)
from scripts.physppi.prepare_semantic_50epoch import validate_semantic_config

CUDA_DEVICES = "0,1,2,3,4,5,6,7"


@dataclass(frozen=True)
class GPUProcess:
    gpu: int | str
    pid: int
    process_name: str
    used_memory_mib: int


@dataclass(frozen=True)
class LaunchResult:
    launcher_pid: int
    child_pid: int
    pgid: int
    command: tuple[str, ...]
    log_path: Path


class _ProbabilityColumn:
    def __init__(self, size: int) -> None:
        self._values = np.ones(size, dtype=np.float64)

    def to_numpy(self) -> np.ndarray:
        return self._values


class _PrefixDataset:
    def __init__(self, name: str, sample_ids: list[str], dataset_key: str) -> None:
        self.name = name
        self.sample_ids = sample_ids
        self.dataset_key = dataset_key
        self.datapoint_cache = {
            "datapoint_probabilities": _ProbabilityColumn(len(sample_ids))
        }

    def __len__(self) -> int:
        return len(self.sample_ids)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def gpu_processes() -> tuple[GPUProcess, ...]:
    """Return every visible compute process; unresolved occupancy is blocking."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: list[GPUProcess] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        gpu, pid, process_name, memory = (field.strip() for field in line.split(",", 3))
        processes.append(
            GPUProcess(
                gpu=gpu,
                pid=int(pid),
                process_name=process_name,
                used_memory_mib=int(memory),
            )
        )
    return tuple(processes)


def materialize_sampler_prefix(
    config: dict[str, Any],
    *,
    prefix_micro_steps: int,
) -> dict[int, list[dict[str, int | float | str]]]:
    """Materialize a data-only rank-local sampler prefix from a semantic YAML."""
    if prefix_micro_steps <= 0:
        raise ValueError("prefix_micro_steps must be positive")
    data_module = config["data_module_args"]
    epoch_len = int(data_module["epoch_len"])
    world_size = 8
    prefix_slots = prefix_micro_steps * world_size
    if prefix_slots > epoch_len:
        raise ValueError("sampler prefix exceeds configured epoch_len")
    if not bool(data_module["distinct_samples_per_rank_group"]):
        raise ValueError("sampler prefix requires distinct rank-group mode")

    train = config["dataset_configs"]["train"]
    manifest_paths = {
        Path(dataset_config["config"]["physppi"]["manifest"])
        for dataset_config in train.values()
    }
    if len(manifest_paths) != 1:
        raise ValueError("sampler prefix requires one shared manifest")
    manifest = json.loads(manifest_paths.pop().read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("sampler prefix requires manifest schema_version 2")

    datasets: list[_PrefixDataset] = []
    weights: list[float] = []
    for name, dataset_config in train.items():
        physppi = dataset_config["config"]["physppi"]
        dataset_key = str(physppi["manifest_dataset_key"])
        partition = str(physppi["manifest_partition"])
        sample_ids = [
            str(sample_id) for sample_id in manifest["datasets"][dataset_key][partition]
        ]
        datasets.append(_PrefixDataset(name, sample_ids, dataset_key))
        weights.append(float(dataset_config["weight"]))

    composite = SamplerDataset(datasets=datasets, epoch_len=epoch_len)
    data_seed = int(data_module["data_seed"])
    rows_by_rank: dict[int, list[dict[str, int | float | str]]] = {}
    for rank in range(world_size):
        sampler = OF3DistributedSampler(
            dataset=composite,
            dataset_probabilities=weights,
            next_dataset_indices={},
            epoch_len=epoch_len,
            num_replicas=world_size,
            rank=rank,
            seed=data_seed,
            distinct_samples_per_rank_group=True,
        )
        occurrences = list(sampler)[:prefix_micro_steps]
        rows: list[dict[str, int | float | str]] = []
        for micro_step, occurrence in enumerate(occurrences):
            dataset = datasets[occurrence.dataset_idx]
            sample_id = dataset.sample_ids[occurrence.datapoint_idx]
            occurrence_rng = build_occurrence_rng(
                data_seed,
                occurrence.epoch,
                occurrence.global_slot,
                dataset.name,
                sample_id,
            )
            rows.append(
                {
                    "rank": rank,
                    "epoch": occurrence.epoch,
                    "micro_step": micro_step,
                    "dataset": dataset.dataset_key,
                    "sample_id": sample_id,
                    "occurrence_seed": occurrence_rng.seed,
                    "requested_level": int(occurrence_rng.slots.integers(0, 5)),
                    "contact_cutoff": float(occurrence_rng.contact.uniform(8.0, 18.0)),
                }
            )
        rows_by_rank[rank] = rows
    return rows_by_rank


def _semantic_parity_projection(config: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(config)
    projected["experiment_settings"].pop("output_dir")
    projected["data_module_args"].pop("epoch_len")
    projected["pl_trainer_args"].pop("max_epochs")
    projected["pl_trainer_args"].pop("max_steps")
    return projected


def _run_dir_for_trace(trace_dir: Path) -> Path:
    for candidate in (trace_dir, *trace_dir.parents):
        if (candidate / "evidence" / "artifact-hashes.json").is_file():
            return candidate
    raise ValueError("cannot locate run evidence for sampler trace")


def verify_sampler_prefix(
    formal_config_path: Path,
    smoke_trace_dir: Path,
    *,
    prefix_micro_steps: int,
) -> dict[str, Any]:
    """Prove formal data assignments are the exact deterministic smoke prefix."""
    formal_config = yaml.safe_load(formal_config_path.read_text())
    run_dir = _run_dir_for_trace(smoke_trace_dir)
    artifacts = json.loads((run_dir / "evidence" / "artifact-hashes.json").read_text())
    smoke_config_path = Path(artifacts["runner_yaml"]["path"])
    smoke_config = yaml.safe_load(smoke_config_path.read_text())
    if _semantic_parity_projection(formal_config) != _semantic_parity_projection(
        smoke_config
    ):
        raise ValueError("smoke/formal semantic configs differ outside allowed fields")

    expected_smoke = materialize_sampler_prefix(
        smoke_config,
        prefix_micro_steps=prefix_micro_steps,
    )
    expected_formal = materialize_sampler_prefix(
        formal_config,
        prefix_micro_steps=prefix_micro_steps,
    )
    if expected_formal != expected_smoke:
        raise ValueError("formal sampler prefix differs from smoke sampler prefix")

    compared_fields = (
        "rank",
        "epoch",
        "micro_step",
        "dataset",
        "sample_id",
        "occurrence_seed",
        "requested_level",
        "contact_cutoff",
    )
    effective_levels: set[int] = set()
    for rank, expected_rows in expected_formal.items():
        path = smoke_trace_dir / f"rank-{rank}.jsonl"
        actual_rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if len(actual_rows) < prefix_micro_steps:
            raise ValueError(
                f"rank {rank} trace has {len(actual_rows)} rows, "
                f"expected at least {prefix_micro_steps}"
            )
        for expected, actual in zip(
            expected_rows,
            actual_rows[:prefix_micro_steps],
            strict=True,
        ):
            for field in compared_fields:
                if field == "contact_cutoff":
                    matches = math.isclose(
                        float(actual[field]),
                        float(expected[field]),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                else:
                    matches = actual[field] == expected[field]
                if not matches:
                    raise ValueError(
                        f"sampler prefix mismatch rank={rank} "
                        f"micro_step={expected['micro_step']} field={field}: "
                        f"actual={actual[field]!r} expected={expected[field]!r}"
                    )
            effective_level = int(actual["effective_level"])
            requested_level = int(actual["requested_level"])
            if not 0 <= effective_level <= requested_level <= 4:
                raise ValueError(
                    f"invalid effective/requested level in rank {rank} trace"
                )
            effective_levels.add(effective_level)

    for micro_step in range(prefix_micro_steps):
        ids = {expected_formal[rank][micro_step]["sample_id"] for rank in range(8)}
        if len(ids) != 8:
            raise ValueError(f"duplicate rank-group ID at micro_step={micro_step}")

    return {
        "clean": True,
        "formal_config": str(formal_config_path),
        "formal_config_sha256": _sha256(formal_config_path),
        "smoke_config": str(smoke_config_path),
        "smoke_config_sha256": _sha256(smoke_config_path),
        "smoke_trace_dir": str(smoke_trace_dir),
        "prefix_micro_steps": prefix_micro_steps,
        "ranks": 8,
        "rows_compared": 8 * prefix_micro_steps,
        "compared_fields": list(compared_fields),
        "effective_levels_observed": sorted(effective_levels),
        "effective_level_proof": (
            "identical sample occurrence and semantic config; deterministic "
            "feature availability and cumulative-level resolution"
        ),
    }


def _manifest_path(config: dict[str, Any]) -> Path:
    train = config["dataset_configs"]["train"]
    paths = {
        Path(train[name]["config"]["physppi"]["manifest"])
        for name in ("qbiolip10k", "afdb10k")
    }
    if len(paths) != 1:
        raise ValueError("both training sources must use one screened manifest")
    return paths.pop()


def _validate_launch_config(
    config: dict[str, Any],
    expected_manifest_sha256: str,
) -> tuple[Path, Path]:
    manifest = _manifest_path(config)
    if _sha256(manifest) != expected_manifest_sha256:
        raise ValueError("screened manifest SHA256 mismatch")
    manifest_payload = json.loads(manifest.read_text())
    if manifest_payload.get("schema_version") != 2:
        raise ValueError("screened manifest schema_version must be 2")
    complete = manifest.parent / "COMPLETE"
    if not complete.is_file():
        raise ValueError("screened manifest is not backed by COMPLETE")
    trainer = config["pl_trainer_args"]
    smoke = trainer["max_epochs"] == 1 and trainer["max_steps"] == 32
    validate_semantic_config(
        config,
        smoke=smoke,
        manifest_sha256=expected_manifest_sha256,
    )
    checkpoint = Path(config["experiment_settings"]["restart_checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"base checkpoint does not exist: {checkpoint}")
    return manifest, checkpoint


def _run_text(command: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _cache_paths(config: object, prefix: str = "") -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if isinstance(config, dict):
        for key, value in config.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if "cache" in str(key).lower() and isinstance(value, str):
                candidate = Path(value)
                if candidate.is_file():
                    paths[label] = candidate
            else:
                paths.update(_cache_paths(value, label))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            paths.update(_cache_paths(value, f"{prefix}[{index}]"))
    return paths


def _capture_static_evidence(
    *,
    evidence_dir: Path,
    repository: Path,
    config_path: Path,
    manifest: Path,
    checkpoint: Path,
    config: dict[str, Any],
) -> None:
    (evidence_dir / "git-head.txt").write_text(
        _run_text(["git", "rev-parse", "HEAD"], cwd=repository)
    )
    (evidence_dir / "git-status.txt").write_text(
        _run_text(["git", "status", "--short"], cwd=repository)
    )
    patch = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    (evidence_dir / "working-tree.patch").write_bytes(patch)
    (evidence_dir / "nvidia-smi-q.txt").write_text(
        _run_text(["nvidia-smi", "-q"], cwd=repository)
    )
    process_snapshot = _run_text(
        ["ps", "-eo", "pid,ppid,pgid,lstart,cwd,args"],
        cwd=repository,
    )
    (evidence_dir / "process-list.txt").write_text(process_snapshot)
    hashes = {
        "runner_yaml": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "base_checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "caches": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in sorted(_cache_paths(config).items())
        },
    }
    _atomic_json(evidence_dir / "artifact-hashes.json", hashes)


def launch_semantic_training(
    config_path: Path,
    output: Path,
    expected_manifest_sha256: str,
) -> LaunchResult:
    """Validate ownership/evidence gates, then start one new process group."""
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    busy = gpu_processes()
    if busy:
        rendered = ", ".join(
            f"gpu={process.gpu} pid={process.pid} {process.process_name}"
            for process in busy
        )
        raise RuntimeError(f"GPU occupancy is unresolved: {rendered}")
    config = yaml.safe_load(config_path.read_text())
    if Path(config["experiment_settings"]["output_dir"]) != output:
        raise ValueError("launcher output must equal YAML experiment output_dir")
    manifest, checkpoint = _validate_launch_config(
        config,
        expected_manifest_sha256,
    )

    output.mkdir(parents=True)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    repository = Path(__file__).resolve().parents[2]
    _capture_static_evidence(
        evidence_dir=evidence_dir,
        repository=repository,
        config_path=config_path,
        manifest=manifest,
        checkpoint=checkpoint,
        config=config,
    )
    command = (
        str(repository / ".pixi/envs/openfold3-cuda12/bin/run_openfold"),
        "train",
        "--runner-yaml",
        str(config_path),
        "--seed",
        "42",
        "--data-seed",
        "42",
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repository) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    log_path = output / "train.log"
    with log_path.open("ab", buffering=0) as log_handle:
        child = subprocess.Popen(
            command,
            cwd=repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = LaunchResult(
        launcher_pid=os.getpid(),
        child_pid=child.pid,
        pgid=os.getpgid(child.pid),
        command=command,
        log_path=log_path,
    )
    _atomic_json(
        evidence_dir / "launcher-process.json",
        {
            **asdict(result),
            "command": list(result.command),
            "log_path": str(result.log_path),
            "start_utc": datetime.now(UTC).isoformat(),
        },
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-yaml", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--verify-prefix-against", type=Path)
    parser.add_argument("--prefix-micro-steps", type=int, default=64)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_prefix_against is not None:
        report = verify_sampler_prefix(
            args.runner_yaml,
            args.verify_prefix_against,
            prefix_micro_steps=args.prefix_micro_steps,
        )
        if args.parity_report is not None:
            args.parity_report.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(args.parity_report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.preflight_only:
            return
    if args.output is None or args.manifest_sha256 is None:
        raise SystemExit("--output and --manifest-sha256 are required for launch")
    launch_semantic_training(
        args.runner_yaml,
        args.output,
        args.manifest_sha256,
    )


if __name__ == "__main__":
    main()
