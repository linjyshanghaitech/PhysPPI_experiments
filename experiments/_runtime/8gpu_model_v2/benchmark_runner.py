#!/usr/bin/env python3
"""Benchmark PhysPPI v2 length buckets and fit cost coefficients.

This runner has two layers:

1. Pure analysis helpers (`fit_time_model`, `recommend_batch_sizes`) that turn
   measured JSONL records into coefficients and a README-ready recommendation.
2. A CLI that prepares fixed-case benchmark YAMLs and can run one candidate
   through the real `run_openfold train` entrypoint, recording status, wall time,
   and log paths.  Current production code still has known `batch_size > 1`
   assumptions; failed candidates are recorded as data, not hidden.
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_YAML = REPO_ROOT / "experiments/_runtime/4gpu_train_v1.yaml"
DEFAULT_RESULTS_JSONL = V2_DIR / "benchmark_results.jsonl"
DEFAULT_SUMMARY_JSON = V2_DIR / "benchmark_fit_summary.json"
DEFAULT_CASES_JSON = V2_DIR / "benchmark_cases.json"
DEFAULT_CONDA_ENV = Path("/101045/jinyu/.local/share/mamba/envs/PhysPPI")
DEFAULT_WEIGHTS = Path(
    "/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/sampling_weights.pt"
)
DEFAULT_MANIFEST = REPO_ROOT / "experiments/split_training_val/split_manifest.json"
DEFAULT_PRECOMPUTED_ROOT = Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3")
DEFAULT_SEQ_LEN_BIN_LABELS = DEFAULT_PRECOMPUTED_ROOT / "seq_len_bin_label.pt"

BUCKETS = [
    ("001_256", 1, 256, [1, 2, 4, 8]),
    ("257_384", 257, 384, [1, 2, 4]),
    ("385_512", 385, 512, [1, 2, 4]),
    ("513_640", 513, 640, [1, 2]),
    ("641_768", 641, 768, [1]),
]

RECOMMENDED_LOCAL_BATCH_BY_BUCKET = {
    "001_256": 4,
    "257_384": 4,
    "385_512": 2,
    "513_640": 1,
    "641_768": 1,
}


@dataclass(frozen=True)
class Candidate:
    bucket: str
    n_pad: int
    batch_size: int

    @property
    def name(self) -> str:
        return f"{self.bucket}_B{self.batch_size}"


def _ok_measurements(measurements: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in measurements if row.get("status", "ok") == "ok"]
    if not rows:
        raise ValueError("no successful benchmark measurements available")
    return rows


def _least_squares(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    coeffs, *_ = np.linalg.lstsq(features, targets, rcond=None)
    pred = features @ coeffs
    ss_res = float(np.sum((targets - pred) ** 2))
    ss_tot = float(np.sum((targets - float(np.mean(targets))) ** 2))
    r2 = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else 1.0 - ss_res / ss_tot
    return coeffs, r2


def fit_time_model(measurements: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Fit T(B,L) = t0 + B * (a0 + a1 L + a2 L^2)."""
    rows = _ok_measurements(measurements)
    x = []
    y = []
    for row in rows:
        n_pad = float(row["n_pad"])
        batch_size = float(row["batch_size"])
        x.append([1.0, batch_size, batch_size * n_pad, batch_size * n_pad * n_pad])
        y.append(float(row["median_step_seconds"]))
    features = np.asarray(x, dtype=float)
    coeffs, r2 = _least_squares(features, np.asarray(y, dtype=float))
    rank = int(np.linalg.matrix_rank(features))
    return {
        "t0": float(coeffs[0]),
        "a0": float(coeffs[1]),
        "a1": float(coeffs[2]),
        "a2": float(coeffs[3]),
        "r2": float(r2),
        "n": float(len(rows)),
        "rank": float(rank),
        "full_rank": bool(rank == features.shape[1]),
    }


def fit_b1_length_model(measurements: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Fit the identifiable B=1 curve T1(L) = b0 + b1 L + b2 L^2."""
    rows = [
        row
        for row in _ok_measurements(measurements)
        if int(row.get("batch_size", 0)) == 1
    ]
    if len(rows) < 3:
        raise ValueError("at least three successful B=1 measurements are required")
    x = []
    y = []
    for row in rows:
        n_pad = float(row["n_pad"])
        x.append([1.0, n_pad, n_pad * n_pad])
        y.append(float(row["median_step_seconds"]))
    features = np.asarray(x, dtype=float)
    coeffs, r2 = _least_squares(features, np.asarray(y, dtype=float))
    return {
        "b0": float(coeffs[0]),
        "b1": float(coeffs[1]),
        "b2": float(coeffs[2]),
        "r2": float(r2),
        "n": float(len(rows)),
    }


def fit_memory_model(measurements: Iterable[dict[str, Any]]) -> dict[str, float] | None:
    """Fit M(B,L,A) = m0 + B * (m1 L + m2 L^2 + m_atom A)."""
    rows = [
        row
        for row in _ok_measurements(measurements)
        if row.get("peak_reserved_gb") is not None
    ]
    if len(rows) < 4:
        return None
    x = []
    y = []
    for row in rows:
        n_pad = float(row["n_pad"])
        batch_size = float(row["batch_size"])
        atom_count = float(row.get("atom_count", 0.0) or 0.0)
        x.append([1.0, batch_size * n_pad, batch_size * n_pad * n_pad, batch_size * atom_count])
        y.append(float(row["peak_reserved_gb"]))
    coeffs, r2 = _least_squares(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {
        "m0": float(coeffs[0]),
        "m1": float(coeffs[1]),
        "m2": float(coeffs[2]),
        "m_atom": float(coeffs[3]),
        "r2": float(r2),
        "n": float(len(rows)),
    }


def _time_for_batch(time_fit: dict[str, float], n_pad: int, batch_size: int) -> float:
    return float(
        time_fit["t0"]
        + batch_size
        * (time_fit["a0"] + time_fit["a1"] * n_pad + time_fit["a2"] * n_pad * n_pad)
    )


def _memory_for_batch(
    memory_fit: dict[str, float], n_pad: int, batch_size: int, atom_count: float = 0.0
) -> float:
    return float(
        memory_fit["m0"]
        + batch_size
        * (
            memory_fit["m1"] * n_pad
            + memory_fit["m2"] * n_pad * n_pad
            + memory_fit["m_atom"] * atom_count
        )
    )


def recommend_batch_sizes(
    *,
    n_pads: Iterable[int],
    time_fit: dict[str, float],
    memory_fit: dict[str, float] | None,
    target_seconds: float,
    memory_limit: float | None,
    max_batch_size: int,
    representative_atom_counts: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    recommendations = []
    for n_pad in n_pads:
        atom_count = 0.0
        if representative_atom_counts is not None:
            atom_count = float(representative_atom_counts.get(int(n_pad), 0.0))
        time_ok = [
            batch
            for batch in range(1, max_batch_size + 1)
            if _time_for_batch(time_fit, int(n_pad), batch) <= target_seconds
        ]
        time_limited = max(time_ok) if time_ok else 1

        memory_limited = max_batch_size
        if memory_fit is not None and memory_limit is not None:
            memory_ok = [
                batch
                for batch in range(1, max_batch_size + 1)
                if _memory_for_batch(memory_fit, int(n_pad), batch, atom_count=atom_count)
                <= memory_limit
            ]
            memory_limited = max(memory_ok) if memory_ok else 1

        recommended = max(1, min(max_batch_size, time_limited, memory_limited))
        recommendations.append(
            {
                "n_pad": int(n_pad),
                "recommended_batch_size": int(recommended),
                "time_limited_batch_size": int(time_limited),
                "memory_limited_batch_size": int(memory_limited),
                "predicted_seconds": _time_for_batch(time_fit, int(n_pad), recommended),
                "predicted_peak_reserved_gb": (
                    None
                    if memory_fit is None
                    else _memory_for_batch(
                        memory_fit,
                        int(n_pad),
                        recommended,
                        atom_count=atom_count,
                    )
                ),
            }
        )
    return recommendations


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_step_timings(
    timing_rows: Iterable[dict[str, Any]],
    *,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    """Summarize per-rank CUDA-synchronized step timing rows.

    DDP step time is limited by the slowest rank, so each global batch uses the
    maximum elapsed time across ranks before taking the median across measured
    batches.
    """
    first_measured = int(warmup_steps)
    last_measured = first_measured + int(measured_steps)
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in timing_rows:
        batch_idx = int(row["batch_idx"])
        if first_measured <= batch_idx < last_measured:
            by_batch.setdefault(batch_idx, []).append(row)

    if not by_batch:
        raise ValueError(
            f"no measured step timing rows for batch_idx in "
            f"[{first_measured}, {last_measured})"
        )

    step_seconds = []
    peak_allocated = []
    peak_reserved = []
    token_count = []
    atom_count = []
    for batch_idx in sorted(by_batch):
        rows = by_batch[batch_idx]
        step_seconds.append(max(float(row["elapsed_seconds"]) for row in rows))
        for row in rows:
            if row.get("peak_allocated_gb") is not None:
                peak_allocated.append(float(row["peak_allocated_gb"]))
            if row.get("peak_reserved_gb") is not None:
                peak_reserved.append(float(row["peak_reserved_gb"]))
            if row.get("token_count") is not None:
                token_count.append(float(row["token_count"]))
            if row.get("atom_count") is not None:
                atom_count.append(float(row["atom_count"]))

    return {
        "measured_step_count": len(step_seconds),
        "median_step_seconds": float(statistics.median(step_seconds)),
        "mean_step_seconds": float(statistics.fmean(step_seconds)),
        "min_step_seconds": float(min(step_seconds)),
        "max_step_seconds": float(max(step_seconds)),
        "peak_allocated_gb": max(peak_allocated) if peak_allocated else None,
        "peak_reserved_gb": max(peak_reserved) if peak_reserved else None,
        "token_count": max(token_count) if token_count else None,
        "atom_count": max(atom_count) if atom_count else None,
    }


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(bucket=bucket, n_pad=hi, batch_size=batch_size)
        for bucket, _lo, hi, batches in BUCKETS
        for batch_size in batches
    ]


def _mask_len(mask: Any) -> int:
    if hasattr(mask, "sum"):
        return int(mask.sum())
    return int(sum(mask))


def _case_len_from_seq_pt(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    total = 0
    for meta in payload.values():
        if isinstance(meta, dict):
            mask = meta.get("mask")
            total += len(meta.get("full_seq_1aa", "")) if mask is None else _mask_len(mask)
    return int(total)


def _find_seq_pt(dataset_root: Path, case_id: str) -> Path | None:
    matches = list(dataset_root.glob(f"chunk*/{case_id}_seq.pt"))
    if len(matches) > 1:
        raise RuntimeError(f"duplicate _seq.pt for {case_id}: {matches}")
    return matches[0] if matches else None


def build_case_table(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    precomputed_root: Path = DEFAULT_PRECOMPUTED_ROOT,
    weights_path: Path = DEFAULT_WEIGHTS,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    weights_raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = weights_raw.get("sampling_weights", weights_raw)
    specs = {
        "afdb": (precomputed_root / "AFDB-10k-1", manifest["datasets"]["afdb"]["train_ids"]),
        "qbiolip": (
            precomputed_root / "Q-BioLiP-10k-new",
            manifest["datasets"]["qbiolip"]["train_ids"],
        ),
    }
    rows = []
    for dataset, (root, case_ids) in specs.items():
        for case_id in case_ids:
            seq_pt = _find_seq_pt(root, case_id)
            if seq_pt is None:
                continue
            length = _case_len_from_seq_pt(seq_pt)
            n_crop = min(length, 768)
            bucket = next(
                bucket for bucket, lo, hi, _batches in BUCKETS if lo <= n_crop <= hi
            )
            rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "length_total": length,
                    "n_crop": n_crop,
                    "bucket": bucket,
                    "identity_weight": float(weights[case_id]),
                }
            )
    return rows


def _manifest_train_ids(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, list[str]]:
    manifest = json.loads(manifest_path.read_text())
    return {
        "afdb": list(manifest["datasets"]["afdb"]["train_ids"]),
        "qbiolip": list(manifest["datasets"]["qbiolip"]["train_ids"]),
    }


def _bucket_boundaries() -> list[dict[str, Any]]:
    return [
        {"bucket": bucket, "min_n_crop": lo, "max_n_crop": hi, "n_pad": hi}
        for bucket, lo, hi, _batches in BUCKETS
    ]


def build_seq_len_bin_payload(
    rows: Iterable[dict[str, Any]],
    *,
    expected_train_ids: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    rows = sorted(list(rows), key=lambda row: (str(row["dataset"]), str(row["case_id"])))
    bucket_to_case_ids = {bucket: [] for bucket, _lo, _hi, _batches in BUCKETS}
    case_to_label: dict[str, dict[str, Any]] = {}
    raw_length_weights: dict[str, float] = {}
    bucket_lookup = {bucket: (lo, hi) for bucket, lo, hi, _batches in BUCKETS}

    for row in rows:
        bucket = str(row["bucket"])
        if bucket not in bucket_lookup:
            raise ValueError(f"unknown bucket for {row['case_id']}: {bucket}")
        case_id = str(row["case_id"])
        local_batch_size = int(RECOMMENDED_LOCAL_BATCH_BY_BUCKET[bucket])
        raw_length_weight = 1.0 / float(local_batch_size)
        raw_length_weights[case_id] = raw_length_weight
        _lo, n_pad = bucket_lookup[bucket]
        label = {
            "dataset": str(row["dataset"]),
            "case_id": case_id,
            "length_total": int(row["length_total"]),
            "n_crop": int(row["n_crop"]),
            "n_pad": int(n_pad),
            "bucket": bucket,
            "seq_len_bin_label": bucket,
            "local_batch_size": local_batch_size,
            "identity_loss_weight": float(row["identity_weight"]),
            "length_loss_weight_none": 1.0,
            "length_loss_weight_raw_inverse_batch": raw_length_weight,
        }
        case_to_label[case_id] = label
        bucket_to_case_ids[bucket].append(case_id)

    mean_raw_length_weight = (
        float(np.mean(list(raw_length_weights.values()))) if raw_length_weights else 1.0
    )
    if mean_raw_length_weight <= 0.0:
        raise ValueError("mean inverse-batch length weight must be positive")
    for case_id, raw_length_weight in raw_length_weights.items():
        case_to_label[case_id]["length_loss_weight_inverse_batch"] = (
            raw_length_weight / mean_raw_length_weight
        )

    for case_ids in bucket_to_case_ids.values():
        case_ids.sort()

    present_by_dataset: dict[str, set[str]] = {"afdb": set(), "qbiolip": set()}
    for row in rows:
        present_by_dataset.setdefault(str(row["dataset"]), set()).add(str(row["case_id"]))
    expected_train_ids = expected_train_ids or {
        "afdb": sorted(present_by_dataset.get("afdb", set())),
        "qbiolip": sorted(present_by_dataset.get("qbiolip", set())),
    }
    missing_seq_pt_ids = {
        dataset: [
            case_id
            for case_id in expected_train_ids.get(dataset, [])
            if case_id not in present_by_dataset.get(dataset, set())
        ]
        for dataset in ("afdb", "qbiolip")
    }
    bucket_counts = {
        bucket: len(case_ids) for bucket, case_ids in bucket_to_case_ids.items()
    }
    dataset_counts = {
        dataset: sum(1 for row in rows if str(row["dataset"]) == dataset)
        for dataset in ("afdb", "qbiolip")
    }
    mean_identity_weight = (
        float(np.mean([float(row["identity_weight"]) for row in rows])) if rows else 1.0
    )
    mean_length_weight = (
        float(
            np.mean(
                [
                    float(label["length_loss_weight_inverse_batch"])
                    for label in case_to_label.values()
                ]
            )
        )
        if case_to_label
        else 1.0
    )

    return {
        "schema": "physppi_seq_len_bin_labels_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(DEFAULT_MANIFEST),
        "source_precomputed_root": str(DEFAULT_PRECOMPUTED_ROOT),
        "source_sampling_weights": str(DEFAULT_WEIGHTS),
        "bin_boundaries": _bucket_boundaries(),
        "recommended_local_batch_by_bucket": dict(RECOMMENDED_LOCAL_BATCH_BY_BUCKET),
        "length_loss_weight_modes": {
            "none": "always 1.0",
            "inverse_batch_size": "normalized 1 / recommended_local_batch_by_bucket",
        },
        "case_to_label": case_to_label,
        "bucket_to_case_ids": bucket_to_case_ids,
        "missing_seq_pt_ids": missing_seq_pt_ids,
        "summary": {
            "num_cases": len(rows),
            "num_expected_train_ids": sum(len(v) for v in expected_train_ids.values()),
            "num_missing_seq_pt": sum(len(v) for v in missing_seq_pt_ids.values()),
            "bucket_counts": bucket_counts,
            "dataset_counts": dataset_counts,
            "mean_identity_loss_weight": mean_identity_weight,
            "mean_length_loss_weight_inverse_batch": mean_length_weight,
            "mean_raw_inverse_batch_weight": mean_raw_length_weight,
        },
    }


def write_seq_len_bin_labels(
    *,
    output_path: Path = DEFAULT_SEQ_LEN_BIN_LABELS,
) -> dict[str, Any]:
    rows = build_case_table()
    payload = build_seq_len_bin_payload(
        rows,
        expected_train_ids=_manifest_train_ids(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def prepare_fixed_cases(
    *,
    output_path: Path = DEFAULT_CASES_JSON,
    cases_per_dataset_bucket: int = 4,
) -> dict[str, Any]:
    table = build_case_table()
    selected: dict[str, dict[str, list[str]]] = {}
    for bucket, _lo, _hi, _batches in BUCKETS:
        selected[bucket] = {}
        for dataset in ("afdb", "qbiolip"):
            candidates = [row for row in table if row["bucket"] == bucket and row["dataset"] == dataset]
            candidates.sort(key=lambda row: (abs(row["n_crop"] - _hi), row["case_id"]))
            selected[bucket][dataset] = [
                row["case_id"] for row in candidates[:cases_per_dataset_bucket]
            ]
    payload = {
        "schema": "physppi_8gpu_v2_benchmark_cases_v1",
        "cases_per_dataset_bucket": cases_per_dataset_bucket,
        "buckets": selected,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _write_candidate_yaml(
    *,
    candidate: Candidate,
    base_yaml: Path,
    output_dir: Path,
    cases_json: Path,
    max_steps: int,
) -> Path:
    data = yaml.safe_load(base_yaml.read_text())
    data["experiment_settings"]["output_dir"] = str(output_dir)
    data["experiment_settings"]["restart_checkpoint_path"] = None
    data["experiment_settings"]["ckpt_load_settings"] = {
        "manual_checkpoint_loading": False,
        "init_from_ema_weights": False,
        "restore_lr_scheduler": False,
        "restore_time_step": False,
        "strict_loading": True,
    }
    data["data_module_args"]["batch_size"] = int(candidate.batch_size)
    data["data_module_args"]["global_datapoint_sampling"] = False
    data["pl_trainer_args"]["devices"] = 8
    data["pl_trainer_args"]["max_steps"] = int(max_steps)
    data["pl_trainer_args"]["max_epochs"] = 1
    data["data_module_args"]["epoch_len"] = max(
        8,
        int(max_steps)
        * int(data["pl_trainer_args"]["devices"])
        * int(candidate.batch_size),
    )
    data["pl_trainer_args"]["val_check_interval"] = 1
    data["pl_trainer_args"]["limit_val_batches"] = 0
    data["pl_trainer_args"]["num_sanity_val_steps"] = 0
    settings = data["model_update"]["custom"].setdefault("settings", {})
    settings["gradient_clipping"] = {"per_sample_clipping": False}
    data["checkpoint_config"]["save_last"] = False
    data["checkpoint_config"]["save_top_k"] = 0
    data["checkpoint_config"]["every_n_train_steps"] = 0
    data["checkpoint_config"]["every_n_epochs"] = 0
    data["checkpoint_config"]["dirpath"] = str(output_dir / "checkpoint")

    cases = json.loads(cases_json.read_text())["buckets"][candidate.bucket]
    train_cfg = data["dataset_configs"]["train"]
    for name, dataset_key in (("qbiolip_train", "qbiolip"), ("afdb_train", "afdb")):
        physppi = train_cfg[name]["config"]["physppi"]
        physppi.pop("manifest", None)
        physppi.pop("sampling_weights", None)
        physppi.pop("manifest_dataset_key", None)
        physppi.pop("manifest_partition", None)
        physppi["sample_ids"] = cases.get(dataset_key, [])
        physppi["max_samples"] = len(physppi["sample_ids"])
        train_cfg[name]["config"]["crop"]["token_crop"]["token_budget"] = int(candidate.n_pad)

    # Avoid validation during benchmark candidates.
    data["dataset_configs"].pop("validation", None)
    candidate_yaml = output_dir / f"{candidate.name}.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    return candidate_yaml


def build_worker_command(
    *,
    conda_env: Path,
    candidate_yaml: Path,
    output_dir: Path,
    warmup_steps: int,
    measured_steps: int,
    seed: int,
    nproc_per_node: int,
) -> list[str]:
    return [
        str(conda_env / "bin/torchrun"),
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={int(nproc_per_node)}",
        str(V2_DIR / "benchmark_worker.py"),
        "--runner-yaml",
        str(candidate_yaml),
        "--timing-dir",
        str(output_dir),
        "--warmup-steps",
        str(warmup_steps),
        "--measured-steps",
        str(measured_steps),
        "--seed",
        str(seed),
        "--data-seed",
        str(seed),
    ]


def run_candidate(args: argparse.Namespace) -> dict[str, Any]:
    candidate = Candidate(args.bucket, int(args.n_pad), int(args.batch_size))
    output_dir = Path(args.output_dir) / candidate.name
    cases_json = Path(args.cases_json)
    if not cases_json.exists():
        prepare_fixed_cases(output_path=cases_json)
    max_steps = int(args.warmup_steps) + int(args.measured_steps)
    candidate_yaml = _write_candidate_yaml(
        candidate=candidate,
        base_yaml=Path(args.base_yaml),
        output_dir=output_dir,
        cases_json=cases_json,
        max_steps=max_steps,
    )
    for stale_timing in output_dir.glob("benchmark_step_rank-*.jsonl"):
        stale_timing.unlink()
    conda_env = Path(args.conda_env)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    nvidia_lib = conda_env / "lib/python3.10/site-packages/nvidia"
    cue_lib = conda_env / "lib/python3.10/site-packages/cuequivariance_ops/lib"
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(cue_lib),
            str(nvidia_lib / "cublas/lib"),
            str(nvidia_lib / "cuda_runtime/lib"),
            str(nvidia_lib / "cuda_nvrtc/lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    command = build_worker_command(
        conda_env=conda_env,
        candidate_yaml=candidate_yaml,
        output_dir=output_dir,
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(args.measured_steps),
        seed=int(args.seed),
        nproc_per_node=8,
    )
    log_path = output_dir / "benchmark.log"
    start = time.perf_counter()
    status = "ok"
    return_code = 0
    timed_out = False
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(
                timeout=None
                if args.timeout_seconds is None
                else float(args.timeout_seconds)
            )
            return_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
            log.write(f"\nBENCHMARK_TIMEOUT after {args.timeout_seconds} seconds\n")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                proc.wait(timeout=30)
    elapsed = time.perf_counter() - start
    if timed_out:
        status = "timeout"
    elif return_code != 0:
        status = "failed"
    timing_files = sorted(output_dir.glob("benchmark_step_rank-*.jsonl"))
    timing_rows = [
        row
        for timing_file in timing_files
        for row in load_jsonl(timing_file)
    ]
    timing_summary: dict[str, Any] = {}
    timing_error = None
    if status == "ok":
        try:
            timing_summary = summarize_step_timings(
                timing_rows,
                warmup_steps=int(args.warmup_steps),
                measured_steps=int(args.measured_steps),
            )
        except Exception as exc:
            status = "failed"
            timing_error = str(exc)
    row = {
        "candidate": candidate.name,
        "bucket": candidate.bucket,
        "n_pad": candidate.n_pad,
        "batch_size": candidate.batch_size,
        "warmup_steps": int(args.warmup_steps),
        "measured_steps": int(args.measured_steps),
        "median_step_seconds": timing_summary.get("median_step_seconds"),
        "mean_step_seconds": timing_summary.get("mean_step_seconds"),
        "min_step_seconds": timing_summary.get("min_step_seconds"),
        "max_step_seconds": timing_summary.get("max_step_seconds"),
        "wall_seconds": elapsed,
        "peak_allocated_gb": timing_summary.get("peak_allocated_gb"),
        "peak_reserved_gb": timing_summary.get("peak_reserved_gb"),
        "token_count": timing_summary.get("token_count"),
        "atom_count": timing_summary.get("atom_count"),
        "status": status,
        "return_code": return_code,
        "timing_error": timing_error,
        "timing_files": [str(path) for path in timing_files],
        "yaml": str(candidate_yaml),
        "log": str(log_path),
    }
    append_jsonl(Path(args.results_jsonl), row)
    return row


def summarize(
    *,
    measurements_path: Path = DEFAULT_RESULTS_JSONL,
    output_path: Path = DEFAULT_SUMMARY_JSON,
    target_seconds: float | None = None,
    memory_limit: float | None = None,
    max_batch_size: int = 8,
) -> dict[str, Any]:
    measurements = load_jsonl(measurements_path)
    time_fit = fit_time_model(measurements)
    b1_length_fit = fit_b1_length_model(measurements)
    memory_fit = fit_memory_model(measurements)
    target = target_seconds
    if target is None:
        baseline = [
            row
            for row in measurements
            if row.get("status", "ok") == "ok"
            and int(row.get("n_pad", 0)) == 768
            and int(row.get("batch_size", 0)) == 1
        ]
        target = float(baseline[0]["median_step_seconds"]) if baseline else 1.0
    representative_atom_counts: dict[int, float] = {}
    for row in measurements:
        if row.get("status", "ok") != "ok" or row.get("atom_count") is None:
            continue
        n_pad = int(row["n_pad"])
        representative_atom_counts[n_pad] = max(
            representative_atom_counts.get(n_pad, 0.0),
            float(row["atom_count"]),
        )
    recommendations = []
    if time_fit["full_rank"]:
        recommendations = recommend_batch_sizes(
            n_pads=[256, 384, 512, 640, 768],
            time_fit=time_fit,
            memory_fit=memory_fit,
            target_seconds=float(target),
            memory_limit=memory_limit,
            max_batch_size=max_batch_size,
            representative_atom_counts=representative_atom_counts,
        )
    payload = {
        "time_fit": time_fit,
        "b1_length_fit": b1_length_fit,
        "memory_fit": memory_fit,
        "target_seconds": target,
        "memory_limit": memory_limit,
        "representative_atom_counts": representative_atom_counts,
        "recommendations": recommendations,
        "measurements": measurements,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| `N_pad` | recommended B | time-limited B | memory-limited B | predicted sec | predicted GB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gb = row["predicted_peak_reserved_gb"]
        lines.append(
            "| {n_pad} | {recommended_batch_size} | {time_limited_batch_size} | "
            "{memory_limited_batch_size} | {sec:.3f} | {gb} |".format(
                n_pad=row["n_pad"],
                recommended_batch_size=row["recommended_batch_size"],
                time_limited_batch_size=row["time_limited_batch_size"],
                memory_limited_batch_size=row["memory_limited_batch_size"],
                sec=row["predicted_seconds"],
                gb="n/a" if gb is None else f"{gb:.3f}",
            )
        )
    return "\n".join(lines)


def write_readme_section(summary_path: Path, readme_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    full_rank = bool(summary["time_fit"]["full_rank"])
    time_fit_note = (
        "full-rank fit available"
        if full_rank
        else "full-rank fit unavailable: all successful runs currently have B=1"
    )
    if full_rank:
        fit_lines = [
            f"t0 = {summary['time_fit']['t0']:.6g}",
            f"a0 = {summary['time_fit']['a0']:.6g}",
            f"a1 = {summary['time_fit']['a1']:.6g}",
            f"a2 = {summary['time_fit']['a2']:.6g}",
            f"time R2 = {summary['time_fit']['r2']:.6g}",
            f"design rank = {summary['time_fit']['rank']:.0f}/4",
        ]
    else:
        fit_lines = [
            "Full B,L model is not identifiable from the current ok rows.",
            f"design rank = {summary['time_fit']['rank']:.0f}/4",
            "Do not use the minimum-norm t0/a0/a1/a2 values as a batch plan.",
        ]
    fit_lines.extend(
        [
            "",
            "B=1 identifiable length fit:",
            f"b0 = {summary['b1_length_fit']['b0']:.6g}",
            f"b1 = {summary['b1_length_fit']['b1']:.6g}",
            f"b2 = {summary['b1_length_fit']['b2']:.6g}",
            f"B=1 R2 = {summary['b1_length_fit']['r2']:.6g}",
        ]
    )
    ok_measurements = [
        row for row in summary["measurements"] if row.get("status") == "ok"
    ]
    measurement_lines = [
        "| candidate | `N_pad` | B | median sec | peak reserved GB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ok_measurements:
        measurement_lines.append(
            "| {candidate} | {n_pad} | {batch_size} | {sec:.3f} | {gb:.3f} |".format(
                candidate=row["candidate"],
                n_pad=row["n_pad"],
                batch_size=row["batch_size"],
                sec=row["median_step_seconds"],
                gb=row["peak_reserved_gb"],
            )
        )
    recommendation_block = (
        _markdown_table(summary["recommendations"])
        if summary["recommendations"]
        else "\n".join(
            [
                "当前没有 full-rank `B,L` 测量矩阵；不生成正式动态 batch size 表。",
                "",
                "B>1 diagnostic: `001_256_B2` 在当前数据管线中失败，latest blocker 是 "
                "collator 无法 stack batch 内第三维不同的 PhysPPI tensor "
                "（例如 `[256, 256, 5]` vs `[256, 256, 7]`）。benchmark worker "
                "已经为测量目的绕过了 sampler trace、per-sample clipping、DockQ target "
                "和 step metrics 的单样本限制；正式训练要支持 B>1，仍需要把这些路径改成 "
                "batch-aware。",
            ]
        )
    )
    section = "\n".join(
        [
            "## 14. Benchmark 拟合结果",
            "",
            "由 `benchmark_runner.py summarize` 生成。",
            "",
            f"Full model status: {time_fit_note}.",
            "",
            "```text",
            *fit_lines,
            "```",
            "",
            "\n".join(measurement_lines),
            "",
            recommendation_block,
            "",
        ]
    )
    marker = "\n## 14. Benchmark 拟合结果\n"
    text = readme_path.read_text()
    if marker in text:
        text = text[: text.index(marker)] + "\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    readme_path.write_text(text.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("matrix")

    prep = sub.add_parser("prepare-cases")
    prep.add_argument("--output", default=str(DEFAULT_CASES_JSON))
    prep.add_argument("--cases-per-dataset-bucket", type=int, default=4)

    labels = sub.add_parser("write-bin-labels")
    labels.add_argument("--output", default=str(DEFAULT_SEQ_LEN_BIN_LABELS))

    run = sub.add_parser("run-candidate")
    run.add_argument("--bucket", required=True)
    run.add_argument("--n-pad", type=int, required=True)
    run.add_argument("--batch-size", type=int, required=True)
    run.add_argument("--warmup-steps", type=int, default=2)
    run.add_argument("--measured-steps", type=int, default=5)
    run.add_argument("--base-yaml", default=str(DEFAULT_BASE_YAML))
    run.add_argument("--output-dir", default=str(V2_DIR / "benchmark_runs"))
    run.add_argument("--cases-json", default=str(DEFAULT_CASES_JSON))
    run.add_argument("--results-jsonl", default=str(DEFAULT_RESULTS_JSONL))
    run.add_argument("--conda-env", default=str(DEFAULT_CONDA_ENV))
    run.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--timeout-seconds", type=float, default=None)

    summ = sub.add_parser("summarize")
    summ.add_argument("--measurements", default=str(DEFAULT_RESULTS_JSONL))
    summ.add_argument("--output", default=str(DEFAULT_SUMMARY_JSON))
    summ.add_argument("--target-seconds", type=float, default=None)
    summ.add_argument("--memory-limit", type=float, default=None)
    summ.add_argument("--max-batch-size", type=int, default=8)
    summ.add_argument("--readme", default=None)

    args = parser.parse_args()
    if args.cmd == "matrix":
        print(json.dumps([candidate.__dict__ | {"name": candidate.name} for candidate in candidate_matrix()], indent=2))
    elif args.cmd == "prepare-cases":
        payload = prepare_fixed_cases(
            output_path=Path(args.output),
            cases_per_dataset_bucket=args.cases_per_dataset_bucket,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.cmd == "write-bin-labels":
        payload = write_seq_len_bin_labels(output_path=Path(args.output))
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    elif args.cmd == "run-candidate":
        print(json.dumps(run_candidate(args), indent=2, sort_keys=True))
    elif args.cmd == "summarize":
        payload = summarize(
            measurements_path=Path(args.measurements),
            output_path=Path(args.output),
            target_seconds=args.target_seconds,
            memory_limit=args.memory_limit,
            max_batch_size=args.max_batch_size,
        )
        if args.readme:
            write_readme_section(Path(args.output), Path(args.readme))
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
