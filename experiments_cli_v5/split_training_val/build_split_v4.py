#!/usr/bin/env python3
"""Build chain-level homology-aware train/val split for cli_v4 (AF-Multimer protocol).

Algorithm is identical to v3:
1. Parse MMseqs2 30% cluster TSV -> per-chain membership.
2. Build case-level homology graph (shared 30% cluster => edge).
3. Find connected components; greedy largest-first into val until ~20%.
4. Parse 40% clusters -> per-case sampling weights (1/max cluster size, mean=1).
5. Write split_manifest.json, sampling_weights.pt, train/val IDs, split_report.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_cluster_tsv(
    cluster_tsv: Path,
    chain_index: dict[str, dict[str, str]],
) -> dict[str, str]:
    valid_ids = set(chain_index)
    memberships: dict[str, str] = {}
    with cluster_tsv.open("r", encoding="ascii") as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            if rep not in valid_ids:
                raise ValueError(f"unknown representative in cluster TSV: {rep}")
            if member not in valid_ids:
                raise ValueError(f"unknown member in cluster TSV: {member}")
            if member in memberships:
                raise ValueError(f"chain assigned to multiple clusters: {member}")
            memberships[member] = rep

    clustered = set(memberships)
    for seq_id in sorted(valid_ids - clustered):
        memberships[seq_id] = seq_id
    for rep in set(memberships.values()):
        if rep not in clustered:
            memberships[rep] = rep
    return memberships


def build_homology_components(
    chain_memberships_30: dict[str, str],
    case_chains: dict[str, list[str]],
) -> list[set[str]]:
    cluster_to_cases: dict[str, set[str]] = defaultdict(set)
    for case_id, chain_ids in case_chains.items():
        for chain_id in chain_ids:
            seq_id = f"{case_id}|{chain_id}"
            cluster_rep = chain_memberships_30.get(seq_id)
            if cluster_rep is not None:
                cluster_to_cases[cluster_rep].add(case_id)

    parent: dict[str, str] = {}
    all_cases = set(case_chains)
    for case_id in all_cases:
        parent[case_id] = case_id

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cluster_rep, cases in cluster_to_cases.items():
        cases_list = sorted(cases)
        if len(cases_list) < 2:
            continue
        first = cases_list[0]
        for other in cases_list[1:]:
            union(first, other)

    components: dict[str, set[str]] = defaultdict(set)
    for case_id in all_cases:
        components[find(case_id)].add(case_id)
    return sorted(components.values(), key=len, reverse=True)


def split_components(
    components: list[set[str]],
    target_val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str], dict[str, Any]]:
    import random

    total = sum(len(c) for c in components)
    target_val = int(total * target_val_ratio)

    by_size: dict[int, list[set[str]]] = defaultdict(list)
    for comp in components:
        by_size[len(comp)].append(comp)

    rng = random.Random(seed)
    ordered: list[set[str]] = []
    for size in sorted(by_size, reverse=True):
        group = by_size[size]
        rng.shuffle(group)
        ordered.extend(group)

    val_cases: set[str] = set()
    val_count = 0
    for comp in ordered:
        if val_count + len(comp) <= target_val:
            val_cases.update(comp)
            val_count += len(comp)

    train_cases = set()
    for comp in components:
        if not (comp & val_cases):
            train_cases.update(comp)

    actual_val_ratio = len(val_cases) / total if total else 0.0
    stats = {
        "total_cases": total,
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "train_ratio": len(train_cases) / total if total else 0.0,
        "val_ratio": actual_val_ratio,
        "connected_components": len(components),
        "components_in_val": sum(1 for c in components if c & val_cases),
        "components_in_train": sum(1 for c in components if c & train_cases),
        "largest_component": max((len(c) for c in components), default=0),
        "singleton_components": sum(1 for c in components if len(c) == 1),
    }
    return train_cases, val_cases, stats


def calculate_sampling_weights(
    chain_memberships_40: dict[str, str],
    case_chains: dict[str, list[str]],
    train_cases: set[str],
) -> dict[str, float]:
    cluster_sizes: dict[str, int] = defaultdict(int)
    for seq_id, rep in chain_memberships_40.items():
        cluster_sizes[rep] += 1

    raw_weights: dict[str, float] = {}
    for case_id, chain_ids in case_chains.items():
        max_size = 1
        for chain_id in chain_ids:
            seq_id = f"{case_id}|{chain_id}"
            rep = chain_memberships_40.get(seq_id, seq_id)
            sz = cluster_sizes.get(rep, 1)
            if sz > max_size:
                max_size = sz
        raw_weights[case_id] = 1.0 / max_size

    train_raw = {cid: raw_weights[cid] for cid in train_cases if cid in raw_weights}
    if not train_raw:
        return {}
    mean_raw = sum(train_raw.values()) / len(train_raw)
    return {cid: raw / mean_raw for cid, raw in train_raw.items()}


def build_split(
    output_dir: Path,
    final_dir: Path,
    *,
    seed: int = 42,
    target_val_ratio: float = 0.2,
    mmseqs_params: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    chain_index = _load_json(output_dir / "chain_index.json")
    case_chains = _load_json(output_dir / "case_chains.json")
    case_chains = {k: list(v) for k, v in case_chains.items()}
    print(f"Loaded {len(case_chains)} cases, {len(chain_index)} chains")

    cluster_tsv_30 = output_dir / "mmseqs_30" / "clusters_cluster.tsv"
    if not cluster_tsv_30.is_file():
        raise FileNotFoundError(f"30% cluster TSV not found: {cluster_tsv_30}")
    chain_memberships_30 = parse_cluster_tsv(cluster_tsv_30, chain_index)
    print(
        f"30% clusters: {len(set(chain_memberships_30.values()))} clusters "
        f"for {len(chain_memberships_30)} chains"
    )

    cluster_tsv_40 = output_dir / "mmseqs_40" / "clusters_cluster.tsv"
    if not cluster_tsv_40.is_file():
        raise FileNotFoundError(f"40% cluster TSV not found: {cluster_tsv_40}")
    chain_memberships_40 = parse_cluster_tsv(cluster_tsv_40, chain_index)
    print(
        f"40% clusters: {len(set(chain_memberships_40.values()))} clusters "
        f"for {len(chain_memberships_40)} chains"
    )

    components = build_homology_components(chain_memberships_30, case_chains)
    print(
        f"Connected components: {len(components)} "
        f"(largest={len(components[0]) if components else 0}, "
        f"singletons={sum(1 for c in components if len(c) == 1)})"
    )

    train_cases, val_cases, split_stats = split_components(
        components, target_val_ratio=target_val_ratio, seed=seed
    )
    print(
        f"Split: train={len(train_cases)} ({split_stats['train_ratio']:.1%}), "
        f"val={len(val_cases)} ({split_stats['val_ratio']:.1%})"
    )

    sampling_weights = calculate_sampling_weights(
        chain_memberships_40, case_chains, train_cases
    )
    print(
        f"Sampling weights for {len(sampling_weights)} training cases "
        f"(min={min(sampling_weights.values()):.4f}, "
        f"max={max(sampling_weights.values()):.4f})"
    )

    dataset_cases: dict[str, list[str]] = defaultdict(list)
    for case_id in case_chains:
        for seq_id, info in chain_index.items():
            if info["case_id"] == case_id:
                dataset_cases[info["dataset"]].append(case_id)
                break

    dataset_splits: dict[str, dict[str, list[str]]] = {}
    for dataset_name in sorted(dataset_cases):
        all_ds = set(dataset_cases[dataset_name])
        dataset_splits[dataset_name] = {
            "train_ids": sorted(all_ds & train_cases),
            "val_ids": sorted(all_ds & val_cases),
        }
        print(
            f"  {dataset_name}: train={len(dataset_splits[dataset_name]['train_ids'])}, "
            f"val={len(dataset_splits[dataset_name]['val_ids'])}"
        )

    chain_clusters_30: dict[str, dict[str, object]] = {}
    cluster_sizes_30: dict[str, int] = defaultdict(int)
    for seq_id, rep in chain_memberships_30.items():
        cluster_sizes_30[rep] += 1
    for seq_id, rep in chain_memberships_30.items():
        chain_clusters_30[seq_id] = {"rep": rep, "size": cluster_sizes_30[rep]}

    chain_clusters_40: dict[str, dict[str, object]] = {}
    cluster_sizes_40: dict[str, int] = defaultdict(int)
    for seq_id, rep in chain_memberships_40.items():
        cluster_sizes_40[rep] += 1
    for seq_id, rep in chain_memberships_40.items():
        chain_clusters_40[seq_id] = {"rep": rep, "size": cluster_sizes_40[rep]}

    case_clusters_30: dict[str, list[str]] = {}
    for case_id, chain_ids in case_chains.items():
        reps: list[str] = []
        for chain_id in chain_ids:
            seq_id = f"{case_id}|{chain_id}"
            rep = chain_memberships_30.get(seq_id, seq_id)
            if rep not in reps:
                reps.append(rep)
        case_clusters_30[case_id] = reps

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_splits,
    }
    _write_json(output_dir / "split_manifest.json", manifest)
    _write_json(final_dir / "split_manifest.json", manifest)

    _write_json(output_dir / "train_ids.json", sorted(train_cases))
    _write_json(output_dir / "val_ids.json", sorted(val_cases))

    mmseqs_params_filled = mmseqs_params or {
        "30": {"min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 0, "cluster_mode": 2},
        "40": {"min_seq_id": 0.4, "coverage": 0.8, "cov_mode": 0, "cluster_mode": 2},
    }
    pt_payload = {
        "schema": "physppi_identity_split_v1",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_splits,
        "sampling_weights": sampling_weights,
        "chain_clusters_30": chain_clusters_30,
        "chain_clusters_40": chain_clusters_40,
        "case_clusters_30": case_clusters_30,
        "mmseqs_params": mmseqs_params_filled,
        "split_stats": split_stats,
    }
    pt_path = output_dir / "sampling_weights.pt"
    torch.save(pt_payload, pt_path)
    final_pt_path = final_dir / "sampling_weights.pt"
    torch.save(pt_payload, final_pt_path)
    print(f"Saved sampling_weights.pt -> {final_pt_path}")

    report = {
        **split_stats,
        "per_dataset": {
            ds: {
                "train": len(info["train_ids"]),
                "val": len(info["val_ids"]),
            }
            for ds, info in dataset_splits.items()
        },
        "cluster_stats_30": {
            "n_clusters": len(set(chain_memberships_30.values())),
            "max_size": max(cluster_sizes_30.values()) if cluster_sizes_30 else 0,
            "mean_size": (
                sum(cluster_sizes_30.values())
                / len(set(chain_memberships_30.values()))
                if cluster_sizes_30
                else 0
            ),
        },
        "cluster_stats_40": {
            "n_clusters": len(set(chain_memberships_40.values())),
            "max_size": max(cluster_sizes_40.values()) if cluster_sizes_40 else 0,
            "mean_size": (
                sum(cluster_sizes_40.values())
                / len(set(chain_memberships_40.values()))
                if cluster_sizes_40
                else 0
            ),
        },
        "weight_stats": {
            "min": min(sampling_weights.values()) if sampling_weights else 0,
            "max": max(sampling_weights.values()) if sampling_weights else 0,
            "mean": (
                sum(sampling_weights.values()) / len(sampling_weights)
                if sampling_weights
                else 0
            ),
        },
    }
    _write_json(output_dir / "split_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_split(
        output_dir=args.output_dir,
        final_dir=args.final_dir,
        seed=args.seed,
        target_val_ratio=args.val_ratio,
    )
    print("\n=== Split Report ===")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
