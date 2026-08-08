#!/usr/bin/env python3
"""Run the full identity-aware train/val split pipeline for cli_v4."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 70}\n")
    subprocess.run(cmd, check=True, cwd=str(HERE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=min(32, (os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--mmseqs", type=str, default=os.environ.get("MMSEQS", "mmseqs"))
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    fasta_path = HERE / "sequences.fasta"
    chain_index_path = HERE / "chain_index.json"
    cluster_tsv_30 = HERE / "mmseqs_30" / "clusters_cluster.tsv"
    cluster_tsv_40 = HERE / "mmseqs_40" / "clusters_cluster.tsv"

    if not args.skip_extract:
        if chain_index_path.is_file() and fasta_path.is_file():
            print("[skip] Step 1: sequences already extracted")
        else:
            run_step("Step 1: Extract chain sequences", [
                sys.executable,
                str(HERE / "extract_sequences_v4.py"),
                "--output-dir", str(HERE),
            ])

    if not args.skip_cluster:
        if cluster_tsv_30.is_file() and cluster_tsv_40.is_file():
            print("[skip] Step 2: clusters already exist")
        else:
            run_step("Step 2: MMseqs2 clustering (30% + 40%)", [
                sys.executable,
                str(HERE / "run_mmseqs_v4.py"),
                "--fasta", str(fasta_path),
                "--output-dir", str(HERE),
                "--mmseqs", args.mmseqs,
                "--threads", str(args.threads),
            ])

    if not args.skip_split:
        print("[info] Step 3 will always re-compute the split (idempotent, fast)")
        run_step("Step 3: Build homology-aware split", [
            sys.executable,
            str(HERE / "build_split_v4.py"),
            "--output-dir", str(HERE),
            "--final-dir", str(args.final_dir),
            "--seed", str(args.seed),
            "--val-ratio", str(args.val_ratio),
        ])

    print(f"\n{'=' * 70}")
    print("  Pipeline complete.")
    print(f"  Output:  {HERE}/split_manifest.json")
    print(f"  Output:  {HERE}/sampling_weights.pt")
    print(f"  Output:  {HERE}/split_report.json")
    print(f"  Copied:  {args.final_dir}/split_manifest.json")
    print(f"  Copied:  {args.final_dir}/sampling_weights.pt")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
