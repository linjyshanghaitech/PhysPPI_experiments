#!/usr/bin/env python3
"""Run MMseqs2 easy-cluster at 30% and 40% identity for cli_v4 split."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_mmseqs(
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    *,
    executable: str = "mmseqs",
    threads: int = 16,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    cov_mode: int = 0,
    cluster_mode: int = 2,
) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for stale in output_prefix.parent.glob(output_prefix.name + "*"):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()

    cmd = [
        executable,
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--cluster-mode", str(cluster_mode),
        "--threads", str(threads),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    cluster_tsv = output_prefix.with_name(output_prefix.name + "_cluster.tsv")
    if not cluster_tsv.is_file():
        raise RuntimeError(f"MMseqs2 did not produce {cluster_tsv}")
    return cluster_tsv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        default=os.environ.get("MMSEQS", "mmseqs"),
    )
    parser.add_argument("--threads", type=int, default=min(32, (os.cpu_count() or 1)))
    parser.add_argument("--coverage", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for min_seq_id, label in [(0.3, "30"), (0.4, "40")]:
        out_prefix = args.output_dir / f"mmseqs_{label}" / "clusters"
        tmp_dir = args.output_dir / f"mmseqs_{label}" / "tmp"
        print(f"\n=== Clustering at {min_seq_id * 100:.0f}% identity ===")
        cluster_tsv = run_mmseqs(
            fasta_path=args.fasta,
            output_prefix=out_prefix,
            tmp_dir=tmp_dir,
            executable=args.mmseqs,
            threads=args.threads,
            min_seq_id=min_seq_id,
            coverage=args.coverage,
        )
        print(f"  -> {cluster_tsv}")
    print(f"\nDone. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
