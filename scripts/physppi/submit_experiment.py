#!/usr/bin/env python3
# ruff: noqa: E402
"""Prepare a PhysPPI Slurm run and submit it only when explicitly requested."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from scripts.physppi._bootstrap import ensure_repository_root
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import ensure_repository_root

ensure_repository_root()

import argparse
from collections.abc import Sequence

from scripts.physppi.experiment_config import load_experiment
from scripts.physppi.experiment_slurm import git_state, prepare_run, submit_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--local-config", type=Path)
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_experiment(
        args.config,
        repo_root=repo_root,
        local_path=args.local_config,
    )
    git_commit, git_dirty = git_state(repo_root)
    prepared = prepare_run(
        loaded,
        repo_root=repo_root,
        python_executable=Path(sys.executable),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    print(prepared.script_path)
    if args.submit:
        print(submit_run(prepared))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
