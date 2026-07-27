#!/usr/bin/env python3
# ruff: noqa: E402
"""Screen both canonical PhysPPI sources through the production DataLoader."""

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
from pathlib import Path

from openfold3.projects.of3_all_atom.physppi.screening.dataset import (
    load_cached_source_ids,
)
from openfold3.projects.of3_all_atom.physppi.screening.history import (
    load_historical_seeds,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    ScreeningOptions,
)
from openfold3.projects.of3_all_atom.physppi.screening.preflight import (
    preflight_audit_view,
)
from openfold3.projects.of3_all_atom.physppi.screening.runner import run_screening


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen exactly 10,000 Q-BioLiP and 10,000 AFDB IDs through "
            "seven deterministic production feature views."
        )
    )
    parser.add_argument("--qbiolip-root", type=Path, required=True)
    parser.add_argument("--afdb-root", type=Path, required=True)
    parser.add_argument("--qbiolip-index-cache", type=Path, required=True)
    parser.add_argument("--afdb-index-cache", type=Path, required=True)
    parser.add_argument("--qbiolip-canonical-report", type=Path, required=True)
    parser.add_argument("--afdb-canonical-report", type=Path, required=True)
    parser.add_argument("--history-runs-root", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--runtime-ledger", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--components-cif", type=Path, required=True)
    parser.add_argument("--reference-molecule-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(8,), default=8)
    parser.add_argument("--sample-timeout-seconds", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    return parser


def _require_resources(options: ScreeningOptions) -> None:
    directories = (
        options.qbiolip_root,
        options.afdb_root,
        options.history_runs_root,
        options.manifests_dir,
        options.reference_molecule_directory,
    )
    files = (
        options.qbiolip_index_cache,
        options.afdb_index_cache,
        options.qbiolip_canonical_report,
        options.afdb_canonical_report,
        options.runtime_ledger,
        options.prior_audit,
        options.components_cif,
    )
    missing = [str(path) for path in (*directories, *files) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing explicit screening resources: " + ", ".join(missing)
        )
    not_directories = [str(path) for path in directories if not path.is_dir()]
    not_files = [str(path) for path in files if not path.is_file()]
    if not_directories or not_files:
        raise ValueError(
            "invalid explicit resource types: " + ", ".join(not_directories + not_files)
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = ScreeningOptions(
        **vars(args),
        max_attempts=2,
        crop_strategies=("spatial", "spatial_interface"),
    )
    _require_resources(options)
    source_ids = {
        DatasetName.QBIOLIP: load_cached_source_ids(
            DatasetName.QBIOLIP, options.qbiolip_index_cache
        ),
        DatasetName.AFDB: load_cached_source_ids(
            DatasetName.AFDB, options.afdb_index_cache
        ),
    }
    historical_seeds = load_historical_seeds(
        options.history_runs_root,
        options.manifests_dir,
        options.runtime_ledger,
        options.prior_audit,
    )
    summary = run_screening(
        options,
        source_ids=source_ids,
        historical_seeds=historical_seeds,
        preflight=preflight_audit_view,
    )
    return 0 if summary.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
