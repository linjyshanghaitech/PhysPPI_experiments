"""Summarize baseline inference results from a PhysPPI test set run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> tuple[int, int, int, str | None]:
    if not path.exists():
        return 0, 0, 0, f"Manifest not found: {path}"

    manifest = json.loads(path.read_text())
    if isinstance(manifest, list):
        requested = len(manifest)
        succeeded = sum(1 for entry in manifest if entry.get("status") == "ok")
        skipped = sum(1 for entry in manifest if entry.get("status") == "skipped")
        return requested, succeeded, skipped, None

    if isinstance(manifest, dict):
        return (
            int(manifest.get("total", 0)),
            int(manifest.get("succeeded", 0)),
            int(manifest.get("skipped", 0)),
            None,
        )

    return 0, 0, 0, f"Unsupported manifest type: {type(manifest).__name__}"


def _load_run_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_query_count(run_info: dict[str, Any]) -> tuple[int | None, str | None]:
    query_path_value = run_info.get("query_json")
    if not query_path_value:
        return None, "run_info.json does not contain query_json"
    query_path = Path(query_path_value)
    if not query_path.exists():
        return None, f"Query JSON not found: {query_path}"
    query = json.loads(query_path.read_text())
    queries = query.get("queries")
    if not isinstance(queries, dict):
        return None, f"Query JSON has no object-valued queries field: {query_path}"
    return len(queries), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested, succeeded, skipped, warning = _load_manifest(args.manifest)

    run_info = _load_run_info(args.run_dir)
    attempted_queries, query_warning = _load_query_count(run_info)

    pred_files: list[Path] = []
    if args.run_dir.exists():
        pred_files = sorted(args.run_dir.rglob("*.pdb")) + sorted(
            args.run_dir.rglob("*.cif")
        )
    elif warning is None:
        warning = f"Run directory not found: {args.run_dir}"

    completed = len(pred_files)

    if attempted_queries is None:
        attempted_queries = succeeded
        if warning is None:
            warning = query_warning
    elif query_warning is not None and warning is None:
        warning = query_warning

    failed = max(attempted_queries - completed, 0) if args.run_dir.exists() else 0

    summary: dict[str, Any] = {
        "manifest": str(args.manifest),
        "run_dir": str(args.run_dir),
        "manifest_requested_samples": requested,
        "manifest_ok_samples": succeeded,
        "requested_samples": requested,
        "succeeded": succeeded,
        "skipped": skipped,
        "attempted_queries": attempted_queries,
        "completed_predictions": completed,
        "failed_predictions": failed,
        "warning": warning,
        "prediction_files": [str(path) for path in pred_files],
    }
    summary.update(run_info)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
