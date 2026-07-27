"""Find candidate OpenFold3-ready training data for PhysPPI runs.

This does not convert legacy OpenFold v2 feature caches. It only identifies paths
that look usable by OpenFold3 or worth manual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOTS = [
    Path("/101045/jinyu/PhysPPI/openfold-3/output"),
    Path("/101045/jinyu/PhysPPI/openfold-3/examples"),
    Path("/101045/jinyu/PhysPPI"),
    Path("/shared/openfold3"),
    Path("/shared"),
]

OF3_MARKERS = {
    "training_yaml": ["training.yaml", "initial_training.yml", "physppi_training.yml"],
    "feature_files": [".npz", ".pt", ".pkl", ".pickle"],
    "dataset_dirs": ["processed", "features", "feature_cache", "pdb_data", "mmcif"],
}


def score_path(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if path.is_file() and path.suffix in OF3_MARKERS["feature_files"]:
        score += 2
    if path.is_file() and name in OF3_MARKERS["training_yaml"]:
        score += 4
    if path.is_dir() and name in OF3_MARKERS["dataset_dirs"]:
        score += 2
    if "openfold3" in str(path).lower() or "openfold-3" in str(path).lower():
        score += 1
    return score


def iter_candidates(root: Path, max_entries: int) -> list[dict[str, object]]:
    if not root.exists():
        return []
    candidates: list[dict[str, object]] = []
    for visited, path in enumerate(root.rglob("*"), 1):
        if visited > max_entries:
            break
        score = score_path(path)
        if score > 0:
            candidates.append(
                {
                    "path": str(path),
                    "kind": "dir" if path.is_dir() else "file",
                    "score": score,
                }
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-entries-per-root", type=int, default=20000)
    parser.add_argument("--root", action="append", type=Path, default=[])
    args = parser.parse_args()

    roots = args.root or DEFAULT_ROOTS
    manifest = {
        "roots": [str(root) for root in roots],
        "candidates": [],
        "selected": {
            "single_sample_id": "",
            "single_sample_feature_path": "",
            "full_train_data_path": "",
            "full_train_yaml": "",
            "baseline_inference_ready": True,
            "training_feature_ready": False,
        },
        "notes": [
            "Legacy OpenFold v2 feature caches are not valid OpenFold3 training"
            " data without conversion.",
            "Do not add ESM features; use existing OpenFold3 MSA/profile features.",
            (
                "Baseline inference test set can be built from old"
                " PhysPPI test split PDBs."
            ),
        ],
    }

    for root in roots:
        manifest["candidates"].extend(
            iter_candidates(root=root, max_entries=args.max_entries_per_root)
        )

    manifest["candidates"].sort(key=lambda item: (-int(item["score"]), item["path"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={args.out}")
    print(f"candidates={len(manifest['candidates'])}")


if __name__ == "__main__":
    main()
