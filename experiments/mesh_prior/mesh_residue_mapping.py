#!/usr/bin/env python3
"""Mesh-to-residue mapping via KDTree nearest-heavy-atom assignment.

Reads v3-format prior PKL files (containing mesh vertex data per chain) and
GT PDB files, then assigns each mesh vertex to the residue of its nearest
heavy atom via scipy.spatial.cKDTree.  Reports the distribution of assigned
mesh vertices per residue to help choose the K (neighbourhood size) for the
new surface representation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# --- Constants ---------------------------------------------------------------

HYDROGEN_ELEMENTS = {"H", "D", "T"}
VERTEX_DIMS = 10  # x, y, z, charge, hbond, hphob, nx, ny, nz, si
OUTPUT_DIMS = 9  # drop hbond: x, y, z, charge, hphob, nx, ny, nz, si
HOH_RESNAME = "HOH"


# --- PDB parsing -------------------------------------------------------------


def parse_gt_heavy_atoms(pdb_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse a GT PDB, returning (coords [M,3], res_ids [M,]) per chain.

    Only non-hydrogen, non-HOH atoms are included.
    """
    chain_atoms: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    chain_residues: dict[str, list[int]] = defaultdict(list)

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 55:
                continue
            resname = line[17:20].strip().upper()
            if resname == HOH_RESNAME:
                continue
            element = (
                line[76:78].strip().upper()
                if len(line) >= 78
                else line[13:14].strip().upper()
            )
            if element in HYDROGEN_ELEMENTS:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            chain_id = line[21].strip()
            if not chain_id:
                continue
            chain_atoms[chain_id].append((x, y, z))
            chain_residues[chain_id].append(resseq)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chain_id in chain_atoms:
        coords = np.array(chain_atoms[chain_id], dtype=np.float32)
        residues = np.array(chain_residues[chain_id], dtype=np.int32)
        result[chain_id] = (coords, residues)
    return result


# --- Mesh loading ------------------------------------------------------------


def load_mesh_vertices(pkl_path: Path) -> dict[str, list[list[float]]]:
    """Load per-chain mesh vertices from a v3-format prior PKL."""
    payload = pickle.loads(pkl_path.read_bytes())
    # v3 PKL: top-level key is case_id → SimpleNamespace with .mesh
    case_id = next(iter(payload))
    obj = payload[case_id]
    mesh = getattr(obj, "mesh", None)
    if mesh is None:
        raise KeyError(f"no 'mesh' field in {pkl_path}")
    if not isinstance(mesh, dict):
        raise TypeError(f"'mesh' is not dict in {pkl_path}: {type(mesh)}")
    return {str(chain): list(vertices) for chain, vertices in mesh.items()}


# --- KDTree mapping ----------------------------------------------------------


def map_mesh_to_residues(
    mesh_chain_vertices: list[list[float]],
    heavy_coords: np.ndarray,
    heavy_residues: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map mesh vertices to residues via nearest-heavy-atom KDTree.

    Returns (vertex_residue_ids, vertex_coords_9d).
    """
    if heavy_coords.size == 0:
        return np.array([], dtype=np.int32), np.empty((0, OUTPUT_DIMS), dtype=np.float32)

    vertices = np.array(mesh_chain_vertices, dtype=np.float32)
    if vertices.size == 0:
        return np.array([], dtype=np.int32), np.empty((0, OUTPUT_DIMS), dtype=np.float32)

    vertex_xyz = vertices[:, :3]
    tree = cKDTree(heavy_coords)
    _, atom_idx = tree.query(vertex_xyz, k=1)

    residue_ids = heavy_residues[atom_idx]

    # Build 9-dim output: drop hbond (column index 4)
    # Original: x,y,z(0:3), charge(3), hbond(4), hphob(5), nx,ny,nz(6:9), si(9)
    # Output:   x,y,z(0:3), charge(3), hphob(4), nx,ny,nz(5:8), si(8)
    keep_indices = [0, 1, 2, 3, 5, 6, 7, 8, 9]
    vertex_9d = vertices[:, keep_indices]

    return residue_ids, vertex_9d


# --- Residue-level aggregation -----------------------------------------------


def aggregate_by_residue(
    residue_ids: np.ndarray,
    vertex_data: np.ndarray,
    chain_id: str,
) -> dict[int, np.ndarray]:
    """Group mesh vertices by their assigned residue.

    Returns {resseq: np.ndarray[N_vertices, 9]}.
    """
    result: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, resseq in enumerate(residue_ids):
        result[int(resseq)].append(vertex_data[i])
    return {res: np.stack(verts, axis=0) for res, verts in result.items()}


# --- Main audit --------------------------------------------------------------


def audit_mesh_distribution(
    dataset_name: str,
    chunk_dir: Path,
    max_samples: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Audit mesh-to-residue mapping for one dataset chunk tree.

    Returns (report, per_residue_counts_array).
    """
    prior_pkl_dir = chunk_dir / "prior_pkl"
    bb_input_dir = chunk_dir / "bb_input"

    pkl_paths = sorted(prior_pkl_dir.glob("*.pkl"))
    if max_samples is not None:
        pkl_paths = pkl_paths[:max_samples]

    all_residue_counts: list[int] = []  # vertices per residue across all samples
    max_counts_per_sample: list[int] = []
    sample_details: list[dict] = []
    skipped = 0

    for pkl_path in pkl_paths:
        case_id = pkl_path.stem

        try:
            mesh_by_chain = load_mesh_vertices(pkl_path)
            # Load heavy atoms from per-chain bb_input PDBs (same coords as mesh)
            heavy_by_chain: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for chain_id in mesh_by_chain:
                chain_pdb = bb_input_dir / f"{case_id}_{chain_id}.pdb"
                if chain_pdb.exists():
                    heavy_by_chain.update(parse_gt_heavy_atoms(chain_pdb))
        except Exception as exc:
            print(f"  [{dataset_name}] SKIP {case_id}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        sample_residue_counts: list[int] = []
        sample_chain_details: dict = {}

        for chain_id in sorted(mesh_by_chain):
            if chain_id not in heavy_by_chain:
                continue

            mesh_verts = mesh_by_chain[chain_id]
            heavy_coords, heavy_residues = heavy_by_chain[chain_id]

            if not mesh_verts or heavy_coords.size == 0:
                continue

            residue_ids, vertex_9d = map_mesh_to_residues(
                mesh_verts, heavy_coords, heavy_residues
            )
            by_residue = aggregate_by_residue(residue_ids, vertex_9d, chain_id)

            chain_counts = [v.shape[0] for v in by_residue.values()]
            sample_residue_counts.extend(chain_counts)
            sample_chain_details[chain_id] = {
                "n_vertices": len(mesh_verts),
                "n_residues_assigned": len(by_residue),
                "max_per_residue": max(chain_counts) if chain_counts else 0,
                "min_per_residue": min(chain_counts) if chain_counts else 0,
                "mean_per_residue": (
                    float(np.mean(chain_counts)) if chain_counts else 0.0
                ),
                "residues_over_64": sum(1 for c in chain_counts if c > 64),
                "residues_over_128": sum(1 for c in chain_counts if c > 128),
            }

        all_residue_counts.extend(sample_residue_counts)
        max_counts_per_sample.append(
            max(sample_residue_counts) if sample_residue_counts else 0
        )
        sample_details.append(
            {
                "case_id": case_id,
                "total_mesh_vertices": sum(
                    len(v) for v in mesh_by_chain.values()
                ),
                "n_chains": len(mesh_by_chain),
                "total_residues_assigned": len(sample_residue_counts),
                "max_per_residue": max(sample_residue_counts) if sample_residue_counts else 0,
                "chains": sample_chain_details,
            }
        )

    counts = np.array(all_residue_counts, dtype=np.int32)
    report = {
        "dataset": dataset_name,
        "chunk_dir": str(chunk_dir),
        "pkl_files_found": len(pkl_paths),
        "samples_processed": len(sample_details),
        "samples_skipped": skipped,
        "total_residues_scored": int(counts.size),
        "per_residue_stats": {
            "mean": float(np.mean(counts)) if counts.size > 0 else 0.0,
            "std": float(np.std(counts)) if counts.size > 0 else 0.0,
            "min": int(np.min(counts)) if counts.size > 0 else 0,
            "max": int(np.max(counts)) if counts.size > 0 else 0,
            "p50": float(np.percentile(counts, 50)) if counts.size > 0 else 0.0,
            "p90": float(np.percentile(counts, 90)) if counts.size > 0 else 0.0,
            "p95": float(np.percentile(counts, 95)) if counts.size > 0 else 0.0,
            "p99": float(np.percentile(counts, 99)) if counts.size > 0 else 0.0,
            "p999": float(np.percentile(counts, 99.9)) if counts.size > 0 else 0.0,
            "fraction_over_64": float((counts > 64).mean()) if counts.size > 0 else 0.0,
            "fraction_over_128": float((counts > 128).mean()) if counts.size > 0 else 0.0,
            "residues_over_64": int((counts > 64).sum()),
            "residues_over_128": int((counts > 128).sum()),
        },
        "sample_details": sample_details,
    }
    return report, counts


# --- Plotting ----------------------------------------------------------------


def plot_distribution(
    counts: np.ndarray,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Plot the per-residue mesh vertex count distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Mesh Vertices per Residue — {dataset_name}\n"
        f"(N={counts.size:,} residues, "
        f"μ={np.mean(counts):.1f}, σ={np.std(counts):.1f}, "
        f"max={np.max(counts)})",
        fontsize=12,
    )

    # --- Histogram (full range) ---
    ax = axes[0, 0]
    ax.hist(counts, bins=100, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(64, color="crimson", linestyle="--", linewidth=1.5, label="K=64")
    ax.axvline(128, color="darkorange", linestyle="--", linewidth=1.5, label="K=128")
    ax.set_xlabel("Mesh vertices per residue")
    ax.set_ylabel("Residue count")
    ax.set_title("Full distribution")
    ax.legend()
    ax.set_yscale("log")

    # --- Zoomed: 0–200 ---
    ax = axes[0, 1]
    zoomed = counts[counts <= 200]
    ax.hist(zoomed, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(64, color="crimson", linestyle="--", linewidth=1.5, label="K=64")
    ax.axvline(128, color="darkorange", linestyle="--", linewidth=1.5, label="K=128")
    ax.set_xlabel("Mesh vertices per residue")
    ax.set_ylabel("Residue count")
    ax.set_title(f"Zoomed 0–200 ({zoomed.size:,} / {counts.size:,} residues)")
    ax.legend()

    # --- CDF ---
    ax = axes[1, 0]
    sorted_counts = np.sort(counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax.plot(sorted_counts, cdf, color="steelblue", linewidth=1.5)
    ax.axvline(64, color="crimson", linestyle="--", linewidth=1.5, label="K=64")
    ax.axvline(128, color="darkorange", linestyle="--", linewidth=1.5, label="K=128")
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Mesh vertices per residue")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF")
    ax.legend()
    ax.set_xlim(0, sorted_counts[-1])

    # --- Percentile summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    percentiles = [50, 90, 95, 99, 99.5, 99.9, 100]
    values = np.percentile(counts, percentiles)
    table_data = [
        ["Percentile", "Vertices", "≤ 64?", "≤ 128?"],
    ]
    for p, v in zip(percentiles, values):
        table_data.append(
            [
                f"P{p}",
                f"{v:.0f}",
                "✓" if v <= 64 else "✗",
                "✓" if v <= 128 else "✗",
            ]
        )
    table_data.append(
        [
            "Fraction >64",
            f"{(counts > 64).mean():.4%}",
            "",
            "",
        ]
    )
    table_data.append(
        [
            "Fraction >128",
            f"{(counts > 128).mean():.4%}",
            "",
            "",
        ]
    )

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.2, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title("Percentile summary", fontsize=11, pad=20)

    plt.tight_layout()
    output_path = output_dir / f"distribution_{dataset_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit mesh-to-residue KDTree mapping distribution."
    )
    parser.add_argument(
        "--afdb-dir",
        type=Path,
        default=Path(
            "/101045/jinyu/PPI/Prior_out/training/AFDB-10k-1"
        ),
        help="AFDB v3 chunk tree root",
    )
    parser.add_argument(
        "--qbiolip-dir",
        type=Path,
        default=Path(
            "/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new"
        ),
        help="Q-BioLiP v3 chunk tree root",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per dataset for quick runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for reports and figures",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_counts: dict[str, np.ndarray] = {}
    all_reports: dict[str, dict] = {}

    for dataset_name, chunk_root in [
        ("AFDB-10k-1", args.afdb_dir),
        ("Q-BioLiP-10k-new", args.qbiolip_dir),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"Chunk root: {chunk_root}")
        print(f"{'='*60}")

        all_chunk_counts: list[int] = []
        chunk_reports: list[dict] = []
        remaining = args.max_samples

        chunk_parent = chunk_root / "chunk"
        if chunk_parent.is_dir():
            chunk_dirs = sorted(chunk_parent.iterdir())
        else:
            chunk_dirs = sorted(chunk_root.glob("chunk*"))
        chunk_dirs = [d for d in chunk_dirs if d.is_dir()]
        if not chunk_dirs:
            chunk_dirs = [chunk_root]

        for chunk_dir in chunk_dirs:
            if remaining is not None and remaining <= 0:
                break
            prior_pkl_dir = chunk_dir / "prior_pkl"
            if not prior_pkl_dir.is_dir():
                continue
            print(f"  Chunk: {chunk_dir.name} (remaining={remaining}) ...")
            report, residue_counts = audit_mesh_distribution(
                dataset_name=f"{dataset_name}/{chunk_dir.name}",
                chunk_dir=chunk_dir,
                max_samples=remaining,
            )
            chunk_reports.append(report)
            if remaining is not None:
                remaining -= report["samples_processed"]

            all_chunk_counts.extend(residue_counts.tolist())

        # Merge across chunks
        merged_counts = np.array(all_chunk_counts, dtype=np.int32)

        # Save JSON report
        report_path = output_dir / f"report_{dataset_name}.json"
        merged_report = {
            "dataset": dataset_name,
            "chunks_processed": len(chunk_reports),
            "per_chunk": chunk_reports,
        }
        report_path.write_text(json.dumps(merged_report, indent=2, default=str))

        # Plot
        if merged_counts.size > 0:
            plot_distribution(merged_counts, dataset_name, output_dir)
            print(f"  Plotted → {output_dir}/distribution_{dataset_name}.png")
            print(
                f"  N={merged_counts.size:,}  μ={merged_counts.mean():.1f}  "
                f"σ={merged_counts.std():.1f}  "
                f"max={merged_counts.max()}  "
                f">64: {(merged_counts > 64).sum()} ({(merged_counts > 64).mean():.2%})  "
                f">128: {(merged_counts > 128).sum()} ({(merged_counts > 128).mean():.2%})"
            )
        else:
            print(f"  WARNING: No residues scored for {dataset_name}")

        all_counts[dataset_name] = merged_counts
        all_reports[dataset_name] = merged_report

    # --- Combined plot ---
    if all_counts:
        combined_list = [c for c in all_counts.values() if c.size > 0]
        if combined_list:
            combined = np.concatenate(combined_list)
            plot_distribution(combined, "Combined", output_dir)
            print(f"\nCombined plot → {output_dir}/distribution_Combined.png")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
