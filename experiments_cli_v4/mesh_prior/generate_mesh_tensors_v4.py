#!/usr/bin/env python3
"""Generate pre-computed per-residue mesh tensors for the v4 prior pipeline.

For each sample, the v4 prior PKL already contains:
    {case_id}:            SimpleNamespace(mesh={chain: [V, 10]}, ...)
    {case_id}_point_to_residue: {chain: [resseq, ...]}   # 1 per vertex, 1-based PDB resseq

This script converts that into the same per-sample `.pt` layout used by the v3
`experiments/mesh_prior/generate_mesh_tensors.py`:

    {
        "data":   {chain_id: torch.Tensor[N, K, 9]},   # rows follow input-PDB residue order
        "resseq": {chain_id: [resseq_0, resseq_1, ...]},
    }

9 columns = x, y, z, charge, hphob, nx, ny, nz, si  (the hbond column is dropped).
Vertices are grouped by residue via the precomputed point_to_residue map and
sorted by distance to the nearest heavy atom (closest first), then truncated to
K per residue, matching the v3 conversion method.

The all-atom PDB in ``chunk*/input/`` is used to:
  * determine per-chain residue order (row order of the output tensor),
  * build the heavy-atom KDTree (fallback mapping when point_to_residue is
    missing; also used for the per-vertex nearest-atom distance used in sorting).

Output layout:
    {output_root}/{dataset_name}/chunkNNNN/{case_id}_mesh.pt
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


HYDROGEN_ELEMENTS = {"H", "D", "T"}
HOH_RESNAME = "HOH"

# Original vertex columns: x, y, z, charge, hbond, hphob, nx, ny, nz, si
# Output columns (hbond dropped): x, y, z, charge, hphob, nx, ny, nz, si
VERTEX_KEEP = [0, 1, 2, 3, 5, 6, 7, 8, 9]
OUTPUT_DIM = 9


def parse_input_pdb(
    pdb_path: Path,
) -> dict[str, tuple[list[int], np.ndarray, np.ndarray]]:
    """Parse an all-atom complex PDB.

    Returns {chain_id: (ordered_unique_resseq, heavy_coords[M,3], heavy_resseq[M,])}.
    Residue order = first appearance order of ATOM/HETATM records per chain.
    """
    chain_order: dict[str, list[int]] = defaultdict(list)
    chain_coords: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    chain_resseq: dict[str, list[int]] = defaultdict(list)

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
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
            order = chain_order[chain_id]
            if not order or order[-1] != resseq:
                order.append(resseq)
            chain_coords[chain_id].append((x, y, z))
            chain_resseq[chain_id].append(resseq)

    result: dict[str, tuple[list[int], np.ndarray, np.ndarray]] = {}
    for chain_id in chain_order:
        coords = np.array(chain_coords[chain_id], dtype=np.float32)
        resseqs = np.array(chain_resseq[chain_id], dtype=np.int32)
        result[chain_id] = (chain_order[chain_id], coords, resseqs)
    return result


def load_v4_prior(pkl_path: Path) -> tuple[str, dict[str, list[list[float]]], dict[str, list[int]] | None]:
    """Load (case_id, mesh, point_to_residue) from a v4 prior PKL."""
    payload = pickle.loads(pkl_path.read_bytes())
    case_id = next(iter(payload))
    obj = payload[case_id]
    mesh = getattr(obj, "mesh", None)
    if mesh is None:
        raise KeyError(f"no 'mesh' field in {pkl_path}")
    if not isinstance(mesh, dict):
        raise TypeError(f"'mesh' is not dict in {pkl_path}: {type(mesh)}")
    p2r = payload.get(f"{case_id}_point_to_residue")
    if p2r is not None and not isinstance(p2r, dict):
        raise TypeError(f"'point_to_residue' is not dict in {pkl_path}: {type(p2r)}")
    return case_id, {str(ch): list(v) for ch, v in mesh.items()}, p2r


def build_chain_tensor(
    mesh_vertices: list[list[float]],
    p2r_chain: list[int] | None,
    ordered_resseq: list[int],
    heavy_coords: np.ndarray,
    heavy_resseq: np.ndarray,
    k: int,
) -> tuple[torch.Tensor, list[int]]:
    """Build [N, K, 9] per-residue tensor for one chain.

    Vertex -> residue assignment uses the precomputed point_to_residue map when
    available, otherwise the nearest heavy atom.  Vertices are sorted by
    nearest-heavy-atom distance (closest first) before truncating to K.
    """
    n = len(ordered_resseq)
    data = torch.zeros(n, k, OUTPUT_DIM, dtype=torch.float32)
    if not mesh_vertices:
        return data, ordered_resseq

    vertices = np.array(mesh_vertices, dtype=np.float32)
    row_index = {resseq: i for i, resseq in enumerate(ordered_resseq)}

    # Nearest-heavy-atom distances (used for sorting; also the fallback mapping).
    if heavy_coords.shape[0] > 0:
        tree = cKDTree(heavy_coords)
        distances, atom_idx = tree.query(vertices[:, :3], k=1)
        distances = np.asarray(distances, dtype=np.float64)
        atom_idx = np.asarray(atom_idx, dtype=np.int64)
    else:
        distances = np.zeros(vertices.shape[0], dtype=np.float64)
        atom_idx = None

    if p2r_chain is not None:
        if len(p2r_chain) != vertices.shape[0]:
            raise ValueError(
                f"point_to_residue length {len(p2r_chain)} != vertex count {vertices.shape[0]}"
            )
        vertex_res = np.asarray(p2r_chain, dtype=np.int32)
    else:
        if atom_idx is None:
            raise ValueError("no heavy atoms available for KDTree mapping")
        vertex_res = heavy_resseq[atom_idx]

    # Group vertex indices by residue, sort by distance (closest first).
    by_res: dict[int, list[int]] = defaultdict(list)
    for v in range(vertices.shape[0]):
        by_res[int(vertex_res[v])].append(v)

    for resseq, vert_idx in by_res.items():
        row = row_index.get(resseq)
        if row is None:
            continue
        vert_idx.sort(key=lambda v: float(distances[v]))
        m = min(len(vert_idx), k)
        stacked = torch.from_numpy(vertices[vert_idx[:m]][:, VERTEX_KEEP].copy())
        data[row, :m, :] = stacked

    return data, ordered_resseq


def generate_sample(
    pkl_path: Path,
    pdb_path: Path,
    case_id: str,
    k: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Generate per-chain [N, K, 9] mesh tensors for one sample."""
    _, mesh, p2r = load_v4_prior(pkl_path)
    pdb_info = parse_input_pdb(pdb_path) if pdb_path.exists() else {}

    result: dict[str, torch.Tensor] = {}
    result_resseq: dict[str, list[int]] = {}
    for chain_id in sorted(mesh):
        ordered_resseq, heavy_coords, heavy_resseq = pdb_info.get(
            chain_id, ([], np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32))
        )
        if not ordered_resseq and p2r is not None and chain_id in p2r:
            # Fallback row order when the chain is absent from the input PDB.
            ordered_resseq = sorted(set(p2r[chain_id]))
        if not ordered_resseq:
            continue
        p2r_chain = (p2r or {}).get(chain_id)
        data, resseq_list = build_chain_tensor(
            mesh[chain_id], p2r_chain, ordered_resseq, heavy_coords, heavy_resseq, k
        )
        result[chain_id] = data
        result_resseq[chain_id] = resseq_list

    return result, result_resseq


def _process_one_sample(args: tuple) -> tuple[str, dict | None, str | None, int, int, int]:
    pkl_path, pdb_path, case_id, k, output_path, force = args
    if output_path.exists() and not force:
        return (case_id, None, "exists", 0, 0, 0)
    try:
        mapping, resseq_lists = generate_sample(pkl_path, pdb_path, case_id, k)
    except Exception as exc:
        return (case_id, None, f"error: {exc}", 0, 0, 0)

    if not mapping:
        return (case_id, None, "empty_mapping", 0, 0, 0)

    payload = {"data": mapping, "resseq": resseq_lists}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)

    n_residues = sum(len(v) for v in resseq_lists.values())
    n_vertices = 0
    max_verts = 0
    for data in mapping.values():
        per_res = (data.abs().sum(dim=-1) > 0).sum(dim=1).int()
        n_vertices += int(per_res.sum().item())
        if per_res.max().item() > max_verts:
            max_verts = int(per_res.max().item())
    return (case_id, payload, None, n_residues, n_vertices, max_verts)


def run_chunk(
    chunk_dir: Path,
    output_dir: Path,
    k: int,
    max_samples: int | None,
    num_workers: int,
    force: bool,
) -> dict[str, Any]:
    prior_pkl_dir = chunk_dir / "prior_pkl"
    input_dir = chunk_dir / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths = sorted(prior_pkl_dir.glob("*.pkl")) if prior_pkl_dir.is_dir() else []
    if max_samples is not None:
        pkl_paths = pkl_paths[:max_samples]

    tasks = []
    for p in pkl_paths:
        case_id = p.stem
        tasks.append(
            (
                p,
                input_dir / f"{case_id}.pdb",
                case_id,
                k,
                output_dir / f"{case_id}_mesh.pt",
                force,
            )
        )

    stats: dict[str, Any] = {
        "chunk": str(chunk_dir),
        "total": len(tasks),
        "processed": 0,
        "skipped_existing": 0,
        "skipped_error": 0,
        "total_residues_assigned": 0,
        "total_vertices": 0,
        "max_verts_per_residue": 0,
    }

    def consume(results) -> None:
        for case_id, payload, error, n_res, n_verts, max_verts in results:
            if error:
                if error == "exists":
                    stats["skipped_existing"] += 1
                else:
                    stats["skipped_error"] += 1
                    if not error.startswith("empty"):
                        print(f"    SKIP {case_id}: {error}", file=sys.stderr)
                continue
            stats["processed"] += 1
            stats["total_residues_assigned"] += n_res
            stats["total_vertices"] += n_verts
            stats["max_verts_per_residue"] = max(
                stats["max_verts_per_residue"], max_verts
            )

    if num_workers <= 1:
        consume(map(_process_one_sample, tasks))
    else:
        from multiprocessing import Pool

        with Pool(processes=num_workers) as pool:
            consume(pool.imap_unordered(_process_one_sample, tasks, chunksize=4))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-residue mesh tensors from v4 prior PKL + input PDB."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset root containing chunk/ (e.g., .../training/cli_v4/AFDB-10k-1-old)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
        help="Root output directory (chunk dirs are created under {root}/{dataset-name})",
    )
    parser.add_argument("--dataset-name", type=str, default="AFDB-10k-1")
    parser.add_argument("--k", type=int, default=64, help="Max vertices per residue")
    parser.add_argument("--chunks", type=str, default=None, help="Comma-separated chunk names (default: all)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total samples per run")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Regenerate existing .pt files")
    args = parser.parse_args()

    chunk_parent = args.dataset_dir / "chunk"
    if chunk_parent.is_dir():
        all_chunk_dirs = sorted(d for d in chunk_parent.iterdir() if d.is_dir())
    else:
        all_chunk_dirs = sorted(d for d in args.dataset_dir.glob("chunk*") if d.is_dir())
    if not all_chunk_dirs:
        all_chunk_dirs = [args.dataset_dir]

    if args.chunks is not None:
        allowed = set(c.strip() for c in args.chunks.split(","))
        all_chunk_dirs = [d for d in all_chunk_dirs if d.name in allowed]

    output_root = args.output_root / args.dataset_name
    print(f"Dataset: {args.dataset_dir}")
    print(f"Chunks: {len(all_chunk_dirs)}")
    print(f"K: {args.k}")
    print(f"Output: {output_root}")
    print()

    global_stats: dict[str, Any] = {"chunks": {}}
    for chunk_dir in all_chunk_dirs:
        pkl_dir = chunk_dir / "prior_pkl"
        if not pkl_dir.is_dir():
            continue
        chunk_output = output_root / chunk_dir.name
        print(f"Chunk {chunk_dir.name} ...", end=" ", flush=True)
        stats = run_chunk(
            chunk_dir=chunk_dir,
            output_dir=chunk_output,
            k=args.k,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            force=args.force,
        )
        global_stats["chunks"][chunk_dir.name] = stats
        print(
            f"{stats['processed']} processed, {stats['skipped_existing']} existing, "
            f"{stats['skipped_error']} errors, max_verts={stats['max_verts_per_residue']}"
        )

    summary_path = output_root / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
