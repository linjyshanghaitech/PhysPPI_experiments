#!/usr/bin/env python3
"""Generate pre-computed mesh-to-residue tensors from v3 prior PKL files.

For each sample, builds a KDTree of heavy atoms from the GT PDB, assigns every
mesh vertex to the residue of its nearest heavy atom, and saves a per-sample
.pt file containing:

    {
        (chain_id, resseq): np.ndarray[M_dynamic, 9],  # only residues with ≥1 vertex
    }

9 columns: x, y, z, charge, hphob, nx, ny, nz, si  (hbond dropped).

At training time the adapter pads/truncates to a fixed K and builds a
[N_tokens, K, 9] tensor + [N_tokens, K] mask following the token_index order.
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

# --- Constants ---------------------------------------------------------------

HYDROGEN_ELEMENTS = {"H", "D", "T"}
HOH_RESNAME = "HOH"

# Original vertex columns:  x,  y,  z,  charge,  hbond,  hphob,  nx,  ny,  nz,  si
# Output columns (drop hbond):  x,  y,  z,  charge,  hphob,  nx,  ny,  nz,  si
VERTEX_KEEP = [0, 1, 2, 3, 5, 6, 7, 8, 9]  # index 4 (hbond) removed
OUTPUT_DIM = 9


# --- Parsing ----------------------------------------------------------------


def parse_gt_residues(
    pdb_path: Path,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Parse a GT PDB, returning per-chain ordered residues with heavy-atom coords.

    Returns {chain_id: [(resseq, heavy_coords_Mx3), ...]} in PDB order.
    """
    chain_atoms: dict[str, dict[int, list[tuple[float, float, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

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
            chain_atoms[chain_id][resseq].append((x, y, z))

    result: dict[str, list[tuple[int, np.ndarray]]] = {}
    for chain_id in sorted(chain_atoms):
        # Sort by resseq to match PDB/token order
        residues = []
        for resseq in sorted(chain_atoms[chain_id]):
            coords = np.array(chain_atoms[chain_id][resseq], dtype=np.float32)
            residues.append((resseq, coords))
        result[chain_id] = residues
    return result


def load_mesh_vertices(pkl_path: Path) -> dict[str, list[list[float]]]:
    """Load per-chain mesh vertices from a v3-format prior PKL."""
    payload = pickle.loads(pkl_path.read_bytes())
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
    residues: list[tuple[int, np.ndarray]],
) -> dict[int, list[np.ndarray]]:
    """Map mesh vertices to residues via nearest-heavy-atom KDTree.

    Returns {resseq: sorted_vertex_list} where vertices are sorted by distance
    to nearest heavy atom (closest first).
    """
    if not mesh_chain_vertices or not residues:
        return {}

    # Build flat heavy-atom coords + residue index lookup
    all_coords: list[np.ndarray] = []
    atom_to_res: list[int] = []
    resseq_list = [r[0] for r in residues]

    for resseq, coords in residues:
        for i in range(coords.shape[0]):
            all_coords.append(coords[i])
            atom_to_res.append(resseq)

    if not all_coords:
        return {}

    heavy_coords = np.stack(all_coords, axis=0)
    atom_to_res_arr = np.array(atom_to_res, dtype=np.int32)

    vertices = np.array(mesh_chain_vertices, dtype=np.float32)
    vertex_xyz = vertices[:, :3]

    tree = cKDTree(heavy_coords)
    distances, atom_idx = tree.query(vertex_xyz, k=1)

    # Group vertices by residue, sort by distance
    residue_verts: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for i in range(vertices.shape[0]):
        resseq = int(atom_to_res_arr[atom_idx[i]])
        vert_9d = vertices[i, VERTEX_KEEP]
        residue_verts[resseq].append((float(distances[i]), vert_9d))

    # Sort each residue's vertices by distance (closest first)
    result: dict[int, list[np.ndarray]] = {}
    for resseq in residue_verts:
        sorted_verts = sorted(residue_verts[resseq], key=lambda x: x[0])
        result[resseq] = [v[1] for v in sorted_verts]

    return result


# --- Main generation ---------------------------------------------------------


def _parse_gt_residue_order(gt_pdb: Path) -> dict[str, list[int]]:
    """Parse GT PDB to get per-chain ordered residue list (all residues)."""
    chain_residues: dict[str, list[int]] = {}
    with gt_pdb.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
                continue
            chain_id = line[21].strip()
            if not chain_id:
                continue
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            residues = chain_residues.setdefault(chain_id, [])
            if not residues or residues[-1] != resseq:
                residues.append(resseq)
    return chain_residues


def _parse_bb_residue_names(bb_pdb: Path) -> list[tuple[int, str]]:
    """Parse bb_input PDB to get ordered (resseq, res_name_3) pairs."""
    order: list[tuple[int, str]] = []
    with bb_pdb.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
                continue
            try:
                resseq = int(line[22:26].strip())
                resname = line[17:20].strip().upper()
            except ValueError:
                continue
            if not order or order[-1][0] != resseq:
                order.append((resseq, resname))
    return order


def generate_sample(
    pkl_path: Path,
    bb_input_dir: Path,
    gt_dir: Path,
    seq_cache: dict,
    case_id: str,
    k: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Generate per-chain [N, K, 9] mesh tensor for one sample.

    N = len(full_seq_1aa) from seq_cache.pkl.
    Residue ordering follows the GT PDB (same as token_index).
    Non-interface residues get all-zero rows.

    Returns (data_dict, resseq_dict) where:
      data_dict: {chain_id: torch.Tensor[N_chain, K, 9]}
      resseq_dict: {chain_id: list[int]}  # resseq per row
    """
    mesh_by_chain = load_mesh_vertices(pkl_path)
    gt_pdb = gt_dir / f"{case_id}.pdb"
    gt_res_order = _parse_gt_residue_order(gt_pdb) if gt_pdb.exists() else {}

    case_seqs = seq_cache.get(case_id, {})

    result: dict[str, torch.Tensor] = {}
    result_resseq: dict[str, list[int]] = {}
    for chain_id in sorted(mesh_by_chain):
        chain_pdb = bb_input_dir / f"{case_id}_{chain_id}.pdb"
        if not chain_pdb.exists():
            continue

        mesh_verts = mesh_by_chain[chain_id]
        bb_residues = parse_gt_residues(chain_pdb).get(chain_id, [])
        if not mesh_verts or not bb_residues:
            continue

        # KDTree mapping (monomer-local coords)
        assigned = map_mesh_to_residues(mesh_verts, bb_residues)

        # Build [N, K, 9] using seq_cache + bb_input residue order.
        # bb_input may have terminal artifacts (e.g. expression tags) that
        # are absent from seq_cache.  Align bb_input residues to seq_cache
        # by sequential residue-name matching so each seq_cache position
        # gets data from the structurally corresponding PDB residue.
        seq_info = case_seqs.get(chain_id, {})
        seq_3aa = seq_info.get("full_seq_3aa", [])
        n_full = len(seq_3aa)
        if n_full == 0:
            n_full = len(gt_res_order.get(chain_id, []))
        if n_full == 0:
            continue

        # Use seq_cache's built-in resseq→sequence_index mapping if available.
        resseq_to_seq_idx: dict[int, int] = seq_info.get("resseq_to_sequence_index", {})
        if not resseq_to_seq_idx:
            # Fallback: greedy alignment of bb_input residues to seq_cache
            bb_order = _parse_bb_residue_names(chain_pdb)
            bb_idx = 0
            for seq_idx, seq_res in enumerate(seq_3aa):
                while bb_idx < len(bb_order):
                    bb_resseq, bb_res = bb_order[bb_idx]
                    bb_idx += 1
                    if bb_res == seq_res:
                        resseq_to_seq_idx[bb_resseq] = seq_idx
                        break

        data = torch.zeros(n_full, k, 9)
        for resseq, verts in assigned.items():
            seq_idx = resseq_to_seq_idx.get(resseq)
            if seq_idx is None:
                continue
            if seq_idx >= n_full:
                continue
            m = min(len(verts), k)
            stacked = torch.from_numpy(np.stack(verts[:m], axis=0))
            data[seq_idx, :m, :] = stacked

        # Build resseq list matching data rows
        resseq_list = [-1] * n_full
        for resseq, seq_idx in resseq_to_seq_idx.items():
            if seq_idx < n_full:
                resseq_list[seq_idx] = resseq

        result[chain_id] = data
        result_resseq[chain_id] = resseq_list

    return result, result_resseq


def _process_one_sample(args: tuple) -> tuple[str, dict | None, str | None]:
    """Process a single sample (module-level for multiprocessing)."""
    pkl_path, bb_input_dir, gt_dir, seq_cache, case_id, k, output_dir = args

    chain_pdbs = list(bb_input_dir.glob(f"{case_id}_*.pdb"))
    if not chain_pdbs:
        return (case_id, None, "no_bb_input")

    try:
        mapping, resseq_lists = generate_sample(
            pkl_path, bb_input_dir, gt_dir, seq_cache, case_id, k
        )
    except Exception as exc:
        return (case_id, None, str(exc))

    if not mapping:
        return (case_id, None, "empty_mapping")

    # Save atomically
    output_path = output_dir / f"{case_id}_mesh.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": mapping, "resseq": resseq_lists}
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)
    return (case_id, payload, None)


def run_generation(
    chunk_dir: Path,
    output_dir: Path,
    k: int,
    max_samples: int | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Generate .pt files for one chunk of samples."""
    prior_pkl_dir = chunk_dir / "prior_pkl"
    bb_input_dir = chunk_dir / "bb_input"
    gt_dir = chunk_dir / "gt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load seq_cache for N per chain
    seq_cache_path = chunk_dir / "seq_cache.pkl"
    seq_cache: dict = {}
    if seq_cache_path.exists():
        seq_cache = pickle.loads(seq_cache_path.read_bytes())

    pkl_paths = sorted(prior_pkl_dir.glob("*.pkl"))
    if max_samples is not None:
        pkl_paths = pkl_paths[:max_samples]

    # Build task args
    tasks = [
        (p, bb_input_dir, gt_dir, seq_cache, p.stem, k, output_dir)
        for p in pkl_paths
    ]

    stats = {
        "chunk": str(chunk_dir),
        "total": len(pkl_paths),
        "processed": 0,
        "skipped_no_bb_input": 0,
        "skipped_error": 0,
        "total_residues_assigned": 0,
        "total_vertices": 0,
        "max_verts_per_residue": 0,
    }

    if num_workers <= 1:
        results = map(_process_one_sample, tasks)
    else:
        from multiprocessing import Pool
        with Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(_process_one_sample, tasks, chunksize=5)

    for case_id, payload, error in results:
        if error:
            if error == "no_bb_input":
                stats["skipped_no_bb_input"] += 1
            else:
                stats["skipped_error"] += 1
                if error != "empty_mapping":
                    print(f"    SKIP {case_id}: {error}", file=sys.stderr)
            continue

        for data in payload["data"].values():
            per_res = (data.abs().sum(dim=-1) > 0).sum(dim=1).int()
            n_assigned = int((per_res > 0).sum().item())
            n_verts = int(per_res.sum().item())
            stats["total_residues_assigned"] += n_assigned
            stats["total_vertices"] += n_verts
            if per_res.max().item() > stats["max_verts_per_residue"]:
                stats["max_verts_per_residue"] = int(per_res.max().item())

        stats["processed"] += 1

    return stats


# --- Main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-residue mesh tensors from v3 PKL + GT PDB."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset root (e.g., /path/to/AFDB-10k-1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .npz files",
    )
    parser.add_argument("--k", type=int, default=64, help="Max vertices per residue")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total samples")
    parser.add_argument("--chunks", type=str, default=None, help="Comma-separated chunk names to process (default: all)")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers (default: 1 = sequential)")
    args = parser.parse_args()

    # Resolve chunk directories
    chunk_parent = args.dataset_dir / "chunk"
    if chunk_parent.is_dir():
        all_chunk_dirs = sorted(
            d for d in chunk_parent.iterdir() if d.is_dir()
        )
    else:
        all_chunk_dirs = sorted(
            d for d in args.dataset_dir.glob("chunk*") if d.is_dir()
        )
    if not all_chunk_dirs:
        all_chunk_dirs = [args.dataset_dir]

    if args.chunks is not None:
        allowed = set(c.strip() for c in args.chunks.split(","))
        all_chunk_dirs = [d for d in all_chunk_dirs if d.name in allowed]

    print(f"Dataset: {args.dataset_dir}")
    print(f"Chunks: {len(all_chunk_dirs)}")
    print(f"K: {args.k}")
    print(f"Output: {args.output_dir}")
    print()

    remaining = args.max_samples
    global_stats: dict[str, Any] = {"chunks": {}}

    for chunk_dir in all_chunk_dirs:
        if remaining is not None and remaining <= 0:
            break
        pkl_dir = chunk_dir / "prior_pkl"
        if not pkl_dir.is_dir():
            continue

        chunk_output = args.output_dir / chunk_dir.name
        print(f"Chunk {chunk_dir.name} (remaining={remaining}) ...", end=" ", flush=True)

        stats = run_generation(
            chunk_dir=chunk_dir,
            output_dir=chunk_output,
            k=args.k,
            max_samples=remaining,
        )
        global_stats["chunks"][chunk_dir.name] = stats

        n = stats["processed"]
        print(
            f"{n} samples, "
            f"{stats['total_residues_assigned']} residues, "
            f"max_verts={stats['max_verts_per_residue']}"
        )
        if remaining is not None:
            remaining -= n

    # Save summary
    summary_path = args.output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
