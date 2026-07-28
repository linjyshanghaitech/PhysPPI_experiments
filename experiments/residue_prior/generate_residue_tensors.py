#!/usr/bin/env python3
"""Generate pre-computed residue prior tensors from v3 prior PKL files.

Extracts .dmap, .interactions, and .opm from each sample's prior_pkl and saves
three independent .pt files alongside the existing mesh .pt:

    {case_id}_interface_residues.pt       – CA-CA contact pairs (dmap)
    {case_id}_interaction_pairs.pt        – 5 typed interactions
    {case_id}_membrane_relative_location.pt – OPM membrane intervals + plane

All three store PDB resseq so the training adapter can map directly via
token_index.get((chain_id, resseq)), which naturally handles crops.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

_INTERACTION_TYPE_MAP: dict[str, int] = {
    "hbond": 0,
    "saltbridge": 1,
    "hydrophobic": 2,
    "pistacking": 3,
    "pication": 4,
}


def _normalize_atom_group(value: object) -> tuple[str, ...] | None:
    """Normalize a scalar or grouped atom endpoint to canonical atom names."""
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip().upper()
        return (name,) if name else None
    if isinstance(value, (list, tuple)):
        names = tuple(
            atom.strip().upper()
            for atom in value
            if isinstance(atom, str) and atom.strip()
        )
        if len(names) == len(value) and names:
            return names
    return None


def generate_sample(pkl_path: Path) -> tuple[dict | None, dict | None, dict | None]:
    """Extract interface residues, interaction pairs, and membrane location.

    Returns (dmap_payload, interaction_payload, membrane_payload), each is
    None when the corresponding source data is empty/absent.
    """
    try:
        raw = pickle.loads(pkl_path.read_bytes())
    except Exception:
        return None, None, None

    case_id = pkl_path.stem
    if not isinstance(raw, dict) or case_id not in raw:
        return None, None, None

    obj = raw[case_id]

    # --- dmap (interface residues) --------------------------------------------
    raw_dmap: dict = {}
    if hasattr(obj, "dmap") and isinstance(obj.dmap, dict):
        raw_dmap = obj.dmap

    dmap_out: dict[str, torch.Tensor] = {}
    for chain_pair, edges in raw_dmap.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        if not isinstance(edges, (list, tuple)) or not edges:
            continue
        rows: list[list[float]] = []
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 3:
                continue
            try:
                r_a = int(edge[0])
                r_b = int(edge[1])
                dist = float(edge[2])
            except (ValueError, TypeError):
                continue
            rows.append([r_a, r_b, dist])
        if rows:
            dmap_out[chain_pair] = torch.tensor(rows, dtype=torch.float32)

    # --- interactions ---------------------------------------------------------
    raw_interactions: dict = {}
    if (
        hasattr(obj, "interactions")
        and obj.interactions is not None
        and isinstance(obj.interactions, dict)
    ):
        raw_interactions = obj.interactions

    interactions_out: dict[str, dict[str, Any]] = {}
    for chain_pair, records in raw_interactions.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        if not isinstance(records, (list, tuple)) or not records:
            continue

        pairs: list[list[int]] = []
        atom_1_list: list[tuple[str, ...]] = []
        atom_2_list: list[tuple[str, ...]] = []

        for item in records:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("t", ""))
            type_id = _INTERACTION_TYPE_MAP.get(kind)
            if type_id is None:
                continue
            try:
                r_a = int(item["r1"])
                r_b = int(item["r2"])
            except (KeyError, ValueError, TypeError):
                continue

            a1 = _normalize_atom_group(item.get("a1"))
            a2 = _normalize_atom_group(item.get("a2"))
            if a1 is None or a2 is None:
                continue

            pairs.append([r_a, r_b, type_id])
            atom_1_list.append(a1)
            atom_2_list.append(a2)

        if pairs:
            interactions_out[chain_pair] = {
                "pairs": torch.tensor(pairs, dtype=torch.int32),
                "atom_1": atom_1_list,
                "atom_2": atom_2_list,
            }

    # --- OPM (membrane) -------------------------------------------------------
    membrane_out: dict[str, Any] | None = None
    raw_opm: dict = {}
    if hasattr(obj, "opm") and obj.opm is not None and isinstance(obj.opm, dict):
        raw_opm = obj.opm

    if raw_opm:
        res_loc_raw = raw_opm.get("res_loc")
        plane = {
            "normal": torch.tensor(raw_opm.get("outer_normal", [0, 0, 0]), dtype=torch.float32),
            "centroid": torch.tensor(raw_opm.get("outer_centroid", [0, 0, 0]), dtype=torch.float32),
            "thicken": float(raw_opm.get("thicken", 0.0)),
        }
        # res_loc is list of [chain_id, start_resseq, end_resseq, side]
        res_loc_out: dict[str, torch.Tensor] = {}
        if isinstance(res_loc_raw, (list, tuple)):
            by_chain: dict[str, list[list[int]]] = {}
            for row in res_loc_raw:
                if not isinstance(row, (list, tuple)) or len(row) < 4:
                    continue
                try:
                    chain_id = str(row[0])
                    start = int(row[1])
                    end = int(row[2])
                    side = int(row[3])
                except (ValueError, TypeError):
                    continue
                by_chain.setdefault(chain_id, []).append([start, end, side])
            for chain_id, intervals in by_chain.items():
                res_loc_out[chain_id] = torch.tensor(intervals, dtype=torch.int32)

        if res_loc_out:
            membrane_out = {
                "res_loc": res_loc_out,
                "plane": plane,
            }

    return (
        {"dmap": dmap_out} if dmap_out else None,
        {"interactions": interactions_out} if interactions_out else None,
        membrane_out,
    )


def run_generation(
    chunk_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Generate .pt files for one chunk."""
    prior_pkl_dir = chunk_dir / "prior_pkl"
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths = sorted(prior_pkl_dir.glob("*.pkl"))
    if max_samples is not None:
        pkl_paths = pkl_paths[:max_samples]

    stats: dict[str, Any] = {
        "chunk": str(chunk_dir),
        "total": len(pkl_paths),
        "interface_residues": 0,
        "interaction_pairs": 0,
        "membrane": 0,
        "skipped_error": 0,
        "total_dmap_pairs": 0,
        "total_interaction_pairs": 0,
        "total_tm_intervals": 0,
    }

    for pkl_path in pkl_paths:
        case_id = pkl_path.stem

        try:
            dmap_payload, inter_payload, memb_payload = generate_sample(pkl_path)
        except Exception as exc:
            print(f"  SKIP {case_id}: {exc}", file=sys.stderr)
            stats["skipped_error"] += 1
            continue

        if dmap_payload is not None:
            dmap_pairs = sum(t.shape[0] for t in dmap_payload["dmap"].values())
            stats["total_dmap_pairs"] += dmap_pairs
            _save_pt(output_dir, f"{case_id}_interface_residues.pt", dmap_payload)
            stats["interface_residues"] += 1

        if inter_payload is not None:
            inter_pairs = sum(
                v["pairs"].shape[0] for v in inter_payload["interactions"].values()
            )
            stats["total_interaction_pairs"] += inter_pairs
            _save_pt(output_dir, f"{case_id}_interaction_pairs.pt", inter_payload)
            stats["interaction_pairs"] += 1

        if memb_payload is not None:
            tm_intervals = sum(
                t.shape[0] for t in memb_payload["res_loc"].values()
            )
            stats["total_tm_intervals"] += tm_intervals
            _save_pt(output_dir, f"{case_id}_membrane_relative_location.pt", memb_payload)
            stats["membrane"] += 1

    return stats


def _save_pt(output_dir: Path, filename: str, payload: dict) -> None:
    tmp_path = output_dir / (filename + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_dir / filename)


def _run_one_chunk(args: tuple[Path, Path, int | None]) -> dict[str, Any]:
    """Worker entry point for ProcessPoolExecutor."""
    chunk_dir, output_dir, max_samples = args
    return run_generation(
        chunk_dir=chunk_dir,
        output_dir=output_dir,
        max_samples=max_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-sample residue prior tensors from v3 PKL."
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
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap total samples"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Comma-separated chunk names to process (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel workers (default: 64)",
    )
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

    # Filter to chunks with prior_pkl
    work_items: list[tuple[Path, Path, int | None]] = []
    for chunk_dir in all_chunk_dirs:
        pkl_dir = chunk_dir / "prior_pkl"
        if pkl_dir.is_dir():
            chunk_output = args.output_dir / chunk_dir.name
            work_items.append((chunk_dir, chunk_output, args.max_samples))

    print(f"Dataset: {args.dataset_dir}")
    print(f"Chunks: {len(work_items)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output_dir}")
    print()

    global_stats: dict[str, Any] = {"chunks": {}}
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one_chunk, item): item[0].name
            for item in work_items
        }
        for future in futures:
            chunk_name = futures[future]
            try:
                stats = future.result()
            except Exception as exc:
                print(f"Chunk {chunk_name} FAILED: {exc}", file=sys.stderr)
                stats = {
                    "chunk": chunk_name,
                    "total": 0,
                    "interface_residues": 0,
                    "interaction_pairs": 0,
                    "membrane": 0,
                    "skipped_error": 1,
                    "total_dmap_pairs": 0,
                    "total_interaction_pairs": 0,
                    "total_tm_intervals": 0,
                }
            global_stats["chunks"][chunk_name] = stats
            completed += 1
            print(
                f"[{completed}/{len(work_items)}] {chunk_name}: "
                f"{stats['interface_residues']} iface, "
                f"{stats['interaction_pairs']} inter, "
                f"{stats['membrane']} memb, "
                f"{stats['skipped_error']} errors"
            )

    # Save summary
    summary_path = args.output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
