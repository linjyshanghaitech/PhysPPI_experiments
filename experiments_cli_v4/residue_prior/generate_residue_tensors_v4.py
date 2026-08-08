#!/usr/bin/env python3
"""Generate pre-computed residue prior tensors from v4 prior PKL files.

Extracts .dmap, .interactions, and .tmdet from each sample's prior_pkl and
saves three independent .pt files:

    {case_id}_interface_residues.pt       — CA-CA contact pairs (dmap)
    {case_id}_interaction_pairs.pt        — 5 typed interactions
    {case_id}_membrane_relative_location.pt — membrane intervals + plane

The v4 prior PKL stores membrane information in the ``tmdet`` field
(``res_loc`` / ``outer_centroid`` / ``outer_normal`` / ``thicken``); the
legacy v3 ``opm`` field is used as a fallback.

Output layout:
    {output_root}/{dataset_name}/{chunk_name}/{case_id}_{suffix}.pt
    {output_root}/{dataset_name}/residue_generation_summary.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
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


def _load_pdb_atom_sets(pdb_path: Path) -> dict[tuple[str, int], set[str]]:
    atom_sets: dict[tuple[str, int], set[str]] = {}
    try:
        lines = pdb_path.read_text(errors="replace").splitlines()
    except OSError:
        return atom_sets
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) > 100:
            continue
        atom_name = line[12:16].strip().upper()
        chain_id = line[21].strip()
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        if not atom_name:
            continue
        atom_sets.setdefault((chain_id, resseq), set()).add(atom_name)
    return atom_sets


def _load_membrane_attr(obj: Any, name: str) -> dict:
    value = getattr(obj, name, None)
    if value is not None and isinstance(value, dict):
        return value
    return {}


def generate_sample(
    pkl_path: Path,
) -> tuple[dict | None, dict | None, dict | None]:
    try:
        raw = pickle.loads(pkl_path.read_bytes())
    except Exception:
        return None, None, None

    case_id = pkl_path.stem
    if not isinstance(raw, dict) or case_id not in raw:
        return None, None, None

    obj = raw[case_id]
    gt_pdb = pkl_path.parent.parent / "gt" / f"{case_id}.pdb"
    atom_sets = _load_pdb_atom_sets(gt_pdb) if gt_pdb.is_file() else {}

    # --- dmap (interface residues) ----------------------------------------
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

    # --- interactions ------------------------------------------------------
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
        left_ch, right_ch = chain_pair.split("|", 1)

        pairs: list[list[int]] = []
        atom_1_list: list[tuple[str, ...]] = []
        atom_2_list: list[tuple[str, ...]] = []
        dist_list: list[float] = []

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
            if atom_sets:
                left_atoms = atom_sets.get((left_ch, r_a), ())
                right_atoms = atom_sets.get((right_ch, r_b), ())
                direct_ok = all(a in left_atoms for a in a1) and all(
                    a in right_atoms for a in a2
                )
                if not direct_ok:
                    swapped_ok = all(a in right_atoms for a in a1) and all(
                        a in left_atoms for a in a2
                    )
                    if swapped_ok:
                        a1, a2 = a2, a1
            try:
                dist = float(item["d"]) if item.get("d") is not None else float("nan")
            except (KeyError, ValueError, TypeError, OverflowError):
                dist = float("nan")

            pairs.append([r_a, r_b, type_id])
            atom_1_list.append(a1)
            atom_2_list.append(a2)
            dist_list.append(dist)

        if pairs:
            interactions_out[chain_pair] = {
                "pairs": torch.tensor(pairs, dtype=torch.int32),
                "atom_1": atom_1_list,
                "atom_2": atom_2_list,
                "dist": torch.tensor(dist_list, dtype=torch.float32),
            }

    # --- membrane (tmdet preferred, opm fallback) --------------------------
    membrane_out: dict[str, Any] | None = None
    raw_membrane = _load_membrane_attr(obj, "tmdet")
    if not raw_membrane:
        raw_membrane = _load_membrane_attr(obj, "opm")

    if raw_membrane:
        res_loc_raw = raw_membrane.get("res_loc")
        plane = {
            "normal": torch.tensor(
                raw_membrane.get("outer_normal", [0, 0, 0]),
                dtype=torch.float32,
            ),
            "centroid": torch.tensor(
                raw_membrane.get("outer_centroid", [0, 0, 0]),
                dtype=torch.float32,
            ),
            "thicken": float(raw_membrane.get("thicken", 0.0)),
        }
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
            membrane_out = {"res_loc": res_loc_out, "plane": plane}

    return (
        {"dmap": dmap_out} if dmap_out else None,
        {"interactions": interactions_out} if interactions_out else None,
        membrane_out,
    )


def _save_pt(output_dir: Path, filename: str, payload: dict) -> None:
    tmp_path = output_dir / (filename + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_dir / filename)


def _process_one_sample(args: tuple) -> dict[str, Any]:
    pkl_path, output_dir, force = args
    case_id = pkl_path.stem
    interface_pt = output_dir / f"{case_id}_interface_residues.pt"
    interaction_pt = output_dir / f"{case_id}_interaction_pairs.pt"
    membrane_pt = output_dir / f"{case_id}_membrane_relative_location.pt"

    if all(p.exists() for p in (interface_pt, interaction_pt, membrane_pt)) and not force:
        return {
            "case_id": case_id,
            "status": "exists",
            "error": None,
            "interface": 0,
            "interaction": 0,
            "membrane": 0,
        }

    try:
        dmap_payload, inter_payload, memb_payload = generate_sample(pkl_path)
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "interface": 0,
            "interaction": 0,
            "membrane": 0,
        }

    n_iface = 0
    n_inter = 0
    n_memb = 0
    try:
        if dmap_payload is not None:
            n_iface = sum(t.shape[0] for t in dmap_payload["dmap"].values())
            if not interface_pt.exists() or force:
                _save_pt(output_dir, f"{case_id}_interface_residues.pt", dmap_payload)
        if inter_payload is not None:
            n_inter = sum(
                v["pairs"].shape[0]
                for v in inter_payload["interactions"].values()
            )
            if not interaction_pt.exists() or force:
                _save_pt(output_dir, f"{case_id}_interaction_pairs.pt", inter_payload)
        if memb_payload is not None:
            n_memb = sum(t.shape[0] for t in memb_payload["res_loc"].values())
            if not membrane_pt.exists() or force:
                _save_pt(
                    output_dir,
                    f"{case_id}_membrane_relative_location.pt",
                    memb_payload,
                )
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"save failed: {exc}",
            "interface": 0,
            "interaction": 0,
            "membrane": 0,
        }

    return {
        "case_id": case_id,
        "status": "ok",
        "error": None,
        "interface": n_iface,
        "interaction": n_inter,
        "membrane": n_memb,
    }


def collect_tasks(
    chunk_dir: Path,
    output_chunk_dir: Path,
    max_samples: int | None,
    force: bool,
) -> list[tuple]:
    prior_pkl_dir = chunk_dir / "prior_pkl"
    if not prior_pkl_dir.is_dir():
        return []
    pkl_paths = sorted(prior_pkl_dir.glob("*.pkl"))
    if max_samples is not None:
        pkl_paths = pkl_paths[:max_samples]
    return [(p, output_chunk_dir, force) for p in pkl_paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-sample residue prior tensors from v4 PKL."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--dataset-name", type=str, default="AFDB-10k-1")
    parser.add_argument("--chunks", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
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
    output_root.mkdir(parents=True, exist_ok=True)

    chunk_tasks: dict[str, list[tuple]] = {}
    for chunk_dir in all_chunk_dirs:
        if not (chunk_dir / "prior_pkl").is_dir():
            continue
        chunk_output = output_root / chunk_dir.name
        tasks = collect_tasks(chunk_dir, chunk_output, args.max_samples, args.force)
        if tasks:
            chunk_tasks[chunk_dir.name] = tasks

    total_tasks = sum(len(t) for t in chunk_tasks.values())
    print(f"Dataset: {args.dataset_dir}")
    print(f"Chunks: {len(chunk_tasks)}")
    print(f"Samples: {total_tasks}")
    print(f"Output: {output_root}")

    flat_tasks = [
        task + (chunk_name,)
        for chunk_name, tasks in chunk_tasks.items()
        for task in tasks
    ]

    def _run(task: tuple) -> dict[str, Any]:
        chunk_name = task[-1]
        result = _process_one_sample(task[:-1])
        result["chunk"] = chunk_name
        return result

    global_stats: dict[str, Any] = {"chunks": {}}
    for chunk_name in chunk_tasks:
        global_stats["chunks"][chunk_name] = {
            "chunk": chunk_name,
            "total": len(chunk_tasks[chunk_name]),
            "ok": 0,
            "exists": 0,
            "errors": 0,
            "error_list": [],
            "total_dmap_pairs": 0,
            "total_interaction_pairs": 0,
            "total_tm_intervals": 0,
        }

    if args.num_workers <= 1:
        results = map(_run, flat_tasks)
    else:
        with Pool(processes=args.num_workers) as pool:
            results = pool.imap_unordered(_run, flat_tasks, chunksize=4)

    for result in results:
        stats = global_stats["chunks"][result["chunk"]]
        status = result["status"]
        if status in stats:
            stats[status] += 1
        else:
            stats["errors"] += 1
        stats["total_dmap_pairs"] += result["interface"]
        stats["total_interaction_pairs"] += result["interaction"]
        stats["total_tm_intervals"] += result["membrane"]
        if result.get("error"):
            stats["error_list"].append(
                {"case_id": result["case_id"], "error": result["error"]}
            )
            print(f"    SKIP {result['case_id']}: {result['error']}", file=sys.stderr)

    for chunk_name, stats in global_stats["chunks"].items():
        print(
            f"  {chunk_name}: {stats['ok']} ok, {stats['exists']} existing, "
            f"{stats['errors']} errors"
        )

    summary_path = output_root / "residue_generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
