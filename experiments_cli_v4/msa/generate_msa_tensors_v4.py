#!/usr/bin/env python3
"""Generate pre-computed MSA tensors for the v4 prior pipeline.

Reads msa_local/{main,paired}/{case}/{case}.npz, projects MSA columns onto GT
residues, computes profile/deletion_mean, and writes {case}_msa.pt.

Output layout mirrors the v4 mesh generator:
    {output_root}/{dataset_name}/{chunk_name}/{case_id}_msa.pt
    {output_root}/{dataset_name}/msa_generation_summary.json

The npz files in v4 are stored under msa_local/{main,paired}/{case}/ and their
internal keys may carry unpaired/ or paired/ prefixes (e.g.
"unpaired/1abc_1" or "paired/1abc_1"); parse_msas_preparsed handles them and
this script consumes the parsed MsaArray values directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openfold3.core.data.io.sequence.msa import parse_msas_preparsed
from openfold3.core.data.primitives.sequence.msa import MsaArray
from openfold3.core.data.pipelines.sample_processing.msa import calculate_profile
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
)

MAX_ROWS = 16384
MAX_ROWS_PAIRED = 8192

_RESIDUES_ARR = np.array(STANDARD_RESIDUES_WITH_GAP_1)
_CHAR_TO_IDX: dict[str, int] = {}
for _i, _r in enumerate(STANDARD_RESIDUES_WITH_GAP_1):
    if _r not in _CHAR_TO_IDX:
        _CHAR_TO_IDX[_r] = _i
_GAP_CHAR = "-"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}
_STANDARD_1 = set("ACDEFGHIKLMNPQRSTVWYX")


def _parse_gt_chain_data(gt_pdb: Path) -> dict[str, dict]:
    chain_atoms: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    with gt_pdb.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip().upper()
            residue_1 = AA3_TO_1.get(resname)
            if residue_1 is None:
                continue
            chain_id = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            insertion = line[26].strip()
            if resseq not in chain_atoms[chain_id]:
                chain_atoms[chain_id][resseq] = (residue_1, insertion)

    result: dict[str, dict] = {}
    for chain_id in sorted(chain_atoms):
        residues = sorted(chain_atoms[chain_id].items())
        sequence = "".join(info[0] for _, info in residues)
        resseq_list = [r for r, _ in residues]
        if sequence and all(c in _STANDARD_1 for c in sequence):
            result[chain_id] = {"sequence": sequence, "resseq": resseq_list}
    return result


def _msa_to_chars(msa: np.ndarray) -> np.ndarray:
    if msa.dtype == np.dtype("<U1") or msa.dtype.kind == "U":
        return msa
    if msa.dtype == np.uint8 or msa.dtype.kind in ("i", "u"):
        return _RESIDUES_ARR[msa.astype(int)]
    return msa.astype(str).astype("<U1")


def _read_msa_npz_chars(path: Path, max_rows: int | None = None) -> MsaArray:
    parsed = parse_msas_preparsed([path])
    if not parsed:
        raise ValueError(f"MSA npz does not contain any arrays: {path}")
    arrays = list(parsed.values())
    if len(arrays) == 1:
        result = arrays[0]
    else:
        result = MsaArray.multi_concatenate(arrays, axis=0)

    if max_rows is not None and len(result) > max_rows:
        result = result.truncate(max_rows)
        result.msa = result.msa.copy()

    if result.deletion_matrix.dtype == np.int64:
        result.deletion_matrix = result.deletion_matrix.astype(np.int32)
    result.msa = _msa_to_chars(result.msa)
    return result


def _msa_query_sequence(msa: MsaArray) -> str:
    if len(msa) == 0:
        return ""
    row = np.asarray(msa.msa[0])
    if row.dtype.kind == "U":
        return "".join(row.astype(str).tolist())
    return "".join(_RESIDUES_ARR[row.astype(int)].tolist())


def _project_msa_to_gt_residues(msa: MsaArray, gt_sequence: str) -> MsaArray | None:
    source = np.asarray(msa.msa)
    if source.ndim != 2 or source.shape[0] == 0 or not gt_sequence:
        return None
    if np.asarray(msa.deletion_matrix).shape != source.shape:
        return None

    source_query_chars = _msa_to_chars(source[0:1])
    source_columns = np.flatnonzero(source_query_chars[0] != _GAP_CHAR)
    source_query = "".join(source_query_chars[0, source_columns].astype(str).tolist())
    if not source_query:
        return None

    if source_query == gt_sequence and source.shape[1] == len(gt_sequence):
        return msa

    def _select_columns(allow_unknown: bool) -> list[int] | None:
        selected: list[int] = []
        target_index = 0
        for column in source_columns:
            if target_index == len(gt_sequence):
                break
            residue = str(_msa_to_chars(source[0:1, column:column + 1])[0, 0])
            target_residue = gt_sequence[target_index]
            if residue == target_residue or (allow_unknown and target_residue == "X"):
                selected.append(int(column))
                target_index += 1
        return selected if target_index == len(gt_sequence) else None

    selected_columns = _select_columns(allow_unknown=False)
    if selected_columns is None:
        selected_columns = _select_columns(allow_unknown=True)
    if selected_columns is None:
        return None

    selected = np.asarray(selected_columns, dtype=int)
    return MsaArray(
        msa=source[:, selected],
        deletion_matrix=np.asarray(msa.deletion_matrix)[:, selected],
        metadata=msa.metadata,
    )


def _find_main_npz(msa_main_dir: Path, case_id: str) -> dict[str, Path]:
    case_dir = msa_main_dir / case_id
    if case_dir.is_dir():
        search_dir = case_dir
        glob_pattern = "*.npz"
    else:
        search_dir = msa_main_dir
        glob_pattern = f"{case_id}_*.npz"

    result: dict[str, Path] = {}
    for f in sorted(search_dir.glob(glob_pattern)):
        stem = f.stem
        if stem == case_id:
            result[case_id] = f
        elif stem.startswith(case_id + "_"):
            suffix = stem[len(case_id) + 1:]
            if "chain=" in suffix:
                chain = suffix.split("chain=")[-1]
                result[chain] = f
                continue
            parts = suffix.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) <= 3 and parts[1].isalpha():
                chain = parts[1]
            elif len(parts) == 1 and len(parts[0]) <= 3 and parts[0].isalpha():
                chain = parts[0]
            else:
                chain = suffix
            result[chain] = f
    return result


def _find_paired_npz(msa_paired_dir: Path, case_id: str) -> dict[str, Path]:
    return _find_main_npz(msa_paired_dir, case_id)


def _process_one_sample_data(
    case_id: str,
    gt_pdb: Path,
    msa_main_dir: Path,
    msa_paired_dir: Path,
) -> dict | None:
    chain_data = _parse_gt_chain_data(gt_pdb)
    if not chain_data:
        return None
    chain_ids = sorted(chain_data)

    main_files = _find_main_npz(msa_main_dir, case_id)
    paired_files = _find_paired_npz(msa_paired_dir, case_id)

    main_msa: dict[str, np.ndarray] = {}
    main_deletion: dict[str, np.ndarray] = {}
    main_loaded: dict[str, MsaArray] = {}

    for chain_id in chain_ids:
        npz_path = main_files.get(chain_id, main_files.get(case_id))
        if npz_path is None:
            continue
        try:
            msa = _read_msa_npz_chars(npz_path, max_rows=MAX_ROWS)
        except Exception:
            continue
        projected = _project_msa_to_gt_residues(msa, chain_data[chain_id]["sequence"])
        if projected is None:
            n_res = len(chain_data[chain_id]["sequence"])
            seq = chain_data[chain_id]["sequence"]
            projected = MsaArray(
                msa=np.array([list(seq)], dtype="<U1"),
                deletion_matrix=np.zeros((1, n_res), dtype=np.int32),
                metadata=[],
            )
        main_loaded[chain_id] = projected
        main_msa[chain_id] = np.asarray(projected.msa)
        main_deletion[chain_id] = np.asarray(projected.deletion_matrix)

    if not main_msa:
        return None

    paired_msa: dict[str, np.ndarray] = {}
    paired_deletion: dict[str, np.ndarray] = {}
    for chain_id in chain_ids:
        npz_path = paired_files.get(chain_id, paired_files.get(case_id))
        if npz_path is None:
            continue
        try:
            msa = _read_msa_npz_chars(npz_path, max_rows=MAX_ROWS_PAIRED)
        except Exception:
            continue
        if msa.msa.shape[0] <= 1 and msa.msa.shape[1] <= 1:
            continue
        projected = _project_msa_to_gt_residues(msa, chain_data[chain_id]["sequence"])
        if projected is None:
            continue
        paired_msa[chain_id] = np.asarray(projected.msa)
        paired_deletion[chain_id] = np.asarray(projected.deletion_matrix)

    profile: dict[str, np.ndarray] = {}
    deletion_mean: dict[str, np.ndarray] = {}
    for chain_id in chain_ids:
        if chain_id in main_loaded:
            profile[chain_id] = calculate_profile(
                msa_array=main_loaded[chain_id].msa,
                molecule_type=MoleculeType.PROTEIN,
                chunk_size=1000,
            ).astype(np.float32)
            deletion_mean[chain_id] = np.mean(
                main_loaded[chain_id].deletion_matrix, axis=0
            ).astype(np.float32)
        else:
            n_res = len(chain_data[chain_id]["sequence"])
            profile[chain_id] = np.zeros((n_res, 32), dtype=np.float32)
            deletion_mean[chain_id] = np.zeros(n_res, dtype=np.float32)

    return {
        "chain_ids": chain_ids,
        "main_msa": main_msa,
        "main_deletion": main_deletion,
        "paired_msa": paired_msa,
        "paired_deletion": paired_deletion,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "resseq": {c: chain_data[c]["resseq"] for c in chain_ids},
    }


def _process_one_sample(args: tuple) -> dict[str, Any]:
    case_id, gt_pdb, msa_main_dir, msa_paired_dir, output_path, force = args
    if output_path.exists() and not force:
        return {"case_id": case_id, "status": "exists", "error": None}
    try:
        data = _process_one_sample_data(case_id, gt_pdb, msa_main_dir, msa_paired_dir)
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if data is None:
        return {"case_id": case_id, "status": "skipped", "error": None}
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(data, tmp_path)
        tmp_path.replace(output_path)
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": f"save failed: {exc}"}
    finally:
        del data
        gc.collect()
    return {"case_id": case_id, "status": "ok", "error": None}


def collect_tasks(
    chunk_dir: Path,
    output_chunk_dir: Path,
    max_samples: int | None,
    force: bool,
) -> list[tuple]:
    msa_main_dir = chunk_dir / "msa_local" / "main"
    msa_paired_dir = chunk_dir / "msa_local" / "paired"
    gt_dir = chunk_dir / "gt"
    if not msa_main_dir.is_dir():
        return []

    case_ids = sorted({d.name for d in msa_main_dir.iterdir() if d.is_dir()})
    if not case_ids:
        case_ids = sorted({f.stem for f in gt_dir.glob("*.pdb") if f.is_file()})
    if max_samples is not None:
        case_ids = case_ids[:max_samples]

    tasks = []
    for case_id in case_ids:
        gt_pdb = gt_dir / f"{case_id}.pdb"
        if not gt_pdb.is_file():
            continue
        output_path = output_chunk_dir / f"{case_id}_msa.pt"
        tasks.append(
            (case_id, gt_pdb, msa_main_dir, msa_paired_dir, output_path, force)
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed MSA .pt tensors for cli_v4."
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
        if not (chunk_dir / "msa_local" / "main").is_dir():
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
            "skipped": 0,
            "errors": 0,
            "error_list": [],
        }

    if args.num_workers <= 1:
        results = map(_run, flat_tasks)
    else:
        with Pool(processes=args.num_workers) as pool:
            results = pool.imap_unordered(_run, flat_tasks, chunksize=4)

    for result in results:
        stats = global_stats["chunks"][result["chunk"]]
        status = result["status"]
        stats[status if status in stats else "errors"] += 1
        if result.get("error"):
            stats["error_list"].append({"case_id": result["case_id"], "error": result["error"]})
            print(f"    SKIP {result['case_id']}: {result['error']}", file=sys.stderr)

    for chunk_name, stats in global_stats["chunks"].items():
        print(
            f"  {chunk_name}: {stats['ok']} ok, {stats['exists']} existing, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )

    summary_path = output_root / "msa_generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
