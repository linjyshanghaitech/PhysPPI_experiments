#!/usr/bin/env python3
"""Generate pre-computed PDB tensors for the v4 prior pipeline.

For each sample produces:
    {case_id}_gt.pt                — canonized GT AtomArray (all heavy atoms)
    {case_id}_{chain}_monomer.pt   — per-chain processed monomer AtomArray
    {case_id}_seq.pt               — per-chain sequence metadata + derived
                                     resseq_to_sequence_index

The v4 final chunk layout has no per-chain PDB files; monomers are split from
the all-atom complex PDB at chunk/input/{case_id}.pdb by chain.  H atoms are
removed during canonicalization; all heavy atoms are preserved.

Output layout:
    {output_root}/{dataset_name}/{chunk_name}/{case_id}_{suffix}.pt
    {output_root}/{dataset_name}/pdb_generation_summary.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PRIOR_SRC = Path("/101045/jinyu/PPI/Prior/src")
if str(_PRIOR_SRC) not in sys.path:
    sys.path.insert(0, str(_PRIOR_SRC))

import biotite.structure as struc  # noqa: E402
from biotite.structure.io import pdbx  # noqa: E402
from biotite.structure.io.pdb.file import encode_hybrid36  # noqa: E402

from openfold3.core.data.io.structure.mol import (  # noqa: E402
    read_single_annotated_sdf,
)
from openfold3.core.data.io.structure.pdb import parse_pdb_af2  # noqa: E402
from openfold3.core.data.pipelines.sample_processing.structure import (  # noqa: E402
    process_target_atom_array_of3,
)
from openfold3.core.data.primitives.caches.format import (  # noqa: E402
    PreprocessingChainData,
)
from openfold3.core.data.primitives.structure.cleanup import (  # noqa: E402
    convert_MSE_to_MET,
    remove_hydrogens,
    remove_std_residue_terminal_atoms,
)
from openfold3.core.data.primitives.structure.tokenization import (  # noqa: E402
    add_token_positions,
)
from openfold3.core.data.resources.residues import MoleculeType  # noqa: E402
from experiments.pdb_cache.pdb_adapter import save_atomarray_as_pt  # noqa: E402

_PROTEIN_REFERENCE_MOLS = (
    _REPO_ROOT
    / "openfold3/core/data/resources/canonical_protein_data/reference_mols"
)

_AA3_TO_1: dict[str, str] = {}
try:
    from physppi_prior.aa_map import AA_MAP as _AA3_TO_1  # noqa: E402
except ImportError:
    _AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
        "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
        "VAL": "V",
    }

# Worker-global CCD (loaded once per worker, lazily)
_CCD_PATH: str | None = None
_CCD_CACHE: Any = None


@functools.lru_cache(maxsize=64)
def _canonical_atom_names(res_name: str) -> tuple[str, ...]:
    mol = read_single_annotated_sdf(
        _PROTEIN_REFERENCE_MOLS / f"{res_name}.sdf"
    )
    return tuple(
        atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()
    )


def _parse_pdb(path: Path):
    try:
        parsed = parse_pdb_af2(path, allow_malformed_charge=True)
    except Exception as exc:
        # Some OpenMM-written PDBs have wrapped/duplicated ATOM serials (biotite
        # requires strictly increasing ids) and/or merged occupancy/bfactor
        # columns.  Repair a temporary copy and parse that instead; the original
        # file is never modified.
        fixed_path = _repair_pdb(path)
        try:
            parsed = parse_pdb_af2(fixed_path, allow_malformed_charge=True)
        except Exception:
            raise exc
        finally:
            fixed_path.unlink(missing_ok=True)
    if not hasattr(parsed, "atom_array"):
        raise ValueError(f"failed to parse PDB: {path}")
    atom_array = parsed.atom_array
    if len(atom_array) > 0 and "occupancy" in atom_array.get_annotation_categories():
        occupancy = np.asarray(atom_array.occupancy)
        if not np.any(occupancy > 0):
            atom_array.occupancy[:] = 1.0
    return atom_array


def _renumber_atom_ids(path: Path) -> Path:
    """Return a temp PDB copy with sequential ATOM/HETATM serial numbers."""
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    serial = 0
    out_lines: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
            serial += 1
            line = line[:6] + f"{serial:5d}" + line[11:]
        out_lines.append(line)
    fd, tmp_name = tempfile.mkstemp(prefix="atomid_fix_", suffix=".pdb")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("".join(out_lines))
    return Path(tmp_name)


_TAIL_FLOAT_PAIR = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)"
)


def _repair_pdb_lines(lines: list[str]) -> list[str]:
    """Repair OpenMM-style atom lines in place (hybrid36 serials, no CONECT).

    - ATOM/HETATM serials are renumbered strictly increasing with biotite's
      5-character hybrid36 encoding (supports >99,999 atoms);
    - CONECT records are dropped (old serials are ambiguous after wrapping;
      bonds are inferred/rebuilt downstream);
    - merged occupancy/tempFactor fields like ``1.00144.63`` are split into
      correct fixed-width columns.
    """

    out: list[str] = []
    serial = 0
    for line in lines:
        if line.startswith("CONECT"):
            continue
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            out.append(line)
            continue
        serial += 1
        fixed = f"{line[:6]}{encode_hybrid36(serial, 5).rjust(5)}{line[11:]}"
        tail = fixed[54:]
        match = _TAIL_FLOAT_PAIR.match(tail)
        if match:
            try:
                occupancy = float(match.group(1))
                temp_factor = float(match.group(2))
            except ValueError:
                out.append(fixed)
                continue
            rest = tail[match.end() :]
            fixed = fixed[:54] + f"{occupancy:6.2f}{temp_factor:6.2f}" + rest
        out.append(fixed)
    return out


def _repair_pdb(path: Path) -> Path:
    """Write a repaired temporary copy and return its path."""

    lines = path.read_text(errors="replace").splitlines(keepends=True)
    repaired = _repair_pdb_lines(lines)
    fd, tmp_name = tempfile.mkstemp(prefix="pdb_repair_", suffix=".pdb")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("".join(repaired))
    return Path(tmp_name)


def canonize_atom_array(atom_array, *, ccd=None):
    """Canonize protein heavy atoms, preserving non-standard residues.

    Standard residues are filtered to their reference-mol atom names and
    reordered to the reference atom order (identical to the CCD order for
    standard residues).  Non-standard residues (modified amino acids such as
    FME/L2C/SEP) are preserved: when a CCD entry exists and matches the PDB
    atom names they are filtered and ordered by CCD (previous behavior);
    otherwise all heavy atoms are kept in original order so the residue never
    disappears (OF3 atomized-token semantics).
    """
    aa = atom_array.copy()
    convert_MSE_to_MET(aa)
    aa = aa[aa.molecule_type_id == MoleculeType.PROTEIN]
    if len(aa) == 0:
        raise ValueError("no protein heavy atoms")
    aa = remove_hydrogens(aa)
    aa = remove_std_residue_terminal_atoms(aa)

    starts = struc.get_residue_starts(aa)
    stops = np.append(starts[1:], len(aa))
    final_idx: list[int] = []

    for s, e in zip(starts, stops):
        rn = str(aa.res_name[s])
        idx = np.arange(s, e)
        ref_path = _PROTEIN_REFERENCE_MOLS / f"{rn}.sdf"
        if ref_path.is_file():
            allowed = _canonical_atom_names(rn)
            residue_keep = np.isin(aa.atom_name[s:e], allowed)
            kept_idx = idx[residue_keep]
            order = np.argsort(
                [allowed.index(name) for name in aa.atom_name[kept_idx]]
            )
            final_idx.extend(kept_idx[order].tolist())
            continue

        # Non-standard residue: prefer CCD filtering/ordering when the CCD
        # entry exists and matches at least one atom; otherwise keep the
        # whole residue (all heavy atoms, original order).
        ccd_mask = None
        if ccd is not None and rn in ccd:
            ccd_entry = ccd[rn]
            if "chem_comp_atom" in ccd_entry:
                ccd_atoms = ccd_entry["chem_comp_atom"]["atom_id"].as_array()
                match = np.isin(aa.atom_name[s:e], ccd_atoms)
                if np.any(match):
                    ccd_mask = match
        if ccd_mask is None:
            final_idx.extend(idx.tolist())
        else:
            kept_idx = idx[ccd_mask]
            ccd_order = ccd[rn]["chem_comp_atom"]["atom_id"].as_array().tolist()
            order = np.argsort(
                [ccd_order.index(name) for name in aa.atom_name[kept_idx]]
            )
            final_idx.extend(kept_idx[order].tolist())

    aa = aa[final_idx]
    if len(aa) == 0:
        raise ValueError("no heavy atoms after cleanup")
    return aa


def annotate_entity_ids(atom_array):
    chains = sorted(set(str(c) for c in atom_array.chain_id))
    seq_to_eid: dict[str, int] = {}
    entity_ids = np.zeros(len(atom_array), dtype=np.int32)
    for ch in chains:
        mask = np.asarray(atom_array.chain_id).astype(str) == ch
        ch_aa = atom_array[mask]
        ch_starts = struc.get_residue_starts(ch_aa)
        seq = "".join(
            _AA3_TO_1.get(str(ch_aa.res_name[s]), "X")
            for s in ch_starts
        )
        if seq not in seq_to_eid:
            seq_to_eid[seq] = len(seq_to_eid) + 1
        entity_ids[np.flatnonzero(mask)] = seq_to_eid[seq]
    atom_array.set_annotation("entity_id", entity_ids)
    atom_array.set_annotation("mol_entity_id", entity_ids.copy())
    return atom_array


def build_chain_metadata(atom_array) -> dict[str, PreprocessingChainData]:
    return {
        str(c): PreprocessingChainData(
            label_asym_id=str(c),
            auth_asym_id=str(c),
            entity_id=1,
            molecule_type=MoleculeType.PROTEIN,
            reference_mol_id=None,
        )
        for c in np.unique(atom_array.chain_id)
    }


def _needs_ccd(atom_array) -> bool:
    protein = atom_array[atom_array.molecule_type_id == MoleculeType.PROTEIN]
    residue_names: set[str] = set(map(str, protein.res_name))
    residue_names.discard("MSE")
    return any(
        not (_PROTEIN_REFERENCE_MOLS / f"{name}.sdf").is_file()
        for name in residue_names
    )


def _init_worker(ccd_path: str | None) -> None:
    global _CCD_PATH
    _CCD_PATH = ccd_path


def _worker_ccd(atom_array):
    if not _CCD_PATH:
        return None
    if not _needs_ccd(atom_array):
        return None
    global _CCD_CACHE
    if _CCD_CACHE is None:
        _CCD_CACHE = pdbx.CIFFile.read(Path(_CCD_PATH))
    return _CCD_CACHE


def process_monomer_array(raw_chain, case_id: str, ccd):
    canonized = canonize_atom_array(raw_chain, ccd=ccd)
    annotated = annotate_entity_ids(canonized)
    processed = process_target_atom_array_of3(
        atom_array=annotated,
        pdb_id=case_id,
        crop_config={
            "chain_crop": {"enabled": False},
            "token_crop": {"enabled": False},
        },
        preferred_chain_or_interface=None,
        per_chain_metadata=build_chain_metadata(annotated),
    )
    add_token_positions(processed.atom_array_gt)
    return processed.atom_array_gt


def process_gt_array(raw_gt, chain_ids: list[str], ccd):
    restricted = raw_gt[
        np.isin(np.asarray(raw_gt.chain_id).astype(str), chain_ids)
    ]
    canonized = canonize_atom_array(restricted, ccd=ccd)
    annotated = annotate_entity_ids(canonized)
    return annotated


def parse_chain_residues(atom_array) -> dict[str, list[tuple[int, str]]]:
    """Return {chain_id: [(resseq, resname_3), ...]} in first-appearance order."""
    result: dict[str, list[tuple[int, str]]] = {}
    chains = sorted(set(str(c) for c in np.asarray(atom_array.chain_id).astype(str)))
    for ch in chains:
        mask = np.asarray(atom_array.chain_id).astype(str) == ch
        ch_aa = atom_array[mask]
        starts = struc.get_residue_starts(ch_aa)
        residues: list[tuple[int, str]] = []
        for s in starts:
            residues.append(
                (int(ch_aa.res_id[s]), str(ch_aa.res_name[s]).strip().upper())
            )
        result[ch] = residues
    return result


def derive_resseq_to_sequence_index(
    chain_residues: list[tuple[int, str]],
    seq_3aa: list[str],
) -> dict[int, int]:
    """Greedy 3-letter name matching, mirroring the v3 mesh fallback."""
    mapping: dict[int, int] = {}
    r_idx = 0
    for seq_idx, seq_res in enumerate(seq_3aa):
        while r_idx < len(chain_residues):
            rseq, rname = chain_residues[r_idx]
            r_idx += 1
            if rname == seq_res:
                mapping[rseq] = seq_idx
                break
    if not mapping and len(chain_residues) == len(seq_3aa):
        mapping = {rseq: i for i, (rseq, _) in enumerate(chain_residues)}
    return mapping


def extract_seq_sample(
    seq_cache: dict,
    case_id: str,
    input_aa,
) -> dict[str, Any] | None:
    case_seqs = seq_cache.get(case_id, {})
    if not case_seqs:
        return None
    chain_residues = parse_chain_residues(input_aa)
    result: dict[str, Any] = {}
    for chain_id, info in case_seqs.items():
        entry: dict[str, Any] = {
            "full_seq_1aa": info.get("full_seq_1aa", ""),
            "full_seq_3aa": info.get("full_seq_3aa", []),
        }
        r2s = derive_resseq_to_sequence_index(
            chain_residues.get(str(chain_id), []),
            entry["full_seq_3aa"],
        )
        if r2s:
            entry["resseq_to_sequence_index"] = {
                int(k): int(v) for k, v in r2s.items()
            }
        for opt_key in ("mask", "msa_eligible"):
            if opt_key in info:
                entry[opt_key] = info[opt_key]
        result[str(chain_id)] = entry
    return result


def _load_seq_cache(chunk_dir: Path) -> dict:
    path = chunk_dir / "seq_cache.pkl"
    if not path.is_file():
        return {}
    return pickle.loads(path.read_bytes())


def _process_one_sample(args: tuple) -> dict[str, Any]:
    chunk_dir, case_id, output_dir, force = args
    gt_pdb = chunk_dir / "gt" / f"{case_id}.pdb"
    input_pdb = chunk_dir / "input" / f"{case_id}.pdb"
    if not input_pdb.is_file():
        return {"case_id": case_id, "status": "skipped", "error": "no input pdb"}
    try:
        raw_gt = _parse_pdb(gt_pdb) if gt_pdb.is_file() else None
        raw_input = _parse_pdb(input_pdb)
        chain_ids = sorted(set(str(c) for c in np.unique(raw_input.chain_id)))
        if not chain_ids:
            return {"case_id": case_id, "status": "skipped", "error": "no chains"}

        gt_pt = output_dir / f"{case_id}_gt.pt"
        if not gt_pt.exists() or force:
            if raw_gt is None:
                raise ValueError("missing gt pdb")
            ccd = _worker_ccd(raw_gt)
            gt_aa = process_gt_array(raw_gt, chain_ids, ccd)
            save_atomarray_as_pt(gt_aa, gt_pt)

        for chain_id in chain_ids:
            mono_pt = output_dir / f"{case_id}_{chain_id}_monomer.pt"
            if mono_pt.exists() and not force:
                continue
            mask = np.asarray(raw_input.chain_id).astype(str) == chain_id
            raw_chain = raw_input[mask]
            ccd = _worker_ccd(raw_chain)
            mono_aa = process_monomer_array(raw_chain, case_id, ccd)
            save_atomarray_as_pt(mono_aa, mono_pt)

        seq_pt = output_dir / f"{case_id}_seq.pt"
        if not seq_pt.exists() or force:
            seq_cache = _load_seq_cache(chunk_dir)
            seq_data = extract_seq_sample(seq_cache, case_id, raw_input)
            if seq_data:
                tmp = seq_pt.with_suffix(".pt.tmp")
                torch.save(seq_data, tmp)
                tmp.replace(seq_pt)
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"case_id": case_id, "status": "ok", "error": None}


def _run_sample(task: tuple) -> dict[str, Any]:
    """Module-level worker for multiprocessing (local closures are not picklable)."""
    chunk_name = task[-1]
    result = _process_one_sample(task[:-1])
    result["chunk"] = chunk_name
    return result


def collect_tasks(
    chunk_dir: Path,
    output_chunk_dir: Path,
    max_samples: int | None,
    force: bool,
) -> list[tuple]:
    input_dir = chunk_dir / "input"
    if not input_dir.is_dir():
        return []
    case_ids = sorted({p.stem for p in input_dir.glob("*.pdb") if p.is_file()})
    if max_samples is not None:
        case_ids = case_ids[:max_samples]
    return [
        (chunk_dir, case_id, output_chunk_dir, force)
        for case_id in case_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed PDB tensors (GT + monomer + seq) for cli_v4."
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
    parser.add_argument("--ccd-file", type=Path, default=None)
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
        if not (chunk_dir / "input").is_dir():
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
    print(f"CCD: {args.ccd_file}")

    flat_tasks = [
        task + (chunk_name,)
        for chunk_name, tasks in chunk_tasks.items()
        for task in tasks
    ]

    global_stats: dict[str, Any] = {"chunks": {}}
    for chunk_name in chunk_tasks:
        global_stats["chunks"][chunk_name] = {
            "chunk": chunk_name,
            "total": len(chunk_tasks[chunk_name]),
            "ok": 0,
            "skipped": 0,
            "errors": 0,
            "error_list": [],
        }

    if args.num_workers <= 1:
        # Sequential mode has no Pool initializer; set the worker-global CCD
        # path explicitly so _worker_ccd() can lazy-load it when needed.
        _init_worker(str(args.ccd_file) if args.ccd_file is not None else None)
        results = map(_run_sample, flat_tasks)
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(str(args.ccd_file) if args.ccd_file is not None else None,),
        ) as executor:
            results = executor.map(_run_sample, flat_tasks, chunksize=16)

    for result in results:
        stats = global_stats["chunks"][result["chunk"]]
        status = result["status"]
        if status in stats:
            stats[status] += 1
        else:
            stats["errors"] += 1
        if result.get("error"):
            stats["error_list"].append(
                {"case_id": result["case_id"], "error": result["error"]}
            )
            print(f"    SKIP {result['case_id']}: {result['error']}", file=sys.stderr)

    for chunk_name, stats in global_stats["chunks"].items():
        print(
            f"  {chunk_name}: {stats['ok']} ok, {stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )

    summary_path = output_root / "pdb_generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
