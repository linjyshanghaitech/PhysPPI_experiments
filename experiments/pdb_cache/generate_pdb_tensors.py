#!/usr/bin/env python3
"""Generate pre-computed PDB tensors for PhysPPI training acceleration.

For each sample, produces per-chunk cache files:

    {case_id}_gt.pt                     — canonized GT AtomArray (complex coords)
    {case_id}_{chain_id}_monomer.pt     — processed monomer AtomArray per chain
    {case_id}_seq.pt                    — per-chain sequence metadata

Goal: skip PDB text parsing + atom canonicalization during training.

- GT:  parsed → chain-restricted → canonized → entity-annotated.
       Stopped BEFORE ``process_target_atom_array_of3`` so cropping still
       happens at training time (crop depends on runtime config).
- Monomer: parsed → canonized → entity-annotated → processed (no-crop)
       → token_positions added.  Full AtomArray preserved via
       ``save_atomarray_as_pt`` so ``build_monomer_input_geometry`` and
       template featurization both work.
- Seq:   per-chain ``full_seq_1aa``, ``full_seq_3aa``,
       ``resseq_to_sequence_index`` extracted from chunk-level seq_cache.pkl.

Directory layout mirrors the existing mesh / residue prior caches:

    {output_dir}/{dataset_name}/{chunk_name}/
        {case_id}_gt.pt
        {case_id}_A_monomer.pt
        {case_id}_B_monomer.pt
        {case_id}_seq.pt

Requires: PhysPPI conda env (biotite + openfold3 imports).
Does NOT import from _runtime.py (avoids CUDA library dependency).
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# -- Ensure the PhysPPI repo root is on sys.path for openfold3 imports --
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also add the Prior package path (for aa_map)
_PRIOR_SRC = Path("/101045/jinyu/PPI/Prior/src")
if str(_PRIOR_SRC) not in sys.path:
    sys.path.insert(0, str(_PRIOR_SRC))

import biotite.structure as struc  # noqa: E402
from biotite.structure.io import pdbx  # noqa: E402

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
    canonicalize_atom_order,
    convert_MSE_to_MET,
    remove_hydrogens,
    remove_non_CCD_atoms,
    remove_std_residue_terminal_atoms,
)
from openfold3.core.data.primitives.structure.tokenization import (  # noqa: E402
    add_token_positions,
)
from openfold3.core.data.resources.residues import MoleculeType  # noqa: E402
from experiments.pdb_cache.pdb_adapter import save_atomarray_as_pt  # noqa: E402

# --- Constants ---------------------------------------------------------------

_PROTEIN_REFERENCE_MOLS = (
    _REPO_ROOT
    / "openfold3/core/data/resources/canonical_protein_data/reference_mols"
)

_AA3_TO_1: dict[str, str] = {}

try:
    from physppi_prior.aa_map import AA_MAP as _AA3_TO_1  # noqa: E402
except ImportError:
    # Fallback: standard amino acids only (should not happen in production)
    _AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
        "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
        "VAL": "V",
    }

# --- Canonical atom-name cache -----------------------------------------------


@functools.lru_cache(maxsize=64)
def _canonical_atom_names(res_name: str) -> frozenset[str]:
    """Return the set of allowed atom names for a canonical residue."""
    mol = read_single_annotated_sdf(
        _PROTEIN_REFERENCE_MOLS / f"{res_name}.sdf"
    )
    return frozenset(
        atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()
    )


# --- PDB parsing -------------------------------------------------------------


def _parse_pdb(path: Path):
    """Parse a PDB file into a biotite AtomArray."""
    parsed = parse_pdb_af2(path, allow_malformed_charge=True)
    if not hasattr(parsed, "atom_array"):
        raise ValueError(f"failed to parse PDB: {path}")
    return parsed.atom_array


# --- Atom canonicalization (inline, avoids _runtime.py CUDA import chain) ----


def canonize_atom_array(atom_array, *, ccd=None):
    """Canonicalize heavy atoms: remove H, convert MSE→MET, filter to reference.

    Replicates ``_canonical_ppi_heavy_atom_array`` from ``_runtime.py``
    without triggering the full model import chain.
    """
    aa = atom_array.copy()
    convert_MSE_to_MET(aa)
    aa = aa[aa.molecule_type_id == MoleculeType.PROTEIN]
    if len(aa) == 0:
        raise ValueError("no protein heavy atoms")
    aa = remove_hydrogens(aa)
    aa = remove_std_residue_terminal_atoms(aa)

    keep = np.ones(len(aa), dtype=bool)
    needs_ccd = False
    noncanon: set[str] = set()
    starts = struc.get_residue_starts(aa)
    stops = np.append(starts[1:], len(aa))

    for s, e in zip(starts, stops):
        rn = str(aa.res_name[s])
        ref_path = _PROTEIN_REFERENCE_MOLS / f"{rn}.sdf"
        if not ref_path.is_file():
            needs_ccd = True
            noncanon.add(rn)
            continue
        allowed = _canonical_atom_names(rn)
        keep[s:e] = np.isin(aa.atom_name[s:e], list(allowed))

    aa = aa[keep]
    if needs_ccd:
        if ccd is None:
            raise ValueError(
                "non-canonical residues require CCD: " + ", ".join(sorted(noncanon))
            )
        aa = remove_non_CCD_atoms(aa, ccd)
        aa = canonicalize_atom_order(aa, ccd)
    if len(aa) == 0:
        raise ValueError("no heavy atoms after cleanup")
    return aa


def annotate_entity_ids(atom_array):
    """Add entity_id / mol_entity_id annotations.

    Replicates ``_annotate_canonical_mol_entity_ids`` without the
    sample argument (sequences are inferred from the atom array directly).
    """
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
    """Build per-chain PreprocessingChainData for process_target_atom_array_of3."""
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


# --- CCD helpers -------------------------------------------------------------


def _needs_ccd(atom_array) -> bool:
    """Return True if any residue lacks a reference SDF."""
    protein = atom_array[atom_array.molecule_type_id == MoleculeType.PROTEIN]
    residue_names: set[str] = set(map(str, protein.res_name))
    residue_names.discard("MSE")
    return any(
        not (_PROTEIN_REFERENCE_MOLS / f"{name}.sdf").is_file()
        for name in residue_names
    )


def _load_ccd(ccd_path: Path):
    if not ccd_path.is_file():
        raise FileNotFoundError(f"CCD file not found: {ccd_path}")
    return pdbx.CIFFile.read(ccd_path)


def _resolve_ccd(atom_array, ccd_path: Path | None):
    """Load CCD if needed, else return None."""
    if ccd_path is not None and _needs_ccd(atom_array):
        return _load_ccd(ccd_path)
    return None


# --- Monomer processing ------------------------------------------------------


def process_monomer_pdb(pdb_path: Path, ccd) -> "AtomArray":
    """Parse → canonize → entity-id → process (no-crop) → token_positions."""
    raw = _parse_pdb(pdb_path)
    canonized = canonize_atom_array(raw, ccd=ccd)
    annotated = annotate_entity_ids(canonized)
    processed = process_target_atom_array_of3(
        atom_array=annotated,
        pdb_id=pdb_path.stem,
        crop_config={
            "chain_crop": {"enabled": False},
            "token_crop": {"enabled": False},
        },
        preferred_chain_or_interface=None,
        per_chain_metadata=build_chain_metadata(annotated),
    )
    add_token_positions(processed.atom_array_gt)
    return processed.atom_array_gt


# --- GT processing -----------------------------------------------------------


def process_gt_pdb(
    gt_pdb: Path,
    case_id: str,
    monomer_pdb_dir: Path,
    ccd,
) -> "AtomArray":
    """Parse → chain-restrict → canonize → entity-annotate (pre-crop)."""
    raw = _parse_pdb(gt_pdb)

    # Discover chains from monomer PDBs
    chain_ids = sorted({
        p.stem.rsplit("_", 1)[-1]
        for p in monomer_pdb_dir.glob(f"{case_id}_*.pdb")
    })
    if not chain_ids:
        chain_ids = sorted(set(str(c) for c in np.unique(raw.chain_id)))

    # Restrict to sample chains
    restricted = raw[
        np.isin(np.asarray(raw.chain_id).astype(str), chain_ids)
    ]
    canonized = canonize_atom_array(restricted, ccd=ccd)
    annotated = annotate_entity_ids(canonized)
    return annotated


# --- Seq cache ---------------------------------------------------------------


def extract_seq_sample(
    seq_cache: dict, case_id: str
) -> dict[str, Any] | None:
    """Extract per-chain sequence info for one sample."""
    case_seqs = seq_cache.get(case_id, {})
    if not case_seqs:
        return None
    result: dict[str, Any] = {}
    for chain_id, info in case_seqs.items():
        entry: dict[str, Any] = {
            "full_seq_1aa": info.get("full_seq_1aa", ""),
            "full_seq_3aa": info.get("full_seq_3aa", []),
        }
        r2s = info.get("resseq_to_sequence_index", {})
        if r2s:
            entry["resseq_to_sequence_index"] = {
                int(k): int(v) for k, v in r2s.items()
            }
        for opt_key in ("mask", "msa_eligible"):
            if opt_key in info:
                entry[opt_key] = info[opt_key]
        result[str(chain_id)] = entry
    return result


# --- Chunk-level runner (single-process per chunk) --------------------------


def _run_one_chunk(args: tuple) -> dict[str, Any]:
    """Worker entry point for ProcessPoolExecutor (one chunk per worker)."""
    chunk_dir, output_dir, ccd_path, max_samples = args
    return run_generation(
        chunk_dir=chunk_dir,
        output_dir=output_dir,
        ccd_path=ccd_path,
        max_samples=max_samples,
        num_workers=0,  # sequential inside the worker
    )


def run_generation(
    chunk_dir: Path,
    output_dir: Path,
    ccd_path: Path | None,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Generate cache files for one chunk (single-process inside chunk)."""
    gt_dir = chunk_dir / "gt"
    bb_input_dir = chunk_dir / "bb_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shared seq_cache for this chunk
    seq_cache_path = chunk_dir / "seq_cache.pkl"
    seq_cache: dict = {}
    if seq_cache_path.exists():
        seq_cache = pickle.loads(seq_cache_path.read_bytes())

    # Discover samples
    gt_paths = sorted(gt_dir.glob("*.pdb"))
    if max_samples is not None:
        gt_paths = gt_paths[:max_samples]

    stats: dict[str, Any] = {
        "chunk": str(chunk_dir),
        "total": len(gt_paths),
        "processed": 0,
        "skipped_error": 0,
        "errors": [],
    }

    ccd = None  # lazy-loaded per chunk

    for gt_pdb in gt_paths:
        case_id = gt_pdb.stem
        monomer_pdbs = sorted(bb_input_dir.glob(f"{case_id}_*.pdb"))
        if not monomer_pdbs:
            continue

        try:
            # ---- GT ----
            gt_pt_path = output_dir / f"{case_id}_gt.pt"
            if not gt_pt_path.exists():
                raw_gt = _parse_pdb(gt_pdb)
                if ccd is None:
                    ccd = _resolve_ccd(raw_gt, ccd_path)

                gt_aa = process_gt_pdb(
                    gt_pdb, case_id, monomer_pdbs[0].parent, ccd
                )
                save_atomarray_as_pt(gt_aa, gt_pt_path)

            # ---- Monomers (one .pt per chain) ----
            for chain_pdb in monomer_pdbs:
                chain_id = chain_pdb.stem.rsplit("_", 1)[-1]
                mono_pt_path = output_dir / f"{case_id}_{chain_id}_monomer.pt"
                if mono_pt_path.exists():
                    continue

                raw_mono = _parse_pdb(chain_pdb)
                if ccd is None:
                    ccd = _resolve_ccd(raw_mono, ccd_path)

                mono_aa = process_monomer_pdb(chain_pdb, ccd)
                save_atomarray_as_pt(mono_aa, mono_pt_path)

            # ---- Seq cache ----
            seq_pt_path = output_dir / f"{case_id}_seq.pt"
            if not seq_pt_path.exists():
                seq_data = extract_seq_sample(seq_cache, case_id)
                if seq_data:
                    tmp = seq_pt_path.with_suffix(".pt.tmp")
                    torch.save(seq_data, tmp)
                    tmp.replace(seq_pt_path)

            stats["processed"] += 1

        except Exception as exc:
            stats["skipped_error"] += 1
            stats["errors"].append({
                "case_id": case_id,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return stats


# --- Main --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed PDB tensors (GT + monomer + seq)."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Dataset root (e.g., /path/to/AFDB-10k-1).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output root for .pt files (e.g., .../precomputed_mesh/cli_v3/AFDB-10k-1).",
    )
    parser.add_argument(
        "--ccd-file", type=Path, default=None,
        help="CCD mmCIF file for non-canonical residues (optional for AFDB).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap total samples (for testing).",
    )
    parser.add_argument(
        "--chunks", type=str, default=None,
        help="Comma-separated chunk names (default: all).",
    )
    parser.add_argument(
        "--workers", type=int, default=64,
        help="Parallel chunk workers (default: 64).",
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

    # Build work items: one per chunk
    work_items: list[tuple] = []
    for chunk_dir in all_chunk_dirs:
        gt_dir = chunk_dir / "gt"
        if not gt_dir.is_dir():
            continue
        n_samples = len(list(gt_dir.glob("*.pdb")))
        if n_samples == 0:
            continue
        chunk_output_dir = args.output_dir / chunk_dir.name
        work_items.append((
            chunk_dir, chunk_output_dir, args.ccd_file, args.max_samples,
        ))

    print(f"Dataset:      {args.dataset_dir}")
    print(f"Output:       {args.output_dir}")
    print(f"Chunks:       {len(work_items)}")
    print(f"Workers:      {args.workers}")
    print(f"CCD file:     {args.ccd_file}")
    print()

    global_stats: dict[str, Any] = {"chunks": {}}
    completed = 0

    if args.workers <= 1 or len(work_items) <= 1:
        # Sequential
        for chunk_dir, chunk_output_dir, ccd_path, max_samp in work_items:
            stats = _run_one_chunk((
                chunk_dir, chunk_output_dir, ccd_path, max_samp,
            ))
            global_stats["chunks"][chunk_dir.name] = stats
            completed += 1
            print(
                f"[{completed}/{len(work_items)}] {chunk_dir.name}: "
                f"{stats['processed']} ok, {stats['skipped_error']} errors"
            )
    else:
        from concurrent.futures import ProcessPoolExecutor

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
                    print(
                        f"Chunk {chunk_name} FAILED: {exc}", file=sys.stderr
                    )
                    stats = {
                        "chunk": chunk_name,
                        "total": 0,
                        "processed": 0,
                        "skipped_error": 1,
                        "errors": [{"error": str(exc)}],
                    }
                global_stats["chunks"][chunk_name] = stats
                completed += 1
                print(
                    f"[{completed}/{len(work_items)}] {chunk_name}: "
                    f"{stats['processed']} ok, {stats['skipped_error']} errors"
                )

    # Summary
    summary_path = args.output_dir / "pdb_generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
