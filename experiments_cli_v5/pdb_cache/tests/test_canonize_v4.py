"""Regression tests for pdb_cache canonize (non-standard residue preservation).

Root cause (2026-08-20 audit, pt_problem.md #1-4): ``canonize_atom_array``
filtered every residue's atoms through the CCD.  When the PDB atom names do not
match the CCD entry (e.g. L2C with ASN-like atom names), all atoms were removed
and the residue disappeared from gt.pt / monomer.pt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure.io import pdbx

_GEN_PATH = Path(
    "/101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments_cli_v5/pdb_cache"
    "/generate_pdb_tensors_v4.py"
)
_CCD_PATH = Path(
    "/101045/jinyu/PhysPPI/assets/openfold3-physppi-20260724/ccd/components.cif"
)
_TRAIN = Path(
    "/101045/jinyu/PPI/Prior_out/training/cli_v4/Q-BioLiP-10k-new/chunk"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_pdb_v4_test", _GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator: {_GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GEN = _load_generator()
_CCD = pdbx.CIFFile.read(_CCD_PATH)


def _chain_from_pdb(case_id: str, chain_id: str):
    pdb = next(_TRAIN.glob(f"chunk*/gt/{case_id}.pdb"))
    raw = _GEN._parse_pdb(pdb)
    return raw[np.asarray(raw.chain_id).astype(str) == chain_id]


def test_canonize_preserves_l2c_with_mismatched_ccd():
    """The L2C residue (CCD atom names mismatch) must survive with all atoms."""
    chain = _chain_from_pdb("2hvj_1_G1", "G")
    canon = _GEN.canonize_atom_array(chain, ccd=_CCD)

    starts = struc.get_residue_starts(canon)
    resnames = [str(canon.res_name[s]) for s in starts]
    resids = [int(canon.res_id[s]) for s in starts]

    assert len(starts) == 218  # was 217 before the fix
    assert "L2C" in resnames
    l2c = canon[canon.res_name == "L2C"]
    assert len(l2c) == 8
    assert sorted(set(map(str, l2c.atom_name))) == [
        "C",
        "CA",
        "CB",
        "CG",
        "N",
        "ND2",
        "O",
        "OD1",
    ]
    assert 201 in resids


def test_canonize_keeps_fme_matching_ccd():
    """CCD-matching non-standard residues keep their filtered CCD atoms."""
    chain = _chain_from_pdb("3tgu_1_G1", "C")
    canon = _GEN.canonize_atom_array(chain, ccd=_CCD)
    starts = struc.get_residue_starts(canon)
    resnames = [str(canon.res_name[s]) for s in starts]
    assert len(starts) == 380
    assert "FME" in resnames
    fme = canon[canon.res_name == "FME"]
    ccd_atoms = set(_CCD["FME"]["chem_comp_atom"]["atom_id"].as_array().tolist())
    assert set(map(str, fme.atom_name)) <= ccd_atoms


def test_canonize_unknown_residue_keeps_all_atoms():
    """A residue missing from both reference mols and CCD keeps every atom."""
    chain = _chain_from_pdb("2hvj_1_G1", "G")
    # Replace one L2C residue name with a name absent from the CCD and rerun.
    renamed = chain.copy()
    renamed.res_name[renamed.res_name == "L2C"] = "ZZZ"
    canon = _GEN.canonize_atom_array(renamed, ccd=_CCD)
    zzz = canon[canon.res_name == "ZZZ"]
    assert len(zzz) == 8


def test_canonize_standard_residue_atom_filter_unchanged():
    """Standard residues are still filtered to reference-mol atom names."""
    chain = _chain_from_pdb("2hvj_1_G1", "G")
    canon = _GEN.canonize_atom_array(chain, ccd=_CCD)
    ala_atoms = set(map(str, canon.atom_name[canon.res_name == "ALA"]))
    assert ala_atoms <= {"N", "CA", "C", "O", "CB", "OXT"}
    assert ala_atoms  # chain G actually contains ALA


def _synthetic_atom_array():
    """ARG-like residue with 5 kept / 4 junk atoms, ALA, and ZZZ (no CCD)."""
    names = [
        "N", "CA", "C", "O", "CB", "QA", "QB", "QC", "QD",
        "N", "CA", "C", "O", "CB", "OXT",
        "N", "CA", "C", "O", "CB", "CG", "ND2", "OD1",
    ]
    resnames = ["ARG"] * 9 + ["ALA"] * 6 + ["ZZZ"] * 8
    resids = [1] * 9 + [2] * 6 + [3] * 8
    elements = [name[0] if name[0] in "NOC" else "C" for name in names]
    aa = struc.AtomArray(len(names))
    aa.chain_id = np.array(["A"] * len(names), dtype="U1")
    aa.res_id = np.array(resids, dtype=np.int32)
    aa.res_name = np.array(resnames, dtype="U3")
    aa.atom_name = np.array(names, dtype="U4")
    aa.element = np.array(elements, dtype="U1")
    aa.coord = np.zeros((len(names), 3), dtype=float)
    aa.hetero = np.zeros(len(names), dtype=bool)
    aa.occupancy = np.ones(len(names), dtype=float)
    aa.bfactor = np.zeros(len(names), dtype=float)
    aa.set_annotation("molecule_type_id", np.zeros(len(names), dtype=np.int32))
    return aa


def test_canonize_orders_kept_atoms_of_partial_residue():
    """Residues with missing atoms must not raise and keep only canonical atoms."""
    aa = _synthetic_atom_array()
    canon = _GEN.canonize_atom_array(aa, ccd=_CCD)
    starts = struc.get_residue_starts(canon)
    assert len(starts) == 3
    assert len(canon) == 19  # 5 ARG + 6 ALA + 8 ZZZ
    arg = canon[canon.res_name == "ARG"]
    assert list(map(str, arg.atom_name)) == ["N", "CA", "C", "O", "CB"]
    zzz = canon[canon.res_name == "ZZZ"]
    assert list(map(str, zzz.atom_name)) == [
        "N", "CA", "C", "O", "CB", "CG", "ND2", "OD1",
    ]


def test_canonize_keeps_unl_without_chem_comp_atom():
    """CCD entries without chem_comp_atom (e.g. UNL) fall back to keep-all."""
    aa = _synthetic_atom_array()
    aa.res_name[aa.res_name == "ZZZ"] = "UNL"
    canon = _GEN.canonize_atom_array(aa, ccd=_CCD)
    unl = canon[canon.res_name == "UNL"]
    assert len(unl) == 8


def test_canonize_round_trip_keeps_l2c_annotations():
    """Save/load round trip preserves the fixed residue set."""
    from experiments.pdb_cache.pdb_adapter import (  # noqa: PLC0415
        load_atomarray_from_pt,
        save_atomarray_as_pt,
    )

    chain = _chain_from_pdb("2hvj_1_G1", "G")
    canon = _GEN.canonize_atom_array(chain, ccd=_CCD)
    tmp = Path("/tmp") / f"_canonize_l2c_roundtrip_{id(canon)}.pt"
    try:
        save_atomarray_as_pt(canon, tmp)
        loaded = load_atomarray_from_pt(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    assert len(loaded) == len(canon)
    assert 201 in {int(r) for r in loaded.res_id[loaded.res_name == "L2C"]}
