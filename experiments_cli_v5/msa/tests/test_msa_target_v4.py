"""Regression tests for the v4 MSA generator target sequence.

Root cause (2026-08-20 audit, pt_problem.md #5): ``_parse_gt_chain_data``
dropped non-UNK non-standard residues (FME/L2C/...) and collapsed residues by
resseq only, so the projection target had fewer columns than gt.pt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from openfold3.core.data.primitives.sequence.msa import MsaArray

_GEN_PATH = Path(
    "/101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments_cli_v5/msa"
    "/generate_msa_tensors_v4.py"
)
_TRAIN = Path(
    "/101045/jinyu/PPI/Prior_out/training/cli_v4/Q-BioLiP-10k-new/chunk"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_msa_v4_test", _GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator: {_GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GEN = _load_generator()


def _write_pdb(tmp_path: Path, lines: list[str]) -> Path:
    pdb = tmp_path / "test.pdb"
    pdb.write_text("".join(lines))
    return pdb


def _ca_line(serial: int, name: str, chain: str, resseq: int, ins: str = "") -> str:
    resname = f"{name:>3}"
    return (
        f"ATOM  {serial:5d}  CA  {resname} {chain}{resseq:4d}{ins}   "
        f"   1.000  50.00           C  \n"
    )


def test_parse_keeps_nonstandard_residues_as_x(tmp_path):
    """FME/L2C/UNK all become X positions in the target sequence."""
    pdb = _write_pdb(
        tmp_path=tmp_path,
        lines=[
            _ca_line(1, "ALA", "A", 1),
            _ca_line(2, "UNK", "A", 2),
            _ca_line(3, "FME", "A", 3),
            _ca_line(4, "L2C", "A", 4),
        ],
    )
    data = _GEN._parse_gt_chain_data(pdb)
    assert data["A"]["sequence"] == "AXXX"
    assert data["A"]["resseq"] == [1, 2, 3, 4]


def test_parse_real_fme_chain_matches_gt_residue_count():
    """3tgu chain C keeps its 380 residues (FME included as X)."""
    pdb = next(_TRAIN.glob("chunk*/gt/3tgu_1_G1.pdb"))
    data = _GEN._parse_gt_chain_data(pdb)
    seq = data["C"]["sequence"]
    assert len(seq) == 380  # was 379 before the fix
    assert seq[0] == "X"
    assert all(c in _GEN._STANDARD_1 for c in seq)


def test_parse_counts_insertion_code_residues_separately(tmp_path):
    """Two residues sharing a resseq with different insertion codes count twice."""
    pdb = _write_pdb(
        tmp_path=tmp_path,
        lines=[
            _ca_line(1, "ALA", "A", 1),
            _ca_line(2, "GLY", "A", 2, ins="A"),
            _ca_line(3, "GLY", "A", 2, ins="B"),
            _ca_line(4, "VAL", "A", 3),
        ],
    )
    data = _GEN._parse_gt_chain_data(pdb)
    assert data["A"]["sequence"] == "AGGV"
    assert len(data["A"]["sequence"]) == 4
    assert len(data["A"]["resseq"]) == 4


def test_parse_counts_residue_without_ca(tmp_path):
    """A residue without CA (side-chain only) still occupies a target column."""
    pdb = _write_pdb(
        tmp_path=tmp_path,
        lines=[
            _ca_line(1, "ALA", "A", 1),
            (
                "ATOM      2  CB  ZZZ A   2      1.000  50.00           C  \n"
            ),
            _ca_line(3, "VAL", "A", 3),
        ],
    )
    data = _GEN._parse_gt_chain_data(pdb)
    assert data["A"]["sequence"] == "AXV"
    assert data["A"]["resseq"] == [1, 2, 3]


def test_projection_keeps_x_matching_semantics():
    """Target X matches an arbitrary source column (allow_unknown fallback)."""
    msa = MsaArray(
        msa=_GEN._msa_to_chars(np.array([["A", "M", "V"]], dtype="<U1")),
        deletion_matrix=np.zeros((1, 3), dtype=np.int32),
        metadata=[],
    )
    projected = _GEN._project_msa_to_gt_residues(msa, "AXV")
    assert projected is not None
    assert projected.msa.shape == (1, 3)
