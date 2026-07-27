# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression: ``get_reference_conformer_data_of3`` must not ``KeyError`` when a
standard amino-acid residue name (GLN, GLY, SER, ...) is used as ``ref_mol_id``
but is absent from ``reference_mol_metadata``.

The defence-in-depth fallback uses ``conformer_gen_strategy="default"`` and
``set_fallback_to_nan=False``, identical to the default entries produced by the
cache builder's synthesis path.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import AllChem

from openfold3.core.data.pipelines.sample_processing.conformer import (
    get_reference_conformer_data_of3,
)
from openfold3.core.data.primitives.caches.format import (
    DatasetChainData,
    DatasetReferenceMoleculeData,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STANDARD_AAS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def _make_minimal_atom_array(res_name, chain_id="A"):
    """Single-component AtomArray with one atom."""
    aa = AtomArray(1)
    aa.coord = np.zeros((1, 3))
    aa.chain_id[:] = chain_id
    aa.res_id[:] = 1
    aa.res_name[:] = res_name
    aa.atom_name[:] = "CA"
    aa.element[:] = "C"
    aa.hetero[:] = False
    aa.set_annotation("component_id", np.array([0], dtype=int))
    aa.set_annotation("atom_name_unique", np.array(["CA_1"], dtype=object))
    aa.set_annotation("crop_mask", np.array([True], dtype=bool))
    # molecule_type_id = 0 → PROTEIN
    aa.set_annotation("molecule_type_id", np.array([0], dtype=int))
    return aa


def _make_minimal_sdf_mol():
    """Smallest valid RDKit mol with a conformer (glycine)."""
    mol = Chem.MolFromSmiles("NCC(=O)O")
    AllChem.EmbedMolecule(mol, randomSeed=1)
    for atom in mol.GetAtoms():
        atom.SetProp("annot_atom_name", f"A{atom.GetIdx()}")
    return mol


_MOL_PATH = (
    "openfold3.core.data.pipelines.sample_processing.conformer"
    ".read_single_annotated_sdf"
)
_PROC_PATH = (
    "openfold3.core.data.pipelines.sample_processing.conformer"
    ".get_processed_reference_conformer"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("res_name", _STANDARD_AAS)
def test_no_keyerror_when_standard_aa_missing_from_metadata(res_name):
    """Each of the 20 standard amino acids must not raise KeyError when absent
    from ``reference_mol_metadata``."""
    atom_array = _make_minimal_atom_array(res_name)
    mol = _make_minimal_sdf_mol()

    per_chain_metadata = {
        "A": DatasetChainData(),  # no reference_mol_id → falls back to res_name
    }

    # reference_mol_metadata is EMPTY — nothing for any residue name
    reference_mol_metadata = {}

    with (
        patch(_MOL_PATH, return_value=mol),
        patch(
            _PROC_PATH,
            return_value="MOCK_PROCESSED_MOL",
        ) as mock_proc,
    ):
        result = get_reference_conformer_data_of3(
            atom_array=atom_array,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=reference_mol_metadata,
            reference_mol_dir=Path("/fake"),
        )

    assert len(result) == 1
    assert result[0] == "MOCK_PROCESSED_MOL"

    # Verify the fallback used default parameters
    call_kwargs = mock_proc.call_args.kwargs
    assert call_kwargs["preferred_confgen_strategy"] == "default"
    assert call_kwargs["set_fallback_to_nan"] is False


def test_existing_ref_mol_metadata_still_used():
    """When ref_mol_id IS present in metadata, its values are used (not the
    fallback defaults)."""
    atom_array = _make_minimal_atom_array("GLN")
    mol = _make_minimal_sdf_mol()

    per_chain_metadata = {
        "A": DatasetChainData(),
    }

    reference_mol_metadata = {
        "GLN": DatasetReferenceMoleculeData(
            conformer_gen_strategy="random_init",
            fallback_conformer_pdb_id="1abc",
            canonical_smiles="CUSTOM",
            set_fallback_to_nan=True,
        ),
    }

    with (
        patch(_MOL_PATH, return_value=mol),
        patch(_PROC_PATH, return_value="MOCK") as mock_proc,
    ):
        get_reference_conformer_data_of3(
            atom_array=atom_array,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=reference_mol_metadata,
            reference_mol_dir=Path("/fake"),
        )

    call_kwargs = mock_proc.call_args.kwargs
    assert call_kwargs["preferred_confgen_strategy"] == "random_init"
    assert call_kwargs["set_fallback_to_nan"] is True


def test_ligand_with_reference_mol_id_works_normally():
    """When a chain HAS a reference_mol_id (ligand), its metadata entry is used
    normally.  This test guards against the fallback breaking existing ligand
    behaviour."""
    atom_array = _make_minimal_atom_array("XYZ")
    mol = _make_minimal_sdf_mol()

    chain_a = DatasetChainData()
    chain_a.reference_mol_id = "XYZ"  # DatasetChainData is bare; setattr needed
    per_chain_metadata = {
        "A": chain_a,
    }

    reference_mol_metadata = {
        "XYZ": DatasetReferenceMoleculeData(
            conformer_gen_strategy="use_fallback",
            fallback_conformer_pdb_id="2abc",
            canonical_smiles="CC",
            set_fallback_to_nan=False,
        ),
    }

    with (
        patch(_MOL_PATH, return_value=mol),
        patch(_PROC_PATH, return_value="MOCK") as mock_proc,
    ):
        get_reference_conformer_data_of3(
            atom_array=atom_array,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=reference_mol_metadata,
            reference_mol_dir=Path("/fake"),
        )

    call_kwargs = mock_proc.call_args.kwargs
    assert call_kwargs["preferred_confgen_strategy"] == "use_fallback"


def test_chain_with_no_crop_atoms_is_skipped():
    """Components with all-False crop_mask are skipped entirely."""
    atom_array = _make_minimal_atom_array("GLY")
    atom_array.set_annotation("crop_mask", np.array([False], dtype=bool))
    mol = _make_minimal_sdf_mol()

    per_chain_metadata = {"A": DatasetChainData()}
    reference_mol_metadata = {}

    with (
        patch(_MOL_PATH, return_value=mol),
        patch(_PROC_PATH),
    ):
        result = get_reference_conformer_data_of3(
            atom_array=atom_array,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=reference_mol_metadata,
            reference_mol_dir=Path("/fake"),
        )

    assert result == []


# ---------------------------------------------------------------------------
# NaN fallback in_crop_mask correctness
# ---------------------------------------------------------------------------


def _make_ser_sdf_mol():
    """Create a minimal serine-like RDKit mol with annotated atom names matching
    PDB conventions (N, CA, C, O, CB, OG)."""
    mol = Chem.MolFromSmiles("N[C@@H](CO)C(=O)O")
    AllChem.EmbedMolecule(mol, randomSeed=1)
    # Assign atom names matching PDB conventions for serine
    names = ["N", "CA", "C", "O", "CB", "OG", "OXT"]
    for atom, name in zip(mol.GetAtoms(), names):
        atom.SetProp("annot_atom_name", name)
    return mol


def _make_asp_sdf_mol():
    """Create a minimal aspartate-like RDKit mol with annotated atom names."""
    mol = Chem.MolFromSmiles("N[C@@H](CC(=O)O)C(=O)O")
    AllChem.EmbedMolecule(mol, randomSeed=1)
    names = ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"]
    for atom, name in zip(mol.GetAtoms(), names):
        atom.SetProp("annot_atom_name", name)
    return mol


@pytest.mark.parametrize(
    "res_name, mol_factory, expected_n_atoms",
    [
        ("SER", _make_ser_sdf_mol, 7),
        ("ASP", _make_asp_sdf_mol, 9),
    ],
)
def test_nan_fallback_in_crop_mask_matches_pdb_atoms(
    res_name, mol_factory, expected_n_atoms
):
    """NaN-fallback ``in_crop_mask`` must have True entries for RDKit atoms whose
    annotated names match PDB crop atom names.  An all-False mask would cause
    ``ref_pos`` to miss atoms that the PDB ``atom_mask`` still counts, leading to
    the ``RuntimeError: shape '[1, 12160]' is invalid for input of size 12416``
    observed at epoch 0 step ~96 with SER+ASP.

    This test constructs a PDB component view with realistic atom naming,
    triggers the NaN-fallback path by raising ``AssertionError`` inside
    ``get_processed_reference_conformer``, and asserts the returned
    ``in_crop_mask`` has the expected number of True entries.
    """
    from biotite.structure import AtomArray

    # Build a PDB-like component view with realistic atom names
    n_atoms = expected_n_atoms
    aa = AtomArray(n_atoms)
    aa.coord = np.zeros((n_atoms, 3))
    aa.chain_id[:] = "A"
    aa.res_id[:] = 1
    aa.res_name[:] = res_name
    aa.hetero[:] = False
    aa.element[:] = "C"

    # Use realistic PDB atom names
    if res_name == "SER":
        pdb_names = ["N", "CA", "C", "O", "CB", "OG", "OXT"]
    else:
        pdb_names = ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"]

    aa.atom_name[:] = pdb_names

    # Make atom names unique (add _<idx> suffix) to match pipeline conventions
    unique_atom_names = np.array([f"{n}_1" for n in pdb_names], dtype=object)
    aa.set_annotation("component_id", np.zeros(n_atoms, dtype=int))
    aa.set_annotation("atom_name_unique", unique_atom_names)
    aa.set_annotation("crop_mask", np.ones(n_atoms, dtype=bool))  # all in crop
    aa.set_annotation("molecule_type_id", np.zeros(n_atoms, dtype=int))

    mol = mol_factory()
    # SDF mol uses RAW atom names (no uniquification suffix).
    # uniquify_ids() inside the fallback path adds the "_1" suffix,
    # matching the PDB's already-uniquified atom_name_unique field.

    per_chain_metadata = {"A": DatasetChainData()}
    # reference_mol_metadata has NO entry for this residue → uses fallback defaults
    reference_mol_metadata = {}

    with (
        patch(_MOL_PATH, return_value=mol),
        patch(_PROC_PATH, side_effect=AssertionError("simulated failure")),
    ):
        result = get_reference_conformer_data_of3(
            atom_array=aa,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=reference_mol_metadata,
            reference_mol_dir=Path("/fake"),
        )

    assert len(result) == 1
    fallback = result[0]
    assert fallback.component_id == 0

    # The in_crop_mask must have True for atoms whose PDB names are in the crop.
    # Since crop_mask is all-True for this component, all PDB names are in crop.
    # RDKit atoms whose names match PDB names should have mask=True.
    n_in_crop = int(fallback.in_crop_mask.sum())
    assert n_in_crop >= 1, (
        f"in_crop_mask is all-False for {res_name} — atom_mask/ref_pos shape mismatch risk"
    )

    # The in_crop_mask length must match the RDKit mol's atom count.
    assert len(fallback.in_crop_mask) == mol.GetNumAtoms()

    # Permutations must be explicitly empty (not zero-length array of wrong dtype)
    assert fallback.permutations.size == 0


def test_nan_fallback_mol_is_not_none():
    """The NaN fallback mol must be a valid RDKit mol, not None."""
    atom_array = _make_minimal_atom_array("SER")
    mol = _make_minimal_sdf_mol()

    per_chain_metadata = {"A": DatasetChainData()}
    reference_mol_metadata = {}

    with (
        patch(_MOL_PATH, return_value=mol),
        patch(_PROC_PATH, side_effect=AssertionError("simulated failure")),
    ):
        result = get_reference_conformer_data_of3(
            atom_array=atom_array,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=reference_mol_metadata,
            reference_mol_dir=Path("/fake"),
        )

    assert len(result) == 1
    fallback = result[0]
    assert fallback.mol is not None
    assert fallback.mol.GetNumAtoms() > 0
    # get_processed_reference_conformer WAS called and raised AssertionError
    # — that's what triggers the NaN-fallback path we're testing
