import importlib

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from openfold3.core.data.resources.residues import MoleculeType


def _structure_adapter_module():
    importlib.import_module(
        "openfold3.core.data.framework.single_datasets.prior_complex"
    )
    return importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.data.adapters._runtime"
    )


def test_canonical_ppi_input_excludes_standard_protein_terminal_oxt():
    module = _structure_adapter_module()
    atom_array = AtomArray(3)
    atom_array.coord = np.zeros((3, 3), dtype=float)
    atom_array.chain_id = np.asarray(["A", "A", "A"])
    atom_array.res_id = np.asarray([1, 1, 1])
    atom_array.res_name = np.asarray(["MET", "MET", "MET"])
    atom_array.atom_name = np.asarray(["CA", "O", "OXT"])
    atom_array.element = np.asarray(["C", "O", "O"])
    atom_array.set_annotation(
        "molecule_type_id", np.full(3, MoleculeType.PROTEIN, dtype=int)
    )

    prepared = module._canonical_ppi_heavy_atom_array(atom_array)

    assert prepared.atom_name.tolist() == ["CA", "O"]


def test_canonical_ppi_input_excludes_noncanonical_standard_residue_atoms():
    module = _structure_adapter_module()
    atom_array = AtomArray(5)
    atom_array.coord = np.zeros((5, 3), dtype=float)
    atom_array.chain_id = np.asarray(["A", "A", "A", "A", "A"])
    atom_array.res_id = np.asarray([1, 1, 1, 1, 1])
    atom_array.res_name = np.asarray(["MET", "MET", "MET", "MET", "MET"])
    atom_array.atom_name = np.asarray(["N", "CA", "C", "CN", "O1"])
    atom_array.element = np.asarray(["N", "C", "C", "C", "O"])
    atom_array.set_annotation(
        "molecule_type_id", np.full(5, MoleculeType.PROTEIN, dtype=int)
    )

    prepared = module._canonical_ppi_heavy_atom_array(atom_array)

    assert prepared.atom_name.tolist() == ["N", "CA", "C"]


def test_canonical_ppi_input_converts_mse_to_met_reference_vocabulary():
    module = _structure_adapter_module()
    atom_array = AtomArray(3)
    atom_array.coord = np.zeros((3, 3), dtype=float)
    atom_array.chain_id = np.asarray(["A", "A", "A"])
    atom_array.res_id = np.asarray([1, 1, 1])
    atom_array.res_name = np.asarray(["MSE", "MSE", "MSE"])
    atom_array.atom_name = np.asarray(["CA", "CB", "SE"])
    atom_array.element = np.asarray(["C", "C", "SE"])
    atom_array.set_annotation(
        "molecule_type_id", np.full(3, MoleculeType.PROTEIN, dtype=int)
    )

    prepared = module._canonical_ppi_heavy_atom_array(atom_array)

    assert prepared.res_name.tolist() == ["MET", "MET", "MET"]
    assert prepared.atom_name.tolist() == ["CA", "CB", "SD"]
    assert prepared.element.tolist() == ["C", "C", "S"]


def test_canonical_ppi_input_excludes_nonprotein_components():
    module = _structure_adapter_module()
    atom_array = AtomArray(5)
    atom_array.coord = np.zeros((5, 3), dtype=float)
    atom_array.chain_id = np.asarray(["A", "A", "A", "L", "L"])
    atom_array.res_id = np.asarray([1, 1, 1, 2, 2])
    atom_array.res_name = np.asarray(["MET", "MET", "MET", "LIG", "LIG"])
    atom_array.atom_name = np.asarray(["N", "CA", "C", "C1", "O1"])
    atom_array.element = np.asarray(["N", "C", "C", "C", "O"])
    atom_array.set_annotation(
        "molecule_type_id",
        np.asarray(
            [
                MoleculeType.PROTEIN,
                MoleculeType.PROTEIN,
                MoleculeType.PROTEIN,
                MoleculeType.LIGAND,
                MoleculeType.LIGAND,
            ],
            dtype=int,
        ),
    )

    prepared = module._canonical_ppi_heavy_atom_array(atom_array)

    assert prepared.chain_id.tolist() == ["A", "A", "A"]
    assert prepared.res_name.tolist() == ["MET", "MET", "MET"]


def test_ccd_reference_cache_generates_annotated_fme_sdf(tmp_path, monkeypatch):
    module = _structure_adapter_module()
    molecule = Chem.MolFromSmiles("NC")
    for atom, atom_name in zip(molecule.GetAtoms(), ["N", "CA"], strict=True):
        atom.SetProp("annot_atom_name", atom_name)
        atom.SetBoolProp("annot_used_atom_mask", True)
    monkeypatch.setattr(module, "mol_from_ccd_entry", lambda *_: molecule)
    monkeypatch.setattr(
        module,
        "resolve_and_format_fallback_conformer",
        lambda value: (value, "default"),
    )

    reference_path = module._ensure_ccd_reference_molecule(
        "FME", tmp_path, ccd=object()
    )

    molecule = module.read_single_annotated_sdf(reference_path)
    assert reference_path == tmp_path / "FME.sdf"
    assert molecule.GetNumAtoms() == 2
    assert {atom.GetProp("annot_atom_name") for atom in molecule.GetAtoms()} >= {
        "N",
    }


def test_fme_formyl_oxygen_is_named_for_the_ccd():
    module = _structure_adapter_module()
    atom_array = AtomArray(5)
    atom_array.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],  # N
            [1.0, 0.0, 0.0],  # CN
            [1.1, 0.0, 0.0],  # formyl O: nearest to CN
            [4.0, 0.0, 0.0],  # backbone C
            [4.1, 0.0, 0.0],  # backbone O
        ]
    )
    atom_array.chain_id = np.asarray(["A"] * 5)
    atom_array.res_id = np.asarray([1] * 5)
    atom_array.res_name = np.asarray(["FME"] * 5)
    atom_array.atom_name = np.asarray(["N", "CN", "O", "C", "O"])
    atom_array.element = np.asarray(["N", "C", "O", "C", "O"])
    atom_array.set_annotation(
        "molecule_type_id", np.full(5, MoleculeType.PROTEIN, dtype=int)
    )

    prepared = module._normalize_fme_carbonyl_oxygen(atom_array)

    assert prepared.atom_name.tolist() == ["N", "CN", "O1", "C", "O"]
