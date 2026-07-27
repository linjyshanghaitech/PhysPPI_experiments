import importlib
from types import SimpleNamespace

import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.resources.residues import MoleculeType


def _atom_array_for_sequences(chain_sequences):
    residues = [
        (chain_id, residue_index, residue_name)
        for chain_id, residue_names in chain_sequences.items()
        for residue_index, residue_name in enumerate(residue_names, start=1)
    ]
    atom_array = AtomArray(len(residues))
    atom_array.coord = np.zeros((len(residues), 3), dtype=float)
    atom_array.chain_id = np.asarray([chain_id for chain_id, _, _ in residues])
    atom_array.res_id = np.asarray([residue_index for _, residue_index, _ in residues])
    atom_array.res_name = np.asarray([residue_name for _, _, residue_name in residues])
    atom_array.atom_name = np.asarray(["CA"] * len(residues))
    atom_array.element = np.asarray(["C"] * len(residues))
    atom_array.set_annotation(
        "molecule_type_id",
        np.full(len(residues), int(MoleculeType.PROTEIN), dtype=int),
    )
    return atom_array


def test_adapter_labels_canonical_atom_array_entities_before_structure_processing(
    monkeypatch,
):
    from openfold3.core.data.framework.single_datasets.prior_complex import (
        PhysPPIPriorComplexDatasetAdapter,
    )

    module = importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.data.adapters._runtime"
    )
    sample = SimpleNamespace(
        sample_id="case1",
        paths=SimpleNamespace(gt_pdb="case1.pdb"),
        chain_sequences={"A": "AC", "B": "AC", "C": "AD"},
    )
    parsed = _atom_array_for_sequences(
        {"A": ["ALA", "CYS"], "B": ["ALA", "CYS"], "C": ["ALA", "ASP"]}
    )
    captured = {}

    monkeypatch.setattr(module, "_parse_pdb_atom_array", lambda path: parsed.copy())

    def fake_process_target_atom_array_of3(**kwargs):
        atom_array = kwargs["atom_array"]
        assert "entity_id" in atom_array.get_annotation_categories()
        assert "mol_entity_id" in atom_array.get_annotation_categories()
        captured["entity_id"] = atom_array.entity_id.copy()
        captured["mol_entity_id"] = atom_array.mol_entity_id.copy()
        return SimpleNamespace(
            atom_array_gt=atom_array,
            crop_strategy="none",
            n_tokens=len(atom_array),
        )

    monkeypatch.setattr(
        module,
        "process_target_atom_array_of3",
        fake_process_target_atom_array_of3,
    )
    monkeypatch.setattr(module, "get_reference_conformer_data_of3", lambda **kwargs: [])
    monkeypatch.setattr(
        module,
        "separate_cropped_and_gt",
        lambda **kwargs: (kwargs["atom_array_gt"], kwargs["atom_array_gt"]),
    )
    monkeypatch.setattr(module, "add_token_positions", lambda atom_array: None)
    monkeypatch.setattr(
        module,
        "featurize_target_gt_structure_of3",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        module,
        "featurize_reference_conformers_of3",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        PhysPPIPriorComplexDatasetAdapter,
        "_ccd_for_atom_array",
        lambda self, atom_array: None,
    )
    monkeypatch.setattr(
        PhysPPIPriorComplexDatasetAdapter,
        "_ensure_ccd_reference_molecules",
        lambda self, atom_array, ccd: None,
    )

    adapter = object.__new__(PhysPPIPriorComplexDatasetAdapter)
    adapter.crop = {"token_crop": {"enabled": False}}
    adapter.dataset_cache = SimpleNamespace(reference_molecule_data={})
    adapter.reference_molecule_directory = None
    adapter._physppi_reference_cache_directory = None

    adapter.create_canonical_structure_features(
        sample,
        preferred_chain_or_interface=None,
        return_full_atom_array=False,
        return_crop_strategy=False,
    )

    assert captured["entity_id"].tolist() == [1, 1, 1, 1, 2, 2]
    assert captured["mol_entity_id"].tolist() == [1, 1, 1, 1, 2, 2]


def test_canonical_ppi_input_excludes_explicit_hydrogens():
    module = importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.data.adapters._runtime"
    )
    atom_array = AtomArray(2)
    atom_array.coord = np.zeros((2, 3), dtype=float)
    atom_array.chain_id = np.asarray(["A", "A"])
    atom_array.res_id = np.asarray([1, 1])
    atom_array.res_name = np.asarray(["ALA", "ALA"])
    atom_array.atom_name = np.asarray(["CA", "H"])
    atom_array.element = np.asarray(["C", "H"])
    atom_array.set_annotation(
        "molecule_type_id",
        np.full(2, int(MoleculeType.PROTEIN), dtype=int),
    )

    prepared = module._canonical_ppi_heavy_atom_array(atom_array)

    assert prepared.atom_name.tolist() == ["CA"]
    assert prepared.element.tolist() == ["C"]
