import biotite.structure as struc
import numpy as np
from biotite.structure import BondList

from openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
from openfold3.core.data.resources.residues import MoleculeType


def _make_two_chain_atom_array() -> struc.AtomArray:
    atoms = struc.AtomArray(4)
    atoms.chain_id = np.asarray(["A", "A", "B", "B"])
    atoms.res_id = np.asarray([1, 2, 1, 2])
    atoms.ins_code = np.asarray(["", "", "", ""])
    atoms.res_name = np.asarray(["ALA", "GLY", "ALA", "GLY"])
    atoms.atom_name = np.asarray(["CA", "CA", "CA", "CA"])
    atoms.element = np.asarray(["C", "C", "C", "C"])
    atoms.coord = np.zeros((4, 3), dtype=float)
    atoms.occupancy = np.ones(4, dtype=float)
    atoms.set_annotation("token_id", np.arange(4))
    atoms.set_annotation(
        "molecule_type_id",
        np.full(4, MoleculeType.PROTEIN, dtype=int),
    )
    atoms.set_annotation("is_atomized", np.zeros(4, dtype=bool))
    atoms.set_annotation("entity_id", np.asarray([7, 7, 7, 7]))
    atoms.set_annotation("mol_entity_id", np.asarray([7, 7, 7, 7]))
    atoms.set_annotation("mol_sym_id", np.asarray([1, 1, 2, 2]))
    atoms.set_annotation("mol_sym_token_index", np.asarray([0, 1, 0, 1]))
    atoms.set_annotation("mol_sym_component_id", np.arange(4))
    atoms.bonds = BondList(4, np.empty((0, 3), dtype=np.uint32))
    return atoms


def test_ground_truth_structure_features_keep_chain_labels() -> None:
    features = featurize_structure_of3(
        _make_two_chain_atom_array(),
        n_tokens=4,
        is_gt=True,
    )

    assert features["asym_id"].tolist() == [1, 1, 2, 2]
    assert features["entity_id"].tolist() == [7, 7, 7, 7]
    assert features["sym_id"].tolist() == [1, 1, 2, 2]


def test_target_structure_features_mark_crop_segments_at_residue_gaps() -> None:
    atom_array = _make_two_chain_atom_array()
    atom_array.res_id = np.asarray([1, 2, 10, 11])
    atom_array.chain_id = np.asarray(["A", "A", "A", "A"])

    features = featurize_structure_of3(
        atom_array,
        n_tokens=4,
        is_gt=False,
    )

    assert features["residue_index"].tolist() == [1, 2, 10, 11]
    assert features["asym_id"].tolist() == [1, 1, 1, 1]
    assert features["crop_segment_id"].tolist() == [1, 1, 2, 2]


def test_target_structure_features_increment_crop_segments_at_chain_boundaries() -> None:
    atom_array = _make_two_chain_atom_array()
    atom_array.res_id = np.asarray([1, 2, 5, 6])
    atom_array.chain_id = np.asarray(["A", "A", "B", "B"])

    features = featurize_structure_of3(
        atom_array,
        n_tokens=4,
        is_gt=False,
    )

    assert features["asym_id"].tolist() == [1, 1, 2, 2]
    assert features["crop_segment_id"].tolist() == [1, 1, 2, 2]


def test_target_structure_features_pad_crop_segment_id() -> None:
    atom_array = _make_two_chain_atom_array()

    features = featurize_structure_of3(
        atom_array,
        n_tokens=6,
        is_gt=False,
    )

    assert features["token_mask"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert features["crop_segment_id"].tolist() == [1, 1, 2, 2, 0, 0]
