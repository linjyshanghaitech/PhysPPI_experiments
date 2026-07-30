import numpy as np
import pytest
from biotite.structure import AtomArray

from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.physppi.features import (
    PhysPPIDataError,
)


def _atoms(
    rows: list[tuple[str, int, str, tuple[float, float, float]]],
) -> AtomArray:
    atoms = AtomArray(len(rows))
    atoms.chain_id = np.asarray([row[0] for row in rows])
    atoms.res_id = np.asarray([row[1] for row in rows])
    atoms.res_name = np.asarray(["ALA"] * len(rows))
    atoms.atom_name = np.asarray([row[2] for row in rows])
    atoms.coord = np.asarray([row[3] for row in rows], dtype=np.float32)
    return atoms


def _structure_adapter_module():
    import openfold3.core.data.framework.single_datasets.afdb10k  # noqa: F401
    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module

    return module


def _token_atoms(
    chain_lengths: dict[str, int],
    *,
    chain_spacing: float = 20.0,
) -> AtomArray:
    rows = []
    token_id = 0
    for chain_index, (chain_id, length) in enumerate(chain_lengths.items()):
        for residue_id in range(1, length + 1):
            rows.append(
                (
                    chain_id,
                    residue_id,
                    token_id,
                    (
                        float(residue_id),
                        float(chain_index) * chain_spacing,
                        0.0,
                    ),
                )
            )
            token_id += 1
    atoms = AtomArray(len(rows))
    atoms.chain_id = np.asarray([row[0] for row in rows])
    atoms.res_id = np.asarray([row[1] for row in rows])
    atoms.res_name = np.asarray(["ALA"] * len(rows))
    atoms.atom_name = np.asarray(["CA"] * len(rows))
    atoms.coord = np.asarray([row[3] for row in rows], dtype=np.float32)
    atoms.token_id = np.asarray([row[2] for row in rows], dtype=np.int64)
    atoms.token_center_atom = np.ones(len(rows), dtype=bool)
    atoms.molecule_type_id = np.full(
        len(rows),
        int(MoleculeType.PROTEIN),
        dtype=np.int64,
    )
    return atoms


def _kept_residues(atom_array: AtomArray, mask: np.ndarray, chain_id: str) -> set[int]:
    kept = atom_array[mask]
    chain = kept[np.asarray(kept.chain_id).astype(str) == chain_id]
    return set(int(res_id) for res_id in np.asarray(chain.res_id))


def test_interface_preserving_crop_keeps_dmap_and_minimum_chain_residues():
    module = _structure_adapter_module()
    atom_array = _token_atoms({"A": 30, "B": 30})
    dmap = {
        "A|B": tuple((residue, residue, 2.0) for residue in range(10, 13))
    }

    mask = module._physppi_interface_preserving_crop_mask(
        atom_array,
        dmap,
        token_budget=40,
        min_residues_per_chain=15,
        rng=np.random.default_rng(1),
    )

    assert int(mask.sum()) <= 40
    kept_a = _kept_residues(atom_array, mask, "A")
    kept_b = _kept_residues(atom_array, mask, "B")
    assert len(kept_a) >= 15
    assert len(kept_b) >= 15
    assert {10, 11, 12} <= kept_a
    assert {10, 11, 12} <= kept_b


def test_interface_preserving_crop_preserves_minimum_before_trimming_dmap():
    module = _structure_adapter_module()
    atom_array = _token_atoms({"A": 60, "B": 60})
    dmap = {"A|B": tuple((residue, residue, float(residue)) for residue in range(1, 31))}

    mask = module._physppi_interface_preserving_crop_mask(
        atom_array,
        dmap,
        token_budget=40,
        min_residues_per_chain=15,
        rng=np.random.default_rng(2),
    )

    kept_a = _kept_residues(atom_array, mask, "A")
    kept_b = _kept_residues(atom_array, mask, "B")
    assert int(mask.sum()) == 40
    assert set(range(1, 16)) <= kept_a
    assert set(range(1, 16)) <= kept_b
    assert 30 not in kept_a
    assert 30 not in kept_b


def test_interface_preserving_crop_adds_nearest_remaining_context():
    module = _structure_adapter_module()
    atom_array = _token_atoms({"A": 30, "B": 30})
    dmap = {"A|B": ((15, 15, 2.0),)}

    mask = module._physppi_interface_preserving_crop_mask(
        atom_array,
        dmap,
        token_budget=14,
        min_residues_per_chain=5,
        rng=np.random.default_rng(3),
    )

    kept_a = _kept_residues(atom_array, mask, "A")
    kept_b = _kept_residues(atom_array, mask, "B")
    assert len(kept_a) + len(kept_b) == 14
    assert kept_a <= set(range(10, 21))
    assert kept_b <= set(range(10, 21))
    assert {1, 30}.isdisjoint(kept_a)
    assert {1, 30}.isdisjoint(kept_b)


def test_interface_preserving_crop_keeps_all_tokens_for_mapped_residue():
    module = _structure_adapter_module()
    atom_array = AtomArray(4)
    atom_array.chain_id = np.asarray(["A", "A", "B", "B"])
    atom_array.res_id = np.asarray([1, 1, 1, 2])
    atom_array.res_name = np.asarray(["ALA"] * 4)
    atom_array.atom_name = np.asarray(["CA", "CB", "CA", "CA"])
    atom_array.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    atom_array.token_id = np.asarray([0, 1, 2, 3], dtype=np.int64)
    atom_array.token_center_atom = np.ones(4, dtype=bool)
    atom_array.molecule_type_id = np.full(
        4,
        int(MoleculeType.PROTEIN),
        dtype=np.int64,
    )

    mask = module._physppi_interface_preserving_crop_mask(
        atom_array,
        {"A|B": ((1, 1, 2.0),)},
        token_budget=4,
        min_residues_per_chain=1,
        rng=np.random.default_rng(4),
    )

    assert mask.tolist()[:3] == [True, True, True]


def test_interface_preserving_crop_is_deterministic_for_same_seed():
    module = _structure_adapter_module()
    atom_array = _token_atoms({"A": 40, "B": 40})
    dmap = {
        "A|B": tuple((residue, residue, 3.0) for residue in range(10, 20))
    }

    first = module._physppi_interface_preserving_crop_mask(
        atom_array,
        dmap,
        token_budget=45,
        min_residues_per_chain=15,
        rng=np.random.default_rng(5),
    )
    replay = module._physppi_interface_preserving_crop_mask(
        atom_array,
        dmap,
        token_budget=45,
        min_residues_per_chain=15,
        rng=np.random.default_rng(5),
    )

    assert np.array_equal(first, replay)


def test_monomer_geometry_matches_cropped_atom_identity_order_without_gt_coordinates():
    module = _structure_adapter_module()
    assert hasattr(module, "build_monomer_input_geometry")
    cropped = _atoms(
        [
            ("A", 2, "CA", (102.0, 102.0, 102.0)),
            ("A", 1, "N", (101.0, 101.0, 101.0)),
            ("A", 4, "CA", (104.0, 104.0, 104.0)),
            ("A", 1, "CA", (101.5, 101.5, 101.5)),
            ("A", 3, "CA", (103.0, 103.0, 103.0)),
        ]
    )
    monomer = _atoms(
        [
            ("A", 1, "CA", (0.0, 0.0, 0.0)),
            ("A", 1, "N", (-1.0, 0.0, 0.0)),
            ("A", 2, "CA", (1.0, 0.0, 0.0)),
            ("A", 3, "CA", (0.0, 1.0, 0.0)),
            ("A", 4, "CA", (0.0, 0.0, 1.0)),
        ]
    )

    result = module.build_monomer_input_geometry(cropped, {"A": monomer})

    expected = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(result["physppi_monomer_atom_positions"], expected)
    assert result["physppi_monomer_atom_mask"].dtype == np.bool_
    assert result["physppi_monomer_atom_mask"].tolist() == [True] * len(cropped)
    assert not np.array_equal(result["physppi_monomer_atom_positions"], cropped.coord)


def test_monomer_geometry_rejects_chain_without_four_non_collinear_representatives():
    module = _structure_adapter_module()
    assert hasattr(module, "build_monomer_input_geometry")
    cropped = _atoms(
        [
            ("A", 1, "CA", (101.0, 0.0, 0.0)),
            ("A", 2, "CA", (102.0, 0.0, 0.0)),
            ("A", 3, "CA", (103.0, 0.0, 0.0)),
            ("A", 4, "CA", (104.0, 0.0, 0.0)),
        ]
    )
    collinear_monomer = _atoms(
        [
            ("A", 1, "CA", (0.0, 0.0, 0.0)),
            ("A", 2, "CA", (1.0, 0.0, 0.0)),
            ("A", 3, "CA", (2.0, 0.0, 0.0)),
            ("A", 4, "CA", (3.0, 0.0, 0.0)),
        ]
    )

    with pytest.raises(PhysPPIDataError, match="missing_mandatory_monomer_geometry"):
        module.build_monomer_input_geometry(cropped, {"A": collinear_monomer})
