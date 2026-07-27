from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray

from openfold3.projects.of3_all_atom.physppi.features import (
    build_surface_features,
)


def _atom_array(rows: list[tuple[str, int, str, str, tuple[float, float, float]]]):
    atoms = AtomArray(len(rows))
    atoms.chain_id = np.asarray([row[0] for row in rows])
    atoms.res_id = np.asarray([row[1] for row in rows])
    atoms.res_name = np.asarray([row[2] for row in rows])
    atoms.atom_name = np.asarray([row[3] for row in rows])
    atoms.coord = np.asarray([row[4] for row in rows], dtype=np.float32)
    return atoms


def test_surface_crop_uses_cb_center_local_frame_and_stable_nearest_order():
    atoms = _atom_array(
        [
            ("A", 1, "ALA", "N", (0.0, 1.0, 0.0)),
            ("A", 1, "ALA", "CA", (0.0, 0.0, 0.0)),
            ("A", 1, "ALA", "C", (1.0, 0.0, 0.0)),
            ("A", 1, "ALA", "CB", (0.0, 0.0, 1.0)),
        ]
    )
    mesh = {
        "A": np.asarray(
            [
                [0.0, 0.0, 2.0, 10.0, 99.0, 20.0, 1.0, 0.0, 0.0, 30.0],
                [0.0, 1.0, 1.0, 11.0, 98.0, 21.0, 0.0, 1.0, 0.0, 31.0],
                [1.0, 0.0, 1.0, 12.0, 97.0, 22.0, 0.0, 0.0, 1.0, 32.0],
            ],
            dtype=np.float32,
        )
    }

    result = build_surface_features(
        mesh,
        {("A", 1): 0},
        {"A": (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
        2,
        atom_array=atoms,
    )

    assert result["mask"].tolist() == [[True, True]]
    assert np.allclose(result["coords"][0], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert np.allclose(
        result["features"][0],
        [[10.0, 20.0, 1.0, 0.0, 0.0, 30.0], [11.0, 21.0, 0.0, 1.0, 0.0, 31.0]],
    )


def test_surface_crop_falls_back_to_gly_ca_without_a_dmap_interface_gate():
    atoms = _atom_array(
        [
            ("A", 1, "GLY", "N", (0.0, 1.0, 0.0)),
            ("A", 1, "GLY", "CA", (0.0, 0.0, 0.0)),
            ("A", 1, "GLY", "C", (1.0, 0.0, 0.0)),
            ("A", 2, "ALA", "N", (0.0, 1.0, 0.0)),
            ("A", 2, "ALA", "CA", (0.0, 0.0, 0.0)),
            ("A", 2, "ALA", "C", (1.0, 0.0, 0.0)),
            ("A", 2, "ALA", "CB", (0.0, 0.0, 1.0)),
        ]
    )
    mesh = {
        "A": np.asarray(
            [[0.0, 0.0, 1.0, 1.0, 9.0, 2.0, 0.0, 0.0, 1.0, 3.0]],
            dtype=np.float32,
        )
    }

    result = build_surface_features(
        mesh,
        {("A", 1): 0, ("A", 2): 1},
        {"A": (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
        1,
        atom_array=atoms,
    )

    assert result["mask"].tolist() == [[True], [True]]
    assert np.allclose(result["coords"][0, 0], [0.0, 0.0, 1.0])
    assert np.allclose(result["features"][0, 0], [1.0, 2.0, 0.0, 0.0, 1.0, 3.0])
    assert np.allclose(result["coords"][1, 0], [0.0, 0.0, 0.0])
    assert np.allclose(result["features"][1, 0], [1.0, 2.0, 0.0, 0.0, 1.0, 3.0])


def test_surface_crop_rejects_invalid_backbone_frame():
    atoms = _atom_array(
        [
            ("A", 1, "ALA", "N", (1.0, 0.0, 0.0)),
            ("A", 1, "ALA", "CA", (0.0, 0.0, 0.0)),
            ("A", 1, "ALA", "C", (2.0, 0.0, 0.0)),
            ("A", 1, "ALA", "CB", (0.0, 0.0, 1.0)),
        ]
    )
    mesh = {
        "A": np.asarray(
            [[0.0, 0.0, 2.0, 1.0, 9.0, 2.0, 0.0, 0.0, 1.0, 3.0]],
            dtype=np.float32,
        )
    }

    result = build_surface_features(
        mesh,
        {("A", 1): 0},
        {"A": (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
        1,
        atom_array=atoms,
    )

    assert result["valid"] is False
    assert not result["mask"].any()
    assert np.array_equal(result["coords"], np.zeros((1, 1, 3), dtype=np.float32))
    assert np.array_equal(result["features"], np.zeros((1, 1, 6), dtype=np.float32))


def test_surface_crop_has_no_owner_or_neighbor_from_a_dropped_chain(tmp_path):
    import openfold3.core.data.framework.single_datasets.afdb10k  # noqa: F401
    from openfold3.projects.of3_all_atom.physppi.data.adapters import (
        restrict_prior_sample_to_chains,
    )
    from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
        PriorComplexSample,
        PriorSamplePaths,
    )

    sample = PriorComplexSample(
        sample_id="surface",
        dmap={"A|B": ((1, 1, 5.0),)},
        interactions=(),
        mesh={
            "A": ((0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0),),
            "B": ((99.0, 99.0, 99.0, 9.0, 0.0, 9.0, 0.0, 0.0, 1.0, 9.0),),
        },
        opm={},
        paths=PriorSamplePaths(
            sample_id="surface",
            gt_pdb=tmp_path / "gt.pdb",
            monomer_pdbs={"A": tmp_path / "A.pdb", "B": tmp_path / "B.pdb"},
            main_msa_npzs={},
            paired_msa_npzs={},
            prior_pkl=tmp_path / "prior.pkl",
        ),
        chain_sequences={"A": "A", "B": "A"},
    )
    restricted = restrict_prior_sample_to_chains(sample, ("A",))
    atoms = _atom_array(
        [
            ("A", 1, "ALA", "N", (0.0, 1.0, 0.0)),
            ("A", 1, "ALA", "CA", (0.0, 0.0, 0.0)),
            ("A", 1, "ALA", "C", (1.0, 0.0, 0.0)),
            ("A", 1, "ALA", "CB", (0.0, 0.0, 1.0)),
        ]
    )

    result = build_surface_features(
        restricted.mesh,
        {("A", 1): 0},
        {"A": (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
        1,
        atom_array=atoms,
    )

    assert set(restricted.mesh) == {"A"}
    assert result["valid"] is True
    assert np.allclose(result["coords"][0, 0], [0.0, 0.0, 1.0])
    assert np.allclose(result["features"][0, 0], [1.0, 2.0, 0.0, 0.0, 1.0, 3.0])
