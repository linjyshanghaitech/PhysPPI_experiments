from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray

from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    InteractionRecord,
)
from openfold3.projects.of3_all_atom.physppi.features import (
    build_interaction_features,
    build_interface_residue_features,
    build_membrane_target,
    build_surface_features,
    build_tm_features,
)


def _atom_array(rows: list[tuple[str, int, str, str, tuple[float, float, float]]]):
    atoms = AtomArray(len(rows))
    atoms.chain_id = np.asarray([row[0] for row in rows])
    atoms.res_id = np.asarray([row[1] for row in rows])
    atoms.res_name = np.asarray([row[2] for row in rows])
    atoms.atom_name = np.asarray([row[3] for row in rows])
    atoms.coord = np.asarray([row[4] for row in rows], dtype=np.float32)
    return atoms


def test_interface_residue_reduces_cross_chain_dmap_to_per_token_mask():
    token_index = {("A", 1): 0, ("B", 2): 1}
    interface = build_interface_residue_features(
        {"A|B": ((1, 2, 9.0), (1, 2, 17.0))}, token_index, 10.0
    )
    interaction = build_interaction_features((), token_index)

    assert interface["available"] is True
    assert interface["valid"] is True
    assert interface["interface_mask"].dtype == np.bool_
    assert interface["interface_mask"].shape == (2,)
    assert interface["interface_mask"].tolist() == [True, True]
    assert interface["cutoff"] == 10.0
    assert "pair_mask" not in interface
    assert "surface_interface_mask" not in interface
    narrow = build_interface_residue_features(
        {"A|B": ((1, 2, 9.0),)}, token_index, 8.0
    )
    assert not narrow["interface_mask"].any()
    assert narrow["valid"] is False
    assert narrow["cutoff"] == 8.0
    assert interaction["available"] is False
    assert interaction["valid"] is False
    assert not interaction["type_mask"].any()


def test_empty_interactions_are_unavailable_not_negative():
    result = build_interaction_features((), {})

    assert result["available"] is False
    assert result["valid"] is False
    assert result["type_mask"].shape == (0, 0, 5)
    assert result["aux"] == ()


def test_interaction_has_five_symmetric_channels_and_drops_invalid_endpoints():
    token_index = {("A", 1): 0, ("B", 2): 1}
    records = (
        InteractionRecord("A|B", 1, 2, "hbond", 2.5, "NZ", "O"),
        InteractionRecord("A|B", 99, 2, "saltbridge", 3.0, "N", "O"),
    )
    result = build_interaction_features(records, token_index)

    assert result["available"] is True
    assert result["valid"] is True
    assert result["type_mask"].shape == (2, 2, 5)
    assert result["type_mask"][0, 1, 0]
    assert result["type_mask"][1, 0, 0]
    assert not result["type_mask"][..., 1].any()
    assert result["aux"] == ((0, 1, 0, 2.5, "NZ", "O"),)


def test_interaction_scalar_atom_endpoints_use_structure_distance():
    atoms = _atom_array(
        [
            ("A", 1, "ALA", "NZ", (0.0, 0.0, 0.0)),
            ("B", 2, "ALA", "O", (3.0, 4.0, 0.0)),
        ]
    )
    result = build_interaction_features(
        (InteractionRecord("A|B", 1, 2, "hbond", 99.0, "NZ", "O"),),
        {("A", 1): 0, ("B", 2): 1},
        atom_array=atoms,
    )

    assert result["valid"] is True
    assert result["aux"][0][3] == 5.0


def test_interaction_group_endpoints_use_centroid_distance():
    atoms = _atom_array(
        [
            ("A", 1, "ALA", "N", (0.0, 0.0, 0.0)),
            ("A", 1, "ALA", "CA", (2.0, 0.0, 0.0)),
            ("B", 2, "ALA", "O", (1.0, 4.0, 0.0)),
            ("B", 2, "ALA", "OXT", (1.0, 6.0, 0.0)),
        ]
    )
    result = build_interaction_features(
        (
            InteractionRecord(
                "A|B", 1, 2, "saltbridge", 99.0, ("N", "CA"), ["O", "OXT"]
            ),
        ),
        {("A", 1): 0, ("B", 2): 1},
        atom_array=atoms,
    )

    assert result["valid"] is True
    assert result["aux"][0][3] == 5.0


def test_interaction_missing_group_member_keeps_type_label_but_no_auxiliary():
    atoms = _atom_array(
        [
            ("A", 1, "ALA", "N", (0.0, 0.0, 0.0)),
            ("B", 2, "ALA", "O", (3.0, 0.0, 0.0)),
        ]
    )
    result = build_interaction_features(
        (InteractionRecord("A|B", 1, 2, "saltbridge", 3.0, ("N", "CA"), "O"),),
        {("A", 1): 0, ("B", 2): 1},
        atom_array=atoms,
    )

    assert result["type_mask"][0, 1, 1]
    assert result["type_mask"][1, 0, 1]
    assert result["valid"] is True
    assert result["aux"] == ()


def test_interaction_nonfinite_group_member_keeps_type_label_but_no_auxiliary():
    atoms = _atom_array(
        [
            ("A", 1, "ALA", "N", (np.nan, 0.0, 0.0)),
            ("A", 1, "ALA", "CA", (2.0, 0.0, 0.0)),
            ("B", 2, "ALA", "O", (3.0, 0.0, 0.0)),
        ]
    )
    result = build_interaction_features(
        (InteractionRecord("A|B", 1, 2, "saltbridge", 3.0, ("N", "CA"), "O"),),
        {("A", 1): 0, ("B", 2): 1},
        atom_array=atoms,
    )

    assert result["type_mask"][0, 1, 1]
    assert result["type_mask"][1, 0, 1]
    assert result["valid"] is True
    assert result["aux"] == ()


def test_all_invalid_interactions_remain_available_but_not_valid():
    records = (InteractionRecord("A|B", 99, 100, "hbond", None, None, None),)
    result = build_interaction_features(records, {("A", 1): 0, ("B", 2): 1})

    assert result["available"] is True
    assert result["valid"] is False
    assert not result["type_mask"].any()
    assert result["aux"] == ()


def test_surface_uses_six_network_attributes_and_shared_transform():
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
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
            dtype=np.float32,
        )
    }
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    translation = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    result = build_surface_features(
        mesh,
        {("A", 1): 0},
        {"A": (rotation, translation)},
        1,
        atom_array=atoms,
    )

    assert result["coords"].shape == (1, 1, 3)
    assert result["features"].shape == (1, 1, 6)
    assert np.allclose(result["coords"][0, 0], np.asarray([1.0, 2.0, 2.0]))
    assert np.allclose(result["features"][0, 0], [4.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert result["mask"].tolist() == [[True]]
    assert result["available"] is True
    assert result["valid"] is True


def test_surface_keeps_nan_charge_neighbor_with_charge_mask_and_zero_channel():
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
            [[0.0, 0.0, 2.0, np.nan, 99.0, 2.0, 0.0, 0.0, 1.0, 3.0]],
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

    assert result["mask"].tolist() == [[True]]
    assert result["charge_mask"].tolist() == [[False]]
    assert result["features"].shape == (1, 1, 6)
    assert np.isfinite(result["features"]).all()
    assert np.allclose(result["features"][0, 0], [0.0, 2.0, 0.0, 0.0, 1.0, 3.0])
    assert result["valid"] is True


def test_surface_builder_has_no_dmap_derived_interface_gate():
    assert "surface_interface_mask" not in build_surface_features.__annotations__
    assert "surface_interface_mask" not in build_surface_features.__code__.co_varnames


def test_surface_skips_nonfinite_transforms():
    mesh = {"A": np.asarray([[1.0, 2.0, 3.0] + list(range(4, 11))])}
    result = build_surface_features(
        mesh,
        {("A", 1): 0},
        {"A": (np.full((3, 3), np.nan), np.zeros(3))},
        1,
    )

    assert result["valid"] is False
    assert not result["mask"].any()
    assert np.isfinite(result["coords"]).all()


def test_tm_uses_only_side_zero_and_tag_zero():
    tm = build_tm_features(
        {"A": {1: 0, 2: 1, 3: "0", 4: "1"}},
        {("A", 1): 0, ("A", 2): 1, ("A", 3): 2, ("A", 4): 3},
    )

    assert tm.dtype == np.bool_
    assert tm.tolist() == [True, False, True, False]


def test_tm_accepts_canonical_opm_interval_rows_from_training_pickles():
    tm = build_tm_features(
        [
            ["A", 27, 29, 0],
            ["A", 30, 31, 1],
            ["B", 10, 12, -1],
            ["C", 5, 5, "0"],
        ],
        {
            ("A", 27): 0,
            ("A", 28): 1,
            ("A", 29): 2,
            ("A", 30): 3,
            ("B", 11): 4,
            ("C", 5): 5,
        },
    )

    assert tm.dtype == np.bool_
    assert tm.tolist() == [True, True, True, False, False, True]


def test_membrane_plane_is_normalized_directed_label_only():
    target = build_membrane_target(
        {"outer_normal": [0.0, 0.0, 2.0], "outer_centroid": [0.0, 0.0, 4.0]},
        True,
    )

    assert target["valid"] is True
    assert np.allclose(target["normal"], [0.0, 0.0, 1.0])
    assert np.allclose(target["point"], [0.0, 0.0, 4.0])
    assert "input" not in target


def test_membrane_plane_is_zero_and_invalid_for_non_tm_or_bad_geometry():
    for opm, is_membrane in (
        ({"outer_normal": [0.0, 0.0, 1.0], "outer_centroid": [0.0, 0.0, 4.0]}, False),
        ({"outer_normal": [0.0, 0.0, 0.0], "outer_centroid": [0.0, 0.0, 4.0]}, True),
        ({"outer_normal": [0.0, 0.0, 1.0]}, True),
    ):
        target = build_membrane_target(opm, is_membrane)
        assert target["valid"] is False
        assert np.array_equal(target["normal"], np.zeros(3, dtype=np.float32))
        assert np.array_equal(target["point"], np.zeros(3, dtype=np.float32))
