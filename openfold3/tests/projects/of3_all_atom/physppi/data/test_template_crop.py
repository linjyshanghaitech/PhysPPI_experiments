import numpy as np
import pytest
from biotite.structure import AtomArray

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
