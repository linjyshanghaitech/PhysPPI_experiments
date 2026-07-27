from pathlib import Path

import pytest

import openfold3.core.data.framework  # noqa: F401
from openfold3.core.data.io.structure.pdb import parse_pdb_af2
from openfold3.projects.of3_all_atom.physppi.data.adapters import (
    _parse_pdb_atom_array,
)


def test_physppi_adapter_recovers_bfactor_overflow_charge_column(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "overflow.pdb"
    pdb_path.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.002019.00           N  \n"
        "END\n"
    )

    with pytest.raises(ValueError, match="invalid literal for int"):
        parse_pdb_af2(pdb_path)

    atom_array = _parse_pdb_atom_array(pdb_path)

    assert atom_array.element.tolist() == ["N"]
    assert atom_array.charge.tolist() == [0]
