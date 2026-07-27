from pathlib import Path

from experiments.permutation_alignment.audit_v1_v32_data_errors import (
    category_payload,
    scan_pdb,
)


def _atom_line(
    *,
    serial: int,
    residue_name: str = "ALA",
    charge: str = "  ",
) -> str:
    columns = [" "] * 80
    columns[0:6] = "ATOM  "
    columns[6:11] = f"{serial:>5}"
    columns[12:16] = " CA "
    columns[17:20] = f"{residue_name:>3}"
    columns[21:22] = "A"
    columns[22:26] = f"{1:>4}"
    columns[78:80] = charge
    line = "".join(columns)
    assert len(line) == 80
    return line


def test_scan_pdb_reports_fixed_width_data_errors(tmp_path: Path) -> None:
    path = tmp_path / "anomalous.pdb"
    path.write_text(
        "\n".join(
            [
                _atom_line(serial=2, residue_name="ALA", charge="2+"),
                _atom_line(serial=1, residue_name="GLY", charge="XX"),
            ]
        )
        + "\n"
    )

    stats = scan_pdb(path, surface_neighbors=1)

    assert stats == {
        "atom_record_count": 2,
        "token_count": 1,
        "invalid_charge_field_count": 1,
        "nonincreasing_atom_id_count": 1,
        "microheterogeneous_residue_count": 1,
        "surface_neighbor_axis_collision": True,
    }


def test_scan_pdb_ignores_non_atoms_and_accepts_charge_directions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.pdb"
    path.write_text(
        "\n"
        "REMARK    charge XX is not an atom record\n"
        f"{_atom_line(serial=1)}\n"
        f"{_atom_line(serial=2, charge='+2')}\n"
        f"{_atom_line(serial=3, charge='2-')}\n"
    )

    stats = scan_pdb(path, surface_neighbors=2)

    assert stats["atom_record_count"] == 3
    assert stats["invalid_charge_field_count"] == 0
    assert stats["nonincreasing_atom_id_count"] == 0
    assert stats["surface_neighbor_axis_collision"] is False


def test_category_payload_deduplicates_and_sorts_ids() -> None:
    payload = category_payload(
        qbiolip_ids=["qbio-2", "qbio-1", "qbio-2"],
        afdb_ids=["afdb-2", "afdb-1", "afdb-2"],
        source="audit-source",
        severity="fatal_data",
    )

    assert payload == {
        "source": "audit-source",
        "severity": "fatal_data",
        "qbiolip_ids": ["qbio-1", "qbio-2"],
        "afdb_ids": ["afdb-1", "afdb-2"],
        "counts": {
            "qbiolip": 2,
            "afdb": 2,
            "total": 4,
        },
    }
