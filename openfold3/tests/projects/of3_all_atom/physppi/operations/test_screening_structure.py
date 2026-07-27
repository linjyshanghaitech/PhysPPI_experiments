import hashlib
import importlib
from pathlib import Path
from types import ModuleType

import pytest

from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Severity,
)


def _structure_module() -> ModuleType:
    try:
        return importlib.import_module("openfold3.projects.of3_all_atom.physppi.screening.structure")
    except ModuleNotFoundError as exc:
        pytest.fail(f"structure validator is missing: {exc}")


def _coordinate_field(value: float | str) -> str:
    if isinstance(value, str):
        return f"{value:>8}"
    return f"{value:8.3f}"


def _atom_line(
    serial: int | str,
    residue_id: int,
    *,
    record: str = "ATOM",
    residue_name: str = "ALA",
    atom_name: str = "CA",
    element: str = "C",
    charge: str = "  ",
    x: float | str = 0.0,
) -> str:
    serial_field = f"{serial:5d}" if isinstance(serial, int) else f"{serial:>5}"
    return (
        f"{record:<6}{serial_field} {atom_name:^4} {residue_name:>3} A"
        f"{residue_id:4d}    "
        f"{_coordinate_field(x)}{0.0:8.3f}{0.0:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}{'':10}{element:>2}{charge:2}\n"
    )


def _write_pdb(path: Path, *lines: str) -> Path:
    path.write_text("".join(lines) + "END\n")
    return path


def _valid_mse_sdf() -> str:
    return """MSE
  PhysPPI

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Se  0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$
"""


def _bad_mse_sdf() -> str:
    return """MSE
  PhysPPI

  1  0  0  0  0  0  0  0  0  0999 V2000
not an atom record
M  END
$$$$
"""


def test_fixed_width_and_independent_parsers_report_expected_findings(
    tmp_path: Path,
) -> None:
    pdb = _write_pdb(
        tmp_path / "malformed.pdb",
        _atom_line(2, 1, charge=" C"),
        _atom_line(1, 1, residue_name="SER"),
    )

    findings = _structure_module().scan_structure_file(pdb, role="gt")
    by_code = {finding.code: finding for finding in findings}

    assert by_code["pdb_charge_field"].severity is Severity.HARD_DATA
    assert by_code["atom_id_order"].severity is Severity.HARD_DATA
    assert by_code["microheterogeneous_residue"].severity is Severity.WARNING
    assert by_code["biotite_parser_failure"].exception_type == "ValueError"
    assert "gemmi_parser_failure" not in by_code
    assert all(
        finding.evidence_sha256 == hashlib.sha256(pdb.read_bytes()).hexdigest()
        for finding in findings
    )


@pytest.mark.parametrize(
    "serials",
    [
        (2, 1),
        ("A0000", "99999"),
    ],
    ids=["decimal", "hybrid-36"],
)
def test_nonincreasing_atom_ids_support_decimal_and_hybrid36(
    tmp_path: Path,
    serials: tuple[int | str, int | str],
) -> None:
    pdb = _write_pdb(
        tmp_path / "decreasing.pdb",
        _atom_line(serials[0], 1),
        _atom_line(serials[1], 2),
    )

    findings = _structure_module().scan_structure_file(pdb, role="template")

    atom_id_finding = next(
        finding for finding in findings if finding.code == "template_atom_id_order"
    )
    assert atom_id_finding.metrics["count"] == 1
    assert atom_id_finding.metrics["role"] == "template"


def test_model_boundaries_reset_fixed_width_scopes(tmp_path: Path) -> None:
    pdb = _write_pdb(
        tmp_path / "multi-model.pdb",
        "MODEL        1\n",
        _atom_line(1, 1, residue_name="ALA"),
        "ENDMDL\n",
        "MODEL        2\n",
        _atom_line(1, 1, residue_name="SER"),
        "ENDMDL\n",
    )

    finding_codes = {
        finding.code
        for finding in _structure_module().scan_structure_file(pdb, role="gt")
    }
    assert "atom_id_order" not in finding_codes
    assert "template_atom_id_order" not in finding_codes
    assert "microheterogeneous_residue" not in finding_codes


def test_nonfinite_coordinate_is_hard_data(tmp_path: Path) -> None:
    pdb = _write_pdb(tmp_path / "nan.pdb", _atom_line(1, 1, x="nan"))

    findings = _structure_module().scan_structure_file(pdb, role="gt")

    finding = next(
        finding for finding in findings if finding.code == "nonfinite_coordinate"
    )
    assert finding.severity is Severity.HARD_DATA
    assert finding.metrics["count"] == 1


def test_empty_pdb_is_hard_data(tmp_path: Path) -> None:
    pdb = tmp_path / "empty.pdb"
    pdb.write_text("HEADER    EMPTY STRUCTURE\nEND\n")

    findings = _structure_module().scan_structure_file(pdb, role="gt")

    empty_findings = [
        finding for finding in findings if finding.code == "empty_structure"
    ]
    assert len(empty_findings) == 1
    assert empty_findings[0].severity is Severity.HARD_DATA


def test_valid_mse_structure_and_reference_parse_cleanly(tmp_path: Path) -> None:
    pdb = _write_pdb(
        tmp_path / "mse.pdb",
        _atom_line(
            1,
            1,
            record="HETATM",
            residue_name="MSE",
            atom_name="SE",
            element="SE",
        ),
    )
    sdf = tmp_path / "MSE.sdf"
    sdf.write_text(_valid_mse_sdf())

    assert _structure_module().scan_structure_file(pdb, role="gt") == []
    assert _structure_module().scan_reference_molecule(sdf, residue_name="MSE") == []


def test_bad_mse_reference_has_one_bounded_hard_finding(tmp_path: Path) -> None:
    sdf = tmp_path / "MSE.sdf"
    sdf.write_text(_bad_mse_sdf())

    findings = _structure_module().scan_reference_molecule(
        sdf,
        residue_name="MSE",
    )

    assert {finding.code for finding in findings} == {"invalid_reference_molecule"}
    finding = findings[0]
    assert finding.severity is Severity.HARD_DATA
    assert finding.exception_type == "ValueError"
    assert finding.evidence_sha256 == hashlib.sha256(sdf.read_bytes()).hexdigest()
    assert "not an atom record" not in finding.message


def test_reference_role_aliases_have_deterministic_finding_and_hash_order(
    tmp_path: Path,
) -> None:
    gt = _write_pdb(tmp_path / "sample.pdb", _atom_line(1, 1))
    template = _write_pdb(tmp_path / "sample_A.pdb", _atom_line(1, 1))
    singular_reference = tmp_path / "SINGULAR.sdf"
    singular_reference.write_text(_bad_mse_sdf())
    plural_reference = tmp_path / "PLURAL.sdf"
    plural_reference.write_text(_bad_mse_sdf())

    findings, source_hashes = _structure_module().scan_sample_structures(
        DatasetName.QBIOLIP,
        "q-reference-order",
        {
            "gt": [gt],
            "templates": [template],
            "reference_molecules": [plural_reference],
            "reference_molecule": [singular_reference],
        },
    )

    assert [
        finding.path
        for finding in findings
        if finding.code == "invalid_reference_molecule"
    ] == [singular_reference, plural_reference]
    assert list(source_hashes) == [
        "gt",
        "reference_molecule",
        "reference_molecules",
        "templates",
    ]


def test_missing_sample_gt_is_hard_but_shared_resources_are_infrastructure(
    tmp_path: Path,
) -> None:
    findings, source_hashes = _structure_module().scan_sample_structures(
        DatasetName.QBIOLIP,
        "q-missing",
        {
            "gt": [tmp_path / "missing-gt.pdb"],
            "components_cif": [tmp_path / "missing-components.cif"],
            "reference_molecule_directory": [tmp_path / "missing-reference-mols"],
        },
    )
    by_code = {finding.code: finding for finding in findings}

    assert by_code["missing_gt_pdb"].severity is Severity.HARD_DATA
    assert by_code["missing_components_cif"].severity is Severity.INFRASTRUCTURE
    assert (
        by_code["missing_reference_molecule_directory"].severity
        is Severity.INFRASTRUCTURE
    )
    assert source_hashes == {}


def test_sample_scan_hashes_reader_resources_and_scans_both_afdb_templates(
    tmp_path: Path,
) -> None:
    gt = _write_pdb(tmp_path / "sample.pdb", _atom_line(1, 1))
    template_a = _write_pdb(
        tmp_path / "sample_A.pdb",
        _atom_line(2, 1),
        _atom_line(1, 2),
    )
    template_b = _write_pdb(
        tmp_path / "sample_B.pdb",
        _atom_line("A0000", 1),
        _atom_line("99999", 2),
    )
    prior = tmp_path / "sample.pkl"
    prior.write_bytes(b"prior payload")
    main_msa = tmp_path / "main.npz"
    main_msa.write_bytes(b"main msa")
    paired_msa = tmp_path / "paired.npz"
    paired_msa.write_bytes(b"paired msa")
    ccd = tmp_path / "components.cif"
    ccd.write_text("data_components\n")
    reference_directory = tmp_path / "reference_mols"
    reference_directory.mkdir()
    mse = reference_directory / "MSE.sdf"
    mse.write_text(_valid_mse_sdf())

    findings, source_hashes = _structure_module().scan_sample_structures(
        DatasetName.AFDB,
        "a-structure",
        {
            "gt_pdb": [gt],
            "monomer_pdbs": [template_b, template_a],
            "prior_pkl": [prior],
            "main_msa_npzs": [main_msa],
            "paired_msa_npzs": [paired_msa],
            "surface": [prior],
            "interaction": [prior],
            "opm": [prior],
            "components_cif": [ccd],
            "reference_molecule_directory": [reference_directory],
            "reference_molecules": [mse],
        },
    )

    template_findings = [
        finding
        for finding in findings
        if finding.code == "template_atom_id_order"
        and finding.metrics["role"] == "template"
    ]
    assert {finding.path for finding in template_findings} == {
        template_a,
        template_b,
    }
    assert set(source_hashes) == {
        "gt",
        "templates:0",
        "templates:1",
        "prior",
        "main_msa",
        "paired_msa",
        "surface",
        "interaction",
        "opm",
        "components_cif",
        "reference_molecules",
    }
    assert all(len(digest) == 64 for digest in source_hashes.values())
