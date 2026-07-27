from pathlib import Path

import pytest

from openfold3.projects.of3_all_atom.physppi.screening.fixtures import (
    fixture_cases,
    run_fixture,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import Severity

EXPECTED = {
    "malformed_charge": "pdb_charge_field",
    "nonincreasing_gt_atom_id": "atom_id_order",
    "nonincreasing_template_atom_id": "template_atom_id_order",
    "microheterogeneous_residue": "microheterogeneous_residue",
    "token_center_out_of_bounds": "token_center_out_of_bounds",
    "frame_atom_out_of_bounds": "frame_atom_out_of_bounds",
    "surface_shape": "surface_shape",
    "msa_width": "msa_width_mismatch",
    "template_width": "template_alignment_mismatch",
    "incompatible_chain_group": "chain_compatibility",
    "nonfinite_feature": "nonfinite_feature",
    "collator_shape": "collated_shape",
}


def test_fixture_matrix_is_complete_and_ordered() -> None:
    cases = fixture_cases()

    assert tuple(case.name for case in cases) == tuple(EXPECTED)
    assert {case.name: case.expected_code for case in cases} == EXPECTED


@pytest.mark.parametrize("case", fixture_cases(), ids=lambda case: case.name)
def test_every_fixture_has_one_stable_primary_finding(
    case,
    tmp_path: Path,
) -> None:
    result = run_fixture(case, tmp_path)

    assert result.findings
    assert result.findings[0].code == EXPECTED[case.name]
    assert result.findings[0].metrics["stage"] == case.stage
    assert result.findings[0].severity in {
        Severity.WARNING,
        Severity.HARD_DATA,
        Severity.LATEST_PREFLIGHT,
    }

    if result.source_path is not None:
        primary = result.findings[0]
        assert primary.path == result.source_path
        assert primary.evidence_sha256
        assert isinstance(primary.metrics.get("count"), int)


def test_malformed_charge_preserves_independent_parser_diagnostics(
    tmp_path: Path,
) -> None:
    case = next(case for case in fixture_cases() if case.name == "malformed_charge")

    result = run_fixture(case, tmp_path)
    parser_failure = next(
        finding
        for finding in result.findings
        if finding.code == "biotite_parser_failure"
    )

    assert parser_failure.exception_type == "ValueError"
    assert parser_failure.path == result.source_path
    assert parser_failure.evidence_sha256
