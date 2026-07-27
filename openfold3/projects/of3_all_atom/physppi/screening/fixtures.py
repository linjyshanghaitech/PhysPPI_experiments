"""Synthetic regression fixtures for stable PhysPPI screening signatures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from openfold3.projects.of3_all_atom.physppi.screening.models import Finding
from openfold3.projects.of3_all_atom.physppi.screening.preflight import (
    _validate_collated,
    validate_physppi_features,
)
from openfold3.projects.of3_all_atom.physppi.screening.structure import (
    scan_structure_file,
)


@dataclass(frozen=True)
class FixtureCase:
    """One deterministic historical failure fixture."""

    name: str
    expected_code: str
    stage: str


@dataclass(frozen=True)
class FixtureResult:
    """Observed findings plus optional source evidence."""

    findings: tuple[Finding, ...]
    source_path: Path | None = None


_CASES = (
    FixtureCase("malformed_charge", "pdb_charge_field", "structure"),
    FixtureCase("nonincreasing_gt_atom_id", "atom_id_order", "structure"),
    FixtureCase(
        "nonincreasing_template_atom_id",
        "template_atom_id_order",
        "structure",
    ),
    FixtureCase(
        "microheterogeneous_residue",
        "microheterogeneous_residue",
        "structure",
    ),
    FixtureCase(
        "token_center_out_of_bounds",
        "token_center_out_of_bounds",
        "preflight",
    ),
    FixtureCase(
        "frame_atom_out_of_bounds",
        "frame_atom_out_of_bounds",
        "preflight",
    ),
    FixtureCase("surface_shape", "surface_shape", "preflight"),
    FixtureCase("msa_width", "msa_width_mismatch", "preflight"),
    FixtureCase(
        "template_width",
        "template_alignment_mismatch",
        "preflight",
    ),
    FixtureCase(
        "incompatible_chain_group",
        "chain_compatibility",
        "preflight",
    ),
    FixtureCase("nonfinite_feature", "nonfinite_feature", "preflight"),
    FixtureCase("collator_shape", "collated_shape", "collator"),
)


def fixture_cases() -> tuple[FixtureCase, ...]:
    """Return the canonical ordered historical fixture matrix."""

    return _CASES


def _atom_line(
    serial: int,
    residue_id: int,
    *,
    residue_name: str = "ALA",
    atom_name: str = "CA",
    charge: str = "  ",
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:^4} {residue_name:>3} A"
        f"{residue_id:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}{'':10}{'C':>2}{charge:2}\n"
    )


def _write_structure_fixture(case: FixtureCase, root: Path) -> Path:
    if case.name == "malformed_charge":
        rows = [_atom_line(1, 1, charge=" C")]
    elif case.name.startswith("nonincreasing_"):
        rows = [_atom_line(2, 1), _atom_line(1, 2)]
    elif case.name == "microheterogeneous_residue":
        rows = [
            _atom_line(1, 1, residue_name="ALA"),
            _atom_line(2, 1, residue_name="SER", atom_name="CB"),
        ]
    else:
        raise ValueError(f"not a structure fixture: {case.name}")
    path = root / f"{case.name}.pdb"
    path.write_text("".join(rows) + "END\n")
    return path


def _valid_features() -> dict[str, object]:
    n_tokens = 2
    n_atoms = 4
    features: dict[str, object] = {
        "token_mask": torch.ones(n_tokens, dtype=torch.bool),
        "atom_mask": torch.ones(n_atoms, dtype=torch.bool),
        "atom_to_token_index": torch.tensor([0, 0, 1, 1]),
        "entity_id": torch.tensor([1, 2]),
        "asym_id": torch.tensor([1, 2]),
        "msa": torch.zeros((1, n_tokens, 32)),
        "msa_mask": torch.ones((1, n_tokens)),
        "has_deletion": torch.zeros((1, n_tokens)),
        "deletion_value": torch.zeros((1, n_tokens)),
        "deletion_mean": torch.zeros(n_tokens),
        "profile": torch.zeros((n_tokens, 32)),
        "physppi_paired_msa_chain_mask": torch.ones(2, dtype=torch.bool),
        "template_restype": torch.zeros((1, n_tokens, 32)),
        "template_pseudo_beta_mask": torch.ones((1, n_tokens)),
        "template_backbone_frame_mask": torch.ones((1, n_tokens)),
        "template_distogram": torch.zeros((1, n_tokens, n_tokens, 39)),
        "template_unit_vector": torch.zeros((1, n_tokens, n_tokens, 3)),
        "ref_pos": torch.zeros((n_atoms, 3)),
        "ref_space_uid": torch.tensor([0, 0, 1, 1]),
        "ref_space_uid_to_perm": {
            0: torch.tensor([[0, 1]]),
            1: torch.tensor([[0, 1]]),
        },
        "physppi_token_center_atom_index": torch.tensor([1, 3]),
        "physppi_backbone_atom_indices": torch.tensor([[0, 1, 0], [2, 3, 2]]),
        "physppi_backbone_frame_mask": torch.ones(n_tokens, dtype=torch.bool),
        "physppi_contact_pair_mask": torch.zeros(
            (n_tokens, n_tokens), dtype=torch.bool
        ),
        "physppi_interaction_type_mask": torch.zeros(
            (n_tokens, n_tokens, 7), dtype=torch.bool
        ),
        "physppi_interaction_distance": torch.zeros((n_tokens, n_tokens)),
        "physppi_interaction_aux_valid": torch.zeros(
            (n_tokens, n_tokens), dtype=torch.bool
        ),
        "physppi_surface_local_coords": torch.zeros((n_tokens, 128, 3)),
        "physppi_surface_features": torch.zeros((n_tokens, 128, 6)),
        "physppi_surface_mask": torch.zeros((n_tokens, 128), dtype=torch.bool),
        "physppi_surface_charge_mask": torch.zeros((n_tokens, 128), dtype=torch.bool),
        "physppi_prior_requested_level": torch.tensor(4),
        "physppi_prior_effective_level": torch.tensor(4),
        "physppi_occurrence_seed": torch.tensor(7, dtype=torch.uint64),
    }
    for slot in ("contact", "interaction", "surface", "tm"):
        for state in ("available", "selected", "valid", "provenance", "active"):
            features[f"physppi_{slot}_{state}"] = torch.tensor(True)
    return features


def _run_preflight_fixture(case: FixtureCase) -> list[Finding]:
    features = _valid_features()
    matcher_diagnostics = None
    if case.name == "token_center_out_of_bounds":
        features["physppi_token_center_atom_index"][-1] = 4
    elif case.name == "frame_atom_out_of_bounds":
        features["physppi_backbone_atom_indices"][0, 0] = 4
    elif case.name == "surface_shape":
        features["physppi_surface_local_coords"] = torch.zeros((2, 127, 3))
    elif case.name == "msa_width":
        features["msa"] = torch.zeros((1, 3, 32))
    elif case.name == "template_width":
        features["template_restype"] = torch.zeros((1, 3, 32))
    elif case.name == "incompatible_chain_group":
        matcher_diagnostics = {
            "accepted": False,
            "fallback": "none",
            "reason": "no_legal_supervised_correspondence",
            "identity_rmsd": 0.0,
            "candidate_rmsd": 0.0,
            "matched_coverage": 0,
        }
    elif case.name == "nonfinite_feature":
        features["profile"][0, 0] = torch.nan
    else:
        raise ValueError(f"not a preflight fixture: {case.name}")
    return validate_physppi_features(
        features,
        matcher_diagnostics=matcher_diagnostics,
    )


def _run_collator_fixture() -> list[Finding]:
    n_tokens = 2
    batch = {
        "physppi_surface_local_coords": torch.zeros((1, n_tokens, 128, 3)),
        "physppi_surface_features": torch.zeros((1, n_tokens, 128, 6)),
        "physppi_surface_mask": torch.zeros((1, n_tokens, 128), dtype=torch.bool),
        "physppi_surface_charge_mask": torch.zeros(
            (1, n_tokens, 128), dtype=torch.bool
        ),
        "physppi_paired_msa_chain_mask": torch.ones((1, 2), dtype=torch.bool),
        "physppi_contact_pair_mask": torch.zeros(
            (1, n_tokens, n_tokens), dtype=torch.bool
        ),
        "physppi_interaction_type_mask": torch.zeros(
            (1, n_tokens, n_tokens, 7), dtype=torch.bool
        ),
        "physppi_interaction_distance": torch.zeros((1, n_tokens, n_tokens)),
        "physppi_interaction_aux_valid": torch.zeros((1, n_tokens), dtype=torch.bool),
    }
    return _validate_collated(batch, n_tokens)


def run_fixture(case: FixtureCase, root: Path) -> FixtureResult:
    """Execute one fixture through the same production screening validators."""

    root.mkdir(parents=True, exist_ok=True)
    if case.stage == "structure":
        source_path = _write_structure_fixture(case, root)
        role = "template" if case.name == "nonincreasing_template_atom_id" else "gt"
        findings = scan_structure_file(source_path, role=role)
        return FixtureResult(tuple(findings), source_path)
    if case.stage == "collator":
        return FixtureResult(tuple(_run_collator_fixture()))
    return FixtureResult(tuple(_run_preflight_fixture(case)))


__all__ = [
    "FixtureCase",
    "FixtureResult",
    "fixture_cases",
    "run_fixture",
]
