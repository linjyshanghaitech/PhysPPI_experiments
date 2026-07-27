"""Bounded, independent validation of PhysPPI structure resources."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import gemmi
from biotite.structure.io.mol import MOLFile
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdb.hybrid36 import decode_hybrid36

from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Finding,
    Severity,
)

_ROLE_ALIASES = {
    "gt_pdb": "gt",
    "monomer_pdbs": "templates",
    "main_msa_npzs": "main_msa",
    "paired_msa_npzs": "paired_msa",
    "prior_pkl": "prior",
}
_SHARED_FILE_ROLES = {"components_cif"}
_SHARED_DIRECTORY_ROLES = {"reference_molecule_directory"}
_REFERENCE_ROLES = ("reference_molecule", "reference_molecules")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid_charge(raw_charge: str) -> bool:
    if not raw_charge.strip():
        return False
    normalized = raw_charge if raw_charge[0] in "+-" else raw_charge[::-1]
    try:
        int(normalized)
    except ValueError:
        return True
    return False


def _fixed_width_scan(path: Path) -> tuple[dict[str, int], str]:
    digest = hashlib.sha256()
    atom_records = 0
    invalid_charge_fields = 0
    nonincreasing_atom_ids = 0
    nonfinite_coordinates = 0
    previous_atom_id: int | None = None
    residue_names: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    microheterogeneous_residues = 0

    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            line = raw_line.decode("utf-8", errors="replace")
            if line.startswith(("MODEL ", "ENDMDL")):
                microheterogeneous_residues += sum(
                    len(names) > 1 for names in residue_names.values()
                )
                previous_atom_id = None
                residue_names.clear()
                continue
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_records += 1
            invalid_charge_fields += int(_invalid_charge(line.rstrip("\r\n")[78:80]))

            try:
                atom_id = decode_hybrid36(line[6:11])
            except ValueError:
                atom_id = None
            if atom_id is not None:
                if previous_atom_id is not None and atom_id <= previous_atom_id:
                    nonincreasing_atom_ids += 1
                previous_atom_id = atom_id

            try:
                coordinates = tuple(
                    float(line[start:end])
                    for start, end in ((30, 38), (38, 46), (46, 54))
                )
            except ValueError:
                nonfinite_coordinates += 1
            else:
                nonfinite_coordinates += int(
                    not all(math.isfinite(value) for value in coordinates)
                )

            residue_key = (line[21:22], line[22:26], line[26:27])
            residue_names[residue_key].add(line[17:20].strip())
    microheterogeneous_residues += sum(
        len(names) > 1 for names in residue_names.values()
    )

    return (
        {
            "atom_record_count": atom_records,
            "invalid_charge_field_count": invalid_charge_fields,
            "nonincreasing_atom_id_count": nonincreasing_atom_ids,
            "nonfinite_coordinate_count": nonfinite_coordinates,
            "microheterogeneous_residue_count": microheterogeneous_residues,
        },
        digest.hexdigest(),
    )


def _evidenced_finding(
    *,
    code: str,
    severity: Severity,
    message: str,
    path: Path,
    evidence_sha256: str,
    role: str,
    count: int | None = None,
    exception_type: str | None = None,
) -> Finding:
    metrics: dict[str, int | str] = {"role": role, "stage": "structure"}
    if count is not None:
        metrics["count"] = count
    return Finding(
        code=code,
        severity=severity,
        message=message,
        path=path,
        exception_type=exception_type,
        evidence_sha256=evidence_sha256,
        evidence_path=path,
        metrics=metrics,
    )


def _parser_severity(role: str) -> Severity:
    if role.startswith("shared_"):
        return Severity.INFRASTRUCTURE
    return Severity.HARD_DATA


def scan_structure_file(path: Path, *, role: str) -> list[Finding]:
    """Run fixed-width, Gemmi, and Biotite checks independently."""

    path = Path(path)
    if not path.is_file():
        code = "missing_gt_pdb" if role == "gt" else f"missing_{role}_pdb"
        return [
            Finding(
                code=code,
                severity=_parser_severity(role),
                message=f"missing {role} structure",
                path=path,
                metrics={"role": role},
            )
        ]

    try:
        metrics, evidence_sha256 = _fixed_width_scan(path)
    except OSError as exc:
        return [
            Finding(
                code="structure_read_failure",
                severity=Severity.INFRASTRUCTURE,
                message=f"could not read {role} structure",
                path=path,
                exception_type=type(exc).__name__,
                metrics={"role": role},
            )
        ]

    findings: list[Finding] = []
    fixed_width_findings = (
        (
            "pdb_charge_field",
            Severity.HARD_DATA,
            "invalid_charge_field_count",
            "invalid fixed-width charge fields",
        ),
        (
            ("template_atom_id_order" if role == "template" else "atom_id_order"),
            Severity.HARD_DATA,
            "nonincreasing_atom_id_count",
            "nonincreasing fixed-width atom IDs",
        ),
        (
            "nonfinite_coordinate",
            Severity.HARD_DATA,
            "nonfinite_coordinate_count",
            "nonfinite or invalid fixed-width coordinates",
        ),
        (
            "microheterogeneous_residue",
            Severity.WARNING,
            "microheterogeneous_residue_count",
            "microheterogeneous residue positions",
        ),
    )
    for code, severity, metric, label in fixed_width_findings:
        count = metrics[metric]
        if count:
            findings.append(
                _evidenced_finding(
                    code=code,
                    severity=severity,
                    message=f"{count} {label}",
                    path=path,
                    evidence_sha256=evidence_sha256,
                    role=role,
                    count=count,
                )
            )
    if metrics["atom_record_count"] == 0:
        findings.append(
            _evidenced_finding(
                code="empty_structure",
                severity=Severity.HARD_DATA,
                message="structure has 0 fixed-width atom records",
                path=path,
                evidence_sha256=evidence_sha256,
                role=role,
                count=0,
            )
        )

    try:
        structure = gemmi.read_structure(str(path))
        gemmi_atom_count = sum(
            1
            for model in structure
            for chain in model
            for residue in chain
            for _ in residue
        )
        if gemmi_atom_count == 0 and metrics["atom_record_count"] > 0:
            raise ValueError(f"Gemmi parsed 0 atoms from {role} structure")
    except Exception as exc:
        findings.append(
            _evidenced_finding(
                code="gemmi_parser_failure",
                severity=_parser_severity(role),
                message=f"Gemmi failed to parse {role} structure",
                path=path,
                evidence_sha256=evidence_sha256,
                role=role,
                exception_type=type(exc).__name__,
            )
        )

    try:
        pdb_file = PDBFile.read(path)
        array = pdb_file.get_structure(
            model=1,
            extra_fields=["b_factor", "occupancy", "charge"],
        )
        if array.array_length() == 0 and metrics["atom_record_count"] > 0:
            raise ValueError(f"Biotite parsed 0 atoms from {role} structure")
    except Exception as exc:
        findings.append(
            _evidenced_finding(
                code="biotite_parser_failure",
                severity=_parser_severity(role),
                message=f"Biotite failed to parse {role} structure",
                path=path,
                evidence_sha256=evidence_sha256,
                role=role,
                exception_type=type(exc).__name__,
            )
        )

    return findings


def scan_reference_molecule(
    path: Path,
    *,
    residue_name: str,
) -> list[Finding]:
    """Validate one sample-used SDF without exposing its contents."""

    path = Path(path)
    if not path.is_file():
        return [
            Finding(
                code="missing_reference_molecule",
                severity=Severity.HARD_DATA,
                message=f"missing {residue_name} reference molecule",
                path=path,
                metrics={"residue_name": residue_name},
            )
        ]
    evidence_sha256: str | None = None
    try:
        evidence_sha256 = _sha256_file(path)
        mol_file = MOLFile.read(path)
        reference_array = mol_file.get_structure()
        if reference_array.array_length() == 0:
            raise ValueError(f"{residue_name} reference molecule has no atoms")
    except Exception as exc:
        return [
            Finding(
                code="invalid_reference_molecule",
                severity=Severity.HARD_DATA,
                message=f"failed to parse {residue_name} reference molecule",
                path=path,
                exception_type=type(exc).__name__,
                evidence_sha256=evidence_sha256,
                evidence_path=path if evidence_sha256 is not None else None,
                metrics={"residue_name": residue_name},
            )
        ]
    return []


def _canonical_paths(
    paths: Mapping[str, Sequence[Path]],
) -> dict[str, tuple[Path, ...]]:
    canonical: dict[str, list[Path]] = defaultdict(list)
    for input_role, values in paths.items():
        role = _ROLE_ALIASES.get(input_role, input_role)
        canonical[role].extend(Path(path) for path in values)
    return {
        role: tuple(sorted(set(values), key=str)) for role, values in canonical.items()
    }


def _missing_resource_finding(
    *,
    code: str,
    severity: Severity,
    message: str,
    path: Path | None,
    role: str,
    dataset: DatasetName,
    sample_id: str,
) -> Finding:
    return Finding(
        code=code,
        severity=severity,
        message=message,
        path=path,
        metrics={
            "role": role,
            "dataset": dataset.value,
            "sample_id": sample_id,
        },
    )


def _source_hash_key(role: str, index: int, total: int) -> str:
    return role if total == 1 else f"{role}:{index}"


def scan_sample_structures(
    dataset: DatasetName,
    sample_id: str,
    paths: Mapping[str, Sequence[Path]],
) -> tuple[list[Finding], dict[str, str]]:
    """Validate reader-located files while preserving infrastructure severity."""

    dataset = DatasetName(dataset)
    canonical = _canonical_paths(paths)
    findings: list[Finding] = []

    gt_paths = canonical.get("gt", ())
    if not gt_paths:
        findings.append(
            _missing_resource_finding(
                code="missing_gt_pdb",
                severity=Severity.HARD_DATA,
                message="sample has no reader-located GT structure",
                path=None,
                role="gt",
                dataset=dataset,
                sample_id=sample_id,
            )
        )
    for path in gt_paths:
        findings.extend(scan_structure_file(path, role="gt"))

    template_paths = canonical.get("templates", ())
    minimum_templates = 2 if dataset is DatasetName.AFDB else 1
    if len(template_paths) < minimum_templates:
        findings.append(
            _missing_resource_finding(
                code="missing_template_pdb",
                severity=Severity.HARD_DATA,
                message=(
                    f"sample has {len(template_paths)} reader-located templates; "
                    f"{minimum_templates} required"
                ),
                path=None,
                role="template",
                dataset=dataset,
                sample_id=sample_id,
            )
        )
    for path in template_paths:
        findings.extend(scan_structure_file(path, role="template"))

    for role in _SHARED_FILE_ROLES:
        role_paths = canonical.get(role)
        if role_paths is None:
            continue
        if not role_paths:
            role_paths = (None,)
        for path in role_paths:
            if path is None or not path.is_file():
                findings.append(
                    _missing_resource_finding(
                        code=f"missing_{role}",
                        severity=Severity.INFRASTRUCTURE,
                        message=f"shared {role} file is unavailable",
                        path=path,
                        role=role,
                        dataset=dataset,
                        sample_id=sample_id,
                    )
                )

    for role in _SHARED_DIRECTORY_ROLES:
        role_paths = canonical.get(role)
        if role_paths is None:
            continue
        if not role_paths:
            role_paths = (None,)
        for path in role_paths:
            if path is None or not path.is_dir():
                findings.append(
                    _missing_resource_finding(
                        code=f"missing_{role}",
                        severity=Severity.INFRASTRUCTURE,
                        message=f"shared {role} is unavailable",
                        path=path,
                        role=role,
                        dataset=dataset,
                        sample_id=sample_id,
                    )
                )

    for role in _REFERENCE_ROLES:
        for path in canonical.get(role, ()):
            findings.extend(scan_reference_molecule(path, residue_name=path.stem))

    source_hashes: dict[str, str] = {}
    for role, role_paths in sorted(canonical.items()):
        if role in _SHARED_DIRECTORY_ROLES:
            continue
        for index, path in enumerate(role_paths):
            if not path.is_file():
                if (
                    role not in {"gt", "templates"}
                    and role not in _SHARED_FILE_ROLES
                    and role not in _REFERENCE_ROLES
                ):
                    findings.append(
                        _missing_resource_finding(
                            code=f"missing_{role}_resource",
                            severity=Severity.HARD_DATA,
                            message=f"sample-specific {role} resource is missing",
                            path=path,
                            role=role,
                            dataset=dataset,
                            sample_id=sample_id,
                        )
                    )
                continue
            try:
                source_hashes[_source_hash_key(role, index, len(role_paths))] = (
                    _sha256_file(path)
                )
            except OSError as exc:
                findings.append(
                    Finding(
                        code="source_hash_failure",
                        severity=Severity.INFRASTRUCTURE,
                        message=f"could not hash {role} resource",
                        path=path,
                        exception_type=type(exc).__name__,
                        metrics={
                            "role": role,
                            "dataset": dataset.value,
                            "sample_id": sample_id,
                        },
                    )
                )

    return findings, source_hashes


__all__ = [
    "scan_reference_molecule",
    "scan_sample_structures",
    "scan_structure_file",
]
