import json
import pickle
from collections.abc import Mapping
from math import inf, nan
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

from openfold3.projects.of3_all_atom.physppi.screening import models
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Finding,
    SampleRecord,
    SampleStatus,
    ScanSummary,
    Severity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[6]


def _complete_summary_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "complete": True,
        "source_counts": {
            DatasetName.QBIOLIP: 10_000,
            DatasetName.AFDB: 10_000,
        },
        "terminal_record_count": 20_000,
        "duplicate_terminal_record_ids": [],
        "status_counts": {SampleStatus.PASS: 20_000},
        "category_counts": {},
        "unresolved_infrastructure_ids": [],
        "artifact_sha256": {},
    }
    values.update(overrides)
    return values


def _sample_record_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "ordinal": 4,
        "dataset": DatasetName.AFDB,
        "sample_id": "a1",
        "findings": [],
        "audit_views": [
            models.AuditView(
                crop_strategy="spatial",
                requested_level=0,
                interface_cutoff=8.0,
                view_index=0,
            )
        ],
        "historical_signatures": [],
        "elapsed_seconds": 0.2,
        "source_hashes": {"gt": "abc"},
        "validator_versions": {"scanner": "2"},
    }
    values.update(overrides)
    return values


def _scan_metadata_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "scanner_version": "2",
        "git_head": "abc123",
        "source_roots": {
            DatasetName.QBIOLIP: Path("/data/qbiolip"),
            DatasetName.AFDB: Path("/data/afdb"),
        },
        "reader_cache_sha256": {
            DatasetName.QBIOLIP: "q-cache",
            DatasetName.AFDB: "a-cache",
        },
        "canonical_report_sha256": {
            DatasetName.QBIOLIP: "q-report",
            DatasetName.AFDB: "a-report",
        },
        "dependency_versions": {"pydantic": "2.13.4"},
        "worker_count": 8,
        "sample_timeout_seconds": 600,
        "max_attempts": 2,
        "crop_strategies": ("spatial", "spatial_interface"),
        "cli_options": {"resume": False, "ids": ["q1", "a1"]},
    }
    values.update(overrides)
    return values


def test_screening_dependencies_are_bounded() -> None:
    expected = {
        "gemmi": ">=0.7,<0.8",
        "biotite": ">=1.6,<2",
        "scipy": ">=1.17,<2",
        "pydantic": ">=2.12,<3",
    }

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project_dependencies = pyproject["project"]["dependencies"]
    assert {f"{name}{constraint}" for name, constraint in expected.items()} <= set(
        project_dependencies
    )

    pixi = tomllib.loads((PROJECT_ROOT / "pixi.toml").read_text())
    pixi_dependencies = pixi["feature"]["openfold3-cpu"]["dependencies"]
    assert {name: pixi_dependencies.get(name) for name in expected} == expected


def test_audit_view_is_strict_bounded_and_immutable() -> None:
    view = models.AuditView(
        crop_strategy="spatial_interface",
        requested_level=4,
        interface_cutoff=13.0,
        view_index=6,
        multichain_probe=True,
    )

    assert view.requested_level == 4
    assert view.multichain_probe is True
    with pytest.raises(ValidationError, match="less than or equal to 18"):
        models.AuditView(
            crop_strategy="spatial",
            requested_level=0,
            interface_cutoff=18.1,
            view_index=0,
        )
    with pytest.raises(ValidationError, match="Input should be 0, 1, 2, 3 or 4"):
        models.AuditView(
            crop_strategy="spatial",
            requested_level="4",
            interface_cutoff=13.0,
            view_index=0,
        )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        view.view_index = 7


def test_sample_record_tracks_audit_views() -> None:
    audit_view = models.AuditView(
        crop_strategy="spatial",
        requested_level=0,
        interface_cutoff=8.0,
        view_index=0,
    )
    record = SampleRecord.classify(
        ordinal=4,
        dataset=DatasetName.AFDB,
        sample_id="a1",
        findings=[],
        audit_views=[audit_view],
        historical_signatures=[],
        elapsed_seconds=0.2,
        source_hashes={"gt": "abc"},
        validator_versions={"scanner": "2"},
    )

    assert record.schema_version == models.SCHEMA_VERSION == 2
    assert record.audit_views == (audit_view,)


def test_excluded_record_requires_excluding_finding() -> None:
    with pytest.raises(ValidationError, match="excluding finding"):
        SampleRecord(
            ordinal=0,
            dataset=DatasetName.QBIOLIP,
            sample_id="q1",
            status=SampleStatus.EXCLUDED,
            findings=[],
            audit_views=[],
            elapsed_seconds=0.1,
            historical_signatures=[],
            source_hashes={},
            validator_versions={"scanner": "2"},
        )


def test_warning_record_cannot_contain_hard_finding() -> None:
    hard = Finding(
        code="missing_gt_pdb",
        severity=Severity.HARD_DATA,
        message="missing",
        path=Path("/data/q1.pdb"),
    )
    with pytest.raises(ValidationError, match="warning record"):
        SampleRecord(
            ordinal=0,
            dataset=DatasetName.QBIOLIP,
            sample_id="q1",
            status=SampleStatus.WARNING,
            findings=[hard],
            audit_views=[],
            elapsed_seconds=0.1,
            historical_signatures=[],
            source_hashes={},
            validator_versions={"scanner": "2"},
        )


def test_passed_historical_sample_is_recovered() -> None:
    record = SampleRecord.classify(
        ordinal=4,
        dataset=DatasetName.AFDB,
        sample_id="a1",
        findings=[],
        audit_views=[],
        historical_signatures=["surface_shape"],
        elapsed_seconds=0.2,
        source_hashes={"gt": "abc"},
        validator_versions={"scanner": "2"},
    )
    assert record.status is SampleStatus.RECOVERED


def test_complete_summary_requires_exact_20k_partition() -> None:
    with pytest.raises(ValidationError, match="10,000 IDs per source"):
        ScanSummary(
            **_complete_summary_values(
                source_counts={
                    DatasetName.QBIOLIP: 10_000,
                    DatasetName.AFDB: 9_999,
                },
                terminal_record_count=19_999,
                status_counts={SampleStatus.PASS: 19_999},
            )
        )


def test_complete_summary_rejects_infrastructure_failures() -> None:
    with pytest.raises(ValidationError, match="infrastructure failures"):
        ScanSummary(
            **_complete_summary_values(unresolved_infrastructure_ids=["afdb:a1"])
        )


def test_complete_summary_requires_exactly_20k_terminal_records() -> None:
    with pytest.raises(ValidationError, match="exactly 20,000 terminal records"):
        ScanSummary(
            **_complete_summary_values(
                terminal_record_count=19_999,
                status_counts={SampleStatus.PASS: 19_999},
            )
        )


def test_complete_summary_rejects_duplicate_dataset_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate terminal records"):
        ScanSummary(
            **_complete_summary_values(
                duplicate_terminal_record_ids=[(DatasetName.QBIOLIP, "q1")]
            )
        )


def test_complete_summary_accepts_exact_unique_terminal_partition() -> None:
    summary = ScanSummary(**_complete_summary_values())

    assert summary.schema_version == models.SCHEMA_VERSION == 2
    assert summary.complete is True


def test_infrastructure_and_hard_findings_cannot_be_excluded() -> None:
    infrastructure = Finding(
        code="read_timeout",
        severity=Severity.INFRASTRUCTURE,
        message="timeout",
    )
    hard = Finding(
        code="missing_gt_pdb",
        severity=Severity.HARD_DATA,
        message="missing",
    )
    with pytest.raises(ValidationError, match="canonical status infrastructure_error"):
        SampleRecord(
            ordinal=0,
            dataset=DatasetName.QBIOLIP,
            sample_id="q1",
            status=SampleStatus.EXCLUDED,
            findings=[infrastructure, hard],
            audit_views=[],
            elapsed_seconds=0.1,
            historical_signatures=[],
            source_hashes={},
            validator_versions={"scanner": "2"},
        )


def test_hard_finding_cannot_be_pass() -> None:
    hard = Finding(
        code="missing_gt_pdb",
        severity=Severity.HARD_DATA,
        message="missing",
    )
    with pytest.raises(ValidationError, match="canonical status excluded"):
        SampleRecord(
            ordinal=0,
            dataset=DatasetName.QBIOLIP,
            sample_id="q1",
            status=SampleStatus.PASS,
            findings=[hard],
            audit_views=[],
            elapsed_seconds=0.1,
            historical_signatures=[],
            source_hashes={},
            validator_versions={"scanner": "2"},
        )


def test_canonical_infrastructure_status_is_accepted() -> None:
    infrastructure = Finding(
        code="read_timeout",
        severity=Severity.INFRASTRUCTURE,
        message="timeout",
    )
    hard = Finding(
        code="missing_gt_pdb",
        severity=Severity.HARD_DATA,
        message="missing",
    )
    record = SampleRecord(
        ordinal=0,
        dataset=DatasetName.QBIOLIP,
        sample_id="q1",
        status=SampleStatus.INFRASTRUCTURE_ERROR,
        findings=[infrastructure, hard],
        audit_views=[],
        elapsed_seconds=0.1,
        historical_signatures=[],
        source_hashes={},
        validator_versions={"scanner": "2"},
    )

    assert record.status is SampleStatus.INFRASTRUCTURE_ERROR


def test_sample_record_nested_state_cannot_mutate_after_validation() -> None:
    warning = Finding(
        code="legacy_warning",
        severity=Severity.WARNING,
        message="warning",
        metrics={"count": 1},
    )
    record = SampleRecord.classify(
        **_sample_record_values(findings=[warning], audit_views=[])
    )

    with pytest.raises(AttributeError):
        record.findings.append(
            Finding(
                code="late_hard_failure",
                severity=Severity.HARD_DATA,
                message="must not be injected after validation",
            )
        )
    with pytest.raises(TypeError):
        record.findings[0].metrics["count"] = nan
    with pytest.raises(TypeError):
        record.source_hashes["gt"] = "mutated"

    assert json.loads(record.model_dump_json())["status"] == "warning"


def test_complete_summary_nested_state_cannot_mutate_after_validation() -> None:
    summary = ScanSummary(**_complete_summary_values())

    with pytest.raises(TypeError):
        summary.source_counts[DatasetName.QBIOLIP] = 9_999
    with pytest.raises(AttributeError):
        summary.duplicate_terminal_record_ids.append((DatasetName.AFDB, "a1"))
    with pytest.raises(AttributeError):
        summary.unresolved_infrastructure_ids.append("afdb:a1")

    assert json.loads(summary.model_dump_json())["complete"] is True


def test_sample_record_json_round_trip_preserves_immutable_state() -> None:
    record = SampleRecord.classify(**_sample_record_values())
    payload = record.model_dump_json()
    restored = SampleRecord.model_validate_json(payload)

    assert restored == record
    assert isinstance(restored.findings, tuple)
    assert isinstance(restored.source_hashes, Mapping)
    assert isinstance(json.loads(payload)["findings"], list)
    assert isinstance(json.loads(payload)["source_hashes"], dict)
    with pytest.raises(TypeError):
        restored.validator_versions["scanner"] = "invalid"


def test_sample_record_is_pickle_safe_for_dataloader_workers() -> None:
    record = SampleRecord.classify(**_sample_record_values())

    restored = pickle.loads(pickle.dumps(record))

    assert restored == record
    assert restored.model_dump_json() == record.model_dump_json()
    with pytest.raises(TypeError):
        restored.source_hashes["gt"] = "invalid"


def test_complete_summary_json_round_trip_preserves_immutable_state() -> None:
    summary = ScanSummary(**_complete_summary_values())
    payload = summary.model_dump_json()
    restored = ScanSummary.model_validate_json(payload)

    assert restored == summary
    assert isinstance(restored.duplicate_terminal_record_ids, tuple)
    assert isinstance(restored.source_counts, Mapping)
    assert isinstance(json.loads(payload)["duplicate_terminal_record_ids"], list)
    assert isinstance(json.loads(payload)["source_counts"], dict)
    with pytest.raises(TypeError):
        restored.status_counts[SampleStatus.PASS] = 19_999


def test_historical_seed_nested_state_is_immutable_and_serialization_safe() -> None:
    seed = models.HistoricalSeed(
        dataset=DatasetName.QBIOLIP,
        sample_id="q1",
        signatures=["atom_ids_not_increasing"],
        sources=["v1-v32"],
    )

    with pytest.raises(AttributeError):
        seed.signatures.append("late_failure")
    with pytest.raises(AttributeError):
        seed.sources.append("mutated_source")

    restored_json = models.HistoricalSeed.model_validate_json(seed.model_dump_json())
    restored_pickle = pickle.loads(pickle.dumps(seed))
    assert restored_json == seed
    assert restored_pickle == seed
    assert isinstance(restored_json.signatures, tuple)
    assert isinstance(json.loads(seed.model_dump_json())["signatures"], list)
    with pytest.raises(AttributeError):
        restored_pickle.sources.append("invalid")


def test_scan_metadata_nested_state_is_immutable_and_serialization_safe() -> None:
    metadata = models.ScanMetadata(**_scan_metadata_values())

    with pytest.raises(TypeError):
        metadata.source_roots[DatasetName.QBIOLIP] = Path("/mutated")
    with pytest.raises(TypeError):
        metadata.dependency_versions["pydantic"] = "mutated"
    with pytest.raises(TypeError):
        metadata.cli_options["resume"] = True
    with pytest.raises(AttributeError):
        metadata.cli_options["ids"].append("late-id")

    payload = metadata.model_dump_json()
    restored_json = models.ScanMetadata.model_validate_json(payload)
    restored_pickle = pickle.loads(pickle.dumps(metadata))
    assert restored_json == metadata
    assert restored_pickle == metadata
    assert isinstance(restored_json.source_roots, Mapping)
    assert isinstance(restored_json.cli_options["ids"], tuple)
    assert isinstance(json.loads(payload)["source_roots"], dict)
    assert isinstance(json.loads(payload)["cli_options"]["ids"], list)
    with pytest.raises(AttributeError):
        restored_pickle.cli_options["ids"].append("invalid")


@pytest.mark.parametrize("value", [nan, inf, -inf], ids=["nan", "inf", "neg_inf"])
def test_finding_rejects_nonfinite_numeric_metrics(value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            code="nonfinite_metric",
            severity=Severity.WARNING,
            message="invalid metric",
            metrics={"value": value},
        )

    assert "finite_number" in {error["type"] for error in exc_info.value.errors()}


@pytest.mark.parametrize("value", [nan, inf, -inf], ids=["nan", "inf", "neg_inf"])
def test_audit_view_rejects_nonfinite_numeric_fields(value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        models.AuditView(
            crop_strategy="spatial",
            requested_level=0,
            interface_cutoff=value,
            view_index=0,
        )

    assert {error["type"] for error in exc_info.value.errors()} == {"finite_number"}


def test_sample_record_rejects_negative_elapsed_seconds() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SampleRecord.classify(**_sample_record_values(elapsed_seconds=-0.1))


@pytest.mark.parametrize("value", [nan, inf, -inf], ids=["nan", "inf", "neg_inf"])
def test_sample_record_rejects_nonfinite_elapsed_seconds(value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SampleRecord.classify(**_sample_record_values(elapsed_seconds=value))

    assert {error["type"] for error in exc_info.value.errors()} == {"finite_number"}
