from __future__ import annotations

import json
from pathlib import Path

import pytest

from openfold3.projects.of3_all_atom.physppi.screening.artifacts import (
    finalize_artifacts,
    verify_sha256s,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Finding,
    SampleRecord,
    ScanMetadata,
    Severity,
)


def _record(
    dataset: DatasetName,
    ordinal: int,
    sample_id: str,
    *,
    historical: bool = False,
    excluded: bool = False,
) -> SampleRecord:
    findings = (
        [
            Finding(
                code="fixture_error",
                severity=Severity.HARD_DATA,
                message="fixture",
            )
        ]
        if excluded
        else []
    )
    return SampleRecord.classify(
        ordinal=ordinal,
        dataset=dataset,
        sample_id=sample_id,
        findings=findings,
        audit_views=[],
        historical_signatures=["historical"] if historical else [],
        elapsed_seconds=0.01,
        source_hashes={},
        validator_versions={"scanner": "fixture"},
    )


def _production_fixture(
    *,
    recovered_id: str = "q-00000",
    excluded_id: str = "a-00000",
) -> tuple[
    list[SampleRecord],
    dict[DatasetName, tuple[str, ...]],
    dict[DatasetName, tuple[str, ...]],
]:
    qbiolip_ids = tuple(f"q-{index:05d}" for index in range(10_000))
    afdb_ids = tuple(f"a-{index:05d}" for index in range(10_000))
    records = [
        _record(
            DatasetName.QBIOLIP,
            index,
            sample_id,
            historical=sample_id == recovered_id,
        )
        for index, sample_id in enumerate(qbiolip_ids)
    ]
    records.extend(
        _record(
            DatasetName.AFDB,
            10_000 + index,
            sample_id,
            excluded=sample_id == excluded_id,
        )
        for index, sample_id in enumerate(afdb_ids)
    )
    inventories = {
        DatasetName.QBIOLIP: qbiolip_ids,
        DatasetName.AFDB: afdb_ids,
    }
    historical = {
        DatasetName.QBIOLIP: (recovered_id,),
        DatasetName.AFDB: (),
    }
    return records, inventories, historical


def _metadata(tmp_path: Path) -> ScanMetadata:
    return ScanMetadata(
        scanner_version="fixture",
        git_head="1" * 40,
        source_roots={
            DatasetName.QBIOLIP: tmp_path / "qbiolip",
            DatasetName.AFDB: tmp_path / "afdb",
        },
        reader_cache_sha256={
            DatasetName.QBIOLIP: "2" * 64,
            DatasetName.AFDB: "3" * 64,
        },
        canonical_report_sha256={
            DatasetName.QBIOLIP: "4" * 64,
            DatasetName.AFDB: "5" * 64,
        },
        dependency_versions={"scanner": "fixture"},
        worker_count=8,
        sample_timeout_seconds=600,
        max_attempts=2,
        crop_strategies=("spatial", "spatial_interface"),
        cli_options={},
    )


def test_historical_quarantine_is_union_even_when_recovered(
    tmp_path: Path,
) -> None:
    records, inventories, historical = _production_fixture()

    finalize_artifacts(
        records=records,
        historical_ids=historical,
        source_ids=inventories,
        metadata=_metadata(tmp_path),
        output=tmp_path / "artifacts",
        implementation_git_head="a" * 40,
        screen_config_sha256="b" * 64,
        evidence={"fixture": True},
    )

    root = tmp_path / "artifacts"
    errors = json.loads((root / "data-error-ids.json").read_text())
    assert "q-00000" in errors["datasets"]["qbiolip"]["ids"]
    assert "a-00000" in errors["datasets"]["afdb"]["ids"]
    manifest = json.loads((root / "screened-training-manifest.json").read_text())
    assert "q-00000" not in manifest["datasets"]["qbiolip"]["train_ids"]
    assert "a-00000" not in manifest["datasets"]["afdb"]["train_ids"]
    assert manifest["datasets"]["qbiolip"]["train_ids"][:2] == [
        "q-00001",
        "q-00002",
    ]
    assert manifest["source_inventory"] == {"afdb": 10_000, "qbiolip": 10_000}


def test_complete_is_written_last_and_all_hashes_verify(tmp_path: Path) -> None:
    records, inventories, historical = _production_fixture()
    output = tmp_path / "artifacts"

    finalize_artifacts(
        records=records,
        historical_ids=historical,
        source_ids=inventories,
        metadata=_metadata(tmp_path),
        output=output,
        implementation_git_head="a" * 40,
        screen_config_sha256="b" * 64,
        evidence={"fixture": True},
    )

    assert (output / "COMPLETE").is_file()
    assert verify_sha256s(output / "SHA256SUMS")
    written = [
        line
        for line in (output / "screen-results.jsonl").read_text().splitlines()
        if line
    ]
    assert len(written) == 20_000
    assert SampleRecord.model_validate_json(written[-1]).sample_id == "a-09999"


def test_finalization_rejects_incomplete_or_infrastructure_records(
    tmp_path: Path,
) -> None:
    records, inventories, historical = _production_fixture()
    with pytest.raises(ValueError, match="exactly 20,000"):
        finalize_artifacts(
            records=records[:-1],
            historical_ids=historical,
            source_ids=inventories,
            metadata=_metadata(tmp_path),
            output=tmp_path / "incomplete",
            implementation_git_head="a" * 40,
            screen_config_sha256="b" * 64,
            evidence={},
        )
