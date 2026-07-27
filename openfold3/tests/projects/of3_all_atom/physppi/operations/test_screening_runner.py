from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from pathlib import Path

import pytest

from openfold3.projects.of3_all_atom.physppi.screening.dataset import (
    ScreeningDataset,
    build_screening_items,
    identity_collate,
    load_cached_source_ids,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    Finding,
    HistoricalSeed,
    SampleRecord,
    SampleStatus,
    ScreeningOptions,
    Severity,
)
from openfold3.projects.of3_all_atom.physppi.screening.runner import (
    build_screening_dataloader,
    run_screening,
)


def _touch(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _options(tmp_path: Path, *, resume: bool = False) -> ScreeningOptions:
    qbiolip_root = tmp_path / "qbiolip"
    afdb_root = tmp_path / "afdb"
    reference = tmp_path / "reference"
    runs = tmp_path / "runs"
    manifests = tmp_path / "manifests"
    for directory in (qbiolip_root, afdb_root, reference, runs, manifests):
        directory.mkdir(parents=True, exist_ok=True)
    return ScreeningOptions(
        qbiolip_root=qbiolip_root,
        afdb_root=afdb_root,
        qbiolip_index_cache=_touch(tmp_path / "q-index.pkl", "q-index"),
        afdb_index_cache=_touch(tmp_path / "a-index.pkl", "a-index"),
        qbiolip_canonical_report=_touch(tmp_path / "q-report.json", "{}"),
        afdb_canonical_report=_touch(tmp_path / "a-report.json", "{}"),
        history_runs_root=runs,
        manifests_dir=manifests,
        runtime_ledger=_touch(tmp_path / "ledger.jsonl", ""),
        prior_audit=_touch(tmp_path / "prior.json", '{"categories": {}}'),
        components_cif=_touch(tmp_path / "components.cif"),
        reference_molecule_directory=reference,
        output_root=tmp_path / "output",
        workers=8,
        sample_timeout_seconds=60,
        max_attempts=2,
        resume=resume,
    )


def _seed(dataset: DatasetName, sample_id: str) -> HistoricalSeed:
    return HistoricalSeed(
        dataset=dataset,
        sample_id=sample_id,
        signatures=["historical"],
        sources=["fixture"],
    )


def _record(
    dataset: DatasetName,
    ordinal: int,
    sample_id: str,
    *,
    severity: Severity | None = None,
    historical_seed: HistoricalSeed | None = None,
) -> SampleRecord:
    findings = (
        []
        if severity is None
        else [
            Finding(
                code=f"{severity.value}_fixture",
                severity=severity,
                message="fixture",
            )
        ]
    )
    return SampleRecord.classify(
        ordinal=ordinal,
        dataset=dataset,
        sample_id=sample_id,
        findings=findings,
        audit_views=[],
        historical_signatures=(
            [] if historical_seed is None else historical_seed.signatures
        ),
        elapsed_seconds=0.01,
        source_hashes={},
        validator_versions={},
    )


def _direct_loader(dataset: ScreeningDataset) -> Iterable[list[SampleRecord]]:
    for index in range(len(dataset)):
        yield identity_collate([dataset[index]])


def test_history_is_front_loaded_without_duplicate_ids() -> None:
    source_ids = {
        DatasetName.QBIOLIP: ("q1", "q2", "q3"),
        DatasetName.AFDB: ("a1", "a2"),
    }
    history = {
        (DatasetName.AFDB, "a2"): _seed(DatasetName.AFDB, "a2"),
        (DatasetName.QBIOLIP, "q2"): _seed(DatasetName.QBIOLIP, "q2"),
        (DatasetName.QBIOLIP, "not-in-source"): _seed(
            DatasetName.QBIOLIP, "not-in-source"
        ),
    }

    items = build_screening_items(source_ids, history)

    assert [(item.dataset, item.sample_id) for item in items] == [
        (DatasetName.QBIOLIP, "q2"),
        (DatasetName.AFDB, "a2"),
        (DatasetName.QBIOLIP, "q1"),
        (DatasetName.QBIOLIP, "q3"),
        (DatasetName.AFDB, "a1"),
    ]
    assert len({(item.dataset, item.sample_id) for item in items}) == len(items)
    assert items[0].ordinal == 1
    assert items[1].ordinal == 4


def test_dataset_runs_all_audit_views_and_emits_one_record(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    calls: list[tuple[str, int]] = []

    def preflight(dataset, index, sample_id, view):
        calls.append((view.crop_strategy, view.view_index))
        return []

    item = build_screening_items(
        {DatasetName.QBIOLIP: ("q1",), DatasetName.AFDB: ()},
        {},
    )[0]
    dataset = ScreeningDataset((item,), options, preflight=preflight)

    record = dataset[0]

    assert record.status is SampleStatus.PASS
    assert record.sample_id == "q1"
    assert [view.view_index for view in record.audit_views] == list(range(7))
    assert calls == [
        ("spatial", 0),
        ("spatial_interface", 1),
        ("spatial", 2),
        ("spatial_interface", 3),
        ("spatial", 4),
        ("spatial_interface", 5),
        ("spatial", 6),
    ]


def test_dataset_classifies_worker_protocol_exception_as_infrastructure(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)

    def crashing_preflight(*_args):
        raise RuntimeError("worker protocol broke")

    item = build_screening_items(
        {DatasetName.QBIOLIP: ("q1",), DatasetName.AFDB: ()},
        {},
    )[0]

    record = ScreeningDataset((item,), options, preflight=crashing_preflight)[0]

    assert record.status is SampleStatus.INFRASTRUCTURE_ERROR
    assert {finding.severity for finding in record.findings} == {
        Severity.INFRASTRUCTURE
    }
    assert len(record.audit_views) == 1


def test_dataloader_uses_exact_production_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = _options(tmp_path)
    captured = {}

    class CapturingLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured.update(kwargs)

    monkeypatch.setattr(
        "openfold3.projects.of3_all_atom.physppi.screening.runner.DataLoader", CapturingLoader
    )
    dataset = ScreeningDataset((), options, preflight=lambda *_: [])

    loader = build_screening_dataloader(dataset)

    assert isinstance(loader, CapturingLoader)
    assert captured == {
        "dataset": dataset,
        "batch_size": 1,
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "shuffle": False,
        "multiprocessing_context": "forkserver",
        "collate_fn": identity_collate,
    }


def test_parent_retries_only_infrastructure_once_in_fresh_generation(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    source_ids = {
        DatasetName.QBIOLIP: ("q-data", "q-infra"),
        DatasetName.AFDB: (),
    }
    calls: dict[str, int] = {}
    generations: list[tuple[str, ...]] = []

    def preflight(dataset, index, sample_id, view):
        calls[sample_id] = calls.get(sample_id, 0) + 1
        if sample_id == "q-data":
            return [
                Finding(
                    code="bad_data",
                    severity=Severity.LATEST_PREFLIGHT,
                    message="bad data",
                )
            ]
        if calls[sample_id] == 1:
            return [
                Finding(
                    code="temporary_worker",
                    severity=Severity.INFRASTRUCTURE,
                    message="temporary",
                )
            ]
        return []

    def loader_factory(dataset):
        generations.append(tuple(item.sample_id for item in dataset.items))
        return _direct_loader(dataset)

    summary = run_screening(
        options,
        source_ids=source_ids,
        historical_seeds={},
        preflight=preflight,
        loader_factory=loader_factory,
        expected_count=2,
    )

    assert generations == [("q-data", "q-infra"), ("q-infra",)]
    assert calls["q-data"] == 7
    assert calls["q-infra"] == 8
    assert summary.complete is False
    assert summary.unresolved_infrastructure_ids == []
    records = [
        SampleRecord.model_validate_json(line)
        for line in (options.output_root / "records.jsonl").read_text().splitlines()
    ]
    assert [(record.sample_id, record.status) for record in records] == [
        ("q-data", SampleStatus.EXCLUDED),
        ("q-infra", SampleStatus.PASS),
    ]


def test_second_infrastructure_failure_stays_incomplete(tmp_path: Path) -> None:
    options = _options(tmp_path)
    source_ids = {DatasetName.QBIOLIP: ("q1",), DatasetName.AFDB: ()}

    def preflight(*_args):
        return [
            Finding(
                code="worker_dead",
                severity=Severity.INFRASTRUCTURE,
                message="dead",
            )
        ]

    summary = run_screening(
        options,
        source_ids=source_ids,
        historical_seeds={},
        preflight=preflight,
        loader_factory=_direct_loader,
        expected_count=1,
    )

    assert summary.complete is False
    assert summary.unresolved_infrastructure_ids == ["qbiolip:q1"]
    assert not (options.output_root / "COMPLETE").exists()


def test_resume_skips_terminal_records_without_duplication(tmp_path: Path) -> None:
    options = _options(tmp_path)
    source_ids = {
        DatasetName.QBIOLIP: ("q1", "q2"),
        DatasetName.AFDB: (),
    }
    calls: list[str] = []

    def preflight(dataset, index, sample_id, view):
        calls.append(sample_id)
        return []

    run_screening(
        options,
        source_ids={DatasetName.QBIOLIP: ("q1",), DatasetName.AFDB: ()},
        historical_seeds={},
        preflight=preflight,
        loader_factory=_direct_loader,
        expected_count=1,
    )
    resume_options = options.model_copy(update={"resume": True})
    run_screening(
        resume_options,
        source_ids=source_ids,
        historical_seeds={},
        preflight=preflight,
        loader_factory=_direct_loader,
        expected_count=2,
    )

    assert calls == ["q1"] * 7 + ["q2"] * 7
    lines = (options.output_root / "records.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["sample_id"] for line in lines] == ["q1", "q2"]


def test_resume_rejects_duplicate_records_and_metadata_mismatch(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    source_ids = {DatasetName.QBIOLIP: ("q1",), DatasetName.AFDB: ()}
    run_screening(
        options,
        source_ids=source_ids,
        historical_seeds={},
        preflight=lambda *_: [],
        loader_factory=_direct_loader,
        expected_count=1,
    )
    records = options.output_root / "records.jsonl"
    records.write_text(records.read_text() + records.read_text())
    resume_options = options.model_copy(update={"resume": True})

    with pytest.raises(ValueError, match="duplicate terminal record"):
        run_screening(
            resume_options,
            source_ids=source_ids,
            historical_seeds={},
            preflight=lambda *_: [],
            loader_factory=_direct_loader,
            expected_count=1,
        )

    records.write_text(records.read_text().splitlines()[0] + "\n")
    metadata = options.output_root / "metadata.json"
    payload = json.loads(metadata.read_text())
    payload["metadata_hash"] = "0" * 64
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        run_screening(
            resume_options,
            source_ids=source_ids,
            historical_seeds={},
            preflight=lambda *_: [],
            loader_factory=_direct_loader,
            expected_count=1,
        )


def test_refuses_existing_output_without_resume(tmp_path: Path) -> None:
    options = _options(tmp_path)
    options.output_root.mkdir()

    with pytest.raises(FileExistsError, match="--resume"):
        run_screening(
            options,
            source_ids={DatasetName.QBIOLIP: (), DatasetName.AFDB: ()},
            historical_seeds={},
            preflight=lambda *_: [],
            loader_factory=_direct_loader,
            expected_count=0,
        )


def test_cached_source_ids_preserve_reader_inventory_order(tmp_path: Path) -> None:
    ordered_ids = [f"q-{index:05d}" for index in range(9_999, -1, -1)]
    cache = tmp_path / "reader-index.pkl"
    with cache.open("wb") as handle:
        pickle.dump(
            {"index": {sample_id: f"/source/{sample_id}" for sample_id in ordered_ids}},
            handle,
        )

    assert load_cached_source_ids(DatasetName.QBIOLIP, cache) == tuple(ordered_ids)
