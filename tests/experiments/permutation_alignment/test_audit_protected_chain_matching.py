import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from experiments.permutation_alignment import audit_protected_chain_matching as audit
from openfold3.projects.of3_all_atom.physppi.screening.models import DatasetName


def _write_ledger(path: Path, *records: str) -> None:
    path.write_text("\n".join(records) + "\n")


def test_run_protected_match_audit_writes_one_terminal_row_per_id(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "runtime.jsonl"
    _write_ledger(
        ledger,
        '{"dataset":"qbiolip","sample_id":"q-1"}',
        '{"dataset":"afdb","sample_id":"AF-2"}',
    )
    output = tmp_path / "audit.jsonl"

    records = audit.run_protected_match_audit(
        ledger,
        output,
        lambda dataset, sample_id: {
            "accepted": False,
            "reason": f"fixture:{dataset.value}:{sample_id}",
            "identity_rmsd": 2.0,
            "candidate_rmsd": 1.5,
            "matched_coverage": 4,
        },
    )

    assert [(record.dataset, record.sample_id) for record in records] == [
        (DatasetName.QBIOLIP, "q-1"),
        (DatasetName.AFDB, "AF-2"),
    ]
    assert [record.status for record in records] == [
        "identity_fallback",
        "identity_fallback",
    ]
    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(written) == 2
    assert [row["reason"] for row in written] == [
        "fixture:qbiolip:q-1",
        "fixture:afdb:AF-2",
    ]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            '{"dataset":"qbiolip","sample_id":"q-1"}\n'
            '{"dataset":"qbiolip","sample_id":"q-1"}\n',
            "duplicate runtime ID",
        ),
        ('{"dataset":\n', "invalid runtime ledger record"),
        ('{"dataset":"afdb","sample_id":""}\n', "empty sample ID"),
    ],
)
def test_load_runtime_ids_rejects_invalid_records(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    ledger = tmp_path / "runtime.jsonl"
    ledger.write_text(contents)

    with pytest.raises(ValueError, match=message):
        audit.load_runtime_ids(ledger)


def test_run_protected_match_audit_rejects_existing_output(tmp_path: Path) -> None:
    ledger = tmp_path / "runtime.jsonl"
    _write_ledger(ledger, '{"dataset":"qbiolip","sample_id":"q-1"}')
    output = tmp_path / "audit.jsonl"
    output.write_text("keep\n")

    with pytest.raises(FileExistsError, match="audit output already exists"):
        audit.run_protected_match_audit(
            ledger,
            output,
            lambda _dataset, _sample_id: pytest.fail("evaluator must not run"),
        )

    assert output.read_text() == "keep\n"


@pytest.mark.parametrize(
    ("identity_rmsd", "candidate_rmsd"),
    [(float("inf"), 1.0), (1.0, float("nan"))],
)
def test_run_protected_match_audit_rejects_nonfinite_rmsd_without_output(
    tmp_path: Path,
    identity_rmsd: float,
    candidate_rmsd: float,
) -> None:
    ledger = tmp_path / "runtime.jsonl"
    _write_ledger(ledger, '{"dataset":"afdb","sample_id":"AF-2"}')
    output = tmp_path / "audit.jsonl"

    with pytest.raises(ValueError, match="RMSD diagnostics must be finite"):
        audit.run_protected_match_audit(
            ledger,
            output,
            lambda _dataset, _sample_id: {
                "accepted": False,
                "reason": "fixture",
                "identity_rmsd": identity_rmsd,
                "candidate_rmsd": candidate_rmsd,
                "matched_coverage": 4,
            },
        )

    assert not output.exists()


def test_production_paths_derive_all_formal_inputs(tmp_path: Path) -> None:
    paths = audit.ProductionPaths(
        qbiolip_root=tmp_path / "qbiolip",
        afdb_root=tmp_path / "afdb",
        formal_root=tmp_path / "formal",
        components_cif=tmp_path / "components.cif",
        reference_molecules=tmp_path / "reference",
    )
    work = tmp_path / "work"

    options = paths.screening_options(work)

    assert options.qbiolip_root == paths.qbiolip_root
    assert options.afdb_root == paths.afdb_root
    assert options.qbiolip_index_cache == (
        paths.formal_root / "reader-index-cache/qbiolip10k-v1.pkl"
    )
    assert options.afdb_index_cache == (
        paths.formal_root / "reader-index-cache/afdb10k-v1.pkl"
    )
    assert options.qbiolip_canonical_report == (
        paths.formal_root / "manifests/qbiolip-audit.json"
    )
    assert options.afdb_canonical_report == (
        paths.formal_root / "manifests/afdb-audit.json"
    )
    assert options.history_runs_root == paths.formal_root
    assert options.manifests_dir == paths.formal_root / "manifests"
    assert options.runtime_ledger == (
        paths.formal_root / "manifests/runtime-failure-ledger.jsonl"
    )
    assert options.prior_audit == (
        paths.formal_root / "audits/v1-v32-full-20k-data-errors/data-error-ids.json"
    )
    assert options.components_cif == paths.components_cif
    assert options.reference_molecule_directory == paths.reference_molecules
    assert options.output_root == work
    assert options.workers == 1
    with pytest.raises(FrozenInstanceError):
        paths.formal_root = tmp_path / "other"


def test_production_dataset_cache_isolated_by_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = audit.ProductionPaths(
        qbiolip_root=tmp_path / "qbiolip-1",
        afdb_root=tmp_path / "afdb-1",
        formal_root=tmp_path / "formal-1",
        components_cif=tmp_path / "components-1.cif",
        reference_molecules=tmp_path / "reference-1",
    )
    second = audit.ProductionPaths(
        qbiolip_root=tmp_path / "qbiolip-2",
        afdb_root=tmp_path / "afdb-2",
        formal_root=tmp_path / "formal-2",
        components_cif=tmp_path / "components-2.cif",
        reference_molecules=tmp_path / "reference-2",
    )
    built: list[tuple[DatasetName, object, Path]] = []

    def fake_build(dataset: DatasetName, options: object, cache: Path) -> object:
        source = object()
        built.append((dataset, options, cache))
        return source

    monkeypatch.setattr(audit, "build_preflight_dataset", fake_build)
    audit._PRODUCTION_DATASETS.clear()
    audit._PRODUCTION_WORK_ROOTS.clear()
    try:
        first_source = audit._dataset_for(first, DatasetName.QBIOLIP)
        assert audit._dataset_for(first, DatasetName.QBIOLIP) is first_source
        second_source = audit._dataset_for(second, DatasetName.QBIOLIP)

        assert second_source is not first_source
        assert len(built) == 2
        assert built[0][1].qbiolip_root == first.qbiolip_root
        assert built[1][1].qbiolip_root == second.qbiolip_root
        assert built[0][2].parents[2] != built[1][2].parents[2]
    finally:
        for temporary in audit._PRODUCTION_WORK_ROOTS.values():
            temporary.cleanup()
        audit._PRODUCTION_WORK_ROOTS.clear()
        audit._PRODUCTION_DATASETS.clear()


def test_cleanup_production_cache_releases_only_target_before_temporary_root(
    tmp_path: Path,
) -> None:
    first = audit.ProductionPaths(
        qbiolip_root=tmp_path / "qbiolip-1",
        afdb_root=tmp_path / "afdb-1",
        formal_root=tmp_path / "formal-1",
        components_cif=tmp_path / "components-1.cif",
        reference_molecules=tmp_path / "reference-1",
    )
    second = audit.ProductionPaths(
        qbiolip_root=tmp_path / "qbiolip-2",
        afdb_root=tmp_path / "afdb-2",
        formal_root=tmp_path / "formal-2",
        components_cif=tmp_path / "components-2.cif",
        reference_molecules=tmp_path / "reference-2",
    )
    cleanup_events: list[tuple[str, audit.ProductionPaths]] = []

    class TemporaryRoot:
        def __init__(self, paths: audit.ProductionPaths) -> None:
            self.paths = paths

        def cleanup(self) -> None:
            assert all(
                cached_paths != self.paths
                for cached_paths, _dataset in audit._PRODUCTION_DATASETS
            )
            assert self.paths not in audit._PRODUCTION_WORK_ROOTS
            cleanup_events.append(("cleanup", self.paths))

    audit._PRODUCTION_DATASETS.clear()
    audit._PRODUCTION_WORK_ROOTS.clear()
    audit._PRODUCTION_DATASETS[(first, DatasetName.QBIOLIP)] = object()
    audit._PRODUCTION_DATASETS[(first, DatasetName.AFDB)] = object()
    retained_dataset = object()
    audit._PRODUCTION_DATASETS[(second, DatasetName.QBIOLIP)] = retained_dataset
    audit._PRODUCTION_WORK_ROOTS[first] = TemporaryRoot(first)
    retained_root = TemporaryRoot(second)
    audit._PRODUCTION_WORK_ROOTS[second] = retained_root
    try:
        audit.cleanup_production_cache(first)
        audit.cleanup_production_cache(first)

        assert cleanup_events == [("cleanup", first)]
        assert {
            (second, DatasetName.QBIOLIP): retained_dataset
        } == audit._PRODUCTION_DATASETS
        assert {second: retained_root} == audit._PRODUCTION_WORK_ROOTS
    finally:
        audit._PRODUCTION_DATASETS.clear()
        audit._PRODUCTION_WORK_ROOTS.clear()


def test_main_cleans_production_cache_when_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls = 0

    class TemporaryRoot:
        def cleanup(self) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

    def fail_audit(
        _runtime_ledger: Path,
        _output: Path,
        evaluator: audit.MatchEvaluator,
    ) -> tuple[audit.ProtectedMatchAuditRecord, ...]:
        paths = evaluator.args[0]
        audit._PRODUCTION_DATASETS[(paths, DatasetName.QBIOLIP)] = object()
        audit._PRODUCTION_WORK_ROOTS[paths] = TemporaryRoot()
        raise RuntimeError("fixture audit failure")

    monkeypatch.setattr(audit, "run_protected_match_audit", fail_audit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit-protected",
            "--runtime-ledger",
            str(tmp_path / "runtime.jsonl"),
            "--output",
            str(tmp_path / "audit.jsonl"),
            "--qbiolip-root",
            str(tmp_path / "qbiolip"),
            "--afdb-root",
            str(tmp_path / "afdb"),
            "--formal-root",
            str(tmp_path / "formal"),
            "--components-cif",
            str(tmp_path / "components.cif"),
        ],
    )
    audit._PRODUCTION_DATASETS.clear()
    audit._PRODUCTION_WORK_ROOTS.clear()
    try:
        with pytest.raises(RuntimeError, match="fixture audit failure"):
            audit.main()

        assert cleanup_calls == 1
        assert audit._PRODUCTION_DATASETS == {}
        assert audit._PRODUCTION_WORK_ROOTS == {}
    finally:
        audit._PRODUCTION_DATASETS.clear()
        audit._PRODUCTION_WORK_ROOTS.clear()


def test_main_with_custom_evaluator_does_not_clean_production_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = audit.ProductionPaths(
        qbiolip_root=tmp_path / "qbiolip",
        afdb_root=tmp_path / "afdb",
        formal_root=tmp_path / "formal",
        components_cif=tmp_path / "components.cif",
        reference_molecules=tmp_path / "reference",
    )

    class TemporaryRoot:
        cleanup_calls = 0

        def cleanup(self) -> None:
            self.cleanup_calls += 1

    retained_dataset = object()
    retained_root = TemporaryRoot()
    audit._PRODUCTION_DATASETS.clear()
    audit._PRODUCTION_WORK_ROOTS.clear()
    audit._PRODUCTION_DATASETS[(paths, DatasetName.QBIOLIP)] = retained_dataset
    audit._PRODUCTION_WORK_ROOTS[paths] = retained_root
    monkeypatch.setattr(audit, "_load_evaluator", lambda _specification: object())
    monkeypatch.setattr(
        audit,
        "run_protected_match_audit",
        lambda _runtime_ledger, _output, _evaluator: (),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit-protected",
            "--runtime-ledger",
            str(tmp_path / "runtime.jsonl"),
            "--output",
            str(tmp_path / "audit.jsonl"),
            "--evaluator",
            "fixture:evaluate",
        ],
    )
    try:
        audit.main()

        assert {
            (paths, DatasetName.QBIOLIP): retained_dataset
        } == audit._PRODUCTION_DATASETS
        assert {paths: retained_root} == audit._PRODUCTION_WORK_ROOTS
        assert retained_root.cleanup_calls == 0
    finally:
        audit._PRODUCTION_DATASETS.clear()
        audit._PRODUCTION_WORK_ROOTS.clear()


def test_custom_evaluator_does_not_require_production_paths() -> None:
    parser = audit.build_parser()
    args = parser.parse_args(
        [
            "--runtime-ledger",
            "runtime.jsonl",
            "--output",
            "audit.jsonl",
            "--evaluator",
            "fixture:evaluate",
        ]
    )

    assert audit._production_paths_from_args(parser, args) is None


@pytest.mark.parametrize(
    "missing",
    [
        "--qbiolip-root",
        "--afdb-root",
        "--formal-root",
        "--components-cif",
    ],
)
def test_default_evaluator_requires_each_production_path(
    missing: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = audit.build_parser()
    values = {
        "--runtime-ledger": "runtime.jsonl",
        "--output": "audit.jsonl",
        "--qbiolip-root": "qbiolip",
        "--afdb-root": "afdb",
        "--formal-root": "formal",
        "--components-cif": "components.cif",
    }
    argv = [
        item
        for option, value in values.items()
        if option != missing
        for item in (option, value)
    ]
    args = parser.parse_args(argv)

    with pytest.raises(SystemExit, match="2"):
        audit._production_paths_from_args(parser, args)

    assert (
        f"{missing} is required when --evaluator is not set" in capsys.readouterr().err
    )


def test_parser_defaults_reference_molecules_from_repository_root() -> None:
    args = audit.build_parser().parse_args(
        [
            "--runtime-ledger",
            "runtime.jsonl",
            "--output",
            "audit.jsonl",
            "--evaluator",
            "fixture:evaluate",
        ]
    )

    assert args.reference_molecules == (
        Path(__file__).resolve().parents[3]
        / "openfold3/core/data/resources/canonical_protein_data/reference_mols"
    )
