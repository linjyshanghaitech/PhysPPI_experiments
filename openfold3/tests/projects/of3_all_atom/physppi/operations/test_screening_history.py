import json
from pathlib import Path

from openfold3.projects.of3_all_atom.physppi.screening.history import (
    SIGNATURE_PATTERNS,
    historical_ids_by_dataset,
    load_historical_seeds,
    load_versioned_log_run_summaries,
    load_versioned_log_sample_ids,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    HistoricalSeed,
)

EXPECTED_SIGNATURES = {
    "reference_molecule",
    "entity_id",
    "msa_width",
    "malformed_charge",
    "canonical_mapping",
    "template_broadcast",
    "atom_id_order",
    "label_index",
    "paired_msa_contract",
    "token_center",
    "frame_start",
    "surface_shape",
    "nonfinite_feature",
    "missing_resource",
}


def test_historical_ids_by_dataset_validates_keys_and_sorts() -> None:
    seeds = {
        (DatasetName.QBIOLIP, "q-z"): HistoricalSeed(
            dataset=DatasetName.QBIOLIP,
            sample_id="q-z",
            signatures=["fixture"],
            sources=["fixture"],
        ),
        (DatasetName.QBIOLIP, "q-a"): HistoricalSeed(
            dataset=DatasetName.QBIOLIP,
            sample_id="q-a",
            signatures=["fixture"],
            sources=["fixture"],
        ),
    }

    ids = historical_ids_by_dataset(seeds)

    assert ids == {
        DatasetName.QBIOLIP: ("q-a", "q-z"),
        DatasetName.AFDB: (),
    }


def _write_log(runs_root: Path, version: int, text: str) -> Path:
    run_dir = runs_root / f"stage1-full-v{version}-smoke"
    run_dir.mkdir()
    log_path = run_dir / "train.log"
    log_path.write_text(text)
    return log_path


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_signature_patterns_cover_every_historical_failure_family() -> None:
    assert set(SIGNATURE_PATTERNS) == EXPECTED_SIGNATURES


def test_load_historical_seeds_unions_bounded_exact_id_evidence(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    manifests = tmp_path / "manifests"
    runs.mkdir()
    manifests.mkdir()
    v1_log = _write_log(
        runs,
        1,
        "ScatterGatherKernel index out of bounds\n"
        "Train step failed with pdb id q-label with preferred chain\n",
    )
    _write_log(
        runs,
        4,
        "physppi_paired_msa_chain_mask must have shape [B, N_chains]\n"
        "Train step failed with pdb id q-msa with preferred chain\n"
        "ScatterGatherKernel index out of bounds\n"
        "Train step failed with pdb id duplicate with preferred chain\n",
    )
    _write_log(
        runs,
        7,
        "invalid-frame-start\n"
        "Train step failed with pdb id q-frame with preferred chain\n",
    )
    v31_log = _write_log(
        runs,
        31,
        "physppi_surface_local_coords must have trailing shape [N, K, 3]\n"
        "Train step failed with pdb id a-surface with preferred chain\n"
        "physppi_surface_local_coords must have trailing shape [N, K, 3]\n"
        "Train step failed with pdb id AF-0000000066488931-model_v1 with preferred chain\n",
    )
    _write_log(
        runs,
        33,
        "ScatterGatherKernel index out of bounds\n"
        "Train step failed with pdb id ignored-v33 with preferred chain\n",
    )
    manifest_path = _write_json(
        manifests / "formal-manifest-v7.json",
        {
            "runtime_skips": [
                {
                    "dataset": "qbiolip",
                    "sample_id": "duplicate",
                    "reason": "ScatterGatherKernel index out of bounds",
                },
                {
                    "dataset": "qbiolip",
                    "sample_id": "q-manifest",
                    "reason": "No such file or directory",
                },
            ]
        },
    )
    ledger = _write_json(
        tmp_path / "runtime-failure-ledger.jsonl",
        {
            "dataset": "qbiolip",
            "sample_id": "duplicate",
            "reason": "ScatterGatherKernel index out of bounds",
        },
    )
    ledger.write_text(
        ledger.read_text()
        + "\n"
        + json.dumps(
            {
                "dataset": "qbiolip",
                "sample_id": "q-ledger",
                "reason": "invalid-frame-start",
            }
        )
        + "\n"
    )
    prior_audit = _write_json(
        tmp_path / "data-error-ids.json",
        {
            "categories": {
                "canonical_residue_mapping_fatal": {
                    "qbiolip_ids": ["q-prior"],
                    "afdb_ids": [],
                },
                "malformed_gt_pdb_charge": {
                    "qbiolip_ids": ["q-charge"],
                    "afdb_ids": [],
                },
                "nonincreasing_template_atom_ids": {
                    "qbiolip_ids": ["q-atoms"],
                    "afdb_ids": [],
                },
            }
        },
    )

    seeds = load_historical_seeds(runs, manifests, ledger, prior_audit)

    assert seeds[(DatasetName.QBIOLIP, "q-label")].signatures == ["label_index"]
    assert seeds[(DatasetName.QBIOLIP, "q-msa")].signatures == ["paired_msa_contract"]
    assert seeds[(DatasetName.AFDB, "a-surface")].signatures == ["surface_shape"]
    assert seeds[(DatasetName.AFDB, "AF-0000000066488931-model_v1")].signatures == [
        "surface_shape"
    ]
    assert seeds[(DatasetName.QBIOLIP, "q-prior")].signatures == ["canonical_mapping"]
    assert seeds[(DatasetName.QBIOLIP, "q-charge")].signatures == ["malformed_charge"]
    assert seeds[(DatasetName.QBIOLIP, "q-atoms")].signatures == ["atom_id_order"]
    assert (DatasetName.QBIOLIP, "ignored-v33") not in seeds
    assert seeds[(DatasetName.QBIOLIP, "duplicate")].sources == sorted(
        seeds[(DatasetName.QBIOLIP, "duplicate")].sources
    )
    assert str(v1_log) in seeds[(DatasetName.QBIOLIP, "q-label")].sources
    assert str(v31_log) in seeds[(DatasetName.AFDB, "a-surface")].sources
    assert str(manifest_path) in seeds[(DatasetName.QBIOLIP, "duplicate")].sources


def test_versioned_log_run_summaries_keep_no_id_error_evidence(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    log_path = _write_log(runs, 5, "failed to parse MSE reference molecule\n")
    _write_log(runs, 33, "failed to parse MSE reference molecule\n")

    assert load_versioned_log_run_summaries(runs) == [
        {
            "run": "stage1-full-v5-smoke",
            "version": 5,
            "signatures": ["reference_molecule"],
            "source": str(log_path),
        }
    ]


def test_historical_seeds_union_all_bounded_manifest_evidence(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    manifests = tmp_path / "manifests"
    runs.mkdir()
    manifests.mkdir()
    first_manifest = _write_json(
        manifests / "formal-manifest-v1.json",
        {
            "runtime_skips": [
                {
                    "dataset": "qbiolip",
                    "sample_id": "duplicate",
                    "reason": "ScatterGatherKernel index out of bounds",
                },
                {
                    "dataset": "qbiolip",
                    "sample_id": "q-nan-in-id",
                    "reason": "ordinary validated condition",
                    "metadata": {"comment": "nan is only metadata"},
                },
            ]
        },
    )
    second_manifest = _write_json(
        manifests / "formal-manifest-v7.json",
        {
            "runtime_skips": [
                {
                    "dataset": "qbiolip",
                    "sample_id": "duplicate",
                    "exception_message": "invalid-frame-start",
                },
            ]
        },
    )
    _write_json(
        manifests / "formal-manifest-v33.json",
        {
            "runtime_skips": [
                {
                    "dataset": "qbiolip",
                    "sample_id": "duplicate",
                    "message": "token center atom index is out of bounds",
                },
                {
                    "dataset": "qbiolip",
                    "sample_id": "future-manifest-id",
                    "reason": "ScatterGatherKernel index out of bounds",
                },
            ]
        },
    )
    ledger = _write_json(tmp_path / "runtime-failure-ledger.jsonl", {})
    audit = _write_json(tmp_path / "audit.json", {"categories": {}})

    seeds = load_historical_seeds(runs, manifests, ledger, audit)

    duplicate = seeds[(DatasetName.QBIOLIP, "duplicate")]
    assert duplicate.signatures == ["frame_start", "label_index"]
    assert duplicate.sources == [str(first_manifest), str(second_manifest)]
    assert seeds[(DatasetName.QBIOLIP, "q-nan-in-id")].signatures == []
    assert (DatasetName.QBIOLIP, "future-manifest-id") not in seeds


def test_versioned_log_evidence_is_bounded_to_one_error_block(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    filler = "\n".join("normal progress" for _ in range(33))
    _write_log(
        runs,
        6,
        "Train step failed with pdb id q-traceback with preferred chain\n"
        "Traceback (most recent call last):\n"
        "token center atom index is out of bounds\n"
        f"{filler}\n"
        "Train step failed with pdb id q-unrelated with preferred chain\n",
    )

    assert load_versioned_log_sample_ids(runs) == [
        {
            "sample_id": "q-traceback",
            "run": "stage1-full-v6-smoke",
            "version": 6,
            "signature": "token_center",
        }
    ]


def test_versioned_log_evidence_rejects_stale_prior_id(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    filler = "\n".join("normal progress" for _ in range(33))
    _write_log(
        runs,
        6,
        "Train step failed with pdb id q-stale with preferred chain\n"
        f"{filler}\n"
        "token center atom index is out of bounds\n",
    )

    assert load_versioned_log_sample_ids(runs) == []


def test_versioned_log_evidence_unions_signatures_preceding_one_id(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_log(
        runs,
        6,
        "physppi_paired_msa_chain_mask must have shape [B, N_chains]\n"
        "ScatterGatherKernel index out of bounds\n"
        "Train step failed with pdb id q-multiple with preferred chain\n",
    )

    assert load_versioned_log_sample_ids(runs) == [
        {
            "sample_id": "q-multiple",
            "run": "stage1-full-v6-smoke",
            "version": 6,
            "signature": "label_index",
        },
        {
            "sample_id": "q-multiple",
            "run": "stage1-full-v6-smoke",
            "version": 6,
            "signature": "paired_msa_contract",
        },
    ]


def test_versioned_log_evidence_consumes_preceding_signature_after_one_id(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_log(
        runs,
        6,
        "ScatterGatherKernel index out of bounds\n"
        "Train step failed with pdb id q-intended with preferred chain\n"
        "Train step failed with pdb id q-unrelated with preferred chain\n",
    )
    assert load_versioned_log_sample_ids(runs) == [
        {
            "sample_id": "q-intended",
            "run": "stage1-full-v6-smoke",
            "version": 6,
            "signature": "label_index",
        }
    ]


def test_manifest_failure_fields_exercise_every_historical_signature_family(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    manifests = tmp_path / "manifests"
    runs.mkdir()
    manifests.mkdir()
    messages = {
        "reference_molecule": "failed to parse MSE reference molecule",
        "entity_id": "missing entity_id",
        "msa_width": "msa width mismatch",
        "malformed_charge": "invalid literal for int() with base 10",
        "canonical_mapping": "canonical residue mapping failed",
        "template_broadcast": "template could not be broadcast",
        "atom_id_order": "atom ids are not strictly increasing",
        "label_index": "ScatterGatherKernel index out of bounds",
        "paired_msa_contract": "physppi_paired_msa_chain_mask must have shape",
        "token_center": "token center atom index is out of bounds",
        "frame_start": "invalid-frame-start",
        "surface_shape": "physppi_surface_local_coords must have trailing shape",
        "nonfinite_feature": "non-finite feature value",
        "missing_resource": "No such file or directory",
    }
    _write_json(
        manifests / "formal-manifest-v8.json",
        {
            "runtime_skips": [
                {
                    "dataset": "qbiolip",
                    "sample_id": f"q-{signature}",
                    "failure_category": message,
                }
                for signature, message in messages.items()
            ]
        },
    )
    ledger = _write_json(tmp_path / "runtime-failure-ledger.jsonl", {})
    audit = _write_json(tmp_path / "audit.json", {"categories": {}})

    seeds = load_historical_seeds(runs, manifests, ledger, audit)

    assert {
        seed.signatures[0]
        for (_, sample_id), seed in seeds.items()
        if sample_id.startswith("q-")
    } == EXPECTED_SIGNATURES
