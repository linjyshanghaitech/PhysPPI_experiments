from __future__ import annotations

import copy
import os
import random
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.projects.of3_all_atom.physppi.data.adapters import (
    PhysPPIPriorComplexDatasetAdapter,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    OptionalSlot,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    QBioLiP10kReader,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    ScreeningOptions,
    Severity,
)
from openfold3.projects.of3_all_atom.physppi.screening.preflight import (
    build_preflight_dataset,
    initialize_worker,
    preflight_audit_view,
    preflight_sample,
    validate_physppi_features,
)
from openfold3.projects.of3_all_atom.physppi.screening.views import audit_views
from openfold3.tests.projects.of3_all_atom.physppi.data.test_afdb10k_reader import (
    _write_afdb_case,
)

SAMPLE_ID = "AF-0000000000000001-model_v1"
CROP_STRATEGIES = ("spatial", "spatial_interface")


def _feature_ready_pdb() -> str:
    atoms = {
        "ALA": ("N", "CA", "C", "O", "CB"),
        "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    }
    lines: list[str] = []
    serial = 1
    residue_offsets = (
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 3.0),
    )
    for chain_index, chain_id in enumerate(("A", "B")):
        for residue_index, residue in enumerate(("ALA", "CYS", "ALA", "CYS"), start=1):
            offset_x, offset_y, offset_z = residue_offsets[residue_index - 1]
            for atom_index, atom_name in enumerate(atoms[residue]):
                x = 10.0 * chain_index + offset_x + atom_index * 0.2
                y = offset_y + atom_index * 0.1
                z = offset_z + atom_index * 0.15
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} {residue:>3s} {chain_id}"
                    f"{residue_index:4d}    {x:8.3f}{y:8.3f}"
                    f"{z:8.3f}{1.0:6.2f}{20.0:6.2f}          "
                    f"{atom_name[0]:>2s}\n"
                )
                serial += 1
        lines.append("TER\n")
    return "".join(lines) + "END\n"


def _make_feature_ready(root: Path) -> None:
    chunk = root / "chunk" / "chunk0001"
    pdb = _feature_ready_pdb()
    (chunk / "gt" / f"{SAMPLE_ID}.pdb").write_text(pdb)
    for chain_id in ("A", "B"):
        chain_lines = [
            line
            for line in pdb.splitlines()
            if line.startswith("ATOM") and line[21] == chain_id
        ]
        (chunk / "bb_input" / f"{SAMPLE_ID}_{chain_id}.pdb").write_text(
            "\n".join(chain_lines) + "\nEND\n"
        )


def _valid_features() -> dict:
    n_tokens = 2
    n_atoms = 4
    n_chains = 2
    return {
        "token_mask": torch.ones(n_tokens, dtype=torch.bool),
        "atom_mask": torch.ones(n_atoms, dtype=torch.bool),
        "atom_to_token_index": torch.tensor([0, 0, 1, 1]),
        "start_atom_index": torch.tensor([0, 2]),
        "entity_id": torch.tensor([1, 2]),
        "asym_id": torch.tensor([1, 2]),
        "msa": torch.zeros((1, n_tokens, 32)),
        "msa_mask": torch.ones((1, n_tokens)),
        "has_deletion": torch.zeros((1, n_tokens)),
        "deletion_value": torch.zeros((1, n_tokens)),
        "deletion_mean": torch.zeros(n_tokens),
        "profile": torch.zeros((n_tokens, 32)),
        "physppi_paired_msa_chain_mask": torch.ones(n_chains, dtype=torch.bool),
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
        **{
            f"physppi_{slot}_{state}": torch.tensor(value)
            for slot in ("contact", "interaction", "surface", "tm")
            for state, value in (
                ("available", True),
                ("selected", True),
                ("valid", True),
                ("provenance", True),
                ("active", True),
            )
        },
    }


def _mutate_missing_entity(features: dict) -> None:
    features.pop("entity_id")


def _mutate_msa_width(features: dict) -> None:
    features["msa"] = torch.zeros((1, 3, 32))


def _mutate_paired_msa(features: dict) -> None:
    features["physppi_paired_msa_chain_mask"] = torch.ones(1, dtype=torch.bool)


def _mutate_canonical_mapping(features: dict) -> None:
    features["ref_space_uid_to_perm"][0] = torch.tensor([[0, 2]])


def _mutate_template_alignment(features: dict) -> None:
    features["template_restype"] = torch.zeros((1, 3, 32))


def _mutate_label_index(features: dict) -> None:
    features["atom_to_token_index"][-1] = 2


def _mutate_token_center(features: dict) -> None:
    features["physppi_token_center_atom_index"][-1] = 4


def _mutate_frame_atom(features: dict) -> None:
    features["physppi_backbone_atom_indices"][0, 0] = 4


def _mutate_surface(features: dict) -> None:
    features["physppi_surface_local_coords"] = torch.zeros((2, 127, 3))


def _mutate_nonfinite(features: dict) -> None:
    features["profile"][0, 0] = torch.nan


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        pytest.param(_mutate_missing_entity, "missing_entity_id", id="entity"),
        pytest.param(_mutate_msa_width, "msa_width_mismatch", id="msa"),
        pytest.param(_mutate_paired_msa, "paired_msa_shape", id="paired-msa"),
        pytest.param(
            _mutate_canonical_mapping,
            "canonical_mapping_invalid",
            id="canonical-mapping",
        ),
        pytest.param(
            _mutate_template_alignment,
            "template_alignment_mismatch",
            id="template",
        ),
        pytest.param(_mutate_label_index, "label_index_out_of_bounds", id="label"),
        pytest.param(
            _mutate_token_center,
            "token_center_out_of_bounds",
            id="token-center",
        ),
        pytest.param(_mutate_frame_atom, "frame_atom_out_of_bounds", id="frame"),
        pytest.param(_mutate_surface, "surface_shape", id="surface"),
        pytest.param(_mutate_nonfinite, "nonfinite_feature", id="nonfinite"),
    ],
)
def test_feature_contract_reports_one_exact_code(mutation, expected_code) -> None:
    features = _valid_features()
    mutation(features)

    findings = validate_physppi_features(features)

    assert {finding.code for finding in findings} == {expected_code}
    assert {finding.severity for finding in findings} == {Severity.LATEST_PREFLIGHT}


def test_paired_msa_ignores_masked_asym_id_padding() -> None:
    features = _valid_features()
    n_tokens = 4
    features.update(
        {
            "token_mask": torch.tensor([True, True, False, False]),
            "entity_id": torch.tensor([1, 2, 0, 0]),
            "asym_id": torch.tensor([1, 2, 0, 0]),
            "msa": torch.zeros((1, n_tokens, 32)),
            "msa_mask": torch.ones((1, n_tokens)),
            "has_deletion": torch.zeros((1, n_tokens)),
            "deletion_value": torch.zeros((1, n_tokens)),
            "deletion_mean": torch.zeros(n_tokens),
            "profile": torch.zeros((n_tokens, 32)),
            "template_restype": torch.zeros((1, n_tokens, 32)),
            "template_pseudo_beta_mask": torch.ones((1, n_tokens)),
            "template_backbone_frame_mask": torch.ones((1, n_tokens)),
            "template_distogram": torch.zeros((1, n_tokens, n_tokens, 39)),
            "template_unit_vector": torch.zeros((1, n_tokens, n_tokens, 3)),
            "physppi_token_center_atom_index": torch.tensor([1, 3, -1, -1]),
            "physppi_backbone_atom_indices": torch.tensor(
                [[0, 1, 0], [2, 3, 2], [-1, -1, -1], [-1, -1, -1]]
            ),
            "physppi_backbone_frame_mask": torch.tensor([True, True, False, False]),
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
            "physppi_surface_charge_mask": torch.zeros(
                (n_tokens, 128), dtype=torch.bool
            ),
        }
    )
    assert validate_physppi_features(features) == []


def test_valid_feature_contract_has_no_findings() -> None:
    assert validate_physppi_features(_valid_features()) == []


def test_legacy_contact_and_surface_interface_fields_are_rejected() -> None:
    features = _valid_features()
    features["physppi_contact_token_mask"] = torch.zeros(2, dtype=torch.bool)
    features["physppi_surface_interface_mask"] = torch.zeros((2, 128), dtype=torch.bool)

    assert {finding.code for finding in validate_physppi_features(features)} == {
        "legacy_prior_field"
    }


def test_pair_interaction_auxiliary_must_share_token_pair_axes() -> None:
    features = _valid_features()
    features["physppi_interaction_aux_valid"] = torch.zeros((2,), dtype=torch.bool)

    assert {finding.code for finding in validate_physppi_features(features)} == {
        "interaction_shape"
    }


def test_prior_levels_are_cumulative_and_match_active_prefix() -> None:
    features = _valid_features()
    features["physppi_prior_effective_level"] = torch.tensor(2)
    features["physppi_surface_active"] = torch.tensor(True)
    features["physppi_tm_active"] = torch.tensor(False)

    assert {finding.code for finding in validate_physppi_features(features)} == {
        "prior_level_contract"
    }


def test_prior_levels_reject_fractional_scalars() -> None:
    features = _valid_features()
    features["physppi_prior_requested_level"] = torch.tensor(4.5)

    assert {finding.code for finding in validate_physppi_features(features)} == {
        "prior_level_contract"
    }


def test_surface_charge_mask_is_part_of_the_surface_shape_contract() -> None:
    features = _valid_features()
    features.pop("physppi_surface_charge_mask")

    assert {finding.code for finding in validate_physppi_features(features)} == {
        "surface_shape"
    }


def test_retained_view_and_matcher_diagnostics_are_validated() -> None:
    features = _valid_features()
    retained_view = {
        "parent_sample_id": "parent",
        "original_chain_ids": ("A", "B", "C", "D", "E"),
        "source_chain_ids": ("A", "C"),
        "component_chain_ids": ("A", "B", "C"),
        "seed_edge": ("A", "C"),
        "trigger": "token_budget",
        "target_chain_count": 2,
        "dmap_cutoff": 18.0,
        "occurrence_seed": 7,
        "matcher_result": "pending",
    }
    matcher = {
        "accepted": True,
        "fallback": "none",
        "reason": "accepted",
        "identity_rmsd": 1.0,
        "candidate_rmsd": 0.5,
        "matched_coverage": 8,
    }

    assert (
        validate_physppi_features(
            features,
            retained_view=retained_view,
            matcher_diagnostics=matcher,
        )
        == []
    )

    retained_view["source_chain_ids"] = ("A",)
    retained_view["target_chain_count"] = 1
    assert {
        finding.code
        for finding in validate_physppi_features(
            features,
            retained_view=retained_view,
            matcher_diagnostics=matcher,
        )
    } == {"retained_view_provenance"}
    retained_view["source_chain_ids"] = ("A", "C")
    retained_view["target_chain_count"] = 2

    matcher["accepted"] = False
    matcher["fallback"] = "none"
    matcher["reason"] = "no_legal_supervised_correspondence"
    assert {
        finding.code
        for finding in validate_physppi_features(
            features,
            retained_view=retained_view,
            matcher_diagnostics=matcher,
        )
    } == {"chain_compatibility"}


def _touch(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _screening_options(tmp_path: Path) -> ScreeningOptions:
    chain_sequences = {
        "A": ["ALA", "CYS", "ALA", "CYS"],
        "B": ["ALA", "CYS", "ALA", "CYS"],
    }
    qbiolip_root = _write_afdb_case(
        tmp_path / "qbiolip",
        chain_sequences=chain_sequences,
        main_sequence="ACAC",
    )
    _make_feature_ready(qbiolip_root)
    qbiolip_main = qbiolip_root / "chunk" / "chunk0001" / "msa_local" / "main"
    source_msa = qbiolip_main / SAMPLE_ID / f"{SAMPLE_ID}.npz"
    for chain_id in ("A", "B"):
        shutil.copy2(
            source_msa, qbiolip_main / f"{SAMPLE_ID}_{SAMPLE_ID}_chain={chain_id}.npz"
        )
    shutil.rmtree(source_msa.parent)
    afdb_root = _write_afdb_case(
        tmp_path / "afdb",
        chain_sequences=chain_sequences,
        main_sequence="ACAC",
    )
    _make_feature_ready(afdb_root)
    qbiolip_index = tmp_path / "shared-cache" / "qbiolip.pkl"
    afdb_index = tmp_path / "shared-cache" / "afdb.pkl"
    QBioLiP10kReader(qbiolip_root, index_cache_file=qbiolip_index)
    AFDB10kReader(afdb_root, index_cache_file=afdb_index)
    history_root = tmp_path / "history"
    manifests_dir = tmp_path / "manifests"
    history_root.mkdir()
    manifests_dir.mkdir()
    output_root = tmp_path / "audit"
    output_root.mkdir()
    return ScreeningOptions(
        qbiolip_root=qbiolip_root,
        afdb_root=afdb_root,
        qbiolip_index_cache=qbiolip_index,
        afdb_index_cache=afdb_index,
        qbiolip_canonical_report=_touch(tmp_path / "qbiolip-report.json"),
        afdb_canonical_report=_touch(tmp_path / "afdb-report.json"),
        history_runs_root=history_root,
        manifests_dir=manifests_dir,
        runtime_ledger=_touch(tmp_path / "runtime-ledger.jsonl", ""),
        prior_audit=_touch(tmp_path / "prior-audit.json"),
        components_cif=_touch(tmp_path / "components.cif", "data_components\n#\n"),
        reference_molecule_directory=Path(
            "openfold3/core/data/resources/canonical_protein_data/reference_mols"
        ).resolve(),
        output_root=output_root,
        workers=1,
    )


def _worker_cache_dir(
    options: ScreeningOptions, dataset: DatasetName, crop_strategy: str
) -> Path:
    return (
        options.output_root
        / "worker-cache"
        / str(os.getpid())
        / f"{dataset.value}-{crop_strategy}"
    )


@pytest.mark.parametrize("dataset_name", list(DatasetName))
@pytest.mark.parametrize("crop_strategy", CROP_STRATEGIES)
def test_preflight_config_is_private_manifest_free_and_forces_one_crop(
    tmp_path: Path,
    dataset_name: DatasetName,
    crop_strategy: str,
) -> None:
    options = _screening_options(tmp_path)
    source_index = (
        options.qbiolip_index_cache
        if dataset_name is DatasetName.QBIOLIP
        else options.afdb_index_cache
    )
    source_before = source_index.read_bytes()
    source_mtime = source_index.stat().st_mtime_ns
    lock_path = source_index.with_name(f"{source_index.name}.lock")
    lock_mtime = lock_path.stat().st_mtime_ns
    worker_cache_dir = _worker_cache_dir(options, dataset_name, crop_strategy)

    dataset = build_preflight_dataset(dataset_name, options, worker_cache_dir)

    assert dataset.name == (
        "qbiolip10k"
        if dataset_name is DatasetName.QBIOLIP
        else "afdb10k"
    )
    assert dataset.physppi.manifest is None
    assert dataset.physppi.sample_ids == []
    assert dataset.physppi.root == (
        options.qbiolip_root
        if dataset_name is DatasetName.QBIOLIP
        else options.afdb_root
    )
    assert dataset.physppi.num_surface_neighbors == 128
    assert Path(dataset.physppi.index_cache_file) == source_index
    assert not (worker_cache_dir / "reader-index-cache.pkl").exists()
    assert Path(dataset._dataset_cache_file) == worker_cache_dir / "cache.json"
    assert dataset._physppi_reference_cache_directory.is_relative_to(worker_cache_dir)
    weights = dataset.crop["token_crop"]["crop_weights"]
    assert dataset.crop["token_crop"]["token_budget"] == 768
    assert weights == {
        "contiguous": 0.0,
        "spatial": float(crop_strategy == "spatial"),
        "spatial_interface": float(crop_strategy == "spatial_interface"),
    }
    assert source_index.read_bytes() == source_before
    assert source_index.stat().st_mtime_ns == source_mtime
    assert lock_path.stat().st_mtime_ns == lock_mtime


@pytest.mark.parametrize("dataset_name", list(DatasetName))
def test_preflight_dataset_instances_reuse_shared_reader_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: DatasetName,
) -> None:
    options = _screening_options(tmp_path)
    reader_class = (
        QBioLiP10kReader if dataset_name is DatasetName.QBIOLIP else AFDB10kReader
    )

    def forbidden_rebuild(*args, **kwargs):
        raise AssertionError("preflight worker must reuse the shared reader index")

    monkeypatch.setattr(reader_class, "_index", forbidden_rebuild)
    datasets = [
        build_preflight_dataset(
            dataset_name,
            options,
            _worker_cache_dir(options, dataset_name, crop_strategy),
        )
        for crop_strategy in CROP_STRATEGIES
    ]

    assert len(datasets) == 2
    assert {Path(dataset.physppi.index_cache_file) for dataset in datasets} == {
        options.qbiolip_index_cache
        if dataset_name is DatasetName.QBIOLIP
        else options.afdb_index_cache
    }


@pytest.mark.parametrize("dataset_name", list(DatasetName))
@pytest.mark.parametrize("crop_strategy", CROP_STRATEGIES)
def test_miniature_reader_reaches_latest_features_and_collator(
    tmp_path: Path,
    dataset_name: DatasetName,
    crop_strategy: str,
) -> None:
    options = _screening_options(tmp_path)
    dataset = build_preflight_dataset(
        dataset_name,
        options,
        _worker_cache_dir(options, dataset_name, crop_strategy),
    )
    matching = dataset.datapoint_cache.index[
        dataset.datapoint_cache["pdb_id"] == SAMPLE_ID
    ].tolist()
    assert matching == [0]

    features = dataset[matching[0]]
    n_tokens = int(features["token_mask"].shape[0])
    batch = openfold_batch_collator([copy.deepcopy(features)])

    assert validate_physppi_features(features) == []
    assert batch["physppi_surface_local_coords"].shape[-3:] == (
        n_tokens,
        128,
        3,
    )
    assert batch["physppi_paired_msa_chain_mask"].ndim == 2


def test_preflight_seed_is_dataset_id_and_crop_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openfold3.projects.of3_all_atom.physppi.screening.preflight as module

    options = _screening_options(tmp_path)
    draws: list[tuple[float, float, float]] = []

    class SeedProbeDataset:
        datapoint_cache = pd.DataFrame({"pdb_id": [SAMPLE_ID]})

        def set_current_datapoint_index(self, index: int) -> None:
            assert index == 0

        def set_current_occurrence(self, occurrence) -> None:
            assert occurrence.datapoint_idx == 0
            assert occurrence.global_slot == 0

        def __getitem__(self, index: int) -> dict:
            assert index == 0
            draws.append(
                (random.random(), float(np.random.random()), float(torch.rand(())))
            )
            return _valid_features()

    monkeypatch.setattr(
        module,
        "build_preflight_dataset",
        lambda dataset, options, worker_cache_dir: SeedProbeDataset(),
    )
    initialize_worker(options)

    first = preflight_sample(DatasetName.QBIOLIP, 0, SAMPLE_ID, "spatial_interface")
    second = preflight_sample(DatasetName.QBIOLIP, 0, SAMPLE_ID, "spatial_interface")

    assert first == second == []
    assert draws[0] == draws[1]


def test_audit_override_forces_level_and_cutoff_without_changing_rng_stream() -> None:
    adapter = object.__new__(PhysPPIPriorComplexDatasetAdapter)
    adapter._physppi_audit_view_override = None
    available = {slot: True for slot in OptionalSlot}
    slot_rng = np.random.default_rng(17)
    slot_control = np.random.default_rng(17)
    contact_rng = np.random.default_rng(23)
    contact_control = np.random.default_rng(23)

    with adapter.audit_prior_view(requested_level=4, contact_cutoff=8.0):
        selection = adapter._select_prior_level(available, slot_rng)
        cutoff = adapter._select_contact_cutoff(contact_rng)

    slot_control.integers(0, 5)
    contact_control.uniform(8.0, 18.0)
    assert selection.requested_level == 4
    assert cutoff == 8.0
    assert slot_rng.integers(0, 5) == slot_control.integers(0, 5)
    assert contact_rng.uniform(8.0, 18.0) == contact_control.uniform(8.0, 18.0)
    assert adapter._physppi_audit_view_override is None


def test_audit_override_is_restored_after_dataset_exception() -> None:
    adapter = object.__new__(PhysPPIPriorComplexDatasetAdapter)
    adapter._physppi_audit_view_override = None

    with (
        pytest.raises(RuntimeError, match="dataset failed"),
        adapter.audit_prior_view(requested_level=2, contact_cutoff=13.0),
    ):
        raise RuntimeError("dataset failed")

    assert adapter._physppi_audit_view_override is None


def test_preflight_audit_view_applies_and_clears_dataset_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openfold3.projects.of3_all_atom.physppi.screening.preflight as module

    options = _screening_options(tmp_path)
    events: list[tuple[str, int, float]] = []

    class AuditProbeDataset:
        datapoint_cache = pd.DataFrame({"pdb_id": [SAMPLE_ID]})

        def set_current_datapoint_index(self, index: int) -> None:
            assert index == 0

        def set_current_occurrence(self, occurrence) -> None:
            assert occurrence.datapoint_idx == 0

        @contextmanager
        def audit_prior_view(self, *, requested_level: int, contact_cutoff: float):
            events.append(("enter", requested_level, contact_cutoff))
            try:
                yield
            finally:
                events.append(("exit", requested_level, contact_cutoff))

        def __getitem__(self, index: int) -> dict:
            assert index == 0
            return _valid_features()

    monkeypatch.setattr(
        module,
        "build_preflight_dataset",
        lambda dataset, options, worker_cache_dir: AuditProbeDataset(),
    )
    initialize_worker(options)
    view = audit_views(DatasetName.AFDB, SAMPLE_ID)[3]

    findings = preflight_audit_view(DatasetName.AFDB, 0, SAMPLE_ID, view)

    assert findings == []
    assert events == [
        ("enter", view.requested_level, view.contact_cutoff),
        ("exit", view.requested_level, view.contact_cutoff),
    ]


def test_dataset_exception_is_latest_preflight_with_sample_id_evidence(
    tmp_path: Path,
) -> None:
    options = _screening_options(tmp_path)
    gt_path = options.qbiolip_root / "chunk" / "chunk0001" / "gt" / f"{SAMPLE_ID}.pdb"
    gt_path.write_text("ATOM\n")
    initialize_worker(options)

    findings = preflight_sample(DatasetName.QBIOLIP, 0, SAMPLE_ID, "spatial")

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity is Severity.LATEST_PREFLIGHT
    assert SAMPLE_ID in finding.message
    assert finding.metrics["sample_id"] == SAMPLE_ID
    assert finding.exception_type


def test_missing_shared_resource_is_infrastructure_with_sample_id_evidence(
    tmp_path: Path,
) -> None:
    options = _screening_options(tmp_path).model_copy(
        update={"components_cif": tmp_path / "missing-components.cif"}
    )
    initialize_worker(options)

    findings = preflight_sample(DatasetName.AFDB, 0, SAMPLE_ID, "spatial_interface")

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity is Severity.INFRASTRUCTURE
    assert SAMPLE_ID in finding.message
    assert finding.metrics["sample_id"] == SAMPLE_ID
