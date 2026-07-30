import numpy as np
import pytest
from pydantic import ValidationError

from openfold3.core.data.framework.single_datasets.abstract_single import (
    DATASET_REGISTRY,
)
from openfold3.core.data.framework.single_datasets.prior_complex import (
    PhysPPIPriorComplexDatasetAdapter,
)
from openfold3.core.data.framework.single_datasets.qbiolip10k import (
    QBioLiP10kDataset,
)
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    QBioLiP10kConfig,
    QBioLiP10kSettings,
)


def test_qbiolip_config_has_one_root_and_no_legacy_fields(tmp_path):
    root = tmp_path / "Q-BioLiP-10k-new"
    root.mkdir()
    settings = QBioLiP10kSettings.model_validate({"root": root})
    assert settings.root.name == "Q-BioLiP-10k-new"
    with pytest.raises(ValidationError):
        QBioLiP10kSettings.model_validate(
            {"root": root, "pkl_directory": str(tmp_path / "old")}
        )


def test_qbiolip_config_accepts_explicit_sampling_weights(tmp_path):
    root = tmp_path / "Q-BioLiP-10k-new"
    root.mkdir()
    weights = tmp_path / "sampling_weights.pt"
    weights.touch()

    settings = QBioLiP10kSettings.model_validate(
        {
            "root": root,
            "sampling_weights": weights,
        }
    )

    assert settings.sampling_weights == weights


def test_qbiolip_config_accepts_positive_total_loss_weight(tmp_path):
    root = tmp_path / "Q-BioLiP-10k-new"
    root.mkdir()

    settings = QBioLiP10kSettings.model_validate(
        {
            "root": root,
            "loss_weight": 0.8,
        }
    )

    assert settings.loss_weight == pytest.approx(0.8)
    with pytest.raises(ValidationError):
        QBioLiP10kSettings.model_validate(
            {
                "root": root,
                "loss_weight": 0.0,
            }
        )


def test_qbiolip_config_defaults_to_interface_preserving_crop(tmp_path):
    root = tmp_path / "Q-BioLiP-10k-new"
    root.mkdir()

    settings = QBioLiP10kSettings.model_validate({"root": root})

    assert settings.interface_preserving_crop is True
    assert settings.min_residues_per_retained_protein_chain == 15
    assert settings.geometry_crop_retry_attempts == 16
    assert settings.retained_view_candidate_cap == 1024


def test_physppi_training_uses_explicit_sampling_weights(tmp_path):
    from types import SimpleNamespace

    import torch

    from openfold3.projects.of3_all_atom.physppi.data.adapters._runtime import (
        _resolve_physppi_sampling_weights,
    )

    manifest = tmp_path / "manifests" / "split_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}")
    weights = tmp_path / "weights" / "sampling_weights.pt"
    weights.parent.mkdir()
    torch.save(
        {"sampling_weights": {"case-a": 1.0, "case-b": 3.0}},
        weights,
    )
    physppi = SimpleNamespace(
        manifest=manifest,
        manifest_partition="train_ids",
        sampling_weights=weights,
    )

    probabilities = _resolve_physppi_sampling_weights(
        ["case-a", "case-b"],
        physppi,
    )

    assert probabilities == pytest.approx([1.0, 3.0])


def test_physppi_training_rejects_incomplete_sampling_weights(tmp_path):
    from types import SimpleNamespace

    import torch

    from openfold3.projects.of3_all_atom.physppi.data.adapters._runtime import (
        _resolve_physppi_sampling_weights,
    )

    manifest = tmp_path / "split_manifest.json"
    manifest.write_text("{}")
    weights = tmp_path / "sampling_weights.pt"
    torch.save({"sampling_weights": {"case-a": 1.0}}, weights)
    physppi = SimpleNamespace(
        manifest=manifest,
        manifest_partition="train_ids",
        sampling_weights=weights,
    )

    with pytest.raises(ValueError, match="case-b"):
        _resolve_physppi_sampling_weights(["case-a", "case-b"], physppi)


def test_physppi_validation_does_not_load_sampling_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime

    manifest = tmp_path / "split_manifest.json"
    manifest.write_text("{}")
    weights = tmp_path / "sampling_weights.pt"
    weights.touch()
    physppi = SimpleNamespace(
        manifest=manifest,
        manifest_partition="val_ids",
        sampling_weights=weights,
    )
    monkeypatch.setattr(
        _runtime._torch,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "validation must not load training sampling weights"
        ),
    )

    probabilities = _runtime._resolve_physppi_sampling_weights(
        ["val-a", "val-b"],
        physppi,
    )

    assert probabilities == [1.0, 1.0]


def _prior_sample(tmp_path, sample_id, dmap):
    from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
        PriorComplexSample,
        PriorSamplePaths,
    )

    return PriorComplexSample(
        sample_id=sample_id,
        dmap=dmap,
        interactions=(),
        mesh={},
        opm={},
        paths=PriorSamplePaths(
            sample_id=sample_id,
            gt_pdb=tmp_path / f"{sample_id}.pdb",
            monomer_pdbs={
                "A": tmp_path / f"{sample_id}_A.pdb",
                "B": tmp_path / f"{sample_id}_B.pdb",
            },
            main_msa_npzs={},
            paired_msa_npzs={},
            prior_pkl=tmp_path / f"{sample_id}.pkl",
        ),
        chain_sequences={"A": "A" * 20, "B": "A" * 20},
    )


def test_datapoint_cache_excludes_samples_without_valid_dmap(tmp_path, caplog):
    from types import SimpleNamespace

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime

    samples = {
        "keep": _prior_sample(tmp_path, "keep", {"A|B": ((1, 1, 12.0),)}),
        "empty": _prior_sample(tmp_path, "empty", {}),
        "empty-edge": _prior_sample(tmp_path, "empty-edge", {"A|B": ()}),
    }
    reader = SimpleNamespace(
        sample_ids=lambda: list(samples),
        read=lambda sample_id: samples[sample_id],
    )
    physppi = SimpleNamespace(
        manifest=None,
        manifest_dataset_key=None,
        manifest_partition="train_ids",
        sampling_weights=None,
        sample_ids=[],
        max_samples=None,
    )
    adapter = SimpleNamespace()

    with caplog.at_level("WARNING"):
        _runtime.PhysPPIPriorComplexDatasetAdapter.create_datapoint_cache_from_reader(
            adapter,
            reader,
            physppi,
        )

    assert adapter.datapoint_cache["pdb_id"].tolist() == ["keep"]
    assert adapter.datapoint_cache["datapoint_probabilities"].tolist() == [1.0]
    assert "excluded 2 PhysPPI samples without valid dmap" in caplog.text
    assert "empty" in caplog.text


def test_datapoint_cache_filters_precomputed_dmap_without_heavy_reader_read(
    tmp_path,
    caplog,
):
    from types import SimpleNamespace

    import torch

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime

    paths = {}
    for sample_id in ("keep", "missing", "empty-edge"):
        chunk = tmp_path / sample_id
        chunk.mkdir()
        paths[sample_id] = SimpleNamespace(
            gt_pdb=chunk / f"{sample_id}_gt.pt",
            monomer_pdbs={
                "A": chunk / f"{sample_id}_A_monomer.pt",
                "B": chunk / f"{sample_id}_B_monomer.pt",
            },
        )
    torch.save(
        {"dmap": {"A|B": torch.tensor([[1, 1, 12.0]], dtype=torch.float32)}},
        paths["keep"].gt_pdb.parent / "keep_interface_residues.pt",
    )
    torch.save(
        {"dmap": {"A|B": torch.empty((0, 3), dtype=torch.float32)}},
        paths["empty-edge"].gt_pdb.parent / "empty-edge_interface_residues.pt",
    )

    def fail_read(sample_id):
        raise AssertionError(f"heavy reader.read() should not run for {sample_id}")

    reader = SimpleNamespace(
        precomputed_only=True,
        sample_ids=lambda: list(paths),
        locate=lambda sample_id: paths[sample_id],
        read=fail_read,
    )
    physppi = SimpleNamespace(
        manifest=None,
        manifest_dataset_key=None,
        manifest_partition="train_ids",
        sampling_weights=None,
        sample_ids=[],
        max_samples=None,
    )
    adapter = SimpleNamespace()

    with caplog.at_level("WARNING"):
        _runtime.PhysPPIPriorComplexDatasetAdapter.create_datapoint_cache_from_reader(
            adapter,
            reader,
            physppi,
        )

    assert adapter.datapoint_cache["pdb_id"].tolist() == ["keep"]
    assert "excluded 2 PhysPPI samples without valid dmap" in caplog.text


def test_manifest_bound_datapoint_cache_trusts_manifest_without_eager_dmap_scan(
    tmp_path,
    caplog,
):
    from types import SimpleNamespace

    import torch

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime

    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        '{"datasets": {"qbiolip": {"train_ids": ["keep", "missing-dmap"]}}}'
    )
    paths = {}
    for sample_id in ("keep", "missing-dmap"):
        chunk = tmp_path / sample_id
        chunk.mkdir()
        paths[sample_id] = SimpleNamespace(
            gt_pdb=chunk / f"{sample_id}_gt.pt",
            monomer_pdbs={
                "A": chunk / f"{sample_id}_A_monomer.pt",
                "B": chunk / f"{sample_id}_B_monomer.pt",
            },
        )
    torch.save(
        {"dmap": {"A|B": torch.tensor([[1, 1, 12.0]], dtype=torch.float32)}},
        paths["keep"].gt_pdb.parent / "keep_interface_residues.pt",
    )

    def fail_read(sample_id):
        raise AssertionError(f"heavy reader.read() should not run for {sample_id}")

    reader = SimpleNamespace(
        precomputed_only=True,
        sample_ids=lambda: list(paths),
        locate=lambda sample_id: paths[sample_id],
        read=fail_read,
    )
    physppi = SimpleNamespace(
        manifest=manifest,
        manifest_dataset_key="qbiolip",
        manifest_partition="train_ids",
        sampling_weights=None,
        sample_ids=[],
        max_samples=None,
    )
    adapter = SimpleNamespace()

    with caplog.at_level("WARNING"):
        _runtime.PhysPPIPriorComplexDatasetAdapter.create_datapoint_cache_from_reader(
            adapter,
            reader,
            physppi,
        )

    assert adapter.datapoint_cache["pdb_id"].tolist() == ["keep", "missing-dmap"]
    assert "excluded" not in caplog.text


def test_qbiolip_dataset_is_registered():
    assert DATASET_REGISTRY["QBioLiP10kDataset"] is QBioLiP10kDataset


def test_qbiolip_dataset_class_has_canonical_reader_boundary():
    assert QBioLiP10kDataset.__name__ == "QBioLiP10kDataset"
    assert issubclass(QBioLiP10kDataset, PhysPPIPriorComplexDatasetAdapter)
    assert "reader" in QBioLiP10kDataset.__init__.__code__.co_names


def test_qbiolip_dataset_runtime_does_not_index_dataset_cache_by_prior_id():
    assert "structure_data" not in QBioLiP10kDataset.__getitem__.__code__.co_names


def test_qbiolip_dataset_does_not_limit_reader_index(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from openfold3.core.data.framework.single_datasets import qbiolip10k as module

    reader_args = {}

    def fake_reader(root, max_samples, *, index_cache_file):
        reader_args.update(
            root=root,
            max_samples=max_samples,
            index_cache_file=index_cache_file,
        )
        return object()

    def fake_adapter_init(self, dataset_config):
        self.dataset_cache = SimpleNamespace(release_connections=lambda: None)

    monkeypatch.setattr(module, "QBioLiP10kReader", fake_reader)
    monkeypatch.setattr(
        PhysPPIPriorComplexDatasetAdapter,
        "__init__",
        fake_adapter_init,
    )
    monkeypatch.setattr(
        QBioLiP10kDataset,
        "create_datapoint_cache_from_reader",
        lambda self, reader, physppi: None,
    )

    cache_path = tmp_path / "qbiolip-index.pkl"
    config = SimpleNamespace(
        physppi=SimpleNamespace(
            root=tmp_path,
            max_samples=256,
            index_cache_file=cache_path,
        )
    )

    dataset = QBioLiP10kDataset(config)

    assert reader_args == {
        "root": tmp_path,
        "max_samples": None,
        "index_cache_file": cache_path,
    }
    assert dataset.physppi.max_samples == 256


def test_qbiolip_dataset_error_identifies_sample_id():
    from types import SimpleNamespace

    import pandas as pd
    from biotite import InvalidFileError

    dataset = object.__new__(QBioLiP10kDataset)
    dataset.datapoint_cache = pd.DataFrame(
        {
            "pdb_id": ["bad-sample"],
            "preferred_chain_or_interface": [None],
        }
    )
    dataset.reader = SimpleNamespace(
        read=lambda sample_id: SimpleNamespace(sample_id=sample_id)
    )

    def fail_build(sample, datapoint):
        raise InvalidFileError("broken template")

    dataset.build_sample_features = fail_build

    with pytest.raises(RuntimeError, match="bad-sample") as exc_info:
        dataset[0]

    assert isinstance(exc_info.value.__cause__, InvalidFileError)


def test_physppi_sample_preserves_exact_cropped_atom_array_with_expanded_gt(
    monkeypatch,
):
    from types import SimpleNamespace

    import torch
    from biotite.structure import AtomArray

    from openfold3.core.data.framework.data_module import openfold_batch_collator
    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime

    full_atom_array = AtomArray(2)
    full_atom_array.atom_name = np.asarray(["FULL-0", "FULL-1"])
    cropped_atom_array = AtomArray(1)
    cropped_atom_array.atom_name = np.asarray(["CROPPED"])
    sample = SimpleNamespace(sample_id="case-a")
    view = SimpleNamespace(source_chain_ids=("A",))
    occurrence_rng = SimpleNamespace(chain_view=object(), seed=42)
    dataset = object.__new__(QBioLiP10kDataset)
    dataset.occurrence = None
    dataset._physppi_data_seed = 42
    dataset.name = "qbiolip"
    dataset.debug_mode = False
    dataset.create_canonical_all_features = lambda **kwargs: {
        "atom_array": full_atom_array,
        "atom_array_cropped": cropped_atom_array,
    }
    dataset.finish_physppi_features = lambda restricted_sample, sample_data: {
        "token_mask": torch.tensor([True]),
        "atom_mask": torch.tensor([True]),
        "ground_truth": {
            "atom_positions": torch.zeros((2, 3)),
        },
    }

    monkeypatch.setattr(
        _runtime,
        "build_occurrence_rng",
        lambda *args, **kwargs: occurrence_rng,
    )
    monkeypatch.setattr(
        _runtime,
        "select_retained_chain_view",
        lambda *args, **kwargs: view,
    )
    monkeypatch.setattr(
        _runtime,
        "restrict_prior_sample_to_chains",
        lambda *args, **kwargs: sample,
    )
    monkeypatch.setattr(
        _runtime,
        "encode_retained_chain_view",
        lambda *args, **kwargs: {},
    )

    features = dataset.build_sample_features(
        sample,
        {"preferred_chain_or_interface": None},
    )
    batch = openfold_batch_collator([features])

    assert batch["atom_array"][0] is cropped_atom_array
    assert batch["atom_array"][0].atom_name.tolist() == ["CROPPED"]
    assert len(batch["atom_array"][0]) == batch["atom_mask"].shape[-1]
    assert batch["ground_truth"]["atom_positions"].shape[-2] == 2

    dataset.finish_physppi_features = lambda restricted_sample, sample_data: {
        "token_mask": torch.tensor([True]),
        "atom_mask": torch.tensor([True, True]),
        "ground_truth": {
            "atom_positions": torch.zeros((2, 3)),
        },
    }
    with pytest.raises(ValueError, match=r"atom_array=1, atom_mask=2"):
        dataset.build_sample_features(
            sample,
            {"preferred_chain_or_interface": None},
        )


def test_physppi_retries_second_layer_crop_without_resampling_chain_view(
    monkeypatch,
):
    from types import SimpleNamespace

    import torch
    from biotite.structure import AtomArray

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime
    from openfold3.projects.of3_all_atom.physppi.features import PhysPPIDataError

    cropped_atom_array = AtomArray(1)
    cropped_atom_array.atom_name = np.asarray(["CA"])
    sample = SimpleNamespace(sample_id="case-a")
    view = SimpleNamespace(source_chain_ids=("A", "B"))
    occurrence = SimpleNamespace(epoch=2, global_slot=99)
    occurrence_rng = SimpleNamespace(chain_view=object(), seed=123)
    dataset = object.__new__(QBioLiP10kDataset)
    dataset.occurrence = occurrence
    dataset._physppi_data_seed = 42
    dataset.name = "qbiolip"
    dataset.debug_mode = False
    dataset.physppi = SimpleNamespace(geometry_crop_retry_attempts=2)
    view_calls = []
    attempt_seeds = []
    finish_calls = 0

    def create_canonical_all_features(**kwargs):
        attempt_seeds.append(kwargs["crop_attempt_seed"])
        return {
            "features": {},
            "atom_array_cropped": cropped_atom_array,
        }

    def finish_physppi_features(restricted_sample, sample_data):
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            raise PhysPPIDataError("invalid_physppi_crop", "too few residues")
        return {
            "token_mask": torch.tensor([True]),
            "atom_mask": torch.tensor([True]),
            "ground_truth": {
                "atom_positions": torch.zeros((1, 3)),
            },
        }

    monkeypatch.setattr(
        _runtime,
        "build_occurrence_rng",
        lambda *args, **kwargs: occurrence_rng,
    )
    monkeypatch.setattr(
        _runtime,
        "select_retained_chain_view",
        lambda *args, **kwargs: view_calls.append(args) or view,
    )
    monkeypatch.setattr(
        _runtime,
        "restrict_prior_sample_to_chains",
        lambda *args, **kwargs: sample,
    )
    monkeypatch.setattr(
        _runtime,
        "encode_retained_chain_view",
        lambda *args, **kwargs: {},
    )
    dataset.create_canonical_all_features = create_canonical_all_features
    dataset.finish_physppi_features = finish_physppi_features

    features = dataset.build_sample_features(
        sample,
        {"preferred_chain_or_interface": None},
    )

    assert features["atom_array"] is cropped_atom_array
    assert len(view_calls) == 1
    assert finish_calls == 2
    assert len(attempt_seeds) == 2
    assert attempt_seeds[0] != attempt_seeds[1]


def test_physppi_retries_crop_below_residue_floor_without_finishing(
    monkeypatch,
):
    from types import SimpleNamespace

    import torch
    from biotite.structure import AtomArray

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime

    def protein_ca_array(count):
        atom_array = AtomArray(count)
        atom_array.coord = np.zeros((count, 3), dtype=float)
        atom_array.chain_id = np.asarray(["A"] * count)
        atom_array.res_id = np.arange(1, count + 1)
        atom_array.atom_name = np.asarray(["CA"] * count)
        return atom_array

    sample = SimpleNamespace(sample_id="case-a", dmap={"A|B": ((1, 1, 8.0),)})
    view = SimpleNamespace(source_chain_ids=("A", "B"))
    occurrence_rng = SimpleNamespace(chain_view=object(), seed=123)
    dataset = object.__new__(QBioLiP10kDataset)
    dataset.occurrence = None
    dataset._physppi_data_seed = 42
    dataset.name = "qbiolip"
    dataset.debug_mode = False
    dataset.physppi = SimpleNamespace(
        geometry_crop_retry_attempts=2,
        min_residues_per_retained_protein_chain=2,
    )
    cropped_by_attempt = [protein_ca_array(1), protein_ca_array(2)]
    finish_calls = 0
    view_calls = 0

    def create_canonical_all_features(**kwargs):
        return {
            "features": {},
            "atom_array_cropped": cropped_by_attempt.pop(0),
        }

    def finish_physppi_features(restricted_sample, sample_data):
        nonlocal finish_calls
        finish_calls += 1
        return {
            "token_mask": torch.ones(2, dtype=torch.bool),
            "atom_mask": torch.ones(
                len(sample_data["atom_array_cropped"]),
                dtype=torch.bool,
            ),
            "ground_truth": {
                "atom_positions": torch.zeros(
                    (len(sample_data["atom_array_cropped"]), 3)
                ),
            },
        }

    def select_view(*args, **kwargs):
        nonlocal view_calls
        view_calls += 1
        return view

    monkeypatch.setattr(
        _runtime,
        "build_occurrence_rng",
        lambda *args, **kwargs: occurrence_rng,
    )
    monkeypatch.setattr(_runtime, "select_retained_chain_view", select_view)
    monkeypatch.setattr(
        _runtime,
        "restrict_prior_sample_to_chains",
        lambda *args, **kwargs: sample,
    )
    monkeypatch.setattr(
        _runtime,
        "encode_retained_chain_view",
        lambda *args, **kwargs: {},
    )
    dataset.create_canonical_all_features = create_canonical_all_features
    dataset.finish_physppi_features = finish_physppi_features

    features = dataset.build_sample_features(
        sample,
        {"preferred_chain_or_interface": None},
    )

    assert features["atom_array"].res_id.tolist() == [1, 2]
    assert finish_calls == 1
    assert view_calls == 1


def test_physppi_crop_retry_exhaustion_includes_diagnostics(monkeypatch):
    from types import SimpleNamespace

    from biotite.structure import AtomArray

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime
    from openfold3.projects.of3_all_atom.physppi.features import PhysPPIDataError

    cropped_atom_array = AtomArray(1)
    cropped_atom_array.coord = np.zeros((1, 3), dtype=float)
    cropped_atom_array.chain_id = np.asarray(["A"])
    cropped_atom_array.res_id = np.asarray([1])
    cropped_atom_array.atom_name = np.asarray(["CA"])
    sample = SimpleNamespace(
        sample_id="case-a",
        dmap={"A|B": ((1, 1, 8.0),)},
        chain_sequences={"A": "AA", "B": "AA"},
    )
    view = SimpleNamespace(source_chain_ids=("A", "B"))
    occurrence_rng = SimpleNamespace(chain_view=object(), seed=123)
    dataset = object.__new__(QBioLiP10kDataset)
    dataset.occurrence = None
    dataset._physppi_data_seed = 42
    dataset.name = "qbiolip"
    dataset.debug_mode = False
    dataset.physppi = SimpleNamespace(
        geometry_crop_retry_attempts=16,
        min_residues_per_retained_protein_chain=2,
    )
    create_calls = 0

    def create_canonical_all_features(**kwargs):
        nonlocal create_calls
        create_calls += 1
        return {
            "features": {},
            "atom_array_cropped": cropped_atom_array,
        }

    dataset.create_canonical_all_features = create_canonical_all_features
    dataset.finish_physppi_features = lambda *args, **kwargs: pytest.fail(
        "floor validation should fail before feature finishing"
    )

    monkeypatch.setattr(
        _runtime,
        "build_occurrence_rng",
        lambda *args, **kwargs: occurrence_rng,
    )
    monkeypatch.setattr(
        _runtime,
        "select_retained_chain_view",
        lambda *args, **kwargs: view,
    )
    monkeypatch.setattr(
        _runtime,
        "restrict_prior_sample_to_chains",
        lambda *args, **kwargs: sample,
    )

    with pytest.raises(PhysPPIDataError) as exc_info:
        dataset.build_sample_features(
            sample,
            {"preferred_chain_or_interface": None},
        )

    assert exc_info.value.args[0] == "physppi_crop_retry_exhausted"
    detail = exc_info.value.args[1]
    assert create_calls == 16
    assert "failed 16 crop attempts" in detail
    assert "cropped_residue_counts={'A': 1, 'B': 0}" in detail
    assert "dmap_residue_counts={'A': 1, 'B': 1}" in detail


def test_qbiolip_adapter_owns_canonical_runtime_paths():
    assert (
        "create_canonical_all_features"
        in PhysPPIPriorComplexDatasetAdapter.build_sample_features.__code__.co_names
    )
    assert "create_canonical_msa_features" in dir(PhysPPIPriorComplexDatasetAdapter)
    assert "create_canonical_template_features" in dir(
        PhysPPIPriorComplexDatasetAdapter
    )


def test_qbiolip_config_is_registered():
    config = QBioLiP10kConfig
    assert config.model_fields["physppi"].annotation is QBioLiP10kSettings


def test_adapter_routes_query_only_fallback_only_to_main_msa(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from biotite.structure import AtomArray

    def write_msa(path, sequence, rows=1):
        msa = np.asarray([list(sequence)] * rows, dtype="<U1")
        np.savez(
            path,
            uniref90={
                "msa": msa,
                "deletion_matrix": np.zeros(msa.shape, dtype=int),
                "metadata": pd.DataFrame(),
            },
        )

    atom_array = AtomArray(2)
    atom_array.coord = np.zeros((2, 3), dtype=float)
    atom_array.chain_id = np.asarray(["A", "B"])
    atom_array.res_id = np.asarray([1, 1])
    atom_array.res_name = np.asarray(["ALA", "CYS"])
    atom_array.atom_name = np.asarray(["CA", "CA"])

    main_a = tmp_path / "main_a.npz"
    paired_b = tmp_path / "paired_b.npz"
    write_msa(main_a, "A", rows=2)
    write_msa(paired_b, "C", rows=3)
    sample = SimpleNamespace(
        sample_id="case1",
        paths=SimpleNamespace(
            main_msa_npzs={"A": main_a},
            paired_msa_npzs={"B": paired_b},
        ),
    )
    adapter = object.__new__(PhysPPIPriorComplexDatasetAdapter)
    adapter.n_tokens = 2

    captured = {}

    def fake_featurizer(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    adapter.msa_featurizer_of3 = fake_featurizer

    result = adapter.create_canonical_msa_features(sample, atom_array)
    assert result["ok"] is True
    assert result["physppi_paired_msa_chain_mask"].tolist() == [False, True]

    collection = captured["msa_array_collection"]
    assert collection.chain_id_to_main_msa["A"].msa.shape[0] == 2
    assert collection.chain_id_to_main_msa["B"].msa.tolist() == [["C"]]
    assert collection.chain_id_to_paired_msa["B"].msa.shape[0] == 3
    assert collection.chain_id_to_paired_msa["A"].msa.shape[0] == 0
    assert collection.row_counts.n_rows_paired_subsampled == 3


def test_adapter_emits_latest_contract_feature_keys(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from biotite.structure import AtomArray

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module
    from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
        PRIOR_OPTIONAL_SLOTS,
        SlotState,
    )

    interaction_call = {}

    def fake_interaction(*args, **kwargs):
        interaction_call["atom_array"] = kwargs.get("atom_array")
        return {
            "type_mask": np.zeros((2, 2, 7), dtype=bool),
            "valid": True,
            "aux": ((0, 1, 0, 3.5, ("N",), ("O",)),),
        }

    monkeypatch.setattr(
        module,
        "build_contact_features",
        lambda *args, **kwargs: {
            "pair_mask": np.asarray([[False, True], [True, False]], dtype=bool),
            "valid": True,
        },
    )
    monkeypatch.setattr(
        module,
        "build_interaction_features",
        fake_interaction,
    )
    surface_features = np.tile(np.arange(6, dtype=np.float32), (2, 128, 1))
    monkeypatch.setattr(
        module,
        "build_surface_features",
        lambda *args, **kwargs: {
            "coords": np.zeros((2, 128, 3), dtype=np.float32),
            "features": surface_features,
            "mask": np.ones((2, 128), dtype=bool),
            "charge_mask": np.ones((2, 128), dtype=bool),
            "valid": True,
        },
    )
    monkeypatch.setattr(
        module,
        "build_tm_features",
        lambda *args, **kwargs: np.asarray([False, True]),
    )
    monkeypatch.setattr(
        module,
        "build_membrane_target",
        lambda *args, **kwargs: {
            "normal": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "point": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "valid": True,
        },
    )

    adapter = object.__new__(PhysPPIPriorComplexDatasetAdapter)
    adapter.physppi = SimpleNamespace(num_surface_neighbors=128)
    atom_array = AtomArray(2)
    atom_array.chain_id = np.asarray(["A", "B"])
    sample = SimpleNamespace(dmap={}, interactions=(), mesh={}, opm={})
    states = {
        slot: SlotState(True, True, True, "qbiolip10k") for slot in PRIOR_OPTIONAL_SLOTS
    }

    result = adapter.build_prior_features(
        sample,
        atom_array,
        {("A", 1): 0, ("B", 1): 1},
        states,
        np.random.default_rng(1),
    )

    assert "physppi_contact_mask" not in result
    assert "physppi_contact_token_mask" not in result
    assert result["physppi_contact_pair_mask"].shape == (2, 2)
    assert result["physppi_contact_pair_mask"].tolist() == [
        [False, True],
        [True, False],
    ]
    assert result["physppi_surface_features"].shape == (2, 128, 6)
    assert result["physppi_surface_features"][0, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert interaction_call["atom_array"] is atom_array
    assert result["physppi_interaction_valid"].item() is True
    assert result["physppi_interaction_aux_valid"].shape == (2, 2)
    assert result["physppi_interaction_aux_valid"].tolist() == [
        [False, True],
        [True, False],
    ]
    assert result["physppi_interaction_distance"].shape == (2, 2)
    assert result["physppi_interaction_distance"].tolist() == [
        [0.0, 3.5],
        [3.5, 0.0],
    ]
    assert not any(key.startswith("physppi_paired_msa_") for key in result)


def test_adapter_truncates_later_slots_after_invalid_middle_prior(monkeypatch):
    from types import SimpleNamespace

    from biotite.structure import AtomArray

    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module
    from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
        PRIOR_OPTIONAL_SLOTS,
        SlotState,
    )

    monkeypatch.setattr(
        module,
        "build_contact_features",
        lambda *args, **kwargs: {
            "pair_mask": np.asarray([[False, True], [True, False]], dtype=bool),
            "valid": True,
        },
    )
    monkeypatch.setattr(
        module,
        "build_interaction_features",
        lambda *args, **kwargs: {
            "type_mask": np.zeros((2, 2, 7), dtype=bool),
            "valid": False,
            "aux": [],
        },
    )
    monkeypatch.setattr(
        module,
        "build_surface_features",
        lambda *args, **kwargs: {
            "coords": np.zeros((2, 1, 3), dtype=np.float32),
            "features": np.zeros((2, 1, 6), dtype=np.float32),
            "mask": np.ones((2, 1), dtype=bool),
            "charge_mask": np.ones((2, 1), dtype=bool),
            "valid": True,
        },
    )
    monkeypatch.setattr(
        module, "build_tm_features", lambda *args, **kwargs: np.asarray([True, True])
    )
    monkeypatch.setattr(
        module,
        "build_membrane_target",
        lambda *args, **kwargs: {
            "normal": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "point": np.zeros(3, dtype=np.float32),
            "valid": True,
        },
    )

    adapter = object.__new__(PhysPPIPriorComplexDatasetAdapter)
    adapter.physppi = SimpleNamespace(num_surface_neighbors=1)
    atom_array = AtomArray(2)
    atom_array.chain_id = np.asarray(["A", "B"])
    states = {
        slot: SlotState(True, True, True, "source") for slot in PRIOR_OPTIONAL_SLOTS
    }

    result = adapter.build_prior_features(
        SimpleNamespace(dmap={}, interactions=(), mesh={}, opm={}),
        atom_array,
        {("A", 1): 0, ("B", 1): 1},
        states,
        np.random.default_rng(1),
    )

    assert states[module.OptionalSlot.CONTACT].active is True
    assert states[module.OptionalSlot.INTERACTION].valid is False
    assert states[module.OptionalSlot.INTERACTION].selected is False
    assert states[module.OptionalSlot.SURFACE].active is False
    assert states[module.OptionalSlot.TM].active is False
    assert result["physppi_surface_selected"].item() is False
    assert result["physppi_tm_selected"].item() is False


def test_slot_state_features_encode_source_and_unavailable_provenance():
    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module
    from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
        OptionalSlot,
        SlotState,
    )

    features = module._slot_state_features(
        {
            OptionalSlot.CONTACT: SlotState(True, True, True, "source"),
            OptionalSlot.INTERACTION: SlotState(False, False, False, "unavailable"),
        }
    )

    assert features["physppi_contact_provenance"].item() is True
    assert features["physppi_interaction_provenance"].item() is False


def test_build_surface_features_rejects_malformed_present_mesh():
    from openfold3.projects.of3_all_atom.physppi.features import (
        PhysPPIDataError,
        build_surface_features,
    )

    with pytest.raises(PhysPPIDataError, match="surface"):
        build_surface_features(
            {"A": np.zeros((1, 9), dtype=np.float32)},
            {("A", 1): 0},
            {"A": (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
            1,
        )


def test_build_surface_features_allows_valid_mesh_without_cropped_frame():
    from openfold3.projects.of3_all_atom.physppi.features import (
        build_surface_features,
    )

    result = build_surface_features(
        {"A": np.zeros((1, 10), dtype=np.float32)},
        {("A", 1): 0},
        {"A": (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))},
        1,
    )

    assert result["available"] is True
    assert result["valid"] is False


def test_build_surface_features_validates_off_view_mesh_before_skipping():
    from openfold3.projects.of3_all_atom.physppi.features import (
        PhysPPIDataError,
        build_surface_features,
    )

    token_index = {("A", 1): 0}
    transforms = {"A": (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}

    with pytest.raises(PhysPPIDataError, match="surface"):
        build_surface_features(
            {"B": np.zeros((1, 9), dtype=np.float32)}, token_index, transforms, 1
        )

    result = build_surface_features(
        {"B": np.zeros((1, 10), dtype=np.float32)}, token_index, transforms, 1
    )

    assert result["available"] is True
    assert result["valid"] is False
