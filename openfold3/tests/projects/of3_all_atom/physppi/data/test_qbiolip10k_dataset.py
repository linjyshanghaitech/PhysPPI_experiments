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


def test_qbiolip_dataset_is_registered():
    assert DATASET_REGISTRY["QBioLiP10kDataset"] is QBioLiP10kDataset


def test_qbiolip_dataset_class_has_canonical_reader_boundary():
    assert QBioLiP10kDataset.__name__ == "QBioLiP10kDataset"
    assert issubclass(QBioLiP10kDataset, PhysPPIPriorComplexDatasetAdapter)
    assert "reader" in QBioLiP10kDataset.__init__.__code__.co_names


def test_qbiolip_dataset_runtime_does_not_index_dataset_cache_by_prior_id():
    assert "structure_data" not in QBioLiP10kDataset.__getitem__.__code__.co_names


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
