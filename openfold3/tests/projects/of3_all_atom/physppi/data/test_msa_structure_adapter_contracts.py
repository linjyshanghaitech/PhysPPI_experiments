from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAGE_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(STAGE_ROOT))


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return np.asarray(self.value).tolist()

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)


def _install_structure_adapter_stubs(monkeypatch):
    @dataclass(frozen=True)
    class _PriorLevelSelection:
        requested_level: int
        effective_level: int
        states: dict[object, object]

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _FakeTensor
    torch_stub.tensor = lambda value, **_: _FakeTensor(value)
    torch_stub.from_numpy = lambda value: _FakeTensor(value)
    torch_stub.ones = lambda *shape, **_: _FakeTensor(np.ones(shape, dtype=bool))
    torch_stub.zeros = lambda *shape, **_: _FakeTensor(np.zeros(shape))
    torch_stub.bool = bool
    torch_stub.float32 = np.float32
    torch_stub.int64 = np.int64
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    biotite_stub = types.ModuleType("biotite")
    structure_stub = types.ModuleType("biotite.structure")
    structure_stub.AtomArray = object
    structure_stub.get_residue_starts = lambda atom_array: np.arange(
        len(atom_array.chain_id)
    )
    biotite_stub.structure = structure_stub
    monkeypatch.setitem(sys.modules, "biotite", biotite_stub)
    monkeypatch.setitem(sys.modules, "biotite.structure", structure_stub)

    module_attrs = {
        "openfold3.core.data.framework.single_datasets.base_of3": {
            "BaseOF3Dataset": object,
        },
        "openfold3.core.data.framework.single_datasets.dataset_utils": {
            "check_invalid_feature_dict": lambda features: None,
        },
        "openfold3.core.data.io.sequence.msa": {
            "parse_msas_preparsed": lambda paths: {},
        },
        "openfold3.core.data.io.structure.mol": {
            "read_single_annotated_sdf": lambda path: None,
            "write_annotated_sdf": lambda *args, **kwargs: None,
        },
        "openfold3.core.data.io.structure.pdb": {"parse_pdb_af2": lambda path: None},
        "openfold3.core.data.pipelines.featurization.conformer": {
            "featurize_reference_conformers_of3": lambda **kwargs: {},
        },
        "openfold3.core.data.pipelines.featurization.loss_weights": {
            "set_loss_weights": lambda *args, **kwargs: {},
        },
        "openfold3.core.data.pipelines.featurization.structure": {
            "featurize_target_gt_structure_of3": lambda **kwargs: {},
        },
        "openfold3.core.data.pipelines.featurization.template": {
            "featurize_template_structures_of3": lambda **kwargs: {},
        },
        "openfold3.core.data.pipelines.sample_processing.conformer": {
            "get_reference_conformer_data_of3": lambda **kwargs: [],
        },
        "openfold3.core.data.pipelines.sample_processing.structure": {
            "process_target_atom_array_of3": lambda **kwargs: None,
        },
        "openfold3.core.data.primitives.caches.format": {
            "PreprocessingChainData": object,
        },
        "openfold3.core.data.primitives.permutation.mol_labels": {
            "separate_cropped_and_gt": lambda **kwargs: (None, None),
        },
        "openfold3.core.data.primitives.structure.cleanup": {
            "canonicalize_atom_order": lambda atom_array, ccd: atom_array,
            "convert_MSE_to_MET": lambda atom_array: atom_array,
            "remove_hydrogens": lambda atom_array: atom_array,
            "remove_non_CCD_atoms": lambda atom_array, ccd: atom_array,
            "remove_std_residue_terminal_atoms": lambda atom_array: atom_array,
        },
        "openfold3.core.data.primitives.structure.component": {
            "mol_from_ccd_entry": lambda *args, **kwargs: None,
        },
        "openfold3.core.data.primitives.structure.conformer": {
            "resolve_and_format_fallback_conformer": lambda *args, **kwargs: None,
        },
        "openfold3.core.data.primitives.structure.template": {
            "TemplateSlice": object,
            "TemplateSliceCollection": object,
        },
        "openfold3.core.data.primitives.structure.tokenization": {
            "add_token_positions": lambda atom_array: None,
        },
        "openfold3.core.data.resources.residues": {
            "MoleculeType": types.SimpleNamespace(PROTEIN="protein"),
        },
        "openfold3.projects.of3_all_atom.physppi.data.adapters.chain_views": {
            "CHAIN_VIEW_COMPATIBILITY_TRIGGER": 20,
            "CHAIN_VIEW_TOKEN_TRIGGER": 768,
            "RetainedChainView": object,
            "build_chain_graph": lambda *args, **kwargs: {},
            "needs_resource_view": lambda *args, **kwargs: False,
            "sample_retained_chain_view": lambda *args, **kwargs: None,
        },
        "openfold3.projects.of3_all_atom.physppi.data.priors.occurrence": {
            "build_occurrence_rng": lambda *args, **kwargs: np.random.default_rng(0),
        },
        "openfold3.projects.of3_all_atom.physppi.data.priors.augmentation": {
            "PRIOR_OPTIONAL_SLOTS": {},
            "OptionalSlot": object,
            "PriorLevelSelection": _PriorLevelSelection,
            "SlotState": object,
            "available_optional_slots": lambda sample: {},
            "build_paired_msa_chain_mask": lambda available: np.asarray(
                available, dtype=bool
            ),
            "sample_prior_level": lambda *args, **kwargs: 0,
            "select_prior_level": lambda *args, **kwargs: {},
        },
        "openfold3.projects.of3_all_atom.physppi.data.priors.schema": {
            "PriorComplexSample": object,
            "PriorSamplePaths": object,
        },
        "openfold3.projects.of3_all_atom.physppi.features": {
            "PhysPPIDataError": ValueError,
            "build_interface_residue_features": lambda *args, **kwargs: {},
            "build_interaction_features": lambda *args, **kwargs: {},
            "build_membrane_target": lambda *args, **kwargs: {},
            "build_surface_features": lambda *args, **kwargs: {},
            "build_tm_features": lambda *args, **kwargs: {},
            "restrict_tm_payload_to_chains": (lambda payload, *args, **kwargs: payload),
        },
        "openfold3.projects.of3_all_atom.physppi.model.interaction_auxiliary": {
            "_build_pair_interaction_auxiliary": lambda *args, **kwargs: {},
        },
    }
    for name, attrs in module_attrs.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    msa_module = types.ModuleType("openfold3.core.data.primitives.sequence.msa")

    @dataclass
    class MsaArray:
        msa: np.ndarray
        deletion_matrix: np.ndarray
        metadata: object = None

        def __len__(self):
            return int(self.msa.shape[0])

        def truncate(self, row_slice, inplace=False):
            if isinstance(row_slice, int):
                row_slice = slice(row_slice)
            truncated = MsaArray(
                self.msa[row_slice],
                self.deletion_matrix[row_slice],
                self.metadata,
            )
            if inplace:
                self.msa = truncated.msa
                self.deletion_matrix = truncated.deletion_matrix
                return None
            return truncated

    class MsaArrayCollection:
        latest = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chain_id_to_paired_msa = {}
            self.chain_id_to_main_msa = {}
            self.row_counts = None
            MsaArrayCollection.latest = self

        def set_state_processed(self, **kwargs):
            self.chain_id_to_paired_msa = kwargs["chain_id_to_paired_msa"]
            self.chain_id_to_main_msa = kwargs["chain_id_to_main_msa"]

        def set_row_counts(self, **kwargs):
            self.row_counts = kwargs

    msa_module.MsaArray = MsaArray
    msa_module.MsaArrayCollection = MsaArrayCollection
    msa_module.calculate_profile = lambda **kwargs: np.asarray([0.0])
    monkeypatch.setitem(
        sys.modules, "openfold3.core.data.primitives.sequence.msa", msa_module
    )


def _reload_adapter(monkeypatch):
    _install_structure_adapter_stubs(monkeypatch)
    physppi_root = STAGE_ROOT / "openfold3/projects/of3_all_atom/physppi"
    for package_name, package_path in (
        ("openfold3.projects.of3_all_atom.physppi", physppi_root),
        ("openfold3.projects.of3_all_atom.physppi.data", physppi_root / "data"),
        (
            "openfold3.projects.of3_all_atom.physppi.data.adapters",
            physppi_root / "data" / "adapters",
        ),
    ):
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        monkeypatch.setitem(sys.modules, package_name, package)
    module_path = physppi_root / "data" / "adapters" / "_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "_physppi_task6_structure_adapter", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _msa(sequence: str, rows: int = 2):
    array = np.asarray([list(sequence) for _ in range(rows)], dtype="<U1")
    module = sys.modules["openfold3.core.data.primitives.sequence.msa"]
    return module.MsaArray(array, np.zeros(array.shape, dtype=int))


def test_afdb_homodimer_single_main_msa_is_shared_with_provenance(monkeypatch):
    module = _reload_adapter(monkeypatch)
    query_msas = {"A": _msa("AC", rows=1), "B": _msa("AC", rows=1)}
    sample = types.SimpleNamespace(
        sample_id="afdb10k_homo",
        dataset_name="AFDB",
        paths=types.SimpleNamespace(
            main_msa_npzs={"A": Path("afdb10k/case/A.npz")},
        ),
        chain_sequences={"A": "AC", "B": "AC"},
    )
    monkeypatch.setattr(module, "_read_msa_npz", lambda path: _msa("AC", rows=3))

    main_msas, provenance = module._build_main_msas(sample, ["A", "B"], query_msas)

    assert main_msas["A"] is main_msas["B"]
    assert provenance == {"A": "chain_file", "B": "shared_homodimer_file"}


def test_non_afdb_sample_missing_main_msa_uses_query_only_row(monkeypatch):
    module = _reload_adapter(monkeypatch)
    query_msas = {"A": _msa("AC", rows=1), "B": _msa("AC", rows=1)}
    sample = types.SimpleNamespace(
        sample_id="qbiolip_case",
        paths=types.SimpleNamespace(main_msa_npzs={"A": Path("A.npz")}),
        chain_sequences={"A": "AC", "B": "AC"},
    )
    monkeypatch.setattr(module, "_read_msa_npz", lambda path: _msa("AC", rows=3))

    main_msas, provenance = module._build_main_msas(sample, ["A", "B"], query_msas)
    assert provenance == {"A": "chain_file", "B": "query_only"}
    assert main_msas["B"] is query_msas["B"]


def test_msa_projection_removes_unobserved_structure_residues(monkeypatch):
    module = _reload_adapter(monkeypatch)
    source = _msa("ABXCDE", rows=2)
    source.msa[1] = np.asarray(list("ABYADE"), dtype="<U1")
    target = _msa("ABCDE", rows=1)

    projected = module._project_msa_to_query_tokens(source, target)

    assert projected is not None
    assert ["".join(row) for row in projected.msa] == ["ABCDE", "ABADE"]
    assert projected.deletion_matrix.shape == (2, 5)


def test_msa_projection_rejects_placeholder_that_cannot_map_to_gt(monkeypatch):
    module = _reload_adapter(monkeypatch)

    assert (
        module._project_msa_to_query_tokens(_msa("X", rows=1), _msa("AC", rows=1))
        is None
    )


def test_afdb_homodimer_main_msa_share_requires_recorded_identity(monkeypatch):
    module = _reload_adapter(monkeypatch)
    query_msas = {"A": _msa("AC", rows=1), "B": _msa("AD", rows=1)}
    sample = types.SimpleNamespace(
        sample_id="afdb10k_hetero",
        dataset_name="AFDB",
        paths=types.SimpleNamespace(main_msa_npzs={"A": Path("afdb10k/case/A.npz")}),
        chain_sequences={"A": "AC", "B": "AD"},
    )
    monkeypatch.setattr(module, "_read_msa_npz", lambda path: _msa("AC", rows=3))

    main_msas, provenance = module._build_main_msas(sample, ["A", "B"], query_msas)
    assert provenance == {"A": "chain_file", "B": "query_only"}
    assert main_msas["B"] is query_msas["B"]


def test_paired_msa_available_without_optional_selection_and_masks_missing(monkeypatch):
    module = _reload_adapter(monkeypatch)

    class AtomArray:
        chain_id = np.asarray(["A", "B"])
        res_id = np.asarray([1, 1])
        res_name = np.asarray(["ALA", "ALA"])
        atom_name = np.asarray(["CA", "CA"])
        token_id = np.asarray([0, 1])

        def __getitem__(self, mask):
            result = AtomArray()
            result.chain_id = self.chain_id[mask]
            result.res_id = self.res_id[mask]
            result.res_name = self.res_name[mask]
            result.atom_name = self.atom_name[mask]
            result.token_id = self.token_id[mask]
            return result

    sample = types.SimpleNamespace(
        sample_id="case",
        paths=types.SimpleNamespace(
            main_msa_npzs={"A": Path("A_main.npz"), "B": Path("B_main.npz")},
            paired_msa_npzs={"A": Path("A_pair.npz")},
        ),
        chain_sequences={"A": "A", "B": "A"},
    )
    monkeypatch.setattr(module, "_read_msa_npz", lambda path: _msa("A", rows=2))
    adapter = object.__new__(module.PhysPPIPriorComplexDatasetAdapter)
    adapter.n_tokens = 2
    adapter.msa_featurizer_of3 = lambda **kwargs: {}

    features = adapter.create_canonical_msa_features(sample, AtomArray())
    collection = sys.modules[
        "openfold3.core.data.primitives.sequence.msa"
    ].MsaArrayCollection.latest

    assert features["physppi_paired_msa_chain_mask"].tolist() == [True, False]
    assert set(collection.chain_id_to_paired_msa) == {"A", "B"}
    assert len(collection.chain_id_to_paired_msa["B"]) == 0


def test_unmappable_paired_msa_is_masked_instead_of_reaching_featurization(monkeypatch):
    module = _reload_adapter(monkeypatch)

    class AtomArray:
        chain_id = np.asarray(["A"])
        res_id = np.asarray([1])
        res_name = np.asarray(["ALA"])
        atom_name = np.asarray(["CA"])
        token_id = np.asarray([0])

        def __getitem__(self, mask):
            result = AtomArray()
            result.chain_id = self.chain_id[mask]
            result.res_id = self.res_id[mask]
            result.res_name = self.res_name[mask]
            result.atom_name = self.atom_name[mask]
            result.token_id = self.token_id[mask]
            return result

    sample = types.SimpleNamespace(
        sample_id="case",
        paths=types.SimpleNamespace(
            main_msa_npzs={"A": Path("A_main.npz")},
            paired_msa_npzs={"A": Path("A_pair.npz")},
        ),
        chain_sequences={"A": "A"},
    )
    monkeypatch.setattr(
        module,
        "_read_msa_npz",
        lambda path: _msa("X", rows=1) if "pair" in str(path) else _msa("A", rows=2),
    )
    adapter = object.__new__(module.PhysPPIPriorComplexDatasetAdapter)
    adapter.n_tokens = 1
    adapter.msa_featurizer_of3 = lambda **kwargs: {}

    features = adapter.create_canonical_msa_features(sample, AtomArray())
    collection = sys.modules[
        "openfold3.core.data.primitives.sequence.msa"
    ].MsaArrayCollection.latest

    assert features["physppi_paired_msa_chain_mask"].tolist() == [False]
    assert collection.chain_id_to_paired_msa == {}


def test_monomer_template_processing_disables_both_crop_stages(monkeypatch):
    module = _reload_adapter(monkeypatch)
    template_atom_array = object()
    processed_template = object()
    captured = {}

    monkeypatch.setattr(
        module, "_parse_pdb_atom_array", lambda path: template_atom_array
    )
    monkeypatch.setattr(
        module,
        "_canonical_ppi_heavy_atom_array",
        lambda atom_array, *args, **kwargs: atom_array,
    )
    monkeypatch.setattr(
        module, "_annotate_canonical_mol_entity_ids", lambda atom_array: atom_array
    )
    monkeypatch.setattr(module, "_chain_metadata", lambda atom_array: {})
    monkeypatch.setattr(
        module.PhysPPIPriorComplexDatasetAdapter,
        "_ccd_for_atom_array",
        lambda self, atom_array: None,
    )
    monkeypatch.setattr(
        module,
        "process_target_atom_array_of3",
        lambda **kwargs: (
            captured.update(kwargs)
            or types.SimpleNamespace(atom_array_gt=processed_template)
        ),
    )
    monkeypatch.setattr(module, "add_token_positions", lambda atom_array: None)

    adapter = object.__new__(module.PhysPPIPriorComplexDatasetAdapter)
    adapter.reference_molecule_directory = Path("reference_molecules")
    assert adapter._process_monomer_template(Path("monomer.pdb")) is processed_template
    assert captured["crop_config"] == {
        "chain_crop": {"enabled": False},
        "token_crop": {"enabled": False},
    }


def test_token_index_ignores_canonical_feature_padding(monkeypatch):
    module = _reload_adapter(monkeypatch)

    class AtomArray:
        chain_id = np.asarray(["A", "A", "B", "B"])
        res_id = np.asarray([1, 1, 1, 1])
        token_id = np.asarray([0, 0, 1, 1])

    features = {
        "token_index": _FakeTensor([1148, 1149, 0, 0]),
        "token_mask": _FakeTensor([True, True, False, False]),
    }

    assert module._token_index(AtomArray(), features) == {("A", 1): 0, ("B", 1): 1}
