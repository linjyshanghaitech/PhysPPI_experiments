import pickle
import stat
import sys
import types
from pathlib import Path

from openfold3.projects.of3_all_atom.config.dataset_configs import (
    QBioLiP10kSettings,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    PriorSamplePaths,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader
from openfold3.projects.of3_all_atom.physppi.data.readers.index_cache import (
    load_or_create_reader_index,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    QBioLiP10kReader,
)


def test_reader_index_builder_runs_once_for_matching_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "qbiolip-index.pkl"
    calls = 0

    def build_index() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"sample": "validated"}

    metadata = {"reader": "qbiolip", "root": "/dataset", "max_samples": None}

    first = load_or_create_reader_index(cache_path, metadata, build_index)
    second = load_or_create_reader_index(cache_path, metadata, build_index)

    assert first == {"sample": "validated"}
    assert second == first
    assert calls == 1


def test_matching_read_only_reader_index_is_reused_without_rebuild(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "qbiolip-index.pkl"
    metadata = {"reader": "qbiolip", "root": "/dataset", "max_samples": None}

    first = load_or_create_reader_index(
        cache_path, metadata, lambda: {"sample": "validated"}
    )
    cache_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    def forbidden_builder() -> dict[str, str]:
        raise AssertionError("matching read-only cache must not be rebuilt")

    second = load_or_create_reader_index(cache_path, metadata, forbidden_builder)

    assert first == second


def test_legacy_prior_complex_class_path_is_mapped_without_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "qbiolip-index.pkl"
    metadata = {"reader": "qbiolip", "root": "/dataset", "max_samples": None}
    paths = PriorSamplePaths(
        sample_id="fixture",
        gt_pdb=Path("gt.pdb"),
        monomer_pdbs={"A": Path("A.pdb")},
        main_msa_npzs={},
        paired_msa_npzs={},
        prior_pkl=Path("prior.pkl"),
    )
    legacy_module_name = "openfold3.projects.of3_all_atom.physppi.data.prior_complex"
    legacy_module = types.ModuleType(legacy_module_name)
    legacy_module.PriorSamplePaths = PriorSamplePaths
    monkeypatch.setitem(sys.modules, legacy_module_name, legacy_module)
    monkeypatch.setattr(PriorSamplePaths, "__module__", legacy_module_name)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {"version": 1, "metadata": metadata, "index": {"fixture": paths}},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    monkeypatch.setattr(
        PriorSamplePaths,
        "__module__",
        "openfold3.projects.of3_all_atom.physppi.data.priors.schema",
    )
    monkeypatch.delitem(sys.modules, legacy_module_name)

    def forbidden_builder():
        raise AssertionError("compatible legacy cache must not be rebuilt")

    loaded = load_or_create_reader_index(
        cache_path,
        metadata,
        forbidden_builder,
    )

    assert loaded == {"fixture": paths}
    assert type(loaded["fixture"]).__module__.endswith(".data.priors.schema")


def test_reader_index_cache_path_is_configurable(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "qbiolip.pkl"

    settings = QBioLiP10kSettings(root=tmp_path, index_cache_file=cache_path)

    assert settings.index_cache_file == cache_path


def test_qbiolip_reader_reuses_persistent_index(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "qbiolip"
    (root / "chunk").mkdir(parents=True)
    cache_path = tmp_path / "cache" / "qbiolip.pkl"
    calls = 0

    def fake_index(self) -> dict:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(QBioLiP10kReader, "_index", fake_index)

    QBioLiP10kReader(root, index_cache_file=cache_path)
    QBioLiP10kReader(root, index_cache_file=cache_path)

    assert calls == 1


def test_afdb_reader_reuses_persistent_index(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "afdb"
    (root / "chunk").mkdir(parents=True)
    cache_path = tmp_path / "cache" / "afdb.pkl"
    calls = 0

    def fake_index(self) -> None:
        nonlocal calls
        calls += 1
        self._paths = {}
        self._chain_sequences = {}

    monkeypatch.setattr(AFDB10kReader, "_index", fake_index)

    AFDB10kReader(root, index_cache_file=cache_path)
    AFDB10kReader(root, index_cache_file=cache_path)

    assert calls == 1
