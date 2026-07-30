import pytest

from openfold3.core.data.framework.single_datasets.abstract_single import (
    DATASET_REGISTRY,
)
from openfold3.core.data.framework.single_datasets.afdb10k import AFDB10kDataset
from openfold3.core.data.framework.single_datasets.prior_complex import (
    PhysPPIPriorComplexDatasetAdapter,
)


def test_afdb_dataset_is_registered():
    assert DATASET_REGISTRY["AFDB10kDataset"] is AFDB10kDataset


def test_afdb_dataset_is_thin_shared_prior_complex_adapter():
    assert issubclass(AFDB10kDataset, PhysPPIPriorComplexDatasetAdapter)
    assert "reader" in AFDB10kDataset.__init__.__code__.co_names
    assert "structure_data" not in AFDB10kDataset.__getitem__.__code__.co_names
    assert "build_sample_features" in AFDB10kDataset.__getitem__.__code__.co_names


def test_afdb_dataset_does_not_limit_reader_index(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from openfold3.core.data.framework.single_datasets import afdb10k as module

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

    monkeypatch.setattr(module, "AFDB10kReader", fake_reader)
    monkeypatch.setattr(
        PhysPPIPriorComplexDatasetAdapter,
        "__init__",
        fake_adapter_init,
    )
    monkeypatch.setattr(
        AFDB10kDataset,
        "create_datapoint_cache_from_reader",
        lambda self, reader, physppi: None,
    )

    cache_path = tmp_path / "afdb-index.pkl"
    config = SimpleNamespace(
        physppi=SimpleNamespace(
            root=tmp_path,
            max_samples=256,
            index_cache_file=cache_path,
        )
    )

    dataset = AFDB10kDataset(config)

    assert reader_args == {
        "root": tmp_path,
        "max_samples": None,
        "index_cache_file": cache_path,
    }
    assert dataset.physppi.max_samples == 256


def test_afdb_dataset_error_identifies_sample_id():
    from types import SimpleNamespace

    import pandas as pd
    from biotite import InvalidFileError

    dataset = object.__new__(AFDB10kDataset)
    dataset.datapoint_cache = pd.DataFrame(
        {
            "pdb_id": ["bad-afdb-sample"],
            "preferred_chain_or_interface": [None],
        }
    )
    dataset.reader = SimpleNamespace(
        read=lambda sample_id: SimpleNamespace(sample_id=sample_id)
    )

    def fail_build(sample, datapoint):
        raise InvalidFileError("broken AFDB sample")

    dataset.build_sample_features = fail_build

    with pytest.raises(RuntimeError, match="bad-afdb-sample") as exc_info:
        dataset[0]

    assert isinstance(exc_info.value.__cause__, InvalidFileError)
