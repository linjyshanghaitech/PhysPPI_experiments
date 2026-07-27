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
