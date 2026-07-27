"""Stable PhysPPI data facade contracts."""

from importlib import import_module

import numpy as np


def test_data_facade_exports_runtime_readers_schema_and_adapter() -> None:
    import_module("openfold3.core.data.framework")
    data = import_module("openfold3.projects.of3_all_atom.physppi.data")

    for symbol in (
        "AFDB10kReader",
        "PhysPPIPriorComplexDatasetAdapter",
        "PriorComplexSample",
        "QBioLiP10kReader",
    ):
        assert getattr(data, symbol).__name__ == symbol


def test_source_mixture_sampler_is_deterministic_and_weighted() -> None:
    import_module("openfold3.core.data.framework")
    adapters = import_module("openfold3.projects.of3_all_atom.physppi.data.adapters")

    actual = adapters.deterministic_source_mixture_indices(
        labels=["qbiolip10k"] * 5 + ["afdb10k"] * 3,
        epoch_len=16,
        fractions={"qbiolip10k": 0.8, "afdb10k": 0.2},
    )

    assert isinstance(actual, list)
    np.testing.assert_array_equal(
        actual,
        adapters.deterministic_source_mixture_indices(
            labels=["qbiolip10k"] * 5 + ["afdb10k"] * 3,
            epoch_len=16,
            fractions={"qbiolip10k": 0.8, "afdb10k": 0.2},
        ),
    )
