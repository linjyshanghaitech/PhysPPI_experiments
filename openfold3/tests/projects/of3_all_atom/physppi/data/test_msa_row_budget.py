"""Regression coverage for the PhysPPI pre-parsed MSA row budget."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from biotite.structure import AtomArray

from openfold3.core.data.framework.single_datasets import (
    prior_complex as _prior_complex,  # noqa: F401
)
from openfold3.core.data.primitives.sequence.msa import MsaArray
from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module


def _msa(rows: int, residue: str) -> MsaArray:
    return MsaArray(
        msa=np.full((rows, 1), residue, dtype="<U1"),
        deletion_matrix=np.zeros((rows, 1), dtype=int),
        metadata=pd.DataFrame(),
    )


def test_preparsed_msas_are_limited_to_the_configured_total_row_budget(monkeypatch):
    atom_array = AtomArray(2)
    atom_array.chain_id = np.asarray(["A", "B"])
    atom_array.res_id = np.asarray([1, 1])
    atom_array.res_name = np.asarray(["ALA", "CYS"])
    atom_array.atom_name = np.asarray(["CA", "CA"])
    sample = SimpleNamespace(
        sample_id="case",
        paths=SimpleNamespace(
            main_msa_npzs={"A": "main-a", "B": "main-b"},
            paired_msa_npzs={"A": "paired-a", "B": "paired-b"},
        ),
    )
    arrays = {
        "main-a": _msa(5, "A"),
        "main-b": _msa(5, "C"),
        "paired-a": _msa(5, "A"),
        "paired-b": _msa(5, "C"),
    }
    monkeypatch.setattr(module, "_read_msa_npz", lambda path: arrays[path])
    adapter = object.__new__(module.PhysPPIPriorComplexDatasetAdapter)
    adapter.n_tokens = 2
    adapter.msa_settings = SimpleNamespace(max_rows=4, max_rows_paired=2)
    captured = {}

    def fake_featurizer(**kwargs):
        captured.update(kwargs)
        return {}

    adapter.msa_featurizer_of3 = fake_featurizer

    adapter.create_canonical_msa_features(sample, atom_array)

    collection = captured["msa_array_collection"]
    assert collection.row_counts.n_rows_paired_subsampled == 2
    assert collection.row_counts.n_rows_total == 4
    assert all(len(msa) == 1 for msa in collection.chain_id_to_main_msa.values())
