import numpy as np
import pytest
import torch

from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    PRIOR_OPTIONAL_SLOTS,
    OptionalSlot,
    build_paired_msa_chain_mask,
    sample_prior_level,
)
from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
    validate_optional_prior_features,
)


def test_paired_msa_never_enters_optional_sampling():
    states = sample_prior_level(
        {slot: True for slot in PRIOR_OPTIONAL_SLOTS}, np.random.default_rng(0)
    ).states
    assert "paired_msa" not in {slot.value for slot in states}
    assert OptionalSlot.CONTACT in states


def test_partial_paired_msa_availability_is_mask_only():
    mask = build_paired_msa_chain_mask([True, False])
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False]


def test_sample_level_paired_msa_marks_shared_chains_available(monkeypatch):
    """AFDB sample-level paired files are shared across its homodimer chains."""

    from types import SimpleNamespace

    import numpy as np
    from biotite.structure import AtomArray

    # Complete the framework's registered single-dataset imports before
    # importing the adapter directly; this avoids the registry cycle.
    import openfold3.core.data.framework.single_datasets.afdb10k  # noqa: F401
    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module

    atom_array = AtomArray(2)
    atom_array.chain_id = np.asarray(["A", "B"])
    atom_array.res_id = np.asarray([1, 1])
    atom_array.res_name = np.asarray(["ALA", "ALA"])
    atom_array.atom_name = np.asarray(["CA", "CA"])
    atom_array.token_id = np.asarray([0, 1])
    sample = SimpleNamespace(
        sample_id="afdb-1",
        paths=SimpleNamespace(
            main_msa_npzs={"A": object(), "B": object()},
            paired_msa_npzs={"afdb-1": object()},
        ),
        dataset_name="AFDB",
        chain_sequences={"A": "A", "B": "A"},
    )
    adapter = object.__new__(module.PhysPPIPriorComplexDatasetAdapter)
    adapter.n_tokens = 2
    adapter.msa_featurizer_of3 = lambda **_: {}
    monkeypatch.setattr(module, "_read_msa_npz", lambda _: module._query_only_msa("A"))

    result = adapter.create_canonical_msa_features(sample, atom_array)
    assert result["physppi_paired_msa_chain_mask"].tolist() == [True, True]


def test_missing_chain_main_msa_falls_back_to_its_query_row(monkeypatch):
    from types import SimpleNamespace

    import openfold3.core.data.framework.single_datasets.afdb10k  # noqa: F401
    from openfold3.projects.of3_all_atom.physppi.data.adapters import _runtime as module

    query_msas = {
        "A": module._query_only_msa("AC"),
        "B": module._query_only_msa("DE"),
    }
    sample = SimpleNamespace(
        paths=SimpleNamespace(main_msa_npzs={"A": object()}),
        dataset_name="Q-BioLiP",
        chain_sequences={"A": "AC", "B": "DE"},
    )
    monkeypatch.setattr(module, "_read_msa_npz", lambda _: query_msas["A"])

    main_msas, provenance = module._build_main_msas(
        sample, ["A", "B"], query_msas
    )

    assert provenance == {"A": "chain_file", "B": "query_only"}
    assert np.array_equal(main_msas["B"].msa, query_msas["B"].msa)


def test_paired_msa_chain_mask_is_not_an_active_optional_payload():
    batch = {
        "token_mask": torch.ones(1, 2, dtype=torch.bool),
        "physppi_paired_msa_chain_mask": torch.tensor([[True, False]]),
    }
    validate_optional_prior_features(batch, mode="train")


def test_paired_msa_chain_mask_rejects_rank_three_payload():
    batch = {
        "token_mask": torch.ones(1, 2, dtype=torch.bool),
        "physppi_paired_msa_chain_mask": torch.ones(1, 2, 1, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="shape"):
        validate_optional_prior_features(batch, mode="train")
