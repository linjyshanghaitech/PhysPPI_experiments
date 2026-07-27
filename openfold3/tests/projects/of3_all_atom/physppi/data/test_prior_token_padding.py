import importlib

import numpy as np


def _structure_adapter_module():
    importlib.import_module(
        "openfold3.core.data.framework.single_datasets.prior_complex"
    )
    return importlib.import_module(
        "openfold3.projects.of3_all_atom.physppi.data.adapters"
    )


def test_prior_payloads_pad_to_the_fixed_token_budget():
    module = _structure_adapter_module()
    payloads = {
        "contact": np.array([True, False]),
        "interaction": np.ones((2, 2, 7), dtype=bool),
        "surface": np.ones((2, 128, 6), dtype=np.float32),
    }

    padded = module._pad_prior_payloads_to_token_count(payloads, n_tokens=4)

    assert padded["contact"].shape == (4,)
    assert padded["interaction"].shape == (4, 4, 7)
    assert padded["surface"].shape == (4, 128, 6)
    assert not padded["contact"][2:].any()
    assert not padded["interaction"][2:].any()
    assert not padded["interaction"][:, 2:].any()
    assert np.all(padded["surface"][2:] == 0)


def test_surface_neighbor_axis_is_not_treated_as_pair_axis_at_128_tokens():
    module = _structure_adapter_module()
    payloads = {
        "interaction": np.ones((128, 128, 7), dtype=bool),
        "surface_coords": np.ones((128, 128, 3), dtype=np.float32),
        "surface_features": np.ones((128, 128, 6), dtype=np.float32),
        "surface_mask": np.ones((128, 128), dtype=bool),
    }

    padded = module._pad_prior_payloads_to_token_count(payloads, n_tokens=768)

    assert padded["interaction"].shape == (768, 768, 7)
    assert padded["surface_coords"].shape == (768, 128, 3)
    assert padded["surface_features"].shape == (768, 128, 6)
