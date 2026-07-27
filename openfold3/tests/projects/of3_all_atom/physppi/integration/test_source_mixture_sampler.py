from openfold3.tests.projects.of3_all_atom.physppi.data.test_msa_structure_adapter_contracts import (
    _reload_adapter,
)


def test_deterministic_source_mixture_sampler_uses_labels_for_80_20_split(
    monkeypatch,
) -> None:
    module = _reload_adapter(monkeypatch)
    labels = ["qbiolip10k"] * 5 + ["afdb10k"] * 5

    indices = module.deterministic_source_mixture_indices(labels, epoch_len=10)

    assert indices == [0, 1, 2, 3, 4, 0, 1, 2, 5, 6]
    assert [labels[index] for index in indices].count("qbiolip10k") == 8
    assert [labels[index] for index in indices].count("afdb10k") == 2
