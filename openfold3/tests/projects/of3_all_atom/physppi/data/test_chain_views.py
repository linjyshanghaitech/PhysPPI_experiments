from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from openfold3.projects.of3_all_atom.config.dataset_configs import (
    QBioLiP10kSettings,
)
from openfold3.projects.of3_all_atom.physppi.data.adapters import chain_views
from openfold3.projects.of3_all_atom.physppi.data.adapters.chain_views import (
    RetainedChainView,
    build_chain_graph,
    needs_resource_view,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    PriorComplexSample,
    PriorSamplePaths,
    iter_chain_pair_distances,
)
from openfold3.projects.of3_all_atom.physppi.features import PhysPPIDataError


def test_resource_view_trigger_is_exact() -> None:
    assert not needs_resource_view(
        chain_count=4,
        token_count=900,
        max_compatibility_group=30,
    )
    assert not needs_resource_view(
        chain_count=8,
        token_count=768,
        max_compatibility_group=20,
    )
    assert needs_resource_view(
        chain_count=8,
        token_count=769,
        max_compatibility_group=20,
    )
    assert needs_resource_view(
        chain_count=8,
        token_count=768,
        max_compatibility_group=21,
    )


def test_chain_graph_uses_exact_18_angstrom_threshold() -> None:
    graph = build_chain_graph(
        ("C", "A", "B"),
        {
            "A|B": ((1, 1, 18.0),),
            "A|C": ((1, 1, 18.0001),),
        },
    )

    assert tuple(graph) == ("A", "B", "C")
    assert graph["A"] == {"B"}
    assert graph["B"] == {"A"}
    assert graph["C"] == set()


def test_chain_graph_ignores_nonfinite_distances() -> None:
    graph = build_chain_graph(
        ("A", "B", "C"),
        {
            "A|B": ((1, 1, float("-inf")),),
            "A|C": ((1, 1, float("nan")),),
            "B|C": ((1, 1, 17.0),),
        },
    )

    assert graph == {
        "A": set(),
        "B": {"C"},
        "C": {"B"},
    }


def test_prior_sample_exposes_stable_chain_inventory_and_dmap_edges(
    tmp_path: Path,
) -> None:
    paths = PriorSamplePaths(
        sample_id="sample",
        gt_pdb=tmp_path / "gt.pdb",
        monomer_pdbs={"B": tmp_path / "B.pdb", "A": tmp_path / "A.pdb"},
        main_msa_npzs={},
        paired_msa_npzs={},
        prior_pkl=tmp_path / "prior.pkl",
    )
    sample = PriorComplexSample(
        sample_id="sample",
        dmap={
            "B|C": ((3, 4, 12.0),),
            "A|B": ((2, 1, 18.0), (1, 1, 9.0)),
        },
        interactions=(),
        mesh={"D": ()},
        opm={},
        paths=paths,
        chain_sequences={"C": "AA"},
    )

    assert sample.chain_ids == ("A", "B", "C", "D")
    assert tuple(iter_chain_pair_distances(sample.dmap)) == (
        ("A", "B", 9.0),
        ("A", "B", 18.0),
        ("B", "C", 12.0),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("resource_view_max_chains", 5),
        ("resource_view_token_trigger", 769),
        ("resource_view_compatibility_trigger", 21),
        ("resource_view_dmap_cutoff", 18.0001),
    ),
)
def test_resource_view_settings_are_canonical(
    tmp_path: Path,
    field: str,
    value: int | float,
) -> None:
    settings = QBioLiP10kSettings(root=tmp_path)
    assert settings.resource_view_max_chains == 4
    assert settings.resource_view_token_trigger == 768
    assert settings.resource_view_compatibility_trigger == 20
    assert settings.resource_view_dmap_cutoff == 18.0

    with pytest.raises(ValidationError, match=field):
        QBioLiP10kSettings(root=tmp_path, **{field: value})


def test_retained_chain_view_carries_complete_provenance() -> None:
    view = RetainedChainView(
        source_chain_ids=("A", "B"),
        trigger="token_budget",
        component_chain_ids=("A", "B", "C"),
        seed_edge=("A", "B"),
        target_chain_count=2,
        parent_sample_id="parent",
        original_chain_ids=("A", "B", "C", "D", "E"),
        dmap_cutoff=18.0,
        occurrence_seed=42,
        matcher_result="pending",
    )

    assert view.parent_sample_id == "parent"
    assert view.original_chain_ids == ("A", "B", "C", "D", "E")
    assert view.source_chain_ids == ("A", "B")
    assert view.trigger == "token_budget"
    assert view.dmap_cutoff == 18.0
    assert view.occurrence_seed == 42
    assert view.matcher_result == "pending"


def _induced_subgraph_is_connected(
    graph: dict[str, set[str]],
    chain_ids: tuple[str, ...],
) -> bool:
    remaining = set(chain_ids)
    frontier = {remaining.pop()}
    while frontier:
        chain_id = frontier.pop()
        neighbors = graph[chain_id] & remaining
        remaining -= neighbors
        frontier |= neighbors
    return not remaining


def test_connected_view_is_two_to_four_chains_and_replays() -> None:
    graph = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B", "D"},
        "D": {"C"},
    }

    first = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(9),
        "token_budget",
        parent_sample_id="parent",
        occurrence_seed=99,
    )
    replay = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(9),
        "token_budget",
        parent_sample_id="parent",
        occurrence_seed=99,
    )

    assert first == replay
    assert 2 <= len(first.source_chain_ids) <= 4
    assert _induced_subgraph_is_connected(graph, first.source_chain_ids)
    assert first.parent_sample_id == "parent"
    assert first.original_chain_ids == ("A", "B", "C", "D")
    assert first.dmap_cutoff == 18.0
    assert first.occurrence_seed == 99
    assert first.matcher_result == "pending"


def test_retained_view_prefers_complete_four_chain_candidate() -> None:
    graph = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B", "D"},
        "D": {"C"},
    }

    view = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(1),
        "token_budget",
        chain_lengths={"A": 100, "B": 120, "C": 130, "D": 140},
    )

    assert view.source_chain_ids == ("A", "B", "C", "D")
    assert view.target_chain_count == 4
    assert view.component_chain_ids == ("A", "B", "C", "D")
    assert view.seed_edge is not None


def test_retained_view_rejects_oversized_four_and_prefers_valid_three() -> None:
    graph = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B", "D"},
        "D": {"C"},
    }

    view = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(3),
        "token_budget",
        chain_lengths={"A": 250, "B": 250, "C": 250, "D": 250},
    )

    assert len(view.source_chain_ids) == 3
    assert _induced_subgraph_is_connected(graph, view.source_chain_ids)
    assert sum({"A": 250, "B": 250, "C": 250, "D": 250}[chain] for chain in view.source_chain_ids) <= 768


def test_retained_view_uses_pair_when_no_valid_three_or_four() -> None:
    graph = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B"},
    }

    view = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(4),
        "token_budget",
        chain_lengths={"A": 500, "B": 500, "C": 500},
    )

    assert len(view.source_chain_ids) == 2
    assert _induced_subgraph_is_connected(graph, view.source_chain_ids)


def test_retained_view_raises_without_connected_pair() -> None:
    graph = {
        "A": set(),
        "B": set(),
        "C": set(),
        "D": set(),
        "E": set(),
    }

    with pytest.raises(ValueError, match="connected dmap pair"):
        chain_views.sample_retained_chain_view(
            graph,
            np.random.default_rng(5),
            "token_budget",
            chain_lengths={chain: 100 for chain in graph},
        )


def test_retained_view_candidate_cap_is_deterministic() -> None:
    graph = {
        chain: {other for other in "ABCDEFG" if other != chain}
        for chain in "ABCDEFG"
    }
    chain_lengths = {chain: 10 for chain in graph}

    first = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(6),
        "token_budget",
        chain_lengths=chain_lengths,
        candidate_cap=5,
    )
    replay = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(6),
        "token_budget",
        chain_lengths=chain_lengths,
        candidate_cap=5,
    )

    assert first == replay
    assert len(first.source_chain_ids) == 4


def test_sampling_is_independent_of_graph_insertion_order() -> None:
    ordered = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A", "D"},
        "D": {"B", "C"},
    }
    reversed_order = {
        "D": {"C", "B"},
        "C": {"D", "A"},
        "B": {"D", "A"},
        "A": {"C", "B"},
    }

    first = chain_views.sample_retained_chain_view(
        ordered,
        np.random.default_rng(23),
        "compatibility_group",
    )
    replay = chain_views.sample_retained_chain_view(
        reversed_order,
        np.random.default_rng(23),
        "compatibility_group",
    )

    assert first == replay


class _ScriptedRNG:
    def __init__(self, draws: list[int]) -> None:
        self.draws = iter(draws)
        self.bounds: list[tuple[int, int]] = []

    def integers(self, low: int, high: int) -> int:
        self.bounds.append((low, high))
        return next(self.draws)


def test_edge_size_and_frontier_draws_use_complete_sorted_candidate_lists() -> None:
    graph = {
        "D": {"C"},
        "C": {"D", "B"},
        "B": {"C", "A"},
        "A": {"B"},
    }

    view = chain_views.sample_retained_chain_view(
        graph,
        np.random.default_rng(7),
        "token_budget",
        chain_lengths={chain: 100 for chain in graph},
    )

    assert view.source_chain_ids == ("A", "B", "C", "D")
    assert view.seed_edge in {("A", "B"), ("B", "C"), ("C", "D")}


def test_isolated_chains_raise_instead_of_first_four_fallback() -> None:
    graph = {
        "E": set(),
        "D": set(),
        "C": set(),
        "B": set(),
        "A": set(),
    }

    with pytest.raises(PhysPPIDataError, match="missing_connected_dmap_pair"):
        chain_views.sample_retained_chain_view(
            graph,
            np.random.default_rng(3),
            "token_budget",
            parent_sample_id="parent",
            occurrence_seed=3,
        )
