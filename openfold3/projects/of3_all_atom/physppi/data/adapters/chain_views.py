"""Deterministic bounded chain views for resource-heavy PhysPPI samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    iter_chain_pair_distances,
)
from openfold3.projects.of3_all_atom.physppi.features import PhysPPIDataError

CHAIN_VIEW_MAX_CHAINS = 4
CHAIN_VIEW_TOKEN_TRIGGER = 768
CHAIN_VIEW_COMPATIBILITY_TRIGGER = 20
CHAIN_VIEW_DMAP_CUTOFF = 18.0
CHAIN_VIEW_CANDIDATE_CAP = 1024

ChainGraph: TypeAlias = dict[str, set[str]]


@dataclass(frozen=True)
class RetainedChainView:
    """One retained chain view plus replayable selection provenance."""

    source_chain_ids: tuple[str, ...]
    trigger: str
    component_chain_ids: tuple[str, ...]
    seed_edge: tuple[str, str] | None
    target_chain_count: int
    parent_sample_id: str = ""
    original_chain_ids: tuple[str, ...] = ()
    dmap_cutoff: float = CHAIN_VIEW_DMAP_CUTOFF
    occurrence_seed: int = 0
    matcher_result: str = "pending"


def needs_resource_view(
    chain_count: int,
    token_count: int,
    max_compatibility_group: int,
) -> bool:
    """Return whether the exact canonical resource-view trigger is active."""

    return chain_count > CHAIN_VIEW_MAX_CHAINS and (
        token_count > CHAIN_VIEW_TOKEN_TRIGGER
        or max_compatibility_group > CHAIN_VIEW_COMPATIBILITY_TRIGGER
    )


def build_chain_graph(
    chain_ids: tuple[str, ...],
    dmap: dict[str, tuple[tuple[int, int, float], ...]],
) -> ChainGraph:
    """Build the sorted undirected chain graph at the exact 18 Å cutoff."""

    graph = {chain_id: set() for chain_id in sorted(set(chain_ids))}
    for left, right, distance in iter_chain_pair_distances(dmap):
        if left not in graph or right not in graph:
            raise ValueError(
                f"dmap chain pair {left}|{right} is absent from the chain inventory"
            )
        if (
            left != right
            and np.isfinite(distance)
            and distance <= CHAIN_VIEW_DMAP_CUTOFF
        ):
            graph[left].add(right)
            graph[right].add(left)
    return graph


def _connected_component(graph: ChainGraph, start: str) -> tuple[str, ...]:
    component = {start}
    frontier = [start]
    while frontier:
        chain_id = frontier.pop()
        for neighbor in sorted(graph[chain_id]):
            if neighbor not in component:
                component.add(neighbor)
                frontier.append(neighbor)
    return tuple(sorted(component))


def _sorted_edges(graph: ChainGraph) -> list[tuple[str, str]]:
    return sorted(
        (left, right) for left in graph for right in graph[left] if left < right
    )


def _cap_candidates(
    candidates: list[tuple[str, ...]],
    rng: np.random.Generator,
    candidate_cap: int,
) -> list[tuple[str, ...]]:
    if candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive")
    candidates = sorted(dict.fromkeys(candidates))
    if len(candidates) <= candidate_cap:
        return candidates
    selected = sorted(
        int(index)
        for index in rng.choice(len(candidates), size=candidate_cap, replace=False)
    )
    return [candidates[index] for index in selected]


def _expand_connected_candidates(
    graph: ChainGraph,
    seeds: list[tuple[str, ...]],
    rng: np.random.Generator,
    candidate_cap: int,
) -> list[tuple[str, ...]]:
    expanded: list[tuple[str, ...]] = []
    for candidate in seeds:
        selected = set(candidate)
        frontier = sorted(
            {
                neighbor
                for chain_id in selected
                for neighbor in graph[chain_id]
                if neighbor not in selected
            }
        )
        for chain_id in frontier:
            expanded.append(tuple(sorted((*candidate, chain_id))))
    return _cap_candidates(expanded, rng, candidate_cap)


def _length_filtered_candidates(
    candidates: list[tuple[str, ...]],
    chain_lengths: dict[str, int],
    token_budget: int,
) -> list[tuple[str, ...]]:
    return [
        candidate
        for candidate in candidates
        if sum(int(chain_lengths.get(chain_id, 0)) for chain_id in candidate)
        <= token_budget
    ]


def _candidate_seed_edge(
    graph: ChainGraph,
    candidate: tuple[str, ...],
) -> tuple[str, str]:
    retained = set(candidate)
    edges = sorted(
        (left, right)
        for left in retained
        for right in graph[left]
        if left < right and right in retained
    )
    if not edges:
        raise RuntimeError("connected candidate has no internal edge")
    return edges[0]


def sample_retained_chain_view(
    graph: ChainGraph,
    rng: np.random.Generator,
    trigger: str,
    *,
    chain_lengths: dict[str, int] | None = None,
    token_budget: int = CHAIN_VIEW_TOKEN_TRIGGER,
    candidate_cap: int = CHAIN_VIEW_CANDIDATE_CAP,
    parent_sample_id: str = "",
    occurrence_seed: int = 0,
    matcher_result: str = "pending",
) -> RetainedChainView:
    """Sample one deterministic connected view from prioritized candidates."""

    original_chain_ids = tuple(sorted(graph))
    if not original_chain_ids:
        raise ValueError("cannot sample a retained-chain view from an empty graph")

    edges = _cap_candidates(_sorted_edges(graph), rng, candidate_cap)
    if not edges:
        raise PhysPPIDataError(
            "missing_connected_dmap_pair",
            "resource-triggered retained-chain view has no connected dmap pair",
        )

    chain_lengths = chain_lengths or {}
    triples = _length_filtered_candidates(
        _expand_connected_candidates(graph, edges, rng, candidate_cap),
        chain_lengths,
        token_budget,
    )
    triples = _cap_candidates(triples, rng, candidate_cap) if triples else []
    quads = _length_filtered_candidates(
        _expand_connected_candidates(graph, triples, rng, candidate_cap)
        if triples
        else [],
        chain_lengths,
        token_budget,
    )
    quads = _cap_candidates(quads, rng, candidate_cap) if quads else []

    if quads:
        candidates = quads
    elif triples:
        candidates = triples
    else:
        candidates = edges

    selected_index = int(rng.integers(0, len(candidates)))
    selected = candidates[selected_index]
    seed_edge = _candidate_seed_edge(graph, selected)
    component = _connected_component(graph, seed_edge[0])

    return RetainedChainView(
        source_chain_ids=selected,
        trigger=trigger,
        component_chain_ids=component,
        seed_edge=seed_edge,
        target_chain_count=len(selected),
        parent_sample_id=parent_sample_id,
        original_chain_ids=original_chain_ids,
        occurrence_seed=occurrence_seed,
        matcher_result=matcher_result,
    )


__all__ = [
    "CHAIN_VIEW_COMPATIBILITY_TRIGGER",
    "CHAIN_VIEW_CANDIDATE_CAP",
    "CHAIN_VIEW_DMAP_CUTOFF",
    "CHAIN_VIEW_MAX_CHAINS",
    "CHAIN_VIEW_TOKEN_TRIGGER",
    "ChainGraph",
    "RetainedChainView",
    "build_chain_graph",
    "needs_resource_view",
    "sample_retained_chain_view",
]
