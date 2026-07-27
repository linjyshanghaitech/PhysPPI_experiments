"""Deterministic bounded chain views for resource-heavy PhysPPI samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    iter_chain_pair_distances,
)

CHAIN_VIEW_MAX_CHAINS = 4
CHAIN_VIEW_TOKEN_TRIGGER = 768
CHAIN_VIEW_COMPATIBILITY_TRIGGER = 20
CHAIN_VIEW_DMAP_CUTOFF = 18.0

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


def sample_retained_chain_view(
    graph: ChainGraph,
    rng: np.random.Generator,
    trigger: str,
    *,
    parent_sample_id: str = "",
    occurrence_seed: int = 0,
    matcher_result: str = "pending",
) -> RetainedChainView:
    """Sample one deterministic connected view from sorted candidate lists."""

    original_chain_ids = tuple(sorted(graph))
    if not original_chain_ids:
        raise ValueError("cannot sample a retained-chain view from an empty graph")

    edges = sorted(
        (left, right) for left in graph for right in graph[left] if left < right
    )
    if not edges:
        kept = original_chain_ids[:CHAIN_VIEW_MAX_CHAINS]
        return RetainedChainView(
            source_chain_ids=kept,
            trigger=trigger,
            component_chain_ids=kept,
            seed_edge=None,
            target_chain_count=len(kept),
            parent_sample_id=parent_sample_id,
            original_chain_ids=original_chain_ids,
            occurrence_seed=occurrence_seed,
            matcher_result=matcher_result,
        )

    seed_edge = edges[int(rng.integers(0, len(edges)))]
    component = _connected_component(graph, seed_edge[0])
    max_target_size = min(CHAIN_VIEW_MAX_CHAINS, len(component))
    target_chain_count = int(rng.integers(2, max_target_size + 1))

    selected = set(seed_edge)
    while len(selected) < target_chain_count:
        frontier = sorted(
            {
                neighbor
                for chain_id in selected
                for neighbor in graph[chain_id]
                if neighbor not in selected
            }
        )
        if not frontier:
            raise RuntimeError("connected component exhausted before target size")
        selected.add(frontier[int(rng.integers(0, len(frontier)))])

    return RetainedChainView(
        source_chain_ids=tuple(sorted(selected)),
        trigger=trigger,
        component_chain_ids=component,
        seed_edge=seed_edge,
        target_chain_count=target_chain_count,
        parent_sample_id=parent_sample_id,
        original_chain_ids=original_chain_ids,
        occurrence_seed=occurrence_seed,
        matcher_result=matcher_result,
    )


__all__ = [
    "CHAIN_VIEW_COMPATIBILITY_TRIGGER",
    "CHAIN_VIEW_DMAP_CUTOFF",
    "CHAIN_VIEW_MAX_CHAINS",
    "CHAIN_VIEW_TOKEN_TRIGGER",
    "ChainGraph",
    "RetainedChainView",
    "build_chain_graph",
    "needs_resource_view",
    "sample_retained_chain_view",
]
