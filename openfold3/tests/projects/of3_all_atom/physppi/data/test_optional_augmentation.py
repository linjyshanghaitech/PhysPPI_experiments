from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    PRIOR_OPTIONAL_SLOTS,
    OptionalSlot,
    SlotState,
    build_paired_msa_chain_mask,
    mask_surface_patches,
    mask_tm_segments,
    sample_prior_level,
    select_prior_level,
    subsample_interactions_by_type,
    subsample_true_contacts,
)


def all_available() -> dict[OptionalSlot, bool]:
    return {slot: True for slot in PRIOR_OPTIONAL_SLOTS}


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (0, ()),
        (1, ("contact",)),
        (2, ("contact", "interaction")),
        (3, ("contact", "interaction", "surface")),
        (4, ("contact", "interaction", "surface", "tm")),
    ],
)
def test_requested_level_is_a_cumulative_prefix(requested, expected):
    selection = select_prior_level(all_available(), requested)

    assert selection.requested_level == requested
    assert selection.effective_level == requested
    assert tuple(
        slot.value for slot, state in selection.states.items() if state.active
    ) == expected


def test_missing_interaction_truncates_requested_l4_to_effective_l1():
    available = all_available() | {OptionalSlot.INTERACTION: False}
    selection = select_prior_level(available, 4)

    assert selection.effective_level == 1
    assert selection.states[OptionalSlot.SURFACE].active is False
    assert selection.states[OptionalSlot.TM].active is False


@pytest.mark.parametrize("requested", [-1, 5])
def test_requested_level_must_be_in_l0_through_l4(requested):
    with pytest.raises(ValueError, match=r"\[0, 4\]"):
        select_prior_level(all_available(), requested)


def test_sampler_uniformly_requests_only_cumulative_l0_through_l4_prefixes():
    selections = [
        sample_prior_level(all_available(), np.random.default_rng(seed))
        for seed in range(512)
    ]

    assert {selection.requested_level for selection in selections} == set(range(5))
    assert all(
        selection.effective_level == selection.requested_level
        and tuple(slot for slot, state in selection.states.items() if state.active)
        == PRIOR_OPTIONAL_SLOTS[: selection.effective_level]
        for selection in selections
    )


def test_active_is_available_selected_and_valid():
    for available in (False, True):
        for selected in (False, True):
            for valid in (False, True):
                state = SlotState(available, selected, valid, "test")
                assert state.active == (available and selected and valid)


def test_slot_state_is_immutable():
    state = SlotState(True, True, True, "qbiolip10k")
    with pytest.raises(FrozenInstanceError):
        state.selected = False


def test_subsample_true_contacts_preserves_one_nonempty_contact():
    edges = np.arange(20, dtype=np.int64).reshape(10, 2)
    out = subsample_true_contacts(edges, np.random.default_rng(1))
    assert out.shape[0] >= 1
    assert any(np.array_equal(row, edges[0]) for row in out)
    assert np.array_equal(edges, np.arange(20, dtype=np.int64).reshape(10, 2))


def test_subsample_true_contacts_empty_and_singleton():
    empty = np.empty((0, 2), dtype=np.int64)
    singleton = np.asarray([[3, 8]], dtype=np.int64)
    assert np.array_equal(
        subsample_true_contacts(empty, np.random.default_rng(1)), empty
    )
    assert np.array_equal(
        subsample_true_contacts(singleton, np.random.default_rng(1)), singleton
    )


def test_subsample_interactions_preserves_each_type_and_is_reproducible():
    edges = np.arange(24, dtype=np.int64).reshape(12, 2)
    type_ids = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    out_edges, out_types = subsample_interactions_by_type(
        edges, type_ids, np.random.default_rng(22)
    )
    repeat_edges, repeat_types = subsample_interactions_by_type(
        edges, type_ids, np.random.default_rng(22)
    )
    assert np.array_equal(out_edges, repeat_edges)
    assert np.array_equal(out_types, repeat_types)
    assert len(out_edges) == len(out_types)
    assert set(out_types.tolist()) == set(type_ids.tolist())
    for type_id in np.unique(type_ids):
        assert np.count_nonzero(out_types == type_id) >= 1


def test_subsample_interactions_empty_input():
    edges = np.empty((0, 2), dtype=np.int64)
    type_ids = np.empty((0,), dtype=np.int64)
    out_edges, out_types = subsample_interactions_by_type(
        edges, type_ids, np.random.default_rng(2)
    )
    assert out_edges.shape == edges.shape
    assert out_types.shape == type_ids.shape


@pytest.mark.parametrize("mask_fn", [mask_surface_patches, mask_tm_segments])
def test_mask_helpers_are_reproducible_without_mutating_input(mask_fn):
    mask = np.ones((2, 10), dtype=bool)
    first = mask_fn(mask, np.random.default_rng(42))
    second = mask_fn(mask, np.random.default_rng(42))
    assert np.array_equal(first, second)
    assert first.shape == mask.shape
    assert np.array_equal(mask, np.ones((2, 10), dtype=bool))


def test_paired_msa_is_not_an_optional_slot_and_chain_mask_is_explicit():
    assert "paired_msa" not in {slot.value for slot in PRIOR_OPTIONAL_SLOTS}
    assert np.array_equal(
        build_paired_msa_chain_mask([True, False, True]),
        np.asarray([True, False, True]),
    )


def test_paired_msa_chain_mask_rejects_non_vector_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        build_paired_msa_chain_mask([[True, False]])
