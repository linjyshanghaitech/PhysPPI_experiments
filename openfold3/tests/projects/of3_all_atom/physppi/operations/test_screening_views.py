from __future__ import annotations

import pytest

from openfold3.projects.of3_all_atom.physppi.data.priors.occurrence import (
    build_occurrence_rng,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    AuditView,
    DatasetName,
)
from openfold3.projects.of3_all_atom.physppi.screening.views import (
    AUDIT_DATA_SEED,
    AUDIT_EPOCH,
    audit_global_slot,
    audit_views,
)

EXPECTED_MATRIX = (
    ("spatial", 0, 8.0, 0, False),
    ("spatial_interface", 1, 18.0, 1, False),
    ("spatial", 2, 13.0, 2, False),
    ("spatial_interface", 3, 8.0, 3, False),
    ("spatial", 4, 18.0, 4, False),
    ("spatial_interface", 4, 13.0, 5, True),
    ("spatial", 4, 13.0, 6, True),
)


def _requested_matrix(views: tuple[AuditView, ...]) -> tuple[tuple, ...]:
    return tuple(
        (
            view.crop_strategy,
            view.requested_level,
            view.contact_cutoff,
            view.view_index,
            view.multichain_probe,
        )
        for view in views
    )


def test_view_matrix_is_the_fixed_seven_view_contract() -> None:
    views = audit_views(
        DatasetName.QBIOLIP,
        "sample",
        source_ordinal=19,
    )

    assert len(views) == 7
    assert all(isinstance(view, AuditView) for view in views)
    assert _requested_matrix(views) == EXPECTED_MATRIX
    assert {view.requested_level for view in views} == {0, 1, 2, 3, 4}
    assert {view.crop_strategy for view in views} == {
        "spatial",
        "spatial_interface",
    }
    assert {view.contact_cutoff for view in views} == {8.0, 13.0, 18.0}
    assert sum(view.multichain_probe for view in views) == 2


def test_afdb_uses_the_same_requested_matrix_as_qbiolip() -> None:
    qbiolip = audit_views("qbiolip", "q-sample", source_ordinal=27)
    afdb = audit_views("afdb", "af-sample", source_ordinal=27)

    assert _requested_matrix(qbiolip) == EXPECTED_MATRIX
    assert _requested_matrix(afdb) == EXPECTED_MATRIX
    assert not hasattr(afdb[0], "effective_level")


def test_each_view_uses_epoch_zero_and_the_approved_occurrence_seed() -> None:
    source_ordinal = 31
    views = audit_views("qbiolip", "sample-uid", source_ordinal=source_ordinal)

    for view in views:
        expected_global_slot = audit_global_slot(source_ordinal, view.view_index)
        expected = build_occurrence_rng(
            AUDIT_DATA_SEED,
            AUDIT_EPOCH,
            expected_global_slot,
            "qbiolip",
            "sample-uid",
        )
        assert view.audit_epoch == 0
        assert view.global_slot == expected_global_slot
        assert view.occurrence_seed == expected.seed


def test_view_seed_replays_and_changes_with_source_ordinal() -> None:
    first = audit_views("afdb", "sample", source_ordinal=4)
    replay = audit_views("afdb", "sample", source_ordinal=4)
    later = audit_views("afdb", "sample", source_ordinal=5)

    assert first == replay
    assert [view.global_slot for view in first] == list(range(28, 35))
    assert {view.occurrence_seed for view in first}.isdisjoint(
        {view.occurrence_seed for view in later}
    )


@pytest.mark.parametrize("source_ordinal", [-1, -7])
def test_global_slot_rejects_negative_source_ordinals(source_ordinal: int) -> None:
    with pytest.raises(ValueError, match="source_ordinal must be nonnegative"):
        audit_views("qbiolip", "sample", source_ordinal=source_ordinal)
