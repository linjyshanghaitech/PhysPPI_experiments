from openfold3.projects.of3_all_atom.physppi.data.priors.occurrence import (
    build_occurrence_rng,
)


def test_occurrence_rng_changes_slot_but_replays_exactly() -> None:
    first = build_occurrence_rng(42, 3, 19, "qbiolip", "1abc_1")
    replay = build_occurrence_rng(42, 3, 19, "qbiolip", "1abc_1")
    other = build_occurrence_rng(42, 3, 20, "qbiolip", "1abc_1")

    assert first.seed == replay.seed
    assert first.contact.uniform(8.0, 18.0) == replay.contact.uniform(8.0, 18.0)
    assert first.seed != other.seed
