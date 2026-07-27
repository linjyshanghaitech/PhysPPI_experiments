# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from openfold3.core.utils.protected_chain_matching import (
    ChainRecord,
    InsufficientRepresentativeAnchors,
    NoLegalSupervisedCorrespondence,
    ProtectedChainMatchError,
    _improves_rmsd,
    align_chain_residues,
    chain_record_from_atom_metadata,
    chains_compatible,
    lexicographic_linear_sum_assignment,
    protected_chain_match,
    representative_anchor_rows,
)


def _representative_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )


def _chain_record(
    chain_id: str,
    sequence: tuple[str, ...],
    *,
    polymer_type: str = "protein",
    representative_xyz: torch.Tensor | None = None,
    representative_mask: torch.Tensor | None = None,
) -> ChainRecord:
    if representative_xyz is None:
        representative_xyz = _representative_points()[: len(sequence)].clone()
    if representative_mask is None:
        representative_mask = torch.ones(len(sequence), dtype=torch.bool)
    return ChainRecord(
        chain_id=chain_id,
        polymer_type=polymer_type,
        sequence=sequence,
        representative_xyz=representative_xyz,
        representative_mask=representative_mask,
    )


def test_chain_records_are_immutable():
    record = _chain_record("A", ("ALA", "GLY", "SER", "THR"))

    with pytest.raises(FrozenInstanceError):
        record.chain_id = "B"


@pytest.mark.parametrize(
    ("sequence", "xyz", "mask", "residue_keys", "error"),
    [
        (
            ["ALA"],
            torch.zeros(1, 3),
            torch.ones(1, dtype=torch.bool),
            (),
            "chain_record_sequence",
        ),
        (
            ("ALA",),
            torch.zeros(1, 3, 1),
            torch.ones(1, dtype=torch.bool),
            (),
            "chain_record_representative_xyz_shape",
        ),
        (
            ("ALA",),
            torch.zeros(1, 3),
            torch.ones(1, 1, dtype=torch.bool),
            (),
            "chain_record_representative_mask_shape",
        ),
        (
            ("ALA", "GLY"),
            torch.zeros(1, 3),
            torch.ones(2, dtype=torch.bool),
            (),
            "chain_record_length_mismatch",
        ),
        (
            ("ALA", "GLY"),
            torch.zeros(2, 3),
            torch.ones(2, dtype=torch.bool),
            (("A", 1, ""),),
            "chain_record_residue_keys_length",
        ),
        (
            ("ALA",),
            torch.zeros(1, 3),
            torch.ones(1, dtype=torch.bool),
            (("A", 1),),
            "chain_record_residue_key_shape",
        ),
    ],
)
def test_chain_record_validates_lengths_and_dimensions(
    sequence, xyz, mask, residue_keys, error
):
    with pytest.raises(ProtectedChainMatchError, match=error):
        ChainRecord(
            chain_id="A",
            polymer_type="protein",
            sequence=sequence,
            representative_xyz=xyz,
            representative_mask=mask,
            residue_keys=residue_keys,
        )


@pytest.mark.parametrize(
    ("polymer_type", "sequence", "atom_names", "token_centers", "expected_row"),
    [
        ("protein", ("ALA",), ("N", "CA"), (True, False), 1),
        ("rna", ("A",), ("C1'", "C4'"), (True, False), 1),
        ("dna", ("DA",), ("C1'", "C4'"), (True, False), 1),
        ("ligand", ("LIG",), ("C1", "C2"), (False, True), 1),
    ],
)
def test_chain_record_factory_enforces_polymer_representatives(
    polymer_type, sequence, atom_names, token_centers, expected_row
):
    atom_xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    record = chain_record_from_atom_metadata(
        chain_id="A",
        polymer_type=polymer_type,
        sequence=sequence,
        atom_xyz=atom_xyz,
        atom_mask=torch.ones(2, dtype=torch.bool),
        atom_names=atom_names,
        atom_to_token_index=torch.zeros(2, dtype=torch.long),
        token_center_atom_mask=torch.tensor(token_centers),
    )

    assert torch.equal(record.representative_xyz, atom_xyz[expected_row].unsqueeze(0))
    assert torch.equal(record.representative_mask, torch.tensor([True]))


def test_compatibility_requires_same_polymer_type_and_compatible_sequence():
    sequence = ("ALA", "GLY", "SER", "THR")
    protein_a = _chain_record("A", sequence)
    protein_b = _chain_record("B", sequence)
    protein_c = _chain_record("C", ("ALA", "GLY", "SER", "VAL"))
    rna = _chain_record("D", ("A", "G", "C", "U"), polymer_type="rna")

    assert chains_compatible(protein_a, protein_b).allowed
    assert chains_compatible(protein_a, protein_c).reason == "sequence_mismatch"
    assert chains_compatible(protein_a, rna).reason == "polymer_type_mismatch"


def test_missing_residue_gap_does_not_make_same_chain_incompatible():
    points = _representative_points()
    source = _chain_record(
        "A",
        ("ALA", "GLY", "SER", "THR", "VAL"),
        representative_xyz=points,
    )
    target_rows = torch.tensor([0, 1, 3, 4])
    target = _chain_record(
        "B",
        ("ALA", "GLY", "THR", "VAL"),
        representative_xyz=points[target_rows],
    )

    alignment = align_chain_residues(source, target)
    compatibility = chains_compatible(source, target)

    assert not alignment.has_conflicting_known_residue
    assert torch.equal(alignment.aligned_source_rows, target_rows)
    assert torch.equal(alignment.aligned_target_rows, torch.arange(4))
    assert compatibility.allowed
    assert compatibility.common_representative_count == 4
    assert torch.equal(compatibility.aligned_source_rows, alignment.aligned_source_rows)
    assert torch.equal(compatibility.aligned_target_rows, alignment.aligned_target_rows)


def test_cropped_chain_aligns_by_stable_residue_keys_before_sequence_alignment():
    target_sequence = ("ALA", "GLY", "SER", "THR", "VAL", "LEU")
    retained_rows = (1, 3, 4, 5)
    source = ChainRecord(
        chain_id="pred-copy",
        polymer_type="protein",
        sequence=tuple(target_sequence[row] for row in retained_rows),
        representative_xyz=_representative_points()[:4].clone(),
        representative_mask=torch.ones(4, dtype=torch.bool),
        residue_keys=tuple(("pred-copy", row, "") for row in retained_rows),
    )
    target = ChainRecord(
        chain_id="gt-copy",
        polymer_type="protein",
        sequence=target_sequence,
        representative_xyz=torch.cat(
            (_representative_points(), torch.tensor([[2.0, 1.0, 0.0]]))
        ),
        representative_mask=torch.ones(6, dtype=torch.bool),
        residue_keys=tuple(("gt-copy", row, "") for row in range(6)),
    )

    alignment = align_chain_residues(source, target)
    compatibility = chains_compatible(source, target)

    assert not alignment.has_conflicting_known_residue
    assert torch.equal(alignment.aligned_source_rows, torch.arange(4))
    assert torch.equal(
        alignment.aligned_target_rows,
        torch.tensor(retained_rows),
    )
    assert compatibility.allowed
    assert compatibility.common_representative_count == 4


def test_unknown_residues_neither_conflict_nor_support_compatibility():
    source = _chain_record("A", ("ALA", "UNK", "GLY", "SER", "THR"))
    target = _chain_record("B", ("ALA", "VAL", "GLY", "SER", "THR"))

    compatibility = chains_compatible(source, target)

    expected_rows = torch.tensor([0, 2, 3, 4])
    assert compatibility.allowed
    assert compatibility.common_representative_count == 4
    assert torch.equal(compatibility.aligned_source_rows, expected_rows)
    assert torch.equal(compatibility.aligned_target_rows, expected_rows)


def test_unknown_only_alignment_has_no_representative_support():
    source = _chain_record("A", ("UNK",) * 4)
    target = _chain_record("B", ("UNK",) * 4)

    compatibility = chains_compatible(source, target)

    assert not compatibility.allowed
    assert compatibility.reason == "insufficient_common_representatives"
    assert compatibility.common_representative_count == 0


def test_nonfinite_common_representative_is_excluded():
    sequence = ("ALA", "GLY", "SER", "THR")
    source = _chain_record("A", sequence)
    target_xyz = _representative_points()[:4].clone()
    target_xyz[2, 0] = torch.nan
    target = _chain_record("B", sequence, representative_xyz=target_xyz)

    compatibility = chains_compatible(source, target)

    assert not compatibility.allowed
    assert compatibility.reason == "insufficient_common_representatives"
    assert compatibility.common_representative_count == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_common_representative_rows_are_device_safe_on_cuda():
    sequence = ("ALA", "GLY", "SER", "THR")
    points = _representative_points()[:4].cuda()
    mask = torch.ones(4, dtype=torch.bool, device="cuda")
    source = _chain_record(
        "A", sequence, representative_xyz=points, representative_mask=mask
    )
    target = _chain_record(
        "B",
        sequence,
        representative_xyz=points.clone(),
        representative_mask=mask.clone(),
    )

    compatibility = chains_compatible(source, target)

    assert compatibility.allowed
    assert compatibility.aligned_source_rows.device.type == "cuda"
    assert compatibility.aligned_target_rows.device.type == "cuda"


def test_chains_compatible_rejects_collinear_exact_aligned_support():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    sequence = ("ALA", "GLY", "SER", "THR")
    source = _chain_record("A", sequence, representative_xyz=points)
    target = _chain_record("B", sequence, representative_xyz=points.clone())

    compatibility = chains_compatible(source, target)

    assert not compatibility.allowed
    assert compatibility.reason == "degenerate_common_representatives"
    assert compatibility.common_representative_count == 4


def test_chains_compatible_rejects_degenerate_aligned_covariance():
    source_points = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    target_points = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    sequence = ("ALA", "GLY", "SER", "THR")
    source = _chain_record("A", sequence, representative_xyz=source_points)
    target = _chain_record("B", sequence, representative_xyz=target_points)

    compatibility = chains_compatible(source, target)

    assert not compatibility.allowed
    assert compatibility.reason == "degenerate_common_representatives"


@pytest.mark.parametrize(
    "points",
    [
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
    ],
)
def test_anchor_rows_reject_fewer_than_four_or_collinear_points(points):
    record = _chain_record(
        "A",
        ("ALA",) * points.shape[0],
        representative_xyz=points,
    )

    with pytest.raises(InsufficientRepresentativeAnchors, match="A"):
        representative_anchor_rows(record)


def test_anchor_rows_return_only_finite_masked_noncollinear_points():
    points = torch.cat([_representative_points(), torch.full((1, 3), torch.nan)])
    mask = torch.tensor([True, True, True, True, False, True])
    record = _chain_record(
        "A",
        ("ALA",) * 6,
        representative_xyz=points,
        representative_mask=mask,
    )

    anchors = representative_anchor_rows(record)

    assert torch.equal(anchors, _representative_points()[:4])


def _translated_chain(
    chain_id: str,
    sequence: tuple[str, ...],
    translation: tuple[float, float, float],
    *,
    representative_mask: torch.Tensor | None = None,
) -> ChainRecord:
    points = _representative_points()[: len(sequence)].clone()
    points += torch.tensor(translation)
    return _chain_record(
        chain_id,
        sequence,
        representative_xyz=points,
        representative_mask=representative_mask,
    )


def test_hungarian_recovers_swapped_equal_sequence_chains():
    sequence = ("ALA", "GLY", "SER", "THR", "VAL")
    gt = (
        _translated_chain("A", sequence, (0.0, 0.0, 0.0)),
        _translated_chain("B", sequence, (13.0, 3.0, -2.0)),
    )
    pred = (
        _translated_chain("A", sequence, (13.0, 3.0, -2.0)),
        _translated_chain("B", sequence, (0.0, 0.0, 0.0)),
    )

    result = protected_chain_match(pred, gt)

    assert result.mapping == {"A": "B", "B": "A"}
    assert result.accepted
    assert result.identity_rmsd - result.candidate_rmsd > 1e-3


def test_equal_or_worse_candidate_returns_identity():
    sequence = ("ALA", "GLY", "SER", "THR", "VAL")
    chains = (
        _translated_chain("A", sequence, (0.0, 0.0, 0.0)),
        _translated_chain("B", sequence, (11.0, 2.0, 1.0)),
    )

    result = protected_chain_match(chains, chains)

    assert result.mapping == {"A": "A", "B": "B"}
    assert not result.accepted
    assert result.reason == "insufficient_rmsd_improvement"


def test_assignment_never_crosses_incompatible_groups():
    sequence_a = ("ALA", "GLY", "SER", "THR", "VAL")
    sequence_b = ("LEU", "ILE", "PHE", "TYR", "TRP")
    gt = (
        _translated_chain("A", sequence_a, (0.0, 0.0, 0.0)),
        _translated_chain("B", sequence_b, (12.0, 0.0, 0.0)),
    )
    pred = (
        _translated_chain("A", sequence_a, (12.0, 0.0, 0.0)),
        _translated_chain("B", sequence_b, (0.0, 0.0, 0.0)),
    )

    result = protected_chain_match(pred, gt)

    assert result.mapping == {"A": "A", "B": "B"}


def test_lexicographic_assignment_is_deterministic_for_exact_ties():
    cost = np.zeros((4, 4), dtype=np.float64)

    rows, columns = lexicographic_linear_sum_assignment(cost)

    assert rows.tolist() == [0, 1, 2, 3]
    assert columns.tolist() == [0, 1, 2, 3]


def test_lexicographic_assignment_skips_infeasible_sparse_tails():
    cost = np.asarray(
        [
            [3.0, 1.0, np.inf],
            [1.0, np.inf, 2.0],
            [2.0, np.inf, 1.0],
        ],
        dtype=np.float64,
    )

    rows, columns = lexicographic_linear_sum_assignment(cost)

    assert rows.tolist() == [0, 1, 2]
    assert columns.tolist() == [1, 0, 2]
    assert cost[rows, columns].sum() == pytest.approx(3.0)


def test_group_larger_than_twenty_returns_identity_with_reason():
    sequence = ("ALA", "GLY", "SER", "THR")
    pred = tuple(
        _translated_chain(f"{index:02d}", sequence, (float(index), 0.0, 0.0))
        for index in range(21)
    )
    gt = tuple(
        _translated_chain(f"{index:02d}", sequence, (float(20 - index), 0.0, 0.0))
        for index in range(21)
    )

    result = protected_chain_match(pred, gt)

    assert result.mapping == {f"{index:02d}": f"{index:02d}" for index in range(21)}
    assert not result.accepted
    assert result.reason == "oversized_compatibility_group"


def test_oversized_group_identity_uses_a_complete_legal_assignment():
    common = ("ALA", "GLY", "SER", "THR")
    pred = (
        _translated_chain("A", (*common, "UNK"), (0.0, 0.0, 0.0)),
        _translated_chain("B", (*common, "VAL"), (1.0, 0.0, 0.0)),
        *(
            _translated_chain(
                chr(ord("C") + index),
                (*common, "UNK"),
                (float(index + 2), 0.0, 0.0),
            )
            for index in range(19)
        ),
    )
    gt = (
        _translated_chain("A", (*common, "VAL"), (0.0, 0.0, 0.0)),
        *(
            _translated_chain(
                chr(ord("B") + index),
                (*common, "LEU"),
                (float(index + 1), 0.0, 0.0),
            )
            for index in range(20)
        ),
    )

    result = protected_chain_match(pred, gt)

    assert result.mapping["A"] == "B"
    assert result.mapping["B"] == "A"
    assert len(result.mapping) == 21
    assert not result.accepted
    assert result.reason == "oversized_compatibility_group"


def test_candidate_with_lower_coverage_is_rejected():
    sequence = ("ALA", "GLY", "SER", "THR", "VAL")
    full_mask = torch.ones(5, dtype=torch.bool)
    partial_mask = torch.tensor([True, True, True, True, False])
    gt = (
        _translated_chain(
            "A", sequence, (0.0, 0.0, 0.0), representative_mask=full_mask
        ),
        _translated_chain(
            "B", sequence, (13.0, 3.0, -2.0), representative_mask=partial_mask
        ),
    )
    pred = (
        _translated_chain(
            "A", sequence, (13.0, 3.0, -2.0), representative_mask=full_mask
        ),
        _translated_chain(
            "B", sequence, (0.0, 0.0, 0.0), representative_mask=partial_mask
        ),
    )

    result = protected_chain_match(pred, gt)

    assert result.mapping == {"A": "A", "B": "B"}
    assert not result.accepted
    assert result.reason == "insufficient_coverage"
    assert result.matched_coverage == 9


def test_no_legal_supervised_correspondence_is_typed():
    pred = (_translated_chain("A", ("ALA", "GLY", "SER", "THR"), (0.0, 0.0, 0.0)),)
    gt = (_translated_chain("A", ("LEU", "ILE", "PHE", "TYR"), (0.0, 0.0, 0.0)),)

    with pytest.raises(NoLegalSupervisedCorrespondence):
        protected_chain_match(pred, gt)


@pytest.mark.parametrize(
    ("identity_rmsd", "candidate_rmsd", "accepted"),
    [
        (1.001, 1.0, False),
        (1.0010001, 1.0, True),
    ],
)
def test_rmsd_improvement_boundary_is_strict(identity_rmsd, candidate_rmsd, accepted):
    assert _improves_rmsd(identity_rmsd, candidate_rmsd) is accepted
