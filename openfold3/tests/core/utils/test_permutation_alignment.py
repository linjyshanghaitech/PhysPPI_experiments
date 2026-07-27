"""Tests for the defensive size check in find_greedy_optimal_mol_permutation."""

import biotite.structure as struc
import numpy as np
import pytest
import torch

import openfold3.core.utils.permutation_alignment as permutation_alignment
from openfold3.core.data.framework.single_datasets.dataset_utils import (
    check_invalid_feature_dict,
)
from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_3
from openfold3.core.utils.permutation_alignment import (
    find_greedy_optimal_mol_permutation,
    reshape_per_sample_inputs,
)
from openfold3.core.utils.protected_chain_matching import (
    ChainRecord,
    ProtectedChainMatchError,
)
from openfold3.projects.of3_all_atom.physppi.model.membrane_geometry import (
    DirectedPlanePrediction,
)


def _make_homodimer_inputs(n_tokens_a, n_tokens_b):
    """Build permutation alignment inputs mimicking a homodimer.

    Creates two symmetric instances of the same entity with the given token
    counts. When n_tokens_a != n_tokens_b this reproduces the mismatch caused
    by the sym_id residue-splitting bug.
    """
    # Build a minimal atom array with two chains
    n_atoms_a, n_atoms_b = n_tokens_a, n_tokens_b  # 1 atom per token
    n_total = n_atoms_a + n_atoms_b

    aa = struc.AtomArray(n_total)
    aa.chain_id[:n_atoms_a] = "A"
    aa.chain_id[n_atoms_a:] = "B"
    aa.res_id[:n_atoms_a] = np.arange(1, n_atoms_a + 1)
    aa.res_id[n_atoms_a:] = np.arange(1, n_atoms_b + 1)
    aa.ins_code[:] = ""
    aa.res_name[:] = "ALA"
    aa.atom_name[:] = "CA"
    aa.element[:] = "C"
    aa.coord[:] = np.random.randn(n_total, 3)

    # Derive tensor inputs from the atom array
    entity_ids = torch.ones(n_total, dtype=torch.long)
    sym_ids = torch.tensor([1] * n_atoms_a + [2] * n_atoms_b)
    sym_token_index = torch.tensor(list(range(n_tokens_a)) + list(range(n_tokens_b)))
    gt_coords = torch.tensor(aa.coord, dtype=torch.float32).unsqueeze(0)
    gt_resolved = torch.ones(n_total)
    pred_coords = torch.randn(n_total, 3)

    return dict(
        gt_token_center_positions_transformed=gt_coords,
        gt_token_center_resolved_mask=gt_resolved,
        gt_mol_entity_ids=entity_ids,
        gt_mol_sym_ids=sym_ids,
        gt_mol_sym_token_index=sym_token_index,
        pred_token_center_positions=pred_coords,
        pred_mol_entity_ids=entity_ids.clone(),
        pred_mol_sym_ids=sym_ids.clone(),
        pred_mol_sym_token_index=sym_token_index.clone(),
    )


def test_mismatched_token_counts_raises():
    """Symmetric instances with different token counts raise a clear error.

    This reproduces the crash caused by partially-unresolved residues where
    biotite's get_residue_starts() splits residues on sym_id boundaries,
    producing different token counts for symmetric chains.
    """
    inputs = _make_homodimer_inputs(n_tokens_a=5, n_tokens_b=3)

    with pytest.raises(
        ValueError,
        match="symmetric instances with different token counts",
    ):
        find_greedy_optimal_mol_permutation(**inputs)


def test_matched_token_counts_succeeds():
    """Symmetric instances with equal token counts produce a valid mapping."""
    inputs = _make_homodimer_inputs(n_tokens_a=4, n_tokens_b=4)

    result = find_greedy_optimal_mol_permutation(**inputs)

    assert isinstance(result, dict)
    assert len(result) == 2


def test_per_sample_reshape_excludes_differentiable_membrane_prediction():
    """Alignment copies data features without severing membrane-loss gradients."""
    differentiable = torch.ones(1, 1, 3, requires_grad=True) * 2.0
    prediction = DirectedPlanePrediction(
        normal=differentiable,
        offset=differentiable[..., 0],
        signed_depth=differentiable,
        normal_error_logits=differentiable,
        offset_error_logits=differentiable,
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    batch = {
        "token_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "physppi_predicted_plane": prediction,
    }

    per_sample_batch, _ = reshape_per_sample_inputs(
        batch=batch,
        atom_positions_predicted=torch.zeros(1, 2, 1, 3),
    )

    assert "physppi_predicted_plane" not in per_sample_batch
    assert batch["physppi_predicted_plane"] is prediction
    assert not batch["physppi_predicted_plane"].normal.is_leaf


def test_invalid_token_center_preserves_losses_and_raises_typed_error(monkeypatch):
    """Invalid token centers remain a typed data failure without mutating losses."""
    calls: list[str] = []

    def fail(name: str):
        def _fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} alignment must not be called")

        return _fail

    monkeypatch.setattr(
        permutation_alignment,
        "multi_chain_permutation_alignment",
        fail("full"),
    )
    monkeypatch.setattr(permutation_alignment, "naive_alignment", fail("naive"))

    predicted = torch.zeros(1, 1, 2, 3)
    batch = {
        "pdb_id": ["exact_bad_sample"],
        "start_atom_index": torch.tensor([[[2]]]),
        "is_protein": torch.zeros(1, 1, 1),
        "is_atomized": torch.ones(1, 1, 1),
        "is_dna": torch.zeros(1, 1, 1),
        "is_rna": torch.zeros(1, 1, 1),
        "restype": torch.zeros(1, 1, 1, 32),
        "token_mask": torch.ones(1, 1, 1),
        "atom_mask": torch.ones(1, 1, 2),
        "ground_truth": {
            "atom_positions": torch.zeros(1, 1, 2, 3),
            "atom_resolved_mask": torch.ones(1, 1, 2),
        },
        "loss_weights": {"diffusion": torch.ones(1, 1)},
    }

    with pytest.raises(ProtectedChainMatchError, match="invalid_token_center_index"):
        permutation_alignment.safe_multi_chain_permutation_alignment(batch, predicted)

    assert calls == []
    assert batch["loss_weights"]["diffusion"].all()


def test_protected_failure_uses_identity_without_zeroing_losses(monkeypatch):
    """A typed protected-matcher failure takes the deterministic identity route."""
    calls: list[str] = []

    def fake_alignment(
        batch,
        atom_positions_predicted,
        *,
        chain_assignment,
        fallback_reason="",
    ):
        calls.append(chain_assignment)
        if chain_assignment in {"token_identity", "protected"}:
            raise ProtectedChainMatchError("historical_token_mismatch")
        batch["physppi_chain_match_diagnostics"] = [
            {
                "accepted": False,
                "fallback": "identity",
                "reason": "historical_token_mismatch",
                "identity_rmsd": 0.0,
                "candidate_rmsd": 0.0,
                "matched_coverage": 2,
            }
        ]
        return {
            "atom_positions": atom_positions_predicted.clone(),
            "atom_resolved_mask": torch.ones_like(
                atom_positions_predicted[..., 0], dtype=torch.bool
            ),
        }

    monkeypatch.setattr(
        permutation_alignment,
        "multi_chain_permutation_alignment",
        fake_alignment,
    )
    predicted = torch.zeros(1, 2, 3)
    batch = {
        "pdb_id": ["historical_token_mismatch"],
        "start_atom_index": torch.tensor([[0, 1]]),
        "is_protein": torch.ones(1, 2),
        "is_atomized": torch.zeros(1, 2),
        "is_dna": torch.zeros(1, 2),
        "is_rna": torch.zeros(1, 2),
        "restype": torch.zeros(1, 2, 32),
        "token_mask": torch.ones(1, 2),
        "atom_mask": torch.ones(1, 2),
        "ground_truth": {
            "atom_positions": torch.zeros(1, 2, 3),
            "atom_resolved_mask": torch.ones(1, 2),
        },
        "loss_weights": {"diffusion": torch.ones(1)},
    }

    permutation_alignment.safe_multi_chain_permutation_alignment(batch, predicted)

    assert calls == ["token_identity", "protected", "identity"]
    assert batch["loss_weights"]["diffusion"].all()
    assert batch["physppi_chain_match_diagnostics"][0]["fallback"] == "identity"
    assert batch["physppi_chain_match_diagnostics"][0]["reason"] == (
        "historical_token_mismatch"
    )


def test_safe_alignment_prefers_token_identity_before_chain_matching(monkeypatch):
    calls: list[str] = []

    def fake_alignment(
        batch,
        atom_positions_predicted,
        *,
        chain_assignment,
        fallback_reason="",
    ):
        calls.append(chain_assignment)
        if chain_assignment != "token_identity":
            raise AssertionError("chain-level alignment should not run first")
        batch["physppi_chain_match_diagnostics"] = [
            {
                "accepted": True,
                "fallback": "none",
                "reason": "token_index_identity",
                "identity_rmsd": 0.0,
                "candidate_rmsd": 0.0,
                "matched_coverage": 1,
            }
        ]
        return {
            "atom_positions": atom_positions_predicted.clone(),
            "atom_resolved_mask": torch.ones_like(
                atom_positions_predicted[..., 0], dtype=torch.bool
            ),
        }

    monkeypatch.setattr(
        permutation_alignment,
        "multi_chain_permutation_alignment",
        fake_alignment,
    )

    predicted = torch.zeros(1, 1, 3)
    batch = {
        "start_atom_index": torch.tensor([[0]]),
        "is_protein": torch.ones(1, 1),
        "is_atomized": torch.zeros(1, 1),
        "is_dna": torch.zeros(1, 1),
        "is_rna": torch.zeros(1, 1),
        "restype": torch.zeros(1, 1, 32),
        "token_mask": torch.ones(1, 1),
        "atom_mask": torch.ones(1, 1),
        "ground_truth": {
            "atom_positions": torch.zeros(1, 1, 3),
            "atom_resolved_mask": torch.ones(1, 1),
        },
    }

    permutation_alignment.safe_multi_chain_permutation_alignment(batch, predicted)

    assert calls == ["token_identity"]
    assert batch["physppi_chain_match_diagnostics"][0]["reason"] == (
        "token_index_identity"
    )


def test_safe_alignment_ignores_padded_token_center_indices(monkeypatch):
    calls: list[str] = []

    def fake_alignment(
        batch,
        atom_positions_predicted,
        *,
        chain_assignment,
        fallback_reason="",
    ):
        calls.append(chain_assignment)
        batch["physppi_chain_match_diagnostics"] = [
            {
                "accepted": True,
                "fallback": "none",
                "reason": "token_index_identity",
                "identity_rmsd": 0.0,
                "candidate_rmsd": 0.0,
                "matched_coverage": 1,
            }
        ]
        return {
            "atom_positions": atom_positions_predicted.clone(),
            "atom_resolved_mask": torch.ones_like(
                atom_positions_predicted[..., 0], dtype=torch.bool
            ),
        }

    monkeypatch.setattr(
        permutation_alignment,
        "multi_chain_permutation_alignment",
        fake_alignment,
    )

    predicted = torch.zeros(1, 1, 3)
    restype = torch.zeros(1, 2, 32)
    restype[..., 0] = 1
    batch = {
        "start_atom_index": torch.tensor([[0, 1]]),
        "is_protein": torch.zeros(1, 2),
        "is_atomized": torch.ones(1, 2),
        "is_dna": torch.zeros(1, 2),
        "is_rna": torch.zeros(1, 2),
        "restype": restype,
        "token_mask": torch.tensor([[True, False]]),
        "atom_mask": torch.ones(1, 1),
        "ground_truth": {
            "atom_positions": torch.zeros(1, 1, 3),
            "atom_resolved_mask": torch.ones(1, 1),
        },
    }

    permutation_alignment.safe_multi_chain_permutation_alignment(batch, predicted)

    assert calls == ["token_identity"]


def test_identity_fallback_finds_complete_sequence_compatible_assignment(monkeypatch):
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    common = ("ALA", "GLY", "SER", "THR")
    pred_records = [
        ChainRecord("A", "protein", (*common, "UNK"), points, torch.ones(5)),
        ChainRecord("B", "protein", (*common, "VAL"), points, torch.ones(5)),
    ]
    gt_records = [
        ChainRecord("A", "protein", (*common, "VAL"), points, torch.ones(5)),
        ChainRecord("B", "protein", (*common, "GLY"), points, torch.ones(5)),
    ]
    records = iter(
        [
            (pred_records, {"A": (1, 1), "B": (1, 2)}),
            (gt_records, {"A": (2, 1), "B": (2, 2)}),
        ]
    )
    monkeypatch.setattr(
        permutation_alignment,
        "_chain_records_from_features",
        lambda *_args, **_kwargs: next(records),
    )

    mapping, diagnostics = (
        permutation_alignment.find_sequence_compatible_identity_mol_permutation(
            pred_features={},
            pred_atom_positions=torch.empty(0, 3),
            pred_atom_mask=torch.empty(0, dtype=torch.bool),
            gt_features={},
            gt_atom_positions=torch.empty(0, 3),
            gt_atom_mask=torch.empty(0, dtype=torch.bool),
            reason="forced_fallback",
        )
    )

    assert mapping == {(1, 1): (2, 2), (1, 2): (2, 1)}
    assert diagnostics["fallback"] == "identity"
    assert diagnostics["matched_coverage"] == 10


def test_safe_alignment_does_not_swallow_untyped_errors(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("programming defect")

    monkeypatch.setattr(
        permutation_alignment,
        "multi_chain_permutation_alignment",
        fail,
    )
    predicted = torch.zeros(1, 1, 3)
    batch = {
        "start_atom_index": torch.tensor([[0]]),
        "is_protein": torch.ones(1, 1),
        "is_atomized": torch.zeros(1, 1),
        "is_dna": torch.zeros(1, 1),
        "is_rna": torch.zeros(1, 1),
        "restype": torch.zeros(1, 1, 32),
        "token_mask": torch.ones(1, 1),
        "atom_mask": torch.ones(1, 1),
        "ground_truth": {},
        "loss_weights": {"diffusion": torch.ones(1)},
    }

    with pytest.raises(RuntimeError, match="programming defect"):
        permutation_alignment.safe_multi_chain_permutation_alignment(batch, predicted)


def test_training_alignment_routes_chain_assignment_through_protected_matcher(
    monkeypatch,
):
    diagnostics = {
        "accepted": True,
        "fallback": "none",
        "reason": "accepted",
        "identity_rmsd": 4.0,
        "candidate_rmsd": 0.0,
        "matched_coverage": 4,
    }
    protected_calls: list[str] = []

    def protected_assignment(**kwargs):
        protected_calls.append("protected")
        return {(1, 1): (1, 1)}, diagnostics

    monkeypatch.setattr(
        permutation_alignment,
        "find_protected_mol_permutation",
        protected_assignment,
    )
    monkeypatch.setattr(
        permutation_alignment,
        "find_greedy_optimal_mol_permutation",
        lambda *_args, **_kwargs: pytest.fail("legacy greedy chain assignment"),
    )
    monkeypatch.setattr(
        permutation_alignment,
        "get_pred_to_permuted_gt_transformation",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        permutation_alignment,
        "apply_transformation",
        lambda positions, _transformation: positions,
    )
    monkeypatch.setattr(
        permutation_alignment,
        "get_final_atom_permutation_index",
        lambda **kwargs: torch.arange(kwargs["pred_positions_aligned"].shape[0]),
    )

    restype = torch.zeros(4, 32)
    restype[:, 0] = 1
    common_tokens = {
        "token_mask": torch.ones(4, dtype=torch.bool),
        "atom_mask": torch.ones(4, dtype=torch.bool),
        "mol_entity_id": torch.ones(4, dtype=torch.long),
        "mol_sym_id": torch.ones(4, dtype=torch.long),
        "mol_sym_component_id": torch.arange(4),
        "mol_sym_token_index": torch.arange(4),
        "num_atoms_per_token": torch.ones(4, dtype=torch.long),
        "start_atom_index": torch.arange(4),
        "restype": restype,
        "is_protein": torch.ones(4, dtype=torch.long),
        "is_rna": torch.zeros(4, dtype=torch.long),
        "is_dna": torch.zeros(4, dtype=torch.long),
        "is_ligand": torch.zeros(4, dtype=torch.long),
        "is_atomized": torch.ones(4, dtype=torch.long),
    }
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    batch = {
        **common_tokens,
        "ref_space_uid": torch.arange(4),
        "ref_space_uid_to_perm": {
            index: torch.zeros(1, 1, dtype=torch.long) for index in range(4)
        },
        "ground_truth": {
            **common_tokens,
            "atom_positions": coordinates.clone(),
            "atom_resolved_mask": torch.ones(4, dtype=torch.bool),
        },
    }

    atom_index = permutation_alignment.single_batch_multi_chain_permutation_alignment(
        batch,
        coordinates.clone(),
        chain_assignment="protected",
    )

    assert protected_calls == ["protected"]
    assert torch.equal(atom_index, torch.arange(4))
    assert batch["physppi_chain_match_diagnostic"] == diagnostics


def _atomized_alignment_token_features(token_count: int) -> dict:
    restype = torch.zeros(token_count, len(STANDARD_RESIDUES_WITH_GAP_3))
    restype[:, 0] = 1
    return {
        "token_index": torch.arange(10, 10 + token_count),
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "atom_mask": torch.ones(token_count, dtype=torch.bool),
        "num_atoms_per_token": torch.ones(token_count, dtype=torch.long),
        "start_atom_index": torch.arange(token_count),
        "restype": restype,
        "is_protein": torch.zeros(token_count, dtype=torch.long),
        "is_rna": torch.zeros(token_count, dtype=torch.long),
        "is_dna": torch.zeros(token_count, dtype=torch.long),
        "is_ligand": torch.zeros(token_count, dtype=torch.long),
        "is_atomized": torch.ones(token_count, dtype=torch.long),
        "asym_id": torch.tensor([1, 1, 2, 2], dtype=torch.long)[:token_count],
        "entity_id": torch.ones(token_count, dtype=torch.long),
        "sym_id": torch.tensor([1, 1, 2, 2], dtype=torch.long)[:token_count],
        "mol_entity_id": torch.ones(token_count, dtype=torch.long),
        "mol_sym_id": torch.tensor([1, 1, 2, 2], dtype=torch.long)[:token_count],
        "mol_sym_component_id": torch.arange(token_count),
        "mol_sym_token_index": torch.arange(token_count),
    }


def test_token_identity_alignment_bypasses_chain_level_matchers(monkeypatch):
    def fail_chain_assignment(*_args, **_kwargs):
        raise AssertionError("chain-level matcher must not run")

    monkeypatch.setattr(
        permutation_alignment,
        "find_protected_mol_permutation",
        fail_chain_assignment,
    )
    monkeypatch.setattr(
        permutation_alignment,
        "find_greedy_optimal_mol_permutation",
        fail_chain_assignment,
    )

    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    common_tokens = _atomized_alignment_token_features(4)
    batch = {
        **common_tokens,
        "ref_space_uid": torch.arange(4),
        "ref_space_uid_to_perm": {
            index: torch.zeros(1, 1, dtype=torch.long) for index in range(4)
        },
        "ground_truth": {
            **common_tokens,
            "atom_positions": coordinates.clone(),
            "atom_resolved_mask": torch.ones(4, dtype=torch.bool),
        },
    }

    atom_index = permutation_alignment.single_batch_multi_chain_permutation_alignment(
        batch,
        coordinates.clone(),
        chain_assignment="token_identity",
    )

    assert torch.equal(atom_index, torch.arange(4))
    assert batch["physppi_chain_match_diagnostic"]["reason"] == "token_index_identity"


def test_token_identity_alignment_ignores_padded_token_center_indices():
    restype = torch.zeros(2, len(STANDARD_RESIDUES_WITH_GAP_3))
    restype[:, 0] = 1
    common_tokens = {
        "token_index": torch.tensor([10, 0]),
        "token_mask": torch.tensor([True, False]),
        "atom_mask": torch.tensor([True]),
        "num_atoms_per_token": torch.tensor([1, 0]),
        "start_atom_index": torch.tensor([0, 1]),
        "restype": restype,
        "is_protein": torch.zeros(2, dtype=torch.long),
        "is_rna": torch.zeros(2, dtype=torch.long),
        "is_dna": torch.zeros(2, dtype=torch.long),
        "is_ligand": torch.zeros(2, dtype=torch.long),
        "is_atomized": torch.ones(2, dtype=torch.long),
        "asym_id": torch.tensor([1, 0]),
        "entity_id": torch.tensor([1, 0]),
        "sym_id": torch.tensor([1, 0]),
    }
    coordinates = torch.zeros(1, 3)
    batch = {
        **common_tokens,
        "ref_space_uid": torch.tensor([0]),
        "ref_space_uid_to_perm": {0: torch.zeros(1, 1, dtype=torch.long)},
        "ground_truth": {
            **common_tokens,
            "atom_positions": coordinates.clone(),
            "atom_resolved_mask": torch.ones(1, dtype=torch.bool),
        },
    }

    atom_index = permutation_alignment.single_batch_multi_chain_permutation_alignment(
        batch,
        coordinates.clone(),
        chain_assignment="token_identity",
    )

    assert torch.equal(atom_index, torch.tensor([0]))


def test_token_identity_alignment_uses_single_atom_tokens_as_centers():
    restype = torch.zeros(1, len(STANDARD_RESIDUES_WITH_GAP_3))
    restype[:, 0] = 1
    common_tokens = {
        "token_index": torch.tensor([10]),
        "token_mask": torch.tensor([True]),
        "atom_mask": torch.tensor([True]),
        "num_atoms_per_token": torch.tensor([1]),
        "start_atom_index": torch.tensor([0]),
        "restype": restype,
        "is_protein": torch.ones(1, dtype=torch.long),
        "is_rna": torch.zeros(1, dtype=torch.long),
        "is_dna": torch.zeros(1, dtype=torch.long),
        "is_ligand": torch.zeros(1, dtype=torch.long),
        "is_atomized": torch.zeros(1, dtype=torch.long),
        "asym_id": torch.tensor([1]),
        "entity_id": torch.tensor([1]),
        "sym_id": torch.tensor([1]),
    }
    coordinates = torch.zeros(1, 3)
    batch = {
        **common_tokens,
        "ref_space_uid": torch.tensor([0]),
        "ref_space_uid_to_perm": {0: torch.zeros(1, 1, dtype=torch.long)},
        "ground_truth": {
            **common_tokens,
            "atom_positions": coordinates.clone(),
            "atom_resolved_mask": torch.ones(1, dtype=torch.bool),
        },
    }

    atom_index = permutation_alignment.single_batch_multi_chain_permutation_alignment(
        batch,
        coordinates.clone(),
        chain_assignment="token_identity",
    )

    assert torch.equal(atom_index, torch.tensor([0]))


def test_token_identity_atom_permutation_resolves_intra_token_symmetry():
    atom_index = permutation_alignment.get_token_identity_atom_permutation_index(
        gt_positions=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        gt_resolved_mask=torch.ones(2, dtype=torch.bool),
        gt_token_index=torch.tensor([42]),
        gt_num_atoms_per_token=torch.tensor([2]),
        gt_asym_id=torch.tensor([1]),
        pred_positions_aligned=torch.tensor([[10.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        pred_token_index=torch.tensor([42]),
        pred_num_atoms_per_token=torch.tensor([2]),
        pred_asym_id=torch.tensor([1]),
        pred_ref_space_uid=torch.tensor([7, 7]),
        pred_ref_space_uid_to_perm={
            7: torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        },
    )

    assert torch.equal(atom_index, torch.tensor([1, 0]))


def _valid_feature_dict() -> dict:
    return {
        "token_index": torch.tensor([0, 1]),
        "token_mask": torch.tensor([True, True]),
        "num_atoms_per_token": torch.tensor([1, 1]),
        "start_atom_index": torch.tensor([0, 1]),
        "atom_to_token_index": torch.tensor([0, 1]),
        "atom_mask": torch.tensor([True, True]),
        "physppi_token_center_atom_index": torch.tensor([0, 1]),
        "ref_pos": torch.zeros(2, 3),
        "is_ligand": torch.tensor([False, False]),
        "is_atomized": torch.tensor([False, False]),
        "ground_truth": {
            "token_index": torch.tensor([0, 1]),
            "token_mask": torch.tensor([True, True]),
            "num_atoms_per_token": torch.tensor([1, 1]),
            "start_atom_index": torch.tensor([0, 1]),
            "atom_mask": torch.tensor([True, True]),
            "atom_positions": torch.zeros(2, 3),
            "atom_resolved_mask": torch.tensor([True, True]),
        },
    }


def test_feature_preflight_is_deterministic_and_never_samples_coordinates(monkeypatch):
    monkeypatch.setattr(
        torch,
        "randn_like",
        lambda *_args, **_kwargs: pytest.fail("random preflight"),
    )

    check_invalid_feature_dict(_valid_feature_dict())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda features: features.update(
                token_index=torch.tensor([1, 1]),
            ),
            "token_index.*strictly increasing",
        ),
        (
            lambda features: features.update(
                atom_to_token_index=torch.tensor([0, 2]),
            ),
            "atom_to_token_index.*bounds",
        ),
        (
            lambda features: features.update(
                physppi_token_center_atom_index=torch.tensor([0, 2]),
            ),
            "representative atom index.*bounds",
        ),
        (
            lambda features: features["ground_truth"].update(
                token_index=torch.tensor([0, 2]),
            ),
            "ground-truth crop coverage",
        ),
    ],
)
def test_feature_preflight_rejects_structural_index_errors(mutate, message):
    features = _valid_feature_dict()
    mutate(features)

    with pytest.raises(ValueError, match=message):
        check_invalid_feature_dict(features)
