import inspect

import torch

from openfold3.core.loss.pdockq import (
    dockq_to_soft_bins,
    pdockq_distribution_loss,
    pdockq_pairwise_ranking_loss,
    per_candidate_complex_dockq,
)
from openfold3.core.model.heads.pdockq import ChainBalancedInterfacePool


def test_soft_target_is_a_normalized_50_bin_distribution():
    target = dockq_to_soft_bins(torch.tensor([0.0, 1.0, 0.5]))
    assert torch.allclose(target.sum(-1), torch.ones(3))
    assert target[0, 0] > 0.99
    assert target[1, -1] > 0.99


def test_chain_balanced_pool_is_invariant_to_chain_pair_size():
    pool = ChainBalancedInterfacePool()
    score_a = pool(
        si=torch.ones(1, 4, 8),
        z=torch.ones(1, 4, 4, 8),
        asym_id=torch.tensor([[0, 0, 1, 1]]),
        token_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    score_b = pool(
        si=torch.ones(1, 8, 8),
        z=torch.ones(1, 8, 8, 8),
        asym_id=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        token_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    assert torch.allclose(score_a, score_b)


def test_chain_balanced_pool_is_invariant_to_zero_asym_id_padding():
    pool = ChainBalancedInterfacePool()
    si = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    z = torch.arange(128, dtype=torch.float32).reshape(1, 4, 4, 8)
    asym_id = torch.tensor([[0, 0, 1, 1]])
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    expected = pool(
        si=si,
        z=z,
        asym_id=asym_id,
        token_mask=token_mask,
    )

    padded_si = torch.cat((si, torch.full((1, 2, 8), 10_000.0)), dim=1)
    padded_z = torch.full((1, 6, 6, 8), 20_000.0)
    padded_z[:, :4, :4] = z
    actual = pool(
        si=padded_si,
        z=padded_z,
        asym_id=torch.tensor([[0, 0, 1, 1, 0, 0]]),
        token_mask=torch.tensor([[True, True, True, True, False, False]]),
    )

    torch.testing.assert_close(actual, expected)


def test_pairwise_ranking_is_zero_for_one_candidate():
    result = pdockq_pairwise_ranking_loss(
        torch.tensor([[0.5]]), torch.tensor([[0.7]]), torch.tensor([[True]])
    )
    assert result.item() == 0.0


def test_pairwise_ranking_uses_point_zero_five_margin():
    result = pdockq_pairwise_ranking_loss(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.8, 0.2]]),
        torch.tensor([[True, True]]),
    )
    assert torch.isclose(result, torch.tensor(0.05))


def test_pairwise_ranking_vectorizes_all_48_candidates_with_finite_gradients():
    predicted = torch.linspace(0.0, 1.0, 48).reshape(1, 48).requires_grad_()
    target = torch.linspace(1.0, 0.0, 48).reshape(1, 48)
    valid = torch.ones(1, 48, dtype=torch.bool)

    loss = pdockq_pairwise_ranking_loss(predicted, target, valid)

    assert torch.isfinite(loss)
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    source = inspect.getsource(pdockq_pairwise_ranking_loss)
    assert "for left" not in source
    assert "for right" not in source


def test_empty_pdockq_support_is_connected_exact_zero_even_for_nan_logits():
    logits = torch.full((1, 2, 50), float("nan"), requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.zeros(1, 2, dtype=torch.bool)

    loss = pdockq_distribution_loss(logits, target, valid)

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_per_candidate_dockq_derives_atomized_inputs_from_training_batch(monkeypatch):
    import openfold3.core.loss.pdockq as module

    captured = {}

    def fake_dockq(**kwargs):
        captured.update(kwargs)
        return type(
            "Result",
            (),
            {
                "chain_pair_to_dockq": {
                    (0, 1, 0): torch.tensor([0.2]),
                    (0, 2, 0): torch.tensor([0.6]),
                    (0, 1, 1): torch.tensor([0.8]),
                }
            },
        )()

    monkeypatch.setattr(module, "dockq", fake_dockq)
    prediction = torch.randn(1, 2, 4, 3, dtype=torch.bfloat16, requires_grad=True)
    batch = {
        "atom_mask": torch.ones(1, 1, 4, dtype=torch.bool),
        # Real collated training batches carry this mask as a floating tensor.
        "token_mask": torch.ones(1, 1, 2),
        "num_atoms_per_token": torch.tensor([[[2, 2]]]),
        "asym_id": torch.tensor([[[1, 2]]]),
        "residue_index": torch.tensor([[[10, 20]]]),
        "ref_atom_name_chars": torch.zeros(1, 1, 4, 4, 64),
        "is_protein": torch.ones(1, 1, 2, dtype=torch.bool),
        "is_rna": torch.zeros(1, 1, 2, dtype=torch.bool),
        "is_dna": torch.zeros(1, 1, 2, dtype=torch.bool),
        "is_atomized": torch.zeros(1, 1, 2, dtype=torch.bool),
        "ground_truth": {
            "atom_resolved_mask": torch.tensor([[[True, True, False, True]]]),
            "inter_filter_atomized": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        },
    }

    score, valid = per_candidate_complex_dockq(
        prediction=prediction,
        gt=torch.zeros(1, 1, 4, 3, dtype=torch.bfloat16),
        batch=batch,
    )

    assert captured["pred_coords"].dtype == torch.float32
    assert captured["gt_coords"].dtype == torch.float32
    assert captured["all_atom_mask"].tolist() == [
        [[True, True, False, True], [True, True, False, True]]
    ]
    assert captured["asym_id_atomized"].tolist() == [[[1, 1, 2, 2], [1, 1, 2, 2]]]
    assert captured["res_id_atomized"].tolist() == [
        [[10, 10, 20, 20], [10, 10, 20, 20]]
    ]
    for name in (
        "is_protein_atomized",
        "is_rna_atomized",
        "is_dna_atomized",
    ):
        assert captured[name].dtype == torch.bool, name
    for name, value in captured.items():
        if torch.is_tensor(value) and value.ndim >= 2:
            assert value.shape[1] == 2, name
    torch.testing.assert_close(score, torch.tensor([[0.4, 0.8]]))
    assert valid.tolist() == [[True, True]]
    assert score.dtype == torch.float32
    assert not score.requires_grad
