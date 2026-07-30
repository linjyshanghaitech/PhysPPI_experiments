"""Candidate-specific pDockQ targets and losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from openfold3.core.metrics.quality import dockq
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms


def dockq_to_soft_bins(scores: torch.Tensor, no_bins: int = 50) -> torch.Tensor:
    """Convert DockQ scores into normalized nearest-center soft labels."""

    if no_bins != 50:
        raise ValueError("PhysPPI pDockQ uses exactly 50 bins")
    centers = torch.linspace(
        0.0, 1.0, no_bins, device=scores.device, dtype=scores.dtype
    )
    index = torch.argmin(
        torch.abs(scores.unsqueeze(-1).clamp(0.0, 1.0) - centers), dim=-1
    )
    target = torch.zeros(
        *scores.shape, no_bins, device=scores.device, dtype=scores.dtype
    )
    return target.scatter_(-1, index.unsqueeze(-1), 1.0)


def _dockq_residue_slots_from_batch(batch: dict) -> torch.Tensor:
    """Build DockQ-private compact residue slots from token order.

    Raw PDB residue IDs can be negative or sparse and are not safe scatter/gather
    slots. DockQ only needs residue grouping, so this allocates per-chain
    0-based slots while preserving adjacent same-residue grouping within each
    chain.
    """

    token_mask = batch["token_mask"].bool()
    num_atoms_per_token = batch["num_atoms_per_token"].to(device=token_mask.device)
    asym_id = batch["asym_id"].to(device=token_mask.device)
    residue_index = batch["residue_index"].to(device=token_mask.device)

    expected_shape = token_mask.shape
    for name, value in (
        ("num_atoms_per_token", num_atoms_per_token),
        ("asym_id", asym_id),
        ("residue_index", residue_index),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must match token_mask shape, got "
                f"{tuple(value.shape)} and {tuple(expected_shape)}"
            )

    effective_token = (
        token_mask & (num_atoms_per_token.long() > 0) & (asym_id.long() > 0)
    )
    slots = torch.zeros_like(residue_index, dtype=torch.long, device=token_mask.device)
    if slots.numel() == 0:
        return slots

    n_token = int(token_mask.shape[-1])
    flat_effective = effective_token.reshape(-1, n_token)
    flat_asym_id = asym_id.long().reshape(-1, n_token)
    flat_residue_index = residue_index.long().reshape(-1, n_token)
    flat_slots = slots.reshape(-1, n_token)

    for row in range(flat_slots.shape[0]):
        chain_state: dict[int, tuple[int, int, int]] = {}
        for token_pos in range(n_token):
            if not bool(flat_effective[row, token_pos].item()):
                continue
            chain_id = int(flat_asym_id[row, token_pos].item())
            residue_id = int(flat_residue_index[row, token_pos].item())
            state = chain_state.get(chain_id)
            if state is not None and state[0] == residue_id:
                slot = state[1]
                next_slot = state[2]
            else:
                slot = state[2] if state is not None else 0
                next_slot = slot + 1
            chain_state[chain_id] = (residue_id, slot, next_slot)
            flat_slots[row, token_pos] = slot

    return slots


def per_candidate_complex_dockq(
    *, prediction: torch.Tensor, gt: torch.Tensor, batch: dict
):
    """Call ``quality.dockq`` and preserve its candidate/sample axis.

    A candidate is valid only when at least one chain pair has a DockQ score;
    no-interface candidates are masked rather than treated as score zero.
    """

    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError(
            "prediction must have shape (batch, candidates, atoms, 3), got "
            f"{tuple(prediction.shape)}"
        )
    batch_size, candidates = prediction.shape[:2]

    def expand_candidates(value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.to(device=prediction.device)
        if value.ndim < 2 or value.shape[0] != batch_size:
            raise ValueError(
                f"{name} must have batch and candidate dimensions, got "
                f"{tuple(value.shape)}"
            )
        if value.shape[1] == candidates:
            return value
        if value.shape[1] != 1:
            raise ValueError(
                f"{name} candidate dimension must be 1 or {candidates}, got "
                f"{value.shape[1]}"
            )
        return value.expand(batch_size, candidates, *value.shape[2:])

    expanded_gt = expand_candidates(gt, "gt").float()
    atom_mask = batch["atom_mask"].bool()
    ground_truth = batch.get("ground_truth")
    if isinstance(ground_truth, dict):
        resolved_mask = ground_truth.get("atom_resolved_mask")
        if resolved_mask is not None:
            atom_mask = atom_mask & resolved_mask.to(
                device=atom_mask.device, dtype=torch.bool
            )

    token_mask = batch["token_mask"]
    num_atoms_per_token = batch["num_atoms_per_token"]
    not_modified = ~batch["is_atomized"].bool()

    def atomize(token_feature: torch.Tensor) -> torch.Tensor:
        return broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=token_feature,
        )

    dockq_residue_slots = _dockq_residue_slots_from_batch(batch)
    dockq_kwargs = {
        "pred_coords": prediction.detach().float(),
        "gt_coords": expanded_gt,
        "all_atom_mask": expand_candidates(atom_mask, "atom_mask"),
        "asym_id_atomized": expand_candidates(
            atomize(batch["asym_id"]), "asym_id_atomized"
        ),
        "res_id_atomized": expand_candidates(
            atomize(dockq_residue_slots), "res_id_atomized"
        ),
        "ref_atom_name_chars_atomized": expand_candidates(
            batch["ref_atom_name_chars"],
            "ref_atom_name_chars_atomized",
        ),
        "is_protein_atomized": expand_candidates(
            atomize(batch["is_protein"].bool() & not_modified).bool(),
            "is_protein_atomized",
        ),
        "is_rna_atomized": expand_candidates(
            atomize(batch["is_rna"].bool() & not_modified).bool(),
            "is_rna_atomized",
        ),
        "is_dna_atomized": expand_candidates(
            atomize(batch["is_dna"].bool() & not_modified).bool(),
            "is_dna_atomized",
        ),
    }
    inter_filter = (
        ground_truth.get("inter_filter_atomized")
        if isinstance(ground_truth, dict)
        else None
    )
    dockq_kwargs["inter_filter_atomized"] = (
        None
        if inter_filter is None
        else expand_candidates(inter_filter, "inter_filter_atomized")
    )
    result = dockq(**dockq_kwargs)
    scores = torch.zeros(
        prediction.shape[:2],
        device=prediction.device,
        dtype=torch.float32,
    )
    valid = torch.zeros(
        prediction.shape[:2], dtype=torch.bool, device=prediction.device
    )
    grouped: dict[int, list[torch.Tensor]] = {}
    for key, value in result.chain_pair_to_dockq.items():
        grouped.setdefault(int(key[-1]), []).append(value)
    for sample_index, values in grouped.items():
        if sample_index >= scores.shape[1]:
            continue
        scores[:, sample_index] = torch.stack(
            [value.detach().to(scores) for value in values], dim=-1
        ).mean(-1)
        valid[:, sample_index] = True
    return scores, valid


def pdockq_distribution_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape[:-1] != valid.shape:
        raise ValueError(
            "pDockQ logits/targets/valid shapes must be (..., bins), "
            f"(..., bins), (...), got {tuple(logits.shape)}, "
            f"{tuple(targets.shape)}, {tuple(valid.shape)}"
        )
    valid = valid.to(device=logits.device, dtype=torch.bool)
    if not valid.any():
        return torch.nan_to_num(logits).sum() * 0.0
    safe_logits = torch.where(valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    safe_targets = torch.where(valid.unsqueeze(-1), targets, torch.zeros_like(targets))
    losses = F.kl_div(
        torch.log_softmax(safe_logits, dim=-1),
        safe_targets,
        reduction="none",
    ).sum(-1)
    weights = valid.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def pdockq_pairwise_ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Rank valid candidates only within each complex."""

    if predicted.shape != target.shape or predicted.shape != valid.shape:
        raise ValueError(
            "pDockQ predicted/target/valid shapes must match, got "
            f"{tuple(predicted.shape)}, {tuple(target.shape)}, "
            f"{tuple(valid.shape)}"
        )
    candidates = predicted.shape[1]
    left, right = torch.triu_indices(
        candidates,
        candidates,
        offset=1,
        device=predicted.device,
    )
    pair_valid = valid[:, left].bool() & valid[:, right].bool()
    difference = target[:, left] - target[:, right]
    safe_difference = torch.where(pair_valid, difference, torch.zeros_like(difference))
    rankable = pair_valid & (safe_difference.abs() > margin)
    prediction_delta = predicted[:, left] - predicted[:, right]
    safe_prediction_delta = torch.where(
        rankable, prediction_delta, torch.zeros_like(prediction_delta)
    )
    terms = torch.relu(margin - torch.sign(safe_difference) * safe_prediction_delta)
    weights = rankable.to(dtype=terms.dtype)
    return (terms * weights).sum() / weights.sum().clamp_min(1.0)


__all__ = [
    "dockq_to_soft_bins",
    "pdockq_distribution_loss",
    "pdockq_pairwise_ranking_loss",
    "per_candidate_complex_dockq",
]
