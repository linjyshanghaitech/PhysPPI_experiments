"""Reduced CPU smoke for the canonical PhysPPI active route."""

import copy
import random
import warnings

import numpy as np
import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3
from openfold3.tests.data_utils import random_of3_features


def _reduced_config():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.msa.msa_module.no_blocks = 1
    cfg.architecture.template.template_pair_stack.no_blocks = 1
    cfg.architecture.diffusion_module.diffusion_transformer.no_blocks = 1
    cfg.architecture.shared.diffusion.no_mini_rollout_steps = 1
    cfg.architecture.shared.diffusion.no_mini_rollout_samples = 1
    cfg.architecture.shared.diffusion.no_full_rollout_steps = 1
    cfg.architecture.shared.diffusion.no_full_rollout_samples = 1
    cfg.architecture.shared.diffusion.no_samples = 1
    cfg.architecture.shared.num_recycles = 0
    cfg.settings.train_confidence_only = True
    cfg.architecture.loss_module.train_confidence_only = True
    cfg.architecture.loss_module.confidence.pae.enabled = True
    cfg.architecture.heads.pae.enabled = True
    cfg.architecture.lora.enabled = False
    return cfg


def _canonical_like_batch(*, membrane: bool):
    batch = random_of3_features(
        batch_size=1,
        n_token=4,
        n_msa=2,
        n_templ=1,
        is_eval=True,
    )
    n_token = batch["token_mask"].shape[-1]
    n_atom = batch["ref_pos"].shape[-2]
    batch["asym_id"] = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    batch["entity_id"] = batch["asym_id"].clone()
    batch.update(
        {
            "mol_entity_id": batch["entity_id"].clone(),
            "mol_sym_id": batch["sym_id"].clone(),
            "mol_sym_component_id": batch["asym_id"].clone(),
            "mol_sym_token_index": batch["token_index"].clone(),
        }
    )
    batch.update(
        {
            "physppi_contact_available": torch.tensor([True]),
            "physppi_contact_selected": torch.tensor([True]),
            "physppi_contact_valid": torch.tensor([True]),
            "physppi_contact_provenance": torch.tensor([True]),
            "physppi_contact_active": torch.tensor([True]),
            "physppi_contact_pair_mask": torch.tensor(
                [
                    [
                        [False, False, True, False],
                        [False, False, False, True],
                        [True, False, False, False],
                        [False, True, False, False],
                    ]
                ],
                dtype=torch.bool,
            ),
            "physppi_interaction_available": torch.tensor([True]),
            "physppi_interaction_selected": torch.tensor([True]),
            "physppi_interaction_valid": torch.tensor([True]),
            "physppi_interaction_provenance": torch.tensor([True]),
            "physppi_interaction_active": torch.tensor([True]),
            "physppi_interaction_type_mask": torch.zeros(
                1, n_token, n_token, 7, dtype=torch.bool
            ),
            "physppi_surface_available": torch.tensor([True]),
            "physppi_surface_selected": torch.tensor([True]),
            "physppi_surface_valid": torch.tensor([True]),
            "physppi_surface_provenance": torch.tensor([True]),
            "physppi_surface_active": torch.tensor([True]),
            "physppi_surface_local_coords": torch.zeros(1, n_token, 128, 3),
            "physppi_surface_features": torch.zeros(1, n_token, 128, 6),
            "physppi_surface_mask": torch.ones(1, n_token, 128, dtype=torch.bool),
            "physppi_surface_charge_mask": torch.ones(
                1, n_token, 128, dtype=torch.bool
            ),
            "physppi_tm_available": torch.tensor([membrane]),
            "physppi_tm_selected": torch.tensor([membrane]),
            "physppi_tm_valid": torch.tensor([membrane]),
            "physppi_tm_provenance": torch.tensor([True]),
            "physppi_tm_active": torch.tensor([membrane]),
            "physppi_tm_token_mask": torch.tensor(
                [[membrane, False, False, False]], dtype=torch.bool
            ),
            "physppi_monomer_atom_positions": batch["ref_pos"].clone(),
            "physppi_monomer_atom_mask": batch["atom_mask"].bool().clone(),
            "physppi_token_center_atom_index": torch.arange(n_token).reshape(1, -1),
            "physppi_backbone_atom_indices": torch.tensor(
                [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]]
            ),
            "physppi_backbone_frame_mask": torch.ones(1, n_token, dtype=torch.bool),
            "physppi_membrane_gt_normal": torch.tensor([[0.0, 0.0, 1.0]]),
            "physppi_membrane_gt_point": torch.zeros(1, 3),
            "physppi_membrane_gt_valid": torch.tensor([membrane]),
            "physppi_paired_msa_chain_mask": torch.tensor([[True, True]]),
        }
    )
    batch["physppi_interaction_type_mask"][0, 0, 2, 0] = True
    batch["physppi_interaction_type_mask"][0, 2, 0, 0] = True
    batch["ground_truth"]["atom_positions"] = torch.randn(1, n_atom, 3)
    batch["ground_truth"]["atom_resolved_mask"] = torch.ones(1, n_atom)
    batch["ref_space_uid_to_perm"] = [{}]
    for key in (
        "token_mask",
        "token_index",
        "asym_id",
        "is_ligand",
        "is_protein",
        "is_dna",
        "is_rna",
        "is_atomized",
        "restype",
        "num_atoms_per_token",
        "start_atom_index",
        "atom_mask",
        "ref_space_uid",
    ):
        batch["ground_truth"][key] = batch[key].clone()
    batch["ground_truth"].update(
        {
            "mol_entity_id": batch["entity_id"].clone(),
            "mol_sym_id": batch["sym_id"].clone(),
            "mol_sym_component_id": batch["asym_id"].clone(),
            "mol_sym_token_index": batch["token_index"].clone(),
        }
    )
    return batch


def test_canonical_prior_route_emits_confidence_and_finite_loss(monkeypatch):
    import openfold3.projects.of3_all_atom.model as model_module

    monkeypatch.setattr(
        model_module, "safe_multi_chain_permutation_alignment", lambda **_: None
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = _reduced_config()
    model = OpenFold3(cfg).eval()
    loss_fn = OpenFold3Loss(cfg.architecture.loss_module)
    assert hasattr(model.aux_heads, "pde")
    assert not hasattr(model.aux_heads, "experimentally_resolved")
    assert "experimentally_resolved" not in cfg.architecture.loss_module.confidence
    assert not any("energy" in name.lower() for name in model.state_dict())
    for membrane in (False, True):
        batch = _canonical_like_batch(membrane=membrane)
        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            processed, output = model(batch)
            loss, breakdown = loss_fn(processed, output, _return_breakdown=True)
        assert not [
            warning
            for warning in caught
            if "PhysPPI" in str(warning.message)
            or "optional" in str(warning.message).lower()
        ]
        assert torch.isfinite(loss)
        assert output["pde_logits"].shape[-1] == 64
        assert "pdockq_logits" in output
        assert "plddt_logits" in output
        assert "pae_logits" in output
        assert "physppi_contact_mask" not in processed
        assert "physppi_membrane_normal" not in output
        assert "physppi_membrane_offset" not in output
        assert "physppi_membrane_plane_valid" not in output
        assert "physppi_predicted_membrane_normal" in output
        assert "physppi_predicted_membrane_offset" in output
        assert "physppi_predicted_membrane_valid" in output
        interaction_distance = output["interaction_distance_pred"]
        assert interaction_distance.shape[-2:] == (4, 4)
        assert torch.equal(interaction_distance, interaction_distance.transpose(-1, -2))
        assert torch.isfinite(torch.stack(list(breakdown.values()))).all()
        if membrane:
            assert "membrane_plane_loss" in breakdown
        else:
            assert breakdown["membrane_plane_loss"].item() == 0.0
