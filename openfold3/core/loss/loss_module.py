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

"""Main loss modules."""

import logging
import math

import torch
import torch.nn as nn

from openfold3.core.loss.confidence import confidence_loss
from openfold3.core.loss.diffusion import diffusion_loss
from openfold3.core.loss.distogram import all_atom_distogram_loss
from openfold3.core.loss.membrane import directed_membrane_plane_loss
from openfold3.core.loss.pdockq import (
    pdockq_distribution_loss,
    pdockq_pairwise_ranking_loss,
)
from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map
from openfold3.projects.of3_all_atom.physppi.model.interaction_auxiliary import (
    interaction_auxiliary_loss,
)

logger = logging.getLogger(__name__)


class OpenFold3Loss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super().__init__()

        # Loss config
        self.config = config

    def loss(self, batch, output):
        cum_loss = 0.0
        losses = {}

        l_confidence, l_confidence_breakdown = confidence_loss(
            batch=batch, output=output, **self.config.confidence
        )
        losses.update(l_confidence_breakdown)

        if l_confidence_breakdown:
            losses["confidence_loss"] = l_confidence.detach().clone()

        # pDockQ supervision is optional at the batch boundary because legacy
        # datasets do not provide candidate DockQ targets.  When present, both
        # distribution (50-bin CE) or regression (MSE) and within-complex
        # ranking losses retain the candidate axis and mask invalid candidates.
        pdockq_logits = output.get("pdockq_logits")
        pdockq_target = batch.get("physppi_pdockq_target")
        pdockq_valid = batch.get("physppi_pdockq_valid")
        if (
            pdockq_logits is not None
            and pdockq_target is not None
            and pdockq_valid is not None
        ):
            if pdockq_logits.shape[-1] == 50:
                l_pdockq = pdockq_distribution_loss(
                    pdockq_logits,
                    pdockq_target.to(
                        device=pdockq_logits.device, dtype=pdockq_logits.dtype
                    ),
                    pdockq_valid.to(device=pdockq_logits.device),
                )
            else:
                expectation = output.get("pdockq_expectation")
                target_score = batch["physppi_pdockq_target_score"].to(
                    device=pdockq_logits.device, dtype=pdockq_logits.dtype
                )
                valid_f = pdockq_valid.to(
                    device=pdockq_logits.device, dtype=pdockq_logits.dtype
                )
                if expectation is not None:
                    se = (expectation - target_score) ** 2
                    l_pdockq = (se * valid_f).sum() / valid_f.sum().clamp_min(1.0)
                else:
                    se = (pdockq_logits.squeeze(-1) - target_score) ** 2
                    l_pdockq = (se * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            expectation = output.get("pdockq_expectation")
            if expectation is not None:
                l_pdockq = l_pdockq + pdockq_pairwise_ranking_loss(
                    expectation,
                    batch["physppi_pdockq_target_score"].to(
                        device=expectation.device, dtype=expectation.dtype
                    ),
                    pdockq_valid.to(device=expectation.device),
                )
            losses["pdockq_loss"] = l_pdockq.detach().clone()
            cum_loss = cum_loss + l_pdockq

        if "physppi_membrane_gt_normal" in batch:
            membrane_cfg = self.config.get("membrane", {})
            membrane_include_in_total = membrane_cfg.get(
                "include_in_total_loss",
                True,
            )
            membrane_loss_kwargs = {
                name: value
                for name, value in membrane_cfg.items()
                if name != "include_in_total_loss"
            }
            if membrane_include_in_total:
                l_membrane = directed_membrane_plane_loss(
                    batch=batch,
                    output=output,
                    **membrane_loss_kwargs,
                )
                cum_loss = cum_loss + l_membrane
            else:
                with torch.no_grad():
                    l_membrane = directed_membrane_plane_loss(
                        batch=batch,
                        output=output,
                        **membrane_loss_kwargs,
                    )
            losses["membrane_plane_loss"] = l_membrane.detach().clone()

        interaction_prediction = output.get("interaction_distance_pred")
        if interaction_prediction is not None and (
            "physppi_interaction_aux_valid" in batch
        ):
            interaction_cfg = self.config.get("interaction_auxiliary", {})
            interaction_include_in_total = interaction_cfg.get(
                "include_in_total_loss",
                True,
            )
            if interaction_include_in_total:
                l_interaction = interaction_auxiliary_loss(
                    batch=batch,
                    output=output,
                    distance_weight=interaction_cfg.get("distance_weight", 1.0),
                )
                cum_loss = cum_loss + l_interaction
            else:
                with torch.no_grad():
                    l_interaction = interaction_auxiliary_loss(
                        batch=batch,
                        output=output,
                        distance_weight=interaction_cfg.get("distance_weight", 1.0),
                    )
            losses["interaction_auxiliary_loss"] = l_interaction.detach().clone()

        # Weighted in confidence_loss()
        cum_loss = cum_loss + l_confidence

        # Do not compute diffusion/distogram losses if only training confidence heads
        if not self.config.train_confidence_only:
            atom_positions_diffusion = output.get("atom_positions_diffusion")
            if atom_positions_diffusion is not None:
                # Compute diffusion losses
                l_diffusion, l_diffusion_breakdown = diffusion_loss(
                    batch=batch,
                    x=atom_positions_diffusion,
                    t=output["noise_level"],
                    **self.config.diffusion,
                )
                losses.update(l_diffusion_breakdown)

                if l_diffusion_breakdown:
                    losses["diffusion_loss"] = l_diffusion.detach().clone()

                # Weighted in diffusion_loss()
                cum_loss = cum_loss + l_diffusion

            # Compute distogram loss
            l_distogram, l_distogram_breakdown = all_atom_distogram_loss(
                batch=batch, logits=output["distogram_logits"], **self.config.distogram
            )
            losses.update(l_distogram_breakdown)

            if l_distogram_breakdown:
                losses["scaled_distogram_loss"] = l_distogram.detach().clone()

            # Weighted in all_atom_distogram_loss()
            cum_loss = cum_loss + l_distogram

        losses["loss"] = cum_loss.detach().clone()

        return cum_loss, losses

    def loss_chunked(self, batch, output, eps=1e-9):
        atom_positions_predicted = output["atom_positions_predicted"]
        batch_dims = atom_positions_predicted.shape[:-2]
        num_samples = batch_dims[-1]

        loss_per_sample_list = []
        loss_breakdown_per_sample_list = []
        for idx in range(math.prod(batch_dims)):

            def fetch_cur_sample(t):
                feat_dims = t.shape[2:]
                t = t.expand(-1, num_samples, *((-1,) * len(feat_dims)))
                t = t.reshape(-1, *feat_dims)
                return t[idx : idx + 1]  # noqa: B023

            cur_batch = tensor_tree_map(fetch_cur_sample, batch, strict_type=False)
            cur_output = tensor_tree_map(fetch_cur_sample, output, strict_type=False)

            loss_sample, loss_breakdown_sample = self.loss(
                batch=cur_batch,
                output=cur_output,
            )
            loss_per_sample_list.append(loss_sample)
            loss_breakdown_per_sample_list.append(loss_breakdown_sample)

        def accum_loss(l: list):
            l = torch.stack(l)
            return l.sum() / (l.shape[0] + eps)

        cum_loss = accum_loss(loss_per_sample_list)
        losses = dict_multimap(accum_loss, loss_breakdown_per_sample_list)

        return cum_loss, losses

    def forward(self, batch, output, _return_breakdown=False):
        """
        Args:
            batch:
                Dict containing input tensors
            output:
                Dict containing output tensors
                (see openfold3/openfold3/model_implementations/of3_all_atom/model.py
                for a list items in batch and output)
            _return_breakdown:
                If True, also return a dictionary of individual
                loss components
        Returns:
            cum_loss: Scalar tensor representing the total loss
            losses: Dict containing individual loss components
        """

        # Having to chunk validation losses per sample is really only
        # needed when training on 40gb GPUs
        num_atoms = output["atom_positions_predicted"].shape[-2]
        apply_per_sample = (
            not torch.is_grad_enabled()
            and self.config.low_mem_validation
            and self.config.per_sample_atom_cutoff is not None
            and num_atoms > self.config.per_sample_atom_cutoff
        )
        if not torch.is_grad_enabled() and apply_per_sample:
            loss, loss_breakdown = self.loss_chunked(batch, output)
        else:
            loss, loss_breakdown = self.loss(batch, output)

        if not _return_breakdown:
            return loss

        return loss, loss_breakdown
