# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
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

import importlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from openfold3.core.model.heads.pdockq import PDockQHead
from openfold3.core.model.heads.prediction_heads import (
    DistogramHead,
    PairformerEmbedding,
    PerResidueLDDTAllAtom,
    PredictedAlignedErrorHead,
    PredictedDistanceErrorHead,
)
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_representative_atoms,
)

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed


class AuxiliaryHeadsAllAtom(nn.Module):
    """
    Auxiliary head for OF3
    Implements AF3 Algorithm 31 with main inference loop (Algorithm 1) line 16 - 17.
    """

    def __init__(self, config):
        """
        Args:
            config: ConfigDict with following keys
                "pairformer_embedding": Pairformer embedding config
                "pae": PAE config
                "pde": PDE config
                "lddt": LDDT config
                "distogram": Distogram config
                "pdockq": pDockQ config
        """
        super().__init__()
        self.config = config
        self.max_atoms_per_token = config.max_atoms_per_token
        self.per_sample_token_cutoff = config.per_sample_token_cutoff

        self.pairformer_embedding = PairformerEmbedding(
            **self.config["pairformer_embedding"],
        )

        # PDE remains an active confidence target for both canonical PhysPPI
        # datasets.  Experimentally-resolved confidence is intentionally not
        # constructed on this route because it has no active supervision.
        self.pde = PredictedDistanceErrorHead(
            **self.config["pde"],
        )

        self.plddt = PerResidueLDDTAllAtom(
            **self.config["lddt"],
        )

        self.distogram = DistogramHead(
            **self.config["distogram"],
        )

        self.pdockq = PDockQHead(
            c_s=self.config["pdockq"]["c_s"],
            c_z=self.config["pdockq"]["c_z"],
            no_bins=self.config["pdockq"].get("no_bins", 50),
        )
        self.pdockq_candidate_chunk_size = int(
            self.config["pdockq"].get("candidate_chunk_size", 1)
        )
        if self.pdockq_candidate_chunk_size < 1:
            raise ValueError("pDockQ candidate_chunk_size must be positive")

        # DDP does not allow unused parameters without performance hit
        # Only initialize PAE head if enabled or using DeepSpeed
        deepspeed_is_initialized = (
            deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        )
        if self.config.pae.enabled or deepspeed_is_initialized:
            self.pae = PredictedAlignedErrorHead(
                **self.config["pae"],
            )

    def forward(
        self,
        batch: dict,
        si_input: torch.Tensor,
        output: dict,
        use_zij_trunk_embedding: bool,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        offload_inference: bool = False,
        _mask_trans: bool = True,
    ):
        """
        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            output:
                Dict containing outputs
                    "si_trunk" ([*, N_token, C_s]):
                        Single representation output from model trunk
                    "zij_trunk" ([*, N_token, N_token, C_z]):
                        Pair representation output from model trunk
                    "atom_positions_predicted" ([*, N_atom, 3]):
                        Predicted atom positions
            use_zij_trunk_embedding:
                Whether to use the zij trunk embedding in the confidence Pairformer
                embedding.
            chunk_size:
                Inference-time subbatch size. Associated with PairFormer embedding.
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_cueq_triangle_kernels:
                Whether to use cuEq triangle attention kernel.
                Mutually exclusive with use_lma
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            offload_inference:
                Whether to offload some computation to CPU
            _mask_trans:
                Whether to mask the output of the transition layers

        Returns:
            aux_out:
                Dict containing following keys:
                    "plddt_logits" ([*, N_atom, 50]):
                        Predicted binned PLDDT logits
                    "pae_logits" ([*, N_token, N_token, 64]):
                        Predicted binned PAE logits
                    "pde_logits" ([*, N_token, N_token, 64]):
                        Predicted binned PDE logits
                    "distogram_logits" ([*, N_token, N_token, 64]):
                        Predicted binned distogram logits
        Note:
            Previous implementations of losses include softmax so all
            heads return logits.
        """
        aux_out = {}

        out_dtype = output["atom_positions_predicted"].dtype
        si = output["si_trunk"]
        zij = output["zij_trunk"]
        atom_positions_predicted = output["atom_positions_predicted"].to(dtype=si.dtype)

        # Distogram head: Main loop (Algorithm 1), line 17
        distogram_logits = self.distogram(z=zij)
        aux_out["distogram_logits"] = distogram_logits

        # Stop grad
        si_input = si_input.detach().clone()
        si = si.detach().clone()
        zij = zij.detach().clone()
        atom_positions_predicted = atom_positions_predicted.detach().clone()

        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Get representative atoms
        repr_x_pred, repr_x_mask = get_token_representative_atoms(
            batch=batch, x=atom_positions_predicted, atom_mask=batch["atom_mask"]
        )

        num_samples = repr_x_pred.shape[-3]
        apply_per_sample = (
            not torch.is_grad_enabled()
            and num_samples > 1
            and self.per_sample_token_cutoff is not None
            and repr_x_pred.shape[-2] > self.per_sample_token_cutoff
        )
        out_device = atom_positions_predicted.device

        if not use_zij_trunk_embedding:
            zij = zij * 0

        # Embed trunk outputs
        # If offload_inference is enabled, si and zij will be returned on the CPU
        si, zij = self.pairformer_embedding(
            si_input=si_input,
            si=si,
            zij=zij,
            x_pred=repr_x_pred,
            single_mask=repr_x_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_triton_triangle_kernels=use_triton_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            offload_inference=offload_inference,
            _mask_trans=_mask_trans,
            apply_per_sample=apply_per_sample,
        )
        # Confidence rollout can contain several diffusion candidates while
        # the shared input features carry one asymmetry-id row.  Broadcast the
        # latter across the candidate axis before pDockQ pooling.
        asym_id = batch["asym_id"]
        if asym_id.ndim == 2:
            asym_id = asym_id.unsqueeze(1)
        if asym_id.ndim == 3 and asym_id.shape[1] == 1 and si.ndim == 4:
            asym_id = asym_id.expand(-1, si.shape[1], -1)

        # Candidate axis is preserved by the confidence Pairformer.  The
        # pDockQ head pools each candidate independently and never averages
        # diffusion samples together.
        aux_out["pdockq_logits"] = self.pdockq(
            si=si.to(device=out_device),
            z=zij.to(device=out_device),
            asym_id=asym_id.to(device=out_device),
            token_mask=token_mask.to(device=out_device),
        )
        aux_out["pdockq_expectation"] = self.pdockq.expectation(
            aux_out["pdockq_logits"]
        )

        # Get atom mask padded to MAX_ATOMS_PER_TOKEN for the flat atom
        # representation used by the retained pLDDT head.
        max_atom_per_token_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=token_mask,
            max_num_atoms_per_token=self.max_atoms_per_token,
        )
        # Expand to match sample dimension
        max_atom_per_token_mask = max_atom_per_token_mask.expand(
            (*atom_positions_predicted.shape[:-2], -1)
        )

        si = si.to(device=out_device)
        aux_out["plddt_logits"] = self.plddt(
            s=si, max_atom_per_token_mask=max_atom_per_token_mask
        )

        # zij is moved back to GPU after the single rep confidence heads
        # because building the max_atom_per_token_mask uses a lot of memory
        zij = zij.to(device=out_device)

        pde_logits = self.pde(zij, apply_per_sample=apply_per_sample)

        if self.config.pae.enabled:
            # Keep only one pairwise confidence tensor resident during
            # inference when offloading is enabled.
            offload_device = "cpu" if offload_inference else out_device
            pde_logits = pde_logits.to(device=offload_device)
            aux_out["pae_logits"] = self.pae(zij, apply_per_sample=apply_per_sample)

        del zij

        aux_out["pde_logits"] = pde_logits.to(device=out_device)

        aux_out = {k: v.to(dtype=out_dtype) for k, v in aux_out.items()}

        return aux_out

    def _forward_pdockq_candidate_chunk(
        self,
        *,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        repr_x_pred: torch.Tensor,
        repr_x_mask: torch.Tensor,
        token_mask: torch.Tensor,
        asym_id: torch.Tensor,
        use_zij_trunk_embedding: bool,
        chunk_size: int | None,
        use_deepspeed_evo_attention: bool,
        use_cueq_triangle_kernels: bool,
        use_triton_triangle_kernels: bool,
        use_lma: bool,
        inplace_safe: bool,
        _mask_trans: bool,
    ) -> torch.Tensor:
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        if not use_zij_trunk_embedding:
            zij = zij * 0
        si, zij = self.pairformer_embedding(
            si_input=si_input,
            si=si,
            zij=zij,
            x_pred=repr_x_pred,
            single_mask=repr_x_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_triton_triangle_kernels=use_triton_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )
        return self.pdockq(
            si=si,
            z=zij,
            asym_id=asym_id,
            token_mask=token_mask,
        )

    def forward_pdockq_candidates(
        self,
        *,
        batch: dict,
        si_input: torch.Tensor,
        output: dict,
        candidate_positions: torch.Tensor,
        use_zij_trunk_embedding: bool,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Assess every training diffusion candidate with the shared trunk."""

        if candidate_positions.ndim != 4:
            raise ValueError(
                "training diffusion candidates must have shape "
                f"(batch, candidates, atoms, 3), got {candidate_positions.shape}"
            )
        out_dtype = candidate_positions.dtype
        si_input = si_input.detach()
        si = output["si_trunk"].detach()
        zij = output["zij_trunk"].detach()
        candidate_positions = candidate_positions.detach().to(dtype=si.dtype)

        repr_x_pred, repr_x_mask = get_token_representative_atoms(
            batch=batch,
            x=candidate_positions,
            atom_mask=batch["atom_mask"],
        )
        candidates = repr_x_pred.shape[1]
        if repr_x_mask.ndim == 2:
            repr_x_mask = repr_x_mask.unsqueeze(1)
        if repr_x_mask.shape[1] == 1:
            repr_x_mask = repr_x_mask.expand(-1, candidates, -1)
        elif repr_x_mask.shape[1] != candidates:
            raise ValueError(
                "representative mask candidate axis must be 1 or match "
                f"diffusion candidates, got {repr_x_mask.shape[1]} and "
                f"{candidates}"
            )
        token_mask = batch["token_mask"]
        if token_mask.ndim == 2:
            token_mask = token_mask.unsqueeze(1)
        if token_mask.shape[1] == 1:
            token_mask = token_mask.expand(-1, candidates, -1)
        elif token_mask.shape[1] != candidates:
            raise ValueError(
                "token_mask candidate axis must be 1 or match diffusion "
                f"candidates, got {token_mask.shape[1]} and {candidates}"
            )
        asym_id = batch["asym_id"]
        if asym_id.ndim == 2:
            asym_id = asym_id.unsqueeze(1)
        if asym_id.ndim == 3 and asym_id.shape[1] == 1:
            asym_id = asym_id.expand(-1, candidates, -1)
        elif asym_id.ndim != 3 or asym_id.shape[1] != candidates:
            raise ValueError(
                "asym_id candidate axis must be 1 or match diffusion "
                f"candidates, got {asym_id.shape} and {candidates}"
            )

        logits_chunks = []
        for start in range(0, candidates, self.pdockq_candidate_chunk_size):
            end = min(start + self.pdockq_candidate_chunk_size, candidates)

            def chunk_forward(
                chunk_repr_x_pred,
                chunk_repr_x_mask,
                chunk_token_mask,
                chunk_asym_id,
            ):
                return AuxiliaryHeadsAllAtom._forward_pdockq_candidate_chunk(
                    self,
                    si_input=si_input,
                    si=si,
                    zij=zij,
                    repr_x_pred=chunk_repr_x_pred,
                    repr_x_mask=chunk_repr_x_mask,
                    token_mask=chunk_token_mask,
                    asym_id=chunk_asym_id,
                    use_zij_trunk_embedding=use_zij_trunk_embedding,
                    chunk_size=chunk_size,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                    use_triton_triangle_kernels=use_triton_triangle_kernels,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    _mask_trans=_mask_trans,
                )

            chunk_args = (
                repr_x_pred[:, start:end],
                repr_x_mask[:, start:end],
                token_mask[:, start:end],
                asym_id[:, start:end],
            )
            if getattr(self, "training", False) and torch.is_grad_enabled():
                chunk_logits = checkpoint(
                    chunk_forward,
                    *chunk_args,
                    use_reentrant=False,
                )
            else:
                chunk_logits = chunk_forward(*chunk_args)
            logits_chunks.append(chunk_logits)

        logits = torch.cat(logits_chunks, dim=1)
        expectation = self.pdockq.expectation(logits)
        return {
            "pdockq_logits": logits.to(dtype=out_dtype),
            "pdockq_expectation": expectation.to(dtype=out_dtype),
        }
