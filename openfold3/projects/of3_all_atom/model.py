# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
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

"""
The main inference and training loops for AlphaFold3.
"""

import random
from enum import Enum

import numpy as np
import torch
from ml_collections import ConfigDict
from torch import nn

from openfold3.core.model.feature_embedders.input_embedders import (
    InputEmbedderAllAtom,
    MSAModuleEmbedder,
)
from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
from openfold3.core.model.latent.msa_module import MSAModuleStack
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.core.model.lora import (
    apply_lora_to_linear_modules,
)
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
    centre_random_augmentation,
    create_noise_schedule,
)
from openfold3.core.utils.permutation_alignment import (
    safe_multi_chain_permutation_alignment,
)
from openfold3.core.utils.tensor_utils import add, tensor_tree_map
from openfold3.projects.of3_all_atom.config.model_config import (
    PHYSPPI_TRAINABLE_PREFIXES,
    PHYSPPI_TRAINING_SCOPE,
    validate_physppi_semantic_config,
)
from openfold3.projects.of3_all_atom.physppi.model.conditioning import (
    OptionalPriorConditioner,
)
from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
    validate_optional_prior_features,
)
from openfold3.projects.of3_all_atom.physppi.model.interaction_auxiliary import (
    InteractionDistanceHead,
)
from openfold3.projects.of3_all_atom.physppi.model.membrane_geometry import (
    DirectedMembranePlaneEstimator,
    DirectedPlanePrediction,
    MembraneDiffusionConditioner,
)

MODEL_VERSION = torch.tensor([1, 0, 0], dtype=torch.float32)


class OffloadModules(Enum):
    TEMPLATE_MODULE = "template_module"
    MSA_MODULE = "msa_module"
    CONFIDENCE_HEADS = "confidence_heads"


class OpenFold3(nn.Module):
    """
    Alphafold 3.

    Implements AF3 Algorithm 1 main loop (but with training).
    """

    def __init__(self, config: ConfigDict):
        """
        Args:
            config:
                The model configuration as a ConfigDict object.
        """
        super().__init__()
        validate_physppi_semantic_config(config)
        self.config = config
        self.settings = self.config.settings
        self.shared = self.config.architecture.shared

        self.synced_generator = np.random.default_rng(seed=self.shared.sync_seed)

        self.input_embedder = InputEmbedderAllAtom(
            **self.config.architecture.input_embedder
        )
        physppi_config = getattr(self.config.architecture, "physppi", None)
        physppi_enabled = physppi_config is not None and physppi_config.enabled
        prior_conditioner_config = (
            getattr(physppi_config, "optional_prior_conditioner", None)
            if physppi_enabled
            else None
        )
        prior_conditioner_enabled = bool(
            prior_conditioner_config is None
            or prior_conditioner_config.get("enabled", True)
        )
        self.physppi_conditioning = (
            OptionalPriorConditioner(
                c_s=self.shared.c_s,
                c_z=self.shared.c_z,
            )
            if physppi_enabled and prior_conditioner_enabled
            else None
        )
        membrane_config = (
            getattr(physppi_config, "membrane_geometry", None)
            if physppi_enabled
            else None
        )
        self.membrane_plane_estimator = (
            DirectedMembranePlaneEstimator(
                c_s=self.shared.c_s,
                normal_bins=int(membrane_config.get("normal_error_bins", 16)),
            )
            if membrane_config is not None and membrane_config.enabled
            else None
        )
        self.interaction_distance_head = (
            InteractionDistanceHead(pair_dim=self.shared.c_z)
            if physppi_enabled
            else None
        )

        self.layer_norm_z = LayerNorm(self.shared.c_z)
        self.linear_z = Linear(
            self.shared.c_z, self.shared.c_z, bias=False, init="final"
        )

        self.template_embedder = TemplateEmbedderAllAtom(
            config=self.config.architecture.template
        )

        self.msa_module_embedder = MSAModuleEmbedder(
            **self.config.architecture.msa.msa_module_embedder
        )
        self.msa_module = MSAModuleStack(**self.config.architecture.msa.msa_module)

        self.layer_norm_s = LayerNorm(self.shared.c_s)
        self.linear_s = Linear(
            self.shared.c_s, self.shared.c_s, bias=False, init="final"
        )

        self.pairformer_stack = PairFormerStack(**self.config.architecture.pairformer)

        self.diffusion_module = DiffusionModule(
            config=self.config.architecture.diffusion_module
        )
        # Membrane signed-depth conditioning is attached only after the
        # directed geometry module is enabled.  The static conditioner itself
        # never owns a second diffusion adapter.
        self.diffusion_module.extra_conditioner = (
            MembraneDiffusionConditioner(
                c_s=self.shared.c_s,
                c_z=self.shared.c_z,
                num_depth_bins=16,
                depth_cutoff=float(membrane_config.get("depth_cutoff", 30.0)),
            )
            if membrane_config is not None and membrane_config.enabled
            else None
        )

        self.sample_diffusion = SampleDiffusion(
            **self.config.architecture.sample_diffusion,
            diffusion_module=self.diffusion_module,
        )

        # Confidence and Distogram Heads
        self.aux_heads = AuxiliaryHeadsAllAtom(config=self.config.architecture.heads)

        self.lora_matches = []
        lora_config = getattr(self.config.architecture, "lora", None)
        if lora_config is not None and lora_config.enabled:
            allow_empty_group = int(self.config.architecture.pairformer.no_blocks) < 48
            if lora_config.get("train_bias", False):
                raise ValueError(
                    "PhysPPI LoRA currently supports train_bias=False only"
                )
            if lora_config.get("init", "b_zero") != "b_zero":
                raise ValueError("PhysPPI LoRA currently supports init='b_zero' only")
            groups = lora_config.get("groups")
            if groups:
                for group in groups:
                    self.lora_matches.extend(
                        apply_lora_to_linear_modules(
                            self,
                            target_patterns=list(group["patterns"]),
                            exclude_patterns=list(group.get("exclude_patterns", [])),
                            rank=int(group["rank"]),
                            alpha=float(group["alpha"]),
                            dropout=float(group.get("dropout", 0.0)),
                            max_trainable_lora_params=int(
                                group.get("max_trainable_lora_params", 8_000_000)
                            ),
                            allow_no_matches=allow_empty_group,
                        )
                    )
            else:
                self.lora_matches = apply_lora_to_linear_modules(
                    self,
                    target_patterns=list(lora_config.target_patterns),
                    exclude_patterns=list(lora_config.get("exclude_patterns", [])),
                    rank=int(lora_config.rank),
                    alpha=float(lora_config.alpha),
                    dropout=float(lora_config.dropout),
                    max_trainable_lora_params=int(
                        lora_config.max_trainable_lora_params
                    ),
                )
            if (
                lora_config.get("target_preset") == "physppi_always_on"
                and int(self.config.architecture.pairformer.no_blocks) == 48
            ):
                match_count = len(self.lora_matches)
                trainable_params = sum(
                    match.trainable_params for match in self.lora_matches
                )
                if match_count != 97 or trainable_params != 131_968:
                    raise ValueError(
                        "canonical PhysPPI LoRA injection must produce exactly "
                        "97 matches / 131968 parameters, got "
                        f"{match_count} matches / {trainable_params} parameters"
                    )

        training_scope = self.settings.get("training_scope", "baseline")
        if physppi_enabled and training_scope == "physppi_priors_only":
            self._configure_physppi_trainable_boundary()

        self.register_buffer("version_tensor", MODEL_VERSION)

    def _configure_physppi_trainable_boundary(self) -> None:
        """Freeze OF3 and expose only the PhysPPI/assessment whitelist."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        trainable_modules = [
            self.physppi_conditioning,
            self.membrane_plane_estimator,
            self.interaction_distance_head,
        ]
        diffusion_bridge = getattr(self.diffusion_module, "extra_conditioner", None)
        trainable_modules.append(diffusion_bridge)
        # The assessment branch owns the confidence heads that are active for
        # canonical PhysPPI data.  PDE must stay in this explicit boundary just
        # like pLDDT/PAE/pDockQ; experimentally-resolved and distogram remain
        # outside the adapter-only route.
        for name in ("pairformer_embedding", "plddt", "pae", "pdockq", "pde"):
            trainable_modules.append(getattr(self.aux_heads, name, None))
        for module in trainable_modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

        # Keep this boundary data-independent: optional-slot availability is a
        # batch property and must never change which parameters are trainable.
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and not name.startswith(PHYSPPI_TRAINABLE_PREFIXES)
        ]
        if unexpected:
            raise ValueError(
                "unexpected trainable OpenFold3 parameters: "
                + ", ".join(unexpected[:10])
            )

    def _disable_activation_checkpointing(self):
        """
        Disable activation checkpointing for the TemplateEmbedder, MSAModule,
        and Pairformer.
        """
        self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.msa_module.blocks_per_ckpt = None
        self.pairformer_stack.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        """
        Enable activation checkpointing for the TemplateEmbedder, MSAModule,
        and Pairformer.
        """
        self.template_embedder.template_pair_stack.blocks_per_ckpt = (
            self.config.architecture.template.template_pair_stack.blocks_per_ckpt
        )
        self.msa_module.blocks_per_ckpt = (
            self.config.architecture.msa.msa_module.blocks_per_ckpt
        )
        self.pairformer_stack.blocks_per_ckpt = (
            self.config.architecture.pairformer.blocks_per_ckpt
        )

    def _get_mode_mem_settings(self):
        """
        Get the memory settings for the current mode (training or evaluation).

        Returns:
            mode_mem_settings: Dict of memory settings
        """
        mode_mem_settings = (
            self.settings.memory.train if self.training else self.settings.memory.eval
        )
        return mode_mem_settings

    def _do_inference_offload(self, seq_len: int, module_name: str) -> bool:
        if self.training:
            return False

        offload_settings = self.settings.memory.eval.offload_inference

        is_above_cutoff = (
            offload_settings.token_cutoff is None
            or seq_len > offload_settings.token_cutoff
        )
        offload_inference = offload_settings[module_name] and is_above_cutoff

        return offload_inference

    def _validate_physppi_features_for_batch(self, batch: dict) -> None:
        """Fail loudly when training with enabled PhysPPI but missing priors."""
        physppi_config = getattr(self.config.architecture, "physppi", None)
        enabled = (
            self.physppi_conditioning is not None
            and physppi_config is not None
            and physppi_config.enabled
        )
        mode = "train" if self.training else "inference"
        if enabled:
            validate_optional_prior_features(
                batch=batch,
                mode=mode,
            )

    def _predict_physppi_membrane_plane(
        self,
        *,
        token_repr: torch.Tensor,
        batch: dict,
        token_mask: torch.Tensor,
    ):
        """Estimate a directed plane from mandatory monomer-input geometry."""
        estimator = self.membrane_plane_estimator
        required = (
            "physppi_monomer_atom_positions",
            "physppi_monomer_atom_mask",
            "physppi_token_center_atom_index",
            "physppi_backbone_atom_indices",
            "physppi_backbone_frame_mask",
            "physppi_tm_token_mask",
            "physppi_membrane_on",
        )
        if estimator is None or not all(key in batch for key in required):
            return None
        return estimator(
            token_repr=token_repr,
            monomer_atom_positions=batch["physppi_monomer_atom_positions"],
            monomer_atom_mask=batch["physppi_monomer_atom_mask"],
            token_center_atom_index=batch["physppi_token_center_atom_index"],
            backbone_atom_indices=batch["physppi_backbone_atom_indices"],
            backbone_frame_mask=batch["physppi_backbone_frame_mask"],
            token_mask=token_mask.bool(),
            tm_token_mask=batch["physppi_tm_token_mask"].bool(),
            membrane_on=batch["physppi_membrane_on"],
        )

    @staticmethod
    def clear_autocast_cache():
        if torch.is_autocast_enabled():
            # Sidestep AMP bug (PyTorch issue #65766)
            # Use after no_grad sections just to be safe (i.e. after rollout)
            torch.clear_autocast_cache()

    def run_trunk(
        self, batch: dict, num_cycles: int, inplace_safe: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements Algorithm 1 lines 1-14.

        Args:
            batch:
                Input feature dictionary
            num_cycles:
                Number of cycles to run
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            s_input:
                [*, N_token, C_s_input] Single (input) representation
            s:
                [*, N_token, C_s] Single representation
            z:
                [*, N_token, N_token, C_z] Pair representation
        """
        mode_mem_settings = self._get_mode_mem_settings()

        offload_msa_module = self._do_inference_offload(
            seq_len=batch["token_mask"].shape[-1],
            module_name=OffloadModules.MSA_MODULE.value,
        )

        offload_template_module = self._do_inference_offload(
            seq_len=batch["token_mask"].shape[-1],
            module_name=OffloadModules.TEMPLATE_MODULE.value,
        )

        s_input, s_init, z_init = self.input_embedder(
            batch=batch,
            inplace_safe=inplace_safe,
            use_high_precision_attention=True,
        )

        # s: [*, N_token, C_s]
        # z: [*, N_token, N_token, C_z]
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # token_mask: [*, N_token]
        # pair_mask: [*, N_token, N_token]
        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        self._validate_physppi_features_for_batch(batch)

        if self.physppi_conditioning is not None:
            conditioned = self.physppi_conditioning(
                s_init=s_init,
                z_init=z_init,
                batch=batch,
                token_mask=token_mask.to(dtype=s_init.dtype),
                pair_mask=pair_mask.to(dtype=z_init.dtype),
            )
            s_init, z_init = conditioned.s_init, conditioned.z_init
            # These are derived masks, not GT membrane-plane inputs.  Keeping
            # them on the batch lets the diffusion bridge hard-bypass
            # non-membrane examples without a second static injection.
            batch["physppi_membrane_on"] = conditioned.membrane_on
            batch["physppi_tm_token_mask"] = conditioned.tm_token_mask
            prediction = self._predict_physppi_membrane_plane(
                token_repr=s_init,
                batch=batch,
                token_mask=token_mask,
            )
            if prediction is not None:
                # Keep aligned GT separate; the prediction is reused by the
                # diffusion bridge and optional membrane supervision only.
                batch["physppi_predicted_plane"] = prediction

        is_grad_enabled = torch.is_grad_enabled()

        for cycle_no in range(num_cycles):
            is_final_iter = cycle_no == (num_cycles - 1)

            # Enable grad when we're training, only enable grad on the last cycle
            enable_grad = (
                is_grad_enabled
                and is_final_iter
                and not self.settings.train_confidence_only
            )
            with torch.set_grad_enabled(enable_grad):
                if is_final_iter:
                    self.clear_autocast_cache()

                # [*, N_token, N_token, C_z]
                z = z_init + self.linear_z(self.layer_norm_z(z))

                z = add(
                    z,
                    self.template_embedder(
                        batch=batch,
                        z=z,
                        pair_mask=pair_mask,
                        chunk_size=mode_mem_settings.chunk_size,
                        _mask_trans=True,
                        use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                        use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                        use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                        use_lma=mode_mem_settings.use_lma,
                        inplace_safe=inplace_safe,
                        offload_inference=offload_template_module,
                    ),
                    inplace=inplace_safe,
                )

                m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)

                # Run MSA + pair embeddings through the MsaModule
                # m: [*, N_seq, N_token, C_m]
                # z: [*, N_token, N_token, C_z]
                swiglu_token_cutoff = (
                    mode_mem_settings.msa_module.swiglu_chunk_token_cutoff
                )
                transition_ckpt_chunk_size = (
                    mode_mem_settings.msa_module.swiglu_seq_chunk_size
                    if swiglu_token_cutoff is None or swiglu_token_cutoff > m.shape[-2]
                    else None
                )
                if offload_msa_module:
                    input_tensors = [m, z]
                    del m, z
                    z = self.msa_module.forward_offload(
                        input_tensors,
                        msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                        pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                        chunk_size=mode_mem_settings.chunk_size,
                        transition_ckpt_chunk_size=transition_ckpt_chunk_size,
                        use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                        use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                        use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                        use_lma=mode_mem_settings.use_lma,
                        _mask_trans=True,
                    )

                    del input_tensors, msa_mask
                else:
                    z = self.msa_module(
                        m,
                        z,
                        msa_mask=msa_mask.to(dtype=m.dtype),
                        pair_mask=pair_mask.to(dtype=z.dtype),
                        chunk_size=mode_mem_settings.chunk_size,
                        transition_ckpt_chunk_size=transition_ckpt_chunk_size,
                        use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                        use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                        use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                        use_lma=mode_mem_settings.use_lma,
                        inplace_safe=inplace_safe,
                        _mask_trans=True,
                    )

                    del m, msa_mask

                s = s_init + self.linear_s(self.layer_norm_s(s))
                s, z = self.pairformer_stack(
                    s=s,
                    z=z,
                    single_mask=token_mask.to(dtype=z.dtype),
                    pair_mask=pair_mask.to(dtype=s.dtype),
                    chunk_size=mode_mem_settings.chunk_size,
                    use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                    use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                    use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                    use_lma=mode_mem_settings.use_lma,
                    inplace_safe=inplace_safe,
                    _mask_trans=True,
                )

        del s_init, z_init

        return s_input, s, z

    def _rollout(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> dict:
        """
        Mini diffusion rollout described in section 4.1.
        Implements Algorithm 1 lines 15-18.

        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            si_trunk:
                [*, N_token, C_s] Single representation output from model trunk
            zij_trunk:
                [*, N_token, N_token, C_z] Pair representation output from model trunk
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            Output dictionary containing the predicted trunk embeddings,
            all-atom positions, and confidence/distogram head logits
        """
        mode_mem_settings = self._get_mode_mem_settings()

        offload_confidence_heads = self._do_inference_offload(
            seq_len=batch["token_mask"].shape[-1],
            module_name=OffloadModules.CONFIDENCE_HEADS.value,
        )

        # Determine number of rollout steps and samples depending on training/eval mode
        no_rollout_steps = (
            self.shared.diffusion.no_mini_rollout_steps
            if self.training
            else self.shared.diffusion.no_full_rollout_steps
        )

        no_rollout_samples = (
            self.shared.diffusion.no_mini_rollout_samples
            if self.training
            else self.shared.diffusion.no_full_rollout_samples
        )

        # Determine whether to use zij trunk embedding in confidence heads
        # Only enabled during training
        use_trunk_embedding = (
            random.random() < self.shared.use_confidence_emb_prob
            if self.training
            else True
        )

        # Compute atom positions
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type="cuda", dtype=torch.float32),
        ):
            noise_schedule = create_noise_schedule(
                no_rollout_steps=no_rollout_steps,
                **self.config.architecture.noise_schedule,
                dtype=si_input.dtype,
                device=si_input.device,
            )

            atom_positions_predicted = self.sample_diffusion(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=noise_schedule,
                no_rollout_samples=no_rollout_samples,
                use_conditioning=True,
                chunk_size=mode_mem_settings.chunk_size,
                use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                use_lma=mode_mem_settings.use_lma,
                _mask_trans=True,
            )

            self.clear_autocast_cache()

        output = {
            "si_trunk": si_trunk,
            "zij_trunk": zij_trunk,
            "atom_positions_predicted": atom_positions_predicted,
        }
        if self.interaction_distance_head is not None:
            output["interaction_distance_pred"] = self.interaction_distance_head(
                zij_trunk,
                token_mask=batch["token_mask"],
            )
        prediction = batch.get("physppi_predicted_plane")
        if prediction is not None:
            output["physppi_predicted_membrane_normal"] = prediction.normal
            output["physppi_predicted_membrane_offset"] = prediction.offset
            output["physppi_predicted_membrane_valid"] = prediction.valid

        cast_dtype = torch.float32 if self.training else si_trunk.dtype
        with torch.amp.autocast(device_type="cuda", dtype=cast_dtype):
            # Compute confidence logits
            output.update(
                self.aux_heads(
                    batch=batch,
                    si_input=si_input,
                    output=output,
                    use_zij_trunk_embedding=use_trunk_embedding,
                    chunk_size=mode_mem_settings.chunk_size,
                    use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                    use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                    use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                    use_lma=mode_mem_settings.use_lma,
                    inplace_safe=inplace_safe,
                    offload_inference=offload_confidence_heads,
                    _mask_trans=True,
                )
            )

        return output

    def _train_diffusion(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> dict:
        """
        Run diffusion training over no_samples noised versions of the input structure.

        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            si_trunk:
                [*, N_token, C_s] Single representation output from model trunk
            zij_trunk:
                [*, N_token, N_token, C_z] Pair representation output from model trunk

        Returns:
            Output dictionary containing the following keys:
                "noise_level" ([*])
                    Noise level at a diffusion step
                "atom_positions_diffusion" ([*, N_samples, N_atom, 3]):
                    Predicted atom positions
        """
        xl_gt = batch["ground_truth"]["atom_positions"]
        atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]

        # Sample noise schedule for training
        no_samples = self.shared.diffusion.no_samples
        batch_size, n_atom = xl_gt.shape[0], xl_gt.shape[-2]
        device, dtype = xl_gt.device, xl_gt.dtype

        xl_gt = xl_gt.tile((1, no_samples, 1, 1))
        xl_gt = centre_random_augmentation(xl=xl_gt, atom_mask=atom_mask_gt)
        # Membrane GT remains label-only; diffusion receives the common batch
        # and the bridge uses the predicted/noisy plane estimate.
        diffusion_batch = batch
        n = torch.randn((batch_size, no_samples), device=device, dtype=dtype)
        t = self.shared.diffusion.sigma_data * torch.exp(-1.2 + 1.5 * n)

        # Sample noise
        noise = t[..., None, None] * torch.randn(
            (batch_size, no_samples, n_atom, 3), device=device, dtype=dtype
        )

        # Sample atom positions
        xl_noisy = xl_gt + noise

        use_conditioning = random.random() < self.shared.diffusion.use_conditioning_prob

        # Run diffusion module
        xl = self.diffusion_module(
            batch=diffusion_batch,
            xl_noisy=xl_noisy,
            token_mask=batch["token_mask"],
            atom_mask=atom_mask_gt,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            xl_clean_context=xl_noisy,
            use_conditioning=use_conditioning,
            use_high_precision_attention=True,
            _mask_trans=True,
        )

        output = {
            "noise_level": t,
            "atom_positions_diffusion": xl,
        }

        return output

    def forward(self, batch: dict) -> tuple[dict, dict]:
        """
        Args:
            batch:
                Dictionary of arguments outlined in supplement section 2.8. Keys must
                include the official names of the features in Table 5, as well as
                additional features noted with a *.

                Features:
                    "residue_index" ([*, N_token])
                        Residue number in the token’s original input chain
                    "token_index" ([*, N_token])
                        Token number
                    "asym_id" ([*, N_token])
                        Unique integer for each distinct chain
                    "entity_id" ([*, N_token])
                        Unique integer for each distinct sequence
                    "sym_id" ([*, N_token])
                        Unique integer within chains of this sequence
                    "restype" ([*, N_token, 32])
                        One-hot encoding of the sequence
                    "is_protein" ([*, N_token])
                        Molecule type mask
                    "is_rna" ([*, N_token])
                        Molecule type mask
                    "is_dna" ([*, N_token])
                        Molecule type mask
                    "is_ligand" ([*, N_token])
                        Molecule type mask
                    "ref_pos" ([*, N_atom, 3])
                        Atom positions (reference conformer)
                    "ref_mask" ([*, N_atom])
                        Atom mask (reference conformer)
                    "ref_element" ([*, N_atom, 128])
                        One-hot encoding of the element atomic
                        number (reference conformer)
                    "ref_charge" ([*, N_atom])
                        Atom charge (reference conformer)
                    "ref_atom_name_chars" ([*, N_atom, 4, 64])
                        One-hot encoding of the unique atom names (reference conformer)
                    "ref_space_uid" ([*, N_atom])
                        Numerical encoding of the chain id and residue
                        index (reference conformer)
                    "msa": ([*, N_msa, N_token, 32])
                        One-hot encoding of the processed MSA
                    "has_deletion" ([*, N_msa, N_token])
                        Binary feature indicating if there is a deletion to
                        the left of each MSA position
                    "deletion_value" ([*, N_msa, N_token])
                        Raw deletion counts
                    "profile" ([*, N_token, 32])
                        Distribution across restypes in the main MSA
                    "deletion_mean" ([*, N_token])
                        Mean number of deletions at each position in the main MSA
                    "template_restype": ([*, N_templ, N_token, 32])
                        One-hot encoding of the template sequence
                    "template_pseudo_beta_mask" ([*, N_templ, N_token])
                        Mask for template C_beta atoms (C_alpha for glycine)
                    "template_backbone_frame_mask" ([*, N_templ, N_token])
                        Mask indicating if required template atoms exist to
                        compute frames
                    "template_distogram" ([*, N_templ, N_token, N_token, 39])
                        A one-hot pairwise feature indicating the distance between
                        C_beta atoms (C_alpha for glycine) in the template
                    "template_unit_vector"([*, N_templ, N_token, N_token, 3])
                        The unit vector between pairs of C_alpha atoms within
                        the local frame of each template residue
                    "token_bonds" ([*, N_token, N_token])
                        A 2D matrix indicating if there is a bond between
                        any atom in token i and token j
                    *"num_atoms_per_token" ([*, N_token])
                        Number of atoms per token
                    *"start_atom_index" ([*, N_token])
                        Starting atom index in each token
                    *"token_mask" ([*, N_token])
                        Token-level mask
                    *"msa_mask" ([*, N_msa, N_token])
                        MSA mask
                    *"num_paired_seqs" ([*])
                        Number of main MSA seqs used in MSA sampling (non-uniprot)
                    "ground_truth" (Dict):
                        "residue_index" ([*, N_token])
                            Residue number in the token’s original input chain
                        "token_index" ([*, N_token])
                            Token number
                        "asym_id" ([*, N_token])
                            Unique integer for each distinct chain
                        "entity_id" ([*, N_token])
                            Unique integer for each distinct sequence
                        "is_protein" ([*, N_token])
                            Molecule type mask
                        "is_ligand" ([*, N_token])
                            Molecule type mask
                        "token_bonds" ([*, N_token, N_token])
                            A 2D matrix indicating if there is a bond between
                            any atom in token i and token j
                        *"num_atoms_per_token" ([*, N_token])
                            Number of atoms per token
                        *"token_mask" ([*, N_token])
                            Token-level mask
                        *"atom_positions" ([*, N_atom, 3])
                            Ground truth atom positions for training
                        *"atom_resolved_mask" ([*, N_atom])
                            Mask for ground truth atom positions
        Returns:
            batch: Updated batch dictionary post permutation alignment
            output: Output dictionary containing the following keys:
                "si_trunk" ([*, N_token, C_s]):
                    Single representation output from model trunk
                "zij_trunk" ([*, N_token, N_token, C_z]):
                    Pair representation output from model trunk
                "atom_positions_predicted" ([*, N_atom, 3]):
                    Predicted atom positions
                "plddt_logits" ([*, N_atom, 50]):
                    Predicted binned PLDDT logits
                "pae_logits" ([*, N_token, N_token, 64]):
                    Predicted binned PAE logits
                "pdockq_logits" ([*, N_sample, 50]):
                    Candidate-specific binned pDockQ logits
                "pdockq_expectation" ([*, N_sample]):
                    Candidate-specific expected pDockQ
                "distogram_logits" ([*, N_token, N_token, 64]):
                    Predicted binned distogram logits
                "noise_level" ([*])
                    Training only, noise level at a diffusion step
                "atom_positions_diffusion" ([*, N_samples, N_atom, 3]):
                    Training only, predicted atom positions

        """
        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # If training, we sample the number of recycles
        # This is the additional number of iterations through the trunk
        num_recycles = (
            int(
                self.synced_generator.integers(low=0, high=self.shared.num_recycles + 1)
            )
            if self.training
            else self.shared.num_recycles
        )
        num_cycles = num_recycles + 1

        output = {"recycles": num_recycles}

        # Compute representations
        si_input, si_trunk, zij_trunk = self.run_trunk(
            batch=batch, num_cycles=num_cycles, inplace_safe=inplace_safe
        )

        # Expand sampling dimension for rollout and diffusion
        si_input = si_input.unsqueeze(1)
        si_trunk = si_trunk.unsqueeze(1)
        zij_trunk = zij_trunk.unsqueeze(1)

        # Expand sampling dimension for batch features
        # Exclude ref_space_uid_to_perm feature since this
        # does not have a proper batch dimension
        ref_space_uid_to_perm = batch.pop("ref_space_uid_to_perm", None)
        # ``tensor_tree_map`` recursively visits NamedTuple values.  Preserve
        # the directed membrane prediction object as a structured prediction
        # rather than converting it to a plain tuple before rollout/loss.
        physppi_predicted_plane = batch.pop("physppi_predicted_plane", None)
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm
        if physppi_predicted_plane is not None:
            if isinstance(physppi_predicted_plane, DirectedPlanePrediction):
                physppi_predicted_plane = DirectedPlanePrediction(
                    *(value.unsqueeze(1) for value in physppi_predicted_plane)
                )
            batch["physppi_predicted_plane"] = physppi_predicted_plane

        # Mini rollout
        rollout_output = self._rollout(
            batch=batch,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            inplace_safe=inplace_safe,
        )

        output.update(rollout_output)

        # Apply permutation alignment for training and validation
        if "ground_truth" in batch:
            # Update the ground-truth coordinates/mask in-place with the correct
            # permutation (and optionally disable losses in case of a
            # critical error)
            with (
                torch.no_grad(),
                torch.amp.autocast(device_type="cuda", dtype=torch.float32),
            ):
                safe_multi_chain_permutation_alignment(
                    batch=batch,
                    atom_positions_predicted=output["atom_positions_predicted"],
                )

                self.clear_autocast_cache()

            if self.training and not self.settings.train_confidence_only:
                # Run training step (if necessary)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    diffusion_output = self._train_diffusion(
                        batch=batch,
                        si_input=si_input,
                        si_trunk=si_trunk,
                        zij_trunk=zij_trunk,
                    )

                    output.update(diffusion_output)

                    if self.settings.training_scope == PHYSPPI_TRAINING_SCOPE:
                        mode_mem_settings = self._get_mode_mem_settings()
                        output.update(
                            self.aux_heads.forward_pdockq_candidates(
                                batch=batch,
                                si_input=si_input,
                                output=output,
                                candidate_positions=output["atom_positions_diffusion"],
                                use_zij_trunk_embedding=True,
                                chunk_size=mode_mem_settings.chunk_size,
                                use_deepspeed_evo_attention=(
                                    mode_mem_settings.use_deepspeed_evo_attention
                                ),
                                use_triton_triangle_kernels=(
                                    mode_mem_settings.use_triton_triangle_kernels
                                ),
                                use_cueq_triangle_kernels=(
                                    mode_mem_settings.use_cueq_triangle_kernels
                                ),
                                use_lma=mode_mem_settings.use_lma,
                                inplace_safe=inplace_safe,
                                _mask_trans=True,
                            )
                        )

        # Memory fragmentation can become a big problem at larger crop sizes
        # due to different sizes of msa/all-atom tensors used between steps
        # Clear the cache between steps if unallocated reserved mem is high
        if self.settings.clear_cache_between_steps:
            torch.cuda.empty_cache()

        return batch, output
