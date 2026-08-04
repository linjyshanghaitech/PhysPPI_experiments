import copy

import pytest
import torch

from openfold3.core.model.lora import (
    LoRALinear,
    apply_lora_to_linear_modules,
    lora_parameter_names,
)
from openfold3.core.model.primitives import Linear
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3


def test_lora_linear_zero_init_preserves_base_output_and_freezes_base():
    torch.manual_seed(0)
    base = Linear(5, 3)
    wrapped = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(7, 5)

    torch.testing.assert_close(wrapped(x), base(x))
    assert wrapped.base.weight.requires_grad is False
    assert wrapped.lora_A.requires_grad is True
    assert wrapped.lora_B.requires_grad is True

    wrapped.lora_B.data.normal_(0, 0.1)
    assert not torch.allclose(wrapped(x), base(x))


def test_apply_lora_rejects_empty_target_matches():
    model = torch.nn.Sequential(Linear(4, 4))

    with pytest.raises(ValueError, match="matched zero"):
        apply_lora_to_linear_modules(
            model,
            [r"does_not_match"],
            rank=2,
            alpha=4,
        )


def test_apply_lora_matches_configured_physppi_groups_exactly():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.pairformer.no_blocks = 48
    cfg.architecture.lora.enabled = True

    model = OpenFold3(cfg)
    matches = model.lora_matches

    template_matches = [
        match for match in matches if match.name == "template_embedder.linear_t"
    ]
    pairformer_matches = [
        match for match in matches if match.name != "template_embedder.linear_t"
    ]
    assert len(matches) == 97
    assert sum(match.trainable_params for match in matches) == 131968
    assert sum(match.trainable_params for match in template_matches) <= 131968
    assert sum(match.trainable_params for match in pairformer_matches) <= 131968
    assert matches[0].name == "template_embedder.linear_t"
    assert all(
        match.name.startswith("pairformer_stack.blocks.4") for match in matches[1:]
    )
    assert lora_parameter_names(model)


def test_lora_parameter_names_requires_exact_parameter_leaf_names():
    module = torch.nn.Module()
    module.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1)))
    module.register_parameter("lora_B", torch.nn.Parameter(torch.ones(1)))
    module.register_parameter("lora_A_shadow", torch.nn.Parameter(torch.ones(1)))
    module.register_parameter("lora_B_extra", torch.nn.Parameter(torch.ones(1)))

    assert lora_parameter_names(module) == {"lora_A", "lora_B"}


def test_lora_target_names_are_not_gated_by_optional_prior_state():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_priors_only"
    cfg.architecture.lora.enabled = True
    cfg.architecture.pairformer.no_blocks = 1
    model = OpenFold3(cfg)

    assert model.lora_matches
    assert all(match.trainable_params > 0 for match in model.lora_matches)
    assert lora_parameter_names(model)
