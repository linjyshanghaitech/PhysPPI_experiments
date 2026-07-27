import copy

import pytest

from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3


def test_lora_config_is_always_on_for_physppi_domain_adaptation():
    cfg = copy.deepcopy(model_config)

    assert cfg.settings.training_scope == "baseline"
    assert cfg.architecture.lora.enabled is True
    assert cfg.architecture.lora.target_patterns == []
    assert cfg.architecture.lora.rank == 8
    assert cfg.architecture.lora.alpha == 16
    assert len(cfg.architecture.lora.groups) == 2
    assert cfg.architecture.lora.target_preset == "physppi_always_on"
    assert cfg.architecture.lora.train_bias is False
    assert cfg.architecture.lora.init == "b_zero"
    assert [
        (
            tuple(group["patterns"]),
            group["rank"],
            group["alpha"],
            group["dropout"],
            group["max_trainable_lora_params"],
        )
        for group in cfg.architecture.lora.groups
    ] == [
        (
            (r"^template_embedder\.linear_t$",),
            2,
            4,
            0.0,
            131_968,
        ),
        (
            (
                r"^pairformer_stack\.blocks\.(4[0-7])\.pair_stack\.tri_att_(start|end)\.mha\.linear_(q|k|v|o)$",
                r"^pairformer_stack\.blocks\.(4[0-7])\.pair_stack\.pair_transition\.(swiglu\.linear_(a|b)|linear_out)$",
                r"^pairformer_stack\.blocks\.(4[0-7])\.attn_pair_bias\.linear_z$",
            ),
            4,
            8,
            0.05,
            131_968,
        ),
    ]


def test_lora_disabled_adds_no_lora_modules_or_state_keys():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = False
    cfg.architecture.lora.enabled = False
    cfg.architecture.pairformer.no_blocks = 1
    model = OpenFold3(cfg)

    module_names = [name for name, _ in model.named_modules()]
    state_keys = list(model.state_dict())

    assert getattr(model, "physppi_conditioning", None) is None
    assert not any("lora" in name.lower() for name in module_names)
    assert not any("lora_A" in key or "lora_B" in key for key in state_keys)


def test_physppi_assessment_boundary_keeps_lora_enabled_for_mandatory_only():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_lora_assessment"
    cfg.architecture.lora.enabled = True
    assert cfg.architecture.lora.enabled
    assert cfg.settings.training_scope == "physppi_lora_assessment"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_preset", "anything_else"),
        ("init", "anything_else"),
        ("train_bias", True),
    ],
)
def test_physppi_route_rejects_noncanonical_lora_metadata(field, value):
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_lora_assessment"
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora[field] = value

    with pytest.raises(ValueError, match=field):
        OpenFold3(cfg)


def test_physppi_route_rejects_noncanonical_lora_group_count():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_lora_assessment"
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.groups = [cfg.architecture.lora.groups[0]]

    with pytest.raises(ValueError, match="exactly two LoRA groups"):
        OpenFold3(cfg)


def test_physppi_route_rejects_noncanonical_lora_group_budget():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_lora_assessment"
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.groups[0]["max_trainable_lora_params"] = 200_000

    with pytest.raises(ValueError, match="max_trainable_lora_params"):
        OpenFold3(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exclude_patterns", [r"pairformer_stack\.blocks\.47"]),
        ("unexpected_field", "changes-the-contract"),
    ],
)
def test_physppi_route_rejects_noncanonical_lora_group_fields(field, value):
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_lora_assessment"
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.groups[1][field] = value

    with pytest.raises(ValueError, match=field):
        OpenFold3(cfg)


@pytest.mark.parametrize(
    "forbidden_name",
    ["pEnergy", "Delta-G", "experimentally_resolved"],
)
@pytest.mark.parametrize("placement", ["head_key", "loss_name"])
def test_physppi_route_statically_rejects_forbidden_assessment_keys(
    forbidden_name, placement
):
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_lora_assessment"
    cfg.architecture.pairformer.no_blocks = 1
    if placement == "head_key":
        cfg.architecture.heads[forbidden_name] = {"enabled": False}
    else:
        cfg.architecture.loss_module.confidence_loss_names.append(forbidden_name)

    with pytest.raises(ValueError, match="forbidden"):
        OpenFold3(cfg)


@pytest.mark.parametrize(
    "forbidden_name",
    ["pEnergy", "Delta-G", "experimentally_resolved"],
)
@pytest.mark.parametrize("section_name", ["heads", "loss_module"])
def test_physppi_route_rejects_forbidden_keys_nested_in_sequences(
    forbidden_name, section_name
):
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.settings.training_scope = "physppi_lora_assessment"
    cfg.architecture.pairformer.no_blocks = 1
    section = getattr(cfg.architecture, section_name)
    section["nested_contract_probe"] = [{forbidden_name: False}]

    with pytest.raises(ValueError, match="forbidden"):
        OpenFold3(cfg)
