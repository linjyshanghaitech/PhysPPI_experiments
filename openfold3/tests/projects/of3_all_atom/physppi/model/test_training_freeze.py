import copy
import types
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from openfold3.core.model.lora import is_lora_parameter_name
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom


class _AuxiliaryModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.pairformer_embedding = nn.Linear(2, 2)
        self.plddt = nn.Linear(2, 2)
        self.pae = nn.Linear(2, 2)
        self.pdockq = nn.Linear(2, 2)
        self.pde = nn.Linear(2, 2)
        self.distogram = nn.Linear(2, 2)


class _DummyModel(nn.Module):
    """Small registered module tree for testing the runner freeze allowlist."""

    def __init__(self, config, physppi_enabled=True):
        super().__init__()
        self.config = config
        self.physppi_conditioning = nn.Linear(2, 2) if physppi_enabled else None
        self.membrane_plane_estimator = nn.Linear(2, 2) if physppi_enabled else None
        self.interaction_distance_head = nn.Linear(2, 1) if physppi_enabled else None
        self.diffusion_module = nn.Module()
        self.diffusion_module.extra_conditioner = (
            nn.Linear(2, 2) if physppi_enabled else None
        )
        self.aux_heads = _AuxiliaryModules()
        self.original_backbone = nn.Linear(2, 2)
        self.lora = nn.Module()
        self.lora.lora_A = nn.Parameter(torch.randn(2, 2))
        self.lora.lora_B = nn.Parameter(torch.zeros(2, 2))
        self.lora.lora_C = nn.Parameter(torch.ones(2, 2))
        self.lora.lora_A_shadow = nn.Parameter(torch.ones(2, 2))
        self.lora.lora_B_extra = nn.Parameter(torch.ones(2, 2))


class _MinimalWrapper:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self._freeze_model_params = types.MethodType(
            OpenFold3AllAtom._freeze_model_params, self
        )
        self._freeze_model_params_for_physppi_priors = types.MethodType(
            OpenFold3AllAtom._freeze_model_params_for_physppi_priors, self
        )
        self._get_physppi_only_exempt_submodules = types.MethodType(
            OpenFold3AllAtom._get_physppi_only_exempt_submodules, self
        )
        self._identify_confidence_params = types.MethodType(
            OpenFold3AllAtom._identify_confidence_params, self
        )
        self._audit_physppi_checkpoint = types.MethodType(
            OpenFold3AllAtom._audit_physppi_checkpoint, self
        )


def _physppi_config(enabled=True):
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = enabled
    cfg.settings.training_scope = "physppi_priors_only"
    cfg.architecture.lora.enabled = True
    return cfg


def test_physppi_training_scope_defaults_to_baseline():
    assert model_config.settings.training_scope == "baseline"


def test_physppi_freeze_leaves_only_whitelist_trainable():
    cfg = _physppi_config()
    model = _DummyModel(cfg)
    wrapper = _MinimalWrapper(model)
    wrapper._freeze_model_params_for_physppi_priors()

    trainable = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    allowed = (
        "physppi_conditioning.",
        "membrane_plane_estimator.",
        "interaction_distance_head.",
        "diffusion_module.extra_conditioner.",
        "aux_heads.pairformer_embedding.",
        "aux_heads.plddt.",
        "aux_heads.pae.",
        "aux_heads.pdockq.",
        "aux_heads.pde.",
    )
    assert trainable
    assert any(name.startswith("physppi_conditioning.") for name in trainable)
    assert any(name.startswith("interaction_distance_head.") for name in trainable)
    assert all(
        name.startswith(allowed) for name in trainable
    )
    assert not any(".lora_A" in name or ".lora_B" in name for name in trainable)
    assert not any(name.startswith("aux_heads.distogram.") for name in trainable)
    assert not any(name.startswith("original_backbone.") for name in trainable)
    assert "lora.lora_C" not in trainable
    assert "lora.lora_A_shadow" not in trainable
    assert "lora.lora_B_extra" not in trainable


def test_physppi_freeze_is_independent_of_optional_slot_activation():
    for enabled_optional_slots in (
        (),
        ("interface_residue", "interaction", "surface", "tm"),
    ):
        cfg = _physppi_config()
        cfg.architecture.physppi.enabled_optional_slots = list(enabled_optional_slots)
        model = _DummyModel(cfg)
        wrapper = _MinimalWrapper(model)
        wrapper._freeze_model_params_for_physppi_priors()

        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        assert any(name.startswith("aux_heads.pde.") for name in trainable)
        assert not any(name.endswith("lora_A") for name in trainable)


def test_confidence_identification_includes_pde():
    cfg = _physppi_config()
    model = _DummyModel(cfg)
    wrapper = _MinimalWrapper(model)
    wrapper._identify_confidence_params()

    assert any(
        name.startswith("aux_heads.pde.") for name in wrapper.confidence_param_names
    )


def test_physppi_checkpoint_audit_requires_pde_but_allows_old_lora_keys():
    cfg = _physppi_config()
    model = _DummyModel(cfg)
    wrapper = _MinimalWrapper(model)

    params = model.state_dict()
    old_params = {
        key: value for key, value in params.items() if not is_lora_parameter_name(key)
    }
    wrapper._audit_physppi_checkpoint(old_params)

    without_pde = {
        key: value
        for key, value in old_params.items()
        if not key.startswith("aux_heads.pde.")
    }
    with pytest.raises(ValueError, match="missing active PDE"):
        wrapper._audit_physppi_checkpoint(without_pde)


@pytest.mark.parametrize("fake_lora_leaf", ["lora_A_shadow", "lora_B_extra"])
def test_physppi_checkpoint_audit_rejects_missing_fake_lora_keys(fake_lora_leaf):
    cfg = _physppi_config()
    model = _DummyModel(cfg)
    wrapper = _MinimalWrapper(model)
    params = model.state_dict()
    without_fake_lora = {
        key: value for key, value in params.items() if not key.endswith(fake_lora_leaf)
    }

    with pytest.raises(ValueError, match="missing non-LoRA"):
        wrapper._audit_physppi_checkpoint(without_fake_lora)


def test_physppi_scope_requires_enabled_physppi():
    cfg = _physppi_config(enabled=False)
    model = _DummyModel(cfg, physppi_enabled=False)
    wrapper = _MinimalWrapper(model)

    with pytest.raises(ValueError, match="training_scope requires"):
        wrapper._get_physppi_only_exempt_submodules()


def test_physppi_training_yaml_enables_explicit_scope():
    repo_root = Path(__file__).resolve().parents[6]
    data = yaml.safe_load(
        (repo_root / "examples/training_yamls/physppi_training.yml").read_text()
    )
    settings = data["model_update"]["custom"]["settings"]
    assert settings["training_scope"] == "physppi_priors_only"
    assert settings.get("train_confidence_only", False) is False


def test_physppi_route_c_metric_value_filters_nonfinite():
    runner = OpenFold3AllAtom.__new__(OpenFold3AllAtom)
    assert runner._route_c_metric_value(torch.tensor(3.0)) == 3.0
    assert runner._route_c_metric_value(torch.tensor(float("nan"))) is None
    assert runner._route_c_metric_value(torch.empty(0)) is None
