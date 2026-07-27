import copy

import pytest
import torch

from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3


def test_physppi_conditioning_module_is_created_when_enabled():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.enabled = False
    model = OpenFold3(cfg)
    assert hasattr(model, "physppi_conditioning")
    assert model.physppi_conditioning is not None
    assert model.interaction_distance_head is not None


def test_physppi_conditioning_module_absent_when_disabled():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = False
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.enabled = False
    model = OpenFold3(cfg)
    assert getattr(model, "physppi_conditioning", None) is None
    assert model.interaction_distance_head is None


def test_physppi_model_validation_allows_mandatory_only_batch():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.enabled = False
    model = OpenFold3(cfg)
    model.train()

    model._validate_physppi_features_for_batch({"token_mask": torch.ones(1, 3)})


def test_physppi_model_validation_rejects_active_optional_slot_without_payload():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = True
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.enabled = False
    model = OpenFold3(cfg)
    model.train()

    with pytest.raises(ValueError, match="active optional contact slot is missing"):
        model._validate_physppi_features_for_batch(
            {
                "token_mask": torch.ones(1, 3),
                "physppi_contact_active": torch.ones(1, dtype=torch.bool),
            }
        )


def test_physppi_model_validation_passes_when_disabled():
    cfg = copy.deepcopy(model_config)
    cfg.architecture.physppi.enabled = False
    cfg.architecture.pairformer.no_blocks = 1
    cfg.architecture.lora.enabled = False
    model = OpenFold3(cfg)
    model.train()

    model._validate_physppi_features_for_batch({"token_mask": torch.ones(1, 3)})
