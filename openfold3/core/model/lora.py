"""Small LoRA utilities for optional OpenFold3 fine-tuning."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAMatch:
    name: str
    weight_shape: tuple[int, int]
    rank: int
    alpha: float
    trainable_params: int


class LoRALinear(nn.Module):
    """Wrap a Linear-like module with zero-init low-rank residual weights."""

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(base, "weight"):
            raise TypeError("LoRALinear requires a module with a weight tensor")
        if base.weight.ndim != 2:
            raise ValueError("LoRALinear only supports 2D linear weights")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

        out_features, in_features = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        self.reset_parameters()

        for param in self.base.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        dropped = self.dropout(x)
        lora_hidden = torch.nn.functional.linear(
            dropped,
            self.lora_A.to(dtype=dropped.dtype),
        )
        lora_out = torch.nn.functional.linear(
            lora_hidden,
            self.lora_B.to(dtype=lora_hidden.dtype),
        )
        return base_out + lora_out.to(dtype=base_out.dtype) * self.scaling

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        old_weight_key = prefix + "weight"
        new_weight_key = prefix + "base.weight"
        if new_weight_key not in state_dict and old_weight_key in state_dict:
            state_dict[new_weight_key] = state_dict.pop(old_weight_key)

        old_bias_key = prefix + "bias"
        new_bias_key = prefix + "base.bias"
        if new_bias_key not in state_dict and old_bias_key in state_dict:
            state_dict[new_bias_key] = state_dict.pop(old_bias_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _get_child(parent: nn.Module, child_name: str) -> nn.Module:
    children = dict(parent.named_children())
    if child_name not in children:
        raise KeyError(f"Module {parent.__class__.__name__} has no child {child_name}")
    return children[child_name]


def _set_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    setattr(parent, parts[-1], replacement)


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def module_matches(
    name: str,
    include: list[re.Pattern[str]],
    exclude: list[re.Pattern[str]],
) -> bool:
    return any(pattern.search(name) for pattern in include) and not any(
        pattern.search(name) for pattern in exclude
    )


def apply_lora_to_linear_modules(
    model: nn.Module,
    target_patterns: Iterable[str],
    *,
    exclude_patterns: Iterable[str] = (),
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    max_trainable_lora_params: int | None = None,
    allow_no_matches: bool = False,
) -> list[LoRAMatch]:
    """Replace matched Linear-like modules with LoRALinear wrappers."""

    include = _compile_patterns(target_patterns)
    exclude = _compile_patterns(exclude_patterns)
    if not include:
        raise ValueError("LoRA enabled but target_patterns is empty")

    to_wrap: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if module_matches(name, include, exclude):
            to_wrap.append((name, module))

    if not to_wrap:
        if allow_no_matches:
            return []
        raise ValueError("LoRA target_patterns matched zero Linear modules")

    matches: list[LoRAMatch] = []
    total_trainable = 0
    for name, module in to_wrap:
        out_features, in_features = module.weight.shape
        trainable = rank * (in_features + out_features)
        total_trainable += trainable
        matches.append(
            LoRAMatch(
                name=name,
                weight_shape=(out_features, in_features),
                rank=rank,
                alpha=alpha,
                trainable_params=trainable,
            )
        )

    if (
        max_trainable_lora_params is not None
        and total_trainable > max_trainable_lora_params
    ):
        raise ValueError(
            "LoRA trainable parameter count exceeds cap: "
            f"{total_trainable} > {max_trainable_lora_params}"
        )

    for name, module in to_wrap:
        _set_module(
            model,
            name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )

    return matches


def is_lora_parameter_name(name: str) -> bool:
    """Return whether a parameter path ends in a canonical LoRA leaf name."""

    return name.rsplit(".", maxsplit=1)[-1] in {"lora_A", "lora_B"}


def lora_parameter_names(model: nn.Module) -> set[str]:
    return {
        name for name, _ in model.named_parameters() if is_lora_parameter_name(name)
    }


def lora_trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if name in lora_parameter_names(model) and param.requires_grad
    )


def validate_lora_checkpoint_keys(
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    *,
    lora_enabled: bool,
) -> None:
    """Validate non-strict checkpoint key differences for LoRA migrations."""

    bad_missing = [key for key in missing_keys if not is_lora_parameter_name(key)]
    if bad_missing:
        raise ValueError(
            "Checkpoint is missing non-LoRA key(s): " + ", ".join(bad_missing[:10])
        )

    unexpected = list(unexpected_keys)
    if unexpected and not lora_enabled:
        raise ValueError(
            "LoRA-disabled model received unexpected checkpoint key(s): "
            + ", ".join(unexpected[:10])
        )
