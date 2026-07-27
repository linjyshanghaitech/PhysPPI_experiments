# Copyright 2026 AlQuraishi Laboratory
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

"""Initialization functions for network parameters."""

import math

import torch
from torch import nn


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    f = _calculate_fan(weights.shape, fan)
    scale = scale / max(1, f)

    # scipy.stats.truncnorm.rvs can emit NaN/Inf for large tensors with the
    # SciPy version used by the CPU environment.  PyTorch's inverse-CDF
    # implementation is device-aware and remains finite for the same bounds.
    truncated_std = 0.8796256610342398  # std of a standard N(0, 1) clipped to [-2, 2]
    std = math.sqrt(scale) / truncated_std
    with torch.no_grad():
        nn.init.trunc_normal_(
            weights,
            mean=0.0,
            std=std,
            a=-2 * std,
            b=2 * std,
        )


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def kaiming_normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
