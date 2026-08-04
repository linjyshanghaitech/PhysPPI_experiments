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

"""Deterministic random streams for immutable PhysPPI sample occurrences."""

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np


@dataclass(frozen=True)
class OccurrenceRNG:
    """Independent deterministic random streams for one sample occurrence."""

    seed: int
    slots: np.random.Generator
    interface_residue: np.random.Generator
    interaction: np.random.Generator
    surface: np.random.Generator
    tm: np.random.Generator
    chain_view: np.random.Generator


def build_occurrence_rng(
    data_seed: int,
    epoch: int,
    global_slot: int,
    dataset: str,
    sample_uid: str,
) -> OccurrenceRNG:
    """Build reproducible, independent streams for a sample occurrence."""
    payload = f"{data_seed}|{epoch}|{global_slot}|{dataset}|{sample_uid}".encode()
    seed = int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")
    children = np.random.SeedSequence(seed).spawn(6)
    generators = [np.random.default_rng(child) for child in children]
    return OccurrenceRNG(seed, *generators)
