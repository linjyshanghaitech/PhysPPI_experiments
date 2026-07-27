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

import logging
import os
from itertools import cycle, islice

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info

from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_center_atom_indices,
)

logger = logging.getLogger(__name__)
worker_seed_log = logging.getLogger(f"{__name__}.worker_seed")


def validate_monotonic_token_indices(features: dict) -> None:
    """Require strictly increasing token indices on both cropped token axes."""
    for path, token_features in (
        ("token_index", features),
        ("ground_truth.token_index", features["ground_truth"]),
    ):
        token_mask = token_features["token_mask"].bool()
        token_index = token_features["token_index"][token_mask]
        if token_index.numel() > 1 and not token_index.diff().gt(0).all():
            raise ValueError(f"{path} must be strictly increasing")


def _validate_start_atom_ranges(
    token_features: dict,
    *,
    atom_count: int,
    path: str,
) -> None:
    token_mask = token_features["token_mask"].bool()
    starts = token_features["start_atom_index"].long()[token_mask]
    counts = token_features["num_atoms_per_token"].long()[token_mask]
    if (starts < 0).any() or (counts < 0).any() or (starts + counts > atom_count).any():
        raise ValueError(f"{path} atom ranges exceed atom-axis bounds")
    if starts.numel() > 1 and not starts.diff().gt(0).all():
        raise ValueError(f"{path} start_atom_index must be strictly increasing")


def validate_atom_to_token_bounds(features: dict) -> None:
    """Validate atom-to-token and token-to-atom ranges deterministically."""
    token_count = features["token_mask"].shape[-1]
    atom_mask = features["atom_mask"].bool()
    atom_to_token = features["atom_to_token_index"].long()[atom_mask]
    if (atom_to_token < 0).any() or (atom_to_token >= token_count).any():
        raise ValueError("atom_to_token_index exceeds token-axis bounds")
    _validate_start_atom_ranges(
        features,
        atom_count=features["ref_pos"].shape[-2],
        path="input",
    )
    ground_truth = features["ground_truth"]
    _validate_start_atom_ranges(
        ground_truth,
        atom_count=ground_truth["atom_positions"].shape[-2],
        path="ground_truth",
    )


def validate_representative_atom_indices(features: dict) -> None:
    """Require each valid token's representative index to address a valid atom."""
    token_mask = features["token_mask"].bool()
    if "physppi_token_center_atom_index" in features:
        representative_index = features["physppi_token_center_atom_index"].long()
        representative_supported = token_mask
    else:
        representative_index, representative_supported = get_token_center_atom_indices(
            features
        )
        representative_supported = representative_supported.bool() & token_mask
    atom_count = features["ref_pos"].shape[-2]
    valid = (
        (representative_index >= 0)
        & (representative_index < atom_count)
        & representative_supported
    )
    if not valid[token_mask].all():
        raise ValueError("representative atom index exceeds atom-axis bounds")
    if not features["atom_mask"].bool()[representative_index[token_mask]].all():
        raise ValueError("representative atom index addresses a padded atom")


def validate_ground_truth_crop_coverage(features: dict) -> None:
    """Require GT token selection to cover exactly the cropped reference atoms."""
    ground_truth = features["ground_truth"]
    gt_token_ids_atomized = broadcast_token_feat_to_atoms(
        token_mask=ground_truth["token_mask"].bool(),
        num_atoms_per_token=ground_truth["num_atoms_per_token"],
        token_feat=ground_truth["token_index"],
    )
    crop_token_ids = features["token_index"][features["token_mask"].bool()]
    selected_gt_atoms = torch.isin(gt_token_ids_atomized, crop_token_ids)
    expected_atom_count = int(features["atom_mask"].bool().sum())
    if int(selected_gt_atoms.sum()) != expected_atom_count:
        raise ValueError(
            "ground-truth crop coverage does not match the cropped atom count"
        )


def pad_to_world_size(df: pd.DataFrame, world_size: int | None = None) -> pd.DataFrame:
    """Pads a dataframe containing examples to match the world size.

    To avoid the default DistributedSampler behavior of repeating samples
    to match the world size, artificially inflate the dataset and flag the
    repeated samples so that they are ignored in the metrics.

    Args:
        df: starting collection of examples
        world_size: world_size in a distributed setting
    Returns:
        collection of examples padded to world size, with the first examples repeated.
    """
    num_examples = len(df)

    if not world_size or num_examples % world_size == 0:
        padded_df = df.copy()
        padded_df["repeated_sample"] = [False] * num_examples
        return padded_df

    # otherwise we need to pad the dataframe
    num_repeated_examples = world_size - num_examples % world_size
    repeated_indices = list(islice(cycle(range(num_examples)), num_repeated_examples))
    padded_df = pd.concat([df, df.iloc[repeated_indices]], ignore_index=True)
    padded_df["repeated_sample"] = [False] * num_examples + [
        True
    ] * num_repeated_examples

    return padded_df


def check_invalid_feature_dict(features: dict):
    """
    Validate the feature dictionary for a single datapoint. Raises a ValueError
    if a check fails. Error handling in each training dataset class should catch
    these in debug_mode and skip/log the invalid samples.

    Args:
        features (dict):
            Feature dictionary for a single datapoint
    """
    validate_monotonic_token_indices(features)
    validate_atom_to_token_bounds(features)
    validate_representative_atom_indices(features)
    validate_ground_truth_crop_coverage(features)

    # Check that the sum of the number of atoms per token is equal to the
    # number of atoms in the reference conformer
    num_atoms_sum = torch.max(torch.sum(features["num_atoms_per_token"], dim=-1)).int()
    num_ref_atoms = features["ref_pos"].shape[-2]
    if num_atoms_sum != num_ref_atoms:
        raise ValueError(
            f"Size mismatch between sum of 'num_atoms_per_token' {num_atoms_sum} atoms "
            f"and reference conformer {num_ref_atoms} atoms."
        )

    # Check that the crop has some resolved atoms
    if not features["ground_truth"]["atom_resolved_mask"].any():
        raise ValueError("No resolved atoms")

    # Check that the ligand and atomized masks are consistent
    if (features["is_ligand"] & ~features["is_atomized"]).any():
        raise ValueError("Contains non-atomized ligands")

    # Check that the number of tokens per atom is less than the maximum expected
    if (features["num_atoms_per_token"] > 23).any():
        raise ValueError("Token contains number of atoms > max expected (23)")

    # Check that all input features are finite
    for k, v in features.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            raise ValueError(f"Non-finite values for {k}")
        if isinstance(v, dict):
            for i, j in v.items():
                if isinstance(j, torch.Tensor) and not torch.isfinite(j).all():
                    raise ValueError(f"Non-finite values for {i}")


def getitem_debug_log(dataset_name: str = "") -> None:
    wi = get_worker_info()
    worker_id = wi.id if wi is not None else 0
    wi_seed = wi.seed if wi else None
    wi_base_seed = (wi_seed - worker_id) if wi_seed is not None else None
    torch_seed = torch.initial_seed()
    if dist.is_available() and dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = int(os.environ.get("RANK", 0))
    local_rank = os.environ.get("LOCAL_RANK", 0)
    worker_seed_log.debug(
        f"__getitem__ {dataset_name}: rank={global_rank} local_rank={local_rank} "
        f"pid={os.getpid()} worker_id={worker_id} wi.seed={wi_seed} "
        f"wi.base_seed={wi_base_seed} torch.initial_seed={torch_seed}",
    )
