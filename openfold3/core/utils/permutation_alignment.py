# =============================================================================
# Modified by Jinyu Lin (linjyshanghaitech), ShanghaiTech University
# Date: 2026-07-28
# Project: PhysPPI — https://github.com/linjyshanghaitech/PhysPPI_experiments
#
# Modification summary:
#   1. Added `chain_assignment="token_identity"` pathway — directly maps
#      prediction tokens to GT tokens by stable source token_index, with
#      optional asym_id consistency verification.  This is now the PREFERRED
#      alignment strategy for PhysPPI training.
#   2. Changed fallback chain in safe_multi_chain_permutation_alignment()
#      from:  protected → identity
#      to:    token_identity → protected → identity
#   3. Added helper functions:
#        - get_token_identity_gt_token_subset_index()
#        - get_token_identity_atom_permutation_index()
#        - _token_identity_atom_index_before_symmetry()
#        - _alignment_token_center_atom_indices()
#        - _get_valid_token_center_atoms()
#
#   See: experiments/permutation_alignment/README.md for full context.
# =============================================================================
#
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

# NIT: confusing that this needs to be transposed, while the rotation matrix in
# Transformation doesn't
import logging
from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial
from typing import NamedTuple, overload

import numpy as np
import torch

from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_3
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_atom_index_offset,
    get_token_center_atom_indices,
)
from openfold3.core.utils.geometry.kabsch_alignment import (
    Transformation,
    apply_transformation,
    get_optimal_transformation,
)
from openfold3.core.utils.protected_chain_matching import (
    ChainRecord,
    ProtectedChainMatchError,
    align_chain_residues,
    lexicographic_linear_sum_assignment,
    protected_chain_match,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


def check_out_of_bounds_indices(
    indices: torch.Tensor, input_tensor: torch.Tensor, dim: int = 0
):
    """Checks if the indices are out of bounds.

    Args:
        indices: Indices to check
        input_tensor: The tensor for indexing
        dim: Dimension to check
    """
    clamped_indices = torch.clamp(
        indices,
        min=0,
        max=input_tensor.shape[dim] - 1,
    )
    assert (indices == clamped_indices).all()


@overload
def split_feats_by_id(
    feats: torch.Tensor, id: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]: ...


def split_feats_by_id(
    feats: list[torch.Tensor],
    id: torch.Tensor,
) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
    """Splits features up into groups corresponding to an ID.

    Args:
        feats (list[torch.Tensor] | torch.Tensor):
            Either list of features or to split up, or single feature tensor.
        id (torch.Tensor):
            A tensor of IDs to split the features by.

    Returns:
        Either:
            tuple[list[list[torch.Tensor]], torch.Tensor]:
                A list of lists of features, where the outer list corresponds to the
                different input features and the inner list corresponds to the different
                IDs. The second element is the unique IDs, in the order of which the
                features were split.
        or:
            tuple[list[torch.Tensor], torch.Tensor]:
                A list corresponding to the single input feature split by the IDs, and
                the unique IDs in the corresponding order.
    """
    unique_ids = torch.unique(id)

    if isinstance(feats, torch.Tensor):
        split_feats = [feats[id == single_id] for single_id in unique_ids]
    else:
        split_feats = [
            [feat[id == single_id] for single_id in unique_ids] for feat in feats
        ]

    return split_feats, unique_ids


def get_gt_segment_mask(
    segment_mol_sym_token_index: torch.Tensor,
    gt_mol_sym_token_index: torch.Tensor,
    segment_mol_sym_id: int | None = None,
    segment_mol_entity_id: int | None = None,
    gt_mol_sym_id: torch.Tensor | None = None,
    gt_mol_entity_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fetches a mask for the gt-segment matching the predicted segment

    This creates a mask on the ground-truth features that exactly matches the input
    predicted segment. The corresponding segment is selected by pairing the predicted
    sym_token_index with the ground-truth sym_token_index. Optionally, the segment can
    be further filtered by the sym_id and entity_id.

    This is necessary because the ground-truth values are usually more expansive and
    contain more atoms due to symmetry-equivalent atoms. Note that this function
    therefore selects atoms as corresponding to the original (arbitrary) input order,
    and makes no attempt to select a particular symmetry-optimal permutation.

    Args:
        segment_mol_sym_token_index (torch.Tensor):
            The sym_token_index values of the token center atoms of the predicted
            segment the ground-truth is matched to.
        gt_mol_sym_token_index (torch.Tensor):
            The sym_token_index values of the ground-truth's token center atoms.
        segment_mol_sym_id (int):
            The sym_id of the predicted segment. If this is specified, gt_mol_sym_id
            must be specified as well, and the mask will only include ground-truth atoms
            with the same sym_id. Defaults to None.
        segment_mol_entity_id (int):
            The entity_id of the predicted segment. If this is specified,
            gt_mol_entity_id must be specified as well, and the mask will only include
            ground-truth atoms with the same entity_id. Defaults to None.
        gt_mol_sym_id (torch.Tensor):
            The sym_id values of the ground-truth's token center atoms. If
            segment_mol_sym_id is specified, this must be specified as well. Defaults
            to None.
        gt_mol_entity_id (torch.Tensor):
            The entity_id values of the ground-truth's token center atoms. If
            segment_mol_entity_id is specified, this must be specified as well.
            Defaults to None.

    Returns:
        torch.Tensor:
            A boolean mask of the same shape as the ground-truth sym_token_index that
            selects the atoms that match the predicted segment.
    """
    segment_mask = torch.ones_like(gt_mol_sym_token_index, dtype=torch.bool)

    # Optionally subset to particular sym_id
    if segment_mol_sym_id is not None:
        if gt_mol_sym_id is None:
            raise ValueError("Need to pass gt_mol_sym_id if segment_mol_sym_id is set")

        segment_mask &= gt_mol_sym_id == segment_mol_sym_id

    # Optionally subset to particular entity_id
    if segment_mol_entity_id is not None:
        if gt_mol_entity_id is None:
            raise ValueError(
                "Need to pass gt_mol_entity_id if segment_mol_entity_id is set"
            )

        segment_mask &= gt_mol_entity_id == segment_mol_entity_id

    # Subset to the sym_token_index of the predicted segment
    segment_mask &= torch.isin(gt_mol_sym_token_index, segment_mol_sym_token_index)

    return segment_mask


def get_centroid(coords: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4):
    """Computes the centroids of resolved coordinates.

    Args:
        coords (torch.Tensor):
            [*, N, 3] the coordinates to compute the centroid of.
        mask (torch.Tensor):
            [*, N] mask for resolved coordinates.
        eps (float):
            A small value to prevent division by zero.

    Returns:
        torch.Tensor:
            [*, 3] the centroid of the resolved coordinates
    """
    n_observed_atoms = torch.sum(mask, dim=-1, keepdim=True)
    centroid = torch.sum(
        coords * mask[..., None],
        dim=-2,
    ) / (n_observed_atoms + eps)

    return centroid


def get_sym_id_with_most_resolved_atoms(
    resolved_mask: torch.Tensor,
    mol_sym_id: torch.Tensor,
) -> tuple[int, int]:
    """Returns the symmetry ID with the most resolved atoms.

    Takes in an array of different symmetry IDs with a corresponding mask of resolved
    atoms, and returns the symmetry ID with the most corresponding resolved atoms.

    Args:
        resolved_mask (torch.Tensor):
            [N] mask of resolved atoms.
        mol_sym_id (torch.Tensor):
            [N] symmetry IDs corresponding to the resolved atoms.

    Returns:
        tuple[int, int]:
            The symmetry ID with the most resolved atoms, and the number of resolved
            atoms. If there are no resolved atoms in the input, the first symmetry ID is
            returned.
    """
    mol_sym_id = mol_sym_id[resolved_mask]

    unique_sym_ids, n_resolved_atoms = torch.unique(mol_sym_id, return_counts=True)

    # If no resolved atoms, return first sym_id
    if n_resolved_atoms.numel() == 0:
        best_idx = 0
    else:
        best_idx = torch.argmax(n_resolved_atoms)

    best_sym_id = unique_sym_ids[best_idx].item()
    n_resolved = n_resolved_atoms[best_idx].item()

    return best_sym_id, n_resolved


class AnchorCandidate(NamedTuple):
    """Small wrapper class around potential anchor molecule data.

    Attributes:
        entity_id (int):
            The entity ID of this anchor candidate molecule.
        sym_id (int):
            The symmetry ID of this anchor candidate molecule.
        n_resolved (int):
            The number of resolved token center atoms in this anchor candidate molecule.
    """

    entity_id: int
    sym_id: int
    n_resolved: int


def get_least_ambiguous_anchor_candidate(
    entity_sym_id_combinations: torch.Tensor,
    gt_mol_entity_id: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_mol_sym_id: torch.Tensor,
) -> AnchorCandidate:
    """Finds the anchor candidate with the least ambiguity.

    This function takes in a list of entity-symmetry ID combinations and returns the
    entity-symmetry ID pair with the least ambiguity. This is done primarily by
    selecting the entity with the least symmetry mates, corresponding to AF2-Multimer
    7.3.1. To break ties between entities with equivalent stoichiometry, as well as to
    select the particular symmetric molecule instance to use as an anchor, the symmetric
    molecule with the most resolved atoms is selected, in the hope that this will result
    in the most stable alignment.

    Args:
        entity_sym_id_combinations (torch.Tensor):
            [N, 2] a list of entity-symmetry ID combinations.
        gt_mol_entity_id (torch.Tensor):
            [N] the entity IDs of each ground-truth atom.
        gt_token_center_resolved_mask (torch.Tensor):
            [N] mask of resolved atoms.
        gt_mol_sym_id (torch.Tensor):
            [N] the symmetry IDs of each ground-truth atom.

    Returns:
        AnchorCandidate:
            The anchor candidate with the least ambiguity.
    """
    # Get number of symmetry mates for each entity
    entity_stoichiometry = Counter(entity_sym_id_combinations[:, 0].tolist())

    # Revert to get entitity IDs for each number of symmetry mates
    count_to_entity_ids = defaultdict(list)
    for entity_id, count in entity_stoichiometry.items():
        count_to_entity_ids[count].append(entity_id)

    # Traverse in order of least symmetry mates / ambiguity, breaking ties if necessary,
    # to get the best anchor candidate
    best_anchor_candidate = None
    for count in sorted(count_to_entity_ids.keys()):
        entity_ids = count_to_entity_ids[count]

        # In case of tie, take the entity ID and sym ID with the biggest number of
        # resolved ground-truth token center atoms (hopefully resulting in the most
        # stable alignments on average)
        if len(entity_ids) > 1:
            logger.debug(
                f"Multiple entities with {count} symmetry mates found. Breaking tie."
            )

            anchor_candidates = []
            for entity_id in entity_ids:
                entity_mask = gt_mol_entity_id == entity_id

                best_sym_id, n_resolved = get_sym_id_with_most_resolved_atoms(
                    gt_token_center_resolved_mask[entity_mask],
                    gt_mol_sym_id[entity_mask],
                )

                anchor_candidates.append(
                    AnchorCandidate(entity_id, best_sym_id, n_resolved)
                )

            # Select best anchor candidate based on number of resolved atoms
            anchor_candidate = max(anchor_candidates, key=lambda x: x.n_resolved)
        else:
            (anchor_entity_id,) = entity_ids

            best_sym_id, n_resolved = get_sym_id_with_most_resolved_atoms(
                gt_token_center_resolved_mask[gt_mol_entity_id == anchor_entity_id],
                gt_mol_sym_id[gt_mol_entity_id == anchor_entity_id],
            )

            anchor_candidate = AnchorCandidate(
                anchor_entity_id, best_sym_id, n_resolved
            )

        # Break if we have a candidate with more than 3 resolved atoms (necessary for
        # stable alignment)
        if anchor_candidate.n_resolved > 3:
            best_anchor_candidate = anchor_candidate
            best_anchor_n_symmetry_mates = count
            break
        # Otherwise, overwrite best_anchor_candidate if we don't have one yet so that it
        # at least corresponds to the least ambiguous stoichiometry. This will only be
        # returned as the final result if no other candidate with more than 3 resolved
        # atoms is found.
        elif best_anchor_candidate is None:
            best_anchor_candidate = anchor_candidate
            best_anchor_n_symmetry_mates = count

    logger.debug(
        f"Chose anchor candidate with {best_anchor_n_symmetry_mates} symmetry mates."
    )

    return best_anchor_candidate


def get_gt_anchor_mask(
    gt_token_center_resolved_mask: torch.Tensor,
    gt_mol_entity_id: torch.Tensor,
    gt_mol_sym_id: torch.Tensor,
    gt_is_ligand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Finds a suitable ground-truth anchor molecule.

    This function finds a suitable ground-truth anchor molecule for the anchor-chain
    alignment, by selecting the molecule with the least ambiguous alignment. Polymer
    anchor molecules are prioritized over ligand anchor molecules, as ligands are more
    likely to result in noisy anchor alignments due to the presence of symmetric atoms
    whose selected permutation is still arbitrary at this stage. If no suitable polymer
    anchor molecule with more than 3 resolved atoms is found, the function will fall
    back to applying the same above criteria to ligand entities in order to find a more
    suitable ligand anchor molecule.

    The selection logic is further detailed in `get_least_ambiguous_anchor_candidate`.

    Args:
        gt_token_center_resolved_mask (torch.Tensor):
            [N] mask of resolved token center atoms.
        gt_mol_entity_id (torch.Tensor):
            [N] the entity IDs of each ground-truth token center atom.
        gt_mol_sym_id (torch.Tensor):
            [N] the symmetry IDs of each ground-truth token center atom.
        gt_is_ligand (torch.Tensor):
            [N] whether each ground-truth token center atom is part of a ligand.

    Returns:
        An [N] mask corresponding to the anchor molecule's token center atoms within the
        full ground-truth token center atoms.
    """

    gt_token_center_resolved_mask = gt_token_center_resolved_mask.bool()
    gt_is_ligand = gt_is_ligand.bool()

    # Group symmetry-equivalent molecules of the same entity together
    entity_sym_id_combinations = torch.cat(
        [
            gt_mol_entity_id[gt_token_center_resolved_mask].unsqueeze(-1),
            gt_mol_sym_id[gt_token_center_resolved_mask].unsqueeze(-1),
        ],
        dim=-1,
    )
    unique_entity_sym_id_combinations = torch.unique(entity_sym_id_combinations, dim=0)

    # Group into polymeric entities and ligand-only entities. Ligand-only entities are
    # deprioritized as anchor chains because of potentially less stable alignments due
    # to symmetric atoms.
    polymer_entity_sym_id_combinations = []
    ligand_entity_sym_id_combinations = []

    for entity_id, sym_id in unique_entity_sym_id_combinations:
        entity_mask = gt_mol_entity_id == entity_id

        if torch.all(gt_is_ligand[entity_mask]):
            ligand_entity_sym_id_combinations.append((entity_id, sym_id))
        else:
            polymer_entity_sym_id_combinations.append((entity_id, sym_id))

    polymer_entity_sym_id_combinations = torch.tensor(
        polymer_entity_sym_id_combinations, device=gt_mol_entity_id.device
    )
    ligand_entity_sym_id_combinations = torch.tensor(
        ligand_entity_sym_id_combinations, device=gt_mol_entity_id.device
    )

    have_polymer_entities = polymer_entity_sym_id_combinations.numel() > 0

    # Try to get a polymeric anchor
    if have_polymer_entities:
        polymer_anchor_candidate = get_least_ambiguous_anchor_candidate(
            polymer_entity_sym_id_combinations,
            gt_mol_entity_id,
            gt_token_center_resolved_mask,
            gt_mol_sym_id,
        )

    # If no polymeric anchor available or only polymeric anchor has < 3 resolved atoms,
    # try to see if there is a better ligand anchor
    if not have_polymer_entities or polymer_anchor_candidate.n_resolved < 3:
        logger.debug(
            "Polymeric anchor candidate has less than 3 resolved atoms, "
            "trying with ligand candidate."
        )

        have_ligand_entities = ligand_entity_sym_id_combinations.numel() > 0
        if have_ligand_entities:
            ligand_anchor_candidate = get_least_ambiguous_anchor_candidate(
                ligand_entity_sym_id_combinations,
                gt_mol_entity_id,
                gt_token_center_resolved_mask,
                gt_mol_sym_id,
            )

            if ligand_anchor_candidate.n_resolved > 3:
                final_anchor = ligand_anchor_candidate
                logger.warning(
                    "Chose ligand anchor molecule. This could result in an unstable "
                    "alignment if the ligand has symmetric atoms."
                )
            else:
                # Both polymer and ligand anchors have < 3 resolved atoms
                logger.warning(
                    "Chose anchor molecule with less than 3 resolved atoms. "
                    "This will result in an unstable alignment."
                )
                # If no polymer anchors were available at all, must settle for ligand
                # Otherwise, revert to the polymer anchor
                final_anchor = (
                    ligand_anchor_candidate
                    if not have_polymer_entities
                    else polymer_anchor_candidate
                )
        else:
            # No ligand candidates available, must settle for the polymer candidate
            logger.warning(
                "Chose anchor molecule with less than 3 resolved atoms. "
                "This will result in an unstable alignment."
            )
            final_anchor = polymer_anchor_candidate
    else:
        # Chose polymer anchor
        final_anchor = polymer_anchor_candidate

    # Get the final mask for the anchor molecule
    anchor_mask = (gt_mol_entity_id == final_anchor.entity_id) & (
        gt_mol_sym_id == final_anchor.sym_id
    )

    return anchor_mask


def get_anchor_transformations(
    gt_anchor_coords: torch.Tensor,
    gt_anchor_resolved_mask: torch.Tensor,
    gt_anchor_sym_token_index: torch.Tensor,
    gt_anchor_entity_id: int,
    pred_coords: torch.Tensor,
    pred_mol_entity_id: torch.Tensor,
    pred_mol_sym_id: torch.Tensor,
    pred_mol_sym_token_index: torch.Tensor,
):
    """Computes optimal transformations for each predicted molecule onto the anchor.

    This function takes in a ground-truth anchor molecule, detects all molecules in the
    prediction that are symmetric to the anchor, and computes the optimal transformation
    for each of these molecules onto the anchor molecule. The optimal transformation is
    computed using the Kabsch algorithm. This follows AlphaFold2-Multimer 7.3.1, though
    note that the coordinates aligned are the token center atoms, not only C-alpha
    carbons.

    Args:
        gt_anchor_coords (torch.Tensor):
            [N, 3] the coordinates of the anchor molecule's token center atoms.
        gt_anchor_resolved_mask (torch.Tensor):
            [N] mask of resolved token center atoms in the anchor molecule.
        gt_anchor_sym_token_index (torch.Tensor):
            [N] the sym_token_index values of the anchor molecule's token center atoms.
        gt_anchor_entity_id (int):
            The entity ID of the anchor molecule.
        pred_coords (torch.Tensor):
            [N, 3] the coordinates of the predicted molecules' token center atoms.
        pred_mol_entity_id (torch.Tensor):
            [N] the entity IDs of the predicted molecules' token center atoms.
        pred_mol_sym_id (torch.Tensor):
            [N] the symmetry IDs of the predicted molecules' token center atoms.
        pred_mol_sym_token_index (torch.Tensor):
            [N] the sym_token_index values of the predicted molecules' token center
            atoms.

    Returns:
        Transformation:
            A Transformation object detailing the affine transformations of each of the
            M predicted molecules that are symmetric to the anchor onto the anchor.

            Attributes:
                rotation_matrix (torch.Tensor):
                    [M, 3, 3] the rotation matrix of the transformation.
                translation_vector (torch.Tensor):
                    [M, 3] the translation vector of the transformation.
    """

    # Get all the molecules in the pred that are equivalent to the anchor
    pred_entity_mask = pred_mol_entity_id == gt_anchor_entity_id
    pred_coords_entity = pred_coords[pred_entity_mask]
    pred_mol_sym_id_entity = pred_mol_sym_id[pred_entity_mask]
    pred_mol_sym_token_index_entity = pred_mol_sym_token_index[pred_entity_mask]

    pred_sym_ids = torch.unique(pred_mol_sym_id_entity)

    transformations = []

    # Transform each equivalent predicted molecule onto the anchor molecule
    for sym_id in pred_sym_ids:
        pred_sym_mask = pred_mol_sym_id_entity == sym_id

        # Get the part of the full anchor chain that matches the in-crop segment
        gt_segment_mask = get_gt_segment_mask(
            segment_mol_sym_token_index=pred_mol_sym_token_index_entity[pred_sym_mask],
            gt_mol_sym_token_index=gt_anchor_sym_token_index,
        )
        gt_segment_coords = gt_anchor_coords[gt_segment_mask]
        gt_segment_resolved_mask = gt_anchor_resolved_mask[gt_segment_mask]

        pred_coords_sym = pred_coords_entity[pred_sym_mask]

        # Get the optimal transformation
        transformations.append(
            get_optimal_transformation(
                gt_segment_coords,
                pred_coords_sym,
                gt_segment_resolved_mask,
            )
        )

    # Stack the transformations to enable broadcasting
    transformations = Transformation(
        rotation_matrix=torch.stack([t.rotation_matrix for t in transformations]),
        translation_vector=torch.stack([t.translation_vector for t in transformations]),
    )

    return transformations


def find_greedy_optimal_mol_permutation(
    gt_token_center_positions_transformed: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_mol_entity_ids: torch.Tensor,
    gt_mol_sym_ids: torch.Tensor,
    gt_mol_sym_token_index: torch.Tensor,
    pred_token_center_positions: torch.Tensor,
    pred_mol_entity_ids: torch.Tensor,
    pred_mol_sym_ids: torch.Tensor,
    pred_mol_sym_token_index: torch.Tensor,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Finds the optimal (entity_id, sym_id) -> (entity_id, sym_id) mapping.

    This function identifies the optimal mapping between symmetric molecules in the
    prediction and the ground-truth, mostly following the Assignment stage of
    AlphaFold2-Multimer (7.3.2). Molecules are greedily assigned to their best matching
    partner by using greedy token center centroid matching under each anchor alignment.

    NOTE: A slight difference to the AF2-Multimer algorithm is that we do not map the
    predicted molecules to ground-truth molecules in an arbitrary order, but instead
    sort the predicted molecules by the number of resolved atoms in the corresponding
    ground-truth segments first. This can give a slightly more stable mapping,
    especially in edge-cases where a predicted molecules has no unresolved atoms in the
    ground-truth and could arbitrarily "steal" a ground-truth segment that would map
    better to another symmetry-equivalent predicted molecule.

    Args:
        gt_token_center_positions_transformed (torch.Tensor):
            [N_aln, N, 3] the transformed token center positions of the ground-truth
                          molecules under each anchor alignment.
        gt_token_center_resolved_mask (torch.Tensor):
            [N] mask of resolved token center atoms in the ground-truth molecules.
        gt_mol_entity_ids (torch.Tensor):
            [N] the entity IDs of the ground-truth token center atoms.
        gt_mol_sym_ids (torch.Tensor):
            [N] the symmetry IDs of the ground-truth token center atoms.
        gt_mol_sym_token_index (torch.Tensor):
            [N] the sym_token_index values of the ground-truth token center atoms.
        pred_token_center_positions (torch.Tensor):
            [N, 3] the token center positions of the predicted molecules.
        pred_mol_entity_ids (torch.Tensor):
            [N] the entity IDs of the predicted molecules.
        pred_mol_sym_ids (torch.Tensor):
            [N] the symmetry IDs of the predicted molecules.
        pred_mol_sym_token_index (torch.Tensor):
            [N] the sym_token_index values of the predicted molecules.

    Returns:
        dict[tuple[int, int], tuple[int, int]]:
            A dictionary defining a mapping between each predicted molecule and
            corresponding ground-truth molecule, both uniquely identified by their
            (entity_id, sym_id) pair.
    """
    gt_token_center_resolved_mask = gt_token_center_resolved_mask.bool()

    # Get the unique entity IDs
    unique_pred_mol_entity_ids = torch.unique(pred_mol_entity_ids)

    # Keep track of centroid distances and permutations over all anchor alignments
    centroid_dists_sq = []
    permutations = []

    # Iterate through each anchor alignment
    for gt_token_center_coords in gt_token_center_positions_transformed:
        centroid_dists_sq_aln = []
        permutation_aln = {}

        # Resolve permutation for each entity group separately
        for entity_id in unique_pred_mol_entity_ids:
            entity_id = entity_id.item()

            pred_entity_mask = pred_mol_entity_ids == entity_id
            pred_coords_entity = pred_token_center_positions[pred_entity_mask]
            pred_sym_ids_entity = pred_mol_sym_ids[pred_entity_mask]
            pred_sym_token_index_entity = pred_mol_sym_token_index[pred_entity_mask]

            gt_entity_mask = gt_mol_entity_ids == entity_id
            gt_coords_entity = gt_token_center_coords[gt_entity_mask]
            gt_resolved_mask_entity = gt_token_center_resolved_mask[gt_entity_mask]
            gt_sym_ids_entity = gt_mol_sym_ids[gt_entity_mask]
            gt_sym_token_index_entity = gt_mol_sym_token_index[gt_entity_mask]

            # Split ground-truth features into symmetric instances
            (
                (
                    gt_coords_entity_split,
                    gt_sym_token_index_entity_split,
                    gt_resolved_mask_entity_split,
                ),
                gt_unique_sym_ids_entity,
            ) = split_feats_by_id(
                [gt_coords_entity, gt_sym_token_index_entity, gt_resolved_mask_entity],
                gt_sym_ids_entity,
            )

            # Stack to [N_sym, N, 3] and [N_sym, N] respectively
            instance_sizes = [t.shape[0] for t in gt_coords_entity_split]
            if len(set(instance_sizes)) > 1:
                raise ValueError(
                    f"Entity {entity_id} has symmetric instances with "
                    f"different token counts: {instance_sizes}. These chains "
                    f"were identified as symmetry-equivalent but have different "
                    f"numbers of tokens. A common cause is the biotite sym_id "
                    f"annotation having mixed values (0 for resolved, -1 for "
                    f"unresolved atoms) within a residue, which causes "
                    f"get_residue_starts() to split it into multiple tokens."
                )

            gt_coords_entity_split = torch.stack(gt_coords_entity_split)
            gt_resolved_mask_entity_split = torch.stack(gt_resolved_mask_entity_split)

            # TODO: Dev-only, remove later
            assert all(
                torch.equal(sym_token_tensor, gt_sym_token_index_entity_split[0])
                for sym_token_tensor in gt_sym_token_index_entity_split[1:]
            )
            # All these should be equivalent, so we just take the first one
            gt_sym_token_index_segment = gt_sym_token_index_entity_split[0]

            unique_pred_sym_ids_entity = torch.unique(pred_sym_ids_entity)

            # Precompute the segment masks which indicate what segment of each
            # ground-truth symmetry-equivalent molecule corresponds to a particular
            # predicted molecule
            pred_sym_id_to_segment_mask = {
                sym_id: get_gt_segment_mask(
                    segment_mol_sym_token_index=pred_sym_token_index_entity[
                        pred_sym_ids_entity == sym_id
                    ],
                    gt_mol_sym_token_index=gt_sym_token_index_segment,
                )
                for sym_id in unique_pred_sym_ids_entity.tolist()
            }

            # Get the number of resolved ground-truth atoms in the ground-truth segments
            # corresponding to this predicted molecule
            pred_sym_id_to_n_resolved_gt = {
                sym_id: gt_resolved_mask_entity_split[:, mask].sum()
                for sym_id, mask in pred_sym_id_to_segment_mask.items()
            }

            # Sort the predicted symmetry IDs by the number of resolved atoms in the
            # corresponding ground-truth segments, so that predicted molecules with the
            # least ambiguous mappings are considered first (slight deviation from
            # AF2-Multimer)
            unique_pred_sym_ids_entity_sorted = sorted(
                unique_pred_sym_ids_entity.tolist(),
                key=lambda sym_id: pred_sym_id_to_n_resolved_gt[sym_id],
                reverse=True,
            )

            # Track which ground-truth symmetry IDs have already been assigned
            used_gt_sym_ids = []

            # Run greedy assignment for this entity
            for sym_id in unique_pred_sym_ids_entity_sorted:
                # Match the segment of the predicted molecule
                gt_segment_mask = pred_sym_id_to_segment_mask[sym_id]

                # [N_sym, N, 3] -> [N_sym, N_segment, 3]
                gt_coords_entity_segment = gt_coords_entity_split[:, gt_segment_mask, :]
                # [N_sym, N] -> [N_sym, N_segment]
                gt_resolved_mask_entity_segment = gt_resolved_mask_entity_split[
                    :, gt_segment_mask
                ]

                # Get centroids of ground-truth
                # [N_sym, 3]
                gt_coords_entity_centroids = get_centroid(
                    gt_coords_entity_segment, gt_resolved_mask_entity_segment
                )

                # Get centroid of prediction molecule, while matching unresolved atoms
                # on the GT side indvidually for each symmetric ground-truth molecule
                # [N_segment, 3]
                pred_coords_sym = pred_coords_entity[pred_sym_ids_entity == sym_id]
                # [N_sym, 3] (N_sym because a different mask is used for each symm. gt)
                pred_coords_sym_centroid = get_centroid(
                    pred_coords_sym.unsqueeze(0), gt_resolved_mask_entity_segment
                )

                # Get squared distances between centroids
                # [N_sym]
                pred_gt_dists_sq = (
                    (pred_coords_sym_centroid - gt_coords_entity_centroids)
                    .pow(2)
                    .sum(dim=-1)
                )

                # Mask already used IDs
                used_gt_sym_ids_mask = torch.isin(
                    gt_unique_sym_ids_entity,
                    torch.tensor(
                        used_gt_sym_ids, device=gt_unique_sym_ids_entity.device
                    ),
                )
                pred_gt_dists_sq[used_gt_sym_ids_mask] = torch.inf

                # Make sure that entirely unresolved ground-truth centroids are picked
                # last (by setting higher than any other dist but lower than inf)
                gt_any_resolved_mask_centroid = gt_resolved_mask_entity_segment.any(
                    dim=-1
                )
                pred_gt_dists_sq[~gt_any_resolved_mask_centroid] = torch.finfo(
                    pred_gt_dists_sq.dtype
                ).max

                # Get the best matching ground-truth symmetry ID
                best_gt_index = torch.argmin(pred_gt_dists_sq)
                best_gt_sym_id = gt_unique_sym_ids_entity[best_gt_index].item()
                best_gt_sym_id_dist_sq = pred_gt_dists_sq[best_gt_index]

                centroid_dists_sq_aln.append(best_gt_sym_id_dist_sq)
                used_gt_sym_ids.append(best_gt_sym_id)

                # Append the mapping between the predicted and ground-truth symmetry and
                # entity IDs to the permutation
                permutation_aln[(entity_id, sym_id)] = (
                    entity_id,
                    best_gt_sym_id,
                )

        centroid_dists_sq.append(centroid_dists_sq_aln)
        permutations.append(permutation_aln)

    centroid_dists_sq = torch.tensor(centroid_dists_sq, device=gt_mol_sym_ids.device)

    # Change the max-value placeholder of entirely unresolved chains to be just above
    # the max of any resolved centroid distance for numerical stability
    unresolved_vals = centroid_dists_sq == torch.finfo(centroid_dists_sq.dtype).max
    centroid_dists_sq[unresolved_vals] = centroid_dists_sq[~unresolved_vals].max() + 1

    # Single RMSD over all centroid distances (this is a slight deviation from the
    # AF2-Multimer SI which calculates an RMSD per centroid, which would just be
    # equivalent to summing up absolute distances)
    # [N_align, N_centroid] -> [N_align]
    rmsds = torch.sqrt(centroid_dists_sq.mean(dim=-1))
    optimal_alignment_index = torch.argmin(rmsds)
    optimal_permutation = permutations[optimal_alignment_index]

    # This RMSD re-computation is only for better logging, to give the RMSD without the
    # artificial values for unresolved atoms (e.g. to see that it is near-0 in
    # unit-tests)
    centroid_dists_sq_optimal = centroid_dists_sq[optimal_alignment_index]
    unresolved_vals_optimal = unresolved_vals[optimal_alignment_index]
    optimal_rmsd_masked = torch.sqrt(
        centroid_dists_sq_optimal[~unresolved_vals_optimal].mean()
    )
    logger.debug(f"Found optimal permutation with RMSD: {optimal_rmsd_masked}")

    assert optimal_permutation is not None

    return optimal_permutation


# TODO: Could restrict global alignment to non-symmetric atoms which could be slightly
# cleaner
def get_permuted_gt_token_subset_index(
    gt_mol_entity_ids: torch.Tensor,
    gt_mol_sym_ids: torch.Tensor,
    gt_mol_sym_token_index: torch.Tensor,
    pred_mol_entity_ids: torch.Tensor,
    pred_mol_sym_ids: torch.Tensor,
    pred_mol_sym_token_index: torch.Tensor,
    optimal_permutation: dict[tuple[int, int], tuple[int, int]],
) -> torch.Tensor:
    """Gets a selector for the ground-truth tokens to map to the predicted tokens.

    This function creates a selection-index on token-level features that will reorder
    and subset the ground-truth tokens in a way that there is a direct 1:1
    correspondence between predicted and GT-tokens following the optimal molecule
    permutation.

    Note that in the case of ligand molecules, the resulting pred-gt mapping of
    token-center atoms is not yet an optimized intra-ligand atom-wise permutation, but
    represents an arbitrary mapping of the predicted ligand molecules to the
    ground-truth ligand molecules. This may introduce slight noise in the alignment but
    seems negligible in practice for most complexes.

    Args:
        gt_mol_entity_ids (torch.Tensor):
            [N] the molecular entity IDs of the ground-truth token center atoms.
        gt_mol_sym_ids (torch.Tensor):
            [N] the molecular symmetry IDs of the ground-truth token center atoms.
        gt_mol_sym_token_index (torch.Tensor):
            [N] the molecular symmetry token indices of the ground-truth token center
            atoms.
        pred_mol_entity_ids (torch.Tensor):
            [N] the molecular entity IDs of the predicted token center atoms.
        pred_mol_sym_ids (torch.Tensor):
            [N] the molecular symmetry IDs of the predicted token center atoms.
        pred_mol_sym_token_index (torch.Tensor):
            [N] the molecular symmetry token indices of the predicted token center
            atoms.
        optimal_permutation (dict[tuple[int, int], tuple[int, int]]):
            The optimal mapping of (entity_id, sym_id) pairs between the predicted and
            ground-truth molecules.

    Returns:
        torch.Tensor:
            An index tensor of shape [N_pred] that selects the ground-truth token center
            atoms that correspond to the predicted token center atoms.
    """
    device = pred_mol_entity_ids.device

    # Final index that will rearrange gt-token features to exactly match the prediction
    # (both in terms of matching the in-crop segment and the permuted order)
    n_tokens_pred = pred_mol_entity_ids.shape[0]
    token_subset_index = -torch.ones(
        n_tokens_pred,
        dtype=torch.long,
        device=device,
    )

    n_tokens_gt = gt_mol_entity_ids.shape[0]
    gt_token_index = torch.arange(n_tokens_gt, device=device)

    for (pred_entity_id, pred_sym_id), (
        gt_entity_id,
        gt_sym_id,
    ) in optimal_permutation.items():
        # The exact section of the prediction feature tensor that corresponds to the
        # current (entity, sym) segment
        pred_segment_mask = (pred_mol_entity_ids == pred_entity_id) & (
            pred_mol_sym_ids == pred_sym_id
        )

        # The exact section of the ground-truth feature tensor that corresponds to the
        # matching (entity, sym) segment as given by the optimal permutation
        gt_segment_mask = get_gt_segment_mask(
            segment_mol_sym_token_index=pred_mol_sym_token_index[pred_segment_mask],
            gt_mol_sym_token_index=gt_mol_sym_token_index,
            segment_mol_entity_id=gt_entity_id,
            segment_mol_sym_id=gt_sym_id,
            gt_mol_entity_id=gt_mol_entity_ids,
            gt_mol_sym_id=gt_mol_sym_ids,
        )

        # Insert the corresponding token index
        token_subset_index[pred_segment_mask] = gt_token_index[gt_segment_mask]

    assert torch.all(token_subset_index != -1)
    assert torch.equal(gt_mol_entity_ids[token_subset_index], pred_mol_entity_ids)

    return token_subset_index


def get_pred_to_permuted_gt_transformation(
    gt_token_center_positions: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_mol_entity_ids: torch.Tensor,
    gt_mol_sym_ids: torch.Tensor,
    gt_mol_sym_token_index: torch.Tensor,
    pred_token_center_positions: torch.Tensor,
    pred_mol_entity_ids: torch.Tensor,
    pred_mol_sym_ids: torch.Tensor,
    pred_mol_sym_token_index: torch.Tensor,
    optimal_permutation: dict[tuple[int, int], tuple[int, int]],
) -> Transformation:
    """Computes a Kabsch alignment between pred. and GT following optimal permutation.

    This function computes a Kabsch alignment of the predicted coordinates to the
    ground-truth coordinates after the optimal molecule-wise permutation has been
    applied. The resulting aligned structure can be used to resolve the intra-residue
    symmetric atom permutations, following AF3 SI 4.2.

    Note that the alignment computation for symmetric atoms will have to use the
    arbitrary original permutation, as the intra-residue atom assignment is not yet
    known.

    Args:
        gt_token_center_positions (torch.Tensor):
            [N, 3] the token center positions of the ground-truth molecules.
        gt_token_center_resolved_mask (torch.Tensor):
            [N] mask of resolved token center atoms in the ground-truth molecules.
        gt_mol_entity_ids (torch.Tensor):
            [N] the molecular entity IDs of the ground-truth token center atoms.
        gt_mol_sym_ids (torch.Tensor):
            [N] the molecular symmetry IDs of the ground-truth token center atoms.
        gt_mol_sym_token_index (torch.Tensor):
            [N] the molecular symmetry token indices of the ground-truth token center
            atoms.
        pred_token_center_positions (torch.Tensor):
            [N, 3] the token center positions of the predicted molecules.
        pred_mol_entity_ids (torch.Tensor):
            [N] the molecular entity IDs of the predicted token center atoms.
        pred_mol_sym_ids (torch.Tensor):
            [N] the molecular symmetry IDs of the predicted token center atoms.
        pred_mol_sym_token_index (torch.Tensor):
            [N] the molecular symmetry token indices of the predicted token center
            atoms.
        optimal_permutation (dict[tuple[int, int], tuple[int, int]]):
            The optimal mapping of (entity_id, sym_id) pairs between the predicted and
            ground-truth molecules.
    Returns:
        Transformation:
            A NamedTuple containing the rotation matrix and translation vector of the
            optimal transformation, such that:

            pred_token_center_positions @ R + t ≈ gt_token_center_positions
    """

    # Rearrange and subset the ground-truth coordinates to match the prediction (this
    # maps the overall symmetric molecules onto each other following the optimal
    # permutation, but still uses an arbitrary assignment of symmetry-equivalent
    # intra-residue atoms based on the original ordering)
    gt_token_subset_index = get_permuted_gt_token_subset_index(
        gt_mol_entity_ids=gt_mol_entity_ids,
        gt_mol_sym_ids=gt_mol_sym_ids,
        gt_mol_sym_token_index=gt_mol_sym_token_index,
        pred_mol_entity_ids=pred_mol_entity_ids,
        pred_mol_sym_ids=pred_mol_sym_ids,
        pred_mol_sym_token_index=pred_mol_sym_token_index,
        optimal_permutation=optimal_permutation,
    )

    gt_token_center_positions_rearranged = gt_token_center_positions[
        gt_token_subset_index
    ]
    gt_token_center_resolved_mask_rearranged = gt_token_center_resolved_mask[
        gt_token_subset_index
    ]

    transformation = get_optimal_transformation(
        mobile_positions=pred_token_center_positions,
        target_positions=gt_token_center_positions_rearranged,
        positions_mask=gt_token_center_resolved_mask_rearranged,
    )

    return transformation


def get_token_identity_gt_token_subset_index(
    *,
    gt_token_index: torch.Tensor,
    pred_token_index: torch.Tensor,
    gt_asym_id: torch.Tensor | None = None,
    pred_asym_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map prediction tokens to GT tokens by the stable source token index.

    PhysPPI addition (Jinyu Lin, ShanghaiTech):
    This is the core of the token_identity alignment strategy.  It bypasses the
    heavyweight Hungarian chain-matching algorithm and instead directly looks up
    GT token rows by their shared token_index values.  When asym_id is available
    on both sides, it additionally verifies that the mapping preserves chain
    identity — a mismatch means token_index alone is insufficient and the caller
    should fall back to protected chain matching.
    """
    gt_tokens = [int(value) for value in gt_token_index.detach().cpu().tolist()]
    pred_tokens = [int(value) for value in pred_token_index.detach().cpu().tolist()]
    duplicate_gt_tokens = [
        token for token, count in Counter(gt_tokens).items() if count > 1
    ]
    if duplicate_gt_tokens:
        raise ProtectedChainMatchError("token_identity_duplicate_gt_token_index")

    duplicate_pred_tokens = [
        token for token, count in Counter(pred_tokens).items() if count > 1
    ]
    if duplicate_pred_tokens:
        raise ProtectedChainMatchError("token_identity_duplicate_pred_token_index")

    gt_token_to_row = {token: row for row, token in enumerate(gt_tokens)}
    missing_tokens = [token for token in pred_tokens if token not in gt_token_to_row]
    if missing_tokens:
        raise ProtectedChainMatchError("token_identity_missing_gt_token_index")

    gt_token_subset_index = torch.tensor(
        [gt_token_to_row[token] for token in pred_tokens],
        dtype=torch.long,
        device=gt_token_index.device,
    )
    if gt_asym_id is not None and pred_asym_id is not None:
        matched_gt_asym = gt_asym_id[gt_token_subset_index].detach().cpu()
        pred_asym = pred_asym_id.detach().cpu()
        if not torch.equal(matched_gt_asym, pred_asym):
            raise ProtectedChainMatchError("token_identity_asym_id_mismatch")

    return gt_token_subset_index


def _token_atom_start_index(num_atoms_per_token: torch.Tensor) -> torch.Tensor:
    num_atoms_per_token = num_atoms_per_token.long()
    if num_atoms_per_token.numel() == 0:
        return torch.empty_like(num_atoms_per_token)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=num_atoms_per_token.device),
            torch.cumsum(num_atoms_per_token[:-1], dim=0),
        ]
    )


def _token_identity_atom_index_before_symmetry(
    *,
    gt_positions: torch.Tensor,
    gt_token_subset_index: torch.Tensor,
    gt_num_atoms_per_token: torch.Tensor,
    pred_positions_aligned: torch.Tensor,
    pred_num_atoms_per_token: torch.Tensor,
) -> torch.Tensor:
    gt_num_atoms_per_token = gt_num_atoms_per_token.long()
    pred_num_atoms_per_token = pred_num_atoms_per_token.long()
    if int(gt_num_atoms_per_token.sum().item()) != gt_positions.shape[0]:
        raise ProtectedChainMatchError("token_identity_gt_atom_count_mismatch")
    if int(pred_num_atoms_per_token.sum().item()) != pred_positions_aligned.shape[0]:
        raise ProtectedChainMatchError("token_identity_pred_atom_count_mismatch")

    matched_gt_num_atoms = gt_num_atoms_per_token[gt_token_subset_index]
    if not torch.equal(
        matched_gt_num_atoms.detach().cpu(),
        pred_num_atoms_per_token.detach().cpu(),
    ):
        raise ProtectedChainMatchError("token_identity_token_atom_count_mismatch")

    gt_start_index = _token_atom_start_index(gt_num_atoms_per_token)
    atom_subset_index = -torch.ones(
        pred_positions_aligned.shape[0],
        dtype=torch.long,
        device=pred_positions_aligned.device,
    )

    pred_atom_start = 0
    for pred_token_row, gt_token_row in enumerate(
        gt_token_subset_index.detach().cpu().tolist()
    ):
        num_atoms = int(pred_num_atoms_per_token[pred_token_row].item())
        gt_atom_start = int(gt_start_index[gt_token_row].item())
        atom_subset_index[pred_atom_start : pred_atom_start + num_atoms] = torch.arange(
            gt_atom_start,
            gt_atom_start + num_atoms,
            dtype=torch.long,
            device=atom_subset_index.device,
        )
        pred_atom_start += num_atoms

    if torch.any(atom_subset_index < 0):
        raise ProtectedChainMatchError("token_identity_unassigned_atom_index")

    return atom_subset_index


def get_token_identity_atom_permutation_index(
    *,
    gt_positions: torch.Tensor,
    gt_resolved_mask: torch.Tensor,
    gt_token_index: torch.Tensor,
    gt_num_atoms_per_token: torch.Tensor,
    gt_asym_id: torch.Tensor | None = None,
    pred_positions_aligned: torch.Tensor,
    pred_token_index: torch.Tensor,
    pred_num_atoms_per_token: torch.Tensor,
    pred_asym_id: torch.Tensor | None = None,
    pred_ref_space_uid: torch.Tensor,
    pred_ref_space_uid_to_perm: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Select GT atoms by token identity and resolve atom-level symmetry."""
    gt_token_subset_index = get_token_identity_gt_token_subset_index(
        gt_token_index=gt_token_index,
        pred_token_index=pred_token_index,
        gt_asym_id=gt_asym_id,
        pred_asym_id=pred_asym_id,
    )
    identity_atom_index = _token_identity_atom_index_before_symmetry(
        gt_positions=gt_positions,
        gt_token_subset_index=gt_token_subset_index,
        gt_num_atoms_per_token=gt_num_atoms_per_token,
        pred_positions_aligned=pred_positions_aligned,
        pred_num_atoms_per_token=pred_num_atoms_per_token,
    )
    if pred_ref_space_uid.shape[0] != pred_positions_aligned.shape[0]:
        raise ProtectedChainMatchError("token_identity_ref_space_uid_count_mismatch")

    gt_resolved_mask = gt_resolved_mask.bool()
    atom_subset_index = -torch.ones_like(identity_atom_index)
    for ref_space_uid in torch.unique(pred_ref_space_uid).detach().cpu().tolist():
        pred_mask = pred_ref_space_uid == ref_space_uid
        gt_atom_idx_subset = identity_atom_index[pred_mask]
        gt_positions_subset = gt_positions[gt_atom_idx_subset]
        gt_resolved_mask_subset = gt_resolved_mask[gt_atom_idx_subset]
        pred_positions_subset = pred_positions_aligned[pred_mask]

        if ref_space_uid not in pred_ref_space_uid_to_perm:
            raise ProtectedChainMatchError(
                "token_identity_missing_ref_space_permutation"
            )
        permutations = pred_ref_space_uid_to_perm[ref_space_uid].to(
            device=gt_atom_idx_subset.device,
            dtype=torch.long,
        )
        if permutations.ndim == 1:
            permutations = permutations.unsqueeze(0)
        if permutations.shape[-1] != gt_atom_idx_subset.shape[0]:
            raise ProtectedChainMatchError(
                "token_identity_ref_space_permutation_size_mismatch"
            )

        check_out_of_bounds_indices(
            indices=permutations,
            input_tensor=gt_atom_idx_subset,
        )
        if permutations.shape[0] == 1:
            best_permutation = permutations[0]
        else:
            gt_positions_permuted = gt_positions_subset[permutations]
            gt_resolved_mask_permuted = gt_resolved_mask_subset[permutations]
            dists_sq = (
                (pred_positions_subset - gt_positions_permuted)
                .pow(2)
                .sum(dim=-1)
                .mul(gt_resolved_mask_permuted)
            )
            n_resolved_atoms = gt_resolved_mask_permuted.sum(dim=-1)
            rmsd = torch.full(
                (permutations.shape[0],),
                torch.inf,
                dtype=pred_positions_subset.dtype,
                device=pred_positions_subset.device,
            )
            resolved = n_resolved_atoms > 0
            if torch.any(resolved):
                rmsd[resolved] = torch.sqrt(
                    dists_sq.sum(dim=-1)[resolved] / n_resolved_atoms[resolved]
                )
                best_permutation = permutations[torch.argmin(rmsd)]
            else:
                best_permutation = permutations[0]

        atom_subset_index[pred_mask] = gt_atom_idx_subset[best_permutation]

    if torch.any(atom_subset_index < 0):
        raise ProtectedChainMatchError("token_identity_unassigned_final_atom_index")

    return atom_subset_index


def get_final_atom_permutation_index(
    gt_positions: torch.Tensor,
    gt_resolved_mask: torch.Tensor,
    gt_mol_entity_id: torch.Tensor,
    gt_mol_sym_id: torch.Tensor,
    gt_mol_sym_component_id: torch.Tensor,
    gt_num_atoms_per_token: torch.Tensor,
    pred_positions_aligned: torch.Tensor,
    pred_mol_entity_id: torch.Tensor,
    pred_mol_sym_id: torch.Tensor,
    pred_mol_sym_component_id: torch.Tensor,
    pred_num_atoms_per_token: torch.Tensor,
    pred_ref_space_uid: torch.Tensor,
    pred_ref_space_uid_to_perm: dict[int, torch.Tensor],
    optimal_mol_permutation: dict[tuple[int, int], tuple[int, int]],
):
    """Gets the final atom-level index tensor to permute the ground-truth atoms.

    Starting from the optimal molecule-wise permutation and a prediction coordinates
    Kabsch-aligned to the ground-truth, this function computes the optimal atom-level
    index that reorders and subsets the ground-truth atoms to best match the prediction.
    In contrast to the molecule-wise permutation, this also resolved the intra-residue
    symmetric atom permutations by minimizing RMSD, following AF3 SI 4.2.

    Args:
        gt_positions (torch.Tensor):
            [N, 3] the ground-truth all-atom positions.
        gt_resolved_mask (torch.Tensor):
            [N] mask of resolved atoms in the ground-truth.
        gt_mol_entity_id (torch.Tensor):
            [N] the entity IDs of the ground-truth tokens.
        gt_mol_sym_id (torch.Tensor):
            [N] the symmetry IDs of the ground-truth tokens.
        gt_mol_sym_component_id (torch.Tensor):
            [N] the symmetry component IDs of the ground-truth tokens.
        gt_num_atoms_per_token (torch.Tensor):
            [N] the number of atoms per ground-truth token.
        pred_positions_aligned (torch.Tensor):
            [N, 3] the predicted all-atom positions aligned to the ground-truth.
        pred_mol_entity_id (torch.Tensor):
            [N] the entity IDs of the predicted tokens.
        pred_mol_sym_id (torch.Tensor):
            [N] the symmetry IDs of the predicted tokens.
        pred_mol_sym_component_id (torch.Tensor):
            [N] the symmetry component IDs of the predicted tokens.
        pred_num_atoms_per_token (torch.Tensor):
            [N] the number of atoms per predicted token.
        pred_ref_space_uid (torch.Tensor):
            [N] the ref_space_uids (absolute component IDs) of the predicted tokens.
        pred_ref_space_uid_to_perm (dict[int, torch.Tensor]):
            A dictionary mapping ref_space_uids to the possible relative ground-truth
            atom permutations.
        optimal_mol_permutation (dict[tuple[int, int], tuple[int, int]]):
            The optimal mapping of (entity_id, sym_id) pairs between the predicted and
            ground-truth molecules.

    Returns:
        torch.Tensor:
            An index tensor of shape [N_pred] that selects the ground-truth atoms that
            correspond to the predicted atoms.
    """
    gt_resolved_mask = gt_resolved_mask.bool()

    # Will create a final indexing operation into the gt-features (initialize to -1 just
    # for easier assert at the end)
    atom_subset_index = -torch.ones(
        pred_positions_aligned.shape[0],
        dtype=torch.long,
        device=pred_positions_aligned.device,
    )

    # Indices of all atoms in the ground-truth
    gt_all_atom_index = torch.arange(gt_positions.shape[0], device=gt_positions.device)

    # Expand features to atom-wise level
    dummy_token_mask = torch.ones_like(pred_mol_entity_id, dtype=torch.bool)
    pred_token_feat_to_atoms = partial(
        broadcast_token_feat_to_atoms,
        dummy_token_mask,
        pred_num_atoms_per_token,
    )
    pred_mol_entity_id_atom = pred_token_feat_to_atoms(token_feat=pred_mol_entity_id)
    pred_mol_sym_id_atom = pred_token_feat_to_atoms(pred_mol_sym_id)
    pred_sym_conformer_id_atom = pred_token_feat_to_atoms(pred_mol_sym_component_id)

    dummy_token_mask = torch.ones_like(gt_mol_entity_id, dtype=torch.bool)
    gt_token_feat_to_atoms = partial(
        broadcast_token_feat_to_atoms,
        dummy_token_mask,
        gt_num_atoms_per_token,
    )
    gt_mol_entity_id_atom = gt_token_feat_to_atoms(gt_mol_entity_id)
    gt_mol_sym_id_atom = gt_token_feat_to_atoms(gt_mol_sym_id)
    gt_sym_conformer_id_atom = gt_token_feat_to_atoms(gt_mol_sym_component_id)

    for (pred_entity_id, pred_sym_id), (
        gt_entity_id,
        gt_sym_id,
    ) in optimal_mol_permutation.items():
        # Subset the predicted coords to current entity and symmetry ID
        pred_mask = (pred_mol_entity_id_atom == pred_entity_id) & (
            pred_mol_sym_id_atom == pred_sym_id
        )
        pred_positions_subset = pred_positions_aligned[pred_mask]
        pred_ref_space_uids_subset = pred_ref_space_uid[pred_mask]

        pred_sym_conformer_id_atom_subset = pred_sym_conformer_id_atom[pred_mask]
        unique_sym_conformer_ids_subset = torch.unique_consecutive(
            pred_sym_conformer_id_atom_subset
        )

        # Subset ground-truth coords to current entity and symmetry ID
        gt_mask = (gt_mol_entity_id_atom == gt_entity_id) & (
            gt_mol_sym_id_atom == gt_sym_id
        )
        # Subset ground-truth to only the conformer-instances present in the prediction
        # (while still including potential extra symmetry-expanded atoms)
        gt_conformer_mask = torch.isin(
            gt_sym_conformer_id_atom[gt_mask],
            unique_sym_conformer_ids_subset,
        )
        gt_positions_subset = gt_positions[gt_mask][gt_conformer_mask]
        gt_resolved_mask_subset = gt_resolved_mask[gt_mask][gt_conformer_mask]
        gt_atom_idx_subset = gt_all_atom_index[gt_mask][gt_conformer_mask]
        gt_sym_conformer_id_atom_subset = gt_sym_conformer_id_atom[gt_mask][
            gt_conformer_mask
        ]

        # Group predictions by ref-space UID (= absolute conformer ID) - this gets the
        # same split as grouping by symmetry conformer ID but we need the absolute IDs
        # for permutation mapping
        pred_positions_subset_grouped, unique_ref_space_uids_subset = split_feats_by_id(
            pred_positions_subset, pred_ref_space_uids_subset
        )

        (
            (
                gt_positions_subset_grouped,
                gt_resolved_mask_subset_grouped,
                gt_atom_idx_subset_grouped,
            ),
            _,
        ) = split_feats_by_id(
            [gt_positions_subset, gt_resolved_mask_subset, gt_atom_idx_subset],
            gt_sym_conformer_id_atom_subset,
        )

        assert unique_ref_space_uids_subset.shape[0] == len(
            pred_positions_subset_grouped
        )

        # Get the optimal permutation for each conformer
        permuted_atom_idxs = []
        for (
            ref_space_uid,
            pred_positions_subset_conf,
            gt_positions_subset_conf,
            gt_resolved_mask_subset_conf,
            gt_atom_idx_subset_conf,
        ) in zip(
            unique_ref_space_uids_subset,
            pred_positions_subset_grouped,
            gt_positions_subset_grouped,
            gt_resolved_mask_subset_grouped,
            gt_atom_idx_subset_grouped,
            strict=False,
        ):
            # All possible permutations for this conformer
            permutations = pred_ref_space_uid_to_perm[ref_space_uid.item()]

            # If there is only a single permutation (which is the case for a lot of
            # residues) skip the remaining computations
            if permutations.shape[0] == 1:
                identity_permutation = permutations[0]

                check_out_of_bounds_indices(
                    indices=identity_permutation,
                    input_tensor=gt_atom_idx_subset_conf,
                )

                permuted_atom_idxs.extend(
                    gt_atom_idx_subset_conf[identity_permutation].tolist()
                )

                # Ensure that the permutation does not change the order of the atoms
                # (which would be unexpected)
                assert torch.diff(identity_permutation).gt(0).all()

                continue

            # Versions of the ground-truth positions for each permutation
            check_out_of_bounds_indices(
                indices=permutations, input_tensor=gt_positions_subset_conf
            )
            check_out_of_bounds_indices(
                indices=permutations, input_tensor=gt_resolved_mask_subset_conf
            )
            gt_positions_subset_conf_perm = gt_positions_subset_conf[permutations]
            gt_resolved_mask_subset_conf_perm = gt_resolved_mask_subset_conf[
                permutations
            ]

            # Minimize RMSD
            dists_sq = (
                (pred_positions_subset_conf - gt_positions_subset_conf_perm)
                .pow(2)
                .sum(dim=-1)
                .mul(gt_resolved_mask_subset_conf_perm)
            )
            n_resolved_atoms = gt_resolved_mask_subset_conf_perm.sum(dim=-1)
            rmsd = torch.sqrt(dists_sq.sum(dim=-1) / n_resolved_atoms)

            # Get permutation that minimizes RMSD
            check_out_of_bounds_indices(
                indices=torch.argmin(rmsd), input_tensor=permutations
            )
            best_permutation = permutations[torch.argmin(rmsd)]

            check_out_of_bounds_indices(
                indices=best_permutation, input_tensor=gt_atom_idx_subset_conf
            )

            # Append the global atom indices corresponding to this permutation
            permuted_atom_idxs.extend(
                gt_atom_idx_subset_conf[best_permutation].tolist()
            )

        # Add the atom indices to the final atom index that tracks the global
        # permutation. We use pred_mask here to ensure that we directly match the layout
        # of the pred-features, even if the features belonging to a connected molecule
        # are not contiguous.
        atom_subset_index[pred_mask] = torch.tensor(
            permuted_atom_idxs, device=gt_mol_sym_id.device
        )

    assert torch.all(atom_subset_index != -1)

    return atom_subset_index


def permute_gt_atom_features(
    features: list[torch.Tensor],
    atom_indexes: list[torch.Tensor],
) -> tuple[torch.Tensor]:
    """Permute ground-truth atom-wise features according to a given index tensor.

    Args:
        features: (list[torch.Tensor]):
            A list of atom-wise features to permute. Each tensor should have shape [B,
            N, ...] where B is the batch size and N is the number of atoms.
        atom_indexes (list[torch.Tensor]):
            A list of atom indices to permute the features for each batch. The list
            should be of length B, and each tensor should have shape [N'] where N' is
            the number of atoms in the permuted order.

    Returns:
        tuple[torch.Tensor]:
            A tuple of permuted features of shape [B, N', ...], containing the features
            in the same order as the input features list.
    """
    batch_size = features[0].shape[0]
    new_features = []

    for feat in features:
        # Pad to longest sequence
        new_feat = torch.nn.utils.rnn.pad_sequence(
            [feat[batch][atom_indexes[batch]] for batch in range(batch_size)],
            batch_first=True,
        )
        new_features.append(new_feat)

    return tuple(new_features)


def update_gt_position_features(
    batch: dict, gt_atom_indexes: list[torch.Tensor]
) -> dict:
    """Rearranges the ground-truth coordinates and resolved mask by index tensor.

    Args:
        batch (dict):
            The initial feature dictionary containing a mini-batch of features.
        gt_atom_indexes (list[torch.Tensor]):
            A list of index tensors that permute the ground-truth atom-wise features for
            each batch.

    Returns:
        dict:
            The updated feature dictionary with the ground-truth "atom_positions" and
            "atom_resolved_mask" features rearranged to perfectly match the prediction,
            as defined by the given index tensors.
    """
    ground_truth_features = batch["ground_truth"]

    gt_atom_positions_permuted, gt_atom_resolved_mask_permuted = (
        permute_gt_atom_features(
            [
                ground_truth_features["atom_positions"],
                ground_truth_features["atom_resolved_mask"],
            ],
            gt_atom_indexes,
        )
    )

    updated_ground_truth_features = {
        "atom_positions": gt_atom_positions_permuted,
        "atom_resolved_mask": gt_atom_resolved_mask_permuted,
    }

    intra_filter_atomized = ground_truth_features.get("intra_filter_atomized")
    if intra_filter_atomized is not None:
        intra_filter_atomized = permute_gt_atom_features(
            [intra_filter_atomized], gt_atom_indexes
        )[0]
        updated_ground_truth_features["intra_filter_atomized"] = intra_filter_atomized

    inter_filter_atomized = ground_truth_features.get("inter_filter_atomized")
    if inter_filter_atomized is not None:
        inter_filter_atomized = permute_gt_atom_features(
            [inter_filter_atomized], gt_atom_indexes
        )[0]
        inter_filter_atomized = permute_gt_atom_features(
            [inter_filter_atomized.transpose(-1, -2)], gt_atom_indexes
        )[0]
        inter_filter_atomized = inter_filter_atomized.transpose(-1, -2)
        updated_ground_truth_features["inter_filter_atomized"] = inter_filter_atomized

    return updated_ground_truth_features


def _protected_representative_atoms(
    token_features: dict,
    atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select CA/C4' (or canonical atomized centers) without random probes."""
    start_atom_index = token_features["start_atom_index"].long()
    is_atomized = token_features["is_atomized"].bool()
    is_protein = token_features["is_protein"].bool() & ~is_atomized
    is_nucleotide = (
        token_features["is_rna"].bool() | token_features["is_dna"].bool()
    ) & ~is_atomized
    restype = token_features["restype"]
    ca_offset, ca_mask = get_token_atom_index_offset("CA", restype)
    c4_offset, c4_mask = get_token_atom_index_offset("C4'", restype)

    representative_index = start_atom_index.clone()
    representative_index = torch.where(
        is_protein,
        start_atom_index + ca_offset,
        representative_index,
    )
    representative_index = torch.where(
        is_nucleotide,
        start_atom_index + c4_offset,
        representative_index,
    )
    representative_supported = (
        is_atomized | (is_protein & ca_mask.bool()) | (is_nucleotide & c4_mask.bool())
    )
    representative_supported &= token_features["token_mask"].bool()
    representative_supported &= representative_index >= 0
    representative_supported &= representative_index < atom_positions.shape[-2]
    if "num_atoms_per_token" in token_features:
        representative_supported &= representative_index < (
            start_atom_index + token_features["num_atoms_per_token"].long()
        )

    safe_index = torch.where(
        representative_supported,
        representative_index,
        torch.zeros_like(representative_index),
    )
    representative_xyz = atom_positions[safe_index]
    representative_mask = atom_mask[safe_index].bool() & representative_supported
    representative_mask &= torch.isfinite(representative_xyz).all(dim=-1)
    return representative_xyz, representative_mask


def _chain_records_from_features(
    token_features: dict,
    atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
) -> tuple[tuple[ChainRecord, ...], dict[str, tuple[int, int]]]:
    """Build deterministic protected-matcher records from model feature tensors."""
    representative_xyz, representative_mask = _protected_representative_atoms(
        token_features,
        atom_positions,
        atom_mask,
    )
    token_mask = token_features["token_mask"].bool()
    entity_ids = token_features["mol_entity_id"].long()
    sym_ids = token_features["mol_sym_id"].long()
    sym_token_index = token_features["mol_sym_token_index"].long()
    restype_index = token_features["restype"].argmax(dim=-1)

    molecule_ids = sorted(
        {
            (int(entity_id), int(sym_id))
            for entity_id, sym_id in zip(
                entity_ids[token_mask].detach().cpu().tolist(),
                sym_ids[token_mask].detach().cpu().tolist(),
                strict=True,
            )
        }
    )
    records: list[ChainRecord] = []
    record_ids: dict[str, tuple[int, int]] = {}
    for entity_id, sym_id in molecule_ids:
        rows = torch.nonzero(
            token_mask & (entity_ids == entity_id) & (sym_ids == sym_id),
            as_tuple=True,
        )[0]
        rows = rows[torch.argsort(sym_token_index[rows], stable=True)]
        molecule_flags = {
            polymer_type
            for polymer_type, feature_name in (
                ("protein", "is_protein"),
                ("rna", "is_rna"),
                ("dna", "is_dna"),
                ("ligand", "is_ligand"),
            )
            if token_features[feature_name][rows].bool().any()
        }
        polymer_type = (
            next(iter(molecule_flags)) if len(molecule_flags) == 1 else "mixed"
        )
        sequence = tuple(
            STANDARD_RESIDUES_WITH_GAP_3[index]
            for index in restype_index[rows].detach().cpu().tolist()
        )
        chain_id = f"entity={entity_id:020d}:sym={sym_id:020d}"
        records.append(
            ChainRecord(
                chain_id=chain_id,
                polymer_type=polymer_type,
                sequence=sequence,
                representative_xyz=representative_xyz[rows],
                representative_mask=representative_mask[rows],
                residue_keys=tuple(
                    (chain_id, int(token_row), "")
                    for token_row in sym_token_index[rows].detach().cpu().tolist()
                ),
            )
        )
        record_ids[chain_id] = (entity_id, sym_id)
    return tuple(records), record_ids


def find_protected_mol_permutation(
    *,
    pred_features: dict,
    pred_atom_positions: torch.Tensor,
    pred_atom_mask: torch.Tensor,
    gt_features: dict,
    gt_atom_positions: torch.Tensor,
    gt_atom_mask: torch.Tensor,
) -> tuple[
    dict[tuple[int, int], tuple[int, int]],
    dict[str, bool | str | float | int],
]:
    """Translate protected chain matching into OF3 molecule permutation IDs."""
    pred_records, pred_ids = _chain_records_from_features(
        pred_features,
        pred_atom_positions,
        pred_atom_mask,
    )
    gt_records, gt_ids = _chain_records_from_features(
        gt_features,
        gt_atom_positions,
        gt_atom_mask,
    )
    result = protected_chain_match(pred_records, gt_records)
    mapping = {
        pred_ids[pred_chain_id]: gt_ids[gt_chain_id]
        for pred_chain_id, gt_chain_id in result.mapping.items()
    }
    diagnostics: dict[str, bool | str | float | int] = {
        "accepted": result.accepted,
        "fallback": "none",
        "reason": result.reason,
        "identity_rmsd": result.identity_rmsd,
        "candidate_rmsd": result.candidate_rmsd,
        "matched_coverage": result.matched_coverage,
    }
    return mapping, diagnostics


def _sequences_compatible_for_identity(
    source: ChainRecord,
    target: ChainRecord,
) -> bool:
    if source.polymer_type != target.polymer_type:
        return False
    if source.sequence == target.sequence:
        return True
    alignment = align_chain_residues(source, target)
    return (
        not alignment.has_conflicting_known_residue
        and alignment.aligned_source_rows.numel() > 0
    )


def find_sequence_compatible_identity_mol_permutation(
    *,
    pred_features: dict,
    pred_atom_positions: torch.Tensor,
    pred_atom_mask: torch.Tensor,
    gt_features: dict,
    gt_atom_positions: torch.Tensor,
    gt_atom_mask: torch.Tensor,
    reason: str,
) -> tuple[
    dict[tuple[int, int], tuple[int, int]],
    dict[str, bool | str | float | int],
]:
    """Build the deterministic identity/source-order fallback mapping."""
    pred_records, pred_ids = _chain_records_from_features(
        pred_features,
        pred_atom_positions,
        pred_atom_mask,
    )
    gt_records, gt_ids = _chain_records_from_features(
        gt_features,
        gt_atom_positions,
        gt_atom_mask,
    )
    source_records = sorted(pred_records, key=lambda record: record.chain_id)
    target_records = sorted(gt_records, key=lambda record: record.chain_id)
    compatibility_cost = np.asarray(
        [
            [
                (0.0 if _sequences_compatible_for_identity(source, target) else np.inf)
                for target in target_records
            ]
            for source in source_records
        ],
        dtype=np.float64,
    )
    try:
        rows, columns = lexicographic_linear_sum_assignment(compatibility_cost)
    except ProtectedChainMatchError as error:
        raise ProtectedChainMatchError(
            "identity_fallback_no_complete_sequence_compatible_assignment"
        ) from error

    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    coverage = 0
    for row, column in zip(rows, columns, strict=True):
        source = source_records[int(row)]
        target = target_records[int(column)]
        mapping[pred_ids[source.chain_id]] = gt_ids[target.chain_id]
        coverage += int(
            align_chain_residues(source, target).aligned_source_rows.numel()
        )
    diagnostics: dict[str, bool | str | float | int] = {
        "accepted": False,
        "fallback": "identity",
        "reason": reason,
        "identity_rmsd": 0.0,
        "candidate_rmsd": 0.0,
        "matched_coverage": coverage,
    }
    return mapping, diagnostics


def _alignment_token_center_atom_indices(
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "physppi_token_center_atom_index" in batch:
        center_index = batch["physppi_token_center_atom_index"].long()
        return center_index, torch.ones_like(center_index, dtype=torch.bool)

    center_index, token_atom_mask = get_token_center_atom_indices(batch)
    if "num_atoms_per_token" not in batch:
        return center_index.long(), token_atom_mask.bool()

    single_atom_token_mask = batch["num_atoms_per_token"].long() == 1
    center_index = torch.where(
        single_atom_token_mask,
        batch["start_atom_index"].long(),
        center_index.long(),
    )
    token_atom_mask = torch.where(
        single_atom_token_mask,
        torch.ones_like(token_atom_mask, dtype=torch.bool),
        token_atom_mask.bool(),
    )
    return center_index, token_atom_mask


def _get_valid_token_center_atoms(
    batch: dict,
    positions: torch.Tensor,
    atom_mask: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract token centers after dropping padded token rows."""
    center_index, token_atom_mask = _alignment_token_center_atom_indices(batch)
    center_index = center_index[token_mask].long()
    token_atom_mask = token_atom_mask[token_mask].bool()
    if torch.any((center_index < 0) | (center_index >= positions.shape[-2])):
        raise ProtectedChainMatchError("invalid_token_center_index")
    center_positions = positions[center_index]
    center_mask = atom_mask[center_index].bool() & token_atom_mask
    center_mask &= torch.isfinite(center_positions).all(dim=-1)
    return center_positions, center_mask


def single_batch_multi_chain_permutation_alignment(
    single_batch: dict,
    single_predicted_positions: torch.Tensor,
    *,
    chain_assignment: str = "protected",
    fallback_reason: str = "",
) -> torch.Tensor:
    """Runs the whole permutation alignment algorithm for features of a single batch.

    This function takes in a single batch of features without a batch dimension, and
    runs through the whole permutation alignment algorithm to find the optimal atom-wise
    subset & reordering of ground-truth atom features to match the predicted atom
    features as well as possible. This follows section 4.2 of the AF3 SI, with some
    slight deviations documented in corresponding functions.

    Args:
        single_batch (dict):
            A dictionary containing the features of a single batch. The dictionary
            should contain the following keys: - ground_truth
                - token_mask: [N_token]
                - atom_mask: [N_atom]
                - atom_positions: [N_atom, 3]
                - atom_resolved_mask: [N_atom]
                - mol_entity_id: [N_token]
                - mol_sym_id: [N_token]
                - mol_sym_component_id: [N_token]
                - mol_sym_token_index: [N_token]
                - num_atoms_per_token: [N_token]
                - is_ligand: [N_token]
            - token_mask: [N_token]
            - atom_mask: [N_atom]
            - mol_entity_id: [N_token]
            - mol_sym_id: [N_token]
            - mol_sym_component_id: [N_token]
            - mol_sym_token_index: [N_token]
            - num_atoms_per_token: [N_token]
            - ref_space_uid: [N_atom]
            - ref_space_uid_to_perm:
                dict[int, [N_permutations, N_conf_atoms]]

        single_predicted_positions (torch.Tensor):
            [N_atom_pred, 3] the predicted atom positions.

    Returns:
        torch.Tensor:
            An index tensor of shape [N_atom_pred] that subselects and reorders the
            ground-truth atom features to match the predicted atom features as well as
            possible.
    """
    # TODO: Note down what assumptions this function makes, especially with respect to
    # ordering of the mols

    gt_batch = single_batch["ground_truth"]

    # Get relevant features and get rid of padding
    gt_token_pad_mask = gt_batch["token_mask"].bool()
    gt_atom_pad_mask = gt_batch["atom_mask"].bool()
    pred_token_pad_mask = single_batch["token_mask"].bool()
    pred_atom_pad_mask = single_batch["atom_mask"].bool()

    gt_coords = gt_batch["atom_positions"][gt_atom_pad_mask]
    gt_resolved_mask = gt_batch["atom_resolved_mask"][gt_atom_pad_mask].bool()
    pred_coords = single_predicted_positions[pred_atom_pad_mask]

    # Subset coordinates to only token centers (generalization of C-alpha in
    # AF2-Multimer algorithm)
    gt_token_center_coords, gt_token_center_mask = _get_valid_token_center_atoms(
        gt_batch,
        gt_batch["atom_positions"],
        gt_batch["atom_resolved_mask"],
        gt_token_pad_mask,
    )

    # Here everything is "resolved" because the model predicts all coordinates so no
    # need to pass an actual mask
    pred_coords_dummy_mask = torch.ones_like(
        single_predicted_positions[:, 0], dtype=torch.bool
    )
    pred_token_center_coords, _ = _get_valid_token_center_atoms(
        single_batch,
        single_predicted_positions,
        pred_coords_dummy_mask,
        pred_token_pad_mask,
    )

    # ASSIGNMENT STAGE
    # PhysPPI modification (Jinyu Lin, ShanghaiTech):
    #   token_identity is the preferred strategy for PhysPPI because prediction
    #   and GT share the same token_index from the source structure.  It avoids
    #   the combinatorial Hungarian matching that protected chain matching does.
    if chain_assignment == "token_identity":
        gt_token_index = gt_batch["token_index"][gt_token_pad_mask]
        pred_token_index = single_batch["token_index"][pred_token_pad_mask]
        gt_asym_id = (
            gt_batch["asym_id"][gt_token_pad_mask] if "asym_id" in gt_batch else None
        )
        pred_asym_id = (
            single_batch["asym_id"][pred_token_pad_mask]
            if "asym_id" in single_batch
            else None
        )
        gt_token_subset_index = get_token_identity_gt_token_subset_index(
            gt_token_index=gt_token_index,
            pred_token_index=pred_token_index,
            gt_asym_id=gt_asym_id,
            pred_asym_id=pred_asym_id,
        )
        token_identity_target = gt_token_center_coords[gt_token_subset_index]
        token_identity_mask = gt_token_center_mask[gt_token_subset_index].bool()
        token_identity_mask &= torch.isfinite(token_identity_target).all(dim=-1)
        token_identity_mask &= torch.isfinite(pred_token_center_coords).all(dim=-1)
        if not torch.any(token_identity_mask):
            raise ProtectedChainMatchError("token_identity_no_resolved_token_centers")

        pred_to_gt_transformation = get_optimal_transformation(
            mobile_positions=pred_token_center_coords,
            target_positions=token_identity_target,
            positions_mask=token_identity_mask,
        )
        token_identity_prediction = apply_transformation(
            pred_token_center_coords,
            pred_to_gt_transformation,
        )
        token_identity_rmsd = torch.sqrt(
            (
                token_identity_prediction[token_identity_mask]
                - token_identity_target[token_identity_mask]
            )
            .square()
            .sum(dim=-1)
            .mean()
        )
        diagnostics = {
            "accepted": True,
            "fallback": "none",
            "reason": "token_index_identity",
            "identity_rmsd": float(token_identity_rmsd.item()),
            "candidate_rmsd": float(token_identity_rmsd.item()),
            "matched_coverage": int(token_identity_mask.sum().item()),
        }
        single_batch["physppi_chain_match_diagnostic"] = diagnostics

        pred_coords_aligned = apply_transformation(
            pred_coords,
            pred_to_gt_transformation,
        )
        return get_token_identity_atom_permutation_index(
            gt_positions=gt_coords,
            gt_resolved_mask=gt_resolved_mask,
            gt_token_index=gt_token_index,
            gt_num_atoms_per_token=gt_batch["num_atoms_per_token"][gt_token_pad_mask],
            gt_asym_id=gt_asym_id,
            pred_positions_aligned=pred_coords_aligned,
            pred_token_index=pred_token_index,
            pred_num_atoms_per_token=single_batch["num_atoms_per_token"][
                pred_token_pad_mask
            ],
            pred_asym_id=pred_asym_id,
            pred_ref_space_uid=single_batch["ref_space_uid"][pred_atom_pad_mask],
            pred_ref_space_uid_to_perm=single_batch["ref_space_uid_to_perm"],
        )

    gt_mol_entity_id = gt_batch["mol_entity_id"][gt_token_pad_mask]
    gt_mol_sym_id = gt_batch["mol_sym_id"][gt_token_pad_mask]
    gt_mol_sym_component_id = gt_batch["mol_sym_component_id"][
        gt_token_pad_mask
    ]  # TODO: change this to atom-wise?
    gt_mol_sym_token_index = gt_batch["mol_sym_token_index"][gt_token_pad_mask]
    gt_num_atoms_per_token = gt_batch["num_atoms_per_token"][gt_token_pad_mask]
    pred_mol_entity_id = single_batch["mol_entity_id"][pred_token_pad_mask]
    pred_mol_sym_id = single_batch["mol_sym_id"][pred_token_pad_mask]
    pred_mol_sym_component_id = single_batch["mol_sym_component_id"][
        pred_token_pad_mask
    ]
    pred_mol_sym_token_index = single_batch["mol_sym_token_index"][pred_token_pad_mask]
    pred_ref_space_uid = single_batch["ref_space_uid"][pred_atom_pad_mask]
    pred_ref_space_uid_to_perm = single_batch["ref_space_uid_to_perm"]
    pred_num_atoms_per_token = single_batch["num_atoms_per_token"][pred_token_pad_mask]

    # ASSIGNMENT STAGE
    # Protect molecule-level assignment while retaining OF3's atom-level symmetry
    # resolution below.
    if chain_assignment == "protected":
        optimal_mol_permutation, diagnostics = find_protected_mol_permutation(
            pred_features=single_batch,
            pred_atom_positions=single_predicted_positions,
            pred_atom_mask=pred_atom_pad_mask,
            gt_features=gt_batch,
            gt_atom_positions=gt_batch["atom_positions"],
            gt_atom_mask=gt_batch["atom_resolved_mask"].bool(),
        )
    elif chain_assignment == "identity":
        optimal_mol_permutation, diagnostics = (
            find_sequence_compatible_identity_mol_permutation(
                pred_features=single_batch,
                pred_atom_positions=single_predicted_positions,
                pred_atom_mask=pred_atom_pad_mask,
                gt_features=gt_batch,
                gt_atom_positions=gt_batch["atom_positions"],
                gt_atom_mask=gt_batch["atom_resolved_mask"].bool(),
                reason=fallback_reason,
            )
        )
    else:
        raise ValueError(f"Unknown chain assignment mode: {chain_assignment}")
    single_batch["physppi_chain_match_diagnostic"] = diagnostics

    # NOTE: Maybe alignments like these should be restricted to non-ligand token
    # centers?
    # Get alignment of predicted token-center coordinates to the ground-truth
    # token-center coordinates before resolving the atom-level symmetry permutations
    pred_to_gt_transformation = get_pred_to_permuted_gt_transformation(
        gt_token_center_coords,
        gt_token_center_mask,
        gt_mol_entity_id,
        gt_mol_sym_id,
        gt_mol_sym_token_index,
        pred_token_center_coords,
        pred_mol_entity_id,
        pred_mol_sym_id,
        pred_mol_sym_token_index,
        optimal_mol_permutation,
    )

    if chain_assignment == "identity":
        gt_token_subset_index = get_permuted_gt_token_subset_index(
            gt_mol_entity_ids=gt_mol_entity_id,
            gt_mol_sym_ids=gt_mol_sym_id,
            gt_mol_sym_token_index=gt_mol_sym_token_index,
            pred_mol_entity_ids=pred_mol_entity_id,
            pred_mol_sym_ids=pred_mol_sym_id,
            pred_mol_sym_token_index=pred_mol_sym_token_index,
            optimal_permutation=optimal_mol_permutation,
        )
        identity_target = gt_token_center_coords[gt_token_subset_index]
        identity_mask = gt_token_center_mask[gt_token_subset_index].bool()
        identity_prediction = apply_transformation(
            pred_token_center_coords,
            pred_to_gt_transformation,
        )
        identity_mask &= torch.isfinite(identity_target).all(dim=-1)
        identity_mask &= torch.isfinite(identity_prediction).all(dim=-1)
        if identity_mask.any():
            identity_rmsd = torch.sqrt(
                (identity_prediction[identity_mask] - identity_target[identity_mask])
                .square()
                .sum(dim=-1)
                .mean()
            )
            diagnostics["identity_rmsd"] = float(identity_rmsd.item())
            diagnostics["candidate_rmsd"] = float(identity_rmsd.item())
            diagnostics["matched_coverage"] = int(identity_mask.sum().item())

    pred_coords_aligned = apply_transformation(pred_coords, pred_to_gt_transformation)

    # Get the final atom index permutation that applies the optimal permutation found
    # earlier, and simultaneously resolves symmetry-equivalent atoms, resulting in a
    # final atom-wise rearrangement of the entire ground-truth feature tensor
    gt_atom_index = get_final_atom_permutation_index(
        gt_positions=gt_coords,
        gt_resolved_mask=gt_resolved_mask,
        gt_mol_entity_id=gt_mol_entity_id,
        gt_mol_sym_id=gt_mol_sym_id,
        gt_mol_sym_component_id=gt_mol_sym_component_id,
        gt_num_atoms_per_token=gt_num_atoms_per_token,
        pred_positions_aligned=pred_coords_aligned,
        pred_mol_entity_id=pred_mol_entity_id,
        pred_mol_sym_id=pred_mol_sym_id,
        pred_mol_sym_component_id=pred_mol_sym_component_id,
        pred_num_atoms_per_token=pred_num_atoms_per_token,
        pred_ref_space_uid=pred_ref_space_uid,
        pred_ref_space_uid_to_perm=pred_ref_space_uid_to_perm,
        optimal_mol_permutation=optimal_mol_permutation,
    )

    return gt_atom_index


# TODO: Add docstring
def multi_chain_permutation_alignment(
    batch: dict,
    atom_positions_predicted: torch.tensor,
    *,
    chain_assignment: str = "protected",
    fallback_reason: str = "",
):
    batch_size = batch["residue_index"].shape[0]

    # This will store the final per-batch subsetting and reordering of the ground-truth
    # features
    gt_atom_indexes = []
    diagnostics = []

    # Currently has to be run sequentially per batch, so split features batch-wise
    for i in range(batch_size):
        single_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                single_batch[key] = value[i]
            elif key == "ref_space_uid_to_perm":
                # This is a special case that provides a separate dict per batch
                single_batch[key] = value[i]
            elif isinstance(value, dict):
                single_batch[key] = {k: v[i] for k, v in value.items()}
            else:
                # Ignore other types
                pass

        single_predicted_positions = atom_positions_predicted[i]

        # Get the permutation indices for the current batch that subset and reorder the
        # ground-truth features to match the prediction
        gt_atom_index = single_batch_multi_chain_permutation_alignment(
            single_batch,
            single_predicted_positions,
            chain_assignment=chain_assignment,
            fallback_reason=fallback_reason,
        )
        gt_atom_indexes.append(gt_atom_index)
        diagnostics.append(single_batch["physppi_chain_match_diagnostic"])

    # Apply the permutation to the ground-truth position and unresolved mask features
    ground_truth_features = update_gt_position_features(batch, gt_atom_indexes)

    # Sanity-check final results (will raise an error if the shapes do not match)
    verify_shape_match(ground_truth_features, atom_positions_predicted)

    batch["physppi_chain_match_diagnostics"] = diagnostics
    return ground_truth_features


def naive_alignment(batch: dict, atom_positions_predicted: torch.tensor) -> dict:
    """Fallback that directly matches GT and predicted token IDs.

    Matches ground-truth features to predicted features by directly matching the token
    IDs. This therefore takes the arbitrary input permutation and does not attempt to
    resolve any intra-residue symmetric atom permutations.

    Args:
        batch (dict):
            The initial feature dictionary containing a mini-batch of features.
        atom_positions_predicted (torch.tensor):
            The predicted atom positions. These are only required to verify the final
            result shapes.
    Returns:
        dict:
            The updated feature dictionary with the ground-truth "atom_positions" and
            "atom_resolved_mask" features rearranged to match the prediction.
    """
    pred_token_ids = batch["token_index"]
    gt_token_ids = batch["ground_truth"]["token_index"]
    pred_token_mask = batch["token_mask"].bool()
    gt_token_mask = batch["ground_truth"]["token_mask"].bool()

    batch_size = pred_token_ids.shape[0]

    gt_atom_indices = []

    # Match the GT atoms directly to the prediction by using the original token ID,
    # disregarding any potential permutation resolving
    for batch_id in range(batch_size):
        pred_token_ids_batch = pred_token_ids[batch_id][pred_token_mask[batch_id]]
        gt_token_ids_batch = gt_token_ids[batch_id][gt_token_mask[batch_id]]

        # Sorted token IDs are required for the below logic to work
        if not pred_token_ids_batch.diff().gt(0).all():
            raise ValueError("'token_index' is not sorted.")
        if not gt_token_ids_batch.diff().gt(0).all():
            raise ValueError("'ground_truth.token_index' is not sorted.")

        gt_token_ids_atomized_batch = broadcast_token_feat_to_atoms(
            token_mask=gt_token_mask[batch_id],
            num_atoms_per_token=batch["ground_truth"]["num_atoms_per_token"][batch_id],
            token_feat=gt_token_ids[batch_id],
        )

        atom_selection_mask = torch.isin(
            gt_token_ids_atomized_batch, pred_token_ids_batch
        )

        gt_atom_indices.append(torch.nonzero(atom_selection_mask, as_tuple=True)[0])

    ground_truth_features = update_gt_position_features(batch, gt_atom_indices)

    # Sanity-check final results (will raise an error if the shapes do not match)
    verify_shape_match(ground_truth_features, atom_positions_predicted)

    return ground_truth_features


# TODO: Improve for larger batch sizes (where simple shape check only checks longest
# batch basically)
def verify_shape_match(
    ground_truth_feats: dict, atom_positions_predicted: torch.tensor
) -> None:
    """Verifies that the final ground-truth feature match the predicted feature shapes.

    Args:
        ground_truth_feats (dict):
            The final ground-truth features after the permutation alignment. Requires
            the keys "atom_positions" and "atom_resolved_mask".
        atom_positions_predicted (torch.tensor):
            The predicted atom positions.
    """
    permuted_gt_coords = ground_truth_feats["atom_positions"]
    permuted_gt_mask = ground_truth_feats["atom_resolved_mask"]

    if not permuted_gt_coords.shape == atom_positions_predicted.shape:
        raise ValueError(
            f"Shape mismatch between permuted ground-truth and predicted coordinates: "
            f"{permuted_gt_coords.shape} vs. {atom_positions_predicted.shape}"
        )

    if not permuted_gt_mask.shape == atom_positions_predicted.shape[:-1]:
        raise ValueError(
            "Shape mismatch between permuted ground-truth mask and predicted "
            f"coordinates: {permuted_gt_mask.shape} vs. "
            f"{atom_positions_predicted.shape}"
        )


# TODO: Update later, temporary fix to handle more than one sample
#  from the rollout output
def reshape_per_sample_inputs(
    batch: dict, atom_positions_predicted: torch.tensor
) -> tuple[dict, torch.tensor]:
    """
    Reshapes the input features and predicted positions so that the features are first
    repeated to match the number of samples in the predicted positions, and then
    flattened so that there is a single batch dimension of size batch_size * n_samples.

    Args:
        batch (dict):
            The initial feature dictionary containing a mini-batch of features.
            The features input to this function will have initial dimensions
            [batch_size, 1] since the sample dimension was already added before
            the rollout.
        atom_positions_predicted (torch.tensor):
            [batch_size, n_samples, n_atom, 3] The predicted atom positions from
            the rollout.

    Returns:
        per_sample_batch (dict):
            The feature dictionary with flattened batch and sample dimensions.
            The features will have starting dimension [batch_size * n_samples, ...].
        per_sample_atom_pos_pred (torch.tensor):
            [batch_size * n_samples, n_atom, 3] The predicted atom positions from
            the rollout, with the batch and sample dimension flattened.
    """
    atom_pos_pred_shape = atom_positions_predicted.shape

    # No sample dimension, return as is
    if len(atom_pos_pred_shape[:-2]) == 1:
        return batch, atom_positions_predicted

    batch_size, no_samples = atom_pos_pred_shape[:2]

    def reshape_batch_feats(t: torch.tensor):
        """Expands the batch dimension to match the number of samples in the output."""
        feat_dims = t.shape[2:]
        if t.numel() == 0:
            return t.expand(batch_size, no_samples, *feat_dims).reshape(
                batch_size * no_samples, *feat_dims
            )
        t = t.expand(-1, no_samples, *((-1,) * len(feat_dims)))
        return t.reshape(-1, *feat_dims)

    # Alignment consumes static data features only.  Keep differentiable model
    # predictions on the original batch, where rollout/diffusion/loss can
    # retain their autograd graph; PyTorch cannot deepcopy non-leaf tensors.
    alignment_batch = dict(batch)
    ref_space_uid_to_perm = alignment_batch.pop("ref_space_uid_to_perm", None)
    alignment_batch.pop("physppi_predicted_plane", None)
    per_sample_batch = tensor_tree_map(reshape_batch_feats, deepcopy(alignment_batch))

    if ref_space_uid_to_perm is not None:
        ref_space_uid_to_perm_flat = []
        for i in range(batch_size):
            ref_space_uid_to_perm_flat.extend(
                [ref_space_uid_to_perm[i] for _ in range(no_samples)]
            )
        per_sample_batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm_flat

    per_sample_atom_pos_pred = atom_positions_predicted.reshape(
        -1, *atom_pos_pred_shape[2:]
    )

    return per_sample_batch, per_sample_atom_pos_pred


# TODO: Update later, temporary fix to handle more than one sample
#  from the rollout output
def format_output_gt_features(ground_truth_features: dict, batch_dims: tuple) -> dict:
    """
    Unflatten the batch and sample dimensions of the output ground truth
    features to match the input.

    Args:
        ground_truth_features (dict):
            The final ground-truth features after the permutation alignment. Requires
            the keys "atom_positions" and "atom_resolved_mask".
        batch_dims (list):
            The batch dimensions of the input features, including the sample dimension.
    Returns:
        reshaped_gt_features (dict):
            The feature dictionary with the batch and sample dimensions reshaped to
            match the input features.
    """
    batch_size = batch_dims[0]
    has_sample_dim = len(batch_dims) > 1

    if has_sample_dim:
        ground_truth_features = {
            feature: value.reshape(batch_size, -1, *value.shape[1:])
            for feature, value in ground_truth_features.items()
        }

    return ground_truth_features


def safe_multi_chain_permutation_alignment(
    batch: dict, atom_positions_predicted: torch.tensor
) -> None:
    """Runs the multi-chain permutation alignment algorithm with error handling.

    This function attempts to run protected multi-chain assignment and then retains
    OpenFold3's atom-level within-residue symmetry alignment.

    Only typed protected-matcher failures are caught. They use a deterministic,
    sequence-compatible identity mapping without mutating configured loss weights.

    For more details, see the documentation of the `multi_chain_permutation_alignment`
    and `single_batch_multi_chain_permutation_alignment` functions.

    Args:
        batch (dict):
            The initial feature dictionary containing a mini-batch of features.
        atom_positions_predicted (torch.tensor):
            The predicted atom positions from the rollout.

    Returns:
        None, the batch dictionary is modified in-place(!)
    """
    batch_dims = atom_positions_predicted.shape[:-2]
    per_sample_batch, per_sample_atom_pos_pred = reshape_per_sample_inputs(
        batch=batch, atom_positions_predicted=atom_positions_predicted
    )

    center_index, _ = _alignment_token_center_atom_indices(per_sample_batch)
    invalid_center_index = (center_index < 0) | (
        center_index >= per_sample_atom_pos_pred.shape[-2]
    )
    invalid_center_index &= per_sample_batch["token_mask"].bool()
    if torch.any(invalid_center_index):
        pdb_ids = batch.get("pdb_id", ("<missing>",))
        logger.error(
            "Invalid token-center index for pdb_id=%s.",
            ", ".join(map(str, pdb_ids)),
        )
        raise ProtectedChainMatchError(
            "invalid_token_center_index:" + ",".join(map(str, pdb_ids))
        )

    # PhysPPI modification (Jinyu Lin, ShanghaiTech):
    #   Fallback chain changed from:  protected → identity
    #   to:  token_identity → protected → identity
    #   token_identity is preferred for PhysPPI because pred and GT share
    #   the same token_index; it is simpler, faster, and more reliable.
    try:
        new_gt_features = multi_chain_permutation_alignment(
            batch=per_sample_batch,
            atom_positions_predicted=per_sample_atom_pos_pred,
            chain_assignment="token_identity",
        )
    except ProtectedChainMatchError as token_identity_error:
        pdb_ids = batch.get("pdb_id", ("<missing>",))
        logger.warning(
            "Token-index identity alignment failed for pdb_id=%s; trying protected "
            "chain matching: %s",
            ", ".join(map(str, pdb_ids)),
            token_identity_error,
        )
        try:
            new_gt_features = multi_chain_permutation_alignment(
                batch=per_sample_batch,
                atom_positions_predicted=per_sample_atom_pos_pred,
                chain_assignment="protected",
            )
        except ProtectedChainMatchError as protected_error:
            logger.error(
                "Protected chain matching failed for pdb_id=%s; using deterministic "
                "sequence-compatible identity mapping: %s",
                ", ".join(map(str, pdb_ids)),
                protected_error,
            )
            new_gt_features = multi_chain_permutation_alignment(
                batch=per_sample_batch,
                atom_positions_predicted=per_sample_atom_pos_pred,
                chain_assignment="identity",
                fallback_reason=str(protected_error),
            )

    new_gt_features = format_output_gt_features(
        ground_truth_features=new_gt_features,
        batch_dims=batch_dims,
    )
    batch["ground_truth"].update(new_gt_features)
    batch["physppi_chain_match_diagnostics"] = per_sample_batch[
        "physppi_chain_match_diagnostics"
    ]
