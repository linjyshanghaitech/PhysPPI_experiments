#!/usr/bin/env python3
"""Adapter: build residue-level prior tensors from pre-computed .pt files.

Drop-in replacements for:
    build_contact_features()       → build_contact_from_precomputed()
    build_interaction_features()   → build_interaction_from_precomputed()
    build_tm_features()            → build_tm_from_precomputed()
    build_membrane_target()        → build_membrane_target_from_precomputed()

All three load pre-computed .pt files that store data in PDB resseq space.
The adapter maps to token_index at call time, which naturally handles
crops — cropped-out residues return None from token_index.get() and are
silently skipped.

Usage in _runtime.py:

    from experiments.residue_prior.residue_adapter import (
        build_contact_from_precomputed,
        build_interaction_from_precomputed,
        build_tm_from_precomputed,
        build_membrane_target_from_precomputed,
    )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Interface residues (dmap → interface_mask [N])
# ---------------------------------------------------------------------------


def build_interface_residue_from_precomputed(
    pt_path: Path,
    token_index: Mapping[tuple[str, int], int],
    cutoff: float,
) -> dict[str, Any]:
    """Build [N] interface-residue mask from _interface_residues.pt.

    Drop-in for build_interface_residue_features(sample.dmap, token_index,
    cutoff).
    Distance-based cutoff augmentation is applied here — dmap stores raw
    CA-CA distances; residues with at least one cross-chain pair ≤ cutoff are
    marked as interface residues.
    """
    n_tokens = _num_tokens(token_index)
    interface_mask = np.zeros((n_tokens,), dtype=bool)

    try:
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return {
            "interface_mask": interface_mask,
            "available": False,
            "valid": False,
            "cutoff": cutoff,
        }

    dmap = pt.get("dmap", {})
    if not isinstance(dmap, dict):
        return {
            "interface_mask": interface_mask,
            "available": False,
            "valid": False,
            "cutoff": cutoff,
        }

    for chain_pair, edges in dmap.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        left_ch, right_ch = chain_pair.split("|", 1)
        edges_np = edges.numpy() if torch.is_tensor(edges) else np.asarray(edges)
        for row in edges_np:
            r_a, r_b, dist = int(row[0]), int(row[1]), float(row[2])
            if dist > cutoff:
                continue
            ta = token_index.get((left_ch, r_a))
            tb = token_index.get((right_ch, r_b))
            if ta is None or tb is None:
                continue
            ta, tb = int(ta), int(tb)
            if ta >= n_tokens or tb >= n_tokens or ta == tb:
                continue
            interface_mask[ta] = True
            interface_mask[tb] = True

    return {
        "interface_mask": interface_mask,
        "available": True,
        "valid": bool(interface_mask.any()),
        "cutoff": cutoff,
    }


# ---------------------------------------------------------------------------
# Interaction pairs (interactions → type_mask [N, N, 5] + aux)
# ---------------------------------------------------------------------------


def build_interaction_from_precomputed(
    pt_path: Path,
    token_index: Mapping[tuple[str, int], int],
    *,
    atom_array: object | None = None,
) -> dict[str, Any]:
    """Build [N, N, 5] interaction type mask from _interaction_pairs.pt.

    Drop-in for build_interaction_features(sample.interactions, token_index,
    atom_array=...).

    type_mask is always built from the precomputed pairs (model input).
    aux entries (for distance loss) are populated only when atom_array is
    provided — atom names in the .pt are resolved against GT PDB coordinates
    and centroid distances are computed on the fly.
    """
    n_tokens = _num_tokens(token_index)
    type_mask = np.zeros((n_tokens, n_tokens, 5), dtype=bool)
    aux: list[tuple[int, int, int, float, tuple[str, ...], tuple[str, ...]]] = []

    try:
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return {
            "type_mask": type_mask,
            "available": False,
            "valid": False,
            "aux": (),
        }

    interactions = pt.get("interactions", {})
    if not isinstance(interactions, dict):
        return {
            "type_mask": type_mask,
            "available": False,
            "valid": False,
            "aux": (),
        }

    atom_lookup = (
        _interaction_atom_lookup(atom_array) if atom_array is not None else None
    )

    for chain_pair, data in interactions.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        left_ch, right_ch = chain_pair.split("|", 1)
        pairs = data.get("pairs")
        a1_list = data.get("atom_1", [])
        a2_list = data.get("atom_2", [])

        if pairs is None:
            continue
        pairs_np = pairs.numpy() if torch.is_tensor(pairs) else np.asarray(pairs)
        for i, (r_a, r_b, type_id) in enumerate(pairs_np):
            r_a, r_b, type_id = int(r_a), int(r_b), int(type_id)
            ta = token_index.get((left_ch, r_a))
            tb = token_index.get((right_ch, r_b))
            if ta is None or tb is None:
                continue
            ta, tb = int(ta), int(tb)
            if ta >= n_tokens or tb >= n_tokens or ta == tb:
                continue
            type_mask[ta, tb, type_id] = True
            type_mask[tb, ta, type_id] = True

            # Compute distance from GT PDB atom coordinates (loss only)
            distance: float | None = None
            a1 = tuple(a1_list[i]) if i < len(a1_list) else ()
            a2 = tuple(a2_list[i]) if i < len(a2_list) else ()
            if atom_lookup is not None and a1 and a2:
                left_centroid = _interaction_group_centroid(
                    atom_lookup, left_ch, r_a, a1
                )
                right_centroid = _interaction_group_centroid(
                    atom_lookup, right_ch, r_b, a2
                )
                if left_centroid is not None and right_centroid is not None:
                    candidate = float(np.linalg.norm(left_centroid - right_centroid))
                    if np.isfinite(candidate):
                        distance = candidate
            if distance is not None:
                aux.append((ta, tb, type_id, distance, a1, a2))

    return {
        "type_mask": type_mask,
        "available": True,
        "valid": bool(type_mask.any()),
        "aux": tuple(aux),
    }


# ---------------------------------------------------------------------------
# Membrane relative location (OPM → tm_mask [N] + plane)
# ---------------------------------------------------------------------------


def build_tm_from_precomputed(
    pt_path: Path,
    token_index: Mapping[tuple[str, int], int],
) -> np.ndarray:
    """Build [N] boolean TM mask from _membrane_relative_location.pt.

    Drop-in for build_tm_features(res_loc, token_index).
    Only side==0 (transmembrane) residues are marked.
    """
    n_tokens = _num_tokens(token_index)
    tm_mask = np.zeros(n_tokens, dtype=bool)

    try:
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return tm_mask

    res_loc = pt.get("res_loc", {})
    if not isinstance(res_loc, dict):
        return tm_mask

    for chain_id, intervals in res_loc.items():
        intervals_np = (
            intervals.numpy() if torch.is_tensor(intervals) else np.asarray(intervals)
        )
        for start, end, side in intervals_np:
            if int(side) != 0:
                continue
            for r in range(int(start), int(end) + 1):
                t = token_index.get((str(chain_id), r))
                if t is not None and int(t) < n_tokens:
                    tm_mask[int(t)] = True

    return tm_mask


def build_membrane_target_from_precomputed(
    pt_path: Path,
) -> dict[str, Any]:
    """Build membrane plane supervision target from _membrane_relative_location.pt.

    Drop-in for build_membrane_target(opm, is_membrane).
    """
    zero = np.zeros(3, dtype=np.float32)
    try:
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return {"valid": False, "normal": zero, "point": zero}

    plane = pt.get("plane", {})
    if not isinstance(plane, dict):
        return {"valid": False, "normal": zero, "point": zero}

    normal = np.asarray(plane.get("normal", zero), dtype=np.float32)
    point = np.asarray(plane.get("centroid", zero), dtype=np.float32)

    if normal.shape != (3,) or point.shape != (3,):
        return {"valid": False, "normal": zero, "point": zero}
    if not np.isfinite(normal).all() or not np.isfinite(point).all():
        return {"valid": False, "normal": zero, "point": zero}

    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-6:
        return {"valid": False, "normal": zero, "point": zero}

    return {"valid": True, "normal": normal / norm, "point": point}


# ---------------------------------------------------------------------------
# Internal helpers (mirror _builders.py)
# ---------------------------------------------------------------------------


def _num_tokens(token_index: Mapping[tuple[str, int], int]) -> int:
    if not token_index:
        return 0
    indices = [int(v) for v in token_index.values() if int(v) >= 0]
    return max(indices, default=-1) + 1


def _interaction_atom_lookup(
    atom_array: object,
) -> dict[tuple[str, int, str], np.ndarray]:
    """Index finite structure coordinates by (chain, resseq, atom_name)."""
    try:
        chains = np.asarray(atom_array.chain_id)
        residues = np.asarray(atom_array.res_id)
        names = np.asarray(atom_array.atom_name)
        coords = np.asarray(atom_array.coord, dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        return {}

    if (
        coords.ndim != 2
        or coords.shape[1] != 3
        or not (chains.shape == residues.shape == names.shape == coords.shape[:1])
    ):
        return {}

    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    for chain, residue, name, coord in zip(chains, residues, names, coords):
        try:
            key_chain = str(chain)
            key_res = int(residue)
        except (TypeError, ValueError, OverflowError):
            continue
        atom_name = str(name).strip().upper()
        if not atom_name or not np.isfinite(coord).all():
            continue
        lookup.setdefault((key_chain, key_res, atom_name), coord.astype(np.float32))
    return lookup


def _interaction_group_centroid(
    lookup: dict[tuple[str, int, str], np.ndarray],
    chain: str,
    residue: int,
    group: Sequence[str],
) -> np.ndarray | None:
    """Compute centroid of an atom group for one residue."""
    coords = []
    for name in group:
        coord = lookup.get((chain, residue, name))
        if coord is None or coord.shape != (3,) or not np.isfinite(coord).all():
            return None
        coords.append(coord)
    if not coords:
        return None
    centroid = np.mean(np.stack(coords, axis=0), axis=0)
    return centroid.astype(np.float32) if np.isfinite(centroid).all() else None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if pt_path is None or not pt_path.exists():
        print("Usage: python residue_adapter.py <path/to/case_id_*.pt>")
        sys.exit(1)

    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    for key, value in pt.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sk, sv in value.items():
                if torch.is_tensor(sv):
                    print(f"  {sk}: shape={sv.shape}, dtype={sv.dtype}")
                elif isinstance(sv, list):
                    print(f"  {sk}: list of {len(sv)} items")
                else:
                    print(f"  {sk}: {sv}")
        else:
            print(f"{key}: {type(value).__name__}")
