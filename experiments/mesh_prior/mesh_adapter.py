#!/usr/bin/env python3
"""Adapter: build [N, K, 3/6] surface tensors from pre-computed mesh PKL files.

Drop-in replacement for `build_surface_features()` in
`openfold3/projects/of3_all_atom/physppi/features/_builders.py`.

The pre-computed PKL contains per-chain, per-residue mesh vertex arrays stored in
GLOBAL complex coordinates (from GT PDB).  At load time we apply the per-residue
frame transform to produce LOCAL coordinates matching the encoder input format.

Usage in `_runtime.py` `build_prior_features()`:

    from experiments.mesh_prior.mesh_adapter import build_surface_from_precomputed
    surface = build_surface_from_precomputed(
        mesh_pkl_path=f"precomputed/{case_id}_mesh.pkl",
        token_index=token_index,
        atom_array=atom_array,
        monomer_transforms=monomer_transforms,
        k=64,
    )

Requires:
    numpy, torch
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def build_surface_from_precomputed(
    *,
    mesh_pt_path: Path | None,
    token_index: Mapping[tuple[str, int], int],
    atom_array: object | None,
    monomer_transforms: Mapping[str, tuple[np.ndarray, np.ndarray]],
    k: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Build surface tensors from pre-computed KDTree mesh-to-residue mapping.

    Returns a dict matching the `build_surface_features()` contract:
        physppi_surface_local_coords:  torch.Tensor  [N, K, 3]
        physppi_surface_features:      torch.Tensor  [N, K, 6]
        physppi_surface_mask:          torch.Tensor  [N, K]   bool
        physppi_surface_charge_mask:   torch.Tensor  [N, K]   bool
    plus `available`, `valid` flags.
    """
    n_tokens = _num_tokens(token_index)
    coords = torch.zeros(n_tokens, k, 3, device=device)
    features = torch.zeros(n_tokens, k, 6, device=device)
    mask = torch.zeros(n_tokens, k, dtype=torch.bool, device=device)
    charge_mask = torch.zeros(n_tokens, k, dtype=torch.bool, device=device)

    # --- Load pre-computed mapping -------------------------------------------
    # Format: {"data": {chain_id: tensor[N, K, 9]}, "resseq": {chain_id: [r1,...]}}
    mesh_data: dict[str, torch.Tensor] = {}
    mesh_resseq: dict[str, list[int]] = {}
    if mesh_pt_path is not None and mesh_pt_path.exists():
        payload = torch.load(mesh_pt_path, map_location="cpu", weights_only=False)
        mesh_data = payload.get("data", {})
        mesh_resseq = payload.get("resseq", {})

    available = len(mesh_data) > 0
    if not available:
        return {
            "physppi_surface_local_coords": coords,
            "physppi_surface_features": features,
            "physppi_surface_mask": mask,
            "physppi_surface_charge_mask": charge_mask,
            "available": False,
            "valid": False,
        }

    # --- Build per-token frames (same as _surface_token_frames) --------------
    frames = _build_token_frames(atom_array, token_index, n_tokens, monomer_transforms)

    # --- Build resseq → row index lookup per chain --------------------------
    resseq_to_idx: dict[str, dict[int, int]] = {}
    for chain_id, rlist in mesh_resseq.items():
        resseq_to_idx[chain_id] = {r: i for i, r in enumerate(rlist)}

    # --- Fill tensors --------------------------------------------------------
    assigned = 0
    for (chain_id, resseq), token_row in token_index.items():
        token_row = int(token_row)
        if token_row < 0 or token_row >= n_tokens:
            continue

        chain_data = mesh_data.get(chain_id)
        if chain_data is None:
            continue
        ridx_map = resseq_to_idx.get(chain_id, {})
        idx = ridx_map.get(resseq)
        if idx is None:
            continue
        verts = chain_data[idx]  # [K, 9]
        m = int((verts.abs().sum(dim=-1) > 0).sum().item())
        if m == 0:
            continue

        frame = frames.get(token_row)
        if frame is None:
            continue

        center, basis = frame  # center: (3,), basis: (3, 3)
        m = min(m, k)
        assigned += 1

        # verts columns: x(0), y(1), z(2), charge(3), hphob(4), nx,ny,nz(5:8), si(8)
        vert_xyz = verts[:m, :3].numpy()  # monomer-local coords
        vert_attrs = verts[:m, 3:].numpy()

        # Step 1: monomer → complex (same as build_surface_features)
        transform = monomer_transforms.get(chain_id)
        if transform is None:
            continue
        rotation = np.asarray(transform[0], dtype=np.float32)
        translation = np.asarray(transform[1], dtype=np.float32)
        vert_xyz = vert_xyz @ rotation + translation

        # Step 2: complex → per-token local frame
        local_xyz = (vert_xyz - center) @ basis
        local_normals = vert_attrs[:, 3:6] @ rotation @ basis
        # attrs cols: charge(0), hphob(1), nx,ny,nz(2:5), si(5)

        is_charge_finite = np.isfinite(vert_attrs[:, 0])

        coords[token_row, :m, :] = torch.from_numpy(local_xyz.astype(np.float32))
        features[token_row, :m, 0] = torch.from_numpy(
            np.where(is_charge_finite, vert_attrs[:, 0], 0.0).astype(np.float32)
        )
        features[token_row, :m, 1] = torch.from_numpy(
            vert_attrs[:, 1].astype(np.float32)
        )
        features[token_row, :m, 2:5] = torch.from_numpy(
            local_normals.astype(np.float32)
        )
        features[token_row, :m, 5] = torch.from_numpy(
            vert_attrs[:, 5].astype(np.float32)
        )
        mask[token_row, :m] = True
        charge_mask[token_row, :m] = torch.from_numpy(is_charge_finite)

    return {
        "physppi_surface_local_coords": coords,
        "physppi_surface_features": features,
        "physppi_surface_mask": mask,
        "physppi_surface_charge_mask": charge_mask,
        "available": available,
        "valid": bool(mask.any()),
    }


# --- Internal helpers --------------------------------------------------------


def _num_tokens(token_index: Mapping[tuple[str, int], int]) -> int:
    if not token_index:
        return 0
    return max(int(v) for v in token_index.values()) + 1


def _build_token_frames(
    atom_array: object | None,
    token_index: Mapping[tuple[str, int], int],
    n_tokens: int,
    monomer_transforms: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute per-token (center, basis) frames from atom_array.

    Replicates `_surface_token_frames()` from _builders.py.
    """
    if atom_array is None:
        return {}

    try:
        chain_ids = np.asarray(atom_array.chain_id)
        residue_ids = np.asarray(atom_array.res_id)
        residue_names = np.asarray(atom_array.res_name)
        atom_names = np.asarray(atom_array.atom_name)
        coords = np.asarray(atom_array.coord, dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        return {}

    if coords.ndim != 2 or coords.shape[1] != 3:
        return {}

    # Collect atoms by residue in global coords
    atoms_by_residue: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    names_by_residue: dict[tuple[str, int], str] = {}

    for chain, residue, res_name, atom_name, coord in zip(
        chain_ids, residue_ids, residue_names, atom_names, coords, strict=False
    ):
        try:
            key = _residue_key(str(chain), int(residue))
        except (TypeError, ValueError, OverflowError):
            continue
        if key is None or not np.isfinite(coord).all():
            continue
        transform = monomer_transforms.get(key[0])
        if transform is None:
            continue
        try:
            rotation = np.asarray(transform[0], dtype=np.float32)
            translation = np.asarray(transform[1], dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if rotation.shape != (3, 3) or translation.shape != (3,):
            continue
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            continue
        coord = coord @ rotation + translation
        atom_key = str(atom_name).strip().upper()
        atoms_by_residue.setdefault(key, {})[atom_key] = coord.astype(np.float32)
        names_by_residue[key] = str(res_name).strip().upper()

    # Backbone atom priorities for frame computation
    BB_PRIORITY = {"CA": 0, "C": 1, "N": 2, "P": 0, "O3'": 1, "C4'": 2}

    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for residue_key, token in token_index.items():
        token = int(token)
        if token < 0 or token >= n_tokens:
            continue
        atoms = atoms_by_residue.get(residue_key, {})
        if not atoms:
            continue

        # Select 3 backbone atoms for frame basis
        selected = _select_frame_atoms(atoms, BB_PRIORITY)
        if selected is None:
            continue

        a_coords = np.stack(list(selected.values()), axis=0)
        center = a_coords.mean(axis=0).astype(np.float32)

        # basis vectors from last atom to first, plus orthogonal
        v1 = (a_coords[0] - center).astype(np.float32)
        v2 = (a_coords[1] - center).astype(np.float32)
        e1 = v1 / (np.linalg.norm(v1) + 1e-8)
        e2 = v2 - np.dot(v2, e1) * e1
        e2_norm = np.linalg.norm(e2)
        if e2_norm < 1e-6:
            continue
        e2 = e2 / e2_norm
        e3 = np.cross(e1, e2)

        basis = np.stack([e1, e2, e3], axis=0).T.astype(np.float32)  # (3, 3)
        frames[token] = (center, basis)

    return frames


def _select_frame_atoms(
    atoms: dict[str, np.ndarray],
    bb_priority: dict[str, int],
) -> dict[str, np.ndarray] | None:
    """Select 3 atoms for building the local surface frame.

    Priority: backbone atoms (CA, C, N) then any other atoms.
    """
    selected: dict[str, np.ndarray] = {}
    for atom_name in sorted(bb_priority, key=lambda n: bb_priority[n]):
        if atom_name in atoms:
            selected[atom_name] = atoms[atom_name]
            if len(selected) == 3:
                return selected

    # Fill remaining with any atoms
    for atom_name, coord in atoms.items():
        if atom_name not in selected:
            selected[atom_name] = coord
            if len(selected) == 3:
                return selected

    if len(selected) >= 2:
        return selected
    return None


def _residue_key(chain: str, resseq: int) -> tuple[str, int] | None:
    if not chain or resseq < 0:
        return None
    return (chain, resseq)


# --- Standalone test ---------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Test with a pre-computed sample
    pkl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if pkl_path is None or not pkl_path.exists():
        print("Usage: python mesh_adapter.py <path/to/case_id_mesh.pkl>")
        print("(requires a GT PDB and token_index to test fully)")
        sys.exit(1)

    mesh_pt = torch.load(pkl_path, map_location="cpu", weights_only=False)
    for chain_id in sorted(mesh_pt):
        residues = mesh_pt[chain_id]
        vert_counts = [v.shape[0] for v in residues.values()]
        print(
            f"  {chain_id}: {len(residues)} residues, "
            f"verts/res: min={min(vert_counts)}, max={max(vert_counts)}, "
            f"mean={np.mean(vert_counts):.1f}"
        )
