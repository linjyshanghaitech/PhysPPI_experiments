"""NumPy feature builders for the canonical Q-BioLiP-10k prior schema.

The builders in this module are deliberately independent of OpenFold and
PyTorch.  They map residue-level Q-BioLiP records to token-indexed arrays;
selection and batching are handled by the dataset layer.  Membrane planes are
returned as supervision targets only and are never represented as an input
feature here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    INTERACTION_TYPES,
    InteractionRecord,
)

_INTERACTION_CHANNELS = {name: index for index, name in enumerate(INTERACTION_TYPES)}


class PhysPPIDataError(ValueError):
    """Raised when a source-present PhysPPI optional payload is malformed."""


def _num_tokens(token_index: Mapping[tuple[str, int], int]) -> int:
    """Return the dense token count implied by a token-index mapping."""

    if not token_index:
        return 0
    indices = [int(index) for index in token_index.values() if int(index) >= 0]
    return max(indices, default=-1) + 1


def _chain_pair(value: object) -> tuple[str, str] | None:
    """Parse a canonical cross-chain key, returning ``None`` if malformed."""

    if not isinstance(value, str):
        return None
    parts = value.split("|")
    if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] == parts[1]:
        return None
    return parts[0], parts[1]


def _residue_key(chain: object, residue: object) -> tuple[str, int] | None:
    """Normalize a chain/residue pair without accepting non-integral ids."""

    if not isinstance(chain, str) or not chain:
        return None
    if isinstance(residue, bool) or not isinstance(residue, Integral):
        return None
    return chain, int(residue)


def _edge_values(edge: object) -> tuple[object, object, object] | None:
    if isinstance(edge, (str, bytes)) or not isinstance(edge, Sequence):
        return None
    if len(edge) < 3:
        return None
    return edge[0], edge[1], edge[2]


def _normalize_atom_group(value: object) -> tuple[str, ...] | None:
    """Normalize a scalar or grouped atom endpoint to canonical atom names."""

    if isinstance(value, str):
        name = value.strip().upper()
        return (name,) if name else None
    if isinstance(value, (list, tuple)):
        names = tuple(
            atom.strip().upper()
            for atom in value
            if isinstance(atom, str) and atom.strip()
        )
        if len(names) == len(value) and names:
            return names
    return None


def _interaction_atom_lookup(
    atom_array: object,
) -> dict[tuple[str, int, str], np.ndarray]:
    """Index finite structure coordinates by residue and normalized atom name."""

    try:
        chains = np.asarray(atom_array.chain_id)
        residues = np.asarray(atom_array.res_id)
        names = np.asarray(atom_array.atom_name)
        coords = np.asarray(atom_array.coord, dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        return {}
    if (
        chains.ndim != 1
        or residues.ndim != 1
        or names.ndim != 1
        or coords.ndim != 2
        or coords.shape[1] != 3
        or not (chains.shape == residues.shape == names.shape == coords.shape[:1])
    ):
        return {}

    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    for chain, residue, name, coord in zip(
        chains, residues, names, coords, strict=True
    ):
        try:
            residue_key = _residue_key(str(chain), int(residue))
        except (TypeError, ValueError, OverflowError):
            continue
        atom_name = _normalize_atom_group(str(name))
        if residue_key is None or atom_name is None or not np.isfinite(coord).all():
            continue
        lookup.setdefault((*residue_key, atom_name[0]), coord.astype(np.float32))
    return lookup


def _interaction_group_centroid(
    lookup: Mapping[tuple[str, int, str], np.ndarray],
    chain: str,
    residue: int,
    group: object,
) -> np.ndarray | None:
    names = _normalize_atom_group(group)
    if names is None:
        return None
    coords = []
    for name in names:
        coord = lookup.get((chain, residue, name))
        if coord is None or coord.shape != (3,) or not np.isfinite(coord).all():
            return None
        coords.append(coord)
    centroid = np.mean(np.stack(coords, axis=0), axis=0)
    return centroid.astype(np.float32) if np.isfinite(centroid).all() else None


def _finite_record_distance(record: InteractionRecord) -> float | None:
    try:
        distance = float(record.distance)
    except (TypeError, ValueError, OverflowError):
        return None
    return distance if np.isfinite(distance) else None


def iter_mappable_dmap_edges(
    dmap: Mapping[str, Sequence[Sequence[object]]],
    token_index: Mapping[tuple[str, int], int],
) -> Iterator[tuple[str, str, int, int, float]]:
    """Yield validated dmap edges whose endpoints survive the model crop."""

    n_tokens = _num_tokens(token_index)
    if dmap and not isinstance(dmap, Mapping):
        raise PhysPPIDataError("contact payload must be a mapping")
    if isinstance(dmap, Mapping):
        for chain_pair, edges in dmap.items():
            pair = _chain_pair(chain_pair)
            if pair is None:
                raise PhysPPIDataError("contact payload has an invalid chain pair")
            if isinstance(edges, (str, bytes)) or not isinstance(edges, Sequence):
                raise PhysPPIDataError("contact payload has invalid edge records")
            left_chain, right_chain = pair
            for edge in edges:
                values = _edge_values(edge)
                if values is None:
                    raise PhysPPIDataError("contact payload has an invalid edge")
                raw_r1, raw_r2, raw_distance = values
                left_key = _residue_key(left_chain, raw_r1)
                right_key = _residue_key(right_chain, raw_r2)
                if left_key is None or right_key is None:
                    raise PhysPPIDataError("contact payload has an invalid residue")
                try:
                    distance = float(raw_distance)
                except (TypeError, ValueError, OverflowError) as error:
                    raise PhysPPIDataError(
                        "contact payload has an invalid distance"
                    ) from error
                if not np.isfinite(distance):
                    raise PhysPPIDataError("contact payload has an invalid distance")
                left = token_index.get(left_key)
                right = token_index.get(right_key)
                if left is None or right is None:
                    continue
                left, right = int(left), int(right)
                if left < 0 or right < 0 or left >= n_tokens or right >= n_tokens:
                    continue
                if left == right:
                    continue
                yield left_chain, right_chain, left, right, distance


def build_contact_features(
    dmap: Mapping[str, Sequence[Sequence[object]]],
    token_index: Mapping[tuple[str, int], int],
    cutoff: float,
) -> dict[str, Any]:
    """Reduce dmap supervision to a symmetric boolean pair-contact target."""

    n_tokens = _num_tokens(token_index)
    pair_mask = np.zeros((n_tokens, n_tokens), dtype=bool)
    try:
        threshold = float(cutoff)
    except (TypeError, ValueError, OverflowError):
        threshold = float("nan")
    for left_chain, right_chain, left, right, distance in iter_mappable_dmap_edges(
        dmap, token_index
    ):
        if (
            left_chain != right_chain
            and np.isfinite(distance)
            and distance <= threshold
        ):
            pair_mask[left, right] = True
            pair_mask[right, left] = True
    np.fill_diagonal(pair_mask, False)

    return {
        "available": bool(dmap),
        "valid": bool(pair_mask.any()),
        "pair_mask": pair_mask,
        "cutoff": threshold,
    }


def build_interaction_features(
    records: Sequence[InteractionRecord] | None,
    token_index: Mapping[tuple[str, int], int],
    *,
    atom_array: object | None = None,
) -> dict[str, Any]:
    """Build seven typed, symmetric interaction channels.

    A nonempty record sequence is available even when all endpoints fail
    validation.  ``valid`` indicates that at least one record maps to a known
    interaction channel and both token endpoints.  Auxiliary distance fields
    are independent: when ``atom_array`` is supplied, atom endpoints are
    resolved against the structure and the target is the centroid distance
    between the two complete atom groups. The canonical adapter always passes
    structure geometry; ``record.distance`` is retained only as a fallback for
    callers using this pure builder without a structure.
    """

    n_tokens = _num_tokens(token_index)
    type_mask = np.zeros((n_tokens, n_tokens, len(INTERACTION_TYPES)), dtype=bool)
    aux: list[
        tuple[
            int, int, int, float | None, tuple[str, ...] | None, tuple[str, ...] | None
        ]
    ] = []
    if records is not None and (
        isinstance(records, (str, bytes)) or not isinstance(records, Sequence)
    ):
        raise PhysPPIDataError("interaction payload must be a sequence")
    record_values = tuple(records) if records is not None else ()
    atom_lookup = (
        _interaction_atom_lookup(atom_array) if atom_array is not None else None
    )

    for record in record_values:
        if not isinstance(record, InteractionRecord):
            raise PhysPPIDataError("interaction payload has an invalid record")
        pair = _chain_pair(record.chain_pair)
        channel = _INTERACTION_CHANNELS.get(record.interaction_type)
        if pair is None:
            raise PhysPPIDataError("interaction payload has an invalid chain pair")
        if channel is None:
            raise PhysPPIDataError("interaction payload has an invalid type")
        left_chain, right_chain = pair
        left_key = _residue_key(left_chain, record.r1)
        right_key = _residue_key(right_chain, record.r2)
        if left_key is None or right_key is None:
            raise PhysPPIDataError("interaction payload has an invalid residue")
        left = token_index.get(left_key)
        right = token_index.get(right_key)
        if left is None or right is None:
            continue
        left, right = int(left), int(right)
        if left < 0 or right < 0 or left >= n_tokens or right >= n_tokens:
            continue
        if left == right:
            continue
        type_mask[left, right, channel] = True
        type_mask[right, left, channel] = True
        distance = _finite_record_distance(record)
        if atom_lookup is not None:
            left_centroid = _interaction_group_centroid(
                atom_lookup,
                left_chain,
                int(record.r1),
                record.atom_1,
            )
            right_centroid = _interaction_group_centroid(
                atom_lookup,
                right_chain,
                int(record.r2),
                record.atom_2,
            )
            if left_centroid is None or right_centroid is None:
                distance = None
            else:
                candidate = float(np.linalg.norm(left_centroid - right_centroid))
                distance = candidate if np.isfinite(candidate) else None
        if distance is not None:
            aux.append(
                (
                    left,
                    right,
                    channel,
                    distance,
                    record.atom_1,
                    record.atom_2,
                )
            )

    if n_tokens:
        token_ids = np.arange(n_tokens)
        type_mask[token_ids, token_ids, :] = False
    return {
        "available": bool(record_values),
        "valid": bool(type_mask.any()),
        "type_mask": type_mask,
        "aux": tuple(aux),
    }


def build_surface_features(
    mesh_by_chain: Mapping[str, object],
    token_index: Mapping[tuple[str, int], int],
    monomer_transforms: Mapping[str, tuple[np.ndarray, np.ndarray]],
    k: int,
    *,
    atom_array: object | None = None,
) -> dict[str, Any]:
    """Build token-local nearest-neighbor surface crops.

    The canonical Q-BioLiP mesh has ten columns:
    xyz, charge, hbond, hydrophobicity, normal xyz, shape index.  The hbond
    column is excluded from the network input, leaving six attributes:
    charge, hydrophobicity, local normal xyz, and shape index.
    """

    if isinstance(k, bool) or not isinstance(k, Integral) or int(k) < 0:
        raise ValueError("k must be a non-negative integer")
    k = int(k)
    n_tokens = _num_tokens(token_index)
    coords = np.zeros((n_tokens, k, 3), dtype=np.float32)
    features = np.zeros((n_tokens, k, 6), dtype=np.float32)
    mask = np.zeros((n_tokens, k), dtype=bool)
    charge_mask = np.zeros((n_tokens, k), dtype=bool)

    frames = _surface_token_frames(
        atom_array, token_index, n_tokens, monomer_transforms
    )

    if mesh_by_chain and not isinstance(mesh_by_chain, Mapping):
        raise PhysPPIDataError("surface payload must be a mapping")
    if isinstance(mesh_by_chain, Mapping):
        for chain_id, raw_mesh in mesh_by_chain.items():
            if not isinstance(chain_id, str):
                raise PhysPPIDataError("surface payload has an invalid chain")
            try:
                mesh = np.asarray(raw_mesh, dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise PhysPPIDataError("surface payload has an invalid mesh") from error
            if mesh.ndim != 2 or mesh.shape[1] != 10 or mesh.shape[0] == 0:
                raise PhysPPIDataError("surface payload has an invalid mesh")
            xyz = mesh[:, :3]
            raw_attrs = mesh[:, 3:10]
            raw_normals = mesh[:, 6:9]
            finite = (
                np.isfinite(xyz).all(axis=1)
                & np.isfinite(raw_attrs[:, 2])
                & np.isfinite(raw_normals).all(axis=1)
                & np.isfinite(raw_attrs[:, 6])
            )
            if not finite.all():
                raise PhysPPIDataError("surface payload has non-finite geometry")
            if chain_id not in monomer_transforms:
                continue
            transform = monomer_transforms[chain_id]
            if not isinstance(transform, Sequence) or len(transform) != 2:
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
            xyz = xyz[finite]
            raw_attrs = raw_attrs[finite]
            raw_normals = raw_normals[finite]
            if xyz.shape[0] == 0:
                continue
            try:
                xyz = xyz @ rotation + translation
                normals = raw_normals @ rotation
            except (TypeError, ValueError):
                continue
            if not np.isfinite(xyz).all() or not np.isfinite(normals).all():
                continue
            for (token_chain, _), token in token_index.items():
                if token_chain != chain_id:
                    continue
                token = int(token)
                if token < 0 or token >= n_tokens:
                    continue
                frame = frames.get(token)
                if frame is None:
                    continue
                center, basis = frame
                distances = np.sum((xyz - center) ** 2, axis=1)
                order = np.lexsort((np.arange(xyz.shape[0]), distances))
                count = min(k, order.shape[0])
                if count == 0:
                    continue
                selected = order[:count]
                local_xyz = (xyz[selected] - center) @ basis
                local_normals = normals[selected] @ basis
                selected_charge_mask = np.isfinite(raw_attrs[selected, 0])
                charge = np.where(selected_charge_mask, raw_attrs[selected, 0], 0.0)
                attrs = np.concatenate(
                    (
                        charge[:, None].astype(np.float32),
                        raw_attrs[selected, 2:3],
                        local_normals,
                        raw_attrs[selected, 6:7],
                    ),
                    axis=-1,
                )
                if not np.isfinite(local_xyz).all() or not np.isfinite(attrs).all():
                    continue
                coords[token, :count] = local_xyz
                features[token, :count] = attrs
                mask[token, :count] = True
                charge_mask[token, :count] = selected_charge_mask

    return {
        "coords": coords,
        "features": features,
        "mask": mask,
        "charge_mask": charge_mask,
        "available": bool(mesh_by_chain),
        "valid": bool(mask.any()),
    }


def _surface_token_frames(
    atom_array: object | None,
    token_index: Mapping[tuple[str, int], int],
    n_tokens: int,
    monomer_transforms: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
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

    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for residue_key, token in token_index.items():
        token = int(token)
        if token < 0 or token >= n_tokens:
            continue
        atoms = atoms_by_residue.get(residue_key)
        if not atoms:
            continue
        ca = atoms.get("CA")
        n = atoms.get("N")
        c = atoms.get("C")
        if ca is None or n is None or c is None:
            continue
        center = atoms.get("CB")
        if center is None or names_by_residue.get(residue_key) == "GLY":
            center = ca
        basis = _backbone_local_basis(n, ca, c)
        if basis is None:
            continue
        frames[token] = (center.astype(np.float32), basis)
    return frames


def _backbone_local_basis(
    n: np.ndarray, ca: np.ndarray, c: np.ndarray
) -> np.ndarray | None:
    e1_raw = c - ca
    n_dir_raw = n - ca
    e1 = _normalize(e1_raw)
    n_dir = _normalize(n_dir_raw)
    if e1 is None or n_dir is None:
        return None
    e2_raw = n_dir - np.dot(n_dir, e1) * e1
    e2 = _normalize(e2_raw)
    if e2 is None:
        return None
    e3 = _normalize(np.cross(e1, e2))
    if e3 is None:
        return None
    return np.stack((e1, e2, e3), axis=1).astype(np.float32)


def _normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    if not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        return None
    return (vector / norm).astype(np.float32)


def _tm_side_is_zero(side: object) -> bool:
    if isinstance(side, bool):
        return not side
    if isinstance(side, Real):
        return float(side) == 0.0
    if isinstance(side, str):
        try:
            return float(side.strip()) == 0.0
        except (TypeError, ValueError) as error:
            raise PhysPPIDataError("TM payload has an invalid side value") from error
    raise PhysPPIDataError("TM payload has an invalid side value")


def _tm_interval_rows(
    res_loc: object,
) -> Iterator[tuple[str, int, int, object, bool]]:
    rows = res_loc.tolist() if isinstance(res_loc, np.ndarray) else res_loc
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise PhysPPIDataError("TM payload must be a mapping or interval-row sequence")
    for row in rows:
        values = row.tolist() if isinstance(row, np.ndarray) else row
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise PhysPPIDataError("TM interval row must contain four values")
        if len(values) != 4:
            raise PhysPPIDataError("TM interval row must contain four values")
        chain_id, start, end, side = values
        start_key = _residue_key(chain_id, start)
        end_key = _residue_key(chain_id, end)
        if start_key is None or end_key is None:
            raise PhysPPIDataError("TM interval has an invalid chain or residue")
        start_id = start_key[1]
        end_id = end_key[1]
        if end_id < start_id:
            raise PhysPPIDataError("TM interval end precedes its start")
        yield str(chain_id), start_id, end_id, side, _tm_side_is_zero(side)


def restrict_tm_payload_to_chains(
    res_loc: object,
    chain_ids: Sequence[str],
) -> object:
    """Return the source TM payload restricted to retained structure chains."""

    retained = set(chain_ids)
    if isinstance(res_loc, Mapping):
        return {
            chain_id: residue_map
            for chain_id, residue_map in res_loc.items()
            if chain_id in retained
        }
    if res_loc is None:
        return None
    return [
        [chain_id, start, end, side]
        for chain_id, start, end, side, _ in _tm_interval_rows(res_loc)
        if chain_id in retained
    ]


def build_tm_features(
    res_loc: object,
    token_index: Mapping[tuple[str, int], int],
) -> np.ndarray:
    """Return a token mask for TM residues whose side/tag is exactly zero."""

    result = np.zeros(_num_tokens(token_index), dtype=bool)
    if res_loc is None:
        return result
    if isinstance(res_loc, Mapping):
        for chain_id, residue_map in res_loc.items():
            if not isinstance(residue_map, Mapping):
                raise PhysPPIDataError("TM payload has an invalid residue mapping")
            for residue_id, side in residue_map.items():
                key = _residue_key(chain_id, residue_id)
                if key is None:
                    raise PhysPPIDataError("TM payload has an invalid residue")
                if not _tm_side_is_zero(side):
                    continue
                token = token_index.get(key)
                if token is None:
                    continue
                token = int(token)
                if 0 <= token < result.shape[0]:
                    result[token] = True
        return result

    normalized_tokens = [
        (str(chain_id), int(residue_id), int(token))
        for (chain_id, residue_id), token in token_index.items()
    ]
    for chain_id, start, end, _, side_zero in _tm_interval_rows(res_loc):
        if not side_zero:
            continue
        for token_chain, residue_id, token in normalized_tokens:
            if (
                token_chain == chain_id
                and start <= residue_id <= end
                and 0 <= token < result.shape[0]
            ):
                result[token] = True
    return result


def build_membrane_target(
    opm: Mapping[str, object], is_membrane: bool
) -> dict[str, Any]:
    """Build a normalized directed membrane plane as a label-only target."""

    zero = np.zeros(3, dtype=np.float32)
    if not is_membrane or not isinstance(opm, Mapping):
        return {"valid": False, "normal": zero.copy(), "point": zero.copy()}
    try:
        normal = np.asarray(opm.get("outer_normal"), dtype=np.float32)
        point = np.asarray(opm.get("outer_centroid"), dtype=np.float32)
    except (TypeError, ValueError):
        return {"valid": False, "normal": zero.copy(), "point": zero.copy()}
    if normal.shape != (3,) or point.shape != (3,):
        return {"valid": False, "normal": zero.copy(), "point": zero.copy()}
    if not np.isfinite(normal).all() or not np.isfinite(point).all():
        return {"valid": False, "normal": zero.copy(), "point": zero.copy()}
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-6:
        return {"valid": False, "normal": zero.copy(), "point": zero.copy()}
    # Preserve the source direction.  There is deliberately no sign
    # minimization or PCA convention: the plane is a directed label.
    return {"valid": True, "normal": normal / norm, "point": point}


__all__ = [
    "PhysPPIDataError",
    "build_contact_features",
    "build_interaction_features",
    "build_membrane_target",
    "build_surface_features",
    "build_tm_features",
    "restrict_tm_payload_to_chains",
]
