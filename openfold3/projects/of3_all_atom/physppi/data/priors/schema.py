"""Shared canonical prior-complex records and parsers for PhysPPI readers."""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

AtomGroup = tuple[str, ...]

INTERACTION_TYPES = (
    "hbond",
    "saltbridge",
    "hydrophobic",
    "pistacking",
    "pication",
    "waterbridge",
    "halogen",
)

MESH_COLUMNS = (
    "x",
    "y",
    "z",
    "charge",
    "hbond",
    "hphob",
    "nx",
    "ny",
    "nz",
    "si",
)


@dataclass(frozen=True)
class InteractionRecord:
    """One typed prior interaction edge."""

    chain_pair: str
    r1: int
    r2: int
    interaction_type: str
    distance: float | None
    atom_1: AtomGroup | None
    atom_2: AtomGroup | None


@dataclass(frozen=True)
class PriorSamplePaths:
    """Canonical files belonging to one prior-complex sample."""

    sample_id: str
    gt_pdb: Path
    monomer_pdbs: dict[str, Path]
    main_msa_npzs: dict[str, Path]
    paired_msa_npzs: dict[str, Path]
    prior_pkl: Path

    @property
    def main_msa_query_only_chains(self) -> tuple[str, ...]:
        """Monomer chains that must use the mandatory query-only fallback."""

        # AFDB stores one representative main MSA under the sample id because
        # both homodimer chains share that file.  It is a real searched MSA,
        # not a query-only fallback for either chain.
        if len(self.main_msa_npzs) == 1 and self.sample_id in self.main_msa_npzs:
            return ()

        return tuple(
            chain_id
            for chain_id in sorted(self.monomer_pdbs)
            if chain_id not in self.main_msa_npzs
        )

    @property
    def main_msa_query_only(self) -> bool:
        """Whether at least one chain uses the mandatory query-only fallback."""

        return bool(self.main_msa_query_only_chains)


@dataclass(frozen=True)
class PriorComplexSample:
    """Parsed canonical prior labels and input paths."""

    sample_id: str
    dmap: dict[str, tuple[tuple[int, int, float], ...]]
    interactions: tuple[InteractionRecord, ...]
    mesh: dict[str, tuple[tuple[float, ...], ...]]
    opm: dict[str, object]
    paths: PriorSamplePaths
    chain_sequences: dict[str, str]
    dataset_name: str = ""

    @property
    def chain_ids(self) -> tuple[str, ...]:
        """Return the complete stable chain inventory for this sample."""

        chain_ids = set(self.paths.monomer_pdbs)
        chain_ids.update(self.chain_sequences)
        chain_ids.update(self.mesh)
        for chain_pair in self.dmap:
            chain_ids.update(chain_pair.split("|"))
        for interaction in self.interactions:
            chain_ids.update(interaction.chain_pair.split("|"))
        return tuple(sorted(chain_ids))


_MISSING = object()
_CANONICAL_FIELDS = frozenset({"dmap", "interactions", "mesh", "opm"})
_LEGACY_FIELDS = frozenset(
    {"cmap", "attractive", "source1", "source2", "prior_processed"}
)


def sample_field(sample: object, name: str, default: Any = _MISSING) -> Any:
    """Read a canonical field from either a mapping or an attribute object."""

    if isinstance(sample, Mapping):
        value = sample.get(name, _MISSING)
    else:
        value = getattr(sample, name, _MISSING)
    if value is _MISSING:
        return default
    return value


def sample_field_names(sample: object) -> set[str]:
    if isinstance(sample, Mapping):
        return {str(name) for name in sample}
    attributes = getattr(sample, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {str(name) for name in attributes}
    return set()


def pdb_resseq(value: object, field: str) -> int:
    """Validate a PDB residue sequence number without truncating floats."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer PDB residue sequence number")
    return int(value)


def validate_chain_pair(chain_pair: object, field: str) -> str:
    if not isinstance(chain_pair, str):
        raise ValueError(f"{field} must use chain-pair syntax A|B")
    parts = chain_pair.split("|")
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"{field} must use chain-pair syntax A|B")
    if any("|" in part for part in parts):
        raise ValueError(f"{field} must use chain-pair syntax A|B")
    return chain_pair


def normalize_atom_group(value: object) -> AtomGroup | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(
        isinstance(atom, str) for atom in value
    ):
        return tuple(value)
    raise TypeError(
        f"atom endpoint must be a string or atom-name list, got {type(value)!r}"
    )


def parse_dmap(
    raw: object, dataset_name: str
) -> dict[str, tuple[tuple[int, int, float], ...]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{dataset_name} dmap must be a mapping")

    parsed: dict[str, tuple[tuple[int, int, float], ...]] = {}
    for chain_pair, edges in raw.items():
        chain_pair = validate_chain_pair(chain_pair, f"{dataset_name} dmap key")
        if not isinstance(edges, (list, tuple)):
            raise ValueError(f"{dataset_name} dmap must map chain pairs to edge rows")

        parsed_edges: list[tuple[int, int, float]] = []
        for edge in edges:
            if isinstance(edge, (str, bytes)) or not isinstance(edge, Sequence):
                raise ValueError(f"dmap edge must be (r1, r2, distance), got {edge}")
            if len(edge) != 3:
                raise ValueError(f"dmap edge must be (r1, r2, distance), got {edge}")
            r1 = pdb_resseq(edge[0], "dmap r1")
            r2 = pdb_resseq(edge[1], "dmap r2")
            try:
                distance = float(edge[2])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid dmap edge: {edge}") from exc
            if not math.isfinite(distance):
                raise ValueError(f"invalid dmap edge distance: {distance!r}")
            parsed_edges.append((r1, r2, distance))
        parsed[chain_pair] = tuple(parsed_edges)
    return parsed


def iter_chain_pair_distances(
    dmap: Mapping[str, Sequence[tuple[int, int, float]]],
):
    """Yield stable ``(left_chain, right_chain, distance)`` dmap rows."""

    for chain_pair in sorted(dmap):
        validated = validate_chain_pair(chain_pair, "dmap key")
        left, right = validated.split("|")
        rows = sorted(
            dmap[chain_pair],
            key=lambda row: (float(row[2]), int(row[0]), int(row[1])),
        )
        for _, _, distance in rows:
            yield left, right, float(distance)


def parse_interactions(raw: object, dataset_name: str) -> tuple[InteractionRecord, ...]:
    if raw is None or (isinstance(raw, Mapping) and not raw):
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{dataset_name} interactions must be a mapping")

    parsed: list[InteractionRecord] = []
    for chain_pair, records in raw.items():
        chain_pair = validate_chain_pair(chain_pair, f"{dataset_name} interaction key")
        if not isinstance(records, (list, tuple)):
            raise ValueError(f"{dataset_name} interaction values must be lists")
        for item in records:
            if not isinstance(item, Mapping):
                raise ValueError("interaction entry must be a mapping")
            kind = str(item.get("t", ""))
            if kind not in INTERACTION_TYPES:
                raise ValueError(f"unsupported interaction type: {kind}")
            try:
                raw_r1 = item["r1"]
                raw_r2 = item["r2"]
            except (KeyError, TypeError) as exc:
                raise ValueError("interaction entry requires integer r1/r2") from exc
            r1 = pdb_resseq(raw_r1, "interaction r1")
            r2 = pdb_resseq(raw_r2, "interaction r2")
            distance_raw = item.get("d")
            if distance_raw is None:
                distance = None
            else:
                try:
                    distance = float(distance_raw)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("interaction distance must be numeric") from exc
                if not math.isfinite(distance):
                    raise ValueError("interaction distance must be finite")
            try:
                atom_1 = normalize_atom_group(item.get("a1"))
                atom_2 = normalize_atom_group(item.get("a2"))
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
            parsed.append(
                InteractionRecord(
                    chain_pair=chain_pair,
                    r1=r1,
                    r2=r2,
                    interaction_type=kind,
                    distance=distance,
                    atom_1=atom_1,
                    atom_2=atom_2,
                )
            )
    return tuple(parsed)


def parse_mesh(
    raw: object, dataset_name: str
) -> dict[str, tuple[tuple[float, ...], ...]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{dataset_name} mesh must be a mapping")
    parsed: dict[str, tuple[tuple[float, ...], ...]] = {}
    for chain_id, rows in raw.items():
        if not isinstance(chain_id, str) or not chain_id:
            raise ValueError(f"{dataset_name} mesh chain id must be a non-empty string")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError(f"{dataset_name} mesh rows must be sequences")
        parsed_rows: list[tuple[float, ...]] = []
        for row in rows:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise ValueError(f"{dataset_name} mesh row must contain 10 columns")
            if len(row) != len(MESH_COLUMNS):
                raise ValueError(f"{dataset_name} mesh row must contain 10 columns")
            try:
                values = tuple(float(value) for value in row)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{dataset_name} mesh row must be numeric") from exc
            for index, value in enumerate(values):
                if index == MESH_COLUMNS.index("charge") and math.isnan(value):
                    continue
                if not math.isfinite(value):
                    raise ValueError(f"{dataset_name} mesh row must be finite")
            parsed_rows.append(values)
        parsed[chain_id] = tuple(parsed_rows)
    return parsed


def parse_prior_payload(
    raw: object,
    sample_id: str,
    paths: PriorSamplePaths,
    dataset_name: str,
    chain_sequences: Mapping[str, str] | None = None,
) -> PriorComplexSample:
    if not isinstance(raw, Mapping) or sample_id not in raw:
        raise ValueError(f"{dataset_name} PKL does not contain {sample_id}")

    sample = raw[sample_id]
    field_names = sample_field_names(sample)
    legacy_fields = sorted(field_names & _LEGACY_FIELDS)
    if legacy_fields:
        raise ValueError(
            f"legacy {dataset_name} fields are not supported: "
            + ", ".join(legacy_fields)
        )
    unknown_fields = sorted(field_names - _CANONICAL_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"unknown {dataset_name} prior fields: " + ", ".join(unknown_fields)
        )

    dmap = parse_dmap(sample_field(sample, "dmap", None), dataset_name)
    interactions = parse_interactions(
        sample_field(sample, "interactions", None), dataset_name
    )
    mesh = parse_mesh(sample_field(sample, "mesh", {}), dataset_name)
    opm = sample_field(sample, "opm", {})
    if not isinstance(opm, Mapping) and opm is not None:
        raise ValueError(f"{dataset_name} opm must be a mapping")

    return PriorComplexSample(
        sample_id=sample_id,
        dmap=dmap,
        interactions=interactions,
        mesh=mesh,
        opm=dict(opm or {}),
        paths=paths,
        chain_sequences=dict(chain_sequences or {}),
        dataset_name=str(dataset_name),
    )


def read_prior_pkl(
    prior_pkl: Path,
    sample_id: str,
    paths: PriorSamplePaths,
    dataset_name: str,
    chain_sequences: Mapping[str, str] | None = None,
) -> PriorComplexSample:
    with prior_pkl.open("rb") as handle:
        raw = pickle.load(handle)
    return parse_prior_payload(raw, sample_id, paths, dataset_name, chain_sequences)


__all__ = [
    "AtomGroup",
    "INTERACTION_TYPES",
    "MESH_COLUMNS",
    "InteractionRecord",
    "PriorComplexSample",
    "PriorSamplePaths",
    "normalize_atom_group",
    "iter_chain_pair_distances",
    "parse_dmap",
    "parse_interactions",
    "parse_mesh",
    "pdb_resseq",
    "read_prior_pkl",
]
