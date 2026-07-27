"""Audit canonical PhysPPI prior trees for fatal data errors and allowed states."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    MESH_COLUMNS,
    sample_field,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    QBioLiP10kReader,
)

_DATASET_READERS = {"qbiolip": QBioLiP10kReader, "afdb": AFDB10kReader}
_CHARGE_INDEX = MESH_COLUMNS.index("charge")
_ALLOWED_KEYS = ("opm_none", "mesh_charge_nan")
_WARNING_KEYS = ("main_msa_query_only",)


def _empty_dataset_report() -> dict[str, Any]:
    return {
        "scanned": 0,
        "allowed": {key: [] for key in _ALLOWED_KEYS},
        "warnings": {key: [] for key in _WARNING_KEYS},
        "fatal": {"ids": [], "reasons": {}},
    }


def _load_raw_sample(prior_pkl: Path, sample_id: str) -> object:
    with prior_pkl.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, Mapping) or sample_id not in raw:
        raise ValueError(f"prior PKL does not contain {sample_id}")
    return raw[sample_id]


def _pdb_ca_resseq(path: Path) -> dict[str, set[int]]:
    if not path.is_file():
        raise ValueError("missing_gt")
    residues: dict[str, set[int]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip()
            if not chain_id:
                continue
            try:
                resseq = int(line[22:26].strip())
                coords = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
            except ValueError as exc:
                raise ValueError("invalid_gt_ca_record") from exc
            if not all(math.isfinite(value) for value in coords):
                raise ValueError("invalid_gt_ca_coordinates")
            residues.setdefault(chain_id, set()).add(resseq)
    return residues


def _raw_mesh_has_charge_nan(sample: object) -> bool:
    raw_mesh = sample_field(sample, "mesh", {})
    if not isinstance(raw_mesh, Mapping):
        return False
    for rows in raw_mesh.values():
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            continue
        for row in rows:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                continue
            if len(row) <= _CHARGE_INDEX:
                continue
            try:
                if math.isnan(float(row[_CHARGE_INDEX])):
                    return True
            except (TypeError, ValueError, OverflowError):
                continue
    return False


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _validate_resseq_membership(
    sample: Any, ca_resseq: Mapping[str, set[int]]
) -> list[str]:
    reasons: list[str] = []
    for chain_pair, edges in sample.dmap.items():
        chain_1, chain_2 = chain_pair.split("|", 1)
        for r1, r2, _distance in edges:
            if r1 not in ca_resseq.get(chain_1, set()):
                _append_reason(reasons, f"missing_dmap_resseq:{chain_1}:{r1}")
            if r2 not in ca_resseq.get(chain_2, set()):
                _append_reason(reasons, f"missing_dmap_resseq:{chain_2}:{r2}")
    for interaction in sample.interactions:
        chain_1, chain_2 = interaction.chain_pair.split("|", 1)
        if interaction.r1 not in ca_resseq.get(chain_1, set()):
            _append_reason(
                reasons, f"missing_interaction_resseq:{chain_1}:{interaction.r1}"
            )
        if interaction.r2 not in ca_resseq.get(chain_2, set()):
            _append_reason(
                reasons, f"missing_interaction_resseq:{chain_2}:{interaction.r2}"
            )
    return reasons


def audit_dataset(dataset: str, root: str | Path) -> dict[str, Any]:
    dataset = dataset.lower()
    if dataset not in _DATASET_READERS:
        raise ValueError(f"unknown dataset: {dataset}")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"missing {dataset} root: {root}")

    reader = _DATASET_READERS[dataset](root)
    report = _empty_dataset_report()
    for sample_id in reader.sample_ids():
        report["scanned"] += 1
        paths = reader.locate(sample_id)
        reasons: list[str] = []
        raw_sample = None
        try:
            raw_sample = _load_raw_sample(paths.prior_pkl, sample_id)
            if sample_field(raw_sample, "opm", {}) is None:
                report["allowed"]["opm_none"].append(sample_id)
            if _raw_mesh_has_charge_nan(raw_sample):
                report["allowed"]["mesh_charge_nan"].append(sample_id)
            sample = reader.read(sample_id)
            ca_resseq = _pdb_ca_resseq(paths.gt_pdb)
            reasons.extend(_validate_resseq_membership(sample, ca_resseq))
            if sample.paths.main_msa_query_only:
                report["warnings"]["main_msa_query_only"].append(sample_id)
        except Exception as exc:
            _append_reason(reasons, str(exc))
        if reasons:
            report["fatal"]["ids"].append(sample_id)
            report["fatal"]["reasons"][sample_id] = reasons

    for values in report["allowed"].values():
        values.sort()
    for values in report["warnings"].values():
        values.sort()
    report["fatal"]["ids"].sort()
    report["fatal"]["reasons"] = dict(sorted(report["fatal"]["reasons"].items()))
    return report


def build_report(dataset: str, root: str | Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "datasets": {
            dataset: audit_dataset(dataset, root),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(_DATASET_READERS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.dataset, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fatal_ids = report["datasets"][args.dataset]["fatal"]["ids"]
    return 1 if fatal_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
