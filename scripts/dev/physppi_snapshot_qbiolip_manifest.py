"""Snapshot the visible Q-BioLiP-10k-new case tree.

The manifest is deliberately a small, stdlib-only boundary around the data
tree.  It never walks staging (``.chunk``) or legacy directories.  A malformed
prior is retained as a case with unavailable optional slots so that a census
cannot silently disappear because a pickle is temporarily unreadable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MSA_NAME = re.compile(r"^(?P<sample>.+)_(?P=sample)_(?:chain=)?(?P<chain>[^_=|_]+)$")
_CANONICAL_FIELDS = frozenset({"dmap", "interactions", "mesh", "opm"})


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _path_record(path: Path) -> dict[str, str | None]:
    return {"path": str(path), "sha256": _file_digest(path)}


def _field(obj: object | None, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _field_names(obj: object) -> set[str]:
    if isinstance(obj, Mapping):
        return {str(name) for name in obj}
    attributes = getattr(obj, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {str(name) for name in attributes}
    return set()


def _load_prior(path: Path, sample_id: str) -> tuple[object | None, str | None]:
    """Load one canonical prior and return a non-fatal diagnostic on failure."""

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # malformed or unavailable upstream pickle
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return None, "prior payload is not a mapping"
    if sample_id not in payload:
        return None, f"prior payload does not contain sample_id {sample_id!r}"
    sample = payload[sample_id]
    field_names = _field_names(sample)
    unknown = sorted(field_names - _CANONICAL_FIELDS)
    if unknown:
        return None, "non-canonical prior fields: " + ", ".join(unknown)
    if not field_names:
        return None, "prior sample has no canonical fields"
    return sample, None


def _optional_state(
    available: bool, *, available_provenance: str, missing_provenance: str
) -> dict[str, bool | str]:
    return {
        "available": bool(available),
        "selected": False,
        "valid": bool(available),
        "provenance": available_provenance if available else missing_provenance,
    }


def _has_value(value: object) -> bool:
    """Boolean availability that also handles array-like canonical fields."""

    try:
        return bool(value)
    except (TypeError, ValueError):
        try:
            return len(value) > 0  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False


def _sample_from_path(path: Path, sample_id: str) -> str | None:
    """Return a chain for an exact canonical MSA basename, else ``None``."""

    if path.suffix != ".npz":
        return None
    match = _MSA_NAME.fullmatch(path.stem)
    if match is None or match.group("sample") != sample_id:
        return None
    chain = match.group("chain")
    return chain if chain else None


def _monomer_chain(path: Path, sample_id: str) -> str | None:
    if path.suffix.lower() != ".pdb" or not path.stem.startswith(sample_id + "_"):
        return None
    chain = path.stem[len(sample_id) + 1 :]
    if chain.startswith("chain="):
        chain = chain[6:]
    return chain or None


def _candidate_roots(chunk: Path, kind: str) -> list[Path]:
    """Return only the canonical per-chunk MSA root.

    Deliberately do not search dataset-level or legacy aliases: allowing a
    fallback outside the visible chunk would make a manifest depend on files
    that the runtime reader cannot consume.
    """

    path = chunk / "msa_local" / kind
    return [path] if path.is_dir() else []


def _msa_paths(
    chunk: Path, sample_id: str, kind: str
) -> tuple[dict[str, Path], list[str]]:
    paths: dict[str, Path] = {}
    errors: list[str] = []
    for base in _candidate_roots(chunk, kind):
        # Main MSA supports a case directory and a shared flat directory.
        # Paired MSA is intentionally nested by case, but accepting a flat
        # paired root would make unrelated pairings ambiguous.
        locations = [base / sample_id] if kind == "paired" else [base / sample_id, base]
        for location in locations:
            if not location.is_dir():
                continue
            for path in sorted(location.iterdir()):
                if not path.is_file() or path.suffix != ".npz":
                    continue
                chain = _sample_from_path(path, sample_id)
                if chain is None:
                    continue
                if chain in paths:
                    errors.append(f"duplicate_{kind}_msa_chain:{chain}")
                    continue
                paths[chain] = path
    return dict(sorted(paths.items())), errors


def _one_row(root: Path, pkl_path: Path) -> dict[str, Any]:
    chunk = pkl_path.parents[1]
    sample_id = pkl_path.stem
    gt_path = chunk / "gt" / f"{sample_id}.pdb"

    monomers: dict[str, Path] = {}
    input_errors: list[str] = []
    bb_input = chunk / "bb_input"
    if bb_input.is_dir():
        for path in sorted(bb_input.glob(f"{sample_id}_*.pdb")):
            if not path.is_file():
                continue
            chain = _monomer_chain(path, sample_id)
            if not chain:
                continue
            if chain in monomers:
                input_errors.append(f"duplicate_monomer_chain:{chain}")
                continue
            monomers[chain] = path

    main, main_errors = _msa_paths(chunk, sample_id, "main")
    paired, paired_errors = _msa_paths(chunk, sample_id, "paired")
    input_errors.extend(main_errors)
    input_errors.extend(paired_errors)
    unknown_main = sorted(set(main) - set(monomers))
    unknown_paired = sorted(set(paired) - set(monomers))
    input_errors.extend(f"unknown_main_msa_chain:{chain}" for chain in unknown_main)
    input_errors.extend(f"unknown_paired_msa_chain:{chain}" for chain in unknown_paired)
    # An MSA for a chain absent from the mandatory monomer set is not an input
    # for this case.  Ignore it rather than allowing a foreign cache entry to
    # make a sample appear complete.
    main = {key: path for key, path in main.items() if key in monomers}
    paired = {key: path for key, path in paired.items() if key in monomers}

    prior, prior_error = _load_prior(pkl_path, sample_id)
    dmap = _field(prior, "dmap")
    interactions = _field(prior, "interactions")
    mesh = _field(prior, "mesh")
    opm = _field(prior, "opm")
    if isinstance(opm, Mapping):
        res_loc = opm.get("res_loc")
    else:
        res_loc = _field(opm, "res_loc")
    if prior_error is not None:
        # An invalid prior is retained for census diagnostics, never partially
        # promoted into an available optional input.
        dmap = interactions = mesh = res_loc = None

    path_errors: list[str] = list(input_errors)
    if not pkl_path.is_file():
        path_errors.append("missing_prior_pkl")
    if not gt_path.is_file():
        path_errors.append("missing_gt_pdb")
    if not monomers:
        path_errors.append("missing_monomer_pdbs")

    monomer_paths = [monomers[key] for key in sorted(monomers)]
    main_paths = [main[key] for key in sorted(main)]
    paired_paths = [paired[key] for key in sorted(paired)]
    query_only = sorted(set(monomers) - set(main))
    files = {
        "prior_pkl": _path_record(pkl_path),
        "gt_pdb": _path_record(gt_path),
        "monomer_pdbs": {key: _path_record(monomers[key]) for key in sorted(monomers)},
        "main_msa_npzs": {key: _path_record(main[key]) for key in sorted(main)},
        "paired_msa_npzs": {key: _path_record(paired[key]) for key in sorted(paired)},
    }
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "chunk_id": chunk.name,
        # Keep simple path lists as the stable public shape; ``files`` adds
        # per-file SHA256 records without changing existing consumers.
        "prior_pkl": str(pkl_path),
        "gt_pdb": str(gt_path),
        "monomer_pdbs": [str(path) for path in monomer_paths],
        "main_msa_npzs": [str(path) for path in main_paths],
        "paired_msa_npzs": [str(path) for path in paired_paths],
        "main_msa_by_chain": {key: str(main[key]) for key in sorted(main)},
        "paired_msa_by_chain": {key: str(paired[key]) for key in sorted(paired)},
        "main_msa_query_only_chains": query_only,
        "main_msa_query_only": bool(query_only),
        "mandatory_valid": not path_errors,
        "path_errors": path_errors,
        "source_sha256": _file_digest(pkl_path),
        "prior_sha256": _file_digest(pkl_path),
        "prior_pkl_sha256": _file_digest(pkl_path),
        "sha256": {
            "prior_pkl": _file_digest(pkl_path),
            "gt_pdb": _file_digest(gt_path),
            "monomer_pdbs": {
                key: _file_digest(monomers[key]) for key in sorted(monomers)
            },
            "main_msa_npzs": {key: _file_digest(main[key]) for key in sorted(main)},
            "paired_msa_npzs": {
                key: _file_digest(paired[key]) for key in sorted(paired)
            },
        },
        "files": files,
        "optional": {
            "paired_msa": _optional_state(
                _has_value(paired_paths),
                available_provenance="qbiolip10k",
                missing_provenance="unavailable",
            ),
            "contact": _optional_state(
                _has_value(dmap),
                available_provenance="qbiolip10k",
                missing_provenance="unavailable",
            ),
            "interaction": _optional_state(
                _has_value(interactions),
                available_provenance="qbiolip10k",
                missing_provenance="unavailable",
            ),
            "surface": _optional_state(
                _has_value(mesh),
                available_provenance="qbiolip10k",
                missing_provenance="unavailable",
            ),
            "tm": _optional_state(
                _has_value(res_loc),
                available_provenance="qbiolip10k",
                missing_provenance="unavailable",
            ),
        },
    }
    if prior_error is not None:
        row["prior_load_error"] = prior_error
    return row


def build_manifest(root: str | Path) -> list[dict[str, Any]]:
    """Build deterministic rows for ``root/chunk/chunk*/prior_pkl/*.pkl``."""

    root = Path(root)
    final_root = root / "chunk"
    if not final_root.is_dir():
        raise FileNotFoundError(f"missing canonical chunk root: {final_root}")
    pkl_paths = sorted(final_root.glob("chunk*/prior_pkl/*.pkl"))
    # Keep non-file ``*.pkl`` entries as invalid rows so a malformed snapshot
    # cannot silently reduce the census count.
    rows = [_one_row(root, path) for path in pkl_paths]
    rows.sort(key=lambda row: row["sample_id"])
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample id in canonical chunk tree")
    return rows


def write_manifest(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root)
    output = Path(output)
    rows = build_manifest(root)
    payload = {
        "schema_version": 1,
        "dataset_root": str(root),
        "count": len(rows),
        "case_count": len(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_manifest(args.root, args.output)
    print(json.dumps({"output": str(args.output), "count": payload["count"]}))


if __name__ == "__main__":
    main()
