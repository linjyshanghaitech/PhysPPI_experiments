from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

DEFAULT_DATASET_ROOT = Path("/101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k")
DEFAULT_MSA_LOCAL_ROOT = DEFAULT_DATASET_ROOT / "msa_local"
DEFAULT_OUTPUT_JSON = Path(
    "output/physppi_q10k_local_msa_training/frozen/q10k_local_msa_available_20260630.json"
)


def default_rep_id(case_id: str, chain_id: str) -> str:
    return f"{case_id}_{case_id}_chain={chain_id}"


def parse_pdb_chain_sequences(pdb_path: Path) -> dict[str, str]:
    residues: dict[str, list[tuple[tuple[int, str], str]]] = {}
    seen_residues: dict[str, set[tuple[int, str]]] = {}
    with pdb_path.open() as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            resname = line[17:20].strip().upper()
            residue = AA3_TO_1.get(resname)
            if residue is None:
                continue
            chain_id = line[21].strip() or "_"
            resseq = int(line[22:26])
            insertion_code = line[26].strip()
            residue_key = (resseq, insertion_code)
            chain_seen = seen_residues.setdefault(chain_id, set())
            if residue_key in chain_seen:
                continue
            chain_seen.add(residue_key)
            residues.setdefault(chain_id, []).append((residue_key, residue))
    return {
        chain_id: "".join(residue for _, residue in rows)
        for chain_id, rows in residues.items()
        if rows
    }


def parse_pdb_chain_ids(pdb_path: Path) -> list[str]:
    return list(parse_pdb_chain_sequences(pdb_path))


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _candidate_case_ids(dataset_root: Path, msa_local_root: Path) -> list[str]:
    ids: set[str] = set()
    ids.update(path.stem for path in (dataset_root / "total_pkl").glob("*.pkl"))
    ids.update(path.stem for path in (dataset_root / "gt_multimer").glob("*.pdb"))
    paired_root = msa_local_root / "paired"
    if paired_root.is_dir():
        ids.update(path.name for path in paired_root.iterdir() if path.is_dir())
    return sorted(ids)


def _validate_case(
    dataset_root: Path, msa_local_root: Path, case_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    total_pkl = dataset_root / "total_pkl" / f"{case_id}.pkl"
    gt_multimer = dataset_root / "gt_multimer" / f"{case_id}.pdb"
    reasons: list[str] = []
    if not total_pkl.is_file():
        reasons.append("missing_total_pkl")
    if not gt_multimer.is_file():
        reasons.append("missing_gt_multimer")
    if reasons:
        return None, {"case_id": case_id, "reasons": reasons}
    chain_sequences = (
        parse_pdb_chain_sequences(gt_multimer) if gt_multimer.is_file() else {}
    )
    if gt_multimer.is_file() and not chain_sequences:
        reasons.append("missing_protein_chains")

    chains: list[dict[str, Any]] = []
    for chain_id, sequence in chain_sequences.items():
        rep_id = default_rep_id(case_id, chain_id)
        main_msa_path = msa_local_root / "main" / f"{rep_id}.npz"
        paired_msa_path = msa_local_root / "paired" / case_id / f"{rep_id}.npz"
        if not main_msa_path.is_file():
            reasons.append(f"missing_main_msa:{chain_id}")
        if not paired_msa_path.is_file():
            reasons.append(f"missing_paired_msa:{chain_id}")
        row: dict[str, Any] = {
            "chain_id": chain_id,
            "sequence": sequence,
            "representative_id": rep_id,
            "main_msa_path": str(main_msa_path),
            "paired_msa_path": str(paired_msa_path),
        }
        if main_msa_path.is_file():
            row["main_msa"] = _file_info(main_msa_path)
        if paired_msa_path.is_file():
            row["paired_msa"] = _file_info(paired_msa_path)
        chains.append(row)

    if reasons:
        return None, {"case_id": case_id, "reasons": reasons}
    return {
        "case_id": case_id,
        "total_pkl": str(total_pkl),
        "gt_multimer": str(gt_multimer),
        "chains": chains,
        "validation_status": "ok",
    }, None


def build_frozen_case_set(
    *,
    dataset_root: Path,
    msa_local_root: Path,
    frozen_at: str,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    candidates = (
        sorted(case_ids)
        if case_ids is not None
        else _candidate_case_ids(dataset_root, msa_local_root)
    )
    cases: list[dict[str, Any]] = []
    excluded_cases: list[dict[str, Any]] = []
    for case_id in candidates:
        case, excluded = _validate_case(dataset_root, msa_local_root, case_id)
        if case is not None:
            cases.append(case)
        if excluded is not None:
            excluded_cases.append(excluded)
    return {
        "schema_version": 1,
        "frozen_at": frozen_at,
        "dataset_root": str(dataset_root),
        "msa_local_root": str(msa_local_root),
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "cases": cases,
        "excluded_cases": excluded_cases,
    }


def write_frozen_case_set(payload: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")


def _load_case_list(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"case list JSON must be a list: {path}")
        return [str(item) for item in payload]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--msa-local-root", type=Path, default=DEFAULT_MSA_LOCAL_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--case-list", type=Path)
    parser.add_argument(
        "--frozen-at", default=datetime.now().astimezone().isoformat(timespec="seconds")
    )
    args = parser.parse_args()
    payload = build_frozen_case_set(
        dataset_root=args.dataset_root,
        msa_local_root=args.msa_local_root,
        frozen_at=args.frozen_at,
        case_ids=_load_case_list(args.case_list) if args.case_list else None,
    )
    write_frozen_case_set(payload, args.output_json)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "case_count": payload["case_count"],
                "excluded_case_count": len(payload["excluded_cases"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
