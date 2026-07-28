#!/usr/bin/env python3
"""Extract per-chain sequences from AFDB + Q-BioLiP training data for MMseqs2 clustering.

Produces:
  - sequences.fasta          : one entry per chain, header = ``>{case_id}|{chain_id}``
  - chain_index.json         : mapping from seq_id to (dataset, case_id, chain_id, seq_len)
  - case_chains.json         : mapping from case_id to list of chain_ids

All outputs land in --output-dir (default: experiments/split_training_val/).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

DATASET_DIRS = {
    "afdb": Path("/101045/jinyu/PPI/Prior_out/training/AFDB-10k-1"),
    "qbiolip": Path("/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new"),
}


def _write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def extract_sequences(
    dataset_dirs: dict[str, Path],
    output_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Extract chain sequences and return (chain_index, case_chains).

    chain_index: ``{seq_id: {"dataset": ..., "case_id": ..., "chain_id": ..., "seq_len": ...}}``
    case_chains: ``{case_id: [chain_id, ...]}``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / "sequences.fasta"
    chain_index: dict[str, dict[str, str]] = {}
    case_chains: dict[str, list[str]] = {}

    total_cases = 0
    total_chains = 0

    with fasta_path.open("w", encoding="ascii", newline="\n") as fh:
        for dataset_name, dataset_dir in sorted(dataset_dirs.items()):
            chunk_dir = dataset_dir / "chunk"
            if not chunk_dir.is_dir():
                print(f"WARNING: {chunk_dir} not found, skipping {dataset_name}")
                continue

            for chunk_path in sorted(chunk_dir.glob("chunk*")):
                seq_cache_path = chunk_path / "seq_cache.pkl"
                if not seq_cache_path.is_file():
                    print(f"WARNING: {seq_cache_path} not found")
                    continue

                with seq_cache_path.open("rb") as handle:
                    seq_cache: dict[str, dict[str, Any]] = pickle.load(handle)

                for case_id in sorted(seq_cache):
                    chains = seq_cache[case_id]
                    total_cases += 1
                    case_chains[case_id] = []

                    for chain_id in sorted(chains):
                        chain_data = chains[chain_id]
                        seq_1aa = chain_data.get("full_seq_1aa", "")
                        if not seq_1aa:
                            seq_3aa = chain_data.get("full_seq_3aa", [])
                            # Fallback: 3-letter to 1-letter (shouldn't happen)
                            seq_1aa = "".join(
                                {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
                                 "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
                                 "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                                 "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
                                 "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
                                 "MSE": "M", "ASX": "B", "GLX": "Z", "UNK": "X"}
                                .get(aa, "X") for aa in seq_3aa
                            )

                        seq_id = f"{case_id}|{chain_id}"
                        fh.write(f">{seq_id}\n{seq_1aa}\n")

                        chain_index[seq_id] = {
                            "dataset": dataset_name,
                            "case_id": case_id,
                            "chain_id": chain_id,
                            "seq_len": len(seq_1aa),
                        }
                        case_chains[case_id].append(chain_id)
                        total_chains += 1

    print(f"Extracted {total_chains} chains from {total_cases} cases ({len(chain_index)} unique chain IDs)")

    # Verify uniqueness
    if len(chain_index) != total_chains:
        raise ValueError(f"Duplicate chain IDs: {total_chains} entries but {len(chain_index)} unique")

    _write_json(output_dir / "chain_index.json", chain_index)
    _write_json(output_dir / "case_chains.json", case_chains)

    return chain_index, case_chains


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for FASTA and JSON index files",
    )
    parser.add_argument(
        "--dataset-dirs",
        type=Path,
        nargs=2,
        default=None,
        help="Override dataset directories: AFDB_DIR QBIOLIP_DIR",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset_dirs = dict(DATASET_DIRS)
    if args.dataset_dirs is not None:
        dataset_dirs = {
            "afdb": args.dataset_dirs[0],
            "qbiolip": args.dataset_dirs[1],
        }

    extract_sequences(dataset_dirs, args.output_dir)
    print(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
