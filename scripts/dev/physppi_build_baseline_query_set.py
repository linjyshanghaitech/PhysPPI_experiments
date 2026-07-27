#!/usr/bin/env python3
"""Build OpenFold3 query JSON files from PDB files for a PhysPPI dataset split.

Reads a split manifest (JSON), extracts protein chain sequences from the
corresponding PDB files, and produces OpenFold3-compatible query JSON files
suitable for inference.

Outputs:
    <output_dir>/<split>_set_manifest.json   -- per-sample status (ok/skipped)
    <output_dir>/query_<split>.json           -- full query for all ok samples
    <output_dir>/query_<split>_small.json     -- first --limit queries (or all if
                                                 --limit >= total)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from Bio.PDB import PDBParser

# ---------------------------------------------------------------------------
# 3-to-1 letter amino acid mapping
# ---------------------------------------------------------------------------
AA3TO1: dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


# ---------------------------------------------------------------------------
# PDB parsing helpers
# ---------------------------------------------------------------------------
def extract_chain_sequences(pdb_path: Path) -> dict[str, str] | None:
    """Extract protein chain sequences from a PDB file.

    Only ATOM records with standard amino-acid residue names (see AA3TO1)
    are kept.  Residues are deduplicated by (chain_id, resseq, icode).

    Returns
    -------
    dict mapping chain_id -> one-letter-sequence, or *None* on parse failure.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", str(pdb_path))
    except Exception:
        return None

    chain_residues: dict[str, dict[tuple[int, str], str]] = {}
    for model in structure:
        for chain in model:
            cid = chain.get_id()
            if cid not in chain_residues:
                chain_residues[cid] = {}
            for residue in chain:
                # Skip HETATMs and water
                if residue.get_id()[0] != " ":
                    continue
                resname = residue.get_resname().strip().upper()
                one = AA3TO1.get(resname)
                if one is None:
                    continue
                # (hetero_flag, resseq, icode) is the full residue id
                _het, resseq, icode = residue.get_id()
                key = (resseq, icode)
                chain_residues[cid][key] = one

    if not chain_residues:
        return None

    sequences: dict[str, str] = {}
    for cid in sorted(chain_residues.keys()):
        residues = chain_residues[cid]
        if not residues:
            continue
        ordered = "".join(
            residues[k] for k in sorted(residues.keys(), key=lambda x: (x[0], x[1]))
        )
        if ordered:
            sequences[cid] = ordered

    return sequences if sequences else None


def group_identical_sequences(
    sequences: dict[str, str],
) -> list[dict[str, Any]]:
    """Group chains with identical sequences together.

    Returns a list of entries suitable for the OpenFold3 query "chains" list:
        [{"molecule_type": "protein",
          "chain_ids": [...],
          "sequence": "..."}, ...]
    """
    seq_to_chains: dict[str, list[str]] = {}
    for cid, seq in sequences.items():
        seq_to_chains.setdefault(seq, []).append(cid)

    chains_out: list[dict[str, Any]] = []
    for seq, cids in seq_to_chains.items():
        chains_out.append(
            {
                "molecule_type": "protein",
                "chain_ids": sorted(cids),
                "sequence": seq,
            }
        )
    # Stable ordering: shortest sequence first, then alphabetical
    chains_out.sort(key=lambda x: (len(x["sequence"]), x["sequence"], x["chain_ids"]))
    return chains_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OpenFold3 baseline query set from PhysPPI split manifest."
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Path to the v3_split_manifest_clean.json (or similar).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split key to read (test, train, valid).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to process (for small test runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Read manifest
    # -----------------------------------------------------------------------
    if not args.split_manifest.exists():
        print(
            f"ERROR: split manifest not found: {args.split_manifest}",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest_raw = json.loads(args.split_manifest.read_text())
    pdb_root = Path(manifest_raw["pdb_path"])
    pkl_root = Path(manifest_raw["pkl_path"])

    sample_ids: list[str] = manifest_raw.get(args.split, [])
    if not sample_ids:
        print(
            f"ERROR: split '{args.split}' not found or empty in manifest",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit is not None and args.limit < len(sample_ids):
        sample_ids = sample_ids[: args.limit]

    print(f"split={args.split}  total_samples={len(sample_ids)}")

    # -----------------------------------------------------------------------
    # Process each sample
    # -----------------------------------------------------------------------
    queries: dict[str, Any] = {}
    manifest_entries: list[dict[str, Any]] = []

    for idx, sample_id in enumerate(sample_ids, 1):
        pdb_path = pdb_root / f"{sample_id}.pdb"
        pkl_path = pkl_root / f"{sample_id}.pkl"

        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "pdb_path": str(pdb_path),
            "pkl_path": str(pkl_path),
            "status": "ok",
            "errors": [],
        }

        # --- Check PDB exists ---
        if not pdb_path.exists():
            entry["status"] = "skipped"
            entry["errors"].append("PDB file not found")
            manifest_entries.append(entry)
            continue

        # --- PKL existence is informational only ---
        if not pkl_path.exists():
            entry["status"] = "skipped"
            entry["errors"].append("PKL file not found")
            manifest_entries.append(entry)
            continue

        # --- Extract sequences from PDB ---
        try:
            sequences = extract_chain_sequences(pdb_path)
        except Exception as exc:
            entry["status"] = "skipped"
            entry["errors"].append(f"Sequence extraction error: {exc}")
            manifest_entries.append(entry)
            continue

        if sequences is None:
            entry["status"] = "skipped"
            entry["errors"].append("No protein chains extracted (empty or non-protein)")
            manifest_entries.append(entry)
            continue

        # Verify no chain has an empty sequence
        empty_chains = [cid for cid, seq in sequences.items() if not seq]
        if empty_chains:
            entry["status"] = "skipped"
            entry["errors"].append(
                f"Chain(s) with empty sequences: {sorted(empty_chains)}"
            )
            manifest_entries.append(entry)
            continue

        # --- Add metadata ---
        entry["chain_ids"] = sorted(sequences.keys())
        entry["sequence_lengths"] = {
            cid: len(seq) for cid, seq in sorted(sequences.items())
        }

        # --- Build query entry ---
        chains_out = group_identical_sequences(sequences)
        queries[sample_id] = {"chains": chains_out}

        manifest_entries.append(entry)

        if idx % 100 == 0 or idx == len(sample_ids):
            print(f"  processed {idx}/{len(sample_ids)}")

    # -----------------------------------------------------------------------
    # Summarise
    # -----------------------------------------------------------------------
    total = len(manifest_entries)
    ok_count = sum(1 for e in manifest_entries if e["status"] == "ok")
    skipped_count = sum(1 for e in manifest_entries if e["status"] == "skipped")
    print(f"\nResults: {total} total, {ok_count} ok, {skipped_count} skipped")

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest_out = args.output_dir / f"{args.split}_set_manifest.json"
    manifest_out.write_text(json.dumps(manifest_entries, indent=2, default=str) + "\n")
    print(f"manifest={manifest_out}")

    # Full query
    query_out = args.output_dir / f"query_{args.split}.json"
    query_out.write_text(json.dumps({"queries": queries}, indent=2, default=str) + "\n")
    print(f"query_full={query_out}")

    # Small query (first --limit or first 5)
    small_limit = (
        args.limit if args.limit is not None and args.limit < len(queries) else 5
    )
    small_queries = dict(list(queries.items())[:small_limit])
    query_small_out = args.output_dir / f"query_{args.split}_small.json"
    query_small_out.write_text(
        json.dumps({"queries": small_queries}, indent=2, default=str) + "\n"
    )
    print(f"query_small={query_small_out}")


if __name__ == "__main__":
    main()
