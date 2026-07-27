#!/usr/bin/env python3
"""Build and optionally launch MSA jobs for missing broad test128 queries.

This fills only query-record coverage for the broad held-out evaluation. It does
not change training caches or existing fixed 32x32 evaluation inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

BASE = Path("/101045/jinyu/PhysPPI/openfold-3")
V4_SPLIT_BACKBONE = Path(
    "/101045/jinyu/PhysPPI/prior_processed/v4-Qbiolip-M/split_backbone"
)
DEFAULT_COVERAGE = (
    BASE
    / "output/physppi_cmap_only_research/broad_eval_inputs/"
    "query_coverage_report.json"
)
DEFAULT_OUTPUT = (
    BASE
    / "output/physppi_cmap_only_research/broad_eval_inputs/"
    "missing_test128_msa"
)
MSA_SETTINGS = BASE / "output/physppi_full_msa_cache/msa_server_full.yml"
PIXI = Path("/101045/jinyu/.pixi/bin/pixi")

AA3_TO_AA1 = {
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
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-report", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--launch-delay-sec", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def missing_ids(coverage_report: Path) -> list[str]:
    data = json.loads(coverage_report.read_text())
    ids = [
        row["sample_id"]
        for row in data.get("missing", [])
        if row.get("reason") == "missing_query"
    ]
    return sorted(dict.fromkeys(ids))


def sequence_from_pdb(path: Path) -> str:
    residues: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            residues.append(key + (resname,))
    seq = "".join(AA3_TO_AA1.get(resname, "X") for *_, resname in residues)
    return seq


def build_query(sample_id: str) -> dict:
    chain_files = sorted(V4_SPLIT_BACKBONE.glob(f"{sample_id}_*.pdb"))
    if not chain_files:
        raise FileNotFoundError(f"No split_backbone PDBs for {sample_id}")

    chains = []
    for idx, pdb_path in enumerate(chain_files, start=1):
        seq = sequence_from_pdb(pdb_path)
        if not seq:
            raise ValueError(f"Empty sequence parsed from {pdb_path}")
        chains.append(
            {
                "chain_ids": [str(idx)],
                "molecule_type": "protein",
                "sequence": seq,
            }
        )
    return {"chains": chains}


def write_chunks(ids: list[str], output_dir: Path, chunk_size: int) -> list[Path]:
    query_dir = output_dir / "query_json"
    query_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(ids), chunk_size):
        chunk_ids = ids[start : start + chunk_size]
        chunk_name = f"missing_test128_{start // chunk_size:04d}"
        payload = {"queries": {sid: build_query(sid) for sid in chunk_ids}}
        path = query_dir / f"{chunk_name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        chunks.append(path)
    return chunks


def launch_chunk(query_json: Path, output_dir: Path, log_dir: Path) -> subprocess.Popen:
    chunk_name = query_json.stem
    chunk_out = output_dir / "chunks" / chunk_name / "colabfold_msas"
    log_file = log_dir / f"{chunk_name}.log"
    cmd = [
        str(PIXI),
        "run",
        "-e",
        "openfold3-cpu",
        "python",
        "-m",
        "openfold3.run_openfold",
        "align-msa-server",
        "--query-json",
        str(query_json),
        "--output-dir",
        str(chunk_out),
        "--msa-computation-settings-yaml",
        str(MSA_SETTINGS),
    ]
    with log_file.open("w") as handle:
        return subprocess.Popen(
            cmd,
            cwd=BASE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def main() -> None:
    args = parse_args()
    ids = missing_ids(args.coverage_report)
    chunks = write_chunks(ids, args.output_dir, args.chunk_size)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "missing_count": len(ids),
        "chunk_count": len(chunks),
        "chunk_size": args.chunk_size,
        "max_parallel": args.max_parallel,
        "output_dir": str(args.output_dir),
        "query_jsons": [str(path) for path in chunks],
        "dry_run": bool(args.dry_run),
        "launch": bool(args.launch),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "launch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))

    if args.dry_run or not args.launch:
        return

    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, subprocess.Popen] = {}
    pending = list(chunks)
    failures: dict[str, int] = {}

    while pending or running:
        while pending and len(running) < args.max_parallel:
            query_json = pending.pop(0)
            proc = launch_chunk(query_json, args.output_dir, log_dir)
            running[query_json.stem] = proc
            print(f"[{datetime.now().isoformat(timespec='seconds')}] launched "
                  f"{query_json.stem} pid={proc.pid}", flush=True)
            time.sleep(args.launch_delay_sec)

        still_running = {}
        for name, proc in running.items():
            rc = proc.poll()
            if rc is None:
                still_running[name] = proc
                continue
            if rc != 0:
                failures[name] = rc
            print(f"[{datetime.now().isoformat(timespec='seconds')}] finished "
                  f"{name} rc={rc}", flush=True)
        running = still_running
        if pending or running:
            time.sleep(60)

    if failures:
        raise SystemExit(f"MSA chunks failed: {failures}")


if __name__ == "__main__":
    main()
