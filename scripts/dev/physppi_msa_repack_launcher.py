#!/usr/bin/env python3
"""Launch MSA search for the 4 repack chunks.

Re-split the 32 pending queries into 4 evenly-weighted chunks (~28,000 aa each).
Waits for ColabFold API to be available, then launches 2-4 concurrent MSA jobs.

Estimated runtime: 20-60 minutes once API is up (depending on API load and reliability).
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path("/101045/jinyu/PhysPPI/openfold-3")
MSA_CACHE = BASE / "output/physppi_full_msa_cache/colabfold_msas/chunks"
QUERY_JSON_DIR = MSA_CACHE / "query_json_repack"
MSA_SETTINGS = BASE / "output/physppi_full_msa_cache/msa_server_full.yml"
LOG_DIR = BASE / "output/physppi_full_msa_cache/logs"

MAX_WAIT_HOURS = 48  # Wait up to 48h for API
CHECK_INTERVAL = 300  # 5 minutes between API checks
MAX_PARALLEL = 4  # All 4 chunks can run concurrently (small total query count)

PIXI = "/101045/jinyu/.pixi/bin/pixi"


def check_api() -> tuple[bool, str]:
    """Return whether the ColabFold/MMseqs2 API host is reachable.

    The old /token probe returns 404 even when the service is reachable. Probe the
    real ticket endpoint used by the client; a GET without a submitted ticket
    returns 400 (invalid ID), which is a healthy API response for this check.
    """
    try:
        import requests

        response = requests.get("https://api.colabfold.com/ticket/msa", timeout=15)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if response.status_code < 500:
        return True, f"HTTP {response.status_code}: {response.text[:80]!r}"
    return False, f"HTTP {response.status_code}: {response.text[:80]!r}"


def launch_chunk(chunk_id: str) -> subprocess.Popen:
    """Launch MSA search for a single repack chunk."""
    query_json = QUERY_JSON_DIR / f"{chunk_id}.json"
    output_dir = MSA_CACHE / chunk_id / "colabfold_msas"
    log_file = LOG_DIR / f"msa_repack_{chunk_id}.log"

    cmd = [
        PIXI,
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
        str(output_dir),
        "--msa-computation-settings-yaml",
        str(MSA_SETTINGS),
    ]

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Collect repack chunks
    chunks = sorted(
        [
            qf.stem
            for qf in QUERY_JSON_DIR.glob("repack_*.json")
            if qf.stem != "repack_summary"
        ]
    )
    if not chunks:
        print("No repack chunks found — nothing to do.")
        return

    total_queries = 0
    for chunk_id in chunks:
        with open(QUERY_JSON_DIR / f"{chunk_id}.json") as f:
            total_queries += len(json.load(f).get("queries", {}))

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Repack launcher starting")
    print(f"  Chunks to process: {chunks}")
    print(f"  Total queries: {total_queries}")

    # Wait for API
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Waiting for ColabFold API...")
    start = datetime.now()

    while True:
        elapsed = (datetime.now() - start).total_seconds() / 3600
        if elapsed > MAX_WAIT_HOURS:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Timeout after {MAX_WAIT_HOURS}h — giving up")
            sys.exit(1)

        api_ok, api_status = check_api()
        if api_ok:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] API reachable: {api_status}")
            break

        now = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{now}] API down ({api_status}), "
            f"elapsed={elapsed:.1f}h, retrying in {CHECK_INTERVAL}s..."
        )
        time.sleep(CHECK_INTERVAL)

    api_up_time = datetime.now()
    wait_time = (api_up_time - start).total_seconds() / 3600
    print(f"[{api_up_time.strftime('%H:%M:%S')}] API is UP! (waited {wait_time:.1f}h)")

    # Launch all chunks in parallel
    now = datetime.now().strftime("%H:%M:%S")
    print(
        f"\n[{now}] Launching {len(chunks)} repack chunks "
        f"({MAX_PARALLEL} concurrent)..."
    )

    procs = {}
    for chunk_id in chunks:
        print(f"  Launching {chunk_id}...")
        proc = launch_chunk(chunk_id)
        procs[chunk_id] = proc
        time.sleep(3)  # Small stagger between launches

    # Monitor progress
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring {len(procs)} jobs...")
    print("  Estimated runtime: 20-60 minutes from now")
    print(f"  Logs: {LOG_DIR}/msa_repack_repack_*.log")
    print(f"  ETA: {datetime.now().strftime('%H:%M')} + 20-60 min")

    while procs:
        still_running = {}
        for chunk_id, proc in procs.items():
            rc = proc.poll()
            if rc is None:
                still_running[chunk_id] = proc
            else:
                elapsed = (datetime.now() - api_up_time).total_seconds() / 60
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                now = datetime.now().strftime("%H:%M:%S")
                print(f"  [{now}] {chunk_id} finished: {status} (t+{elapsed:.0f}min)")

        procs = still_running
        if procs:
            time.sleep(60)  # Check every minute

    total_time = (datetime.now() - api_up_time).total_seconds() / 60
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] All repack chunks completed in {total_time:.0f} minutes!")

    # Verify completion
    print("\nVerifying results...")
    all_ok = True
    for chunk_id in chunks:
        output_file = MSA_CACHE / chunk_id / "colabfold_msas" / "query_msa.json"
        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
            n_done = len(data.get("queries", {}))
            print(f"  {chunk_id}: {n_done} queries in output")
        else:
            print(f"  {chunk_id}: NO OUTPUT FILE!")
            all_ok = False

    if all_ok:
        print("\n✓ All repack chunks completed successfully!")
    else:
        print("\n⚠ Some chunks may have failed — check logs")


if __name__ == "__main__":
    main()
