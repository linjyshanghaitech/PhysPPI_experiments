#!/usr/bin/env python3
"""Monitor parallel MSA jobs, kill c4 when they finish, re-split remaining chunks.

Strategy:
1. Wait for active parallel jobs (PIDs from args or auto-detect) to exit
2. Kill the slow main c4 job
3. Identify remaining chunks that need MSA search
4. Launch new parallel jobs for remaining chunks (4-8 concurrent)
"""

import json
import os
import subprocess
import time
from pathlib import Path

MSA_CACHE = Path("output/physppi_full_msa_cache/colabfold_msas/chunks")
QUERY_JSON_DIR = MSA_CACHE / "query_json"
MSA_SETTINGS = "output/physppi_full_msa_cache/msa_server_full.yml"
C4_PID_FILE = "output/physppi_full_msa_cache/chunked_msa_parallel_c4.pid"
MAX_PARALLEL = 8  # Number of concurrent chunk jobs (balanced for API rate limits)


def check_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_chunk_status():
    """Return (done_ids, pending_ids) based on query_msa.json completeness."""
    pending = []
    for qf in sorted(QUERY_JSON_DIR.glob("chunk_*.json")):
        chunk_id = qf.stem
        with open(qf) as f:
            data = json.load(f)
        n_queries = len(data.get("queries", {}))

        mapping_file = MSA_CACHE / chunk_id / "colabfold_msas" / "query_msa.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                mapped = len(json.load(f))
        else:
            mapped = 0

        if mapped < n_queries:
            pending.append((chunk_id, n_queries, mapped))

    return pending


def launch_chunk(chunk_id: str, log_dir: Path) -> subprocess.Popen:
    """Launch MSA search for a single chunk."""
    query_json = QUERY_JSON_DIR / f"{chunk_id}.json"
    output_dir = MSA_CACHE / chunk_id / "colabfold_msas"
    log_file = log_dir / f"msa_{chunk_id}.log"

    cmd = [
        "/101045/jinyu/.pixi/bin/pixi",
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
        MSA_SETTINGS,
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watch-pids",
        type=int,
        nargs="*",
        default=[],
        help="PIDs to wait for before re-splitting",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("output/physppi_full_msa_cache/logs")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Check if parallel jobs are still running
    active = [pid for pid in args.watch_pids if check_pid_alive(pid)]

    if active:
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] Waiting for {len(active)} parallel jobs to finish: {active}")
        while active:
            time.sleep(60)  # Check every minute
            active = [pid for pid in args.watch_pids if check_pid_alive(pid)]
            if len(active) < len(args.watch_pids):
                done_count = len(args.watch_pids) - len(active)
                now = time.strftime("%H:%M:%S")
                print(
                    f"[{now}] {done_count} job(s) finished, "
                    f"{len(active)} still running: {active}"
                )

    print(f"[{time.strftime('%H:%M:%S')}] All parallel jobs finished!")

    # Step 2: Kill the c4 main job
    c4_pid_path = Path(C4_PID_FILE)
    if c4_pid_path.exists():
        c4_pid = int(c4_pid_path.read_text().strip())
        if check_pid_alive(c4_pid):
            print(
                f"[{time.strftime('%H:%M:%S')}] Killing c4 main job (PID {c4_pid})..."
            )
            try:
                os.kill(c4_pid, 15)  # SIGTERM
                time.sleep(5)
                if check_pid_alive(c4_pid):
                    os.kill(c4_pid, 9)  # SIGKILL
                print("  c4 job terminated")
            except Exception as e:
                print(f"  Error killing c4: {e}")
        else:
            print(f"  c4 job (PID {c4_pid}) already dead")
    else:
        print("  No c4 PID file found")

    # Step 3: Identify remaining chunks
    pending = get_chunk_status()
    print(
        f"\n[{time.strftime('%H:%M:%S')}] {len(pending)} chunks still need MSA search"
    )

    if not pending:
        print("All chunks complete!")
        return

    # Show top pending
    for chunk_id, nq, mapped in sorted(pending)[:10]:
        print(f"  {chunk_id}: {mapped}/{nq} queries mapped")
    if len(pending) > 10:
        print(f"  ... and {len(pending) - 10} more")

    if args.dry_run:
        print("DRY RUN: would launch parallel jobs. Exiting.")
        return

    # Step 4: Launch parallel jobs (MAX_PARALLEL at a time)
    now = time.strftime("%H:%M:%S")
    print(f"\n[{now}] Launching parallel jobs ({MAX_PARALLEL} concurrent)...")

    launched = []
    remaining = sorted(pending)  # Process in order

    while remaining:
        # Fill up to MAX_PARALLEL slots
        while len(launched) < MAX_PARALLEL and remaining:
            chunk_id, nq, mapped = remaining.pop(0)
            print(f"  Launching {chunk_id} ({mapped}/{nq} mapped)...")
            proc = launch_chunk(chunk_id, args.log_dir)
            launched.append((chunk_id, proc))
            time.sleep(5)  # Small delay between launches

        # Check if any launched jobs have finished
        still_running = []
        for chunk_id, proc in launched:
            if proc.poll() is None:
                still_running.append((chunk_id, proc))
            else:
                rc = proc.returncode
                print(f"  [{time.strftime('%H:%M:%S')}] {chunk_id} finished (rc={rc})")
                if rc != 0:
                    print(f"    Re-queuing {chunk_id} (non-zero exit)")
                    # Re-check this chunk's status and re-add to pending
                    for cid, nq, mapped in get_chunk_status():
                        if cid == chunk_id:
                            remaining.append((cid, nq, mapped))
                            break
        launched = still_running

        if launched:
            time.sleep(30)  # Wait before next check

    print(f"\n[{time.strftime('%H:%M:%S')}] All chunks processed!")


if __name__ == "__main__":
    main()
