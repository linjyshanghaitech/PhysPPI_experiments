#!/usr/bin/env python3
"""Wait for ColabFold API to become available, then launch MSA resplit."""

import subprocess
import sys
import time
from datetime import datetime

MAX_WAIT_HOURS = 24
CHECK_INTERVAL = 300  # 5 minutes

start = datetime.now()
print(f"[{start.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for ColabFold API...")

while True:
    elapsed = (datetime.now() - start).total_seconds() / 3600
    if elapsed > MAX_WAIT_HOURS:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Timeout after {MAX_WAIT_HOURS}h — giving up")
        sys.exit(1)

    try:
        import requests

        r = requests.get("https://api.colabfold.com/ticket/msa", timeout=15)
        if r.status_code < 500:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] API reachable: HTTP {r.status_code}")
            break
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] API server returned {r.status_code}, waiting...")
    except Exception as e:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] API unreachable ({type(e).__name__}), elapsed={elapsed:.1f}h")

    time.sleep(CHECK_INTERVAL)

print(f"[{datetime.now().strftime('%H:%M:%S')}] API is UP! Launching MSA resplit...")

result = subprocess.run(
    [
        "/101045/jinyu/.pixi/bin/pixi",
        "run",
        "-e",
        "openfold3-cpu",
        "python",
        "-u",
        "scripts/dev/physppi_msa_resplit.py",
        "--log-dir",
        "output/physppi_full_msa_cache/logs",
    ],
    capture_output=False,
)

now = datetime.now().strftime("%H:%M:%S")
print(f"[{now}] MSA resplit exited with code {result.returncode}")
sys.exit(result.returncode)
