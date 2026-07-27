# ruff: noqa: E402
"""Strict read-only monitor for an owned PhysPPI semantic training run."""

from __future__ import annotations

if __package__:
    from scripts.physppi._bootstrap import ensure_repository_root
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import ensure_repository_root

ensure_repository_root()

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_IGNORED_LINE_PATTERNS = (
    re.compile(r"DeprecationWarning:.*Python 3\.14", re.IGNORECASE),
    re.compile(r"AccumulateGrad.*stream.*warning", re.IGNORECASE),
)
_STRICT_PATTERNS = {
    "nan": re.compile(r"\bnan\b", re.IGNORECASE),
    "nonfinite": re.compile(r"non[- ]?finite", re.IGNORECASE),
    "traceback": re.compile(r"traceback", re.IGNORECASE),
    "oom": re.compile(
        r"cuda.*out of memory|outofmemoryerror|\boom\b",
        re.IGNORECASE,
    ),
    "nccl": re.compile(
        r"nccl.*(?:error|fail|timeout|abort)",
        re.IGNORECASE,
    ),
    "dataloader": re.compile(
        r"dataloader.*(?:error|fail|worker)",
        re.IGNORECASE,
    ),
    "template": re.compile(
        r"template.*(?:skip|broadcast|error)|broadcast.*(?:template|error)",
        re.IGNORECASE,
    ),
    "cuda": re.compile(r"cuda (?:error|exception)", re.IGNORECASE),
    "scattergather": re.compile(r"scattergather", re.IGNORECASE),
    "token_center": re.compile(
        r"invalid[- ]token[- ]center|token center atom index is out of bounds",
        re.IGNORECASE,
    ),
    "surface": re.compile(r"physppi_surface_local_coords", re.IGNORECASE),
}


class StrictLogScan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    clean: bool
    matches: tuple[str, ...]
    signatures: tuple[str, ...]


def strict_log_scan(text: str) -> StrictLogScan:
    """Reject every configured numeric/data/communication signature."""
    matches: list[str] = []
    signatures: set[str] = set()
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in _IGNORED_LINE_PATTERNS):
            continue
        for name, pattern in _STRICT_PATTERNS.items():
            if pattern.search(line):
                matches.append(line)
                signatures.add(name)
                break
    return StrictLogScan(
        clean=not matches,
        matches=tuple(matches),
        signatures=tuple(sorted(signatures)),
    )


def _atomic_json(path: Path, payload: object) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _owned_process(run_dir: Path) -> dict[str, object]:
    evidence = json.loads((run_dir / "evidence" / "launcher-process.json").read_text())
    pid = int(evidence["child_pid"])
    pgid = int(evidence["pgid"])
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return {"active": False, "pid": pid, "pgid": pgid}
    actual_pgid = os.getpgid(pid)
    cwd = proc.joinpath("cwd").resolve()
    command = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode()
    expected_command = tuple(str(part) for part in evidence["command"])
    expected_repository = Path(expected_command[0].split("/.pixi/", 1)[0]).resolve()
    owned = (
        actual_pgid == pgid
        and "run_openfold train" in command
        and all(part in command for part in expected_command[1:])
        and cwd == expected_repository
        and Path(evidence["log_path"]).parent.resolve() == run_dir.resolve()
    )
    return {
        "active": True,
        "owned": owned,
        "pid": pid,
        "pgid": actual_pgid,
        "cwd": str(cwd),
        "command": command,
    }


def _gpu_telemetry() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    telemetry = []
    for line in result.stdout.splitlines():
        index, used, free, utilization = (
            int(field.strip()) for field in line.split(",", 3)
        )
        telemetry.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "utilization_percent": utilization,
            }
        )
    return telemetry


def _sampler_progress(run_dir: Path) -> dict[str, object]:
    roots = (
        run_dir / "logs" / "evidence" / "sampler",
        run_dir / "evidence" / "sampler",
    )
    root = next((candidate for candidate in roots if candidate.is_dir()), roots[0])
    counts: dict[str, int] = {}
    latest_rank_zero: dict[str, object] | None = None
    for path in sorted(root.glob("rank-*.jsonl")) if root.is_dir() else ():
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        counts[path.stem] = len(rows)
        if path.name == "rank-0.jsonl" and rows:
            latest_rank_zero = rows[-1]
    return {"rows_by_rank": counts, "latest_rank_zero": latest_rank_zero}


def monitor_run(run_dir: Path, *, terminal: bool = False) -> dict[str, object]:
    """Validate ownership, scan new log bytes, and snapshot GPU/progress state."""
    evidence_dir = run_dir / "evidence"
    state_path = evidence_dir / "monitor-state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    log_path = run_dir / "train.log"
    prior_offset = int(state.get("log_offset", 0))
    size = log_path.stat().st_size
    offset = 0 if terminal or size < prior_offset else prior_offset
    with log_path.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode(errors="replace")
    scan = strict_log_scan(text)
    process = _owned_process(run_dir)
    if process.get("active") and not process.get("owned"):
        raise RuntimeError("active process no longer matches launcher ownership")
    report: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "terminal_full_scan": terminal,
        "log_offset_start": offset,
        "log_offset_end": size,
        "strict_scan": scan.model_dump(mode="json"),
        "process": process,
        "sampler_progress": _sampler_progress(run_dir),
        "gpus": _gpu_telemetry(),
    }
    _atomic_json(state_path, {"log_offset": size})
    _atomic_json(evidence_dir / "monitor-latest.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--terminal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = monitor_run(args.run_dir, terminal=args.terminal)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["strict_scan"]["clean"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
