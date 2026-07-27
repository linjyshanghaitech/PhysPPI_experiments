from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PIXI_PYTHON = [
    "/101045/jinyu/.pixi/bin/pixi",
    "run",
    "-e",
    "openfold3-cpu",
    "python",
]


@dataclass(frozen=True)
class ProbeCommand:
    chunk_name: str
    query_json: Path
    output_dir: Path
    stdout_path: Path
    stderr_path: Path
    argv: list[str]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def select_chunk_query_paths(
    *,
    chunk_query_dir: Path,
    chunk_names: list[str],
) -> list[Path]:
    if not chunk_names:
        raise ValueError("at least one chunk name is required")
    paths: list[Path] = []
    for chunk_name in chunk_names:
        path = chunk_query_dir / f"{chunk_name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def build_probe_commands(
    *,
    chunk_query_paths: list[Path],
    output_dir: Path,
    msa_computation_settings_yaml: Path,
) -> list[ProbeCommand]:
    commands: list[ProbeCommand] = []
    log_dir = output_dir / "logs"
    for chunk_query in chunk_query_paths:
        chunk_name = chunk_query.stem
        chunk_output = output_dir / "chunks" / chunk_name / "colabfold_msas"
        commands.append(
            ProbeCommand(
                chunk_name=chunk_name,
                query_json=chunk_query,
                output_dir=chunk_output,
                stdout_path=log_dir / f"{chunk_name}.stdout",
                stderr_path=log_dir / f"{chunk_name}.stderr",
                argv=PIXI_PYTHON
                + [
                    "-m",
                    "openfold3.run_openfold",
                    "align-msa-server",
                    "--query-json",
                    str(chunk_query),
                    "--output-dir",
                    str(chunk_output),
                    "--msa-computation-settings-yaml",
                    str(msa_computation_settings_yaml),
                ],
            )
        )
    return commands


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_probe_command(
    *,
    command: ProbeCommand,
    status_dir: Path,
    timeout_seconds: int,
    launch_delay_seconds: float,
    execute: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    if launch_delay_seconds > 0:
        time.sleep(launch_delay_seconds)

    command.output_dir.mkdir(parents=True, exist_ok=True)
    command.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    status_path = status_dir / f"{command.chunk_name}.json"
    _write_json(
        status_path,
        {
            "chunk": command.chunk_name,
            "status": "running",
            "query_json": str(command.query_json),
            "output_dir": str(command.output_dir),
            "argv": command.argv,
            "started_at": started_at,
        },
    )

    try:
        with (
            command.stdout_path.open("w") as stdout,
            command.stderr_path.open("w") as stderr,
        ):
            completed = execute(
                command.argv,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                text=True,
            )
        return_code = int(completed.returncode)
        status = "passed" if return_code == 0 else "failed"
        error = None
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        status = "failed"
        error = f"timeout after {exc.timeout} seconds"

    finished_at = utc_now()
    report = {
        "chunk": command.chunk_name,
        "status": status,
        "query_json": str(command.query_json),
        "output_dir": str(command.output_dir),
        "stdout": str(command.stdout_path),
        "stderr": str(command.stderr_path),
        "argv": command.argv,
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if error is not None:
        report["error"] = error
    _write_json(status_path, report)
    return report


def run_parallel_probe(
    *,
    commands: list[ProbeCommand],
    output_dir: Path,
    concurrency: int,
    launch_delay_seconds: float,
    timeout_seconds: int,
    dry_run: bool = False,
    execute: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    status_dir = output_dir / "chunk_status"
    planned = [
        {
            "chunk": command.chunk_name,
            "query_json": str(command.query_json),
            "output_dir": str(command.output_dir),
            "stdout": str(command.stdout_path),
            "stderr": str(command.stderr_path),
            "argv": command.argv,
        }
        for command in commands
    ]

    if dry_run:
        report = {
            "status": "dry_run",
            "concurrency": concurrency,
            "launch_delay_seconds": launch_delay_seconds,
            "timeout_seconds": timeout_seconds,
            "chunks": planned,
            "started_at": utc_now(),
            "finished_at": utc_now(),
        }
        _write_json(output_dir / "parallel_msa_probe_report.json", report)
        return report

    started_at = utc_now()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for index, command in enumerate(commands):
            futures.append(
                pool.submit(
                    run_probe_command,
                    command=command,
                    status_dir=status_dir,
                    timeout_seconds=timeout_seconds,
                    launch_delay_seconds=index * launch_delay_seconds,
                    execute=execute,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda item: str(item["chunk"]))
    failed = [result for result in results if result.get("status") != "passed"]
    report = {
        "status": "passed" if not failed else "failed",
        "concurrency": concurrency,
        "launch_delay_seconds": launch_delay_seconds,
        "timeout_seconds": timeout_seconds,
        "started_at": started_at,
        "finished_at": utc_now(),
        "chunk_count": len(results),
        "failed_chunk_count": len(failed),
        "chunks": results,
    }
    _write_json(output_dir / "parallel_msa_probe_report.json", report)
    if failed:
        raise RuntimeError(f"parallel MSA probe failed chunks={len(failed)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-query-dir", type=Path, required=True)
    parser.add_argument("--msa-computation-settings-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk", dest="chunks", action="append", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--launch-delay-seconds", type=float, default=45.0)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunk_paths = select_chunk_query_paths(
        chunk_query_dir=args.chunk_query_dir,
        chunk_names=args.chunks,
    )
    commands = build_probe_commands(
        chunk_query_paths=chunk_paths,
        output_dir=args.output_dir,
        msa_computation_settings_yaml=args.msa_computation_settings_yaml,
    )
    report = run_parallel_probe(
        commands=commands,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        launch_delay_seconds=args.launch_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
