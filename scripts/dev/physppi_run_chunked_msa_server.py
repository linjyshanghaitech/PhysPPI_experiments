from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PIXI_PYTHON = [
    "/101045/jinyu/.pixi/bin/pixi",
    "run",
    "-e",
    "openfold3-cpu",
    "python",
]
MAPPING_FILENAMES = [
    "rep_id_to_seq.json",
    "seq_to_rep_id.json",
    "chain_id_to_rep_id.json",
    "query_name_to_complex_id.json",
]


@dataclass(frozen=True)
class ChunkCommand:
    chunk_name: str
    query_json: Path
    output_dir: Path
    argv: list[str]


def split_query_payload(query_payload: dict, max_queries_per_chunk: int) -> list[dict]:
    if max_queries_per_chunk <= 0:
        raise ValueError("max_queries_per_chunk must be positive")
    queries = query_payload.get("queries")
    if not isinstance(queries, dict) or not queries:
        raise ValueError("query payload must contain a non-empty queries object")

    items = list(queries.items())
    chunks: list[dict] = []
    for start in range(0, len(items), max_queries_per_chunk):
        chunk_queries = dict(items[start : start + max_queries_per_chunk])
        chunk = {key: value for key, value in query_payload.items() if key != "queries"}
        chunk["queries"] = chunk_queries
        chunks.append(chunk)
    return chunks


def write_query_chunks(
    *,
    query_payload: dict,
    chunk_query_dir: Path,
    max_queries_per_chunk: int,
) -> list[Path]:
    chunk_query_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_query_payload(query_payload, max_queries_per_chunk)
    paths: list[Path] = []
    for index, chunk in enumerate(chunks):
        path = chunk_query_dir / f"chunk_{index:04d}.json"
        path.write_text(json.dumps(chunk, indent=2) + "\n")
        paths.append(path)
    return paths


def _status_path(status_dir: Path, chunk_name: str) -> Path:
    return status_dir / f"{chunk_name}.json"


def _read_status(status_dir: Path, chunk_name: str) -> dict | None:
    path = _status_path(status_dir, chunk_name)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _chunk_output_complete(output_dir: Path) -> bool:
    mappings_dir = output_dir / "mappings"
    main_dir = output_dir / "main"
    return (
        all(
            (mappings_dir / mapping_name).is_file()
            for mapping_name in MAPPING_FILENAMES
        )
        and main_dir.is_dir()
        and any(main_dir.glob("*.npz"))
    )


def _write_status(status_dir: Path, chunk_name: str, payload: dict) -> Path:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = _status_path(status_dir, chunk_name)
    path.write_text(json.dumps({"chunk": chunk_name, **payload}, indent=2) + "\n")
    return path


def run_chunk_command(
    *,
    command: ChunkCommand,
    status_dir: Path,
    execute=subprocess.call,
    force: bool = False,
) -> dict:
    previous = _read_status(status_dir, command.chunk_name)
    if (
        previous is not None
        and previous.get("status") == "passed"
        and _chunk_output_complete(command.output_dir)
        and not force
    ):
        return {"chunk": command.chunk_name, "status": "skipped"}

    partial_raw = command.output_dir / "raw"
    if partial_raw.exists() and not force:
        shutil.rmtree(partial_raw)

    return_code = int(execute(command.argv))
    status = "passed" if return_code == 0 else "failed"
    report = {
        "status": status,
        "query_json": str(command.query_json),
        "output_dir": str(command.output_dir),
        "argv": command.argv,
        "return_code": return_code,
    }
    _write_status(status_dir, command.chunk_name, report)
    if return_code != 0:
        raise RuntimeError(
            f"MSA chunk failed: {command.chunk_name} return_code={return_code}"
        )
    return {"chunk": command.chunk_name, **report}


def _run_chunk_command_with_delay(
    *,
    command: ChunkCommand,
    status_dir: Path,
    launch_delay_seconds: float,
    execute,
    force: bool,
) -> dict:
    if launch_delay_seconds > 0:
        time.sleep(launch_delay_seconds)
    return run_chunk_command(
        command=command,
        status_dir=status_dir,
        execute=execute,
        force=force,
    )


def run_chunk_commands_parallel(
    *,
    commands: list[ChunkCommand],
    status_dir: Path,
    concurrency: int,
    launch_delay_seconds: float,
    execute=subprocess.call,
    force: bool = False,
) -> list[dict]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if launch_delay_seconds < 0:
        raise ValueError("launch_delay_seconds must be non-negative")

    results_by_chunk: dict[str, dict] = {}
    pending_commands: list[ChunkCommand] = []
    for command in commands:
        previous = _read_status(status_dir, command.chunk_name)
        if (
            previous is not None
            and previous.get("status") == "passed"
            and _chunk_output_complete(command.output_dir)
            and not force
        ):
            results_by_chunk[command.chunk_name] = {
                "chunk": command.chunk_name,
                "status": "skipped",
            }
        else:
            pending_commands.append(command)

    if concurrency == 1:
        for command in pending_commands:
            results_by_chunk[command.chunk_name] = run_chunk_command(
                command=command,
                status_dir=status_dir,
                execute=execute,
                force=force,
            )
        return [results_by_chunk[command.chunk_name] for command in commands]

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_chunk_command_with_delay,
                command=command,
                status_dir=status_dir,
                launch_delay_seconds=index * launch_delay_seconds,
                execute=execute,
                force=force,
            ): command.chunk_name
            for index, command in enumerate(pending_commands)
        }
        for future in as_completed(futures):
            chunk_name = futures[future]
            try:
                results_by_chunk[chunk_name] = future.result()
            except Exception:
                failures.append(chunk_name)

    if failures:
        failed = ", ".join(sorted(failures))
        raise RuntimeError(f"MSA chunks failed: {failed}")
    return [results_by_chunk[command.chunk_name] for command in commands]


def _load_mapping(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"mapping file is not a JSON object: {path}")
    return data


def _merge_mapping(
    *,
    merged: dict,
    update: dict,
    mapping_name: str,
    chunk_dir: Path,
) -> None:
    for key, value in update.items():
        if key in merged and merged[key] != value:
            raise ValueError(
                "mapping conflict "
                f"{mapping_name}:{key} from {chunk_dir}: "
                f"{merged[key]!r} != {value!r}"
            )
        merged[key] = value


def _copy_main_npzs(chunk_dir: Path, output_dir: Path) -> tuple[int, int]:
    source_main = chunk_dir / "main"
    if not source_main.is_dir():
        raise FileNotFoundError(source_main)
    target_main = output_dir / "main"
    target_main.mkdir(parents=True, exist_ok=True)
    copied = 0
    duplicates = 0
    for source in sorted(source_main.glob("*.npz")):
        target = target_main / source.name
        if target.exists():
            duplicates += 1
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied, duplicates


def merge_chunk_outputs(*, chunk_output_dirs: list[Path], output_dir: Path) -> dict:
    if not chunk_output_dirs:
        raise ValueError("no chunk outputs to merge")
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    merged_mappings = {name: {} for name in MAPPING_FILENAMES}
    main_npz_count = 0
    duplicate_main_npz_count = 0
    for chunk_dir in chunk_output_dirs:
        for mapping_name in MAPPING_FILENAMES:
            update = _load_mapping(chunk_dir / "mappings" / mapping_name)
            _merge_mapping(
                merged=merged_mappings[mapping_name],
                update=update,
                mapping_name=mapping_name,
                chunk_dir=chunk_dir,
            )
        copied, duplicates = _copy_main_npzs(chunk_dir, output_dir)
        main_npz_count += copied
        duplicate_main_npz_count += duplicates

    for mapping_name, payload in merged_mappings.items():
        (mappings_dir / mapping_name).write_text(json.dumps(payload, indent=2) + "\n")
    (mappings_dir / "README.md").write_text(
        "Merged from chunked PhysPPI MSA outputs.\n"
    )

    report = {
        "status": "passed",
        "chunk_count": len(chunk_output_dirs),
        "mapping_counts": {
            name: len(payload) for name, payload in merged_mappings.items()
        },
        "main_npz_count": main_npz_count,
        "duplicate_main_npz_count": duplicate_main_npz_count,
    }
    (output_dir / "chunked_msa_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def build_chunk_commands(
    *,
    chunk_query_paths: list[Path],
    output_dir: Path,
    msa_computation_settings_yaml: Path,
) -> list[ChunkCommand]:
    commands: list[ChunkCommand] = []
    for chunk_query in chunk_query_paths:
        chunk_name = chunk_query.stem
        chunk_output = output_dir / "chunks" / chunk_name / "colabfold_msas"
        commands.append(
            ChunkCommand(
                chunk_name=chunk_name,
                query_json=chunk_query,
                output_dir=chunk_output,
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


def run_chunked_msa(
    *,
    query_json: Path,
    output_dir: Path,
    msa_computation_settings_yaml: Path,
    max_queries_per_chunk: int,
    force_chunk: bool = False,
    concurrency: int = 1,
    launch_delay_seconds: float = 0,
    execute=subprocess.call,
) -> dict:
    query_payload = json.loads(query_json.read_text())
    chunk_query_paths = write_query_chunks(
        query_payload=query_payload,
        chunk_query_dir=output_dir / "chunks" / "query_json",
        max_queries_per_chunk=max_queries_per_chunk,
    )
    commands = build_chunk_commands(
        chunk_query_paths=chunk_query_paths,
        output_dir=output_dir,
        msa_computation_settings_yaml=msa_computation_settings_yaml,
    )
    status_dir = output_dir / "chunk_status"
    chunk_reports = run_chunk_commands_parallel(
        commands=commands,
        status_dir=status_dir,
        concurrency=concurrency,
        launch_delay_seconds=launch_delay_seconds,
        execute=execute,
        force=force_chunk,
    )
    merge_report = merge_chunk_outputs(
        chunk_output_dirs=[command.output_dir for command in commands],
        output_dir=output_dir,
    )
    report = {
        "status": "passed",
        "query_json": str(query_json),
        "output_dir": str(output_dir),
        "max_queries_per_chunk": max_queries_per_chunk,
        "concurrency": concurrency,
        "launch_delay_seconds": launch_delay_seconds,
        "chunk_count": len(commands),
        "chunks": chunk_reports,
        "merge": merge_report,
    }
    (output_dir / "chunked_msa_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msa-computation-settings-yaml", type=Path, required=True)
    parser.add_argument("--max-queries-per-chunk", type=int, default=16)
    parser.add_argument("--force-chunk", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--launch-delay-seconds", type=float, default=0)
    args = parser.parse_args()

    report = run_chunked_msa(
        query_json=args.query_json,
        output_dir=args.output_dir,
        msa_computation_settings_yaml=args.msa_computation_settings_yaml,
        max_queries_per_chunk=args.max_queries_per_chunk,
        force_chunk=args.force_chunk,
        concurrency=args.concurrency,
        launch_delay_seconds=args.launch_delay_seconds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
