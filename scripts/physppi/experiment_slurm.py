from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from scripts.physppi.experiment_config import ExperimentConfig, LoadedExperiment

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


_SLURM_DIRECTIVES = (
    ("job-name", "job_name"),
    ("partition", "partition"),
    ("account", "account"),
    ("qos", "qos"),
    ("nodes", "nodes"),
    ("ntasks-per-node", "ntasks_per_node"),
    ("cpus-per-task", "cpus_per_task"),
    ("gpus-per-node", "gpus_per_node"),
    ("mem", "memory"),
    ("time", "time"),
)
_LOCAL_SLURM_FIELDS = (
    "partition",
    "account",
    "qos",
    "nodes",
    "ntasks_per_node",
    "cpus_per_task",
    "gpus_per_node",
    "memory",
    "time",
)
_SBATCH_SAFE_ATOM = re.compile(r"[A-Za-z0-9_./:@%+=,-]+\Z")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    script_path: Path
    command: tuple[str, ...]


def build_command(
    config: ExperimentConfig,
    *,
    python_executable: Path,
) -> tuple[str, ...]:
    return (
        str(python_executable),
        "-m",
        config.experiment.module,
        *config.experiment.args,
    )


def _render_sbatch_directive(name: str, value: object) -> str:
    rendered = str(value)
    if _SBATCH_SAFE_ATOM.fullmatch(rendered) is None:
        raise ValueError(f"unsafe Slurm directive value for {name}")
    return f"#SBATCH --{name}={rendered}"


def render_sbatch(
    config: ExperimentConfig,
    *,
    repo_root: Path,
    python_executable: Path,
) -> str:
    output_dir = config.artifacts.output_dir
    directives: list[tuple[str, object]] = []
    for directive_name, field_name in _SLURM_DIRECTIVES:
        value = getattr(config.slurm, field_name)
        if value is not None:
            directives.append((directive_name, value))
    directives.extend(
        [
            ("output", output_dir / "stdout.log"),
            ("error", output_dir / "stderr.log"),
        ]
    )

    lines = ["#!/usr/bin/env bash"]
    lines.extend(_render_sbatch_directive(name, value) for name, value in directives)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
        ]
    )
    lines.extend(
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(config.environment.export.items())
    )
    lines.append(
        f"exec {shlex.join(build_command(config, python_executable=python_executable))}"
    )
    return "\n".join(lines) + "\n"


def git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return commit.stdout.strip(), bool(status.stdout.strip())


def redact_scalar_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: redact_scalar_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_scalar_tree(item) for item in value]
    return "<redacted>"


def _render_redacted_value(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_render_redacted_value(item) for item in value) + "]"
    return json.dumps(value)


def render_redacted_local(raw: Mapping[str, object]) -> str:
    redacted = redact_scalar_tree(raw)
    if not isinstance(redacted, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError("local override root must be a mapping")

    lines: list[str] = []
    if "schema_version" in redacted:
        lines.append(
            f"schema_version = {_render_redacted_value(redacted['schema_version'])}"
        )

    slurm = redacted.get("slurm")
    if isinstance(slurm, dict):
        if lines:
            lines.append("")
        lines.append("[slurm]")
        for name in _LOCAL_SLURM_FIELDS:
            if name in slurm:
                lines.append(f"{name} = {_render_redacted_value(slurm[name])}")

    environment = redacted.get("environment")
    if isinstance(environment, dict):
        if lines:
            lines.append("")
        lines.append("[environment]")
        if "required" in environment:
            lines.append(
                "required = " + _render_redacted_value(environment["required"])
            )

        exported = environment.get("export")
        if isinstance(exported, dict):
            lines.extend(["", "[environment.export]"])
            for name in sorted(exported):
                lines.append(f"{name} = {_render_redacted_value(exported[name])}")

    return "\n".join(lines) + "\n"


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, content: str) -> None:
    _write_bytes(path, content.encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")

    ctypes.set_errno(0)
    result = _RENAMEAT2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_run(
    loaded: LoadedExperiment,
    *,
    repo_root: Path,
    python_executable: Path,
    git_commit: str,
    git_dirty: bool,
) -> PreparedRun:
    output_dir = loaded.config.artifacts.output_dir
    if output_dir.exists():
        raise FileExistsError(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        experiment_archive = temporary_dir / "experiment.toml"
        profile_archive = temporary_dir / "slurm-profile.toml"
        _write_bytes(experiment_archive, loaded.experiment_source.content)
        _write_bytes(profile_archive, loaded.profile_source.content)

        config_sha256 = {
            "experiment.toml": _sha256(experiment_archive),
            "slurm-profile.toml": _sha256(profile_archive),
        }

        if loaded.local_path is not None:
            if loaded.local_source is None:
                raise ValueError("local source snapshot is missing")
            raw_local = tomllib.load(BytesIO(loaded.local_source.content))
            local_archive = temporary_dir / "local-overrides.redacted.toml"
            _write_text(local_archive, render_redacted_local(raw_local))
            config_sha256["local-overrides.redacted.toml"] = _sha256(local_archive)

        if loaded.config.openfold is not None:
            runner_sha256 = _sha256(loaded.config.openfold.runner_yaml)
            _write_text(
                temporary_dir / "runner-yaml.sha256",
                runner_sha256 + "\n",
            )
            config_sha256["runner-yaml"] = runner_sha256

        command = build_command(
            loaded.config,
            python_executable=python_executable,
        )
        _write_json(temporary_dir / "command.json", command)

        sbatch = render_sbatch(
            loaded.config,
            repo_root=repo_root,
            python_executable=python_executable,
        )
        script_path = temporary_dir / "job.sbatch"
        _write_text(script_path, sbatch)
        _write_json(
            temporary_dir / "prepared.json",
            {
                "schema_version": 1,
                "prepared_at_utc": _utc_now(),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "config_sha256": config_sha256,
                "command": command,
                "sbatch_sha256": _sha256(script_path),
            },
        )

        _fsync_directory(temporary_dir)
        _rename_noreplace(temporary_dir, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException as failure:
        if temporary_dir.exists():
            try:
                shutil.rmtree(temporary_dir)
            except BaseException as cleanup_failure:
                raise cleanup_failure from failure
        raise

    return PreparedRun(
        run_dir=output_dir,
        script_path=output_dir / "job.sbatch",
        command=command,
    )


__all__ = [
    "PreparedRun",
    "build_command",
    "git_state",
    "prepare_run",
    "redact_scalar_tree",
    "render_redacted_local",
    "render_sbatch",
]
