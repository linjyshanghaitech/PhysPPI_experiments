import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from scripts.physppi.experiment_config import LoadedExperiment, load_experiment
from scripts.physppi.experiment_slurm import (
    build_command,
    git_state,
    prepare_run,
    render_sbatch,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


def _write_loaded_experiment(
    root: Path,
    *,
    local: bool = False,
    runner_yaml: bool = False,
    all_resources: bool = False,
) -> LoadedExperiment:
    profile = root / "profile.toml"
    profile_lines = [
        "schema_version = 1",
        "[slurm]",
        'partition = "compute"',
        "nodes = 2",
        "cpus_per_task = 8",
        'memory = "32G"',
        'time = "02:00:00"',
    ]
    if all_resources:
        profile_lines.extend(
            [
                'account = "research"',
                'qos = "normal"',
                "ntasks_per_node = 4",
                "gpus_per_node = 1",
            ]
        )
    profile.write_text("\n".join(profile_lines) + "\n")

    runner = root / "runner config.yaml"
    if runner_yaml:
        runner.write_text("trainer:\n  precision: bf16\n")

    output_dir = root / "outputs" / "run one"
    experiment_lines = [
        "schema_version = 1",
        "[experiment]",
        'name = "audit"',
        'run_name = "run-one"',
        'module = "tests.fixtures.fake_experiment"',
        'args = ["--label", "value with spaces", "--literal", "$HOME;echo nope"]',
        "[slurm]",
        'profile = "profile.toml"',
        'job_name = "audit-run"',
        "[artifacts]",
        f'output_dir = "{output_dir}"',
        "[environment]",
        'export = { SAFE = "plain", TRICKY = "it\'s $HOME; touch nope" }',
    ]
    if runner_yaml:
        experiment_lines.extend(
            [
                "[openfold]",
                f'runner_yaml = "{runner}"',
            ]
        )
    experiment = root / "source experiment.toml"
    experiment.write_text("\n".join(experiment_lines) + "\n")

    local_path = None
    environ = {}
    if local:
        local_path = root / "private local.toml"
        local_path.write_text(
            "schema_version = 1\n"
            "[slurm]\n"
            'account = "secret-account"\n'
            "nodes = 3\n"
            "[environment]\n"
            'required = ["TOKEN", "SECOND_TOKEN"]\n'
            "[environment.export]\n"
            'API_TOKEN = "top-secret"\n'
            'EMPTY_SECRET = ""\n'
        )
        environ = {"TOKEN": "present", "SECOND_TOKEN": "also-present"}

    return load_experiment(
        experiment,
        repo_root=root,
        local_path=local_path,
        environ=environ,
        validate_module=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_command_uses_python_module_and_exact_args(tmp_path: Path) -> None:
    loaded = _write_loaded_experiment(tmp_path)

    command = build_command(
        loaded.config,
        python_executable=Path("/opt/python env/bin/python"),
    )

    assert command == (
        "/opt/python env/bin/python",
        "-m",
        "tests.fixtures.fake_experiment",
        "--label",
        "value with spaces",
        "--literal",
        "$HOME;echo nope",
    )


def test_render_sbatch_is_deterministic_and_orders_directives(
    tmp_path: Path,
) -> None:
    loaded = _write_loaded_experiment(tmp_path, all_resources=True)

    first = render_sbatch(
        loaded.config,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
    )
    second = render_sbatch(
        loaded.config,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
    )

    assert first == second
    directive_names = [
        line.split("--", 1)[1].split("=", 1)[0]
        for line in first.splitlines()
        if line.startswith("#SBATCH")
    ]
    assert directive_names == [
        "job-name",
        "partition",
        "account",
        "qos",
        "nodes",
        "ntasks-per-node",
        "cpus-per-task",
        "gpus-per-node",
        "mem",
        "time",
        "output",
        "error",
    ]
    assert first.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in first
    assert "shell=True" not in first
    assert first.endswith("\n")


def test_render_sbatch_omits_none_directives_and_targets_artifact_logs(
    tmp_path: Path,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)

    rendered = render_sbatch(
        loaded.config,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
    )

    assert "#SBATCH --account=" not in rendered
    assert "#SBATCH --qos=" not in rendered
    assert "#SBATCH --ntasks-per-node=" not in rendered
    assert "#SBATCH --gpus-per-node=" not in rendered
    assert (
        "#SBATCH --output="
        + shlex.quote(str(loaded.config.artifacts.output_dir / "stdout.log"))
        in rendered
    )
    assert (
        "#SBATCH --error="
        + shlex.quote(str(loaded.config.artifacts.output_dir / "stderr.log"))
        in rendered
    )


def test_render_sbatch_quotes_repo_exports_and_command(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo with spaces"
    repo_root.mkdir()
    loaded = _write_loaded_experiment(repo_root)
    command = build_command(
        loaded.config,
        python_executable=Path("/opt/python env/bin/python"),
    )

    rendered = render_sbatch(
        loaded.config,
        repo_root=repo_root,
        python_executable=Path("/opt/python env/bin/python"),
    )

    assert f"cd {shlex.quote(str(repo_root))}" in rendered
    assert "export SAFE=plain" in rendered
    assert (
        f"export TRICKY={shlex.quote(loaded.config.environment.export['TRICKY'])}"
        in rendered
    )
    assert f"exec {shlex.join(command)}" in rendered
    assert "\ntouch nope\n" not in rendered


def test_prepare_run_writes_exact_prepared_files_without_submission(
    tmp_path: Path,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)

    prepared = prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="a" * 40,
        git_dirty=False,
    )

    assert prepared.run_dir == loaded.config.artifacts.output_dir
    assert prepared.script_path == prepared.run_dir / "job.sbatch"
    assert set(path.name for path in prepared.run_dir.iterdir()) == {
        "experiment.toml",
        "slurm-profile.toml",
        "command.json",
        "job.sbatch",
        "prepared.json",
    }
    assert (prepared.run_dir / "experiment.toml").read_bytes() == (
        loaded.experiment_path.read_bytes()
    )
    assert (prepared.run_dir / "slurm-profile.toml").read_bytes() == (
        loaded.profile_path.read_bytes()
    )
    assert not (prepared.run_dir / "submitted.json").exists()
    assert json.loads((prepared.run_dir / "command.json").read_text()) == list(
        prepared.command
    )


def test_prepare_run_rejects_an_existing_output_directory(tmp_path: Path) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    loaded.config.artifacts.output_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        prepare_run(
            loaded,
            repo_root=tmp_path,
            python_executable=Path("/usr/bin/python3"),
            git_commit="b" * 40,
            git_dirty=True,
        )

    assert list(loaded.config.artifacts.output_dir.iterdir()) == []


def test_prepare_run_redacts_every_local_scalar_and_preserves_containers(
    tmp_path: Path,
) -> None:
    loaded = _write_loaded_experiment(tmp_path, local=True)

    prepared = prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="c" * 40,
        git_dirty=False,
    )

    redacted_path = prepared.run_dir / "local-overrides.redacted.toml"
    with redacted_path.open("rb") as handle:
        redacted = tomllib.load(handle)
    assert redacted == {
        "schema_version": "<redacted>",
        "slurm": {
            "account": "<redacted>",
            "nodes": "<redacted>",
        },
        "environment": {
            "required": ["<redacted>", "<redacted>"],
            "export": {
                "API_TOKEN": "<redacted>",
                "EMPTY_SECRET": "<redacted>",
            },
        },
    }
    text = redacted_path.read_text()
    for secret in ["secret-account", "top-secret", '"TOKEN"', '"SECOND_TOKEN"']:
        assert secret not in text


def test_prepare_run_omits_local_archive_when_no_local_config(
    tmp_path: Path,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)

    prepared = prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="d" * 40,
        git_dirty=False,
    )

    assert not (prepared.run_dir / "local-overrides.redacted.toml").exists()


def test_prepare_run_records_runner_yaml_hash_without_copying_yaml(
    tmp_path: Path,
) -> None:
    loaded = _write_loaded_experiment(tmp_path, runner_yaml=True)
    assert loaded.config.openfold is not None

    prepared = prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="e" * 40,
        git_dirty=False,
    )

    expected = _sha256(loaded.config.openfold.runner_yaml)
    assert (prepared.run_dir / "runner-yaml.sha256").read_text() == expected + "\n"
    assert not any(
        path.suffix in {".yaml", ".yml"} for path in prepared.run_dir.iterdir()
    )


def test_prepare_run_metadata_records_hashes_git_argv_and_utc(
    tmp_path: Path,
) -> None:
    loaded = _write_loaded_experiment(tmp_path, local=True, runner_yaml=True)
    assert loaded.config.openfold is not None

    prepared = prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="f" * 40,
        git_dirty=True,
    )
    metadata = json.loads((prepared.run_dir / "prepared.json").read_text())

    assert metadata["schema_version"] == 1
    assert metadata["prepared_at_utc"].endswith("Z")
    assert metadata["git_commit"] == "f" * 40
    assert metadata["git_dirty"] is True
    assert metadata["command"] == list(prepared.command)
    assert metadata["config_sha256"] == {
        "experiment.toml": _sha256(prepared.run_dir / "experiment.toml"),
        "local-overrides.redacted.toml": _sha256(
            prepared.run_dir / "local-overrides.redacted.toml"
        ),
        "runner-yaml": _sha256(loaded.config.openfold.runner_yaml),
        "slurm-profile.toml": _sha256(prepared.run_dir / "slurm-profile.toml"),
    }
    assert metadata["sbatch_sha256"] == _sha256(prepared.run_dir / "job.sbatch")
    assert (prepared.run_dir / "prepared.json").read_text().endswith("\n")
    assert (prepared.run_dir / "command.json").read_text().endswith("\n")


def test_prepare_run_cleans_temp_directory_after_injected_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    output_parent = loaded.config.artifacts.output_dir.parent

    monkeypatch.setattr(
        "scripts.physppi.experiment_slurm._write_json",
        Mock(side_effect=OSError("injected write failure")),
    )

    with pytest.raises(OSError, match="injected write failure"):
        prepare_run(
            loaded,
            repo_root=tmp_path,
            python_executable=Path("/usr/bin/python3"),
            git_commit="0" * 40,
            git_dirty=False,
        )

    assert not loaded.config.artifacts.output_dir.exists()
    assert output_parent.is_dir()
    assert list(output_parent.iterdir()) == []


def test_git_state_uses_argv_subprocess_calls(tmp_path: Path) -> None:
    runner = Mock(
        side_effect=[
            subprocess.CompletedProcess(
                args=["git", "rev-parse", "HEAD"],
                returncode=0,
                stdout="1" * 40 + "\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "status", "--porcelain"],
                returncode=0,
                stdout=" M tracked.txt\n",
                stderr="",
            ),
        ]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("scripts.physppi.experiment_slurm.subprocess.run", runner)
        state = git_state(tmp_path)

    assert state == ("1" * 40, True)
    assert runner.call_args_list == [
        call(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ),
        call(
            ["git", "status", "--porcelain"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ),
    ]


def test_prepare_run_never_invokes_sbatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    subprocess_run = Mock(side_effect=AssertionError("must not invoke subprocess"))
    monkeypatch.setattr(
        "scripts.physppi.experiment_slurm.subprocess.run",
        subprocess_run,
    )

    prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="2" * 40,
        git_dirty=False,
    )

    subprocess_run.assert_not_called()
