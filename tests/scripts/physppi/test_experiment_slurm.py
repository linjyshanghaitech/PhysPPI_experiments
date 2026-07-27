import errno
import hashlib
import json
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, call

import pytest

import scripts.physppi.experiment_slurm as experiment_slurm
from scripts.physppi.experiment_config import (
    ArtifactsSection,
    LoadedExperiment,
    load_experiment,
)
from scripts.physppi.experiment_slurm import (
    PreparedRun,
    build_command,
    git_state,
    prepare_run,
    render_sbatch,
    submit_run,
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

    output_dir = root / "outputs" / "run-one"
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


def _prepared_submission(
    root: Path,
    *,
    command: tuple[str, ...] = ("/usr/bin/python3", "-m", "example"),
    script: str = "#!/bin/bash\n",
) -> PreparedRun:
    run_dir = root / "prepared-run"
    run_dir.mkdir()
    script_path = run_dir / "job.sbatch"
    script_path.write_text(script)
    return PreparedRun(
        run_dir=run_dir,
        script_path=script_path,
        command=command,
    )


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
    config = loaded.config.model_copy(
        update={"artifacts": ArtifactsSection(output_dir=tmp_path / "safe-output")}
    )
    command = build_command(
        config,
        python_executable=Path("/opt/python env/bin/python"),
    )

    rendered = render_sbatch(
        config,
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


@pytest.mark.parametrize(
    "unsafe_job_name",
    [
        r"audit\run",
        'audit "run one"',
        "audit#comment",
        "audit\n#SBATCH --account=attacker",
    ],
)
def test_render_sbatch_rejects_unsafe_directive_atoms(
    tmp_path: Path,
    unsafe_job_name: str,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    config = loaded.config.model_copy(
        update={
            "slurm": loaded.config.slurm.model_copy(
                update={"job_name": unsafe_job_name}
            )
        }
    )

    with pytest.raises(ValueError, match="job-name"):
        render_sbatch(
            config,
            repo_root=tmp_path,
            python_executable=Path("/usr/bin/python3"),
        )


def test_render_sbatch_rejects_output_paths_with_spaces(tmp_path: Path) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    config = loaded.config.model_copy(
        update={
            "artifacts": ArtifactsSection(output_dir=tmp_path / "output with spaces")
        }
    )

    with pytest.raises(ValueError, match="output"):
        render_sbatch(
            config,
            repo_root=tmp_path,
            python_executable=Path("/usr/bin/python3"),
        )


def test_render_sbatch_emits_only_expected_safe_directives(tmp_path: Path) -> None:
    loaded = _write_loaded_experiment(tmp_path, all_resources=True)

    rendered = render_sbatch(
        loaded.config,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
    )

    directives = [line for line in rendered.splitlines() if line.startswith("#SBATCH")]
    assert len(directives) == 12
    assert all(
        "'" not in directive and '"' not in directive for directive in directives
    )
    assert directives[0] == "#SBATCH --job-name=audit-run"


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


def test_prepare_run_uses_toml_bytes_captured_during_load(tmp_path: Path) -> None:
    loaded = _write_loaded_experiment(tmp_path, local=True)
    experiment_bytes = loaded.experiment_source.content
    profile_bytes = loaded.profile_source.content
    assert loaded.local_source is not None
    assert loaded.experiment_source.path == loaded.experiment_path
    assert loaded.profile_source.path == loaded.profile_path
    assert loaded.local_source.path == loaded.local_path
    assert loaded.config.slurm.account == "secret-account"
    assert loaded.config.slurm.nodes == 3

    loaded.experiment_path.write_text("replaced experiment\n")
    loaded.profile_path.unlink()
    assert loaded.local_path is not None
    loaded.local_path.unlink()

    prepared = prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="3" * 40,
        git_dirty=False,
    )

    assert (prepared.run_dir / "experiment.toml").read_bytes() == experiment_bytes
    assert (prepared.run_dir / "slurm-profile.toml").read_bytes() == profile_bytes
    with (prepared.run_dir / "local-overrides.redacted.toml").open("rb") as handle:
        redacted = tomllib.load(handle)
    assert redacted["slurm"] == {
        "account": "<redacted>",
        "nodes": "<redacted>",
    }
    assert redacted["environment"]["export"] == {
        "API_TOKEN": "<redacted>",
        "EMPTY_SECRET": "<redacted>",
    }
    assert (
        "secret-account"
        not in (prepared.run_dir / "local-overrides.redacted.toml").read_text()
    )
    metadata = json.loads((prepared.run_dir / "prepared.json").read_text())
    assert (
        metadata["config_sha256"]["experiment.toml"]
        == hashlib.sha256(experiment_bytes).hexdigest()
    )
    assert (
        metadata["config_sha256"]["slurm-profile.toml"]
        == hashlib.sha256(profile_bytes).hexdigest()
    )
    assert metadata["config_sha256"]["local-overrides.redacted.toml"] == _sha256(
        prepared.run_dir / "local-overrides.redacted.toml"
    )


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


def test_prepare_run_surfaces_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    monkeypatch.setattr(
        experiment_slurm,
        "_write_json",
        Mock(side_effect=OSError("injected write failure")),
    )

    def fail_unless_errors_are_ignored(
        path: Path,
        *,
        ignore_errors: bool = False,
    ) -> None:
        if not ignore_errors:
            raise OSError(f"injected cleanup failure: {path}")

    monkeypatch.setattr(
        experiment_slurm.shutil,
        "rmtree",
        fail_unless_errors_are_ignored,
    )

    with pytest.raises(OSError, match="cleanup failure") as error:
        prepare_run(
            loaded,
            repo_root=tmp_path,
            python_executable=Path("/usr/bin/python3"),
            git_commit="7" * 40,
            git_dirty=False,
        )

    assert error.value.__cause__ is not None
    assert "write failure" in str(error.value.__cause__)


def test_prepare_run_does_not_replace_target_created_at_publish_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    output_dir = loaded.config.artifacts.output_dir
    output_parent = output_dir.parent
    real_fsync_directory = experiment_slurm._fsync_directory
    target_inode: int | None = None

    def create_competing_target(path: Path) -> None:
        nonlocal target_inode
        if path != output_parent:
            output_dir.mkdir()
            target_inode = output_dir.stat().st_ino
        real_fsync_directory(path)

    monkeypatch.setattr(
        experiment_slurm,
        "_fsync_directory",
        create_competing_target,
    )

    with pytest.raises(FileExistsError):
        prepare_run(
            loaded,
            repo_root=tmp_path,
            python_executable=Path("/usr/bin/python3"),
            git_commit="4" * 40,
            git_dirty=False,
        )

    assert target_inode is not None
    assert output_dir.is_dir()
    assert output_dir.stat().st_ino == target_inode
    assert list(output_dir.iterdir()) == []
    assert set(output_parent.iterdir()) == {output_dir}


def test_prepare_run_cleans_temp_when_temp_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    output_dir = loaded.config.artifacts.output_dir
    rename = Mock(side_effect=AssertionError("rename must not be attempted"))
    monkeypatch.setattr(experiment_slurm, "_rename_noreplace", rename)
    monkeypatch.setattr(
        experiment_slurm,
        "_fsync_directory",
        Mock(side_effect=OSError("injected directory fsync failure")),
    )

    with pytest.raises(OSError, match="injected directory fsync failure"):
        prepare_run(
            loaded,
            repo_root=tmp_path,
            python_executable=Path("/usr/bin/python3"),
            git_commit="5" * 40,
            git_dirty=False,
        )

    rename.assert_not_called()
    assert not output_dir.exists()
    assert output_dir.parent.is_dir()
    assert list(output_dir.parent.iterdir()) == []


def test_prepare_run_fsyncs_temp_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _write_loaded_experiment(tmp_path)
    output_dir = loaded.config.artifacts.output_dir
    events: list[tuple[str, Path, Path | None]] = []
    real_rename = experiment_slurm._rename_noreplace

    def record_rename(source: Path, destination: Path) -> None:
        events.append(("rename", source, destination))
        real_rename(source, destination)

    monkeypatch.setattr(
        experiment_slurm,
        "_fsync_directory",
        lambda path: events.append(("fsync", path, None)),
    )
    monkeypatch.setattr(experiment_slurm, "_rename_noreplace", record_rename)

    prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="6" * 40,
        git_dirty=False,
    )

    assert len(events) == 3
    assert events[0][0] == "fsync"
    assert events[0][1].parent == output_dir.parent
    assert events[0][1].name.startswith(f".{output_dir.name}.")
    assert events[1] == ("rename", events[0][1], output_dir)
    assert events[2] == ("fsync", output_dir.parent, None)


def test_rename_noreplace_fails_closed_when_syscall_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    monkeypatch.setattr(experiment_slurm, "_RENAMEAT2", None)

    with pytest.raises(OSError) as error:
        experiment_slurm._rename_noreplace(source, destination)

    assert error.value.errno == errno.ENOSYS
    assert source.is_dir()
    assert not destination.exists()


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


@pytest.mark.parametrize(
    ("stdout", "expected_job_id"),
    [
        ("12345", "12345"),
        ("12345;cluster\n", "12345"),
    ],
)
def test_submit_run_records_only_valid_job_id(
    tmp_path: Path,
    stdout: str,
    expected_job_id: str,
) -> None:
    prepared = _prepared_submission(tmp_path)
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=0,
            stdout=stdout,
            stderr="",
        )
    )

    job_id = submit_run(prepared, runner=runner)

    assert job_id == expected_job_id
    runner.assert_called_once_with(
        ["sbatch", "--parsable", str(prepared.script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    submitted = json.loads((prepared.run_dir / "submitted.json").read_text())
    assert submitted == {
        "schema_version": 1,
        "submitted_at_utc": submitted["submitted_at_utc"],
        "job_id": expected_job_id,
        "sbatch_sha256": _sha256(prepared.script_path),
    }
    assert submitted["submitted_at_utc"].endswith("Z")
    assert not (prepared.run_dir / "submission-error.json").exists()


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "not-a-job",
        "123 bad",
        "12345;cluster extra",
        "12345;cluster\nextra",
        "12345;cluster\0",
        "12345;cluster\x7f",
    ],
)
def test_submit_run_rejects_invalid_job_id(
    tmp_path: Path,
    stdout: str,
) -> None:
    prepared = _prepared_submission(tmp_path)
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=0,
            stdout=stdout,
            stderr="invalid response",
        )
    )

    with pytest.raises(RuntimeError):
        submit_run(prepared, runner=runner)

    assert not (prepared.run_dir / "submitted.json").exists()
    error = json.loads((prepared.run_dir / "submission-error.json").read_text())
    assert error["schema_version"] == 1
    assert error["failed_at_utc"].endswith("Z")
    assert error["error_type"] == "invalid_job_id"
    assert error["returncode"] == 0
    assert error["stderr"] == "invalid response"


def test_submit_run_records_nonzero_exit_without_submitted_state(
    tmp_path: Path,
) -> None:
    prepared = _prepared_submission(tmp_path)
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=2,
            stdout="12345\n",
            stderr="scheduler rejected request",
        )
    )

    with pytest.raises(RuntimeError):
        submit_run(prepared, runner=runner)

    assert not (prepared.run_dir / "submitted.json").exists()
    error = json.loads((prepared.run_dir / "submission-error.json").read_text())
    assert error["error_type"] == "nonzero_exit"
    assert error["returncode"] == 2
    assert error["stderr"] == "scheduler rejected request"


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("sbatch is missing"),
        PermissionError("sbatch is not executable"),
    ],
)
def test_submit_run_records_runner_os_error_without_submitted_state(
    tmp_path: Path,
    failure: OSError,
) -> None:
    prepared = _prepared_submission(tmp_path)
    runner = Mock(side_effect=failure)

    with pytest.raises(RuntimeError):
        submit_run(prepared, runner=runner)

    assert not (prepared.run_dir / "submitted.json").exists()
    error = json.loads((prepared.run_dir / "submission-error.json").read_text())
    assert error["error_type"] == type(failure).__name__
    assert error["returncode"] is None
    assert error["stderr"] == str(failure)


def test_submit_run_redacts_command_and_export_secrets_and_cleans_stderr(
    tmp_path: Path,
) -> None:
    command_secret = "command-secret-value"
    export_secret = "export-secret-value"
    prepared = _prepared_submission(
        tmp_path,
        command=("/usr/bin/python3", "--token", command_secret),
        script=(
            "#!/bin/bash\n"
            f"export API_TOKEN={shlex.quote(export_secret)}\n"
            "exec /usr/bin/python3\n"
        ),
    )
    stderr = (
        "x" * 9000
        + "\0before\x01"
        + command_secret
        + "\n"
        + export_secret
        + "\x7fafter"
    )
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=1,
            stdout="",
            stderr=stderr,
        )
    )

    with pytest.raises(RuntimeError):
        submit_run(prepared, runner=runner)

    serialized = (prepared.run_dir / "submission-error.json").read_text()
    error = json.loads(serialized)
    assert command_secret not in serialized
    assert export_secret not in serialized
    assert "\0" not in error["stderr"]
    assert "\x01" not in error["stderr"]
    assert "\x7f" not in error["stderr"]
    assert len(error["stderr"]) == 8192
    assert error["stderr"].endswith("before<redacted><redacted>after")


def test_submit_run_redaction_does_not_rewrite_its_placeholder(
    tmp_path: Path,
) -> None:
    prepared = _prepared_submission(
        tmp_path,
        command=("redacted",),
        script="#!/bin/bash\nexport TOKEN=prefix-redacted\n",
    )
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=1,
            stdout="",
            stderr="prefix-redacted redacted",
        )
    )

    with pytest.raises(RuntimeError):
        submit_run(prepared, runner=runner)

    error = json.loads((prepared.run_dir / "submission-error.json").read_text())
    assert error["stderr"] == "<redacted> <redacted>"


def test_submit_run_removes_controls_before_redacting_secrets(
    tmp_path: Path,
) -> None:
    prepared = _prepared_submission(
        tmp_path,
        command=("command-secret",),
    )
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=1,
            stdout="",
            stderr="failure: command-\0secret",
        )
    )

    with pytest.raises(RuntimeError):
        submit_run(prepared, runner=runner)

    serialized = (prepared.run_dir / "submission-error.json").read_text()
    assert "command-secret" not in serialized
    assert json.loads(serialized)["stderr"] == "failure: <redacted>"


def test_submit_run_fails_before_runner_when_already_submitted(
    tmp_path: Path,
) -> None:
    prepared = _prepared_submission(tmp_path)
    submitted_path = prepared.run_dir / "submitted.json"
    original = '{"job_id": "111"}\n'
    submitted_path.write_text(original)
    runner = Mock(side_effect=AssertionError("runner must not be called"))

    with pytest.raises(RuntimeError, match="already submitted"):
        submit_run(prepared, runner=runner)

    runner.assert_not_called()
    assert submitted_path.read_text() == original


def test_submit_run_serializes_concurrent_submission_attempts(
    tmp_path: Path,
) -> None:
    prepared = _prepared_submission(tmp_path)
    start = threading.Barrier(2)
    runner_calls = 0
    calls_lock = threading.Lock()

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal runner_calls
        with calls_lock:
            runner_calls += 1
        time.sleep(0.05)
        return subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=0,
            stdout="12345\n",
            stderr="",
        )

    def attempt() -> str:
        start.wait()
        try:
            return submit_run(prepared, runner=runner)
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))

    assert runner_calls == 1
    assert results.count("12345") == 1
    assert sum("already submitted" in result for result in results) == 1
    assert (
        json.loads((prepared.run_dir / "submitted.json").read_text())["job_id"]
        == "12345"
    )


def test_submit_run_atomically_replaces_status_and_fsyncs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_submission(tmp_path)
    error_path = prepared.run_dir / "submission-error.json"
    error_path.write_text('{"stale": true}\n')
    replacements: list[tuple[Path, Path]] = []
    fsyncs: list[Path] = []
    real_replace = experiment_slurm.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(experiment_slurm.os, "replace", record_replace)
    monkeypatch.setattr(
        experiment_slurm,
        "_fsync_directory",
        lambda path: fsyncs.append(path),
    )
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=0,
            stdout="54321\n",
            stderr="",
        )
    )

    assert submit_run(prepared, runner=runner) == "54321"

    assert len(replacements) == 1
    temporary, destination = replacements[0]
    assert temporary.parent == prepared.run_dir
    assert destination == prepared.run_dir / "submitted.json"
    assert fsyncs == [prepared.run_dir]
    assert not error_path.exists()
