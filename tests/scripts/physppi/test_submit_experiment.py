from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import scripts.physppi.submit_experiment as submit_experiment
from scripts.physppi.experiment_slurm import PreparedRun


@pytest.fixture
def cli_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, Mock, Mock, Mock, PreparedRun]:
    loaded = object()
    prepared = PreparedRun(
        run_dir=tmp_path / "run",
        script_path=tmp_path / "run" / "job.sbatch",
        command=("python", "-m", "example"),
    )
    load = Mock(return_value=loaded)
    state = Mock(return_value=("a" * 40, True))
    prepare = Mock(return_value=prepared)
    submit = Mock(return_value="12345")
    monkeypatch.setattr(submit_experiment, "load_experiment", load)
    monkeypatch.setattr(submit_experiment, "git_state", state)
    monkeypatch.setattr(submit_experiment, "prepare_run", prepare)
    monkeypatch.setattr(submit_experiment, "submit_run", submit)
    return load, state, prepare, submit, prepared


def test_cli_defaults_to_dry_run(
    cli_dependencies: tuple[Mock, Mock, Mock, Mock, PreparedRun],
    capsys: pytest.CaptureFixture[str],
) -> None:
    load, state, prepare, submit, prepared = cli_dependencies

    assert submit_experiment.main(["experiment.toml"]) == 0

    repo_root = Path(submit_experiment.__file__).resolve().parents[2]
    load.assert_called_once_with(
        Path("experiment.toml"),
        repo_root=repo_root,
        local_path=None,
    )
    state.assert_called_once_with(repo_root)
    prepare.assert_called_once_with(
        load.return_value,
        repo_root=repo_root,
        python_executable=Path(sys.executable),
        git_commit="a" * 40,
        git_dirty=True,
    )
    submit.assert_not_called()
    assert capsys.readouterr().out == f"{prepared.script_path}\n"


def test_cli_submits_only_with_explicit_flag_and_passes_local_path(
    cli_dependencies: tuple[Mock, Mock, Mock, Mock, PreparedRun],
    capsys: pytest.CaptureFixture[str],
) -> None:
    load, _, _, submit, prepared = cli_dependencies

    assert (
        submit_experiment.main(
            [
                "experiment.toml",
                "--local-config",
                "private.toml",
                "--submit",
            ]
        )
        == 0
    )

    assert load.call_args.kwargs["local_path"] == Path("private.toml")
    submit.assert_called_once_with(prepared)
    assert capsys.readouterr().out == f"{prepared.script_path}\n12345\n"


def test_cli_direct_execution_help_has_no_import_error() -> None:
    script = Path(submit_experiment.__file__).resolve()

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "ImportError" not in result.stderr


def test_cli_parser_has_no_environment_or_config_submission_switch(
    cli_dependencies: tuple[Mock, Mock, Mock, Mock, PreparedRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, submit, _ = cli_dependencies
    monkeypatch.setenv("PHYS_PPI_SUBMIT", "1")

    assert submit_experiment.main(["experiment.toml"]) == 0

    submit.assert_not_called()
