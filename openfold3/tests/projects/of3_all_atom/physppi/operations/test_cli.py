"""Import and CLI boundaries for supported PhysPPI operations."""

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
CLI_PATHS = (
    "scripts/physppi/screen_full_dataset.py",
    "scripts/physppi/prepare_semantic_50epoch.py",
    "scripts/physppi/launch_semantic_training.py",
    "scripts/physppi/monitor_semantic_training.py",
    "scripts/physppi/validate_semantic_checkpoint.py",
)


@pytest.mark.parametrize("relative_path", CLI_PATHS)
def test_supported_cli_help_succeeds(relative_path: str) -> None:
    result = subprocess.run(
        [sys.executable, relative_path, "--help"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
