"""Completeness checks for the PhysPPI test and script disposition."""

import json
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
DISPOSITION = (
    REPOSITORY_ROOT / "docs/physppi/refactor/test-and-script-disposition.json"
)
ALLOWED_ACTIONS = {"delete", "move", "retain"}


def test_disposition_is_complete_unique_and_resolved() -> None:
    payload = json.loads(DISPOSITION.read_text(encoding="utf-8"))
    baseline = payload["baseline_commit"]
    entries = payload["entries"]
    paths = [entry["path"] for entry in entries]

    baseline_paths = {
        path
        for path in subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", baseline],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).splitlines()
        if (
            path.startswith("openfold3/tests/test_physppi")
            and path.endswith(".py")
        )
        or path.startswith("scripts/dev/physppi")
    }

    assert len(paths) == len(set(paths))
    assert set(paths) == baseline_paths
    assert {entry["action"] for entry in entries} <= ALLOWED_ACTIONS
    for entry in entries:
        assert entry["kind"] in {"script", "test"}
        assert entry["reason"]
        if entry["action"] in {"move", "retain"}:
            assert (REPOSITORY_ROOT / entry["target"]).exists()
        else:
            assert not (REPOSITORY_ROOT / entry["path"]).exists()
            assert entry["replacement"]
