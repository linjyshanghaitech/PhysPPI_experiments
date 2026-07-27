# ruff: noqa: E402
"""Ensure direct CLI execution imports the current repository checkout."""

import sys
from pathlib import Path


def ensure_repository_root() -> None:
    repository_root = str(Path(__file__).resolve().parents[2])
    if sys.path[0] != repository_root:
        sys.path.insert(0, repository_root)
