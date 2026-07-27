#!/usr/bin/env python3
# ruff: noqa: E402
"""Build immutable direct-eight-GPU PhysPPI smoke and formal YAMLs."""

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
import importlib.util
from pathlib import Path

import yaml


def _semantic_module():
    path = Path(__file__).with_name("prepare_semantic_50epoch.py")
    spec = importlib.util.spec_from_file_location("physppi_semantic_50epoch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load semantic config builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_stage1_configs(
    base: dict[str, object], manifest: str, version: str
) -> tuple[dict[str, object], dict[str, object]]:
    return _semantic_module().build_semantic_configs(
        base,
        manifest,
        version,
        manifest_sha256=None,
    )


def validate_stage1_config(config: dict[str, object], *, smoke: bool) -> None:
    data_module_args = config["data_module_args"]
    trainer = config["pl_trainer_args"]
    updates = (
        data_module_args["epoch_len"]
        // trainer["devices"]
        // trainer["accumulate_grad_batches"]
    )
    expected = 32 if smoke else 256
    if updates != expected:
        raise ValueError(f"expected {expected} optimizer updates, got {updates}")
    try:
        _semantic_module().validate_semantic_config(
            config,
            smoke=smoke,
            manifest_sha256=None,
        )
    except ValueError as exc:
        if "optimizer updates per epoch" in str(exc):
            raise ValueError(f"expected {expected} optimizer updates: {exc}") from exc
        raise


def _write_yaml(path: Path, config: dict[str, object]) -> None:
    _semantic_module().write_yaml(path, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    parser.add_argument("--formal-output", type=Path, required=True)
    args = parser.parse_args()

    baseline = yaml.safe_load(args.baseline.read_text())
    smoke, formal = build_stage1_configs(baseline, args.manifest, args.version)
    _write_yaml(args.smoke_output, smoke)
    _write_yaml(args.formal_output, formal)


if __name__ == "__main__":
    main()
