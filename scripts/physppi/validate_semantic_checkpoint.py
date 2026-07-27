# ruff: noqa: E402
"""Read-only validation for complete PhysPPI semantic epoch checkpoints."""

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
import hashlib
import math
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict


class SemanticCheckpointReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Path
    sha256: str
    size_bytes: int
    epoch: int | None
    global_step: int | None
    has_model: bool
    has_optimizer: bool
    has_scheduler: bool
    has_ema: bool
    has_loops: bool
    has_datamodule: bool
    manifest_sha256: str | None
    ema_decay: float | None
    scheduler_step: int | None
    epoch_only_boundary: bool
    finite_tensors: bool
    errors: tuple[str, ...]
    valid: bool


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _nonempty(value: object) -> bool:
    return isinstance(value, (dict, list, tuple)) and bool(value)


def _all_finite(value: object) -> bool:
    if torch.is_tensor(value):
        return not value.dtype.is_floating_point or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _scheduler_step(value: object) -> int | None:
    candidates: list[int] = []

    def collect(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"last_epoch", "_step_count", "step_count"} and isinstance(
                    child, (int, float)
                ):
                    candidates.append(int(child))
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    return max(candidates) if candidates else None


def _ema_decay(value: object) -> float | None:
    if not isinstance(value, dict) or "decay" not in value:
        return None
    decay = value["decay"]
    if torch.is_tensor(decay):
        if decay.numel() != 1:
            return None
        decay = decay.item()
    try:
        result = float(decay)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _manifest_hash(value: object) -> tuple[str | None, bool]:
    if not isinstance(value, dict):
        return None, False
    hashes = value.get("manifest_sha256_by_dataset")
    if not isinstance(hashes, dict) or not hashes:
        return None, False
    unique = {str(digest) for digest in hashes.values()}
    return (next(iter(unique)), True) if len(unique) == 1 else (None, False)


def validate_checkpoint(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_epoch: int | None = None,
    expected_global_step: int | None = None,
    updates_per_epoch: int = 256,
) -> SemanticCheckpointReport:
    """Load a checkpoint on CPU and validate full, resumable training state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")

    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    epoch = int(epoch) if isinstance(epoch, int) else None
    global_step = int(global_step) if isinstance(global_step, int) else None
    has_model = _nonempty(checkpoint.get("state_dict"))
    has_optimizer = _nonempty(checkpoint.get("optimizer_states"))
    has_scheduler = _nonempty(checkpoint.get("lr_schedulers"))
    has_ema = _nonempty(checkpoint.get("ema"))
    has_loops = _nonempty(checkpoint.get("loops"))
    has_datamodule = _nonempty(checkpoint.get("DataModule"))
    manifest_sha256, manifest_unambiguous = _manifest_hash(checkpoint.get("DataModule"))
    ema_decay = _ema_decay(checkpoint.get("ema"))
    scheduler_step = _scheduler_step(checkpoint.get("lr_schedulers"))
    finite_tensors = _all_finite(checkpoint)
    epoch_only_boundary = (
        epoch is not None
        and global_step is not None
        and global_step == (epoch + 1) * updates_per_epoch
    )

    errors: list[str] = []
    for present, code in (
        (has_model, "missing_model_state"),
        (has_optimizer, "missing_optimizer_state"),
        (has_scheduler, "missing_scheduler_state"),
        (has_ema, "missing_ema_state"),
        (has_loops, "missing_loop_state"),
        (has_datamodule, "missing_datamodule_state"),
    ):
        if not present:
            errors.append(code)
    if not finite_tensors:
        errors.append("nonfinite_tensor")
    if not manifest_unambiguous or manifest_sha256 != expected_manifest_sha256:
        errors.append("manifest_sha256_mismatch")
    if ema_decay is None or not math.isclose(
        ema_decay, 0.999, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append("ema_decay_mismatch")
    if (
        global_step is None
        or scheduler_step is None
        or scheduler_step not in {global_step, global_step + 1}
    ):
        errors.append("scheduler_step_mismatch")
    if not epoch_only_boundary:
        errors.append("not_epoch_boundary")
    if expected_epoch is not None and epoch != expected_epoch:
        errors.append("epoch_mismatch")
    if expected_global_step is not None and global_step != expected_global_step:
        errors.append("global_step_mismatch")

    return SemanticCheckpointReport(
        path=path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        epoch=epoch,
        global_step=global_step,
        has_model=has_model,
        has_optimizer=has_optimizer,
        has_scheduler=has_scheduler,
        has_ema=has_ema,
        has_loops=has_loops,
        has_datamodule=has_datamodule,
        manifest_sha256=manifest_sha256,
        ema_decay=ema_decay,
        scheduler_step=scheduler_step,
        epoch_only_boundary=epoch_only_boundary,
        finite_tensors=finite_tensors,
        errors=tuple(errors),
        valid=not errors,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--expected-global-step", type=int)
    parser.add_argument("--updates-per-epoch", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_checkpoint(
        args.checkpoint,
        expected_manifest_sha256=args.manifest_sha256,
        expected_epoch=args.expected_epoch,
        expected_global_step=args.expected_global_step,
        updates_per_epoch=args.updates_per_epoch,
    )
    print(report.model_dump_json(indent=2))
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
