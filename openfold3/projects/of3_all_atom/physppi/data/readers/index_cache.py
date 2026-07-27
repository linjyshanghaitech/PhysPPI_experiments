"""Process-safe persistent cache for validated PhysPPI reader indexes."""

from __future__ import annotations

import fcntl
import os
import pickle
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

IndexT = TypeVar("IndexT")
_CACHE_VERSION = 1
_LEGACY_MODULE_ALIASES = {
    "openfold3.projects.of3_all_atom.physppi.data.prior_complex": (
        "openfold3.projects.of3_all_atom.physppi.data.priors.schema"
    ),
    "openfold3.projects.of3_all_atom.physppi.data.optional_augmentation": (
        "openfold3.projects.of3_all_atom.physppi.data.priors.augmentation"
    ),
    "openfold3.projects.of3_all_atom.physppi.data.occurrence": (
        "openfold3.projects.of3_all_atom.physppi.data.priors.occurrence"
    ),
    "openfold3.projects.of3_all_atom.physppi.data.chain_views": (
        "openfold3.projects.of3_all_atom.physppi.data.adapters.chain_views"
    ),
}


class _ReaderIndexUnpickler(pickle.Unpickler):
    """Resolve only the known PhysPPI module moves in persisted indexes."""

    def find_class(self, module: str, name: str) -> object:
        current_module = _LEGACY_MODULE_ALIASES.get(module, module)
        return super().find_class(current_module, name)


def load_or_create_reader_index(
    cache_path: str | Path,
    metadata: Mapping[str, object],
    builder: Callable[[], IndexT],
) -> IndexT:
    """Load a matching index or build it once under an inter-process lock."""

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_metadata = dict(metadata)
    lock_path = path.with_name(f"{path.name}.lock")

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            try:
                with path.open("rb") as cache_file:
                    payload = _ReaderIndexUnpickler(cache_file).load()
                if (
                    payload.get("version") == _CACHE_VERSION
                    and payload.get("metadata") == expected_metadata
                ):
                    return payload["index"]
            except (
                AttributeError,
                EOFError,
                ImportError,
                OSError,
                pickle.UnpicklingError,
            ):
                pass

        index = builder()
        payload = {
            "version": _CACHE_VERSION,
            "metadata": expected_metadata,
            "index": index,
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                pickle.dump(payload, temporary_file, protocol=pickle.HIGHEST_PROTOCOL)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return index


__all__ = ["load_or_create_reader_index"]
