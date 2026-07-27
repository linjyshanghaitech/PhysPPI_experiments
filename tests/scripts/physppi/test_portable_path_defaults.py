import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "physppi" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(
        f"_portable_{name}",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_semantic_paths_have_portable_home_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENFOLD3_CHECKPOINT", raising=False)
    monkeypatch.delenv("PHYSPPI_COMPONENTS_CIF", raising=False)

    module = _load_script("prepare_semantic_50epoch")

    assert str(Path.home() / ".openfold3/of3-p2-155k.pt") == module.CANONICAL_CHECKPOINT
    assert (
        str(Path.home() / "PhysPPI/openfold-3/output/physppi_msa_cache/components.cif")
        == module.COMPONENTS_CIF
    )


def test_semantic_paths_allow_environment_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    components = tmp_path / "components.cif"
    monkeypatch.setenv("OPENFOLD3_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("PHYSPPI_COMPONENTS_CIF", str(components))

    module = _load_script("prepare_semantic_50epoch")

    assert str(checkpoint) == module.CANONICAL_CHECKPOINT
    assert str(components) == module.COMPONENTS_CIF


def test_snapshot_root_has_portable_home_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHYSPPI_QBIOLIP_ROOT", raising=False)

    module = _load_script("feature_sanity_report")

    assert module._SNAPSHOT_DATASET_ROOTS["qbiolip"] == (
        Path.home() / "PPI/Prior_out/training/Q-BioLiP-10k-new"
    )


def test_snapshot_root_allows_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "qbiolip"
    monkeypatch.setenv("PHYSPPI_QBIOLIP_ROOT", str(root))

    module = _load_script("feature_sanity_report")

    assert module._SNAPSHOT_DATASET_ROOTS["qbiolip"] == root
