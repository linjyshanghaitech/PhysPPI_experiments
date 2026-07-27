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
    module = _load_script("prepare_semantic_50epoch")
    monkeypatch.delenv("OPENFOLD3_CHECKPOINT", raising=False)
    monkeypatch.delenv("PHYSPPI_COMPONENTS_CIF", raising=False)

    assert (
        str(Path.home() / ".openfold3/of3-p2-155k.pt")
        == module._canonical_checkpoint_path()
    )
    assert (
        str(Path.home() / "PhysPPI/openfold-3/output/physppi_msa_cache/components.cif")
        == module._components_cif_path()
    )


def test_semantic_paths_allow_environment_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("prepare_semantic_50epoch")
    checkpoint = tmp_path / "model.pt"
    components = tmp_path / "components.cif"
    monkeypatch.setenv("OPENFOLD3_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("PHYSPPI_COMPONENTS_CIF", str(components))

    assert str(checkpoint) == module._canonical_checkpoint_path()
    assert str(components) == module._components_cif_path()


def test_snapshot_root_has_portable_home_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("feature_sanity_report")
    monkeypatch.delenv("PHYSPPI_QBIOLIP_ROOT", raising=False)

    assert module._snapshot_dataset_root("qbiolip") == (
        Path.home() / "PPI/Prior_out/training/Q-BioLiP-10k-new"
    )


def test_snapshot_root_allows_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("feature_sanity_report")
    root = tmp_path / "qbiolip"
    monkeypatch.setenv("PHYSPPI_QBIOLIP_ROOT", str(root))

    assert module._snapshot_dataset_root("qbiolip") == root
