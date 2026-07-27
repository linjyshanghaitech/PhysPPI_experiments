"""Architecture boundaries for the final PhysPPI package layout."""

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = REPOSITORY_ROOT / "openfold3/projects/of3_all_atom/physppi"
FORBIDDEN_IMPORT_ROOTS = (
    "openfold3.tests",
    "scripts",
)
REQUIRED_PACKAGE_PATHS = (
    "data/readers",
    "data/priors",
    "data/adapters",
    "features",
    "model",
    "screening",
)
LEGACY_FINAL_FILES = (
    "conditioning.py",
    "contact_encoder.py",
    "feature_injection.py",
    "interaction_auxiliary.py",
    "membrane_geometry.py",
    "pair_constraints.py",
    "surface_encoder.py",
    "surface_position.py",
    "tm_topology.py",
)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return imported


def test_production_code_does_not_import_tests_or_scripts() -> None:
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for imported in _imported_modules(path):
            if imported.startswith(FORBIDDEN_IMPORT_ROOTS):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {imported}")

    assert not violations, "\n".join(violations)


def test_final_package_layout_replaces_legacy_root_modules() -> None:
    missing = [
        relative
        for relative in REQUIRED_PACKAGE_PATHS
        if not (PACKAGE_ROOT / relative).is_dir()
    ]
    remaining = [
        filename
        for filename in LEGACY_FINAL_FILES
        if (PACKAGE_ROOT / filename).exists()
    ]

    assert not missing and not remaining, (
        f"missing target packages: {missing}; "
        f"legacy root modules still present: {remaining}"
    )
