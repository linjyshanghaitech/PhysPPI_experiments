"""Characterize the public PhysPPI imports that the refactor must preserve."""

from importlib import import_module

import pytest

PUBLIC_SYMBOLS = {
    "OptionalPriorConditioner": "openfold3.projects.of3_all_atom.physppi",
    "OptionalPriorOutput": "openfold3.projects.of3_all_atom.physppi",
    "InteractionDistanceHead": "openfold3.projects.of3_all_atom.physppi",
    "interaction_auxiliary_loss": "openfold3.projects.of3_all_atom.physppi",
    "DirectedMembranePlaneEstimator": "openfold3.projects.of3_all_atom.physppi",
    "MembraneDiffusionConditioner": "openfold3.projects.of3_all_atom.physppi",
    "QBioLiP10kReader": ("openfold3.projects.of3_all_atom.physppi.data"),
    "AFDB10kReader": "openfold3.projects.of3_all_atom.physppi.data",
    "PriorComplexSample": ("openfold3.projects.of3_all_atom.physppi.data"),
    "PhysPPIPriorComplexDatasetAdapter": (
        "openfold3.projects.of3_all_atom.physppi.data"
    ),
}


@pytest.mark.parametrize(("symbol", "module_name"), PUBLIC_SYMBOLS.items())
def test_public_symbol_remains_importable(symbol: str, module_name: str) -> None:
    if symbol == "PhysPPIPriorComplexDatasetAdapter":
        # Current OpenFold runtime registers framework datasets before importing
        # their shared PhysPPI adapter. The refactor may remove this prerequisite.
        import_module("openfold3.core.data.framework")
    module = import_module(module_name)

    assert getattr(module, symbol).__name__ == symbol
