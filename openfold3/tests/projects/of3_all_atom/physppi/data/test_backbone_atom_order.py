from __future__ import annotations

import numpy as np
import torch
from biotite.structure import AtomArray

# Importing a registered single-dataset module first completes the framework's
# eager dataset registration before the adapter imports ``BaseOF3Dataset``.
# This avoids the package's pre-existing circular-import edge at test startup.
import openfold3.core.data.framework.single_datasets.afdb10k  # noqa: F401
from openfold3.projects.of3_all_atom.physppi.data.adapters import (
    _geometry_features,
)


def test_geometry_features_preserve_ordered_backbone_indices_and_missing_atoms():
    atoms = AtomArray(5)
    atoms.coord = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    atoms.token_id = np.asarray([0, 0, 0, 1, 1])
    atoms.atom_name = np.asarray(["C", "N", "CA", "N", "CA"])

    result = _geometry_features(
        atoms,
        torch.ones(2, dtype=torch.bool),
        np.asarray(atoms.coord, dtype=np.float32),
        np.ones(len(atoms), dtype=bool),
    )

    centers = result["physppi_token_center_atom_index"]
    backbone = result["physppi_backbone_atom_indices"]
    frame_mask = result["physppi_backbone_frame_mask"]
    assert centers.dtype == torch.int64
    assert backbone.dtype == torch.int64
    assert torch.equal(centers, torch.tensor([2, 4], dtype=torch.int64))
    assert torch.equal(
        backbone,
        torch.tensor([[1, 2, 0], [3, 4, -1]], dtype=torch.int64),
    )
    assert torch.equal(frame_mask, torch.tensor([True, False]))
