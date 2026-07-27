"""Private, read-only-source configuration for PhysPPI preflight runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from openfold3.core.data.framework.single_datasets.afdb10k import AFDB10kDataset
from openfold3.core.data.framework.single_datasets.qbiolip10k import QBioLiP10kDataset
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    AFDB10kConfig,
    QBioLiP10kConfig,
)
from openfold3.projects.of3_all_atom.physppi.screening.models import (
    DatasetName,
    ScreeningOptions,
)

TOKEN_BUDGET = 768
SURFACE_NEIGHBORS = 128


def _source_index(dataset: DatasetName, options: ScreeningOptions) -> Path:
    return (
        options.qbiolip_index_cache
        if dataset is DatasetName.QBIOLIP
        else options.afdb_index_cache
    )


def _source_root(dataset: DatasetName, options: ScreeningOptions) -> Path:
    return options.qbiolip_root if dataset is DatasetName.QBIOLIP else options.afdb_root


def _crop_strategy(dataset: DatasetName, worker_cache_dir: Path) -> str:
    prefix = f"{dataset.value}-"
    if not worker_cache_dir.name.startswith(prefix):
        raise ValueError(f"worker cache must be named {prefix}<crop-strategy>")
    crop_strategy = worker_cache_dir.name.removeprefix(prefix)
    if crop_strategy not in {"spatial", "spatial_interface"}:
        raise ValueError(f"unsupported preflight crop strategy: {crop_strategy}")
    return crop_strategy


def _validate_worker_cache(
    dataset: DatasetName, options: ScreeningOptions, worker_cache_dir: Path
) -> Path:
    worker_cache_dir = Path(worker_cache_dir).resolve()
    root = (options.output_root / "worker-cache" / str(os.getpid())).resolve()
    if worker_cache_dir.parent != root:
        raise ValueError("preflight caches must be private to this worker PID")
    _crop_strategy(dataset, worker_cache_dir)
    return worker_cache_dir


def _validate_shared_index(source: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"missing shared reader index cache: {source}")
    return source.resolve()


def _write_private_dataset_cache(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "_type": "ProteinMonomerDatasetCache",
                "name": "physppi-preflight",
                "structure_data": {},
                "reference_molecule_data": {},
            },
            indent=2,
        )
    )


def build_preflight_dataset(
    dataset: DatasetName, options: ScreeningOptions, worker_cache_dir: Path
):
    """Build one crop-specific production dataset without mutating shared state."""
    worker_cache_dir = _validate_worker_cache(dataset, options, worker_cache_dir)
    crop_strategy = _crop_strategy(dataset, worker_cache_dir)
    source_root = _source_root(dataset, options)
    source_index = _source_index(dataset, options)
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing shared dataset root: {source_root}")
    if not options.components_cif.is_file():
        raise FileNotFoundError(
            f"missing shared components.cif: {options.components_cif}"
        )
    if not options.reference_molecule_directory.is_dir():
        raise FileNotFoundError(
            "missing shared canonical reference molecules: "
            f"{options.reference_molecule_directory}"
        )

    worker_cache_dir.mkdir(parents=True, exist_ok=True)
    index_cache = _validate_shared_index(source_index)
    dataset_cache = worker_cache_dir / "cache.json"
    template_arrays = worker_cache_dir / "template-arrays"
    template_arrays.mkdir(exist_ok=True)
    _write_private_dataset_cache(dataset_cache)

    weights = {
        "contiguous": 0.0,
        "spatial": float(crop_strategy == "spatial"),
        "spatial_interface": float(crop_strategy == "spatial_interface"),
    }
    config_class = QBioLiP10kConfig if dataset is DatasetName.QBIOLIP else AFDB10kConfig
    config = config_class.model_validate(
        {
            "name": (
                "qbiolip10k"
                if dataset is DatasetName.QBIOLIP
                else "afdb10k"
            ),
            "dataset_paths": {
                "dataset_cache_file": dataset_cache,
                "alignments_directory": None,
                "alignment_db_directory": None,
                "alignment_array_directory": source_root,
                "target_structures_directory": source_root,
                "target_structure_file_format": "pdb",
                "reference_molecule_directory": options.reference_molecule_directory,
                "template_cache_directory": None,
                "template_structures_directory": None,
                "template_structure_array_directory": template_arrays,
                "template_file_format": "npz",
                "ccd_file": options.components_cif,
            },
            "crop": {
                "token_crop": {
                    "token_budget": TOKEN_BUDGET,
                    "crop_weights": weights,
                }
            },
            "physppi": {
                "root": source_root,
                "index_cache_file": index_cache,
                "manifest": None,
                "sample_ids": [],
                "num_surface_neighbors": SURFACE_NEIGHBORS,
            },
        }
    )
    dataset_class = (
        QBioLiP10kDataset if dataset is DatasetName.QBIOLIP else AFDB10kDataset
    )
    return dataset_class(config)
