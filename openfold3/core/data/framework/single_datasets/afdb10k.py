"""Single-dataset route for the canonical AFDB-10k-1 homodimer tree."""

from __future__ import annotations

from pathlib import Path

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.prior_complex import (
    PhysPPIPriorComplexDatasetAdapter,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader


@register_dataset
class AFDB10kDataset(PhysPPIPriorComplexDatasetAdapter):
    """OF3-native features plus canonical AFDB optional slots."""

    reader_name = "AFDB"

    def __init__(self, dataset_config) -> None:
        root = Path(dataset_config.physppi.root)
        try:
            self.reader = AFDB10kReader(
                root,
                dataset_config.physppi.max_samples,
                index_cache_file=dataset_config.physppi.index_cache_file,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "AFDB10kDataset requires the visible root/chunk tree; "
                "legacy or staging roots are not accepted"
            ) from exc
        super().__init__(dataset_config)
        self.physppi = dataset_config.physppi
        self.create_datapoint_cache_from_reader(self.reader, self.physppi)
        self.dataset_cache.release_connections()

    def __getitem__(self, index: int) -> dict:
        datapoint = self.datapoint_cache.iloc[index]
        sample_id = str(datapoint["pdb_id"])
        try:
            sample = self.reader.read(sample_id)
            return self.build_sample_features(sample, datapoint)
        except Exception as exc:
            raise RuntimeError(f"AFDB sample {sample_id} failed") from exc
