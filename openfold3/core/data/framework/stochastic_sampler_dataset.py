# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the SamplerDataset and OF3DistributedSampler classes.

The SamplerDataset class is a pytorch Dataset class that wraps one or more
SingleDataset instances. The OF3DistributedSampler class samples a desired number
of datapoints from the SamplerDataset based on the provided dataset and datapoint
probabilities. The sampling is done by generating a list of index tuples for a
given dataset-datapoint pair per sample that is regenerated at the start of
each virtual epoch.

The steps below outline how datapoints get from raw datapoints to the model
and highlight where you currently are in the process:

0. Dataset filtering and cache generation
    raw data -> filtered data
1. PreprocessingPipeline
    filtered data -> preprocessed data
2. SampleProcessingPipeline and FeaturePipeline
    preprocessed data -> parsed/processed data -> FeatureDict
3. SingleDataset
    datapoints -> __getitem__ -> FeatureDict
4. OF3DistributedSampler + SamplerDataset (optional) [YOU ARE HERE]
    Sequence[SingleDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule
    SingleDataset/SamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

# TODO: rename module from stochastic_sampler_dataset.py to sampler_dataset.py
import logging
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from openfold3.core.data.framework.single_datasets.abstract_single import SingleDataset

logger = logging.getLogger(__name__)


class SampleOccurrence(NamedTuple):
    """Immutable identity for one sample occurrence in a virtual epoch."""

    dataset_idx: int
    datapoint_idx: int
    epoch: int
    global_slot: int


class SamplerDataset(Dataset):
    """
    A dataset class for combining multiple SingleDataset instances.
    Accepts (dataset_idx, datapoint_idx) tuples from the OF3DistributedSampler
    to fetch the actual data.
    """

    def __init__(
        self,
        datasets: Sequence[SingleDataset],
        epoch_len: int = None,
    ) -> None:
        """
        Args:
            datasets (Sequence[SingleDataset]):
                List of datasets to sample from.
            epoch_len (int):
                Number of datapoints to sample in total for each virtual epoch.
        """
        super().__init__()
        self.datasets = datasets
        self.epoch_len = epoch_len

    def __len__(self):
        # This length is nominal; the Sampler controls the actual __iter__ length
        return self.epoch_len

    def __getitem__(self, occurrence: SampleOccurrence) -> Any:
        """
        Wrapper getitem for indexing into the unrolled examples.

        Args:
            occurrence: Immutable sample identity yielded by the sampler.
        """
        dataset = self.datasets[occurrence.dataset_idx]

        # Set datapoint index in sampled dataset - used for logging in the treadmill
        dataset.set_current_datapoint_index(occurrence.datapoint_idx)
        dataset.set_current_occurrence(occurrence)

        # Index into the list of datasets then datapoints for the given dataset
        # This calls the __getitem__ method of the SingleDataset class
        return dataset[occurrence.datapoint_idx]

    def get_worker_path(self, subdirs: list[str] | None, fname: str) -> str:
        """Returns the path to the worker output directory or file.

        Note: Treadmill logging utility. This function only works if individual datasets
        passed to the StochasticSamplerDataset were wrapped with the LoggingMixin.

        Args:
            subdirs (list[str] | None):
                List of subdirectories to append to the path.
            fname (str):
                Filename to append to the path.
        """
        # Get the dataset-specific path
        dataset_path = self.datasets[0].get_worker_path(subdirs=subdirs, fname=fname)
        return dataset_path


class OF3DistributedSampler(DistributedSampler):
    """
    A dataset class for combining multiple SingleDataset instances and
    sampling from them with the provided probabilities.
    """

    _DISTINCT_PREFIX_CHUNK_SIZE = 512
    _DISTINCT_CHUNK_SEED_STRIDE = 1_000_003

    def __init__(
        self,
        dataset: SamplerDataset,
        dataset_probabilities: Sequence[float],
        next_dataset_indices: dict[str, Any],
        epoch_len: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
        distinct_samples_per_rank_group: bool = False,
    ) -> None:
        """Initializes the SamplerDataset class.

        Args:
            dataset (Dataset):
                Composite SamplerDataset to sample from.
            dataset_probabilities (Sequence[float]):
                Probabilities of sampling each dataset.
            epoch_len (int):
                Number of datapoints to sample in total for each virtual epoch.
            next_dataset_indices: dict[str, Any]
                Record of last used indices for datasets that use in-order
                sampling
            num_replicas:
                Number of processes participating in distributed training
            rank:
                Rank of the current process
            seed:
                Random seed used to shuffle the sampler if shuffle=True
            distinct_samples_per_rank_group:
                Resolve duplicate dataset-datapoint pairs within each contiguous
                distributed rank group before rank slicing.
        """
        logger.debug(
            f"Rank {rank} - Initializing OF3DistributedSampler with "
            f"{dataset_probabilities=}, {next_dataset_indices=}, {epoch_len=}, {seed=}"
        )
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )
        # Access the underlying SingleDatasets
        self.datasets = dataset.datasets
        self.dataset_probabilities = torch.tensor(dataset_probabilities)
        self.epoch_len = epoch_len
        self.next_dataset_indices = next_dataset_indices
        self.distinct_samples_per_rank_group = distinct_samples_per_rank_group

        if (
            self.distinct_samples_per_rank_group
            and self.epoch_len % self.num_replicas != 0
        ):
            raise ValueError(
                "epoch_len must be divisible by num_replicas when "
                "distinct_samples_per_rank_group is enabled"
            )

    @staticmethod
    def _get_random_subset(
        dataset: SingleDataset, num_examples: int, generator: torch.Generator
    ) -> torch.Tensor:
        """Selects random indices from dataset based on dataset probabilities."""
        # Retrieve datapoint probabilities for given dataset
        datapoint_probabilities = torch.tensor(
            dataset.datapoint_cache["datapoint_probabilities"].to_numpy()
        )

        # Sample datapoint indices
        datapoint_indices_i = torch.multinomial(
            input=datapoint_probabilities,
            num_samples=num_examples,
            replacement=True,
            generator=generator,
        )
        return datapoint_indices_i

    def _get_ordered_subset(
        self, dataset: SingleDataset, num_examples: int
    ) -> torch.Tensor:
        """Selects indices based on sliced examples from the dataset."""

        datapoint_probabilities = torch.tensor(
            dataset.datapoint_cache["datapoint_probabilities"].to_numpy()
        )
        if not torch.all(torch.eq(datapoint_probabilities, 1.0)):
            raise ValueError(
                "Ordered slicing of datasets not supported for "
                "datasets with nonuniform probabilities"
            )

        start_idx = self.next_dataset_indices[dataset.name]
        end_idx = start_idx + num_examples

        slice_indices = torch.arange(start_idx, min(end_idx, len(dataset)))

        if end_idx > len(dataset):
            end_idx = end_idx - len(dataset)
            logger.warning(
                f"Reached the end of the dataset {dataset.name},"
                f"missing {end_idx} examples,"
                "Sampling will loop back from beginning of dataset."
            )
            slice_indices = torch.concat((slice_indices, torch.arange(0, end_idx)))

        self.next_dataset_indices[dataset.name] = end_idx

        logger.debug(
            f"Fetching ordered subset for epoch {self.epoch} {dataset.name}: "
            f"start={start_idx}, n={num_examples}, end={end_idx}"
        )

        return slice_indices

    def get_datapoint_indices(
        self, dataset_indices: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        n_datasets = len(self.datasets)
        datapoint_indices = torch.zeros(len(dataset_indices), dtype=torch.long)

        for dataset_idx, num_datapoints_per_dataset in zip(
            torch.arange(n_datasets),
            torch.bincount(dataset_indices, minlength=n_datasets),
            strict=True,
        ):
            if num_datapoints_per_dataset == 0:
                continue

            dataset = self.datasets[dataset_idx]

            # Create a dataset-specific generator derived from the main one
            generator_seed = torch.randint(
                low=0, high=100000, size=(1,), generator=generator
            ).item()
            datapoint_idx_generator = torch.Generator().manual_seed(generator_seed)

            if dataset.name in self.next_dataset_indices:
                datapoint_indices_i = self._get_ordered_subset(
                    dataset=dataset, num_examples=num_datapoints_per_dataset
                )
            else:
                datapoint_indices_i = self._get_random_subset(
                    dataset=dataset,
                    num_examples=num_datapoints_per_dataset,
                    generator=datapoint_idx_generator,
                )

            # Add to datapoint index container to pair with dataset indices
            datapoint_indices[torch.where(dataset_indices == dataset_idx)] = (
                datapoint_indices_i
            )

        return datapoint_indices

    def get_dataset_indices(
        self,
        generator: torch.Generator,
        *,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        return torch.multinomial(
            input=self.dataset_probabilities,
            num_samples=self.epoch_len if num_samples is None else num_samples,
            replacement=True,
            generator=generator,
        )

    def _deduplicate_rank_groups(
        self,
        dataset_indices: torch.Tensor,
        datapoint_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if not self.distinct_samples_per_rank_group:
            return datapoint_indices

        resolved = datapoint_indices.clone()
        for start in range(0, len(dataset_indices), self.num_replicas):
            used: set[tuple[int, int]] = set()
            for position in range(start, start + self.num_replicas):
                dataset_idx = int(dataset_indices[position])
                datapoint_idx = int(resolved[position])
                key = (dataset_idx, datapoint_idx)
                if key in used:
                    probabilities = torch.tensor(
                        self.datasets[dataset_idx]
                        .datapoint_cache["datapoint_probabilities"]
                        .to_numpy(),
                        dtype=torch.float,
                    )
                    for used_dataset_idx, used_datapoint_idx in used:
                        if used_dataset_idx == dataset_idx:
                            probabilities[used_datapoint_idx] = 0.0
                    if int(torch.count_nonzero(probabilities)) == 0:
                        raise ValueError(
                            "dataset cannot provide enough distinct datapoints "
                            f"for a rank group: dataset_idx={dataset_idx}, "
                            f"num_replicas={self.num_replicas}"
                        )
                    datapoint_idx = int(
                        torch.multinomial(
                            probabilities,
                            num_samples=1,
                            replacement=False,
                            generator=generator,
                        ).item()
                    )
                    resolved[position] = datapoint_idx
                    key = (dataset_idx, datapoint_idx)
                used.add(key)
        return resolved

    def __iter__(self) -> Iterator[SampleOccurrence]:
        """
        Sample epoch_len number of samples according to the provided probabilities.
        """

        chunk_size = (
            min(self.epoch_len, self._DISTINCT_PREFIX_CHUNK_SIZE)
            if self.distinct_samples_per_rank_group
            else self.epoch_len
        )
        dataset_chunks: list[torch.Tensor] = []
        datapoint_chunks: list[torch.Tensor] = []
        for chunk_idx, start in enumerate(range(0, self.epoch_len, chunk_size)):
            num_samples = min(chunk_size, self.epoch_len - start)
            # Chunk zero preserves the historical seed exactly. A large prime
            # stride makes later chunks deterministic without colliding with
            # the supported epoch range, so an epoch_len=512 smoke is an exact
            # prefix of an epoch_len=4096 formal run.
            generator = torch.Generator().manual_seed(
                self.seed
                + self.epoch
                + chunk_idx * self._DISTINCT_CHUNK_SEED_STRIDE
            )
            dataset_indices_i = self.get_dataset_indices(
                generator=generator,
                num_samples=num_samples,
            )
            datapoint_indices_i = self.get_datapoint_indices(
                dataset_indices=dataset_indices_i,
                generator=generator,
            )
            datapoint_indices_i = self._deduplicate_rank_groups(
                dataset_indices=dataset_indices_i,
                datapoint_indices=datapoint_indices_i,
                generator=generator,
            )
            dataset_chunks.append(dataset_indices_i)
            datapoint_chunks.append(datapoint_indices_i)
        dataset_indices = torch.cat(dataset_chunks)
        datapoint_indices = torch.cat(datapoint_chunks)

        # Create the global list of dataset-datapoint index tuples
        global_pairs = torch.stack((dataset_indices, datapoint_indices), dim=1).tolist()

        logger.debug(f"Sampled batch indices: {global_pairs=}")

        # super().__iter__() yields the indices for this rank.
        # Use those indices to pick the correct tuples from the global list
        indices_for_this_rank = list(super().__iter__())

        logger.debug(
            f"Called OF3DistributedSampler.__iter__ in rank {self.rank}: "
            f"epoch {self.epoch}, seed {self.seed + self.epoch}, sampled dataset "
            f"indices {dataset_indices.tolist()}, sampled datapoint indices "
            f"{datapoint_indices.tolist()}, indices_for_this_rank "
            f"{indices_for_this_rank}."
        )

        for i in indices_for_this_rank:
            dataset_idx, datapoint_idx = global_pairs[i]
            yield SampleOccurrence(
                dataset_idx=int(dataset_idx),
                datapoint_idx=int(datapoint_idx),
                epoch=self.epoch,
                global_slot=i,
            )
