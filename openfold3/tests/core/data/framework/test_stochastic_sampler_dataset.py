from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
import torch

from openfold3.core.data.framework.stochastic_sampler_dataset import (
    OF3DistributedSampler,
    SampleOccurrence,
    SamplerDataset,
)


@dataclass
class _Dataset:
    name: str
    size: int

    def __post_init__(self) -> None:
        self.datapoint_cache = pd.DataFrame(
            {
                "pdb_id": [f"{self.name}-{index}" for index in range(self.size)],
                "datapoint_probabilities": [1.0] * self.size,
            }
        )

    def __len__(self) -> int:
        return self.size


def _rank_sequences(
    *,
    epoch_len: int = 4096,
    seed: int = 42,
    distinct: bool = True,
    sizes: tuple[int, int] = (997, 991),
    weights: tuple[float, float] = (0.8, 0.2),
) -> list[list[tuple[int, int]]]:
    datasets = [_Dataset("qbiolip", sizes[0]), _Dataset("afdb", sizes[1])]
    composite = SamplerDataset(datasets=datasets, epoch_len=epoch_len)
    return [
        list(
            OF3DistributedSampler(
                dataset=composite,
                dataset_probabilities=weights,
                next_dataset_indices={},
                epoch_len=epoch_len,
                num_replicas=8,
                rank=rank,
                seed=seed,
                distinct_samples_per_rank_group=distinct,
            )
        )
        for rank in range(8)
    ]


def _make_sampler(
    *, epoch_len: int, num_replicas: int, rank: int, seed: int
) -> OF3DistributedSampler:
    datasets = [_Dataset("qbiolip", 16)]
    composite = SamplerDataset(datasets=datasets, epoch_len=epoch_len)
    return OF3DistributedSampler(
        dataset=composite,
        dataset_probabilities=[1.0],
        next_dataset_indices={},
        epoch_len=epoch_len,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
    )


def test_sampler_yields_occurrence_identity_after_rank_slice() -> None:
    sampler = _make_sampler(epoch_len=16, num_replicas=8, rank=3, seed=42)
    sampler.set_epoch(2)

    rows = list(sampler)

    assert all(isinstance(row, SampleOccurrence) for row in rows)
    assert [row.epoch for row in rows] == [2, 2]
    assert [row.global_slot for row in rows] == [3, 11]


def test_distinct_mode_yields_eight_unique_pairs_per_micro_step() -> None:
    sequences = _rank_sequences()

    assert {len(sequence) for sequence in sequences} == {512}
    assert all(len(set(group)) == 8 for group in zip(*sequences, strict=True))


def test_distinct_mode_is_reproducible_for_fixed_seed() -> None:
    assert _rank_sequences(seed=7) == _rank_sequences(seed=7)
    assert _rank_sequences(seed=7) != _rank_sequences(seed=8)


def test_distinct_mode_preserves_smoke_prefix_when_epoch_len_changes() -> None:
    smoke = _rank_sequences(epoch_len=512)
    formal = _rank_sequences(epoch_len=4096)

    assert [sequence[:64] for sequence in formal] == smoke


def test_default_mode_matches_explicit_false() -> None:
    datasets = [_Dataset("qbiolip", 17), _Dataset("afdb", 19)]
    composite = SamplerDataset(datasets=datasets, epoch_len=128)
    common = {
        "dataset": composite,
        "dataset_probabilities": [0.8, 0.2],
        "next_dataset_indices": {},
        "epoch_len": 128,
        "num_replicas": 8,
        "rank": 3,
        "seed": 42,
    }

    implicit = list(OF3DistributedSampler(**common))
    explicit = list(
        OF3DistributedSampler(**common, distinct_samples_per_rank_group=False)
    )

    assert implicit == explicit


def test_global_datapoint_mode_uses_one_flat_multinomial_call(monkeypatch) -> None:
    qbiolip = _Dataset("qbiolip", 2)
    afdb = _Dataset("afdb", 2)
    qbiolip.datapoint_cache["datapoint_probabilities"] = [1.0, 2.0]
    afdb.datapoint_cache["datapoint_probabilities"] = [3.0, 4.0]
    composite = SamplerDataset(datasets=[qbiolip, afdb], epoch_len=4)
    calls = []

    def fake_multinomial(input, num_samples, replacement, generator):
        calls.append(input.clone())
        assert num_samples == 4
        assert replacement is True
        return torch.tensor([0, 1, 2, 3])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    sampler = OF3DistributedSampler(
        dataset=composite,
        dataset_probabilities=[1.0, 0.0],
        next_dataset_indices={},
        epoch_len=4,
        num_replicas=1,
        rank=0,
        seed=42,
        global_datapoint_sampling=True,
    )

    occurrences = list(sampler)

    assert len(calls) == 1
    assert calls[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert [(row.dataset_idx, row.datapoint_idx) for row in occurrences] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_distinct_mode_requires_complete_world_size_groups() -> None:
    datasets = [_Dataset("qbiolip", 16), _Dataset("afdb", 16)]
    composite = SamplerDataset(datasets=datasets, epoch_len=65)

    with pytest.raises(ValueError, match="epoch_len.*num_replicas"):
        OF3DistributedSampler(
            dataset=composite,
            dataset_probabilities=[0.8, 0.2],
            next_dataset_indices={},
            epoch_len=65,
            num_replicas=8,
            rank=0,
            distinct_samples_per_rank_group=True,
        )


def test_distinct_mode_preserves_weighted_source_sampling() -> None:
    sequences = _rank_sequences()
    all_pairs = [pair for sequence in sequences for pair in sequence]
    qbiolip_fraction = sum(pair.dataset_idx == 0 for pair in all_pairs) / len(all_pairs)

    assert abs(qbiolip_fraction - 0.8) < 0.03


def test_distinct_mode_keeps_equal_probability_datapoints_observable() -> None:
    sequences = _rank_sequences(
        epoch_len=4096,
        sizes=(8, 8),
        weights=(1.0, 0.0),
    )
    observed = {
        occurrence.datapoint_idx for sequence in sequences for occurrence in sequence
    }

    assert observed == set(range(8))


def test_distinct_mode_rejects_source_with_insufficient_group_capacity() -> None:
    datasets = [_Dataset("qbiolip", 7), _Dataset("afdb", 16)]
    composite = SamplerDataset(datasets=datasets, epoch_len=8)
    sampler = OF3DistributedSampler(
        dataset=composite,
        dataset_probabilities=[1.0, 0.0],
        next_dataset_indices={},
        epoch_len=8,
        num_replicas=8,
        rank=0,
        seed=42,
        distinct_samples_per_rank_group=True,
    )

    with pytest.raises(ValueError, match="cannot provide enough distinct datapoints"):
        list(sampler)


def test_distinct_mode_deduplicates_only_within_one_rank_group() -> None:
    datasets = [_Dataset("qbiolip", 8), _Dataset("afdb", 8)]
    composite = SamplerDataset(datasets=datasets, epoch_len=16)
    sampler = OF3DistributedSampler(
        dataset=composite,
        dataset_probabilities=[1.0, 0.0],
        next_dataset_indices={},
        epoch_len=16,
        num_replicas=8,
        rank=0,
        seed=42,
        distinct_samples_per_rank_group=True,
    )
    generator = torch.Generator().manual_seed(3)
    dataset_indices = torch.zeros(16, dtype=torch.long)
    datapoint_indices = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)

    resolved = sampler._deduplicate_rank_groups(
        dataset_indices=dataset_indices,
        datapoint_indices=datapoint_indices,
        generator=generator,
    )

    assert len(set(resolved[:8].tolist())) == 8
    assert len(set(resolved[8:].tolist())) == 8
    assert set(resolved[:8].tolist()) == set(resolved[8:].tolist())
