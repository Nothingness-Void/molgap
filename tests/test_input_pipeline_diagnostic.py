"""CPU-only synthetic sampling checks; no graph assets or protected roles."""
import random

import pytest

from molgap.family_training_data import PassBatchSampler
from molgap.family_training_loaders import training_loader_options
from molgap.input_pipeline_diagnostic import (
    DiagnosticBatchSampler, packed_node_counts, padding_statistics,
)


@pytest.mark.parametrize("strategy", ["global", "block", "bucket"])
@pytest.mark.parametrize("drop_last", [True, False])
def test_exact_membership_tail_and_resume(strategy, drop_last):
    counts = [1 + (i * 7 % 47) for i in range(521)]
    expected = list(PassBatchSampler(len(counts), 16, 42,
        algorithm="python-epoch-shuffle", pass_index=3, drop_last=drop_last))
    sampler = DiagnosticBatchSampler(counts, 16, 42, strategy=strategy,
        pass_index=3, drop_last=drop_last)
    batches = list(sampler)
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == sorted(i for batch in expected for i in batch)
    assert len(flat) == len(set(flat))
    assert sorted(map(len, batches)) == sorted(map(len, expected))
    assert len(sampler) == len(expected)
    assert list(DiagnosticBatchSampler(counts, 16, 42, strategy=strategy,
        pass_index=3, start_batch=5, drop_last=drop_last)) == batches[5:]
    if strategy == "global":
        assert batches == expected


@pytest.mark.parametrize("strategy", ["global", "block", "bucket"])
def test_private_rng_and_epoch_variation(strategy):
    counts = [1 + i % 37 for i in range(1024)]
    python_rng = random.getstate()
    first = list(DiagnosticBatchSampler(counts, 16, 42, strategy=strategy))
    assert random.getstate() == python_rng
    assert first == list(DiagnosticBatchSampler(counts, 16, 42, strategy=strategy))
    assert first != list(DiagnosticBatchSampler(counts, 16, 42, strategy=strategy, pass_index=1))
    if strategy == "bucket":
        companions = lambda batches: {frozenset(batch) for batch in batches}
        assert companions(first) != companions(list(DiagnosticBatchSampler(
            counts, 16, 42, strategy=strategy, pass_index=1)))


def test_bucket_reduces_padding_not_samples():
    counts = [1 + (i * 7 % 47) for i in range(4096)]
    observations = {strategy: padding_statistics(counts, list(DiagnosticBatchSampler(
        counts, 16, 42, strategy=strategy))) for strategy in ("global", "bucket")}
    assert observations["global"]["samples"] == observations["bucket"]["samples"]
    assert observations["bucket"]["padded_node_slots"] < observations["global"]["padded_node_slots"]
    assert observations["bucket"]["dense_attention_pair_slots"] < observations["global"]["dense_attention_pair_slots"]


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_shared_worker_options(workers):
    options = training_loader_options(workers)
    assert options["persistent_workers"] is (workers > 0)
    assert options["pin_memory"] is True
    assert ("prefetch_factor" in options) is (workers > 0)


@pytest.mark.parametrize("counts,strategy,window", [([1, 0], "global", 20),
    ([1, True], "global", 20), ([1, 2], "unknown", 20), ([1, 2], "bucket", 1)])
def test_invalid_sampler_rejected(counts, strategy, window):
    with pytest.raises(ValueError):
        DiagnosticBatchSampler(counts, 1, 42, strategy=strategy, window_batches=window)


def test_metadata_uses_packed_slices_without_materializing_graphs():
    import torch
    from types import SimpleNamespace

    class Packed:
        _indices = None
        slices = {"x": torch.tensor([0, 2, 9, 12])}

        def __len__(self):
            return 3

        def __getitem__(self, index):
            raise AssertionError("Graph materialization is not metadata access")

    assert packed_node_counts(Packed()) == [2, 7, 3]
    class Combined(SimpleNamespace):
        def __len__(self):
            return 6
    assert packed_node_counts(Combined(datasets=[Packed(), Packed()])) == [2, 7, 3] * 2
