"""CPU synthetic sampler metadata checks; no data, checkpoints or models."""
import random
import weakref
from unittest.mock import patch

import numpy as np
import pytest
import torch

from molgap.edge_state_training_core import (
    EpochPermutationBatchSampler,
    _epoch_permutation_metadata,
    epoch_permutation_state,
)


@pytest.fixture(autouse=True)
def isolated_cache_and_no_cuda(monkeypatch):
    _epoch_permutation_metadata.cache_clear()
    initialized = torch.cuda.is_initialized()

    def forbidden_cuda(*args, **kwargs):
        pytest.fail("Sampler metadata must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden_cuda)
    yield
    assert torch.cuda.is_initialized() == initialized
    _epoch_permutation_metadata.cache_clear()


@pytest.mark.parametrize("dataset_size,batch_size", [(1, 1), (8, 2), (9, 2), (17, 5)])
@pytest.mark.parametrize("seed", [0, 42, 43])
@pytest.mark.parametrize("epoch", [0, 1, 7])
def test_parity_drop_last_and_all_cursors(dataset_size, batch_size, seed, epoch):
    sampler = EpochPermutationBatchSampler(
        dataset_size, batch_size, seed=seed, epoch=epoch,
    )
    batches = list(sampler)
    assert len(batches) == dataset_size // batch_size
    assert sum(map(len, batches)) == (dataset_size // batch_size) * batch_size
    with patch.object(torch, "randperm", wraps=torch.randperm) as randperm:
        for cursor in range(len(batches) + 1):
            state = epoch_permutation_state(
                dataset_size, batch_size, seed=seed, epoch=epoch, next_batch=cursor,
            )
            assert state == sampler.state_for(cursor)
        assert randperm.call_count == 1
    resumed = EpochPermutationBatchSampler.from_state(
        state, dataset_size=dataset_size, batch_size=batch_size, seed=seed, epoch=epoch,
    )
    assert list(resumed) == []


def test_fresh_dict_mutation_isolation_and_scalar_only_cache(monkeypatch):
    refs = []
    original_init = EpochPermutationBatchSampler.__init__

    def observe_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        refs.extend((weakref.ref(self), weakref.ref(self.indices)))

    monkeypatch.setattr(EpochPermutationBatchSampler, "__init__", observe_init)
    expected = EpochPermutationBatchSampler(9, 2, seed=42, epoch=3).state_for(0)
    first = epoch_permutation_state(9, 2, seed=42, epoch=3, next_batch=0)
    second = epoch_permutation_state(9, 2, seed=42, epoch=3, next_batch=0)
    assert first == second == expected
    assert first is not second
    first.clear()
    second.update(order_sha256="broken", dataset_size=-1, extra=[])
    assert epoch_permutation_state(9, 2, seed=42, epoch=3, next_batch=0) == expected
    metadata = _epoch_permutation_metadata(9, 2, 42, 3)
    assert type(metadata) is tuple
    assert all(type(item) is tuple and type(item[0]) is str
               and type(item[1]) in (int, str) for item in metadata)
    assert all(ref() is None for ref in refs)


class IntSubclass(int):
    pass


@pytest.mark.parametrize("name", ["dataset_size", "batch_size", "seed", "epoch", "next_batch"])
@pytest.mark.parametrize("invalid", [True, False, 1.0, 0.0, "1", None, np.int64(1), IntSubclass(1), -1])
@pytest.mark.parametrize("warm", [False, True])
def test_invalid_types_rejected_before_cache_without_pollution(name, invalid, warm):
    args = dict(dataset_size=1, batch_size=1, seed=1, epoch=1, next_batch=1)
    if warm:
        epoch_permutation_state(**args)
    before = _epoch_permutation_metadata.cache_info()
    args[name] = invalid
    with pytest.raises(ValueError, match=name):
        epoch_permutation_state(**args)
    assert _epoch_permutation_metadata.cache_info() == before


@pytest.mark.parametrize("dataset_size,batch_size,next_batch", [
    (0, 1, 0), (1, 0, 0), (1, 2, 0), (9, 2, 5),
])
@pytest.mark.parametrize("warm", [False, True])
def test_invalid_bounds_match_sampler_and_do_not_touch_cache(dataset_size, batch_size, next_batch, warm):
    if warm:
        epoch_permutation_state(9, 2, seed=42, epoch=3, next_batch=0)
    before = _epoch_permutation_metadata.cache_info()
    with pytest.raises(ValueError) as helper_error:
        epoch_permutation_state(
            dataset_size, batch_size, seed=42, epoch=3, next_batch=next_batch,
        )
    with pytest.raises(ValueError) as sampler_error:
        EpochPermutationBatchSampler(
            dataset_size, batch_size, seed=42, epoch=3,
        ).state_for(next_batch)
    assert str(helper_error.value) == str(sampler_error.value)
    assert _epoch_permutation_metadata.cache_info() == before


@pytest.mark.parametrize("changed", [
    {"dataset_size": 10}, {"batch_size": 3}, {"seed": 43}, {"epoch": 4},
])
def test_each_metadata_coordinate_has_its_own_cache_key(changed):
    args = dict(dataset_size=9, batch_size=2, seed=42, epoch=3, next_batch=0)
    with patch.object(torch, "randperm", wraps=torch.randperm) as randperm:
        epoch_permutation_state(**args)
        epoch_permutation_state(**{**args, **changed})
        assert randperm.call_count == 2


def test_bounded_lru_eviction_and_regeneration():
    def state(epoch, cursor=0):
        return epoch_permutation_state(9, 2, seed=42, epoch=epoch, next_batch=cursor)

    with patch.object(torch, "randperm", wraps=torch.randperm) as randperm:
        first = state(0)
        second = state(1)
        assert state(0, 4) == {**first, "next_batch": 4}
        state(2)
        assert _epoch_permutation_metadata.cache_info().currsize == 2
        assert _epoch_permutation_metadata.cache_info().maxsize == 2
        assert randperm.call_count == 3
        assert state(0) == first
        assert randperm.call_count == 3
        assert state(1) == second
        assert randperm.call_count == 4
        assert _epoch_permutation_metadata.cache_info().currsize == 2


def test_torch_python_numpy_rng_preserved_on_misses_hits_and_eviction():
    torch_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    for epoch, cursor in [(0, 0), (0, 4), (1, 2), (2, 1), (0, 3)]:
        epoch_permutation_state(9, 2, seed=42, epoch=epoch, next_batch=cursor)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
