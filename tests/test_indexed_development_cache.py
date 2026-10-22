"""Bounded immutable development reuse; synthetic generated CPU graphs only."""
from copy import copy

import pytest
import torch
from torch_geometric.loader import DataLoader

from modern2d_fixtures import graph
from molgap.gptrans_execution_tuning import PreparedDevelopmentCache


def make_cache(*, max_bytes=1024*1024):
    members = tuple(3*i+7 for i in range(130))
    graphs = [graph(3, source) for source in members]
    for sample in graphs:
        sample.role_name = "common_dev"
    class Source:
        calls = 0
        def __iter__(self):
            self.calls += 1
            return iter(DataLoader(graphs, batch_size=128, shuffle=False,
                generator=torch.Generator().manual_seed(42)))
    source = Source()
    cache = PreparedDevelopmentCache(source, rows=len(members), batch_size=128,
        max_bytes=max_bytes, source_start=None, source_indices=members)
    return cache, source, members


def test_indexed_warm_cache_preserves_source_role_dtype_and_private_rng():
    cache, source, members = make_cache()
    rng = torch.get_rng_state().clone()
    cold = list(cache)
    # Transfer through a borrowed store cannot relocate retained CPU tensors.
    copy(cold[0]).to("meta", "x", "edge_index", "edge_attr")
    warm = list(cache)
    assert source.calls == 1
    assert [source for batch in warm for source in batch.source_idx.tolist()] == list(members)
    for a, b in zip(cold, warm):
        assert set(a.keys()) == set(b.keys())
        assert all(role == "common_dev" for role in b.role_name)
        for key in ("x", "edge_index", "edge_attr", "y", "source_idx", "random_walk_pe"):
            assert torch.equal(a[key], b[key]) and a[key].dtype == b[key].dtype
            assert b[key].device.type == "cpu"
    assert torch.equal(torch.get_rng_state(), rng)
    assert cache.stats["warm_passes"] == 1


def test_overflow_and_incomplete_pass_never_publish_partial_cache():
    cache, source, members = make_cache(max_bytes=1)
    assert sum(batch.num_graphs for batch in cache) == len(members)
    assert sum(batch.num_graphs for batch in cache) == len(members)
    assert source.calls == 2 and cache.batches is None
    assert cache.disabled_reason == "tensor_storage_budget_exceeded"
    cache, source, _ = make_cache()
    iterator = iter(cache)
    next(iterator)
    iterator.close()
    assert cache.batches is None and not cache.busy
    list(cache)
    assert source.calls == 2 and cache.batches is not None
    cache.clear()
    assert cache.batches is None


def test_indexed_order_change_rejects_before_cache_publication():
    cache, _, members = make_cache()
    cache.source_indices = tuple(i+1 for i in members)
    with pytest.raises(ValueError, match="row count/order"):
        list(cache)
    assert cache.batches is None and not cache.busy


@pytest.mark.parametrize("members", [[1, 4], (1, 1), (4, 1), (True, 4), (-1, 4), (1,)])
def test_indexed_member_declarations_fail_closed(members):
    with pytest.raises(ValueError):
        PreparedDevelopmentCache([], rows=2, batch_size=128, max_bytes=100,
            source_start=None, source_indices=members)


def test_indexed_manifest_cannot_also_claim_a_contiguous_source_range():
    with pytest.raises(ValueError):
        PreparedDevelopmentCache([], rows=2, batch_size=128, max_bytes=100,
            source_start=1, source_indices=(1, 4))
