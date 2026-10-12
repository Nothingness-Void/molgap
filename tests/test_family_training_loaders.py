"""Persistent streams over synthetic CPU graphs only, without model execution."""
from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Data

from molgap.family_training_data import DatasetSelection, PassBatchSampler, TrainingData
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.gptrans_execution_tuning import ExecutionTuning


def make_pool(family, *, workers=0, development=True, cache_bytes=1024 * 1024,
              drop_last=True):
    def graphs(start, rows):
        return [Data(x=torch.zeros(2, 9, dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.zeros(2, 3, dtype=torch.long),
            random_walk_pe=torch.full((2, 16), float(i)),
            y=torch.tensor([float(i % 11)]), source_idx=torch.tensor([i]),
            pos=torch.zeros(2, 3)) for i in range(start, start + rows)]
    selection = DatasetSelection("manifest.json", "1" * 64, development)
    train, dev = graphs(0, 257), graphs(257, 129) if development else None
    sampler = "python-epoch-shuffle" if family == "neural_atom_k1" else "torch-pass-randperm"
    steps = (2 if drop_last else 3) * 2
    recipe = FamilyRecipe(family, "reference", 42, "2" * 64, selection,
        len(train), len(dev) if dev else 0, steps, .001, .01,
        ScheduleSpec("cosine_epoch", 2),
        "best-development-live" if development else "final-live", sampler, 1,
        target_mean_eV=3., target_std_eV=2., drop_last=drop_last)
    data = TrainingData(train, dev, selection.manifest_sha256, len(train), len(dev) if dev else 0,
        0, 257 if dev else None, 3., 2.)
    tuning = ExecutionTuning(training_workers=workers, development_cache_bytes=cache_bytes)
    return FamilyLoaders(recipe, data, tuning)


def order(loader):
    return [batch.source_idx.tolist() for batch in loader]


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("drop_last", [True, False])
def test_one_loader_reused_with_exact_resume_order_and_private_rng(family, workers, drop_last):
    pool = make_pool(family, workers=workers, drop_last=drop_last)
    rng = torch.get_rng_state()
    try:
        loader = pool.train(0, 0)
        first = order(loader)
        worker_iterator = getattr(loader, "_iterator", None)
        assert pool.train(1, 0) is loader
        second = order(loader)
        assert first != second
        assert getattr(loader, "_iterator", None) is worker_iterator
        expected = list(PassBatchSampler(257, 128, 42, algorithm=pool.recipe.sampler,
            pass_index=1, drop_last=drop_last))
        assert second == expected
        assert order(pool.train(1, 1)) == second[1:]
        assert torch.equal(rng, torch.get_rng_state())
        assert len(first[-1]) == (128 if drop_last else 1)
    finally:
        pool.close()
    assert getattr(loader, "_iterator", None) is None
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.train(0, 0)


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("cache_bytes", [1, 1024 * 1024])
def test_development_cache_preserves_family_inputs_order_and_original_storage(family, cache_bytes):
    pool = make_pool(family, cache_bytes=cache_bytes)
    try:
        first = list(pool.dev())
        original = [batch.source_idx.clone() for batch in first]
        first[0].to("meta", "x", "edge_index", "edge_attr")
        second = list(pool.dev())
        assert [idx for batch in second for idx in batch.source_idx.tolist()] == list(range(257, 386))
        assert all(torch.equal(batch.source_idx, index) for batch, index in zip(second, original))
        assert all(batch.x.device.type == "cpu" for batch in second)
        assert all("pos" not in batch for batch in second)
        assert all(("random_walk_pe" in batch) == (family == "neural_atom_k1") for batch in second)
        assert all(("execution_layout" in batch) == (family == "gptrans_t") for batch in second)
        if cache_bytes == 1:
            assert pool.development_cache.disabled_reason == "tensor_storage_budget_exceeded"
        else:
            assert pool.development_cache.stats["warm_passes"] == 1
    finally:
        pool.close()


def test_final_only_has_no_development_loader_and_rejects_mismatched_data():
    pool = make_pool("neural_atom_k1", development=False)
    try:
        assert pool.development is None
        with pytest.raises(ValueError, match="No selected"):
            pool.dev()
        with pytest.raises(ValueError, match="differs"):
            FamilyLoaders(pool.recipe, replace(pool.data, train_rows=258), pool.tuning)
        with pytest.raises(ValueError, match="cursor"):
            pool.train(0, 9)
    finally:
        pool.close()
