"""CPU synthetic integration; no MolGap model, cache, weights or role is loaded."""
from dataclasses import replace
import os
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from molgap.gptrans_execution_backend import PreparedLoader
from molgap.gptrans_execution_tuning import ExecutionTuning, PreparedDevelopmentCache, tuned_backend
from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage


def graphs(rows=130):
    return [Data(x=torch.tensor([[i % 3]], dtype=torch.long),
                 edge_index=torch.empty((2, 0), dtype=torch.long),
                 edge_attr=torch.empty((0, 1), dtype=torch.long),
                 y=torch.tensor([i / 1000.]), source_idx=torch.tensor([100000 + i]))
            for i in range(rows)]


def source(data):
    return PreparedLoader(DataLoader(data, batch_size=128, shuffle=False,
                                    generator=torch.Generator().manual_seed(314159)))


def test_cache_reuses_prepared_batches_and_isolates_pyg_device_transfer():
    cache = PreparedDevelopmentCache(source(graphs()), rows=130, batch_size=128, max_bytes=100000)
    first = list(cache)
    assert [b.num_graphs for b in first] == [128, 2]
    assert cache.stats['cold_passes'] == 1
    assert cache.batches is not None
    first[0].to('meta')
    second = list(cache)
    assert second[0].x.device.type == 'cpu'
    assert second[0].execution_layout['flat_index'].device.type == 'cpu'
    assert second[0].source_idx.tolist() == list(range(100000, 100128))
    assert cache.stats['warm_passes'] == 1
    cache.clear()
    assert cache.batches is None and cache.stats['cached_tensor_storage_bytes'] == 0


def test_cache_overflow_streams_complete_role_and_reports_fallback():
    cache = PreparedDevelopmentCache(source(graphs()), rows=130, batch_size=128, max_bytes=1)
    assert sum(b.num_graphs for b in cache) == 130
    assert cache.batches is None and cache.disabled_reason == 'tensor_storage_budget_exceeded'
    assert sum(b.num_graphs for b in cache) == 130
    assert cache.stats['cold_passes'] == 2 and cache.stats['warm_passes'] == 0


def test_interrupted_pass_is_not_published():
    cache = PreparedDevelopmentCache(source(graphs()), rows=130, batch_size=128, max_bytes=100000)
    iterator = iter(cache)
    next(iterator)
    with pytest.raises(ValueError, match='Concurrent'):
        next(iter(cache))
    iterator.close()
    assert cache.batches is None and not cache.busy
    assert sum(b.num_graphs for b in cache) == 130
    assert cache.batches is not None


@pytest.mark.parametrize('kind', ['order', 'count', 'extra', 'batch'])
def test_cache_rejects_changed_rows_or_batch(kind):
    data = graphs()
    rows, physical = 130, 128
    if kind == 'order':
        data[0].source_idx.fill_(100001)
    elif kind == 'count':
        rows = 131
    elif kind == 'extra':
        data.append(graphs(131)[-1])
    else:
        physical = 64
    with pytest.raises(ValueError, match='row|batch|incomplete'):
        list(PreparedDevelopmentCache(source(data), rows=rows, batch_size=physical, max_bytes=100000))


@pytest.mark.parametrize('field,value', [
    ('cpu_threads', 0), ('cpu_threads', True), ('training_workers', -1),
    ('development_workers', True), ('independent_models', 0), ('total_cpu_budget', 1),
    ('development_cache_bytes', 0), ('cache_development', 1),
])
def test_resource_validation_fails_closed(field, value):
    with pytest.raises(ValueError):
        replace(ExecutionTuning(), **{field: value})


def owner():
    names = ('_make_model', '_training_loader', '_development_loader', '_forward',
             '_make_training_state', '_optimizer_step', '_evaluate', '_save_checkpoint')
    result = SimpleNamespace(**{name: lambda *a, **k: None for name in names},
        PHYSICAL_BATCH=128, DEVELOPMENT_ROWS=130, SEED=42, LOADER_WORKERS=4,
        LEARNING_RATE=.001, _weight_decay=lambda v: .05,
        _scientific_fields=lambda v='reference': {'precision': 'fp32', 'seed': 42})
    result._forward = lambda model, batch: model(batch.y.view(-1, 1)).view(-1)
    return result


def profile(cache):
    return ExecutionTuning(training_workers=0, development_workers=0,
                           independent_models=1, total_cpu_budget=2, cache_development=cache)


def test_cached_live_ema_results_rng_and_owner_restore_match_streaming(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    data = graphs()
    model = torch.nn.Linear(1, 1)
    ema = ExponentialMovingAverage(model, .999)
    with torch.no_grad():
        model.weight.add_(.1)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    outputs, states = [], []
    for cached in (False, True):
        native = owner()
        original = {k: getattr(native, k) for k in vars(native) if k.startswith('_')}
        torch.manual_seed(420)
        with tuned_backend(native, profile(cached)) as report:
            output = [native._evaluate(model, ema, data, torch.tensor(0.), torch.tensor(1.),
                         device='cpu', weights=weights) for weights in ('ema', 'live', 'ema', 'live')]
            outputs.append(output)
            states.append(torch.get_rng_state())
            assert report['phase_timing']['evaluation_passes'] == 4
            assert native._scientific_fields()['execution_backend'].startswith('optimized-tuning-v2:')
            assert all(torch.equal(initial[k], model.state_dict()[k]) for k in initial)
            if cached:
                assert report['development_caches'][0]['cold_passes'] == 1
                assert report['development_caches'][0]['warm_passes'] == 3
        assert all(getattr(native, k) is v for k, v in original.items())
    assert torch.equal(states[0], states[1])
    for baseline, candidate in zip(*outputs):
        assert baseline['mae_eV'] == candidate['mae_eV']
        assert all(torch.equal(baseline[k], candidate[k])
                   for k in ('prediction_eV', 'target_eV', 'source_idx'))


def test_exception_restores_threads_environment_weights_and_methods(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    monkeypatch.delenv('MKL_NUM_THREADS', raising=False)
    native, data = owner(), graphs()
    original_forward = native._forward
    model = torch.nn.Linear(1, 1)
    ema = ExponentialMovingAverage(model, .999)
    with torch.no_grad():
        model.weight.add_(1.)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    threads = torch.get_num_threads()
    def fail(*args):
        raise RuntimeError('synthetic forward fault')
    native._forward = fail
    with pytest.raises(RuntimeError, match='synthetic forward'):
        with tuned_backend(native, profile(True)):
            assert torch.get_num_threads() == 1
            assert os.environ['OMP_NUM_THREADS'] == '1'
            native._evaluate(model, ema, data, torch.tensor(0.), torch.tensor(1.), device='cpu')
    assert torch.get_num_threads() == threads
    assert os.environ['OMP_NUM_THREADS'] == '7' and 'MKL_NUM_THREADS' not in os.environ
    assert native._forward is fail and native._forward is not original_forward
    assert all(torch.equal(initial[k], model.state_dict()[k]) for k in initial)


def test_changed_resource_or_cache_configuration_has_distinct_identity():
    base = ExecutionTuning()
    assert replace(base, cache_development=False).identity != base.identity
    assert replace(base, training_workers=0).identity != base.identity
    assert base.environment() == {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def test_context_does_not_accept_a_different_physical_batch():
    native = owner()
    native.PHYSICAL_BATCH = 64
    with pytest.raises(ValueError, match='BS128'):
        with tuned_backend(native, profile(True)):
            pass


def test_worker_cleanup_fault_still_restores_runtime_and_owner(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    monkeypatch.delenv('MKL_NUM_THREADS', raising=False)
    native, data = owner(), graphs()
    previous = native._development_loader
    threads = torch.get_num_threads()
    def fail():
        raise RuntimeError('synthetic shutdown fault')
    with pytest.raises(RuntimeError, match='shutdown fault'):
        with tuned_backend(native, profile(False)):
            wrapper = native._development_loader(data)
            wrapper.loader._iterator = SimpleNamespace(_shutdown_workers=fail)
    assert torch.get_num_threads() == threads
    assert os.environ['OMP_NUM_THREADS'] == '7' and 'MKL_NUM_THREADS' not in os.environ
    assert native._development_loader is previous


def test_training_loader_matches_native_epoch_order_without_main_rng_draw():
    from molgap import pcqm_gptrans_v4 as recipe
    native = owner()
    native.DeterministicEpochBatchSampler = recipe.DeterministicEpochBatchSampler
    # Exercise the real frozen 100K sampler without loading any molecule data.
    data = [None] * recipe.TRAIN_ROWS
    with tuned_backend(native, profile(True)):
        for epoch in (0, 1, 39):
            torch.manual_seed(123)
            initial = torch.get_rng_state().clone()
            batches = list(native._training_loader(data, epoch).loader.batch_sampler)
            expected = list(recipe.DeterministicEpochBatchSampler(len(data), epoch))
            assert batches == expected
            assert all(len(b) == 128 for b in batches)
            assert torch.equal(initial, torch.get_rng_state())


@pytest.mark.parametrize('kind', ['training', 'development'])
def test_tuned_loaders_reject_transformed_dataset(kind):
    native = owner()
    data = SimpleNamespace(transform=lambda x: x)
    with tuned_backend(native, profile(True)):
        with pytest.raises(ValueError, match='transforms'):
            if kind == 'training':
                native._training_loader(data, 0)
            else:
                native._development_loader(data)


def test_warm_cache_skips_source_and_layout_preparation():
    class CountedSource:
        calls = 0
        def __iter__(self):
            self.calls += 1
            yield from source(graphs())
    counted = CountedSource()
    cache = PreparedDevelopmentCache(counted, rows=130, batch_size=128, max_bytes=100000)
    list(cache)
    list(cache)
    assert counted.calls == 1


@pytest.mark.parametrize('start', [-1, True, '100000'])
def test_cache_rejects_invalid_role_offset(start):
    with pytest.raises(ValueError):
        PreparedDevelopmentCache([], rows=130, batch_size=128, max_bytes=100000,
                                 source_start=start)
