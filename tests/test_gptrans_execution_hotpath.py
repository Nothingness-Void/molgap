"""Synthetic CPU state/transfer tests, not molecular training or inference."""
from copy import copy
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap.gptrans_execution_backend import prepare_layout
from molgap.gptrans_execution_hotpath import (
    BoundForeachEMA, BatchTransferView, ReusedWeightView, development_cpu_view, LoadedDatasetCache,
)
from molgap.pcqm_gptrans_step_profile import ForeachExponentialMovingAverage


def layer():
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Linear(4, 1))
    model.register_buffer('counter', torch.tensor(0))
    return model


@pytest.mark.parametrize('decay', [.999, .9999])
def test_bound_ema_exactly_matches_v1_across_updates_and_resume(decay):
    model = layer()
    expected, actual = ForeachExponentialMovingAverage(model, decay), BoundForeachEMA(model, decay)
    rng = torch.get_rng_state().clone()
    for step in range(10):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(.001 * (step + 1))
            model.counter.add_(1)
        expected.update(model)
        actual.update(model)
        assert all(torch.equal(actual.state[k], v) for k, v in expected.state.items())
        if step == 4:
            actual.load_state_dict(expected.state_dict())
    assert actual.plan_builds == 2 and actual.update_calls == 10
    assert torch.equal(rng, torch.get_rng_state())
    assert all(v.grad_fn is None for v in actual.state.values())


def test_warm_ema_never_reconstructs_model_state_dict(monkeypatch):
    model = layer()
    actual = BoundForeachEMA(model, .999)
    original = model.state_dict
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'state_dict', counted)
    for _ in range(8):
        actual.update(model)
    assert calls == [{'keep_vars': True}]
    assert actual.plan_builds == 1


@pytest.mark.parametrize('mutation', ['parameter', 'buffer', 'module', 'storage', 'stride', 'ema_shape', 'ema_tensor', 'extra'])
def test_ema_rejects_stale_bindings_before_any_update(mutation):
    model = layer()
    actual = BoundForeachEMA(model, .999)
    actual.update(model)
    if mutation == 'parameter':
        model[0].weight = torch.nn.Parameter(model[0].weight.detach().clone())
    elif mutation == 'buffer':
        model.counter = model.counter.clone()
    elif mutation == 'module':
        model[0] = torch.nn.Linear(2, 4)
    elif mutation == 'storage':
        model[0].weight.data = model[0].weight.detach().clone()
    elif mutation == 'ema_tensor':
        actual.state['0.weight'] = actual.state['0.weight'].clone()
    elif mutation == 'stride':
        model[0].weight.data = model[0].weight.detach().reshape(2, 4).t()
    elif mutation == 'ema_shape':
        actual.state['0.weight'].resize_(2, 4)
    else:
        model.register_buffer('unexpected', torch.tensor(1.))
    before = {k: v.clone() for k, v in actual.state.items()}
    with pytest.raises(RuntimeError, match='EMA'):
        actual.update(model)
    assert all(torch.equal(actual.state[k], v) for k, v in before.items())


def test_ema_load_rebinds_replaced_model_tensor():
    model = layer()
    actual = BoundForeachEMA(model, .999)
    actual.update(model)
    model[0].weight = torch.nn.Parameter(model[0].weight.detach().clone())
    actual.load_state_dict(actual.state_dict())
    actual.update(model)
    assert actual.plan_builds == 2


def test_bound_ema_rejects_another_model_and_bad_resume_keys():
    model = layer()
    actual = BoundForeachEMA(model, .999)
    actual.update(model)
    with pytest.raises(RuntimeError, match='binding changed'):
        actual.update(layer())
    with pytest.raises(RuntimeError, match='identity changed'):
        actual.load_state_dict({})


@pytest.mark.parametrize('training', [False, True])
def test_selective_transfer_leaves_unused_fields_and_eval_targets_on_cpu(training):
    batch = prepare_layout(Batch.from_data_list([Data(
        x=torch.ones(2, 1, dtype=torch.long), edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.ones(2, 1, dtype=torch.long), y=torch.tensor([.2]),
        source_idx=torch.tensor([100000]), random_walk_pe=torch.ones(2, 16))]))
    output = BatchTransferView(copy(batch), training=training).to('meta', non_blocking=True)
    assert output.x.device.type == 'meta'
    assert output.execution_layout['mask'].device.type == 'meta'
    assert output.y.device.type == ('meta' if training else 'cpu')
    assert output.random_walk_pe.device.type == 'cpu'
    assert output.source_idx.device.type == 'cpu' and output.ptr.device.type == 'cpu'
    assert batch.x.device.type == 'cpu' and batch.execution_layout['mask'].device.type == 'cpu'


def test_weight_backup_reuses_storage_restores_live_and_preserves_rng():
    model = layer()
    ema = ForeachExponentialMovingAverage(model, .999)
    with torch.no_grad():
        model[0].weight.add_(1.)
        model.counter.add_(3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    view = ReusedWeightView()
    rng = torch.get_rng_state().clone()
    pointers = None
    for _ in range(3):
        with view.use(model, ema, 'ema'):
            assert all(torch.equal(model.state_dict()[k], v) for k, v in ema.state.items())
        assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
        current = {k: v.data_ptr() for k, v in view.backup.items()}
        if pointers is not None:
            assert current == pointers
        pointers = current
    assert view.allocations == 1
    assert view.retained_tensor_bytes == sum(v.numel() * v.element_size() for v in before.values())
    assert torch.equal(rng, torch.get_rng_state())
    view.clear()
    assert view.backup is None
    assert view.retained_tensor_bytes == 0


def test_cpu_optimizer_gradient_and_resume_parity_with_bound_ema():
    # Elementary CPU arithmetic regression, not a MolGap model/training run.
    left, right = layer(), layer()
    right.load_state_dict(left.state_dict())
    expected = ForeachExponentialMovingAverage(left, .999)
    actual = BoundForeachEMA(right, .999)
    opts = [torch.optim.AdamW(m.parameters(), lr=.001, foreach=True) for m in (left, right)]
    x, target = torch.tensor([[.2, .4]]), torch.tensor([[.1]])
    for step in range(4):
        gradients = []
        for model, opt, ema in zip((left, right), opts, (expected, actual)):
            opt.zero_grad(set_to_none=True)
            (model(x) - target).abs().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            gradients.append([p.grad.clone() for p in model.parameters()])
            opt.step()
            ema.update(model)
        assert all(torch.equal(a, b) for a, b in zip(*gradients))
        assert all(torch.equal(left.state_dict()[k], v) for k, v in right.state_dict().items())
        assert all(torch.equal(expected.state[k], v) for k, v in actual.state.items())
        for a, b in zip(opts[0].state.values(), opts[1].state.values()):
            assert all(torch.equal(a[k], b[k]) for k in a)
        if step == 1:
            actual.load_state_dict({k: v.clone() for k, v in actual.state_dict().items()})
            opts[1].load_state_dict(opts[1].state_dict())


@pytest.mark.parametrize('fault', ['body', 'partial_load'])
def test_weight_view_restores_on_body_or_partial_load_failure(fault):
    model = layer()
    ema = ForeachExponentialMovingAverage(model, .999)
    with torch.no_grad():
        model[0].weight.add_(1.)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    if fault == 'partial_load':
        state = dict(ema.state_dict())
        del state['1.bias']
        ema = SimpleNamespace(state_dict=lambda: state)
    view = ReusedWeightView()
    with pytest.raises(RuntimeError):
        with view.use(model, ema, 'ema'):
            raise RuntimeError('synthetic body fault')
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    assert not view.busy


def test_weight_view_live_does_not_allocate_and_nested_views_are_rejected():
    model, view = layer(), ReusedWeightView()
    with view.use(model, None, 'live'):
        with pytest.raises(ValueError, match='nested'):
            with view.use(model, None, 'live'):
                pass
    assert view.allocations == 0 and view.backup is None


def test_weight_view_preserves_native_load_hooks():
    model = layer()
    ema, calls = ForeachExponentialMovingAverage(model, .999), []
    handle = model.register_load_state_dict_post_hook(lambda m, incompatible: calls.append(m))
    with ReusedWeightView().use(model, ema, 'ema'):
        pass
    handle.remove()
    assert calls == [model, model]


def test_development_view_drops_unused_storage_without_mutating_source():
    batch = prepare_layout(Batch.from_data_list([Data(
        x=torch.ones(2, 1), edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, 1), y=torch.tensor([.2]), source_idx=torch.tensor([100000]),
        random_walk_pe=torch.ones(2, 16), pos=torch.ones(2, 3))]))
    result = development_cpu_view(batch)
    assert 'random_walk_pe' not in result and 'pos' not in result
    assert 'random_walk_pe' in batch and 'pos' in batch
    assert 'pos' not in result._slice_dict and 'pos' in batch._slice_dict
    assert result.num_graphs == batch.num_graphs
    assert result.x.data_ptr() == batch.x.data_ptr()
    assert torch.equal(result.y, batch.y) and torch.equal(result.source_idx, batch.source_idx)


def test_dataset_reuse_avoids_duplicate_deserialization_and_clears(tmp_path):
    paths = (tmp_path / 'train.pt', tmp_path / 'dev.pt')
    for p in paths:
        p.write_bytes(b'synthetic data')
    calls = []
    def load(inputs):
        calls.append(inputs)
        return object(), object()
    cache = LoadedDatasetCache(load)
    first = cache((paths[0],))
    assert cache((str(paths[0]),)) is first
    cache((paths[1],))
    assert len(calls) == 2 and cache.statistics == dict(loads=2, hits=1)
    cache.clear()
    assert not cache.entries


@pytest.mark.parametrize('when', ['before_hit', 'during_load'])
def test_dataset_cache_rejects_file_changes(tmp_path, when):
    path = tmp_path / 'train.pt'
    path.write_bytes(b'synthetic')
    def load(inputs):
        if when == 'during_load':
            path.write_bytes(b'synthetic changed')
        return object()
    cache = LoadedDatasetCache(load)
    if when == 'before_hit':
        cache((path,))
        path.write_bytes(b'synthetic changed')
    with pytest.raises(RuntimeError, match='changed'):
        cache((path,))


def test_dataset_cache_rejects_missing_empty_or_third_role(tmp_path):
    cache = LoadedDatasetCache(lambda paths: object())
    with pytest.raises(RuntimeError, match='one frozen'):
        cache(())
    with pytest.raises(FileNotFoundError):
        cache((tmp_path / 'missing',))
    for name in ('train', 'dev'):
        path = tmp_path / name
        path.write_bytes(b'synthetic')
        cache((path,))
    extra = tmp_path / 'extra'
    extra.write_bytes(b'synthetic')
    with pytest.raises(RuntimeError, match='one frozen'):
        cache((extra,))
