from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch

from molgap.k1_execution_efficiency import compare_states, probe_execution_efficiency
from molgap.training_reproducibility import capture_rng_state


class Batch:
    num_graphs = 4

    def __init__(self):
        self.x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7
        self.y = torch.arange(4, dtype=torch.float32)

    def to(self, device, *, non_blocking=False):
        self.x = self.x.to(device, non_blocking=non_blocking)
        self.y = self.y.to(device, non_blocking=non_blocking)
        return self


def factory():
    return torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3),
                               torch.nn.Dropout(0.3), torch.nn.Linear(3, 1))


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5,
                             foreach=False, fused=False)


def step(model, opt, batch, mean, std):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.l1_loss(model(batch.x).view(-1), (batch.y - mean) / std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    opt.step()
    return loss.detach()


def run(**kwargs):
    options = dict(warmup=1, steps=3, repeats=2, wall_seconds=30)
    options.update(kwargs)
    return probe_execution_efficiency(factory, [Batch() for _ in range(4)], 0., 1.,
        optimizer_factory=optimizer, step_fn=step, **options)


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_exact_dropout_buffers_optimizer_and_rng_replay_without_caller_mutation():
    rng = capture_rng_state()
    batches = [Batch() for _ in range(4)]
    before = deepcopy(batches[0].x)
    result = probe_execution_efficiency(factory, batches, 0., 1.,
        optimizer_factory=optimizer, step_fn=step, warmup=1, steps=3, repeats=2)
    assert result['status'] == 'complete'
    assert compare_states(rng, capture_rng_state())['exact']
    assert torch.equal(before, batches[0].x)
    for row in result['cases'].values():
        assert row['equivalence']['exact']
        assert row['execution_only_admissible']
        assert len(row['timings']) == 2
        assert row['timings'][0]['peak_allocated_bytes'] is None
    assert result['timing_order'][0] == result['timing_order'][1][::-1]
    assert not result['contract_promotion']
    assert result['compile']['status'] == 'deferred'


def test_extra_context_order_and_exact_gate():
    events = []

    @contextmanager
    def context(model, batch, device):
        assert batch.x.device.type == 'cpu'
        events.append('enter')
        model.probe_context = True
        yield
        events.append('exit')

    result = run(extra_cases={'cpu_layout': context})
    assert result['cases']['cpu_layout']['equivalence']['exact']
    assert events == ['enter', 'exit'] * 11


def test_changed_buffers_fail_execution_gate():
    @contextmanager
    def context(model, batch, device):
        yield
        model[1].running_mean.add_(0.1)

    row = run(extra_cases={'bad_layout': context})['cases']['bad_layout']
    assert not row['equivalence']['exact']
    assert not row['execution_only_admissible']
    assert not row['speed_candidate']
    assert any('running_mean' in p for p in row['equivalence']['differences'])


def test_optimizer_variants_never_promote_even_when_exact():
    result = run(optimizer_cases=('foreach', 'fused'), atol=1e-6, rtol=1e-5)
    for name in ('foreach', 'fused'):
        row = result['cases'][name]
        assert row['optimizer_contract_change']
        assert not row['execution_only_admissible']
        assert not row['speed_candidate']
        if 'equivalence' in row:
            assert row['equivalence']['atol'] == 1e-6


def test_comparator_detects_full_tensor_keys_shapes_and_nonfinite():
    a = {'p': torch.tensor([1., 2.]), 'rng': torch.tensor([1], dtype=torch.uint8)}
    b = deepcopy(a)
    b['p'][1] += 1e-5
    report = compare_states(a, b, atol=1e-4)
    assert not report['exact'] and report['within_tolerance']
    assert report['max_abs_error'] > 0
    assert not compare_states(a, {**a, 'extra': None})['exact']
    assert not compare_states(a, {**a, 'p': torch.ones(1)})['exact']
    assert not compare_states(torch.tensor([float('nan')]), torch.tensor([float('nan')]))['exact']


def test_budget_exhaustion_preserves_rng_and_does_not_claim_gate(monkeypatch):
    import molgap.k1_execution_efficiency as module
    rng = capture_rng_state()
    clock = iter([0., 2., 3., 4.])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(clock))
    result = run(wall_seconds=1)
    assert result['status'] == 'budget_exhausted'
    assert compare_states(rng, capture_rng_state())['exact']


@pytest.mark.parametrize('kwargs', [dict(steps=1), dict(repeats=1),
    dict(repeats=3), dict(warmup=65), dict(wall_seconds=-1), dict(single_stream=False),
    dict(extra_cases={'baseline': None}), dict(optimizer_cases=('other',))])
def test_invalid_bounds_and_cases_fail(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)


def test_real_api_rejects_non128_without_execution():
    with pytest.raises(ValueError):
        run(real_run=True)


def test_nonfinite_forward_retains_guards():
    @contextmanager
    def context(model, batch, device):
        batch.x.fill_(float('nan'))
        yield

    result = run(extra_cases={'nonfinite': context})
    row = result['cases']['nonfinite']
    assert row['status'] == 'failed_or_unsupported'
    assert 'Non-finite forward' in row['reason']


def test_context_uses_same_batch_object_as_transfer_and_step():
    retained = [Batch() for _ in range(4)]

    @contextmanager
    def context(model, batch, device):
        assert all(batch is not original for original in retained)
        model.context_batch = batch
        yield
        del model.context_batch

    def checked_step(model, opt, batch, mean, std):
        if hasattr(model, 'context_batch'):
            assert model.context_batch is batch
        return step(model, opt, batch, mean, std)

    result = probe_execution_efficiency(factory, retained, 0., 1.,
        optimizer_factory=optimizer, step_fn=checked_step, warmup=1, steps=3,
        extra_cases={'layout': context})
    assert result['cases']['layout']['equivalence']['exact']


def test_comparison_detects_gradient_and_rng_drift():
    value = {'gradients_before_clip': {'unused': None, 'p': torch.tensor([1., 2.])},
             'rng': {'torch': torch.tensor([1, 2], dtype=torch.uint8)}}
    altered = deepcopy(value)
    altered['gradients_before_clip']['p'][0] += 1
    altered['rng']['torch'][1] = 3
    result = compare_states(value, altered)
    assert not result['exact']
    assert set(result['differences']) == {
        'state.gradients_before_clip.p', 'state.rng.torch'}


def test_nonfp32_inputs_and_autocast_rejected():
    batches = [Batch() for _ in range(4)]
    batches[0].x = batches[0].x.double()
    with pytest.raises(ValueError, match='Floating inputs'):
        probe_execution_efficiency(factory, batches, 0., 1.,
            optimizer_factory=optimizer, step_fn=step, warmup=1, steps=3)
    with torch.autocast('cpu'):
        with pytest.raises(ValueError, match='Autocast'):
            run()


def test_separate_equivalence_bound_and_checkpoint_roundtrip(tmp_path):
    result = run(equivalence_steps=2, return_checkpoint_snapshots=True)
    assert result['equivalence_updates'] == 2
    for row in result['cases'].values():
        checkpoint = row['final_checkpoint']
        path = tmp_path / 'scratch.pt'
        torch.save(checkpoint, path)
        restored = torch.load(path, weights_only=False)
        assert compare_states(checkpoint, restored)['exact']
        assert row['end_rng']['torch'] == checkpoint['rng']['torch'].tolist()
        assert row['checkpoint_structure']['keys'] == ['model', 'optimizer', 'rng']
        assert len(row['paired_repeat_savings_fraction']) == 2
        assert row['min_seconds_per_step'] <= row['median_seconds_per_step'] <= row['max_seconds_per_step']


def test_real_prediction_guard_checks_numel_dtype_and_finiteness_cpu_only():
    from molgap.k1_execution_efficiency import _finite_output

    _finite_output(None, (), torch.zeros(128, 1), expected_predictions=128)
    _finite_output(None, (), (torch.zeros(128), torch.zeros(())), expected_predictions=128)
    for prediction in (torch.zeros(127), torch.zeros(128, dtype=torch.float64),
                       torch.full((128,), float('nan'))):
        with pytest.raises(RuntimeError):
            _finite_output(None, (), prediction, expected_predictions=128)
