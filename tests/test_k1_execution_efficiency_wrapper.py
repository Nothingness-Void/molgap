"""AST-extracted wrapper tests: no worker imports, tensors, data or GPU use."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


WRAPPER = Path(__file__).resolve().parents[1] / 'experiments/pcqm_k1_execution_efficiency/run.py'


def source_tree():
    return ast.parse(WRAPPER.read_text(encoding='utf-8'))


def helpers(namespace):
    names = {'_guard_deadline', '_verify_cpu_optimizer_steps', '_checkpoint_resume'}
    functions = [node for node in source_tree().body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(WRAPPER), 'exec'), namespace)
    return namespace


def optimizer_mock(step_device='cpu', **flags):
    return SimpleNamespace(
        param_groups=[{'params': ['p'], 'fused': False, 'capturable': False, **flags}],
        state={'p': {'step': SimpleNamespace(device=SimpleNamespace(type=step_device))}})


class FakePath:
    def __init__(self, events):
        self.events = events
        self.parent = self

    def mkdir(self, **kwargs):
        self.events.append('mkdir')

    def stat(self):
        self.events.append('stat')
        return SimpleNamespace(st_size=1234)

    def __str__(self):
        return 'mock-scratch.pt'


def checkpoint_fixture(*, expire_at=None):
    events, clock, saved = [], {'now': 0.}, {}

    class Model:
        def to(self, device):
            return self

        def train(self):
            return self

        def state_dict(self):
            return {'p': 'mock-state'}

        def load_state_dict(self, state, strict):
            assert strict and state == {'p': 'mock-state'}

    class Optimizer:
        def __init__(self):
            self.param_groups = optimizer_mock().param_groups
            self.state = optimizer_mock().state

        def state_dict(self):
            return {'state': deepcopy(self.state), 'param_groups': deepcopy(self.param_groups)}

        def load_state_dict(self, state):
            self.state = state['state']
            self.param_groups = state['param_groups']

    class Batch:
        def clone(self):
            return Batch()

        def to(self, device):
            return self

    def event(name):
        events.append(name)
        if name == expire_at:
            clock['now'] = 10.

    def step(*args):
        event(f'step{sum(e.startswith("step") for e in events) + 1}')
        return 'mock-loss'

    def save(path, checkpoint):
        event('save')
        saved['checkpoint'] = deepcopy(checkpoint)

    def load(path, *, map_location, weights_only):
        assert map_location == 'cpu'
        assert not weights_only
        event('load')
        return deepcopy(saved['checkpoint'])

    @contextmanager
    def context(*args):
        event('context_enter')
        try:
            yield
        finally:
            events.append('context_exit')

    namespace = helpers({
        'time': SimpleNamespace(perf_counter=lambda: clock['now']),
        'capture_rng_state': lambda: {'rng': 'mock-state'},
        'restore_rng_state': lambda state: event('restore_rng'),
        'atomic_torch_save': save,
        'sha256_file': lambda path: 'mock-hash',
    })
    arguments = dict(torch=SimpleNamespace(load=load, cuda=SimpleNamespace(synchronize=lambda: event('sync'))),
        factory=Model, optimizer_for=lambda model: Optimizer(), step=step, batches=[Batch(), Batch()],
        mean=0., std=1., device='cuda', deadline=10., checkpoint_path=FakePath(events),
        cpu_layout_context=context, compare_states=lambda a, b: {'exact': a == b})
    return namespace, arguments, events, clock


def test_cpu_load_scalar_state_and_publication_timer_order():
    namespace, arguments, events, clock = checkpoint_fixture()
    original_timer = namespace['time'].perf_counter

    def timer():
        events.append('timer')
        return original_timer()

    namespace['time'].perf_counter = timer
    result = namespace['_checkpoint_resume'](**arguments)
    assert result['status'] == 'complete'
    assert result['publication']['bytes'] == 1234
    assert result['verified_cpu_step_states'] == 1
    assert sum(e.startswith('step') for e in events) == 3
    assert events.index('step1') < events.index('sync') < events.index('save') < events.index('load')
    assert events[events.index('save') - 1] == 'timer'
    assert result['comparison']['exact']


@pytest.mark.parametrize('expire_at,forbidden', [
    ('step1', 'save'), ('save', 'load'), ('load', 'step2'),
    ('step2', 'context_enter'), ('context_enter', 'step3'), ('step3', 'compare')])
def test_deadline_stops_followup_work_and_cleans_context(expire_at, forbidden):
    namespace, arguments, events, clock = checkpoint_fixture(expire_at=expire_at)
    arguments['compare_states'] = lambda a, b: events.append('compare')
    with pytest.raises(TimeoutError):
        namespace['_checkpoint_resume'](**arguments)
    assert forbidden not in events
    if 'context_enter' in events:
        assert events[-1] == 'context_exit'


def test_expired_deadline_blocks_model_construction():
    namespace, arguments, events, clock = checkpoint_fixture()
    clock['now'] = 10.
    arguments['factory'] = lambda: pytest.fail('Construction after deadline')
    with pytest.raises(TimeoutError):
        namespace['_checkpoint_resume'](**arguments)


def test_optimizer_cpu_step_check_rejects_gpu_and_missing_state():
    namespace = helpers({})
    check = namespace['_verify_cpu_optimizer_steps']
    assert check(optimizer_mock()) == 1
    with pytest.raises(RuntimeError, match='remain on CPU'):
        check(optimizer_mock('cuda'))
    empty = optimizer_mock()
    empty.state = {}
    with pytest.raises(RuntimeError, match='No unfused'):
        check(empty)


@pytest.mark.parametrize('probe_status,helper_timeout,expected', [
    ('budget_exhausted', False, 'skipped'), ('complete', True, 'budget_exhausted')])
def test_wrapper_emits_missing_resume_without_false_completion(tmp_path, probe_status, helper_timeout, expected):
    run = next(n for n in source_tree().body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'resume' for t in n.targets))
    writes = {}

    def resume(**kwargs):
        if not helper_timeout:
            pytest.fail('Checkpoint work must not follow an exhausted probe')
        raise TimeoutError('mock budget exhausted')

    namespace = dict(report={'status': probe_status, 'cases': {}},
        time=SimpleNamespace(perf_counter=lambda: 5., process_time=lambda: 1.),
        started=0., cpu_started=0., wall_limit=10., deadline=10., output=tmp_path,
        device='cpu', ROOT=tmp_path, torch=None, factory=None, optimizer_for=None,
        step=None, batches=[], mean=0., std=1., cpu_layout_context=None, compare_states=None,
        _checkpoint_resume=resume, atomic_json=lambda path, value: writes.update({path.name: value}),
        sha256_file=lambda path: 'mock-hash', json=SimpleNamespace(dumps=lambda value: value),
        print=lambda value: None)
    exec(compile(ast.Module(body=run.body[start:], type_ignores=[]), str(WRAPPER), 'exec'), namespace)
    assert writes['checkpoint_resume.json']['status'] == expected
    assert writes['checkpoint_resume.json']['comparison'] is None
    assert writes['completion.json']['status'] == 'budget_exhausted'
    assert writes['completion.json']['missing_evidence'] == ['checkpoint_resume']


def test_source_verification_precedes_input_decode_and_probe_uses_remaining_wall():
    run = next(n for n in source_tree().body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    checks = [n for n in ast.walk(run) if isinstance(n, ast.Raise) and n.exc is not None
              and 'Source bytes changed' in ast.unparse(n.exc)]
    loads = [n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and ast.unparse(n.func) == 'torch.load']
    assert checks and loads
    assert checks[0].lineno < min(n.lineno for n in loads)
    probe = next(n for n in ast.walk(run) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'probe_k1_execution_efficiency')
    wall = next(k.value for k in probe.keywords if k.arg == 'wall_seconds')
    assert ast.unparse(wall) == 'deadline - time.perf_counter()'
