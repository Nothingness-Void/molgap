"""CPU-only relocation parity against the frozen pre-move Git implementations.

Run with pytest's basetemp/cache and Python bytecode redirected to E:.
No retained checkpoints, datasets, platform access or accelerators are used.
"""
import ast
import importlib
from io import BytesIO
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
BEFORE_MOVE = "a7bc58de"
MODULES = (
    ("molgap.gptrans", "molgap.models.gptrans", "src/molgap/gptrans.py"),
    ("molgap.model_substrate", "molgap.models.substrate", "src/molgap/model_substrate.py"),
)


def _frozen_source(path):
    return subprocess.run(
        ["git", "show", f"{BEFORE_MOVE}:{path}"], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout


@pytest.fixture
def frozen_gptrans(monkeypatch):
    # Execute the actual old source in memory instead of copying its algorithm.
    name = "_molgap_gptrans_before_namespace_move"
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(_frozen_source(MODULES[0][2]), MODULES[0][2], "exec"), module.__dict__)
    return module


def _tiny_model(module):
    return module.OGBGPTransTiny(
        node_channels=8, pair_channels=4, num_layers=2, num_heads=2,
        shortest_path_cap=3, dropout=0.1, drop_path=0.1,
    ).cpu()


def _inputs():
    import torch

    return (
        torch.zeros(5, 9, dtype=torch.long),
        torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]),
        torch.zeros(6, 3, dtype=torch.long),
        torch.tensor([0, 0, 0, 1, 1]),
    )


def _assert_tensors_equal(actual, expected):
    import torch

    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("old,new,path", MODULES)
def test_aliases_keep_all_original_symbols_and_pickle_identity(old, new, path):
    legacy = importlib.import_module(old)
    implementation = importlib.import_module(new)
    assert legacy is implementation
    original_definitions = {
        node.name: node for node in ast.parse(_frozen_source(path)).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    moved_definitions = {
        node.name: node for node in ast.parse(Path(implementation.__file__).read_text(encoding="utf-8")).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert original_definitions.keys() <= moved_definitions.keys()
    for name, original in original_definitions.items():
        value = getattr(legacy, name)
        assert value is getattr(implementation, name)
        assert value.__module__ == old
        assert pickle.loads(pickle.dumps(value)) is value


def test_substrate_slots_match_frozen_source_forward_and_gradients(monkeypatch):
    from dataclasses import replace
    import torch
    from molgap.models import gptrans, substrate

    name = "_molgap_substrate_before_namespace_move"
    frozen = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, frozen)
    exec(compile(_frozen_source(MODULES[1][2]), MODULES[1][2], "exec"), frozen.__dict__)

    class Shift(torch.nn.Module):
        def __init__(self, field):
            super().__init__()
            self.field = field
            self.shift = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, frame):
            return replace(frame, **{self.field: getattr(frame, self.field) + self.shift})

    observations = []
    with torch.random.fork_rng(devices=[]):
        for module in (frozen, substrate):
            torch.manual_seed(42)
            model = _tiny_model(gptrans).eval()
            owner = module.ModelSubstrate(model, "gptrans_t")
            for slot, field, layer in (
                (module.SlotName.PRE_MESSAGE, "node", 1),
                (module.SlotName.POST_AGGREGATION, "node_update", 1),
                (module.SlotName.PRE_READOUT, "embedding", None),
            ):
                owner.attach(slot.value, Shift(field), module.SlotBinding("gptrans_t", slot, layer))
            owner.seal()
            try:
                output = model(*_inputs())
                loss = output.square().mean()
                loss.backward()
                gradients = {
                    key: parameter.grad.detach().clone()
                    for key, parameter in model.named_parameters() if parameter.grad is not None
                }
                addon_names = [key for key, _ in owner.named_addon_parameters()]
                assert len(addon_names) == 3
                assert all(key in gradients for key in addon_names)
                observations.append((output.detach(), loss.detach(), gradients, model.state_dict()))
            finally:
                assert not owner._active
        for actual, expected in zip(observations[1][:2], observations[0][:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tensors_equal(observations[1][2], observations[0][2])
        _assert_tensors_equal(observations[1][3], observations[0][3])


@pytest.mark.parametrize("first_new", [False, True])
@pytest.mark.parametrize("old,new,path", MODULES)
def test_fresh_import_order_and_lazy_namespace(old, new, path, first_new, tmp_path):
    del path
    code = f"""
import importlib
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
import molgap.models
assert 'torch' not in sys.modules
assert 'molgap.gptrans' not in sys.modules
assert 'molgap.models.gptrans' not in sys.modules
assert 'molgap.model_substrate' not in sys.modules
assert 'molgap.models.substrate' not in sys.modules
first = importlib.import_module({(new if first_new else old)!r})
second = importlib.import_module({(old if first_new else new)!r})
assert first is second
if {old!r} == 'molgap.model_substrate':
    assert 'torch' not in sys.modules
assert 'molgap.experiment_cli' not in sys.modules
assert 'molgap.family_training_core' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gptrans_training_update_is_exactly_equal_to_frozen_source(frozen_gptrans):
    import torch
    from molgap.models import gptrans

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        before = _tiny_model(frozen_gptrans)
        torch.manual_seed(42)
        after = _tiny_model(gptrans)
        _assert_tensors_equal(after.state_dict(), before.state_dict())
        assert sum(p.numel() for p in after.parameters()) == sum(
            p.numel() for p in before.parameters()
        )
        target = torch.tensor([[0.2], [-0.4]])
        observations = []
        for model in (before, after):
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            torch.manual_seed(123)
            output = model(*_inputs())
            loss = torch.nn.functional.l1_loss(output, target)
            loss.backward()
            gradients = {
                name: None if p.grad is None else p.grad.detach().clone()
                for name, p in model.named_parameters()
            }
            optimizer.step()
            observations.append((output.detach(), loss.detach(), gradients))
        for actual, expected in zip(observations[1][:2], observations[0][:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(observations[1][0]).all()
        assert observations[1][2].keys() == observations[0][2].keys()
        for name, actual in observations[1][2].items():
            expected = observations[0][2][name]
            if expected is None:
                assert actual is None, name
            else:
                assert actual is not None and torch.isfinite(actual).all(), name
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
        assert any(
            gradient is not None and torch.count_nonzero(gradient)
            for gradient in observations[1][2].values()
        )
        _assert_tensors_equal(after.state_dict(), before.state_dict())


def test_frozen_state_dict_strict_loads_both_directions(frozen_gptrans):
    import torch
    from molgap.models import gptrans

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        before = _tiny_model(frozen_gptrans).eval()
        after = _tiny_model(gptrans).eval()
        buffer = BytesIO()
        torch.save(before.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        result = after.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        result = before.load_state_dict(after.state_dict(), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        with torch.no_grad():
            torch.testing.assert_close(after(*_inputs()), before(*_inputs()), rtol=0, atol=0)


@pytest.mark.parametrize("old,new,path", MODULES)
def test_real_historical_pickle_resolves_moved_classes(old, new, path, monkeypatch):
    import torch

    implementation = importlib.import_module(new)
    importlib.import_module(old)
    historical = ModuleType(old)
    # Produce historical bytes with the old source at its original module path.
    # Restoring the alias before loads proves old pickle resolution still works.
    with torch.random.fork_rng(devices=[]), monkeypatch.context() as patch:
        patch.setitem(sys.modules, old, historical)
        exec(compile(_frozen_source(path), path, "exec"), historical.__dict__)
        if old == "molgap.gptrans":
            value = _tiny_model(historical).eval()
            with torch.no_grad():
                expected = value(*_inputs())
        else:
            value = historical.SlotBinding("gptrans_t", historical.SlotName.PRE_READOUT)
        payload = pickle.dumps(value)
    assert old.encode() in payload
    restored = pickle.loads(payload)
    assert type(restored) is getattr(implementation, type(value).__name__)
    if old == "molgap.gptrans":
        _assert_tensors_equal(restored.state_dict(), value.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored(*_inputs()), expected, rtol=0, atol=0)
    else:
        assert restored.slot is implementation.SlotName.PRE_READOUT
        assert restored.family == "gptrans_t"


def test_legacy_gptrans_monkeypatch_reaches_implementation(monkeypatch):
    from molgap import gptrans as legacy
    from molgap.models import gptrans

    calls = []
    original = legacy.to_dense_batch

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(legacy, "to_dense_batch", tracked)
    model = _tiny_model(gptrans).eval()
    model(*_inputs())
    assert len(calls) == 2


def test_legacy_private_substrate_monkeypatch_reaches_binding(monkeypatch):
    import torch
    from molgap import model_substrate as legacy
    from molgap.models import gptrans, substrate

    class Identity(torch.nn.Module):
        def forward(self, payload):
            return payload

    calls = []
    original = legacy._argument

    def tracked(*args, **kwargs):
        calls.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(legacy, "_argument", tracked)
    model = _tiny_model(gptrans).eval()
    owner = substrate.bind_substrate(model, "gptrans_t", [
        ("identity", Identity(), substrate.SlotBinding("gptrans_t", substrate.SlotName.PRE_READOUT)),
    ])
    try:
        model(*_inputs())
        assert calls == ["batch", "input"]
    finally:
        owner.detach("identity")
