"""Synthetic metadata/guard parity, not full K1 numerical/native speed qualification.

Only _step is AST-isolated from the immutable checkpoint_io_attempt_001 source
snapshot (lines 207-236); no trainer import or production modification occurs.
SHA256: ac13918013015c27b1f46020b0eabf2b4f7ad0eb18bb091a9e6033b94948dc34.
No actual model, optimizer, gradients, CUDA APIs, or training are executed.
"""
import ast
import builtins
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = (Path(__file__).resolve().parents[1] / "experiments/pcqm_k1_execution_efficiency/"
          "checkpoint_io_attempt_001/source_snapshot/colab_k1_screen.py")
SHA256 = "ac13918013015c27b1f46020b0eabf2b4f7ad0eb18bb091a9e6033b94948dc34"


@pytest.fixture(scope="module")
def steps():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    reference = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "_step")
    candidate = copy.deepcopy(reference)
    target = ast.parse("arm['train_loss_sum'] += float(loss)").body[0]
    indices = [i for i, node in enumerate(candidate.body) if ast.dump(node) == ast.dump(target)]
    assert len(indices) == 1 and ast.dump(candidate.body[-1]) == ast.dump(ast.parse("return float(loss)").body[0])
    index = indices[0]
    candidate.body[index].value = ast.copy_location(ast.Name(id="scalar", ctx=ast.Load()), reference.body[index].value)
    candidate.body[-1].value = ast.copy_location(ast.Name(id="scalar", ctx=ast.Load()), reference.body[-1].value)
    candidate.body.insert(index, ast.copy_location(ast.parse("scalar = float(loss)").body[0], reference.body[index]))
    # Reverse exactly these three changes; all other AST nodes/locations must match.
    restored = copy.deepcopy(candidate)
    del restored.body[index]
    restored.body[index].value = copy.deepcopy(reference.body[index].value)
    restored.body[-1].value = copy.deepcopy(reference.body[-1].value)
    assert ast.dump(restored, include_attributes=True) == ast.dump(reference, include_attributes=True)
    return reference, ast.fix_missing_locations(candidate)


def exercise(step, case="ok", ema=False, device="cpu"):
    events, hooks = [], []

    class Loss:
        tensor = torch.tensor(float("nan") if case == "loss_nan" else 1.25)

        def __float__(self):
            events.append(("float",))
            return self.tensor.item()

    class Arm(dict):
        def __setitem__(self, key, value):
            events.append(("progress", key, value))
            super().__setitem__(key, value)

    class Batch(dict):
        def to(self, destination):
            events.append(("batch.to", destination))
            return self

    batch = Batch({"pos": None} if case == "geometry" else {})
    batch.y = torch.zeros(2, dtype=torch.float64 if case == "target_dtype" else torch.float32)
    batch.random_walk_pe = torch.zeros(2, dtype=torch.float64 if case == "pe_dtype" else torch.float32)
    parameter = torch.zeros(1, dtype=torch.float64 if case == "parameter_dtype" else torch.float32)
    model = SimpleNamespace(train=lambda: events.append(("train",)))

    def parameters():
        events.append(("parameters",))
        return [parameter]

    def register(hook):
        events.append(("hook.add",))
        hooks.append(hook)

        def remove():
            events.append(("hook.remove",))
            hooks.remove(hook)
        return SimpleNamespace(remove=remove)

    def finite(value):
        events.append(("finite", "loss" if isinstance(value, Loss) else "prediction"))
        return torch.isfinite(value.tensor if isinstance(value, Loss) else value)

    def owner_step(actual_model, optimizer, actual_batch, mean, std):
        assert actual_model is model and actual_batch is batch and optimizer is token
        assert (mean, std) == (0.5, 2.0)
        events.extend([("owner.step",), ("forward",)])
        prediction = torch.zeros(1 if case == "forward_shape" else 2)
        if case == "forward_nan":
            prediction[0] = float("nan")
        for hook in tuple(hooks):
            hook(model, (), prediction)
        if case == "owner_error":
            raise RuntimeError("synthetic owner failure")
        events.append(("owner.update.marker",))  # Marker only; no update/gradients.
        return Loss()

    def check(deadline):
        events.append(("deadline", deadline))
        if case == "deadline":
            raise TimeoutError("synthetic deadline")

    model.parameters, model.register_forward_hook = parameters, register
    token = object()
    arm = Arm(model=model, optimizer=token, ema=object() if ema else None,
              offset=3, steps=7, samples=14, train_loss_sum=2.5)
    fake_torch = SimpleNamespace(float32=torch.float32, isfinite=finite,
                                cuda=SimpleNamespace(synchronize=lambda: events.append(("cuda.sync.mock",))))

    def isolated_import(name, *args, **kwargs):
        assert name == "torch", "Only _step's mocked torch import is permitted"
        return fake_torch

    namespace = {"__builtins__": {**vars(builtins), "__import__": isolated_import},
                 "_check": check, "FORBIDDEN_MODEL_FIELDS": ("pos", "geometry"),
                 "owner": SimpleNamespace(BS=2, step=owner_step),
                 "update_ema": lambda *args: events.append(("ema",))}
    exec(compile(ast.Module(body=[copy.deepcopy(step)], type_ignores=[]), str(SOURCE), "exec"), namespace)
    result, error = None, None
    try:
        result = namespace["_step"](arm, batch, 0.5, 2.0, device, 100.0)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        error = (type(exc), str(exc))
    assert not hooks
    return result, error, {key: arm[key] for key in ("offset", "steps", "samples", "train_loss_sum")}, events


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_single_conversion_exact_synthetic_counter_guard_and_event_parity(steps, ema, device):
    reference, candidate = [exercise(step, ema=ema, device=device) for step in steps]
    assert reference[:3] == candidate[:3] == (1.25, None, {"offset": 4, "steps": 8, "samples": 16, "train_loss_sum": 3.75})
    assert type(reference[0]) is type(candidate[0]) is float
    before, after = reference[3], candidate[3]
    assert before.count(("float",)) == 2 and after.count(("float",)) == 1
    assert before[-1] == ("float",) and before[:-1] == after
    assert after.index(("finite", "prediction")) < after.index(("owner.update.marker",))
    assert after.index(("hook.remove",)) < after.index(("finite", "loss"))
    if ema:
        assert after.index(("finite", "loss")) < after.index(("ema",)) < after.index(("float",))
    assert (("cuda.sync.mock",) in after) == device.startswith("cuda")
    if device.startswith("cuda"):
        assert after.index(("cuda.sync.mock",)) < after.index(("progress", "offset", 4))
        if ema:
            assert after.index(("ema",)) < after.index(("cuda.sync.mock",))


@pytest.mark.parametrize("case", ["forward_nan", "forward_shape", "owner_error", "loss_nan",
                                  "geometry", "target_dtype", "pe_dtype", "parameter_dtype", "deadline"])
def test_failclosed_guards_hook_cleanup_no_progress_or_scalar_conversion(steps, case):
    reference, candidate = [exercise(step, case=case, ema=True, device="cuda:0") for step in steps]
    assert reference == candidate
    _, error, progress, events = candidate
    assert error is not None
    assert progress == {"offset": 3, "steps": 7, "samples": 14, "train_loss_sum": 2.5}
    assert not any(event[0] in ("progress", "float", "ema", "cuda.sync.mock") for event in events)
    if case.startswith("forward") or case in ("owner_error", "loss_nan"):
        assert events.count(("hook.remove",)) == 1
    else:
        assert ("forward",) not in events and ("hook.add",) not in events
    assert (("owner.update.marker",) in events) == (case == "loss_nan")
    if case == "deadline":
        assert events == [("deadline", 100.0)]
