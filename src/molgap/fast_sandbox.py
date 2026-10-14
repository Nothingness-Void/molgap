"""One synthetic CPU batch for local model/addon checks, independent of trainers.

Factories build disposable models. This gate checks forward/autograd/tensor
serialization only; it neither qualifies a real cache, optimizer continuation,
accelerator memory, native runtime nor a scientific comparison.
"""
from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import random
import sys
import time


FORMAT = "molgap-fast-local-sandbox-v1"
LIMITATIONS = [
    "Synthetic CPU tensors only; no real dataset or protected role is consumed.",
    "State_dict roundtrip is not optimizer/scheduler/RNG continuation qualification.",
    "No GPU memory, physical-batch128, platform or scientific acceptance is established.",
    "Elapsed API time is measured; cold process startup is not measured here.",
]


@contextmanager
def _cpu_scope(seed, threads):
    import numpy as np
    import torch
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    old_threads, old_dtype = torch.get_num_threads(), torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        # Explicit empty device list avoids CUDA discovery or RNG capture.
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            torch.set_num_threads(threads)
            torch.set_default_dtype(torch.float32)
            torch.use_deterministic_algorithms(True, warn_only=False)
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_num_threads(old_threads)
        torch.set_default_dtype(old_dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def synthetic_batch():
    """Two unequal tiny categorical OGB graphs; all fields are generated in RAM."""
    import torch
    from torch_geometric.data import Batch, Data
    graphs = []
    for index, nodes in enumerate((3, 2)):
        x = torch.zeros(nodes, 9, dtype=torch.long)
        x[:, 0] = torch.arange(nodes) + 5 + index
        source = torch.arange(nodes - 1)
        edges = torch.cat((torch.stack((source, source + 1)), torch.stack((source + 1, source))), 1)
        graphs.append(Data(x=x, edge_index=edges,
            edge_attr=torch.zeros(edges.shape[1], 3, dtype=torch.long),
            random_walk_pe=torch.arange(nodes * 16).reshape(nodes, 16).float() / 100,
            y=torch.tensor([-2. - index]), source_idx=torch.tensor([index])))
    return Batch.from_data_list(graphs)


def _module(value):
    import torch
    from .model_substrate import ModelSubstrate
    model = value.model if isinstance(value, ModelSubstrate) else value
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Sandbox factory must return nn.Module or ModelSubstrate")
    if any(tensor.device.type != "cpu" for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError("Sandbox requires CPU-only model parameters and buffers")
    return model


def _groups(model, requested):
    if callable(requested):
        requested = requested(model)
    parameters = dict(model.named_parameters())
    if requested is None:
        owner = getattr(model, "_molgap_substrate", None)
        requested = {} if owner is None else {
            entry.name: [name for name in parameters if name.startswith(owner.namespace + "." + entry.name + ".")]
            for entry in owner.bindings}
    if type(requested) is not dict:
        raise ValueError("addon_parameters must map addon names to parameter-name sequences")
    groups, seen = {}, set()
    for addon, names in requested.items():
        if type(addon) is not str or not addon or not isinstance(names, (list, tuple)):
            raise ValueError("Invalid addon parameter ownership group")
        if any(type(name) is not str or name not in parameters for name in names) or len(set(names)) != len(names):
            raise ValueError("Addon ownership refers to missing/duplicate model parameters")
        if seen.intersection(names):
            raise ValueError("Addon parameter ownership groups overlap")
        seen.update(names)
        groups[addon] = tuple(names)
    return parameters, groups


def _gradient_report(parameters, names, *, require_present=False):
    import torch
    trainable = [name for name in names if parameters[name].requires_grad]
    missing, zero, nonfinite, connected = [], [], [], []
    for name in trainable:
        gradient = parameters[name].grad
        if gradient is None:
            missing.append(name)
        elif not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        elif not bool(torch.count_nonzero(gradient)):
            zero.append(name)
        else:
            connected.append(name)
    status = ("NOT_APPLICABLE" if not trainable else
              "PASSED" if connected and not nonfinite and (not require_present or not missing) else "FAILED")
    result = {"status": status, "trainable_parameters": len(trainable),
        "connected_nonzero": connected, "missing_gradients": missing,
        "zero_gradients": zero, "nonfinite_gradients": nonfinite}
    if not trainable:
        result["reason"] = "No owned trainable parameters: parameter-free, frozen or initialization-only; trainable addon connectivity is not proved."
    if zero:
        result["zero_gradient_note"] = "A zero-start projection can block upstream gradients on the first batch; this test does not infer permanent disconnection."
    return result


def _modes(model):
    return [(module, module.training) for module in model.modules()]


def _restore_modes(modes):
    for module, training in modes:
        module.training = training


def _tensor_state(model):
    import torch
    state = model.state_dict()
    if any(not torch.is_tensor(value) or value.device.type != "cpu" or not bool(torch.isfinite(value).all())
           for value in state.values()):
        raise ValueError("Sandbox requires a finite CPU tensor state_dict")
    return state


def _same_state(actual, expected):
    import torch
    if actual.keys() != expected.keys() or any(
            actual[name].dtype != value.dtype or actual[name].shape != value.shape
            or not torch.equal(actual[name], value) for name, value in expected.items()):
        raise ValueError("State_dict keys, tensor schema or bytes differ after roundtrip")


def run_sandbox(factory, *, forward=None, batch=None, addon_parameters=None, seed=42, threads=1):
    """Check one backward batch and exact in-memory state/eval reconstruction.

    ``forward(model, batch)`` defaults to ``model(batch)``. Parameter ownership
    accepts ``{addon: [fully_qualified_parameter_names]}`` or a callable returning
    that mapping. Substrate bindings supply groups automatically. Explicit groups
    can describe reviewed addons which deliberately reuse backbone parameters.
    Execution failures return a FAILED report; malformed API arguments raise.
    """
    if not callable(factory) or (forward is not None and not callable(forward)):
        raise TypeError("Expected a model factory and optional callable forward")
    if type(seed) is not int or not 0 <= seed < 2**32 or type(threads) is not int or threads < 1:
        raise ValueError("Sandbox seed/threads are outside supported bounds")
    began = time.perf_counter()
    loaded = {name: name in sys.modules for name in ("torch", "numpy", "torch_geometric")}
    report = {"format": FORMAT, "status": "FAILED", "seed": seed, "device": "cpu",
        "synthetic_data": batch is None, "train_backward_batches": 0,
        "checks": {}, "addons": {}, "limitations": list(LIMITATIONS)}
    try:
        with _cpu_scope(seed, threads):
            import torch
            from .shared_model_adapter import validate_model_output
            current = batch if batch is not None else synthetic_batch()
            target = getattr(current, "y", None)
            if (not torch.is_tensor(target) or target.device.type != "cpu"
                    or target.dtype != torch.float32 or not bool(torch.isfinite(target).all())):
                raise ValueError("Sandbox batch requires finite CPU FP32 y")
            target = target.reshape(-1)
            if not target.numel():
                raise ValueError("Sandbox batch must contain targets")
            call = forward or (lambda model, graph_batch: model(graph_batch))
            model = _module(factory())
            original_modes = _modes(model)
            try:
                parameters, groups = _groups(model, addon_parameters)
                model.train()
                model.zero_grad(set_to_none=True)
                prediction = validate_model_output(call(model, current), rows=target.numel(), device=target.device)
                loss = torch.nn.functional.l1_loss(prediction, target)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Sandbox L1 loss is nonfinite")
                report["checks"]["forward"] = {"status": "PASSED", "rows": target.numel(), "loss": float(loss.detach())}
                loss.backward()
                report["train_backward_batches"] = 1
                owned = {name for names in groups.values() for name in names}
                report["checks"]["backbone_gradients"] = _gradient_report(parameters, [name for name in parameters if name not in owned])
                report["addons"] = {addon: _gradient_report(parameters, names, require_present=True)
                    for addon, names in groups.items()}
                if report["checks"]["backbone_gradients"]["status"] == "FAILED" or any(
                        result["status"] == "FAILED" for result in report["addons"].values()):
                    raise ValueError("Backbone/addon gradients are disconnected or nonfinite")
                buffer = BytesIO()
                state = _tensor_state(model)
                torch.save(state, buffer)
                size = buffer.tell()
                buffer.seek(0)
                retained = torch.load(buffer, map_location="cpu", weights_only=True)
                _same_state(_tensor_state(model), retained)
                model.eval()
                with torch.no_grad():
                    expected = validate_model_output(call(model, current), rows=target.numel(), device=target.device).clone()
                # Release autograd intermediates before the second constructor.
                del loss, prediction, parameters, state
                model.zero_grad(set_to_none=True)
                restored = _module(factory())
                if restored is model or {id(p) for p in restored.parameters()}.intersection(id(p) for p in model.parameters()):
                    raise ValueError("Sandbox factory must create independent models/parameters")
                restored_modes = _modes(restored)
                try:
                    restored.load_state_dict(retained, strict=True)
                    _same_state(_tensor_state(restored), retained)
                    restored.eval()
                    with torch.no_grad():
                        observed = validate_model_output(call(restored, current), rows=target.numel(), device=target.device)
                    if not torch.equal(observed, expected):
                        raise ValueError("Reloaded eval predictions differ from the serialized model")
                    report["checks"]["state_dict_roundtrip"] = {"status": "PASSED", "tensor_keys": len(retained),
                        "serialized_bytes": size, "eval_predictions_exact": True}
                finally:
                    _restore_modes(restored_modes)
            finally:
                _restore_modes(original_modes)
            report["status"] = "PASSED"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["timing"] = {"elapsed_seconds": time.perf_counter() - began,
        "scope": "API entry through checks and process-state restoration",
        "imports_loaded_at_entry": loaded, "cold_process_startup_measured": False}
    return report


def run_family_sandbox(family: str, *, seed=42, threads=1):
    """Random-initialized retained baseline factory/forward, never frozen weights."""
    if family == "neural_atom_k1":
        def factory():
            from .qm9_neural_atom import make_encoder
            return make_encoder("neural_atom_k1")
        def forward(model, batch):
            from .k1_screen_training import _forward
            return _forward(model, batch)
    elif family == "gptrans_t":
        def factory():
            from .pcqm_gptrans_v4 import _make_model
            return _make_model(variant="reference")
        def forward(model, batch):
            from .pcqm_gptrans_v4 import _forward
            return _forward(model, batch)
    else:
        raise ValueError("No reviewed baseline sandbox forward for family")
    result = run_sandbox(factory, forward=forward, seed=seed, threads=threads)
    result.update(family=family, model_identity="random baseline constructor", frozen_initialization_loaded=False)
    return result


def run_arm_sandbox(spec, arm_id: str, *, addon_parameters=None, seed=42, threads=1):
    """Use the source-pinned generic graph ABI without data/initial-state loading."""
    from .experiment_spec import ExperimentSpec
    from .experiment_execution import training_adapter
    from .shared_model_adapter import build_graph_model, forward_normalized_gap
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected a validated ExperimentSpec")
    spec = ExperimentSpec.from_json(spec.to_json())
    arm = next((item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError("Unknown sandbox arm")
    adapter = training_adapter(arm)
    if adapter.model_factory is None or adapter.module != "molgap.graph_screen_training":
        raise ValueError("Arm has no bound generic sandbox forward; use a reviewed factory/forward or baseline preset")
    owned = {}
    def observer(addon, parameters):
        # Retain objects until the final name mapping so removal/replacement
        # cannot recycle Python identities into another addon's group.
        owned[addon["name"]] = tuple(parameters)
    def factory():
        owned.clear()
        return build_graph_model(spec, arm_id, addon_observer=observer)
    def groups(model):
        if addon_parameters is not None:
            return addon_parameters(model) if callable(addon_parameters) else addon_parameters
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        result = {}
        for addon, parameters in owned.items():
            identities = {id(parameter) for parameter in parameters}
            if not identities.issubset(names):
                raise ValueError("A later addon removed an earlier addon's owned parameters")
            result[addon] = [name for identity, name in names.items() if identity in identities]
        return result
    result = run_sandbox(factory, forward=forward_normalized_gap,
        addon_parameters=groups, seed=seed, threads=threads)
    result.update(spec_identity=spec.identity, arm_id=arm_id,
        model_identity="source-pinned random constructor", frozen_initialization_loaded=False)
    return result


__all__ = ["run_sandbox", "run_family_sandbox", "run_arm_sandbox", "synthetic_batch"]
