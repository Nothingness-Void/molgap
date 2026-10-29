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
        if owner is None:
            requested = {}
        else:
            requested = {
                entry.name: [name for name in parameters if name.startswith(owner.namespace + "." + entry.name + ".")]
                for entry in owner.bindings}
            for mutation_name, target_path in getattr(owner, "mutations", ()):
                prefix = target_path + "."
                requested[mutation_name] = [
                    name for name in parameters if name == target_path or name.startswith(prefix)]
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


def _family_forward(family, k1_execution=None, prepare=None):
    """The one accelerated-CPU forward shared by every family sandbox and loaded-model probe."""
    from copy import copy
    from .family_model_ops import k1_forward

    def forward(model, batch):
        if family == "neural_atom_k1":
            from .k1_execution_layout import cpu_layout_context
            from .k1_execution_backend import execution_context
            borrowed = copy(batch)
            with cpu_layout_context(model, borrowed, "cpu"), execution_context(model, borrowed, "cpu", k1_execution):
                return k1_forward(model, borrowed)
        from .gptrans_execution_backend import prepare_layout, prepared_forward
        prepared = prepare_layout(copy(batch))
        if prepare is not None:
            prepared = prepare(prepared, training=model.training)
        return prepared_forward(model, prepared)
    return forward


def run_family_sandbox(family: str, *, addons=(), k1_execution=None, addon_parameters=None, seed=42, threads=1):
    """Random baseline plus reviewed slots through the accelerated CPU forwards."""
    from .family_model_ops import make_gptrans_backbone
    from .shared_model_adapter import apply_reviewed_addons, validate_family_addons
    if family not in {"neural_atom_k1", "gptrans_t"}:
        raise ValueError("No reviewed baseline sandbox forward for family")
    if k1_execution is not None and family != "neural_atom_k1":
        raise ValueError("K1 execution options require the K1 substrate")
    adapter = validate_family_addons(family, addons)
    def factory():
        if family == "neural_atom_k1":
            from .qm9_neural_atom import make_encoder
            model = make_encoder("neural_atom_k1")
        else:
            from .gptrans_execution_backend import bind_model_layout
            model = make_gptrans_backbone(variant="reference")
            bind_model_layout(model)
        model = apply_reviewed_addons(adapter, {"addons": list(addons)}, model)
        if hasattr(model, "_molgap_substrate"):
            model._molgap_substrate.seal()
        return model
    result = run_sandbox(factory, forward=_family_forward(family, k1_execution), addon_parameters=addon_parameters, seed=seed, threads=threads)
    result.update(family=family, model_identity="random baseline with reviewed slots",
                  frozen_initialization_loaded=False)
    return result


def _sam_step_checks(recipe, initial_state, forward, batch):
    """Perturbed-second-pass checks for mode sam: determinism, BN buffers, weight restore."""
    import torch
    from .family_model_ops import k1_optimizer_step
    from .family_training_kernel import load_family_model

    def run(mode, lr):
        model = _module(load_family_model(recipe, initial_state)[0]).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        torch.manual_seed(7)
        k1_optimizer_step(model, optimizer, batch, -2.5, 1.0, mode, clip_norm=1.0,
                          check_finite=True, forward=forward)
        return before, {key: value.detach().clone() for key, value in model.state_dict().items()}

    same = lambda a, b, keys: all(torch.equal(a[key], b[key]) for key in keys)
    _, first = run("sam", 4e-4)
    _, second = run("sam", 4e-4)
    _, plain = run("reference", 4e-4)
    buffers = [key for key in first if key.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    before, frozen_lr = run("sam", 0.0)  # lr 0: any change to a weight is a restore failure
    weights = [key for key, _ in _module(load_family_model(recipe, initial_state)[0]).named_parameters()]
    status = lambda ok: {"status": "PASSED" if ok else "FAILED"}
    return {"sam_first_step_deterministic": status(same(first, second, first)),
            "sam_bn_buffers_match_unperturbed_step": {**status(bool(buffers) and same(first, plain, buffers)),
                                                       "buffers": len(buffers)},
            "sam_weights_restored_bitwise": status(same(before, frozen_lr, weights))}


def run_frozen_arm_sandbox(baseline_recipe, recipe, initial_state, *, seed=42, threads=1):
    """Check one candidate on the real frozen initial state against its baseline, on CPU.

    Step-0 outputs must be bitwise equal to the baseline (eval and train, same RNG).
    Every substrate-owned addon parameter needs a finite non-zero gradient after
    one backward (through verify_loaded_family_model, the single gradient probe).
    Every attached addon module must fire in a forward, and a strict save, fresh
    load roundtrip must reproduce predictions bitwise after moving the addon
    parameters off their start. No dataset rows, optimizer continuation or
    accelerator is involved.
    """
    from io import BytesIO
    import torch
    from .family_training_kernel import load_family_model
    from .shared_model_adapter import validate_model_output
    report = {"format": FORMAT, "status": "FAILED", "seed": seed, "device": "cpu", "synthetic_data": True,
        "family": recipe.family, "model_identity": "frozen initialization",
        "frozen_initialization_loaded": True, "checks": {}, "addons": {}, "limitations": list(LIMITATIONS)}
    bits = lambda t: t.detach().contiguous().view(torch.int32)
    try:
        model = _module(load_family_model(recipe, initial_state)[0])
        probe = verify_loaded_family_model(recipe, model, seed=seed, threads=threads)
        report["addons"] = probe["addon_reports"]
        for result in report["addons"].values():
            if result["zero_gradients"] or result["nonfinite_gradients"] or result["missing_gradients"]:
                result["status"] = "FAILED"
        with _cpu_scope(seed, threads):
            forward = _family_forward(recipe.family, getattr(recipe, "k1_execution", None))
            base = _module(load_family_model(baseline_recipe, initial_state)[0])
            batch = synthetic_batch()
            rows = batch.y.numel()
            predict = lambda m: validate_model_output(forward(m, batch), rows=rows, device=batch.y.device)
            parameters = dict(model.named_parameters())
            added = sorted(set(parameters) - set(base.state_dict()))
            owned = {name for result in report["addons"].values()
                     for name in (*result["connected_nonzero"], *result["missing_gradients"],
                                  *result["zero_gradients"], *result["nonfinite_gradients"])}
            if set(added) - owned:
                raise ValueError("Candidate has parameters absent from the baseline that no attached addon owns")
            fired = {}
            modules = dict(getattr(model, "_molgap_addons", {}))
            hooks = [m.register_forward_hook(lambda _m, _i, _o, key=k: fired.__setitem__(key, fired.get(key, 0) + 1))
                     for k, m in modules.items()]
            try:
                model.eval()
                with torch.no_grad():
                    predict(model)
            finally:
                for hook in hooks:
                    hook.remove()
            silent = sorted(set(modules) - set(fired))
            report["checks"]["addon_taps_fire"] = {"status": "FAILED" if silent or not modules else "PASSED",
                "attached": len(modules), "silent": silent}
            if recipe.mode == "sam" and not modules:
                # SAM is an optimizer-step mode with no forward taps; its own checks follow.
                report["checks"]["addon_taps_fire"] = {"status": "NOT_APPLICABLE", "attached": 0, "silent": []}
                report["checks"].update(_sam_step_checks(recipe, initial_state, forward, batch))
            for mode in ("eval", "train"):
                getattr(base, mode)(), getattr(model, mode)()
                with torch.enable_grad() if mode == "train" else torch.no_grad():
                    torch.manual_seed(7)
                    expected = predict(base)
                    torch.manual_seed(7)
                    observed = predict(model)
                same = bool(torch.equal(bits(expected), bits(observed)))
                report["checks"]["step0_bitwise_" + mode] = {"status": "PASSED" if same else "FAILED"}
            with torch.no_grad():
                for name in added:
                    parameters[name].add_(torch.linspace(-0.3, 0.3, parameters[name].numel()).reshape(parameters[name].shape))
            model.eval()
            with torch.no_grad():
                moved = predict(model).clone()
            buffer = BytesIO()
            torch.save(_tensor_state(model), buffer)
            buffer.seek(0)
            fresh = _module(load_family_model(recipe, initial_state)[0])
            fresh.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
            fresh.eval()
            with torch.no_grad():
                roundtrip = bool(torch.equal(bits(moved), bits(predict(fresh))))
                changed = not added or not torch.equal(bits(moved), bits(predict(base.eval())))
            report["checks"]["save_fresh_load_bitwise"] = {"status": "PASSED" if roundtrip and changed else "FAILED",
                "predictions_bitwise_equal": roundtrip, "moved_parameters_change_output": changed}
        failed = [k for k, v in {**report["checks"], **report["addons"]}.items() if v["status"] == "FAILED"]
        report["status"] = "FAILED" if failed else "PASSED"
        if failed:
            report["error"] = {"type": "SandboxCheckFailed", "message": "Failed: " + ", ".join(failed)}
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return report


def run_arm_sandbox(spec, arm_id: str, *, addon_parameters=None, seed=42, threads=1, frozen=None):
    """Use the source-pinned generic graph ABI without data/initial-state loading.

    ``frozen=(baseline_recipe, recipe, initial_state)`` selects the frozen-state
    candidate checks of run_frozen_arm_sandbox for a shared-family arm.
    """
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
    if adapter.module == "molgap.family_training_runner" and frozen is not None:
        result = run_frozen_arm_sandbox(*frozen, seed=seed, threads=threads)
        result.update(spec_identity=spec.identity, arm_id=arm_id)
        return result
    if adapter.module == "molgap.family_training_runner":
        result = run_family_sandbox(arm["family"]["name"], addons=tuple(arm["addons"]),
            addon_parameters=addon_parameters, seed=seed, threads=threads)
        result.update(spec_identity=spec.identity, arm_id=arm_id)
        return result
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


def verify_loaded_family_model(recipe, model, *, seed=42, threads=1):
    """Probe a loaded CPU model, restoring its tensors, gradients and modes on exit."""
    import torch
    from .shared_model_adapter import validate_model_output
    prepare = None
    if recipe.family != "neural_atom_k1":
        from .family_training_policy import mode_policy
        prepare = mode_policy(recipe.family, recipe.mode).prepare_batch
    forward = _family_forward(recipe.family, getattr(recipe, "k1_execution", None), prepare)
    call = lambda current_model, batch: forward(current_model, batch)

    model = _module(model)
    original_modes = _modes(model)
    with _cpu_scope(seed, threads):
        original_tensors = [(value, value.detach().clone()) for value in (*model.parameters(), *model.buffers())]
        original_gradients = [(parameter, parameter.grad) for parameter in model.parameters()]
        try:
            batch = synthetic_batch()
            target = batch.y.reshape(-1)
            parameters, groups = _groups(model, None)
            model.train()
            model.zero_grad(set_to_none=True)
            prediction = validate_model_output(call(model, batch), rows=target.numel(), device=target.device)
            loss = torch.nn.functional.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Pre-submission sandbox L1 loss is nonfinite")
            loss.backward()
            owned = {name for names in groups.values() for name in names}
            backbone_report = _gradient_report(parameters, [name for name in parameters if name not in owned])
            addon_reports = {
                addon: _gradient_report(parameters, names, require_present=True)
                for addon, names in groups.items()
            }
            if backbone_report["status"] == "FAILED" or any(
                    item["status"] == "FAILED" for item in addon_reports.values()):
                raise ValueError("Pre-submission sandbox detected disconnected or nonfinite backbone/addon gradients")
            model.eval()
            with torch.no_grad():
                eval_pred = validate_model_output(call(model, batch), rows=target.numel(), device=target.device)
                if not bool(torch.isfinite(eval_pred).all()):
                    raise ValueError("Pre-submission sandbox eval predictions are nonfinite")
            return {
                "status": "PASSED",
                "rows": int(target.numel()),
                "backbone_gradients": backbone_report["status"],
                "addons": {name: item["status"] for name, item in addon_reports.items()},
                "addon_reports": addon_reports,
            }
        finally:
            with torch.no_grad():
                for value, snapshot in original_tensors:
                    value.copy_(snapshot)
            for parameter, gradient in original_gradients:
                parameter.grad = gradient
            _restore_modes(original_modes)


__all__ = [
    "run_sandbox",
    "run_family_sandbox",
    "run_arm_sandbox",
    "run_frozen_arm_sandbox",
    "synthetic_batch",
    "verify_loaded_family_model",
]
