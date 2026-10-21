"""Synthetic CPU checks without cache, training loops or platform calls."""
from dataclasses import replace
import random
from types import MappingProxyType, ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from torch import nn

from molgap.fast_sandbox import run_arm_sandbox, run_family_sandbox, run_sandbox, synthetic_batch, verify_loaded_family_model
from molgap.model_substrate import ModelSubstrate, SlotBinding, SlotName


class TinyGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(9, 1)
    def forward(self, batch):
        pooled = batch.x.new_zeros((len(batch.y), 9), dtype=torch.float32).index_add(0, batch.batch, batch.x.float())
        result = self.head(pooled)
        return result + self.delta if hasattr(self, "delta") else result


class TinySlotGraph(TinyGraph):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        del edge_index, edge_attr, random_walk_pe
        pooled = x.new_zeros((2, 9), dtype=torch.float32).index_add(0, batch, x.float())
        return self.head(pooled)


class Addon(nn.Module):
    def __init__(self, kind="live"):
        super().__init__()
        self.kind = kind
        if kind != "stateless":
            self.scale = nn.Parameter(torch.tensor(.5))
    def forward(self, payload):
        if self.kind == "disconnected":
            return payload
        if self.kind == "stateless":
            return replace(payload, embedding=payload.embedding + 1)
        return replace(payload, embedding=payload.embedding * self.scale)


def slot_factory(kind="live"):
    def factory():
        owner = ModelSubstrate(TinySlotGraph(), "neural_atom_k1")
        owner.attach("delta", Addon(kind), SlotBinding("neural_atom_k1", SlotName.PRE_READOUT))
        owner.seal()
        return owner
    return factory


def owning_forward(model, batch):
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe).reshape(-1)


def test_default_synthetic_batch_has_complete_ogb_schema():
    from molgap.pcqm_topology import validate_ogb_gap_batch
    batch = synthetic_batch()
    assert validate_ogb_gap_batch(batch) == 2
    assert torch.equal(batch.ptr, torch.tensor([0, 3, 5]))
    assert batch.x.device.type == "cpu"


def test_one_batch_slot_forward_addon_gradients_and_memory_state_roundtrip():
    report = run_sandbox(slot_factory(), forward=owning_forward)
    assert report["status"] == "PASSED", report
    assert report["train_backward_batches"] == 1 and report["synthetic_data"]
    assert report["checks"]["state_dict_roundtrip"]["eval_predictions_exact"]
    assert report["addons"]["delta"]["connected_nonzero"] == ["_molgap_addons.delta.scale"]
    assert report["timing"]["elapsed_seconds"] >= 0
    assert report["timing"]["cold_process_startup_measured"] is False


def test_disconnected_addon_fails_and_does_not_claim_checkpoint_proof():
    report = run_sandbox(slot_factory("disconnected"), forward=owning_forward)
    assert report["status"] == "FAILED"
    assert report["addons"]["delta"]["missing_gradients"] == ["_molgap_addons.delta.scale"]
    assert "state_dict_roundtrip" not in report["checks"]


def test_one_connected_parameter_does_not_hide_an_unused_addon_parameter():
    class PartlyDisconnected(Addon):
        def __init__(self):
            super().__init__()
            self.unused = nn.Parameter(torch.tensor(1.))
    def factory():
        owner = ModelSubstrate(TinySlotGraph(), "neural_atom_k1")
        owner.attach("delta", PartlyDisconnected(), SlotBinding("neural_atom_k1", SlotName.PRE_READOUT))
        return owner
    report = run_sandbox(factory, forward=owning_forward)
    assert report["status"] == "FAILED"
    assert report["addons"]["delta"]["connected_nonzero"] == ["_molgap_addons.delta.scale"]
    assert report["addons"]["delta"]["missing_gradients"] == ["_molgap_addons.delta.unused"]
    assert report["addons"]["delta"]["status"] == "FAILED"
    assert "state_dict_roundtrip" not in report["checks"]


def test_stateless_addon_reports_gradient_not_applicable():
    report = run_sandbox(slot_factory("stateless"), forward=owning_forward)
    assert report["status"] == "PASSED", report
    assert report["addons"]["delta"]["status"] == "NOT_APPLICABLE"
    assert "not proved" in report["addons"]["delta"]["reason"]


def test_zero_start_reports_blocked_upstream_without_false_rejection():
    class ZeroStart(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(9, 3)
            self.return_projection = nn.Linear(3, 9, bias=False)
            nn.init.zeros_(self.return_projection.weight)
        def forward(self, payload):
            return replace(payload, embedding=payload.embedding + self.return_projection(self.project(payload.embedding)))
    def factory():
        owner = ModelSubstrate(TinySlotGraph(), "neural_atom_k1")
        owner.attach("zero", ZeroStart(), SlotBinding("neural_atom_k1", SlotName.PRE_READOUT))
        return owner
    report = run_sandbox(factory, forward=owning_forward)
    assert report["status"] == "PASSED", report
    assert report["addons"]["zero"]["zero_gradients"] == [
        "_molgap_addons.zero.project.weight", "_molgap_addons.zero.project.bias"]
    assert report["addons"]["zero"]["connected_nonzero"] == ["_molgap_addons.zero.return_projection.weight"]


def test_explicit_parameter_groups_support_reviewed_base_parameter_reuse():
    report = run_sandbox(TinyGraph, addon_parameters=lambda model: {"base_reuse": ["head.weight"]})
    assert report["status"] == "PASSED", report
    assert report["addons"]["base_reuse"]["status"] == "PASSED"
    assert report["checks"]["backbone_gradients"]["connected_nonzero"] == ["head.bias"]
    wrong = run_sandbox(TinyGraph, addon_parameters={"bad": ["unwritten.weight"]})
    assert wrong["status"] == "FAILED" and "missing" in wrong["error"]["message"]


def test_nan_forward_nonfinite_gradients_and_serialization_schema_fail_closed():
    class NaN(TinyGraph):
        def forward(self, batch):
            return super().forward(batch) * float("nan")
    assert run_sandbox(NaN)["status"] == "FAILED"
    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()
        @staticmethod
        def backward(ctx, gradient):
            return gradient * float("nan")
    class Grad(TinyGraph):
        def forward(self, batch):
            return BadGradient.apply(super().forward(batch))
    bad = run_sandbox(Grad)
    assert bad["status"] == "FAILED"
    assert bad["checks"]["backbone_gradients"]["nonfinite_gradients"]
    count = 0
    def different_factory():
        nonlocal count
        count += 1
        model = TinyGraph()
        if count == 2:
            model.register_buffer("missing_first_time", torch.zeros(1))
        return model
    changed = run_sandbox(different_factory)
    assert changed["status"] == "FAILED" and "Missing key" in changed["error"]["message"]


def test_process_rng_threads_flags_and_model_modes_restored_without_cuda_probe(monkeypatch):
    batch = synthetic_batch()
    python, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    made = []
    def factory():
        model = TinyGraph().eval()
        model.head.train()
        made.append(model)
        return model
    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU sandbox probed CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    report = run_sandbox(factory, batch=batch)
    assert report["status"] == "PASSED", report
    assert random.getstate() == python
    current = np.random.get_state()
    assert current[0] == numpy_state[0] and np.array_equal(current[1], numpy_state[1]) and current[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_num_threads() == threads and torch.get_default_dtype() is dtype
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn
    assert all(not model.training and model.head.training for model in made)


def test_factory_must_construct_independent_models():
    model = TinyGraph()
    report = run_sandbox(lambda: model)
    assert report["status"] == "FAILED" and "independent" in report["error"]["message"]


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("failure", [False, True])
def test_loaded_model_probe_restores_tensors_gradients_modes_and_rng(family, failure):
    if family == "neural_atom_k1":
        from molgap.qm9_neural_atom import make_encoder
        model = make_encoder("neural_atom_k1")
    else:
        from molgap.family_model_ops import make_gptrans_backbone
        from molgap.gptrans_execution_backend import bind_model_layout
        model = bind_model_layout(make_gptrans_backbone(variant="reference"))
    model.eval()
    next(module for module in model.modules() if isinstance(module, nn.Linear)).train()
    parameters = list(model.parameters())
    parameters[0].grad = torch.ones_like(parameters[0])
    gradients = [parameter.grad for parameter in parameters]
    tensors = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    handle = None
    if failure:
        def fail_after_forward(*args):
            raise RuntimeError("synthetic loaded-model probe failure")
        handle = model.register_forward_hook(fail_after_forward)
    try:
        recipe = SimpleNamespace(family=family, mode="reference", k1_execution=None)
        if failure:
            with pytest.raises(RuntimeError, match="synthetic loaded-model probe failure"):
                verify_loaded_family_model(recipe, model)
        else:
            assert verify_loaded_family_model(recipe, model)["status"] == "PASSED"
        assert all(torch.equal(value, tensors[name]) for name, value in model.state_dict().items())
        assert all(parameter.grad is gradient for parameter, gradient in zip(parameters, gradients))
        assert [module.training for module in model.modules()] == modes
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
def test_baseline_presets_reuse_factory_and_owning_forward(monkeypatch, family):
    calls = []
    if family == "neural_atom_k1":
        from molgap import qm9_neural_atom as owner
        original = owner.make_encoder
        def factory(mode):
            calls.append(mode)
            return original(mode)
        monkeypatch.setattr(owner, "make_encoder", factory)
    else:
        from molgap import family_model_ops as owner
        original = owner.make_gptrans_backbone
        def factory(*, variant):
            calls.append(variant)
            return original(variant=variant)
        monkeypatch.setattr(owner, "make_gptrans_backbone", factory)
    result = run_family_sandbox(family)
    assert result["status"] == "PASSED", result
    assert calls == (["neural_atom_k1"] * 2 if family == "neural_atom_k1" else ["reference"] * 2)
    assert result["frozen_initialization_loaded"] is False


def test_observer_tracks_parameter_objects_through_wrapper_prefixes(monkeypatch):
    from molgap import shared_model_adapter as shared
    seen = {}
    def hook(adapter, arm, addon):
        if addon["name"] == "delta":
            def apply(model, config):
                model.delta = nn.Parameter(torch.tensor(.5))
            return apply
        class Wrapped(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.body = model
            def forward(self, batch):
                return self.body(batch)
        return lambda model, config: Wrapped(model)
    monkeypatch.setattr(shared, "_addon_hook", hook)
    def observe(addon, parameters):
        seen[addon["name"]] = {id(parameter) for parameter in parameters}
    model = shared.apply_reviewed_addons(None,
        {"addons": [{"name": "delta", "config": {}}, {"name": "wrap", "config": {}}]}, TinyGraph(), addon_observer=observe)
    assert seen["delta"] == {id(model.body.delta)} and not seen["wrap"]
    assert {name for name, parameter in model.named_parameters() if id(parameter) in seen["delta"]} == {"body.delta"}
    with pytest.raises(TypeError, match="observer"):
        shared.apply_reviewed_addons(None, {"addons": []}, TinyGraph(), addon_observer="caller-string")


def test_source_pinned_generic_arm_uses_same_construction_observer(monkeypatch, tmp_path):
    from molgap import experiment_execution as execution, experiment_spec as declaration
    from molgap.v4_runtime import normalized_source_sha256
    from test_experiment_workflow import _arm, _spec_payload
    family = ("sandbox_graph", "1")
    contract = declaration.FamilyContract(*family, "molgap.synthetic_sandbox_model", "graph_gap_screen_v1",
        "ogb-atom9-bond3-rwse16-v1", ("train", "development"), "seed42-epoch-global-randperm-v1", "train-mean-unbiased-std")
    families = MappingProxyType({**declaration.FAMILIES, family: contract})
    addon_contract = declaration.AddonContract(family[0], "sandbox", "molgap.synthetic_sandbox_addon")
    addons = MappingProxyType({**declaration.ADDONS, ("sandbox_delta", "1"): addon_contract})
    monkeypatch.setattr(declaration, "FAMILIES", families)
    monkeypatch.setattr(execution, "FAMILIES", families)
    monkeypatch.setattr(declaration, "ADDONS", addons)
    monkeypatch.setattr(execution, "ADDONS", addons)
    for name in ("synthetic_sandbox_model", "synthetic_sandbox_addon"):
        path = tmp_path / (name + ".py")
        path.write_text("# controlled synthetic hook fixture\n", encoding="utf-8")
        module = ModuleType("molgap." + name)
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    model_module = sys.modules["molgap.synthetic_sandbox_model"]
    addon_module = sys.modules["molgap.synthetic_sandbox_addon"]
    model_module.make_model = lambda arm: TinyGraph()
    def apply_addon(model, config):
        model.delta = nn.Parameter(torch.tensor(.5))
    addon_module.apply_addon = apply_addon
    adapter = execution.graph_training_adapter(family, model_factory="molgap.synthetic_sandbox_model:make_model",
        addons=(execution.TrainingAddon("sandbox_delta", "1", "delta", apply_hook="molgap.synthetic_sandbox_addon:apply_addon"),))
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS", MappingProxyType({**execution.TRAINING_ADAPTERS, family: adapter}))
    arm = _arm("neural_atom_k1", "2", "candidate")
    arm["family"] = dict(name=family[0], version=family[1])
    for key, name in (("recipe", contract.recipe), ("sampler", contract.sampler), ("transform", contract.transform)):
        arm["training"][key]["name"] = name
    arm["base"]["sha256"] = normalized_source_sha256(tmp_path / "synthetic_sandbox_model.py")
    arm["addons"] = [{"name": "sandbox_delta", "version": "1", "config": {},
        "source_sha256": normalized_source_sha256(tmp_path / "synthetic_sandbox_addon.py")}]
    arm["addon_semantics"] = "ordered"
    spec = declaration.ExperimentSpec(_spec_payload([arm], platform="kaggle"))
    result = run_arm_sandbox(spec, "candidate")
    assert result["status"] == "PASSED", result
    assert result["addons"]["sandbox_delta"]["connected_nonzero"] == ["delta"]
    assert result["spec_identity"] == spec.identity
    assert result["frozen_initialization_loaded"] is False
    (tmp_path / "synthetic_sandbox_addon.py").write_text("# changed source\n", encoding="utf-8")
    altered = run_arm_sandbox(spec, "candidate")
    assert altered["status"] == "FAILED" and "source differs" in altered["error"]["message"]


def test_shared_graph_build_seals_only_after_complete_composition(monkeypatch):
    from molgap import shared_model_adapter as shared
    owner = ModelSubstrate(TinySlotGraph(), "neural_atom_k1")
    monkeypatch.setattr(shared, "_validate_factory_contract", lambda *args: None)
    monkeypatch.setattr(shared, "validate_factory_source", lambda *args: None)
    monkeypatch.setattr(shared, "resolve_model_factory", lambda *args: lambda arm: owner.model)
    def apply(adapter, arm, model, **kwargs):
        owner.attach("first", Addon(), SlotBinding("neural_atom_k1", SlotName.PRE_READOUT))
        owner.attach("second", Addon(), SlotBinding("neural_atom_k1", SlotName.PRE_READOUT))
        assert not owner.sealed
        return model
    monkeypatch.setattr(shared, "apply_reviewed_addons", apply)
    result = shared.build_model(SimpleNamespace(model_factory="reviewed"), {})
    assert result is owner.model and owner.sealed
    with pytest.raises(RuntimeError, match="sealed"):
        owner.detach("first")


@pytest.mark.parametrize("options,error", [
    ({"seed": True}, ValueError), ({"threads": 0}, ValueError), ({"forward": "unreviewed"}, TypeError),
])
def test_invalid_api_arguments_raise(options, error):
    with pytest.raises(error):
        run_sandbox(TinyGraph, **options)
