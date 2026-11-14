"""Real family modules with random initial tensors and tiny synthetic CPU graphs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap.family_training_kernel import ContractSchedule, FamilyKernel
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.training_reproducibility import atomic_torch_save
from molgap.v4_runtime import state_dict_sha256


@pytest.mark.parametrize("optimizer_name", ["MuonAdamW", "BatchedMuonAdamW"])
def test_k1_muon_matches_fp32_matrix_update_and_native_adamw(optimizer_name):
    from molgap import k1_muon
    model = torch.nn.Module()
    model.hidden = torch.nn.Linear(2, 3)
    model.head = torch.nn.Linear(3, 1)
    optimizer = getattr(k1_muon, optimizer_name)(k1_muon.parameter_groups(model, .03),
                                               lr=.01, weight_decay=.03)
    assert optimizer.param_groups[0]["param_names"] == ["hidden.weight"]
    native_head = torch.nn.Parameter(model.head.weight.detach().clone())
    native = torch.optim.AdamW([native_head], lr=.01, weight_decay=.03)
    expected = model.hidden.weight.detach().double().clone()
    buffer = torch.zeros_like(expected)
    for scale in (1., .7):
        grad = torch.tensor([[.2, -.4], [.1, .6], [-.5, .3]]) * scale
        model.hidden.weight.grad = grad.clone()
        model.head.weight.grad = torch.full_like(model.head.weight, .2 * scale)
        native_head.grad = model.head.weight.grad.clone()
        buffer = .95 * buffer + .05 * grad.double()
        update = (.05 * grad.double() + .95 * buffer).T
        update = update / update.norm().clamp(min=1e-7)
        for _ in range(5):
            gram = update @ update.T
            update = 3.4445 * update - 4.775 * (gram @ update) + 2.0315 * (gram @ gram @ update)
        expected = expected * (1 - .01 * .03) - .01 * .2 * (3 ** .5) * update.T
        def closure():
            assert torch.is_grad_enabled()
            return model.hidden.weight.sum()
        loss = optimizer.step(closure)
        assert loss.grad_fn is not None
        native.step()
        torch.testing.assert_close(model.hidden.weight, expected.float(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(model.head.weight, native_head, atol=0, rtol=0)
    assert set(optimizer.state[model.hidden.weight]) == {"step", "momentum_buffer"}
    assert optimizer.state[model.hidden.weight]["step"].item() == 2


def test_k1_muon_batched_multistep_and_bidirectional_resume():
    from molgap.k1_muon import BatchedMuonAdamW, MuonAdamW

    shapes = [(7, 3), (7, 3), (7, 3), (3, 7), (3, 7), (4, 4), (4, 4), (7, 3), (2,)]
    initial = [torch.sin(torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
                         + index).reshape(shape) for index, shape in enumerate(shapes)]

    def make_run(optimizer_type, values):
        params = [torch.nn.Parameter(value.detach().clone()) for value in values]
        optimizer = optimizer_type([
            {"params": params[:7], "algorithm": "muon", "lr": .01},
            {"params": params[7:8], "algorithm": "muon", "lr": .02},
            {"params": params[8:], "algorithm": "adamw", "lr": .003},
        ], weight_decay=.03)
        return params, optimizer

    reference = make_run(MuonAdamW, initial)
    batched = make_run(BatchedMuonAdamW, initial)
    runs = [reference, batched]
    active_steps = [0] * len(shapes)
    max_parameter_error = max_momentum_error = 0.
    for step in range(6):
        if step == 2:
            for source, target_type in ((reference, BatchedMuonAdamW), (batched, MuonAdamW)):
                resumed = make_run(target_type, source[0])
                saved = deepcopy(source[1].state_dict())
                resumed[1].load_state_dict(saved)
                assert resumed[1].param_groups == [
                    dict(group, params=resumed[0][start:end])
                    for group, start, end in zip(source[1].param_groups, (0, 7, 8), (7, 8, 9))]
                for old_param, new_param in zip(source[0], resumed[0]):
                    if old_param not in source[1].state:
                        assert new_param not in resumed[1].state
                        continue
                    for key, value in source[1].state[old_param].items():
                        torch.testing.assert_close(resumed[1].state[new_param][key], value, atol=0, rtol=0)
                runs.append(resumed)
        gradients = [torch.cos(value * (step + 1) + index) * (.1 + index)
                     for index, value in enumerate(initial)]
        gradients[1] = None if step < 3 else gradients[1]
        gradients[2] = None
        gradients[4] = None if step == 4 else gradients[4]
        gradients[5] = torch.zeros_like(initial[5])
        if step == 3:
            gradients[0] = torch.zeros_like(initial[0])
        if step == 5:
            gradients = [None] * len(shapes)
        for index, grad in enumerate(gradients):
            active_steps[index] += grad is not None
        for params, optimizer in runs:
            before = [param.detach().clone() for param in params]
            old_buffers = {index: optimizer.state[param]["momentum_buffer"].clone()
                           for index, param in enumerate(params[:-1]) if param in optimizer.state}
            for param, grad in zip(params, gradients):
                param.grad = None if grad is None else grad.clone()
            optimizer.step()
            for index, (param, grad) in enumerate(zip(params, gradients)):
                if grad is not None:
                    torch.testing.assert_close(param.grad, grad, atol=0, rtol=0)
                else:
                    assert param.grad is None
                    torch.testing.assert_close(param, before[index], atol=0, rtol=0)
                    if index in old_buffers:
                        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"],
                                                   old_buffers[index], atol=0, rtol=0)
                if active_steps[index] == 0:
                    assert param not in optimizer.state
                else:
                    state = optimizer.state[param]
                    assert state["step"].item() == active_steps[index]
                    if index < len(shapes) - 1:
                        assert set(state) == {"step", "momentum_buffer"}
                        assert state["step"].shape == torch.Size([])
                        assert state["step"].dtype == torch.float32
                        assert state["momentum_buffer"].shape == param.shape
        for params, optimizer in runs[1:]:
            for expected, actual in zip(reference[0], params):
                max_parameter_error = max(max_parameter_error, (expected - actual).abs().max().item())
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                if expected in reference[1].state and "momentum_buffer" in reference[1].state[expected]:
                    expected_buffer = reference[1].state[expected]["momentum_buffer"]
                    actual_buffer = optimizer.state[actual]["momentum_buffer"]
                    max_momentum_error = max(max_momentum_error,
                                             (expected_buffer - actual_buffer).abs().max().item())
                    torch.testing.assert_close(actual_buffer, expected_buffer, atol=1e-7, rtol=1e-6)
    print(f"Muon CPU multistep/resume max_abs: parameter={max_parameter_error:.9g}, "
          f"momentum={max_momentum_error:.9g}")


def test_initialization_dtype_must_not_be_silently_cast(initial_states, tmp_path):
    from molgap.family_training_kernel import load_family_model
    from molgap.v4_runtime import read_frozen_state_artifact
    path, digest = initial_states["neural_atom_k1"]
    state, _ = read_frozen_state_artifact(path, expected_state_sha256=digest)
    altered = {key: value.double() if value.is_floating_point() else value for key, value in state.items()}
    new_digest = state_dict_sha256(altered)
    wrong = tmp_path / "wrong-dtype.pt"
    atomic_torch_save(wrong, altered)
    recipe = FamilyRecipe("neural_atom_k1", "reference", 42, new_digest,
        DatasetSelection("synthetic.json", "2" * 64, False), 128, 0, 1,
        .001, .01, ScheduleSpec("cosine_epoch", 1), "final-live", "torch-pass-randperm", 1,
        target_mean_eV=0., target_std_eV=1.)
    with torch.random.fork_rng(devices=[]), pytest.raises(ValueError, match="dtype"):
        load_family_model(recipe, wrong)


@pytest.fixture(scope="module")
def initial_states(tmp_path_factory):
    from molgap.qm9_neural_atom import make_encoder
    from molgap.pcqm_gptrans_v4 import _make_model
    root = tmp_path_factory.mktemp("random-family-initial")
    results = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for family, factory in (("neural_atom_k1", lambda: make_encoder("neural_atom_k1")),
                                ("gptrans_t", _make_model)):
            model = factory()
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            digest = state_dict_sha256(state)
            path = root / (family + ".pt")
            atomic_torch_save(path, {"model_state": state, "state_sha256": digest})
            results[family] = path, digest
            del model, state
    return results


def synthetic_batch():
    return Batch.from_data_list([Data(x=torch.zeros(3, 9, dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.zeros(4, 3, dtype=torch.long), random_walk_pe=torch.zeros(3, 16),
        y=torch.tensor([float(i)]), source_idx=torch.tensor([i])) for i in range(2)])


@pytest.mark.parametrize("family,mode", [
    ("neural_atom_k1", "reference"), ("neural_atom_k1", "ssma"),
    ("neural_atom_k1", "mean2"), ("neural_atom_k1", "mean2_clean_second"),
    ("gptrans_t", "reference"), ("gptrans_t", "pair_prenorm"),
    ("gptrans_t", "centered_logits"), ("gptrans_t", "memory_value"), ("gptrans_t", "memory_message"),
])
def test_family_addons_share_kernel_and_do_not_mutate_cpu_data(initial_states, family, mode):
    from dataclasses import replace
    path, digest = initial_states[family]
    recipe = FamilyRecipe(family, mode, 42, digest,
        DatasetSelection("synthetic.json", "2" * 64, True), 128, 2, 1,
        .001, .01, ScheduleSpec("cosine_epoch", 1), "best-development-live",
        "torch-pass-randperm", 1, target_mean_eV=0., target_std_eV=1.,
        torch_version=str(torch.__version__).split("+", 1)[0])
    if mode == "reference":
        recipe = replace(recipe, no_decay_modules=("BatchNorm" if family == "neural_atom_k1" else "LayerNorm",))
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        kernel = FamilyKernel(recipe, path, device="cpu")
        if recipe.no_decay_modules:
            from torch.nn.modules.batchnorm import _BatchNorm
            norm_type = _BatchNorm if family == "neural_atom_k1" else torch.nn.LayerNorm
            exempt = {id(p) for module in kernel.model.modules() if isinstance(module, norm_type)
                      for name, p in module.named_parameters(recurse=False) if name in {"weight", "bias"} and p.requires_grad}
            assert {id(p) for p in kernel.optimizer.param_groups[1]["params"]} == exempt
            assert [g["weight_decay"] for g in kernel.optimizer.param_groups] == [.01, 0.]
            assert all(id(p) not in exempt for p in kernel.optimizer.param_groups[0]["params"])
        batch = synthetic_batch()
        before = deepcopy(batch.to_dict())
        kernel.model.train()
        loss, absolute, rows = kernel.step(batch, check_finite=True)
        assert loss.ndim == 0 and loss.grad_fn is None and torch.isfinite(loss)
        assert torch.isfinite(absolute) and rows == 2
        assert "execution_layout" not in batch
        for key, value in before.items():
            assert torch.equal(batch[key], value)
        assert kernel.optimizer.state
        kernel.model.eval()
        with torch.no_grad():
            assert kernel.forward(batch).shape == (2,)
            if (family, mode) == ("neural_atom_k1", "reference"):
                from molgap.training.core import evaluate_development
                kernel.model.head[-1].bias.fill_(float("nan"))
                with pytest.raises(ValueError, match=r"predictions non-finite: count=2; source_idx=\[0, 1\]"):
                    evaluate_development(kernel.model, [batch], forward=lambda model, batch: kernel.forward(batch),
                                         rows=2, source_start=0, mean=0., std=1.)
        assert not hasattr(kernel.model, "_execution_layout")
        assert not hasattr(kernel.model, "_execution_graph_count")
        kernel.close()
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def test_ema_cpu_checkpoint_preserves_destination_schema_and_does_not_alias():
    from molgap.gptrans_execution_hotpath import BoundForeachEMA
    model = torch.nn.BatchNorm1d(2)
    ema = BoundForeachEMA(model)
    saved = {key: value.clone() for key, value in ema.state_dict().items()}
    ema.load_state_dict(saved)
    saved["weight"].zero_()
    assert torch.equal(ema.state["weight"], torch.ones(2))
    ema.update(model)
    invalid = dict(ema.state_dict())
    invalid["weight"] = torch.zeros(3)
    with pytest.raises(RuntimeError, match="shape/dtype"):
        ema.load_state_dict(invalid)
    destination = BoundForeachEMA(torch.nn.Linear(2, 2, device="meta"))
    destination.load_state_dict(torch.nn.Linear(2, 2).state_dict())
    assert all(value.device.type == "meta" for value in destination.state.values())


@pytest.mark.parametrize("kind", ["cosine_step", "cosine_epoch"])
def test_native_cosine_state_resume_matches_uninterrupted(kind):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    kernel = SimpleNamespace(optimizer=optimizer,
        recipe=SimpleNamespace(schedule=ScheduleSpec(kind, 4), learning_rate=.01))
    schedule = ContractSchedule(kernel)
    def advance(unit):
        optimizer.zero_grad(set_to_none=True)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        if kind == "cosine_step":
            schedule.after_step(unit)
        else:
            schedule.after_pass(unit - 1, None)
    advance(1)
    saved, saved_optimizer = deepcopy(schedule.state_dict()), deepcopy(optimizer.state_dict())
    advance(2)
    expected = optimizer.param_groups[0]["lr"]
    schedule.load_state_dict(saved)
    optimizer.load_state_dict(saved_optimizer)
    advance(2)
    assert optimizer.param_groups[0]["lr"] == expected


@pytest.mark.parametrize("family,prefix,cached", [("neural_atom_k1", "k1", False),
    ("neural_atom_k1", "k1", True), ("gptrans_t", "gptrans", False)])
def test_ordered_registered_addons_reach_accelerated_forward_and_optimizer(initial_states, family, prefix, cached):
    from pathlib import Path
    from molgap.experiment_execution import build_addon_declaration
    from molgap.model_substrate import SlotBinding, SlotName
    from molgap.substrate_scale_addon import SlotScale
    from molgap.k1_execution_backend import K1ExecutionConfig
    path, digest = initial_states[family]
    interface = (family, "3" if prefix == "k1" else "2")
    addons = tuple(build_addon_declaration(interface, prefix + "_" + slot + "_scale",
        repo_root=Path(__file__).resolve().parents[1]) for slot in ("node", "aggregate", "readout"))
    recipe = FamilyRecipe(family, "composed", 42, digest,
        DatasetSelection("synthetic.json", "2" * 64, False), 128, 0, 1,
        .001, .01, ScheduleSpec("cosine_step", 1), "final-live", "torch-pass-randperm", 1,
        target_mean_eV=0., target_std_eV=1., addons=addons,
        k1_execution=K1ExecutionConfig() if cached else None,
        torch_version=str(torch.__version__).split("+", 1)[0])
    assert FamilyRecipe.from_dict(recipe.to_dict()) == recipe
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.random.fork_rng(devices=[]):
            kernel = FamilyKernel(recipe, path, device="cpu")
            owner = kernel.model._molgap_substrate
            assert owner.sealed
            expected_order = ["node_scale", "aggregate_scale", "readout_scale"]
            assert [entry.name for entry in owner.bindings] == expected_order
            calls = []
            for name, module in kernel.model._molgap_addons.items():
                module.register_forward_hook(lambda module, args, output, name=name: calls.append(name))
            loss, _, _ = kernel.step(synthetic_batch(), check_finite=True)
            assert torch.isfinite(loss) and calls == expected_order
            optimized = {id(p) for group in kernel.optimizer.param_groups for p in group["params"]}
            for name, parameter in owner.named_addon_parameters():
                assert id(parameter) in optimized
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert torch.count_nonzero(parameter.grad), name
                assert parameter in kernel.optimizer.state
            with pytest.raises(RuntimeError, match="sealed"):
                owner.attach("late", SlotScale("embedding"), SlotBinding(family, SlotName.PRE_READOUT))
            kernel.close()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill
