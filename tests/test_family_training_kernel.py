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


def test_k1_muon_matches_fp32_matrix_update_and_native_adamw(tmp_path, monkeypatch):
    from molgap.k1_muon import MuonAdamW, parameter_groups
    model = torch.nn.Module()
    model.hidden = torch.nn.Linear(2, 3)
    model.head = torch.nn.Linear(3, 1)
    optimizer = MuonAdamW(parameter_groups(model, .03), lr=.01, weight_decay=.03)
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
        optimizer.step()
        native.step()
        torch.testing.assert_close(model.hidden.weight, expected.float(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(model.head.weight, native_head, atol=0, rtol=0)
    assert set(optimizer.state[model.hidden.weight]) == {"step", "momentum_buffer"}
    assert optimizer.state[model.hidden.weight]["step"].item() == 2

    from molgap.family_training_kernel import make_family_optimizer, parameter_groups_for_recipe
    model.hidden = torch.nn.Linear(2, 2, bias=False)
    model.hidden.weight.data.copy_(torch.tensor([[.1, .2], [-.3, .4]]))
    recipe = SimpleNamespace(optimizer="k1-soap-adamw-fp32-v1", learning_rate=.01, weight_decay=.03)
    optimizer = make_family_optimizer(recipe, parameter_groups_for_recipe(recipe, model))
    native_head = torch.nn.Parameter(model.head.weight.detach().clone())
    native = torch.optim.AdamW([native_head], lr=.01, weight_decay=.03)
    initial = model.hidden.weight.detach().clone()
    for index in range(12):
        model.hidden.weight.grad = torch.tensor([[.2, -.4], [.1, .6]]) + index * torch.tensor([[.003, .002], [-.004, .005]])
        model.head.weight.grad = torch.full_like(model.head.weight, .2)
        native_head.grad = model.head.weight.grad.clone()
        optimizer.step()
        native.step()
        torch.testing.assert_close(model.head.weight, native_head, atol=0, rtol=0)
        if index == 0:
            assert torch.equal(model.hidden.weight, initial)
            assert optimizer.state[model.hidden.weight]["step"] == 0
            path = tmp_path / "soap-initialized.pt"
            torch.save(optimizer.state_dict(), path)
            optimizer = make_family_optimizer(recipe, parameter_groups_for_recipe(recipe, model))
            optimizer.load_state_dict(torch.load(path, weights_only=True))
    # Pinned author's implementation on these fixed gradients, including its QR refresh.
    expected = torch.tensor([[-.04356049746274948, .17716053128242493],
                             [-.27684491872787476, .25547081232070923]])
    torch.testing.assert_close(model.hidden.weight, expected, atol=1e-6, rtol=1e-6)
    assert optimizer.state[model.hidden.weight]["step"] == 11
    assert all(matrix.dtype == torch.float32 for key in ("GG", "Q")
               for matrix in optimizer.state[model.hidden.weight][key])
    def failed_eigh(matrix):
        assert matrix.dtype == torch.float32
        raise torch.linalg.LinAlgError("observed FP32 failure")
    monkeypatch.setattr(torch.linalg, "eigh", failed_eigh)
    optimizer.state.clear()
    with pytest.raises(torch.linalg.LinAlgError, match="observed FP32 failure"):
        optimizer.step()


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
        optimizer="k1-muon-adamw-fp32-v1" if prefix == "k1" and not cached else "adamw",
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
