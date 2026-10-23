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
    path, digest = initial_states[family]
    recipe = FamilyRecipe(family, mode, 42, digest,
        DatasetSelection("synthetic.json", "2" * 64, True), 128, 2, 1,
        .001, .01, ScheduleSpec("cosine_epoch", 1), "best-development-live",
        "torch-pass-randperm", 1, target_mean_eV=0., target_std_eV=1.,
        torch_version=str(torch.__version__).split("+", 1)[0])
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        kernel = FamilyKernel(recipe, path, device="cpu")
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


@pytest.mark.parametrize("learn_scale", [0, 1])
def test_message_norm_accelerated_initialization_gradients_and_checkpoint_resume(initial_states, learn_scale):
    from io import BytesIO
    from pathlib import Path
    from molgap.experiment_execution import build_addon_declaration
    from molgap.family_training_kernel import load_family_model
    from molgap.k1_execution_backend import K1ExecutionConfig
    from molgap.training_reproducibility import capture_rng_state, restore_rng_state
    from test_family_training_session import assert_equal
    path, digest = initial_states["neural_atom_k1"]
    root = Path(__file__).resolve().parents[1]
    def recipe_for(value):
        declaration = build_addon_declaration(("neural_atom_k1", "3"), "k1_message_norm",
            repo_root=root, config={"learn_scale": value})
        return FamilyRecipe("neural_atom_k1", "composed", 42, digest,
            DatasetSelection("synthetic.json", "2" * 64, False), 128, 0, 2,
            .001, .01, ScheduleSpec("cosine_step", 2), "final-live", "torch-pass-randperm", 1,
            target_mean_eV=0., target_std_eV=1., addons=(declaration,),
            k1_execution=K1ExecutionConfig(), ema_decay=.999,
            torch_version=str(torch.__version__).split("+", 1)[0])
    for invalid in (True, False, -1, 2, 1.0):
        with pytest.raises(ValueError):
            recipe_for(invalid)
    recipe = recipe_for(learn_scale)
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
            counterpart, _ = load_family_model(recipe_for(1 - learn_scale), path)
            assert_equal(kernel.model.state_dict(), counterpart.state_dict())
            assert sum(p.numel() for p in kernel.model.parameters()) - sum(
                p.numel() for p in counterpart.parameters()) == (9 if learn_scale else -9)
            kernel.model.eval()
            counterpart.eval()
            batch = synthetic_batch()
            with torch.no_grad():
                assert torch.equal(kernel.forward(batch), kernel.forward(batch, model=counterpart))
            del counterpart
            owner = kernel.model._molgap_substrate
            assert owner.sealed and len(owner.bindings) == 9
            calls = []
            for name, module in kernel.model._molgap_addons.items():
                module.register_forward_hook(lambda module, args, output, name=name: calls.append(name))
            kernel.model.train()
            kernel.step(batch, check_finite=True)
            assert calls == ["message_norm_" + str(i) for i in range(1, 10)]
            assert not owner._k1_nodes
            parameters = list(owner.named_addon_parameters())
            assert len(parameters) == 9 * learn_scale
            for name, parameter in parameters:
                assert parameter.grad is not None and torch.isfinite(parameter.grad)
                assert torch.count_nonzero(parameter.grad), name
                assert parameter in kernel.optimizer.state
            buffer = BytesIO()
            torch.save({"model": kernel.model.state_dict(), "optimizer": kernel.optimizer.state_dict(),
                        "ema": kernel.ema.state_dict(), "rng": capture_rng_state()}, buffer)
            kernel.step(batch, check_finite=True)
            expected = deepcopy((kernel.model.state_dict(), kernel.optimizer.state_dict(), kernel.ema.state_dict()))
            restored = FamilyKernel(recipe, path, device="cpu")
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=False)
            restored.model.load_state_dict(saved["model"], strict=True)
            restored.optimizer.load_state_dict(saved["optimizer"])
            restored.ema.load_state_dict(saved["ema"])
            restore_rng_state(saved["rng"])
            restored.model.train()
            restored.step(batch, check_finite=True)
            assert_equal(expected, (restored.model.state_dict(), restored.optimizer.state_dict(), restored.ema.state_dict()))
            restored.close()
            kernel.close()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill
