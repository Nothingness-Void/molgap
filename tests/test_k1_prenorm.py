"""Focused CPU checks for the registered K1 pre-norm variants."""
from copy import deepcopy
from contextlib import contextmanager
import json
import re

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from molgap.family_training_data import DatasetSelection, TrainingData
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_session import FamilyTrainingSession
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.k1_prenorm import AdaRMSNorm, PreNormLocalBlock, make_model
from molgap.training_reproducibility import atomic_torch_save
from molgap.v4_runtime import state_dict_sha256


def _tiny_batch(graphs=2, *, source_start=0):
    return Batch.from_data_list([
        Data(
            x=torch.zeros(3, 9, dtype=torch.long),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_attr=torch.zeros(4, 3, dtype=torch.long),
            random_walk_pe=torch.zeros(3, 16),
            y=torch.tensor([float(i % 5)]),
            source_idx=torch.tensor([source_start + i]),
        )
        for i in range(graphs)
    ])


def _recipe(mode, digest, *, steps=1, development_rows=2):
    development = development_rows > 0
    return FamilyRecipe(
        "neural_atom_k1", mode, 42, digest,
        DatasetSelection("synthetic.json", "2" * 64, development),
        128, development_rows, steps, .001, .01,
        ScheduleSpec("cosine_epoch", steps),
        "best-development-live" if development else "final-live",
        "python-epoch-shuffle", 1,
        target_mean_eV=0., target_std_eV=1., ema_decay=.999,
        finite_check_every=1,
        torch_version=str(torch.__version__).split("+", 1)[0],
    )


@contextmanager
def _strict_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


class _ShiftNorm(nn.Module):
    def __init__(self, amount, calls, name):
        super().__init__()
        self.amount, self.calls, self.name = amount, calls, name

    def forward(self, value):
        self.calls.append((self.name, value.detach().clone()))
        return value + self.amount


class _ScaleConv(nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def forward(self, value, edge_index, *, edge_attr):
        self.calls.append(("conv", value.detach().clone()))
        return value * 2


class _ScaleMLP(nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def forward(self, value):
        self.calls.append(("mlp", value.detach().clone()))
        return value * 3


def test_local_block_normalizes_each_branch_input_before_adding_residual():
    calls = []
    base = type("Base", (), {})()
    base.conv, base.mlp, base.dropout = _ScaleConv(calls), _ScaleMLP(calls), 0.
    base.norm1 = _ShiftNorm(10., calls, "norm1")
    base.norm3 = _ShiftNorm(1., calls, "norm3")
    base.norm_with_batch = False
    block = PreNormLocalBlock(base, adaptive=False).eval()
    x = torch.tensor([[1., 2.], [3., 4.]])

    actual = block(x, torch.empty(2, 0, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                   edge_attr=torch.empty(0, 3))

    hidden = x + 2 * (x + 10.)
    expected = hidden + 3 * (hidden + 1.)
    assert torch.equal(actual, expected)
    assert [name for name, _ in calls] == ["norm1", "conv", "norm3", "mlp"]
    assert torch.equal(calls[0][1], x)
    assert torch.equal(calls[1][1], x + 10.)
    assert torch.equal(calls[2][1], hidden)
    assert torch.equal(calls[3][1], hidden + 1.)


def test_adarmsnorm_alpha_one_beta_zero_is_identity():
    norm = AdaRMSNorm(channels=4)
    with torch.no_grad():
        norm.alpha.fill_(1.)
        norm.beta.zero_()
    value = torch.tensor([[0., 1., -2., 4.], [3., -1., .5, 2.]])
    assert torch.allclose(norm(value), value, rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def raw_and_prenorm_models():
    from molgap.qm9_neural_atom import make_encoder

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        raw = make_encoder("neural_atom_k1")
        torch.manual_seed(42)
        bn = make_model("prenorm_bn")
        torch.manual_seed(42)
        ada = make_model("prenorm_ada")
    return raw, bn, ada


def test_variants_have_nineteen_norms_retain_dropout_and_preserve_shared_weights(raw_and_prenorm_models):
    from torch.nn.modules.batchnorm import _BatchNorm

    raw, bn, ada = raw_and_prenorm_models
    assert len(bn.local_blocks) == len(ada.local_blocks) == 9
    assert sum(isinstance(module, _BatchNorm) for module in bn.modules()) == 19
    assert sum(isinstance(module, AdaRMSNorm) for module in ada.modules()) == 19
    assert all(block.dropout == pytest.approx(.05) for model in (bn, ada) for block in model.local_blocks)
    assert all(model.neural_atom_mixers["9"].dropout.p == pytest.approx(.05) for model in (bn, ada))

    raw_state = raw.state_dict()
    for variant in (bn, ada):
        state = variant.state_dict()
        shared = [key for key in raw_state
                  if not re.match(r"local_blocks\.\d+\.(?:norm1|norm3)\.", key)]
        assert all(key in state and torch.equal(raw_state[key], state[key]) for key in shared)
        assert "neural_atom_mixers.9.output_norm" in dict(variant.named_modules())


def test_variant_frozen_state_load_and_k1_update_have_finite_gradients_and_final_norm_optimizer_ownership(
        tmp_path, raw_and_prenorm_models):
    state = {key: value.detach().clone() for key, value in raw_and_prenorm_models[2].state_dict().items()}
    digest = state_dict_sha256(state)
    artifact = tmp_path / "prenorm-ada-initial.pt"
    atomic_torch_save(artifact, {"model_state": state, "state_sha256": digest})
    recipe = _recipe("prenorm_ada", digest)

    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with _strict_determinism():
            kernel = FamilyKernel(recipe, artifact, device="cpu")
            try:
                assert all(torch.equal(value, kernel.model.state_dict()[key])
                           for key, value in state.items())
                final_norm = kernel.model.neural_atom_mixers["9"].output_norm
                optimizer_ids = {id(parameter) for group in kernel.optimizer.param_groups
                                 for parameter in group["params"]}
                assert all(id(parameter) in optimizer_ids for parameter in final_norm.parameters())

                kernel.model.train()
                loss, absolute, rows = kernel.step(_tiny_batch(), check_finite=True)
                assert loss.ndim == 0 and torch.isfinite(loss)
                assert torch.isfinite(absolute) and rows == 2
                assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                           for parameter in final_norm.parameters())
            finally:
                kernel.close()
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def test_live_selection_keeps_ema_as_auxiliary_canonical_trace(tmp_path):
    from molgap.family_training_outputs import SessionOutputs
    from molgap.experiment_execution import build_family_recipe
    from molgap.experiment_family_workflow import tensor_digest
    from molgap.family_training_runner import INTERFACES, PROFILES
    from test_family_training_runner import context
    from test_family_training_session import SyntheticKernel

    recipe = _recipe("prenorm_bn", "2" * 64)
    template = _tiny_batch(graphs=1).to_data_list()[0]
    train, development = [], []
    for index in range(128):
        graph = template.clone()
        graph.source_idx = torch.tensor([index])
        graph.y = torch.tensor([float(index % 5)])
        train.append(graph)
    for index in range(2):
        graph = template.clone()
        graph.source_idx = torch.tensor([128 + index])
        graph.y = torch.tensor([float(index % 2)])
        development.append(graph)
    data = TrainingData(train, development, recipe.dataset.manifest_sha256,
        128, 2, 0, 128, 0., 1.)
    tuning = ExecutionTuning(cpu_threads=1, training_workers=0, development_workers=0)
    log = {"checkpoint": [], "selected": [], "passes": []}
    session = FamilyTrainingSession(SyntheticKernel(recipe), FamilyLoaders(recipe, data, tuning),
        checkpoint=lambda progress, state: log["checkpoint"].append(deepcopy(state)),
        selected=lambda progress, weights, state, evaluation:
            log["selected"].append((progress, weights, deepcopy(state), deepcopy(evaluation))),
        pass_finished=lambda progress, metrics:
            log["passes"].append((progress, deepcopy(metrics))))

    source = torch.cat([graph.source_idx for graph in development])
    target = torch.cat([graph.y for graph in development])
    record = build_family_recipe(INTERFACES[recipe.family], addon="k1_prenorm_bn",
        configuration=recipe, train_role_identity="3" * 64,
        development_role_identity="4" * 64,
        source_idx_sha256=tensor_digest(source, role="source_idx"),
        target_sha256=tensor_digest(target, role="target"))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    ctx = context("neural_atom_k1", contract)
    outputs = SessionOutputs(tmp_path / "output", ctx, contract=contract,
        adapter=PROFILES["neural_atom_k1"], trajectory_id="synthetic-prenorm",
        train_role_identity=record["train_role_identity"])
    session.checkpoint, session.selected = outputs.checkpoint, outputs.selected
    session.pass_finished, session.batch_finished = outputs.pass_finished, outputs.batch_finished
    outputs.logger.start(session.progress)
    progress = session.run()
    runtime = {"platform": ctx.platform, "account": ctx.account, "precision": "fp32",
        "source_commit": ctx.source_commit, "source_archive_sha256": ctx.source_archive_sha256}
    costs = [
        {"metric": "wall_seconds", "unit": "seconds", "status": "measured", "value": 1.,
         "semantics": "process_wall", "hardware": "synthetic-cpu"},
        {"metric": "device_seconds", "unit": "seconds", "status": "missing", "value": None,
         "semantics": "allocated_device", "hardware": "synthetic-cpu"},
    ]
    report = outputs.complete(progress, runtime=runtime, costs=costs)
    assert report["status"] == "MECHANICALLY_VERIFIED", report

    trace = json.loads((outputs.root / "canonical_trace.json").read_text(encoding="utf-8"))
    observations = [row for row in trace["observations"] if row["event"] == "observation"]
    assert observations and all(row["live_dev_metric"] is not None for row in observations)
    assert all(row["ema_dev_metric"] is not None for row in observations)
    assert trace["metric_semantics"]["live_dev_metric"]["weights"] == "live"
    assert trace["metric_semantics"]["ema_dev_metric"]["weights"] == "ema"
    selected = torch.load(outputs.root / "selected_model.pt", map_location="cpu", weights_only=True)
    assert selected["weights"] == "live"
    assert selected["selection"] == "best-development-live"
    minimum_live = min(row["live_dev_metric"] for row in observations)
    assert selected["development_mae_eV"] == pytest.approx(minimum_live)
    assert selected["optimizer_step"] in {
        row["optimizer_step"] for row in observations if row["live_dev_metric"] == minimum_live
    }


def test_native_prenorm_factory_supports_small_synthetic_session_resume(tmp_path,
        raw_and_prenorm_models):
    state = {key: value.detach().clone() for key, value in raw_and_prenorm_models[2].state_dict().items()}
    digest = state_dict_sha256(state)
    artifact = tmp_path / "prenorm-ada-session-initial.pt"
    atomic_torch_save(artifact, {"model_state": state, "state_sha256": digest})
    recipe = _recipe("prenorm_ada", digest, steps=2)

    template = _tiny_batch(graphs=1).to_data_list()[0]
    train, development = [], []
    for index in range(128):
        graph = template.clone()
        graph.source_idx = torch.tensor([index])
        graph.y = torch.tensor([float(index % 5)])
        train.append(graph)
    for index in range(2):
        graph = template.clone()
        graph.source_idx = torch.tensor([128 + index])
        graph.y = torch.tensor([float(index % 2)])
        development.append(graph)
    data = TrainingData(train, development, recipe.dataset.manifest_sha256,
        128, 2, 0, 128, 0., 1.)
    tuning = ExecutionTuning(cpu_threads=1, training_workers=0, development_workers=0)
    events = {"checkpoint": [], "selected": [], "passes": []}

    def new_session():
        kernel = FamilyKernel(recipe, artifact, device="cpu")
        loaders = FamilyLoaders(recipe, data, tuning)
        return FamilyTrainingSession(kernel, loaders,
            checkpoint=lambda progress, saved: events["checkpoint"].append(deepcopy(saved)),
            selected=lambda progress, weights, selected_state, evaluation:
                events["selected"].append((progress, weights, deepcopy(selected_state), evaluation)),
            pass_finished=lambda progress, metrics:
                events["passes"].append((progress, deepcopy(metrics))))

    old_threads = torch.get_num_threads()
    rng = torch.get_rng_state()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with _strict_determinism():
            first = new_session()
            paused = first.run(should_pause=lambda: True)
            assert paused.status == "paused" and paused.optimizer_step == 1
            retained = deepcopy(events["checkpoint"][-1])

            resumed = new_session()
            resumed.load_state_dict(retained)
            complete = resumed.run()
            assert complete.status == "complete" and complete.optimizer_step == 2
            assert len(events["passes"]) == 2
            assert all(metrics["development_ema"] is not None for _, metrics in events["passes"])
            assert events["selected"] and all(weights == "live" for _, weights, _, _ in events["selected"])
            assert "neural_atom_mixers.9.output_norm.alpha" in retained["model"]
    finally:
        torch.set_num_threads(old_threads)
        torch.set_rng_state(rng)

