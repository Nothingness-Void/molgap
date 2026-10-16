"""Focused CPU checks for the registered K1 pre-norm variants."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
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
    base_state = {key: value.detach().clone() for key, value in raw_and_prenorm_models[0].state_dict().items()}
    derived_state = {key: value.detach().clone() for key, value in raw_and_prenorm_models[2].state_dict().items()}
    digest = state_dict_sha256(base_state)
    artifact = tmp_path / "prenorm-ada-initial.pt"
    atomic_torch_save(artifact, {"model_state": base_state, "state_sha256": digest})
    recipe = _recipe("prenorm_ada", digest)

    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with _strict_determinism():
            kernel = FamilyKernel(recipe, artifact, device="cpu")
            try:
                loaded_state = kernel.model.state_dict()
                assert set(loaded_state) == set(derived_state)
                assert all(torch.equal(value, loaded_state[key])
                           for key, value in derived_state.items())
                shared = [key for key in base_state
                          if not re.match(r"local_blocks\.\d+\.(?:norm1|norm3)\.", key)]
                assert all(torch.equal(base_state[key], loaded_state[key]) for key in shared)
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
    state = {key: value.detach().clone() for key, value in raw_and_prenorm_models[0].state_dict().items()}
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


def test_same_run_acceptance_accepts_shared_counters_but_rejects_different_initialization(tmp_path):
    from molgap.experiment_family_workflow import check_acceptance_plan
    from molgap.experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL

    expected = {
        "completed_passes": 1,
        "next_batch": 0,
        "optimizer_steps": 2,
        "sample_presentations": 256,
        "development_rows": 2,
        "source_idx_sha256": "d" * 64,
        "target_sha256": "e" * 64,
        "precision": "fp32",
    }
    contract = {
        "acceptance_requirements": expected,
        "development_role_identity": "synthetic-development-role",
        "training_recipe": {"name": "shared-synthetic-recipe"},
    }
    target_manifest = {
        "development_target_sha256": expected["target_sha256"],
        "target_encoding": "little-endian-float64-contiguous-raw-bytes",
    }

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    contract_pointer = write_json(tmp_path / "experiments/synthetic/recipe.json", contract)
    target_pointer = write_json(tmp_path / "experiments/synthetic/target.json", target_manifest)

    def reference(name, digest):
        return {"name": name, "version": "1", "sha256": digest}

    def arm(arm_id, role, *, candidate=False, initialization_sha="c" * 64):
        return {
            "arm_id": arm_id,
            "scientific_role": role,
            "family": {"name": "neural_atom_k1", "version": "3"},
            "base": reference("synthetic-k1-base", "1" * 64),
            "initialization": {
                "kind": "frozen_state", "seed": 42, "state_sha256": initialization_sha,
            },
            "data": {
                "dataset": reference("pcqm4mv2", "2" * 64),
                "split": reference("synthetic-split", "3" * 64),
                "roles": [
                    {"role": data_role, "membership_sha256": "4" * 64,
                     "row_order_sha256": "5" * 64, "usage_sha256": "6" * 64}
                    for data_role in ("train", "development")
                ],
                "feature_schema": "ogb-atom9-bond3-rwse16-v1",
                "feature_sha256": "7" * 64,
                "target": "pcqm4mv2-gap-eV-direct",
            },
            "training": {
                "recipe": reference("shared-family-training", contract_pointer["sha256"]),
                "overrides": {},
                "objective": reference("normalized-gap-l1", "8" * 64),
                "sampler": reference("frozen-family-pass-order", "9" * 64),
                "transform": reference("frozen-train-mean-sample-std", "a" * 64),
            },
            "addons": ([{
                "name": "k1_prenorm_ada", "version": "1", "config": {},
                "source_sha256": "b" * 64,
            }] if candidate else []),
            "addon_semantics": "ordered" if candidate else "baseline",
        }

    def make_spec(candidate_initialization_sha="c" * 64):
        return ExperimentSpec({
            "schema_version": SCHEMA_VERSION_V2,
            "experiment_id": "synthetic-k1-same-run",
            "logical_run_id": "synthetic-k1-run",
            "arms": [
                arm("reference", "reference"),
                arm("candidate", "candidate", candidate=True,
                    initialization_sha=candidate_initialization_sha),
            ],
            "platform": {
                "name": "local", "accelerator": "synthetic-cpu", "device_count": 1,
                "cpu_cores": 2, "memory_gib": 4, "atomic_checkpoints": True,
                "retrievable_chunks": True,
            },
            "prospective": {
                "arms": [
                    {"arm_id": arm_id, "trajectory_id": "T-" + arm_id,
                     "plan_spec_ref": "experiments/synthetic/" + arm_id + ".json",
                     "plan_spec_sha256": "f" * 64,
                     "output": "experiments/synthetic/output-" + arm_id}
                    for arm_id in ("reference", "candidate")
                ],
                "same_run_replay": {
                    "reference_arm_id": "reference", "candidate_arm_ids": ["candidate"],
                },
            },
            "evidence": {
                "policy": reference("molgap-v5", "c" * 64),
                "required_artifacts": [
                    "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact",
                ],
            },
            "terminal_protocol": TERMINAL_PROTOCOL,
        })

    def plan_for(spec):
        return {
            "format": "molgap-family-same-run-acceptance-plan-v1",
            "spec_identity": spec.identity,
            "arms": [
                {"arm_id": arm_id, "adapter": "k1-session-v1", "expected": expected,
                 "contract": contract_pointer, "target_manifest": target_pointer}
                for arm_id in ("reference", "candidate")
            ],
        }

    matching_spec = make_spec()
    accepted = check_acceptance_plan(matching_spec, tmp_path, plan_for(matching_spec))
    assert accepted["status"] == "ACCEPTANCE_INPUTS_AVAILABLE", accepted
    assert all(row["status"] == "AVAILABLE" for row in accepted["arms"])

    mismatched_spec = make_spec("f" * 64)
    blocked = check_acceptance_plan(mismatched_spec, tmp_path, plan_for(mismatched_spec))
    assert blocked["status"] == "BLOCKED", blocked
    candidate_result = next(row for row in blocked["arms"] if row["arm_id"] == "candidate")
    assert candidate_result["blockers"] == ["Same-run peer identity differs: initialization"]

