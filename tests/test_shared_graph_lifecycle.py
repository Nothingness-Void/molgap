"""A new model and tiny addons reuse the actual local release/output lifecycle."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import MappingProxyType

import torch
import pytest

from molgap import experiment_execution as execution
from molgap import experiment_spec as declaration
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_source_inventory import registered_source_files
from molgap.experiment_preflight import _unpack
from molgap.experiment_workflow import accept_workflow, prepare_workflow
from molgap.experiment_resume import restore_resume_bundle, safe_cpu_torch_load
from molgap.experiment_workflow_resume import (
    RESUME_PLAN_FORMAT, build_workflow_resume, prepare_resumed_workflow,
)
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.v4_runtime import normalized_source_sha256
from test_experiment_lifecycle_integration import (
    _git, _launch_receipt, _plan_input, _prepare_reference_inputs,
    _synthetic_policy, _terminal_locations,
)
from test_experiment_workflow import _arm, _spec_payload, _write_strict_retained_reference_plan
from test_comparison_readiness import _target_transform_asset
from test_graph_screen_training import _input_fixture


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ("synthetic_shared_graph", "1")
MODEL_SOURCE = '''import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(9, 1)

    def forward(self, batch):
        nodes = batch.x.float()
        return self.readout(global_mean_pool(nodes, batch.batch)).reshape(-1)

def make_model(arm):
    return Model()
'''
ADDON_SOURCE = '''import torch

def apply_addon(model, config):
    with torch.no_grad():
        model.readout.bias.add_(config["shift"])
'''


def _load_module(name, path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, name, module)
    module_spec.loader.exec_module(module)
    return module


def _install_family(repo, monkeypatch):
    contract = declaration.FamilyContract(*FAMILY, "molgap.synthetic_shared_graph",
        "graph_gap_screen_v1", "ogb-atom9-bond3-rwse16-v1", ("train", "development"),
        "seed42-epoch-global-randperm-v1", "train-mean-unbiased-std")
    addon_contract = declaration.AddonContract(FAMILY[0], "readout-shift",
        "molgap.synthetic_shared_addon", (declaration.AddonConfigField("shift", value=1),))
    families = MappingProxyType({**declaration.FAMILIES, FAMILY: contract})
    addons = MappingProxyType({**declaration.ADDONS, ("synthetic_shift", "1"): addon_contract})
    monkeypatch.setattr(declaration, "FAMILIES", families)
    monkeypatch.setattr(declaration, "ADDONS", addons)
    monkeypatch.setattr(execution, "FAMILIES", families)
    monkeypatch.setattr(execution, "ADDONS", addons)
    adapter = execution.graph_training_adapter(FAMILY,
        model_factory="molgap.synthetic_shared_graph:make_model",
        addons=(execution.TrainingAddon("synthetic_shift", "1", "shift",
                    apply_hook="molgap.synthetic_shared_addon:apply_addon"),))
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS",
        MappingProxyType({**execution.TRAINING_ADAPTERS, FAMILY: adapter}))
    arm = _arm("neural_atom_k1", "2", "new_graph")
    arm["family"] = dict(name=FAMILY[0], version=FAMILY[1])
    for field, name in (("recipe", contract.recipe), ("sampler", contract.sampler),
                        ("transform", contract.transform)):
        arm["training"][field]["name"] = name
    arm["addons"] = [{"name": "synthetic_shift", "version": "1", "config": {"shift": 1},
                     "source_sha256": "a" * 64}]
    arm["addon_semantics"] = "ordered"
    spec = declaration.ExperimentSpec(_spec_payload([arm], platform="kaggle"))
    sources = registered_source_files(spec)
    for relative in sources:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "src/molgap/synthetic_shared_graph.py":
            target.write_text(MODEL_SOURCE, encoding="utf-8")
        elif relative == "src/molgap/synthetic_shared_addon.py":
            target.write_text(ADDON_SOURCE, encoding="utf-8")
        else:
            shutil.copyfile(ROOT / relative, target)
    # A fresh process must see the reviewed registrations from packaged bytes.
    spec_path = repo / "src/molgap/experiment_spec.py"
    with spec_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nFAMILIES = MappingProxyType({{**FAMILIES, {FAMILY!r}: {contract!r}}})\n")
        handle.write(f"ADDONS = MappingProxyType({{**ADDONS, ('synthetic_shift', '1'): {addon_contract!r}}})\n")
    execution_path = repo / "src/molgap/experiment_execution.py"
    source = execution_path.read_text(encoding="utf-8")
    registration = (f"\n    {FAMILY!r}: graph_training_adapter({FAMILY!r}, "
        f"model_factory={adapter.model_factory!r}, addons={adapter.addons!r}),")
    source = source.replace("TRAINING_ADAPTERS = MappingProxyType({",
        "TRAINING_ADAPTERS = MappingProxyType({" + registration, 1)
    execution_path.write_text(source, encoding="utf-8")
    model_module = _load_module("molgap.synthetic_shared_graph", repo / "src/molgap/synthetic_shared_graph.py", monkeypatch)
    _load_module("molgap.synthetic_shared_addon", repo / "src/molgap/synthetic_shared_addon.py", monkeypatch)
    owner = _load_module("molgap.graph_screen_training", repo / "src/molgap/graph_screen_training.py", monkeypatch)
    arm["base"]["sha256"] = normalized_source_sha256(repo / "src/molgap/synthetic_shared_graph.py")
    arm["addons"][0]["source_sha256"] = normalized_source_sha256(repo / "src/molgap/synthetic_shared_addon.py")
    _git(repo, "init")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Shared graph fixture")
    _git(repo, "config", "core.autocrlf", "false")
    return arm, owner, model_module, sources


def _strict_acceptance(repo, spec, expected, recipes):
    relative = _write_strict_retained_reference_plan(repo, spec, expected, recipes)
    transform = repo / "experiments/reference/target_transform.json"
    transform.write_bytes(json_bytes(_target_transform_asset()))
    (repo / "experiments/reference/contract.json").write_bytes(json_bytes({"schema": "contract-v1", "synthetic": True}))
    path = repo / relative
    acceptance = json.loads(path.read_bytes())
    for entry in acceptance["arms"]:
        entry["reference_artifacts"]["target_transform_asset"]["sha256"] = file_digest(transform)
    path.write_bytes(json_bytes(acceptance))
    return relative


def test_new_model_and_tiny_addon_reuse_release_training_and_acceptance(tmp_path, monkeypatch):
    from molgap.shared_model_adapter import build_model
    from molgap.v4_runtime import model_state_sha256

    repo = tmp_path / "repo"
    repo.mkdir()
    arm, owner, _, sources = _install_family(repo, monkeypatch)
    _prepare_reference_inputs(repo)
    inputs, _, manifest_raw = _input_fixture(repo / "fixed_graphs", monkeypatch)
    train, development = inputs.role("train"), inputs.role("development")
    config = {
        "epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0,
        "target_mean_eV": float(train.target_eV.mean()),
        "target_std_eV": float(train.target_eV.std(unbiased=True)),
        "train_rows": train.rows, "development_rows": development.rows,
        "train_source_idx_sha256": train.source_idx_sha256,
        "train_target_sha256": train.target_sha256,
        "manifest_sha256": owner.hashlib.sha256(manifest_raw).hexdigest(),
        "row_order_fingerprint": owner.sampler_order_sha256(train.rows, 1),
    }
    recipe = execution.build_family_recipe(FAMILY, addon="synthetic_shift",
        source_idx_sha256=development.source_idx_sha256,
        target_sha256=development.target_sha256, recipe_config=config)
    recipe_path = repo / "recipes/new_graph.json"
    recipe_path.parent.mkdir()
    recipe_path.write_bytes(json_bytes(recipe))
    arm["training"]["recipe"]["sha256"] = normalized_source_sha256(recipe_path)
    arm["training"]["sampler"]["sha256"] = config["row_order_fingerprint"]
    torch.manual_seed(42)
    initial_model = build_model(execution.training_adapter(arm), arm)
    initial_state = repo / "initial.pt"
    torch.save(initial_model.state_dict(), initial_state)
    arm["initialization"]["state_sha256"] = model_state_sha256(initial_model)
    _git(repo, "add", "--", *sources, "recipes/new_graph.json")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic model and shared release sources")
    payload = _spec_payload([arm], platform="kaggle")
    binding = payload["prospective"]["arms"][0]
    plan_input = _plan_input(repo, arm, binding, 0, _synthetic_policy(repo))
    plan_path = repo / binding["plan_spec_ref"]
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(json_bytes(plan_input))
    binding["plan_spec_sha256"] = file_digest(plan_path)
    spec = declaration.ExperimentSpec(payload)
    expected = recipe["acceptance_requirements"]
    recipes = {"new_graph": "recipes/new_graph.json"}
    acceptance_plan = _strict_acceptance(repo, spec, expected, recipes)
    plan = {
        "format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": [], "arms": [{"arm_id": "new_graph", "device": 0,
            "recipe": recipes["new_graph"], "initial_state": str(initial_state)}],
        "acceptance_plan": acceptance_plan,
        "kaggle": {"account": "synthetic-account",
            "kernel": "synthetic-account/synthetic-pair-run", "title": "Synthetic Pair Run",
            "datasets": ["synthetic-account/source-data"], "source_dataset": "synthetic-account/source-data",
            "accelerator": "NvidiaTeslaT4"},
    }
    prepared = tmp_path / "prepared"
    result = prepare_workflow(spec, repo, plan, prepared)
    assert result["status"] == "PREPARED_FOR_PLATFORM", result
    package = prepared / "package"
    isolated = tmp_path / "isolated_source"
    isolated.mkdir()
    _unpack(package, isolated)
    # Resolve the complete ABI from the actual published source bytes in a
    # fresh process. Host editable imports must never fill a missing file.
    probe = '''import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from molgap.experiment_preflight import _PackageOnly, _origins
root = pathlib.Path(sys.argv[1])
sys.meta_path.insert(0, _PackageOnly(root))
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_execution import training_adapter, validate_training_registry
from molgap.shared_model_adapter import build_model
from molgap import graph_screen_training
spec = ExperimentSpec.from_json(pathlib.Path(sys.argv[2]).read_text())
validate_training_registry()
arm = spec.to_dict()["arms"][0]
adapter = training_adapter(arm)
model = build_model(adapter, arm)
assert model.readout.bias.dtype.is_floating_point
assert all(callable(getattr(graph_screen_training, name)) for name in (
    "run_screen_preflight", "run_screen_arm", "run_screen_resume_preflight", "validate_screen_resume"))
origins = _origins(root)
assert "molgap.synthetic_shared_graph" in origins
assert "molgap.synthetic_shared_addon" in origins
assert "molgap.pcqm_graph_inputs" in origins
print("published model/addon/trainer ABI resolved")
'''
    resolved = subprocess.run([sys.executable, "-I", "-c", probe,
        str(isolated / "src"), str(package / "experiment_spec.json")],
        cwd=isolated, capture_output=True, text=True, timeout=60)
    assert resolved.returncode == 0, resolved.stdout + resolved.stderr
    receipt = _launch_receipt(spec, package, result["package_identity"], repo)
    staged = prepared / "source_dataset"
    shutil.copytree(repo / "fixed_graphs", staged / "fixed_graphs")
    output = repo / "retained/family_outputs/new_graph"
    arguments = dict(spec=spec, package_dir=package,
        expected_package_identity=result["package_identity"], arm_id="new_graph", mode="shift",
        recipe_path=recipe_path, initial_state_path=staged / "initial_states/new_graph.pt",
        input_root=staged, account="synthetic-account", run_reference="synthetic-account/synthetic-pair-run",
        trajectory_id=binding["trajectory_id"])
    # The fixture proves local mechanics; the production device gate is
    # independently tested and an actual T4 allocation remains required.
    monkeypatch.setattr(owner, "_execution_device", lambda spec: torch.device("cpu"), raising=False)
    qualified_cuda_devices = owner._qualified_cuda_devices
    monkeypatch.setattr(owner, "_qualified_cuda_devices",
        lambda runtime, platform: qualified_cuda_devices(runtime, "local"))
    owner.run_screen_preflight(**arguments, output=output)
    trained = owner.run_screen_arm(**arguments, output=output)
    assert trained["status"] == "MECHANICALLY_VERIFIED", trained
    uninterrupted = safe_cpu_torch_load(output / "last_checkpoint.pt")
    partial = repo / "retained/partial/new_graph"
    owner.run_screen_preflight(**arguments, output=partial)
    original_train = owner._train_epoch
    calls = 0
    def interrupted_train(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption after the first acknowledged epoch")
        return original_train(*args, **kwargs)
    monkeypatch.setattr(owner, "_train_epoch", interrupted_train)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        owner.run_screen_arm(**arguments, output=partial)
    monkeypatch.setattr(owner, "_train_epoch", original_train)
    bundle = tmp_path / "resume_bundle"
    bundled = build_workflow_resume(spec, repo, prepared, package, result["package_identity"],
        receipt, bundle, arm_id="new_graph", source_output=partial)
    assert bundled["status"] == "RESUME_VERIFIED", bundled
    assert bundled["cursor"]["epoch"] == 1
    recovery = tmp_path / "recovery"
    recovered = prepare_resumed_workflow(spec, repo, prepared, package, result["package_identity"],
        receipt, recovery, plan={"format": RESUME_PLAN_FORMAT, "spec_identity": spec.identity,
            "arms": {"new_graph": str(bundle)}})
    assert recovered["status"] == "PREPARED_FOR_PLATFORM", recovered
    assert recovered["prospective_published"] is False
    output = repo / "retained/resumed/new_graph"
    context_for_restore = RunContext.for_training(spec, package,
        expected_package_identity=result["package_identity"], arm_id="new_graph",
        account="synthetic-account", run_reference="synthetic-account/synthetic-pair-run")
    trajectory = repo / binding["output"] / "trajectory.json"
    restore_resume_bundle(bundle, output, spec, arm_id="new_graph",
        context=context_for_restore, trajectory=trajectory)
    resume_preflight = tmp_path / "resume_preflight"
    owner.run_screen_resume_preflight(**arguments, output=resume_preflight, resume_output=output)
    trained = owner.run_screen_arm(**arguments, output=output, preflight_dir=resume_preflight)
    assert trained["status"] == "MECHANICALLY_VERIFIED", trained
    resumed = safe_cpu_torch_load(output / "last_checkpoint.pt")
    def assert_exact(left, right):
        if torch.is_tensor(left):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert type(left) is type(right) and len(left) == len(right)
            for one, two in zip(left, right):
                assert_exact(one, two)
        else:
            assert left == right
    for key in ("model", "optimizer", "scheduler", "rng_state", "cursor",
                "optimizer_step", "sample_presentations"):
        assert_exact(uninterrupted[key], resumed[key])
    context = RunContext.from_launch(spec, receipt, package,
        expected_package_identity=result["package_identity"], arm_id="new_graph")
    assert owner.inspect_output(output, context=context, expected=expected)["status"] == "MECHANICALLY_VERIFIED"
    locations = _terminal_locations(repo, spec, {"new_graph": output}, result["source_commit"])
    accepted = accept_workflow(spec, repo, {"new_graph": {"output_dir": output.relative_to(repo).as_posix(),
        "expected": expected}}, receipt_path=receipt, package_dir=package,
        expected_package_identity=result["package_identity"], locations=locations, execute=False)
    assert accepted["status"] == "MECHANICALLY_VERIFIED", accepted
    closed = accept_workflow(spec, repo, {"new_graph": {"output_dir": output.relative_to(repo).as_posix(),
        "expected": expected}}, receipt_path=receipt, package_dir=package,
        expected_package_identity=result["package_identity"], locations=locations, execute=True)
    assert closed["status"] == "COMPLETE", closed
    inventory = json.loads((package / "SOURCE_FILES.json").read_bytes())["files"]
    paths = {entry["path"] for entry in inventory}
    assert "src/molgap/graph_screen_training.py" in paths
    assert "src/molgap/synthetic_shared_graph.py" in paths
    assert "src/molgap/synthetic_shared_addon.py" in paths
    assert "src/molgap/k1_screen_training.py" not in paths
    assert "src/molgap/gptrans_screen_workflow.py" not in paths
