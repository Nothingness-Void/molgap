"""CPU lifecycle proof for the shared new-family graph owner."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import types
from types import MappingProxyType

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap import experiment_execution as execution
from molgap import experiment_spec as declaration
from molgap import graph_screen_training as owner
from molgap import pcqm_topology as topology
from molgap.experiment_family_workflow import RunContext, inspect_output
from molgap.screen_policy import canonical_fingerprint
from molgap.v4_runtime import model_state_sha256, normalized_source_sha256


def _write_json(path: Path, value: dict) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


class _TinyGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(9, 8)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, batch):
        hidden = torch.relu(self.encoder(batch.x.float()))
        pooled = torch.zeros((int(batch.num_graphs), hidden.shape[1]), device=hidden.device)
        pooled.index_add_(0, batch.batch.long(), hidden)
        counts = torch.bincount(batch.batch.long(), minlength=int(batch.num_graphs)).clamp_min(1).to(hidden.dtype).reshape(-1, 1)
        return self.head(pooled / counts).reshape(-1)


def _shard(path: Path, start: int, rows: int = 128) -> dict:
    graphs = []
    for offset in range(rows):
        graphs.append(Data(
            x=torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.zeros((2, 3), dtype=torch.long),
            random_walk_pe=torch.zeros((2, 16), dtype=torch.float32),
            y=torch.tensor([5.0 + 0.01 * (start + offset)], dtype=torch.float32),
            row_index=torch.tensor([start + offset], dtype=torch.long),
        ))
    torch.save(InMemoryDataset.collate(graphs), path)
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def _input_fixture(tmp_path: Path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = []
    for role, starts in (("train", (0, 128)), ("development", (256,))):
        for shard, start in enumerate(starts):
            path = tmp_path / f"{role}_{shard}.pt"
            receipt = _shard(path, start)
            records.append({"role": role, "file": path.name, "rows": 128,
                            "source_idx_min": start, "source_idx_max": start + 127,
                            **receipt})
    manifest = {
        "format": "molgap-pcqm4mv2-fixed-subset-v1", "status": "complete",
        "identity": {"name": "ogb-train-100k", "train_rows": 256,
                      "development_rows": 128, "kaggle1": True,
                      "scnet_compatible": False},
        "source": {"official_archive_sha256": topology.OFFICIAL_ARCHIVE_SHA256,
                   "official_row_manifest_sha256": topology.OFFICIAL_ROW_MANIFEST_SHA256,
                   "external_data_used": False},
        "roles": {"train": {"source_idx_start": 0, "source_idx_stop": 256, "rows": 256},
                  "development": {"source_idx_start": 256, "source_idx_stop": 384, "rows": 128}},
        "graph_contract": {"feature_schema": "ogb", "node_feature_dim": 9,
                           "edge_feature_dim": 3, "rwse_dim": 16,
                           "geometry_method": "ETKDGv3", "optimization_method": "MMFF94s"},
        "assets": {"topology": records, "geometry": []},
        "aggregates": {"topology": "0" * 64, "geometry": "1" * 64},
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    manifest_path = tmp_path / "manifest.json"
    raw = _write_json(manifest_path, manifest)
    accepted = copy.deepcopy(topology.PREFLIGHT_DATASETS)
    accepted[("synthetic_graph_family", "1")] = {"ogb-train-100k": canonical_fingerprint(manifest)}
    monkeypatch.setattr(topology, "PREFLIGHT_DATASETS", accepted)
    inputs = owner.load_graph_inputs(tmp_path, expected_manifest_sha256=hashlib.sha256(raw).hexdigest())
    return inputs, manifest_path, raw


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    key = ("synthetic_graph_family", "1")
    contract = declaration.FamilyContract(
        key[0], key[1], "molgap.graph_screen_training", owner.RECIPE,
        owner.FEATURE_SCHEMA, ("train", "development"), owner.SAMPLER,
        "train-mean-unbiased-std",
    )
    families = MappingProxyType({**declaration.FAMILIES, key: contract})
    monkeypatch.setattr(declaration, "FAMILIES", families)
    monkeypatch.setattr(execution, "FAMILIES", families)
    module = sys.modules["molgap.graph_screen_training"]
    monkeypatch.setattr(module, "make_model", lambda arm: _TinyGraph(), raising=False)
    adapter = execution.graph_training_adapter(key, model_factory="molgap.graph_screen_training:make_model")
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS",
                        MappingProxyType({**execution.TRAINING_ADAPTERS, key: adapter}))
    inputs, manifest_path, manifest_bytes = _input_fixture(tmp_path / "inputs", monkeypatch)
    train = inputs.role("train"); development = inputs.role("development")
    config = {
        "epochs": 2, "learning_rate": 1e-2, "weight_decay": 0.0,
        "target_mean_eV": float(train.target_eV.mean()),
        "target_std_eV": float(train.target_eV.std(unbiased=True)),
        "train_rows": train.rows, "development_rows": development.rows,
        "train_source_idx_sha256": train.source_idx_sha256,
        "train_target_sha256": train.target_sha256,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "row_order_fingerprint": owner.sampler_order_sha256(train.rows, 1),
    }
    recipe = owner.build_screen_recipe("reference", family=key,
        source_idx_sha256=development.source_idx_sha256,
        target_sha256=development.target_sha256, recipe_config=config)
    recipe_path = tmp_path / "recipe.json"
    recipe_bytes = _write_json(recipe_path, recipe)
    # Make the factory's source identity explicit in the Spec and produce a
    # genuine frozen tensor state from the same reviewed factory.
    factory_sha = normalized_source_sha256(Path(module.__file__))
    torch.manual_seed(42)
    initial_model = _TinyGraph()
    initial_path = tmp_path / "initial.pt"
    torch.save(initial_model.state_dict(), initial_path)
    state_sha = model_state_sha256(initial_model)
    arm = {
        "arm_id": "synthetic_arm", "scientific_role": "candidate",
        "family": {"name": key[0], "version": key[1]},
        "base": {"name": "synthetic_factory", "version": "1", "sha256": factory_sha},
        "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": state_sha},
        "data": {
            "dataset": {"name": "pcqm4mv2", "version": "1", "sha256": "a" * 64},
            "split": {"name": "synthetic-split", "version": "1", "sha256": "b" * 64},
            "roles": [{"role": role, "membership_sha256": "c" * 64,
                       "row_order_sha256": "d" * 64, "usage_sha256": "e" * 64}
                      for role in ("train", "development")],
            "feature_schema": owner.FEATURE_SCHEMA, "feature_sha256": "f" * 64,
            "target": "pcqm4mv2-gap-eV-direct",
        },
        "training": {
            "recipe": {"name": owner.RECIPE, "version": "1", "sha256": hashlib.sha256(recipe_bytes).hexdigest()},
            "overrides": {}, "objective": {"name": owner.OBJECTIVE, "version": "1", "sha256": "1" * 64},
            "sampler": {"name": owner.SAMPLER, "version": "1", "sha256": config["row_order_fingerprint"]},
            "transform": {"name": "train-mean-unbiased-std", "version": "1", "sha256": "2" * 64},
        },
        "addons": [], "addon_semantics": "baseline",
    }
    payload = {
        "schema_version": declaration.SCHEMA_VERSION_V2,
        "experiment_id": "synthetic-graph-screen", "logical_run_id": "synthetic-graph-run",
        "platform": {"name": "local", "accelerator": "synthetic-cpu", "device_count": 1,
                      "cpu_cores": 8, "memory_gib": 32, "atomic_checkpoints": True,
                      "retrievable_chunks": True},
        "arms": [arm],
        "prospective": {"arms": [{"arm_id": arm["arm_id"], "trajectory_id": "synthetic-trajectory",
                                    "plan_spec_ref": "plans/synthetic_arm.json",
                                    "plan_spec_sha256": "9" * 64,
                                    "output": "experiments/prospective-synthetic_arm"}]},
        "evidence": {"policy": {"name": "molgap-v5", "version": "1", "sha256": "9" * 64},
                     "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": declaration.TERMINAL_PROTOCOL,
    }
    spec = declaration.ExperimentSpec(payload)
    context = RunContext(
        experiment_id=spec.to_dict()["experiment_id"], logical_run_id=spec.to_dict()["logical_run_id"],
        arm_id=arm["arm_id"], arm_identity=canonical_fingerprint(arm), spec_identity=spec.identity,
        family_name=key[0], family_version=key[1], source_commit="a" * 40,
        source_archive_sha256="b" * 64, package_identity="c" * 64,
        training_recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest(), platform="local",
        account="local", run_reference="local/synthetic", platform_version=None,
    )
    monkeypatch.setattr(RunContext, "for_training", classmethod(lambda cls, *args, **kwargs: context))
    monkeypatch.setattr(execution, "validate_staged_trajectory",
                        lambda *args, **kwargs: {"trajectory_id": "synthetic-trajectory"})
    return {"key": key, "spec": spec, "arm": arm, "inputs": inputs, "recipe": recipe,
            "recipe_path": recipe_path, "initial_path": initial_path, "context": context,
            "input_root": inputs.root}


def test_inputs_reject_geometry_and_preserve_fixed_rows(synthetic_case):
    inputs = synthetic_case["inputs"]
    assert inputs.role("train").rows == 256
    assert inputs.role("development").source_idx.tolist()[:3] == [256, 257, 258]
    assert inputs.role("train").source_idx_sha256
    batch = next(iter(inputs.train_loader(1)))
    assert topology.validate_ogb_gap_batch(batch) == 128
    assert not ({"pos", "edge_distance", "geometry_valid"} & set(batch.keys()))


def test_shared_owner_real_cpu_train_resume_and_inspect(tmp_path, synthetic_case, monkeypatch):
    case = synthetic_case
    preflight = tmp_path / "preflight"
    certificate = owner.run_screen_preflight(
        spec=case["spec"], package_dir=tmp_path, expected_package_identity="c" * 64,
        arm_id=case["arm"]["arm_id"], mode="reference", recipe_path=case["recipe_path"],
        initial_state_path=case["initial_path"], input_root=case["input_root"], output=preflight,
        account="local", run_reference="local/synthetic", trajectory_id="synthetic-trajectory",
    )
    assert certificate["status"] == "accepted"
    assert certificate["resume_roundtrip"]["compared_fields"] == dict.fromkeys(
        ("model", "optimizer", "scheduler", "rng_state", "cursor",
         "optimizer_step", "sample_presentations"), True)
    assert case["recipe"]["metric_semantics"]["live_train_metric"]["timing"].startswith("online-pre-update;")
    output = tmp_path / "output"
    result = owner.run_screen_arm(
        spec=case["spec"], package_dir=tmp_path, expected_package_identity="c" * 64,
        arm_id=case["arm"]["arm_id"], mode="reference", recipe_path=case["recipe_path"],
        initial_state_path=case["initial_path"], input_root=case["input_root"], output=output,
        account="local", run_reference="local/synthetic", trajectory_id="synthetic-trajectory",
        preflight_dir=preflight,
    )
    assert result["status"] == "MECHANICALLY_VERIFIED"
    assert (output / "last_checkpoint.pt").is_file()
    assert (output / "canonical_trace.json").is_file()
    report = inspect_output(output, context=case["context"], expected=case["recipe"]["acceptance_requirements"])
    assert report["status"] == "MECHANICALLY_VERIFIED"
    with pytest.raises(ValueError, match="Completed"):
        owner.validate_screen_resume(
            case["spec"], case["arm"]["arm_id"], None, output,
            owner._retained_artifacts(output),
            {"context": [output / "canonical_trace.json.context.json"],
             "provenance": [output / "training_contract.json"]},
            {"trajectory_id": "synthetic-trajectory"}, context=case["context"].to_dict())

    # A worker interruption after epoch 1 leaves a checkpoint and matching
    # canonical trace.  The same owner resumes at epoch 2 and must reach the
    # same model/prediction endpoint as the uninterrupted run above.
    partial = tmp_path / "partial"
    original_step = owner._train_epoch
    calls = {"count": 0}

    def stop_before_second_epoch(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic worker interruption")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(owner, "_train_epoch", stop_before_second_epoch)
    with pytest.raises(RuntimeError, match="synthetic worker interruption"):
        owner.run_screen_arm(
            spec=case["spec"], package_dir=tmp_path, expected_package_identity="c" * 64,
            arm_id=case["arm"]["arm_id"], mode="reference", recipe_path=case["recipe_path"],
            initial_state_path=case["initial_path"], input_root=case["input_root"], output=partial,
            account="local", run_reference="local/synthetic", trajectory_id="synthetic-trajectory",
            preflight_dir=preflight,
        )
    monkeypatch.setattr(owner, "_train_epoch", original_step)
    def validate_partial():
        return owner.validate_screen_resume(
            case["spec"], case["arm"]["arm_id"], None, partial,
            owner._retained_artifacts(partial),
            {"context": [partial / "canonical_trace.json.context.json"],
             "provenance": [partial / "training_contract.json"]},
            {"trajectory_id": "synthetic-trajectory"}, context=case["context"].to_dict())
    partial_resume = validate_partial()
    assert partial_resume["epoch"] == 1
    before_inspection = owner.tensor_safe_rng_state(owner.capture_rng_state())
    validate_partial()
    assert owner._tree_equal(before_inspection, owner.tensor_safe_rng_state(owner.capture_rng_state()))
    paths = owner._retained_artifacts(partial)
    original_bytes = {key: path.read_bytes() for key, path in paths.items()}
    original_state = {key: owner._checkpoint_payload(paths[key])
                      for key in ("checkpoint", "selected_model", "predictions")}
    trace = json.loads(original_bytes["trace"])
    # Rebind file hashes after each deliberate semantic corruption. This
    # verifies the owner gate independently of transport integrity checks.
    corruptions = (
        ("selected_model", lambda value: value.update(epoch=2), "outside"),
        ("selected_model", lambda value: value.update(optimizer_step=999), "epoch/step"),
        ("selected_model", lambda value: value.update(weights="ema"), "live trace"),
        ("selected_model", lambda value: value["context"].update(source_commit="d" * 40), "context"),
        ("selected_model", lambda value: value["model"]["head.bias"].add_(1), "differs from checkpoint"),
        ("predictions", lambda value: value["prediction_eV"].add_(1), "metric mismatch"),
        ("predictions", lambda value: value["source_idx"].add_(1), "development identity"),
        ("checkpoint", lambda value: value["scheduler"].update(last_epoch=999), "scheduler differs"),
        ("checkpoint", lambda value: value["optimizer"]["param_groups"][0].update(lr=10.0), "learning rate changed"),
        ("checkpoint", lambda value: value["optimizer"]["param_groups"][0].update(weight_decay=10.0), "frozen settings"),
        ("checkpoint", lambda value: value["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1), "moment tensor"),
        ("checkpoint", lambda value: value["optimizer"]["state"][0].update(step=torch.tensor(True)), "parameter step"),
    )
    for role, mutate, message in corruptions:
        changed = copy.deepcopy(original_state[role])
        mutate(changed)
        owner.atomic_torch_save(paths[role], changed)
        checkpoint = changed if role == "checkpoint" else copy.deepcopy(original_state["checkpoint"])
        checkpoint["owner_state"]["selected_endpoint"] = {
            "selected_model_sha256": owner.file_digest(paths["selected_model"]),
            "predictions_sha256": owner.file_digest(paths["predictions"])}
        owner.atomic_torch_save(paths["checkpoint"], checkpoint)
        updated_trace = copy.deepcopy(trace)
        updated_trace["observations"][-1]["checkpoint_identity"] = "sha256:" + owner.file_digest(paths["checkpoint"])
        _write_json(paths["trace"], updated_trace)
        with pytest.raises(ValueError, match=message):
            validate_partial()
        for key, raw in original_bytes.items():
            paths[key].write_bytes(raw)
    changed = copy.deepcopy(original_state["selected_model"])
    changed["model"]["head.bias"].add_(1)
    owner.atomic_torch_save(paths["selected_model"], changed)
    with pytest.raises(ValueError, match="acknowledged checkpoint"):
        validate_partial()
    paths["selected_model"].write_bytes(original_bytes["selected_model"])
    runtime_path = partial / "runtime_manifest.json"
    runtime_bytes = runtime_path.read_bytes()
    runtime_with_cuda = json.loads(runtime_bytes)
    runtime_with_cuda["accelerator"] = {"name": "Tesla T4", "device_count_visible": 1}
    _write_json(runtime_path, runtime_with_cuda)
    with pytest.raises(ValueError, match="CUDA RNG schema"):
        validate_partial()
    runtime_path.write_bytes(runtime_bytes)
    # The resume qualification itself checks the current runtime before
    # publishing a certificate, including its device gate.
    runtime_builder = owner.build_runtime_manifest
    monkeypatch.setattr(owner, "build_runtime_manifest", lambda config: {"runtime_fingerprint": "drift"})
    arguments = dict(spec=case["spec"], package_dir=tmp_path, expected_package_identity="c" * 64,
        arm_id=case["arm"]["arm_id"], mode="reference", recipe_path=case["recipe_path"],
        initial_state_path=case["initial_path"], input_root=case["input_root"],
        account="local", run_reference="local/synthetic", trajectory_id="synthetic-trajectory")
    with pytest.raises(ValueError, match="Recovery runtime differs"):
        owner.run_screen_resume_preflight(**arguments, resume_output=partial, output=tmp_path / "drift")
    monkeypatch.setattr(owner, "build_runtime_manifest", runtime_builder)
    original_device = owner._execution_device
    def bad_device(spec):
        raise RuntimeError("unqualified device")
    monkeypatch.setattr(owner, "_execution_device", bad_device)
    with pytest.raises(RuntimeError, match="unqualified device"):
        owner.run_screen_resume_preflight(**arguments, resume_output=partial, output=tmp_path / "bad_device")
    monkeypatch.setattr(owner, "_execution_device", original_device)
    resumed_result = owner.run_screen_arm(
        spec=case["spec"], package_dir=tmp_path, expected_package_identity="c" * 64,
        arm_id=case["arm"]["arm_id"], mode="reference", recipe_path=case["recipe_path"],
        initial_state_path=case["initial_path"], input_root=case["input_root"], output=partial,
        account="local", run_reference="local/synthetic", trajectory_id="synthetic-trajectory",
        preflight_dir=preflight,
    )
    assert resumed_result["status"] == "MECHANICALLY_VERIFIED"
    continuous_state = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=True)["model"]
    resumed_state = torch.load(partial / "last_checkpoint.pt", map_location="cpu", weights_only=True)["model"]
    assert all(torch.equal(continuous_state[key], resumed_state[key]) for key in continuous_state)
    continuous_predictions = torch.load(output / "development_predictions.pt", map_location="cpu", weights_only=True)
    resumed_predictions = torch.load(partial / "development_predictions.pt", map_location="cpu", weights_only=True)
    assert torch.equal(continuous_predictions["prediction_eV"], resumed_predictions["prediction_eV"])


def test_recipe_config_is_strict(synthetic_case):
    case = synthetic_case
    bad = dict(case["recipe"])
    bad["weight_decay"] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        owner.validate_screen_recipe(case["spec"], case["arm"]["arm_id"], bad)


@pytest.mark.parametrize("platform", ("ims", "scnet", "unsupported"))
def test_shared_runtime_rejects_unsupported_platform(platform):
    spec = types.SimpleNamespace(to_dict=lambda: {"platform": {"name": platform}})
    with pytest.raises(ValueError, match="no execution policy"):
        owner._execution_device(spec)


def test_local_runtime_stays_on_cpu_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    spec = types.SimpleNamespace(to_dict=lambda: {"platform": {"name": "local"}})
    assert owner._execution_device(spec) == torch.device("cpu")


@pytest.mark.parametrize("accelerator,accepted", (
    (None, False), ({"name": "Tesla T4", "device_count_visible": 0}, False),
    ({"name": "Tesla T4", "device_count_visible": 2}, False),
    ({"name": "Tesla P100", "device_count_visible": 1}, False),
    ({"name": "Tesla T4", "device_count_visible": 1}, True)))
def test_resume_runtime_binds_one_visible_t4(accelerator, accepted):
    runtime = {"accelerator": accelerator}
    if accepted:
        assert owner._qualified_cuda_devices(runtime, "kaggle") == 1
    else:
        with pytest.raises(ValueError, match="one visible T4"):
            owner._qualified_cuda_devices(runtime, "kaggle")


@pytest.mark.parametrize("available,devices,name,accepted", (
    (False, 0, "", False), (True, 2, "Tesla T4", False),
    (True, 1, "Tesla P100", False), (True, 1, "Tesla T4", True)))
def test_kaggle_runtime_requires_one_visible_t4(monkeypatch, available, devices, name, accepted):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)
    spec = types.SimpleNamespace(to_dict=lambda: {"platform": {"name": "kaggle"}})
    if accepted:
        assert owner._execution_device(spec) == torch.device("cuda:0")
    else:
        with pytest.raises(RuntimeError, match="Shared Kaggle graph screen"):
            owner._execution_device(spec)


def test_real_fixed_manifest_metadata_allows_geometry_inventory_without_loading():
    from molgap import pcqm_graph_inputs

    path = Path(__file__).resolve().parents[1] / "platforms/_records/ims/pcqm_fixed_datasets_v1/ogb-train-100k.manifest.json"
    manifest = json.loads(path.read_bytes())
    records = pcqm_graph_inputs._validate_manifest_shape(manifest)
    assert len(records) == 3
    assert {item["role"] for item in records} == {"train", "development"}
    assert any(item["file"].startswith("store/topology/") for item in records)
    # The manifest inventories geometry, but the pure graph owner only returns
    # topology records and never attempts to resolve a geometry path.
    assert "geometry" in manifest["assets"]
