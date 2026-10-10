"""Synthetic file-only loader custody; no molecular decode or model execution."""
import hashlib
import json
import struct
from unittest.mock import Mock

import pytest

from molgap import saved_diagnostic_closure as closure
from molgap.research_memory.schemas import validate_cost_event, validate_role_event
from molgap.training_reproducibility import atomic_json, sha256_file


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def publish(folder, inputs, result):
    atomic_json(folder / "inputs.json", inputs)
    result.update(inputs_sha256=sha256_file(folder / "inputs.json"),
                  prospective_sha256=sha256_file(folder / "rml/trajectory.json"))
    atomic_json(folder / "results/result.json", result)
    atomic_json(folder / "results/completion.json", dict(
        complete=True, inputs_sha256=result["inputs_sha256"],
        files={"result.json": sha256_file(folder / "results/result.json")},
    ))


@pytest.fixture
def custody(tmp_path, monkeypatch):
    folder = tmp_path / "experiments/synthetic-loader"
    source = b"synthetic immutable source only\n"
    commit = "a" * 40
    atomic_json(folder / "source_snapshot/metadata.json", {"synthetic": True})
    # Bytes are not a pickle and must only be hashed, never deserialized.
    probe = tmp_path / "retained-probe.bin"
    probe.write_bytes(b"synthetic train probe; not molecular graphs")
    inputs = dict(kind="loader_metadata", ceiling_seconds=60,
                  source_idx=list(range(4095, -1, -1)),
                  train_probe=dict(path=str(probe), sha256=sha256_file(probe)),
                  snapshot_files={"metadata.json": sha256_file(folder / "source_snapshot/metadata.json")},
                  executed_source_files={"src/synthetic.txt": hashlib.sha256(source).hexdigest()})
    atomic_json(folder / "rml/trajectory.json", dict(
        trajectory_id="TB-loader-fixture", actions=[dict(
            action_id="A001", run_ids=["run-loader"], source_commit=commit,
        )],
    ))
    for name in ("protocol.md", "terminal_decision.md", "attribution.md"):
        (folder / name).write_text("Synthetic NO_TRAIN metadata only.\n", encoding="ascii")
    result = dict(format="molgap-k1-saved-analysis-v1", status="complete",
                  kind="loader_metadata", checks={"collation_metadata": True},
                  training_executed=False, inference_executed=False,
                  optimizer_created=False, gradients_computed=False,
                  runtime=dict(device="cpu", cpu_threads=4,
                               cuda_initialized=False, cuda_visible_devices="",
                               cuda_available=False, cuda_device_count=0),
                  wall_seconds=12, process_cpu_seconds=2,
                  total_process_cpu_seconds=8,
                  observed_source_idx=inputs["source_idx"].copy())
    publish(folder, inputs, result)
    git = Mock(return_value=source)
    monkeypatch.setattr(closure.subprocess, "check_output", git)

    def finalize(root, rml, terminal):
        package = read(root / terminal)
        for cost in package["costs"]:
            validate_cost_event(cost)
        for role in package["roles"]:
            validate_role_event(role)
        return {"status": "synthetic-finalizer-delegation"}

    finalizer = Mock(side_effect=finalize)
    monkeypatch.setattr(closure, "finalize", finalizer)
    return tmp_path, folder, inputs, result, probe, source, git, finalizer


@pytest.mark.parametrize("hidden", ["", "-1"])
def test_loader_roles_source_custody_and_total_cpu(custody, hidden):
    root, folder, inputs, result, probe, source, git, finalizer = custody
    result["runtime"]["cuda_visible_devices"] = hidden
    publish(folder, inputs, result)
    result_before = (folder / "results/result.json").read_bytes()
    trajectory_before = (folder / "rml/trajectory.json").read_bytes()
    assert closure.close_saved_diagnostic(root, folder)["status"] == "synthetic-finalizer-delegation"
    acceptance = read(folder / "acceptance.json")
    terminal = read(folder / "terminal.json")
    roles = acceptance["roles"]
    assert len(roles) == 1
    assert roles[0]["role_name"] == "train"
    assert roles[0]["access_kind"] == "labels_read"
    assert roles[0]["selection_used"] is False
    digest = hashlib.sha256(b"".join(struct.pack("<q", i) for i in inputs["source_idx"])).hexdigest()
    assert roles[0]["row_manifest_hash"] == digest
    assert acceptance["role_use"]["train"] == "consumed"
    for role in ("internal_development", "official_validation", "test_dev", "test_challenge", "common", "ood"):
        assert acceptance["role_use"][role] == "untouched"
    measurement = acceptance["costs"][0]["measurement"]
    assert measurement["cpu_hours"] == {"value": 8 / 3600, "status": "measured"}
    assert measurement["wall_hours"] == {"value": 12 / 3600, "status": "measured"}
    assert measurement["device_hours"] == {"value": None, "status": "not_applicable"}
    assert terminal["roles"] == roles
    assert terminal["costs"] == acceptance["costs"]
    assert terminal["evidence"]["role_use"] == acceptance["role_use"]
    assert terminal["decision"]["outcome"] == "NO_TRAIN"
    assert (folder / "executed_source/src/synthetic.txt").read_bytes() == source
    assert terminal["artifact_hashes"]["experiments/synthetic-loader/executed_source/src/synthetic.txt"] == hashlib.sha256(source).hexdigest()
    assert (folder / "results/result.json").read_bytes() == result_before
    assert read(folder / "results/result.json")["process_cpu_seconds"] == 2
    assert (folder / "rml/trajectory.json").read_bytes() == trajectory_before
    git.assert_called_once_with(["git", "show", "a" * 40 + ":src/synthetic.txt"], cwd=root.resolve())
    finalizer.assert_called_once_with(root.resolve(), "experiments/synthetic-loader/rml", "experiments/synthetic-loader/terminal.json")


@pytest.mark.parametrize("total", ["absent", None, True, False, "8", -1, 1, float("nan"), float("inf")])
def test_parent_only_or_invalid_total_cpu_is_measurement_missing(custody, total):
    root, folder, inputs, result, *_ = custody
    if total == "absent":
        result.pop("total_process_cpu_seconds")
    else:
        result["total_process_cpu_seconds"] = total
    publish(folder, inputs, result)
    closure.close_saved_diagnostic(root, folder)
    cost = read(folder / "acceptance.json")["costs"][0]
    assert cost["measurement"]["cpu_hours"] == {"value": None, "status": "measurement_missing"}
    assert read(folder / "results/result.json")["process_cpu_seconds"] == 2


@pytest.mark.parametrize("include_total", [False, True])
def test_partial_parent_plus_worker_sample_never_becomes_total_cpu(custody, include_total):
    root, folder, inputs, result, *_ = custody
    result["observed_parent_plus_worker_cpu_seconds"] = 8
    if include_total:
        result["total_process_cpu_seconds"] = None
    else:
        result.pop("total_process_cpu_seconds")
    publish(folder, inputs, result)
    before = (folder / "results/result.json").read_bytes()
    closure.close_saved_diagnostic(root, folder)
    cpu = read(folder / "acceptance.json")["costs"][0]["measurement"]["cpu_hours"]
    assert cpu == {"value": None, "status": "measurement_missing"}
    assert (folder / "results/result.json").read_bytes() == before
    retained = read(folder / "results/result.json")
    assert retained["process_cpu_seconds"] == 2
    assert retained["observed_parent_plus_worker_cpu_seconds"] == 8


@pytest.mark.parametrize("mutation", [
    "out_of_range", "negative", "duplicate", "bool", "float", "short", "missing",
    "observed_reordered", "observed_altered", "observed_bool", "observed_float",
    "probe_bytes", "probe_hash", "probe_missing", "probe_binding_missing", "probe_path",
    "training_executed", "inference_executed", "optimizer_created", "gradients_computed",
    "model_flag_zero", "model_flag_missing", "false_checks", "numeric_checks", "no_checks",
    "cuda", "cuda_initialized", "cuda_initialized_zero", "cuda_initialized_missing",
    "cuda_visible_devices", "cuda_visible_devices_none", "cuda_visible_devices_missing",
    "cuda_available_true", "cuda_available_zero", "cuda_available_missing",
    "cuda_count_bool", "cuda_count_float", "cuda_count_nonzero", "cuda_count_negative", "cuda_count_missing",
    "threads", "kind", "wall_limit", "parent_cpu_bool",
])
def test_loader_refuses_invalid_evidence_before_closure_writes(custody, mutation):
    root, folder, inputs, result, probe, source, git, finalizer = custody
    if mutation == "out_of_range":
        inputs["source_idx"][0] = 500000
    elif mutation == "negative":
        inputs["source_idx"][0] = -1
    elif mutation == "duplicate":
        inputs["source_idx"][0] = inputs["source_idx"][1]
    elif mutation in ("bool", "float"):
        inputs["source_idx"][-1] = False if mutation == "bool" else 0.0
    elif mutation == "short":
        inputs["source_idx"].pop()
    elif mutation == "missing":
        inputs.pop("source_idx")
    elif mutation == "observed_reordered":
        result["observed_source_idx"].reverse()
    elif mutation == "observed_altered":
        result["observed_source_idx"][0] = 499999
    elif mutation in ("observed_bool", "observed_float"):
        result["observed_source_idx"][-1] = False if mutation == "observed_bool" else 0.0
    elif mutation == "probe_bytes":
        probe.write_bytes(b"changed synthetic bytes")
    elif mutation == "probe_hash":
        inputs["train_probe"]["sha256"] = "0" * 64
    elif mutation == "probe_missing":
        inputs["train_probe"]["path"] = str(root / "missing.bin")
    elif mutation == "probe_binding_missing":
        inputs.pop("train_probe")
    elif mutation == "probe_path":
        inputs["train_probe"]["path"] = ""
    elif mutation in ("training_executed", "inference_executed", "optimizer_created", "gradients_computed"):
        result[mutation] = True
    elif mutation == "model_flag_zero":
        result["training_executed"] = 0
    elif mutation == "model_flag_missing":
        result.pop("training_executed")
    elif mutation in ("false_checks", "numeric_checks", "no_checks"):
        result["checks"] = {} if mutation == "no_checks" else {"collation_metadata": mutation == "numeric_checks" and 1}
    elif mutation == "cuda":
        result["runtime"]["device"] = "cuda"
    elif mutation == "cuda_initialized":
        result["runtime"]["cuda_initialized"] = True
    elif mutation == "cuda_initialized_zero":
        result["runtime"]["cuda_initialized"] = 0
    elif mutation == "cuda_initialized_missing":
        result["runtime"].pop("cuda_initialized")
    elif mutation == "cuda_visible_devices":
        result["runtime"]["cuda_visible_devices"] = "0"
    elif mutation == "cuda_visible_devices_none":
        result["runtime"]["cuda_visible_devices"] = None
    elif mutation == "cuda_visible_devices_missing":
        result["runtime"].pop("cuda_visible_devices")
    elif mutation == "cuda_available_true":
        result["runtime"]["cuda_available"] = True
    elif mutation == "cuda_available_zero":
        result["runtime"]["cuda_available"] = 0
    elif mutation == "cuda_available_missing":
        result["runtime"].pop("cuda_available")
    elif mutation == "cuda_count_bool":
        result["runtime"]["cuda_device_count"] = False
    elif mutation == "cuda_count_float":
        result["runtime"]["cuda_device_count"] = 0.0
    elif mutation == "cuda_count_nonzero":
        result["runtime"]["cuda_device_count"] = 1
    elif mutation == "cuda_count_negative":
        result["runtime"]["cuda_device_count"] = -1
    elif mutation == "cuda_count_missing":
        result["runtime"].pop("cuda_device_count")
    elif mutation == "threads":
        result["runtime"]["cpu_threads"] = 8
    elif mutation == "kind":
        result["kind"] = "trace_pair"
    elif mutation == "wall_limit":
        result["wall_seconds"] = 61
    elif mutation == "parent_cpu_bool":
        result["process_cpu_seconds"] = True
    publish(folder, inputs, result)
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        closure.close_saved_diagnostic(root, folder)
    assert {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before
    git.assert_not_called()
    finalizer.assert_not_called()


@pytest.mark.parametrize("kind", ["trace_pair", "bn_rows"])
def test_old_nonloader_roles_and_parent_cost_unchanged(custody, kind):
    root, folder, inputs, result, *_ = custody
    inputs["kind"] = result["kind"] = kind
    inputs.pop("train_probe")
    inputs.pop("source_idx")
    result["runtime"].pop("cuda_initialized")
    result["runtime"].pop("cuda_visible_devices")
    result["runtime"].pop("cuda_available")
    result["runtime"].pop("cuda_device_count")
    inputs["development_bounds"] = [500000, 500003]
    result["observed_source_idx"] = list(range(500000, 500003)) if kind == "bn_rows" else []
    publish(folder, inputs, result)
    closure.close_saved_diagnostic(root, folder)
    acceptance = read(folder / "acceptance.json")
    assert acceptance["costs"][0]["measurement"]["cpu_hours"] == {"value": 2 / 3600, "status": "measured"}
    assert "train" not in acceptance["role_use"]
    roles = acceptance["roles"]
    if kind == "bn_rows":
        assert [r["access_kind"] for r in roles] == ["labels_read", "metric_computed"]
        assert all(r["role_name"] == "internal_development" for r in roles)
        assert acceptance["role_use"]["internal_development"] == "consumed"
    else:
        assert roles == []


def test_old_nonloader_cannot_gain_train_access(custody):
    root, folder, inputs, result, *_ = custody
    inputs["kind"] = result["kind"] = "trace_pair"
    publish(folder, inputs, result)
    with pytest.raises(ValueError, match="Observed membership differs"):
        closure.close_saved_diagnostic(root, folder)
    assert not (folder / "acceptance.json").exists()
