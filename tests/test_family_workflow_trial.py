import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from molgap import execution_comparison, family_workflow_trial as trial
from molgap.experiment_execution import ExecutionExtension
from molgap.experiment_family_workflow import ROLES
from molgap.experiment_spec import ExperimentSpec


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments" / "family_workflow_trial_100k"
BOOTSTRAP_PATH = ROOT / "platforms" / "kaggle" / "run_experiment.py"
RETRIEVER_PATH = ROOT / "platforms" / "kaggle" / "retrieve_family_outputs.py"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("family_trial_bootstrap_test", BOOTSTRAP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_retriever():
    spec = importlib.util.spec_from_file_location("family_trial_retriever_test", RETRIEVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family_files_with_profile(tmp_path, monkeypatch, *, protocol, missing_role,
                               trace_context=False, trace_context_path=None):
    retriever = _load_retriever()
    monkeypatch.setattr(retriever, "artifact_adapter",
        lambda _adapter, _family: SimpleNamespace(protocol=protocol))
    monkeypatch.setattr(retriever, "_check_context", lambda _observed, _expected: None)
    roles = ROLES - {missing_role}
    manifest = {
        "format": "molgap-family-output-v1",
        "context": {},
        "adapter": "test-profile",
        "artifacts": {
            role: {"path": f"artifacts/{role}.bin", "sha256": "a" * 64}
            for role in roles
        },
    }
    if trace_context:
        manifest["artifacts"]["trace_context"] = {
            "path": trace_context_path or manifest["artifacts"]["trace"]["path"] + ".context.json",
            "sha256": "b" * 64,
        }
    path = tmp_path / "output_manifest.json"
    payload = json.dumps(manifest, sort_keys=True).encode()
    path.write_bytes(payload)
    context = SimpleNamespace(family_name="gptrans_t", family_version="2")
    return retriever._family_files(context, "runs/session/output_manifest.json", path,
                                   hashlib.sha256(payload).hexdigest())


def _frozen_inputs():
    spec = ExperimentSpec(json.loads((FROZEN / "experiment_spec.json").read_text()))
    plan = json.loads((FROZEN / "acceptance_plan.json").read_text())
    return spec, plan


def test_actual_frozen_trial_declaration_passes_real_acceptance_check():
    spec, plan = _frozen_inputs()

    result = trial.check_plan(spec, ROOT, plan)

    assert result["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    assert result["comparison_scope"] == "EXECUTION_ONLY"
    assert result["scientific_acceptance"] == "NOT_APPLICABLE"
    assert [(arm["arm_id"], arm["status"]) for arm in result["arms"]] == [
        ("k1", "AVAILABLE"), ("gptrans", "AVAILABLE")
    ]


def test_frozen_authority_budget_tampering_is_rejected(monkeypatch):
    spec, plan = _frozen_inputs()
    read_json = trial._json

    def tampered_json(path):
        value = read_json(path)
        if Path(path).name == "authority.json":
            value["wall_ceiling_seconds"] = 1801
        return value

    monkeypatch.setattr(trial, "_json", tampered_json)
    with pytest.raises(ValueError, match="authority or budget"):
        trial.check_plan(spec, ROOT, plan)


def test_development_role_addition_is_rejected():
    spec, plan = _frozen_inputs()
    declaration = copy.deepcopy(spec.to_dict())
    added_role = copy.deepcopy(declaration["arms"][0]["data"]["roles"][0])
    added_role["role"] = "development"
    declaration["arms"][0]["data"]["roles"].append(added_role)

    class SpecView:
        identity = spec.identity

        @staticmethod
        def to_dict():
            return declaration

    with pytest.raises(ValueError, match="family or role"):
        trial.check_plan(SpecView(), ROOT, plan)


def test_historical_pair_execution_dispatch_keeps_default_extension():
    from test_experiment_workflow import _candidate_pair_spec

    extension = execution_comparison.execution_extension(_candidate_pair_spec(), ROOT)

    assert extension == ExecutionExtension()
    assert extension.wall_ceiling_seconds is None
    assert extension.configuration is None


def test_kaggle_bootstrap_uses_pinned_trial_profile_and_allocation_cutoff():
    source = BOOTSTRAP_PATH.read_text()

    assert 'workflow_trial = declaration["experiment_id"] == "family-workflow-trial-100k"' in source
    assert 'authority.get("wall_ceiling_seconds") != 1800' in source
    assert 'profile="family-shared" if k1_comparison or modern2d_formal or workflow_trial else' in source
    assert '1800 - (time.perf_counter() - allocation_started) - 120' in source
    tree = ast.parse(source)
    workload = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_workload")
    trial_block = next(node for node in ast.walk(workload)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "workflow_trial"
        and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            and child.func.id == "_run_trial_pair" for child in ast.walk(node)))
    runner = next(node for node in ast.walk(trial_block)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_run_trial_pair")
    timeout = next(keyword.value for keyword in runner.keywords if keyword.arg == "timeout")
    assert "allocation_started" in {node.id for node in ast.walk(timeout) if isinstance(node, ast.Name)}
    assert {1, 1800, 15} <= {node.value for node in ast.walk(timeout) if isinstance(node, ast.Constant)}
    pair_runtime = (ROOT / "src" / "molgap" / "kaggle_pair_runtime.py").read_text()
    assert 'hard_stop_reserve = 45 if diagnostic and diagnostic.get("format") == "molgap-family-workflow-trial-plan-v1" else 60' in pair_runtime


def test_trial_process_group_normal_exit_starts_new_session(monkeypatch):
    bootstrap = _load_bootstrap()
    events = []

    class Process:
        pid = 4812

        def wait(self, timeout):
            events.append(("wait", timeout))
            return 0

    process = Process()
    popen_calls = []

    def fake_popen(command, *, cwd, env, start_new_session):
        popen_calls.append((command, cwd, env, start_new_session))
        return process

    def fake_killpg(pid, sig):
        events.append(("killpg", pid, sig))

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bootstrap.os, "killpg", fake_killpg, raising=False)

    bootstrap._run_trial_pair(["worker"], cwd="/synthetic", env={}, timeout=90)

    assert popen_calls == [(["worker"], "/synthetic", {}, True)]
    assert events == [("wait", 90)]


def test_trial_timeout_kills_group_even_if_leader_exits_before_sigkill(monkeypatch):
    bootstrap = _load_bootstrap()
    events = []

    class Process:
        pid = 4813

        def __init__(self):
            self.wait_count = 0

        def wait(self, timeout):
            self.wait_count += 1
            events.append(("wait", timeout))
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 0  # The leader exits after TERM; the rest of the group may remain.

    process = Process()

    def fake_popen(command, *, cwd, env, start_new_session):
        assert start_new_session is True
        return process

    def fake_killpg(pid, sig):
        events.append(("killpg", pid, sig))

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bootstrap.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(bootstrap.signal, "SIGTERM", getattr(signal, "SIGTERM", 15), raising=False)
    monkeypatch.setattr(bootstrap.signal, "SIGKILL", getattr(signal, "SIGKILL", 9), raising=False)

    with pytest.raises(subprocess.TimeoutExpired):
        bootstrap._run_trial_pair(["worker"], cwd="/synthetic", env={}, timeout=1)

    assert events == [
        ("wait", 1),
        ("killpg", 4813, bootstrap.signal.SIGTERM),
        ("wait", 5),
        ("killpg", 4813, bootstrap.signal.SIGKILL),
        ("wait", 5),
    ]


def test_family_session_final_only_manifest_can_omit_predictions(tmp_path, monkeypatch):
    manifest, required = _family_files_with_profile(tmp_path, monkeypatch,
        protocol="family-session", missing_role="predictions")

    assert set(manifest["artifacts"]) == ROLES - {"predictions"}
    assert "runs/session/artifacts/predictions.bin" not in required
    assert "runs/session/artifacts/selected_model.bin" in required
    assert "runs/session/output_manifest.json" in required


def test_family_session_trace_context_is_pinned_to_companion_path(tmp_path, monkeypatch):
    manifest, required = _family_files_with_profile(tmp_path, monkeypatch,
        protocol="family-session", missing_role="predictions", trace_context=True)

    assert manifest["artifacts"]["trace_context"]["path"] == "artifacts/trace.bin.context.json"
    assert required["runs/session/artifacts/trace.bin.context.json"] == (
        "artifacts/trace.bin.context.json", "b" * 64
    )


def test_family_session_trace_context_wrong_path_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Trace companion is not bound"):
        _family_files_with_profile(tmp_path, monkeypatch, protocol="family-session",
            missing_role="predictions", trace_context=True,
            trace_context_path="artifacts/unrelated.json")


@pytest.mark.parametrize(("protocol", "missing_role"), [
    ("family-session", "trace"),
    ("screen", "predictions"),
])
def test_final_only_manifest_rejects_missing_other_role_or_legacy_profile(
        tmp_path, monkeypatch, protocol, missing_role):
    with pytest.raises(ValueError, match="Incomplete retention manifest"):
        _family_files_with_profile(tmp_path, monkeypatch, protocol=protocol,
                                   missing_role=missing_role)
