"""CPU-only contract checks for the formal Modern2D release boundary.

These tests deliberately stop at metadata, process orchestration and stdlib
bootstrap seams.  They do not construct a model, graph, CUDA context, remote
client, or RML record.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tarfile

import pytest


FORMAL_PLAN = "molgap-modern2d-formal-reference-plan-v1"


class _FormalSpec:
    identity = "a" * 64

    def to_dict(self):
        return {
            "experiment_id": "modern2d-formal-screen",
            "logical_run_id": "modern2d-formal-screen-s42-v1",
            "arms": [
                {"arm_id": "ppgt", "family": {"name": "ppgt_small", "version": "3"},
                 "addons": []},
                {"arm_id": "esa", "family": {"name": "esa_small", "version": "4"},
                 "addons": []},
            ],
            "platform": {
                "name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2,
                "cpu_cores": 4,
            },
        }


def _formal_extension(*, wall=1000, reserve=600):
    from molgap.experiment_execution import ExecutionExtension, ExecutionPhase

    return ExecutionExtension(
        phases=(ExecutionPhase("preflight"), ExecutionPhase("train")),
        configuration={"format": FORMAL_PLAN, "checkpoint_reserve_seconds": reserve},
        wall_ceiling_seconds=wall,
    )


def _pair_fixture(tmp_path, monkeypatch, *, process_factory):
    """Install a small, non-model pair runtime harness and run it."""
    from molgap import kaggle_pair_runtime as runtime
    import molgap.experiment_retention as retention
    import molgap.family_pair_release as pair_release
    import molgap.gptrans_infrastructure_comparison as gptrans_comparison
    import molgap.k1_infrastructure_comparison as k1_comparison

    source_root = tmp_path / "source"
    package_dir = tmp_path / "package"
    input_root = tmp_path / "input"
    source_root.mkdir()
    package_dir.mkdir()
    input_root.mkdir()
    (package_dir / "experiment_spec.json").write_text("{}", encoding="utf-8")
    launch_path = tmp_path / "experiment_launch.json"
    config = {
        "expected_package_identity": "b" * 64,
        "expected_source_archive_sha256": "c" * 64,
        "jobs": [
            {"arm_id": "ppgt", "device": 0},
            {"arm_id": "esa", "device": 1},
        ],
    }
    launch_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "output"
    spec = _FormalSpec()

    class _SpecLoader:
        @staticmethod
        def from_json(path):
            return spec

    class _Ledger:
        instances = []

        def __init__(self, **kwargs):
            self.started = 0.0
            self.prior_segments = []
            self.writes = []
            type(self).instances.append(self)

        def snapshot(self, state):
            return {"prior_unobserved_intervals": []}

        def write(self, output, status, **kwargs):
            self.writes.append(status)

    monkeypatch.setattr(runtime, "ExperimentSpec", _SpecLoader)
    monkeypatch.setattr(runtime, "_json", lambda path: config)
    monkeypatch.setattr(runtime, "validate_execution_plan", lambda spec, jobs: jobs)
    monkeypatch.setattr(runtime, "execution_extension", lambda spec, root: _formal_extension())
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: "Tesla T4\nTesla T4\n")
    monkeypatch.setattr(runtime.subprocess, "Popen", process_factory)
    monkeypatch.setattr(runtime, "AllocationLedger", _Ledger)
    monkeypatch.setattr(runtime, "forward_training_logs", lambda *args, **kwargs: args[1])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: 500.0)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(retention, "seal_execution_retention", lambda *args, **kwargs: None)
    monkeypatch.setattr(k1_comparison, "is_comparison", lambda spec: False)
    monkeypatch.setattr(gptrans_comparison, "is_comparison", lambda spec: False)
    monkeypatch.setattr(pair_release, "bounded_cycle_pair", lambda spec: False)

    manifest = {
        "spec_identity": spec.identity,
        "package_identity": config["expected_package_identity"],
        "archive_sha256": config["expected_source_archive_sha256"],
        "source_commit": "d" * 40,
    }
    report = runtime.run_two_phase_pair(
        source_root=source_root,
        package_dir=package_dir,
        input_root=input_root,
        launch_path=launch_path,
        output=output,
        manifest=manifest,
    )
    return report, output, _Ledger.instances[-1], config


class _MockProcess:
    def __init__(self, command, *, preflight_failure=None, calls=None, **kwargs):
        self.command = list(command)
        self.calls = calls if calls is not None else []
        self.phase = self.command[self.command.index("--phase") + 1]
        self.arm_id = self.command[self.command.index("--arm") + 1]
        self.output = Path(self.command[self.command.index("--output") + 1])
        self.output.mkdir(parents=True, exist_ok=True)
        self.returncode = (preflight_failure[1] if self.phase == "preflight" and
                           preflight_failure is not None and
                           self.arm_id == preflight_failure[0] else
                           0 if self.phase == "preflight" else None)
        self.terminated = False
        self.env = kwargs.get("env", {})
        self.calls.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self.phase == "train":
            (self.output / "last_checkpoint.pt").write_bytes(b"mock-checkpoint")
            (self.output / "pause_receipt.json").write_text(
                json.dumps({"status": "PAUSED", "reason": "STOP_FOR_COST"}),
                encoding="utf-8",
            )
        self.returncode = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_formal_versions_are_additive_and_profile_name_is_pinned():
    from molgap import modern2d_training as formal
    from molgap.experiment_execution import (
        MODERN2D_FORMAL_INTERFACES,
        MODERN2D_FORMAL_PROFILE,
        MODERN2D_PREVIEW_INTERFACES,
        TRAINING_ADAPTERS,
        validate_training_registry,
    )

    validate_training_registry()
    assert formal.INTERFACES == (("ppgt_small", "3"), ("esa_small", "4"))
    assert formal.MODEL_INTERFACES == (("ppgt_small", "2"), ("esa_small", "3"))
    assert formal.FORMAL_STEPS == 46860
    assert formal.PROFILE == MODERN2D_FORMAL_PROFILE == "modern2d-formal-session-v1"
    for formal_family, preview_family in zip(formal.INTERFACES, formal.MODEL_INTERFACES):
        formal_adapter = TRAINING_ADAPTERS[formal_family]
        preview_adapter = TRAINING_ADAPTERS[preview_family]
        assert formal_adapter.formal_training and not formal_adapter.integration_preview
        assert preview_adapter.integration_preview and not preview_adapter.formal_training
        assert MODERN2D_FORMAL_INTERFACES[formal_family][0] == preview_adapter.model_factory
        assert MODERN2D_PREVIEW_INTERFACES[preview_family][0] == formal_adapter.model_factory


def test_formal_workflow_dispatches_explicit_budget_extension(monkeypatch, tmp_path):
    from molgap import execution_comparison, experiment_family_workflow as workflow
    from molgap import modern2d_training_release as release

    spec = _FormalSpec()
    sentinel = object()
    (tmp_path / "formal").mkdir()
    authority = tmp_path / "formal" / "authority.json"
    authority.write_text(json.dumps({"purpose": "FORMAL_100K_TRAINING"}), encoding="utf-8")
    plan = {"format": FORMAL_PLAN, "authority": {
        "path": "formal/authority.json",
        "sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
    }}
    (tmp_path / "formal" / "acceptance_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(release, "plan_path", lambda value: "formal/acceptance_plan.json")
    monkeypatch.setattr(release, "check_plan", lambda *args, **kwargs: None)
    extension = release.execution_extension(spec=spec, source_root=tmp_path)
    assert extension.wall_ceiling_seconds == 16200
    assert extension.configuration == {"format": FORMAL_PLAN, "checkpoint_reserve_seconds": 600}

    monkeypatch.setattr(release, "execution_extension", lambda spec, root: sentinel)
    assert execution_comparison.execution_extension(spec, tmp_path) is sentinel
    monkeypatch.setattr(release, "check_plan", lambda *args, **kwargs: sentinel)
    assert workflow.check_acceptance_plan(spec, tmp_path, {"format": FORMAL_PLAN}) is sentinel


def test_old_preview_is_still_blocked_before_payload(monkeypatch, tmp_path):
    from molgap import family_training_data, family_training_runner as runner

    preview = SimpleNamespace(integration_preview=True, cpu_synthetic_only=False,
                              formal_interface=False)
    monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, preview))
    monkeypatch.setattr(
        family_training_data,
        "load_training_data",
        lambda *args, **kwargs: pytest.fail("preview reached payload loading"),
    )
    with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
        runner._bootstrap(
            spec=None,
            package_dir=tmp_path,
            expected_package_identity="a" * 64,
            arm_id="ppgt",
            mode="reference",
            recipe_path=tmp_path / "preview.json",
            initial_state_path=tmp_path / "initial.pt",
            input_root=tmp_path,
            account="fixture",
            run_reference="fixture/run",
            trajectory_id="fixture",
        )


@pytest.mark.parametrize("value", [None, "", "NaN", "Infinity", "0", "-1"])
def test_formal_pause_deadline_rejects_nonfinite_or_nonpositive_values(monkeypatch, value):
    from molgap import family_training_runner as runner

    if value is None:
        monkeypatch.delenv("MOLGAP_ALLOCATION_PAUSE_AT", raising=False)
    else:
        monkeypatch.setenv("MOLGAP_ALLOCATION_PAUSE_AT", value)
    with pytest.raises(ValueError, match="finite"):
        runner._formal_pause_deadline()


def test_formal_deadline_stops_both_workers_for_cost_and_retains_state(tmp_path, monkeypatch):
    calls = []

    def spawn(command, **kwargs):
        return _MockProcess(command, calls=calls, **kwargs)

    report, output, ledger, _ = _pair_fixture(tmp_path, monkeypatch, process_factory=spawn)

    assert report["status"] == "STOP_FOR_COST"
    assert report["scientific_acceptance"] == "INCONCLUSIVE"
    assert report["exit_reason"] == "STOP_FOR_COST"
    assert [item.phase for item in calls] == ["preflight", "preflight", "train", "train"]
    assert {item.env["CUDA_VISIBLE_DEVICES"] for item in calls} == {"0", "1"}
    assert _FormalSpec().to_dict()["platform"]["cpu_cores"] == 4
    assert all(item.env["OMP_NUM_THREADS"] == "1" for item in calls)
    assert all(item.env["MKL_NUM_THREADS"] == "1" for item in calls)
    assert {float(item.env["MOLGAP_ALLOCATION_PAUSE_AT"]) for item in calls} == {400.0}
    assert all(item.terminated for item in calls if item.phase == "train")
    assert all(state["terminal_status"] == "interrupted"
               for state in report["arms"].values())
    assert all((output / arm / "pause_receipt.json").is_file()
               and (output / arm / "last_checkpoint.pt").is_file()
               for arm in ("ppgt", "esa"))
    assert ledger.writes[-1] == "complete"
    assert "interrupted" not in ledger.writes
    assert "failed" not in ledger.writes


def test_preflight_failure_blocks_formal_training_for_all_arms(tmp_path, monkeypatch):
    calls = []

    def spawn(command, **kwargs):
        return _MockProcess(command, preflight_failure=("esa", 7), calls=calls, **kwargs)

    with pytest.raises(RuntimeError, match="Arm failed during preflight"):
        _pair_fixture(tmp_path, monkeypatch, process_factory=spawn)

    assert [item.phase for item in calls] == ["preflight", "preflight"]
    state = json.loads((tmp_path / "output" / "pair_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert all(not item["training_started"] for item in state["arms"].values())
    assert state["arms"]["esa"]["terminal_status"] == "failed"
    assert state["arms"]["ppgt"]["terminal_status"] == "cancelled"


def _formal_context():
    from molgap.data.indexed_inputs import AcceptedIndexedInputPermit
    from molgap.experiment_family_workflow import RunContext
    from molgap.modern2d_training import Modern2DFormalTrainingContext

    run = RunContext(
        "formal-question", "formal-run", "ppgt", "a" * 64, "b" * 64,
        "ppgt_small", "3", "c" * 40, "d" * 64, "e" * 64, "f" * 64,
        "kaggle", "fixture", "fixture/formal", None,
    )
    train = AcceptedIndexedInputPermit("train_100k", "1" * 64, "2" * 64,
        "3" * 64, "4" * 64, "5" * 64, 1024)
    development = AcceptedIndexedInputPermit("common_dev", "1" * 64, "2" * 64,
        "3" * 64, "6" * 64, "7" * 64, 1024)
    return Modern2DFormalTrainingContext(
        run_context=run,
        recipe_identity="8" * 64,
        authority_sha256="9" * 64,
        train_permit=train,
        development_permit=development,
        train_feature_binding_sha256="a" * 64,
        development_feature_binding_sha256="b" * 64,
    )


def test_formal_context_rejects_source_account_and_authority_metadata_changes(
    tmp_path, monkeypatch,
):
    from dataclasses import replace
    from molgap import modern2d_training as formal

    context = _formal_context()
    monkeypatch.setattr(formal.Modern2DFormalTrainingContext, "validate_recipe",
                        lambda self, recipe: None)
    recipe = SimpleNamespace()
    spec = SimpleNamespace(identity=context.run_context.spec_identity)
    changed_source = replace(context.run_context, source_archive_sha256="c" * 64)
    changed_account = replace(context.run_context, account="other", run_reference="other/formal")
    for changed in (changed_source, changed_account):
        with pytest.raises(ValueError, match="source/package/recipe/account/run"):
            context.validate_before_payload(
                recipe=recipe, context=changed, spec=spec,
                authority_path=None, input_root=tmp_path,
            )

    authority = tmp_path / "authority.json"
    authority.write_bytes(b"changed authority bytes")
    with pytest.raises(ValueError, match="authority bytes changed"):
        context.validate_before_payload(
            recipe=recipe, context=context.run_context, spec=spec,
            authority_path=authority, input_root=tmp_path,
        )


def test_formal_context_rejects_role_and_feature_metadata_changes(monkeypatch):
    from dataclasses import replace
    from molgap import modern2d_feature_binding as feature_binding
    from molgap.data.indexed_inputs import IndexedRoleDataset
    from molgap.screen_policy import canonical_fingerprint

    context = _formal_context()
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        replace(context, train_permit=replace(context.train_permit, role="train_500k"))

    train_binding = {"role": "train_100k"}
    development_binding = {"role": "common_dev"}
    bound = replace(
        context,
        train_feature_binding_sha256=canonical_fingerprint(train_binding),
        development_feature_binding_sha256=canonical_fingerprint(development_binding),
    )

    def fake_binding(dataset, *, family, configuration):
        return train_binding if dataset.permit.role == "train_100k" else development_binding

    monkeypatch.setattr(feature_binding, "build_indexed_feature_binding", fake_binding)
    train_dataset = object.__new__(IndexedRoleDataset)
    train_dataset.permit = bound.train_permit
    development_dataset = object.__new__(IndexedRoleDataset)
    development_dataset.permit = bound.development_permit
    data = SimpleNamespace(
        synthetic=False, indexed=True, train_rows=100000,
        train=train_dataset, development=development_dataset,
    )
    recipe = SimpleNamespace(interface=("ppgt_small", "3"), model_config={})
    bound.validate_loaded_data(recipe, data, training_only=False)
    with pytest.raises(ValueError, match="feature/source/config/history"):
        replace(bound, train_feature_binding_sha256="c" * 64).validate_loaded_data(
            recipe, data, training_only=False,
        )


def test_family_shared_environment_pins_torch_271_without_gpu_probe(tmp_path, monkeypatch):
    import sys
    import time
    from molgap import kaggle_python_environment as environment

    probe = json.dumps({
        "python": "3.12", "executable": "fixture-python",
        "packages": {"numpy": "1.26.4", "torch": "2.7.1+cu126",
                      "torch-geometric": "2.6.1", "ogb": "1.3.6", "rdkit": "2025.9.5"},
        "cuda_build": "12.6", "model_constructed": False,
        "model_inference_executed": False, "training_executed": False,
        "graph_role_read": False,
    })
    calls = []

    def fake_run(command, *, env, check, timeout, text, capture_output=False):
        calls.append(command)
        return SimpleNamespace(stdout=probe if capture_output else "")

    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    executable, qualification = environment.prepare_python(
        tmp_path / "family-env", deadline=time.time() + 60,
        source_root=tmp_path, profile="family-shared",
    )
    assert executable.name == "python"
    assert any("torch==2.7.1+cu126" in command for command in calls)
    assert "https://download.pytorch.org/whl/cu126" in calls[2]
    assert qualification["requirements"] == [
        "numpy==1.26.4", "torch==2.7.1+cu126", "torch-geometric==2.6.1",
        "ogb==1.3.6", "rdkit==2025.9.5",
    ]
    assert qualification["model_constructed"] is False
    assert qualification["training_executed"] is False
    assert calls[0][0:3] == [sys.executable, "-m", "pip"]


def _tar_bytes(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("families", [
    [("ppgt_small", "3"), ("esa_small", "4")],
    [("ppgt_small", "3")],
    [("esa_small", "4")],
    [("ppgt_small", "3"), ("ppgt_small", "3")],
    [("esa_small", "4"), ("esa_small", "4")],
])
def test_formal_bootstrap_uses_family_shared_for_namespaced_mount(tmp_path, monkeypatch, families):
    entry = _load_module(
        Path(__file__).parents[1] / "platforms/kaggle/run_experiment.py",
        "formal_release_bootstrap_fixture",
    )
    input_root = tmp_path / "input"
    mount = input_root / "datasets" / "owner" / "accepted-inputs"
    working = tmp_path / "working"
    temp = tmp_path / "temp"
    mount.mkdir(parents=True)
    working.mkdir()
    temp.mkdir()
    declaration = _FormalSpec().to_dict()
    declaration["arms"] = [dict(arm_id="arm" + str(i),
        family=dict(name=family, version=version), addons=[])
        for i, (family, version) in enumerate(families)]
    declaration["prospective"] = dict(arms=[dict(
        plan_spec_ref="experiments/profile/" + a["arm_id"] + "/plan_input.json")
        for a in declaration["arms"]])
    authority = dict(user_authorized=True, scientific_training_authorized=True,
        purpose="CONFIGURED_SCREEN", allocated_t4_hours_ceiling=1,
        gpu=dict(wall_ceiling_seconds=1800, allocated_devices=2, accelerator="NvidiaTeslaT4"))
    authority_bytes = json.dumps(authority).encode()
    plan_bytes = json.dumps(dict(authority=dict(path="experiments/profile/authority.json",
        sha256=hashlib.sha256(authority_bytes).hexdigest()))).encode()
    preflight = (
        "import tarfile\n"
        "def _unpack(package, source):\n"
        "    with tarfile.open(package / 'source.tar.gz', 'r:gz') as archive:\n"
        "        archive.extractall(source)\n"
    ).encode()
    archive_bytes = _tar_bytes([("src/molgap/experiment_preflight.py", preflight),
        ("experiments/profile/acceptance_plan.json", plan_bytes),
        ("experiments/profile/authority.json", authority_bytes)])
    archive = mount / "source_payload.bin"
    archive.write_bytes(archive_bytes)
    (mount / "experiment_launch.json").write_text(json.dumps({
        "expected_source_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
    }), encoding="utf-8")
    (mount / "experiment_spec.json").write_text(json.dumps(declaration), encoding="utf-8")
    for filename in ("package_manifest.json", "SOURCE_COMMIT.txt",
                     "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        (mount / filename).write_text("{}", encoding="utf-8")

    real_path = Path

    def redirect(raw):
        return {
            "/kaggle/input": input_root,
            "/kaggle/working": working,
            "/kaggle/temp/molgap-workflow": temp / "molgap-workflow",
        }.get(str(raw), real_path(raw))

    prepared = []
    runs = []

    def fake_prepare(directory, *, deadline, source_root, profile):
        prepared.append({"directory": directory, "deadline": deadline,
                         "source_root": source_root, "profile": profile})
        directory.mkdir(parents=True)
        return directory / "venv" / "bin" / "python", {
            "status": "IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED",
            "requirements": ["torch==2.7.1+cu126"],
        }

    def fake_run(command, **kwargs):
        runs.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    import runpy
    monkeypatch.setattr(runpy, "run_path", lambda path: {"prepare_python": fake_prepare})
    monkeypatch.setattr(entry, "Path", redirect)
    monkeypatch.setattr(entry.subprocess, "run", fake_run)
    monkeypatch.setattr(entry, "_run_trial_pair", fake_run)
    launch = mount / "experiment_launch.json"
    entry.EXPECTED_LAUNCH_SHA256 = hashlib.sha256(launch.read_bytes()).hexdigest()
    monkeypatch.setattr(entry.time, "perf_counter", lambda: 1000.0)

    entry._workload(1000.0)

    assert prepared and prepared[0]["profile"] == "family-shared"
    assert prepared[0]["directory"].name == "runtime"
    assert runs and runs[0][1]["timeout"] == 1785
    command = runs[0][0]
    assert command[command.index("--input-root") + 1] == str(input_root)
    assert command[command.index("--launch") + 1] == str(launch)


@pytest.mark.parametrize("dirty", [True, False], ids=["defer-dirty-derived", "rebuild-clean"])
def test_prospective_plan_defers_dirty_derived_but_rebuilds_clean(
    tmp_path, monkeypatch, dirty,
):
    """Canonical publication succeeds while only a clean derived tree is rebuilt."""
    from unittest.mock import Mock

    from molgap import experiment_prospective as prospective

    root = tmp_path / "repo"
    root.mkdir()
    if dirty:
        (root / ".git").mkdir()
    plans = [{"spec": {"trajectory": {"trajectory_id": "T-ppgt"}},
              "output": "experiments/planned-ppgt"}]
    bindings = [{"arm_id": "ppgt", "trajectory_id": "T-ppgt",
                 "output": "experiments/planned-ppgt"}]
    canonical_records = [{"trajectory_id": "T-ppgt",
                          "path": "experiments/planned-ppgt", "status": "PLANNED"}]
    canonical_batch = {"size": 1, "batch_id": "canonical-batch"}
    plan_result = {"status": "PLANNED", "batch": canonical_batch,
                   "results": canonical_records}
    monkeypatch.setattr(prospective, "_prospective_inputs",
                        lambda spec, repo: (plans, bindings))
    monkeypatch.setattr(prospective, "plan_many", Mock(return_value=plan_result))
    rebuild = Mock(return_value={"index.json": b"canonical-derived"})
    monkeypatch.setattr(prospective, "rebuild_research_memory", rebuild)
    status = (" M research_memory/derived/index.json\n"
              + "\n".join(
                  f" M research_memory/derived/user-pin-{index:02d}.json"
                  for index in range(14)
              )) if dirty else ""
    git_status = Mock(return_value=SimpleNamespace(returncode=0, stdout=status))
    monkeypatch.setattr(subprocess, "run", git_status)

    result, code = prospective.plan_prospective(object(), root)

    assert code == 0
    assert result["completed_records"] == canonical_records
    assert result["batch"] == canonical_batch
    if dirty:
        git_status.assert_called_once_with(
            ["git", "status", "--porcelain", "--untracked-files=all", "--",
             "research_memory/derived"],
            cwd=root, text=True, capture_output=True,
        )
        assert result["status"] == "PROSPECTIVE_PLANNED_DERIVED_DEFERRED_USER_CHANGES"
        assert result["rml_rebuilt"] is False
        assert result["rml_derived_files"] == []
        rebuild.assert_not_called()
    else:
        git_status.assert_not_called()
        assert result["status"] == "PROSPECTIVE_PLANNED_AND_RML_REBUILT"
        assert result["rml_rebuilt"] is True
        assert result["rml_derived_files"] == ["index.json"]
        rebuild.assert_called_once_with(root)
