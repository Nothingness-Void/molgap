"""Synthetic CPU contracts/output fixtures; not native or scientific qualification."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.experiment_family_workflow import RunContext, inspect_output, tensor_digest
from molgap.experiment_package import _name
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_runner import (
    INTERFACES, PROFILES, _assert_target_statistics, family_arm_bindings, read_recipe,
)
from molgap.training_reproducibility import sha256_file, validate_accelerator_request
from test_family_training_kernel import initial_states
from test_family_training_session import assert_equal, events, make_session


def contract_file(tmp_path, session):
    data, recipe = session.loaders.data, session.recipe
    source = torch.cat([graph.source_idx for graph in data.development]) if data.development else None
    target = torch.cat([graph.y for graph in data.development]) if data.development else None
    record = build_family_recipe(INTERFACES[recipe.family], configuration=recipe,
        train_role_identity="3" * 64, development_role_identity="4" * 64 if data.development else None,
        source_idx_sha256=tensor_digest(source, role="source_idx") if source is not None else None,
        target_sha256=tensor_digest(target, role="target") if target is not None else None)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return path, record


def context(family, path):
    return RunContext("synthetic-question", "synthetic-run", "reference", "a" * 64, "b" * 64,
        *INTERFACES[family], "c" * 40, "d" * 64, "e" * 64, sha256_file(path),
        "local", "fixture", "fixture/run", None)


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("rows,development", [(100000, True), (500000, True), (3378606, False)])
def test_one_recipe_builder_and_training_adapter_for_each_scale(family, rows, development):
    session = make_session(family, development=development, schedule_kind="cosine_epoch", events=events())
    try:
        steps = rows // 128
        recipe = replace(session.recipe, train_rows=rows, optimizer_steps=steps * 2)
        record = build_family_recipe(INTERFACES[family], configuration=recipe,
            train_role_identity="3" * 64, development_role_identity="4" * 64 if development else None,
            source_idx_sha256="5" * 64 if development else None, target_sha256="6" * 64 if development else None)
        assert record["acceptance_requirements"]["sample_presentations"] == steps * 128 * 2
        adapter = training_adapter({"family": {"name": family, "version": INTERFACES[family][1]}, "addons": []})
        assert adapter.module == "molgap.family_training_runner"
        assert adapter.artifact_adapter == PROFILES[family]
        assert "platform" not in recipe.to_dict()
        assert family_arm_bindings(recipe)["transform"]["sha256"] != family_arm_bindings(
            replace(recipe, target_mean_eV=recipe.target_mean_eV + .01))["transform"]["sha256"]
    finally:
        session.close()


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("development", [False, True])
def test_actual_session_output_translation_and_cpu_mechanical_inspection(
        tmp_path, family, development, initial_states):
    report_ema = family == "neural_atom_k1" and development
    session = make_session(family, development=development, schedule_kind="cosine_step", events=events(),
        report_ema=report_ema)
    path, record = contract_file(tmp_path, session)
    ctx = context(family, path)
    outputs = SessionOutputs(tmp_path / "output", ctx, contract=path, adapter=PROFILES[family],
        trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"])
    session.loaders.attach_role_audit(outputs.role_audit)
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    progress = session.run()
    runtime = {"platform": ctx.platform, "account": ctx.account, "precision": "fp32",
        "source_commit": ctx.source_commit, "source_archive_sha256": ctx.source_archive_sha256}
    costs = [{"metric": "wall_seconds", "unit": "seconds", "status": "measured", "value": 1.,
        "semantics": "process_wall", "hardware": "synthetic-cpu"},
        {"metric": "device_seconds", "unit": "seconds", "status": "missing", "value": None,
        "semantics": "allocated_device", "hardware": "synthetic-cpu"}]
    report = outputs.complete(progress, runtime=runtime, costs=costs)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["scientific_acceptance"] == report["replay_readiness"] == "NOT_EVALUATED"
    assert (outputs.root / "development_predictions.pt").is_file() == development
    assert ("development_mae_eV" in report["observed"]) == development
    if report_ema:
        trace = json.loads((outputs.root / "canonical_trace.json").read_text(encoding="utf-8"))
        assert trace["metric_semantics"]["live_dev_metric"]["weights"] == "live"
        assert trace["metric_semantics"]["ema_dev_metric"]["weights"] == "ema"
        assert all(point["live_dev_metric"] is not None and point["ema_dev_metric"] is not None
                   for point in trace["observations"])
    manifest_path = outputs.root / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resume_path = outputs.root / manifest["artifacts"]["resume"]["path"]
    state = torch.load(resume_path, map_location="cpu", weights_only=True)
    state["sample_presentations"] += 1
    torch.save(state, resume_path)
    manifest["artifacts"]["resume"]["sha256"] = sha256_file(resume_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    blocked = inspect_output(outputs.root, context=ctx, expected=record["acceptance_requirements"])
    assert blocked["status"] == "BLOCKED"
    assert any("exposure" in error for error in blocked["blockers"])

    if family == "neural_atom_k1" and development:
        # Exercise the real K1 checkpoint/bundle journey with Clean-EMA state.
        # The fixture is CPU-only and makes no native qualification claim.
        from molgap.experiment_resume import (create_resume_bundle,
            restore_resume_bundle, validate_resume_bundle)
        from molgap.family_training_data import DatasetSelection, TrainingData
        from molgap.family_training_kernel import FamilyKernel
        from molgap.family_training_loaders import FamilyLoaders
        from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
        from molgap.family_training_session import FamilyTrainingSession
        from molgap.family_training_state import CalibrationPolicy, source_order_sha256
        from molgap.gptrans_execution_tuning import ExecutionTuning
        from test_family_training_kernel import synthetic_batch

        initial, digest = initial_states[family]
        template = synthetic_batch().to_data_list()[0]
        def graph(index):
            item = template.clone()
            item.source_idx = torch.tensor([index])
            item.y = torch.tensor([3. + (index % 7 - 3) * .1])
            return item

        selection = DatasetSelection("synthetic.json", "2" * 64, True)
        clean_recipe = FamilyRecipe(family, "reference", 42, digest, selection, 256, 2, 4,
            .001, .01, ScheduleSpec("cosine_step", 4), "best-development-clean-ema",
            "torch-pass-randperm", 1, target_mean_eV=3., target_std_eV=2., ema_decay=.999,
            clean_bn=CalibrationPolicy(selection.manifest_sha256, "3" * 64,
                source_order_sha256(torch.arange(128)), 1), development_reports=("live",),
            torch_version=str(torch.__version__).split("+", 1)[0])
        data = TrainingData([graph(i) for i in range(256)],
            [graph(256 + i) for i in range(2)], selection.manifest_sha256, 256, 2,
            0, 256, 3., 2., synthetic=True)

        def actual_session():
            from molgap.training_reproducibility import configure_fp32_determinism
            configure_fp32_determinism(42)
            return FamilyTrainingSession(FamilyKernel(clean_recipe, initial, device="cpu"),
                FamilyLoaders(clean_recipe, data, ExecutionTuning(training_workers=0, development_workers=0)),
                checkpoint=lambda *args: None, selected=lambda *args: None,
                pass_finished=lambda *args: None)

        contract_dir = tmp_path / "clean-contract"
        contract_dir.mkdir()
        contract_session = actual_session()
        clean_path, clean_record = contract_file(contract_dir, contract_session)
        contract_session.close()
        clean_ctx = context(family, clean_path)
        baseline = actual_session()
        endpoint = baseline.run()
        expected_state = deepcopy(baseline.state_dict())

        split = actual_session()
        clean_output = SessionOutputs(tmp_path / "clean-output", clean_ctx,
            contract=clean_path, adapter=PROFILES[family], trajectory_id="synthetic-trajectory",
            train_role_identity=clean_record["train_role_identity"])
        split.loaders.attach_role_audit(clean_output.role_audit)
        split.checkpoint, split.selected, split.pass_finished = (
            clean_output.checkpoint, clean_output.selected, clean_output.pass_finished)
        split.batch_finished = clean_output.batch_finished
        clean_output.logger.start(split.progress)
        paused = split.run(should_pause=lambda: split.progress.optimizer_step >= 2)
        assert paused.status == "paused" and paused.optimizer_step == 2
        wire = torch.load(clean_output.root / "last_checkpoint.pt", map_location="cpu", weights_only=True)
        assert wire["cursor"]["weight_state"]["clean"] is not None
        assert all(isinstance(value, torch.Tensor) for value in wire["cursor"]["pending_absolute"])
        checkpoint_bytes = (clean_output.root / "last_checkpoint.pt").read_bytes()

        trajectory = {"trajectory_id": "synthetic-trajectory", "record_mode": "prospective",
            "state_at_start": {"source_commit": clean_ctx.source_commit,
                                "source_config_identity": clean_ctx.arm_identity},
            "actions": [{"action_id": "A001", "run_ids": [clean_ctx.logical_run_id],
                         "attempt_ids": ["v1"]}], "decision": {"outcome": "ACTIVE"}}
        bundle = tmp_path / "clean-bundle"
        create_resume_bundle(clean_output.root, clean_ctx, trajectory, bundle)
        verified = validate_resume_bundle(bundle, context=clean_ctx, trajectory=trajectory)
        assert "weight_state" not in verified["manifest"]["cursor"]
        assert "pending_absolute" not in verified["manifest"]["cursor"]
        restored = tmp_path / "clean-restored"
        restore_resume_bundle(bundle, restored, context=clean_ctx, trajectory=trajectory)
        assert (restored / "last_checkpoint.pt").read_bytes() == checkpoint_bytes

        resumed = actual_session()
        restored_output = SessionOutputs(restored, clean_ctx, contract=clean_path,
            adapter=PROFILES[family], trajectory_id="synthetic-trajectory",
            train_role_identity=clean_record["train_role_identity"])
        resumed.loaders.attach_role_audit(restored_output.role_audit)
        resumed.checkpoint, resumed.selected, resumed.pass_finished = (
            restored_output.checkpoint, restored_output.selected, restored_output.pass_finished)
        resumed.batch_finished = restored_output.batch_finished
        resumed.load_state_dict(restored_output.retained_state())
        restored_output.logger.start(resumed.progress)
        assert resumed.run() == endpoint
        assert_equal(expected_state, resumed.state_dict())


def test_worker_startup_failure_closes_kernel_and_session(tmp_path, monkeypatch):
    import signal
    from molgap import family_training_kernel as kernels
    from molgap import family_training_runner as worker
    from molgap import family_training_session as sessions
    from molgap.gptrans_execution_tuning import ExecutionTuning
    from test_family_training_session import SyntheticKernel, FamilyTrainingSession

    template = make_session("neural_atom_k1", development=False,
        schedule_kind="cosine_step", events=events())
    path, record = contract_file(tmp_path, template)
    ctx = context("neural_atom_k1", path)
    failed_root = tmp_path / "startup-failure"
    failed_root.mkdir()
    worker._begin_cost_interval(failed_root, ctx, had_checkpoint=False)
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"synthetic initial state")
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    created, kernel_closes = [], []

    def cpu_kernel(configuration, state_path, *, device, **_):
        assert device == "cuda"
        kernel = SyntheticKernel(configuration)
        close = kernel.close
        def close_kernel():
            kernel_closes.append(True)
            close()
        kernel.close = close_kernel
        return kernel

    def retained_session(*args, **kwargs):
        value = FamilyTrainingSession(*args, **kwargs)
        created.append(value)
        return value

    previous_signal = signal.getsignal(signal.SIGTERM)
    original_signal = signal.signal

    def interrupt_startup(number, handler):
        result = original_signal(number, handler)
        if handler is not previous_signal:
            raise OSError("startup interrupted before session.run")
        return result

    with monkeypatch.context() as boundary:
        boundary.setattr(worker, "_bootstrap", lambda **_: (
            record, template.recipe, template.loaders.data,
            ExecutionTuning(training_workers=0, development_workers=0),
            ctx, {"accelerator": "synthetic-cpu"}, tmp_path))
        boundary.setattr(worker, "_verify_preflight", lambda *a, **k: {})
        boundary.setattr(worker, "_loaded_source", lambda _: tmp_path)
        boundary.setattr("molgap.family_workflow_trial.is_trial", lambda _: False)
        boundary.setattr("molgap.k1_infrastructure_comparison.is_comparison", lambda _: False)
        boundary.setattr("molgap.gptrans_infrastructure_comparison.is_comparison", lambda _: False)
        boundary.setattr(kernels, "FamilyKernel", cpu_kernel)
        boundary.setattr(sessions, "FamilyTrainingSession", retained_session)
        boundary.setattr(signal, "signal", interrupt_startup)
        with pytest.raises(OSError, match="startup interrupted"):
            worker.run_screen_arm(output=failed_root, recipe_path=path,
                initial_state_path=initial, package_dir=package_dir,
                trajectory_id="synthetic-trajectory", spec=object())

    assert signal.getsignal(signal.SIGTERM) is previous_signal
    assert len(created) == len(kernel_closes) == 1
    failed = created[0]
    assert failed.closed and failed.loaders.closed
    assert failed.progress.optimizer_step == failed.progress.sample_presentations == 0
    assert not failed.kernel.optimizer.state
    assert not (failed_root / "last_checkpoint.pt").exists()
    assert not (failed_root / "output_manifest.json").exists()


def test_contract_rejects_counter_changes_and_unknown_fields(tmp_path):
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    try:
        path, original = contract_file(tmp_path, session)
        assert read_recipe(path)[1] == session.recipe
        retained = deepcopy(original)
        retained["initial_development_check"] = {
            "prediction_tolerance_eV": 1e-4, "mae_tolerance_eV": 1e-6}
        path.write_text(json.dumps(retained), encoding="utf-8")
        assert read_recipe(path)[1] == session.recipe
        reported = deepcopy(original)
        reported["development_reports"] = ["ema"]
        path.write_text(json.dumps(reported), encoding="utf-8")
        assert read_recipe(path)[1] == replace(session.recipe, development_reports=("ema",))
        for invalid in ({"prediction_tolerance_eV": 1e-4},
                        {"prediction_tolerance_eV": float("nan"), "mae_tolerance_eV": 1e-6},
                        {"prediction_tolerance_eV": True, "mae_tolerance_eV": 1e-6}):
            retained["initial_development_check"] = invalid
            path.write_text(json.dumps(retained), encoding="utf-8")
            with pytest.raises(ValueError):
                read_recipe(path)
        for change in ("counter", "unknown", "development"):
            record = deepcopy(original)
            if change == "counter":
                record["acceptance_requirements"]["sample_presentations"] += 1
            elif change == "unknown":
                record["callback"] = "external:execute"
            else:
                record["development_role_identity"] = "fake"
            path.write_text(json.dumps(record), encoding="utf-8")
            with pytest.raises(ValueError):
                read_recipe(path)
    finally:
        session.close()


@pytest.mark.parametrize("family,expected", [
    ("neural_atom_k1", (5.3383002281188965, 1.275090217590332)),
    ("gptrans_t", (5.338300266634226, 1.2750902063230574)),
])
def test_target_statistics_match_recipe_in_native_precision(family, expected):
    session = make_session(family, development=False, schedule_kind="cosine_step", events=events())
    try:
        recipe = replace(session.recipe, target_mean_eV=expected[0], target_std_eV=expected[1])
        data = SimpleNamespace(target_mean_eV=5.338300266634226,
                               target_std_eV=1.2750902063230574)
        _assert_target_statistics(recipe, data)
    finally:
        session.close()


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
def test_target_statistics_drift_is_rejected(family):
    session = make_session(family, development=False, schedule_kind="cosine_step", events=events())
    try:
        recipe = replace(session.recipe, target_mean_eV=5.338300266634226,
                         target_std_eV=1.2750902063230574)
        data = SimpleNamespace(target_mean_eV=5.338300266634226 + 1e-4,
                               target_std_eV=1.2750902063230574)
        with pytest.raises(ValueError, match="target statistics"):
            _assert_target_statistics(recipe, data)
    finally:
        session.close()


def test_python_io_owners_are_source_and_artifact_directories_stay_blocked():
    for path in ("src/molgap/family_training_outputs.py", "src/molgap/other_outputs.py"):
        assert _name(path) == path
    for path in ("src/molgap/other_outputs.json", "src/molgap/output_models/worker.py",
                 "experiments/example/outputs/model.py", "data/family_training_outputs.py",
                 "src/molgap/private_token.py"):
        with pytest.raises(ValueError):
            _name(path)


@pytest.mark.parametrize("requested,observed,hip", [
    ("T4", "Tesla T4", False), ("NvidiaTeslaT4", "NVIDIA T4", False),
    ("A100", "NVIDIA A100-SXM4-40GB", False),
    ("RTX 5060", "NVIDIA GeForce RTX 5060", False),
    ("DCU", "opaque-device-name", True),
])
def test_resource_matching_is_metadata_only(requested, observed, hip):
    validate_accelerator_request(requested, observed, hip_runtime=hip)


@pytest.mark.parametrize("requested,observed,hip", [
    ("T4", "NVIDIA A100-SXM4-40GB", False), ("A100", "Tesla T4", False),
    ("RTX 5060", "NVIDIA GeForce RTX 5060 Ti", False),
    ("GPU", "GPU", False), ("DCU", "NVIDIA A100-SXM4-40GB", False),
    ("NVIDIA GPU", "NVIDIA GPU", False), ("NVIDIA", "NVIDIA", False),
    ("---", "---", False), ("CUDA", "CUDA", False),
    ("ACCELERATOR", "ACCELERATOR", False), ("T4", "GPU", False),
    ("DCU", "GPU", True), ("DCU", "NVIDIA", True),
])
def test_wrong_allocation_cannot_be_qualified(requested, observed, hip):
    with pytest.raises(ValueError):
        validate_accelerator_request(requested, observed, hip_runtime=hip)


def test_native_proof_storage_round_trip_and_interrupted_retirement(tmp_path):
    import hashlib
    from molgap.native_proof_storage import (RESUME_ARCHIVE, compact_certificate, load_proof,
        proof_stream, retire_proof_sources)
    from molgap.research_memory.trace import atomic_write, json_bytes

    first, second = tmp_path / "actual-resume-a.pt", tmp_path / "actual-resume-b.pt"
    torch.save({"state": torch.arange(4)}, first)
    torch.save({"state": torch.arange(4)}, second)
    from molgap.training_reproducibility import sha256_file
    certificate = {"proofs": [{"path": first.name, "sha256": sha256_file(first)},
        {"path": second.name, "sha256": sha256_file(second)}]}
    compact_certificate(tmp_path, certificate, actual_resume=True)
    with pytest.raises(FileNotFoundError):
        retire_proof_sources(tmp_path, certificate)
    atomic_write(tmp_path / "actual_resume_certificate.json", json_bytes(certificate))
    archive = tmp_path / RESUME_ARCHIVE
    original = archive.read_bytes()
    archive.write_bytes(b"interrupted archive")
    with pytest.raises(ValueError, match="intact published storage"):
        retire_proof_sources(tmp_path, certificate)
    archive.write_bytes(original)
    retire_proof_sources(tmp_path, certificate)
    assert not first.exists() and not second.exists() and archive.is_file()
    with proof_stream(tmp_path, first.name, certificate["proofs"][0]["sha256"]) as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == certificate["proofs"][0]["sha256"]
    assert load_proof(tmp_path, certificate["proofs"][0])["state"].tolist() == [0, 1, 2, 3]
