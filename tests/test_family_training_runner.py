"""Synthetic CPU contracts/output fixtures; not native or scientific qualification."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.experiment_family_workflow import RunContext, inspect_output, tensor_digest
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_runner import INTERFACES, PROFILES, family_arm_bindings, read_recipe
from molgap.training_reproducibility import configure_fp32_determinism, sha256_file, validate_accelerator_request
from test_family_training_kernel import initial_states, synthetic_batch
from test_family_training_session import assert_equal, events, make_session


@pytest.mark.parametrize("damage", ["missing", "changed", "complete"])
def test_retained_native_dependency_inventory_fails_closed(tmp_path, monkeypatch, damage):
    """Synthetic bytes test admission to the real verifier, never native truth."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    from molgap import family_training_runner as owner
    from molgap.v4_runtime import normalized_source_sha256
    contract = tmp_path / "training_contract.json"
    contract.write_bytes(b"{}\r\n")
    ctx = replace(context("neural_atom_k1", contract), training_recipe_sha256=normalized_source_sha256(contract))
    provenance = {"context": ctx.to_dict(), "recipe_identity": "frozen-recipe"}
    (tmp_path / "runtime_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    proof = tmp_path / "repeat-a.pt"
    proof.write_bytes(b"synthetic dependency, not a tensor qualification")
    digest = sha256_file(proof)
    (tmp_path / "runtime_certificate.json").write_text(json.dumps({"proofs": [
        {"path": proof.name, "sha256": digest}]}), encoding="utf-8")
    monkeypatch.setattr(owner, "read_recipe", lambda _: ({}, SimpleNamespace(identity="frozen-recipe")))
    verifier = Mock()
    monkeypatch.setattr(owner, "_verify_preflight", verifier)
    if damage == "missing":
        proof.unlink()
    elif damage == "changed":
        proof.write_bytes(b"changed")
    if damage == "changed":
        with pytest.raises(ValueError, match="hash mismatch"):
            owner.inspect_retained_native_qualification(tmp_path, recipe_path=contract,
                provenance=provenance, context=ctx)
        verifier.assert_not_called()
        return
    result = owner.inspect_retained_native_qualification(tmp_path, recipe_path=contract,
        provenance=provenance, context=ctx)
    if damage == "missing":
        assert result["status"] == "BLOCKED"
        assert result["missing"] == ["repeat-a.pt"]
        assert result["missing_size_bytes"] is None
        verifier.assert_not_called()
    else:
        assert result["status"] == "VERIFIED"
        assert result["missing"] == []
        verifier.assert_called_once()
    assert result["scientific_acceptance"] == result["replay_readiness"] == "NOT_EVALUATED"


@pytest.fixture
def cpu_replay(monkeypatch):
    from molgap.training_reproducibility import capture_rng_state, restore_rng_state
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before, threads = capture_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    precision = torch.get_float32_matmul_precision()
    for backend, name in ((torch.backends.cuda.matmul, "allow_tf32"),
                          (torch.backends.cudnn, "allow_tf32"),
                          (torch.backends.cudnn, "benchmark"),
                          (torch.backends.cudnn, "deterministic")):
        monkeypatch.setattr(backend, name, getattr(backend, name))
    torch.set_num_threads(1)
    try:
        yield
    finally:
        restore_rng_state(before)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.set_float32_matmul_precision(precision)


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
        tmp_path, cpu_replay, initial_states, family, development, monkeypatch):
    from molgap.experiment_resume import create_resume_bundle, restore_resume_bundle, validate_resume_bundle
    from molgap.family_training_data import DatasetSelection, TrainingData
    from molgap.family_training_kernel import FamilyKernel
    from molgap.family_training_loaders import FamilyLoaders
    from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
    from molgap.family_training_session import FamilyTrainingSession
    from molgap.gptrans_execution_tuning import ExecutionTuning
    from molgap.screen_policy import canonical_fingerprint
    from test_experiment_workflow import _arm, _spec_payload

    initial, digest = initial_states[family]
    template = synthetic_batch().to_data_list()[0]
    def graph(index):
        item = template.clone()
        item.source_idx = torch.tensor([index])
        item.y = torch.tensor([3. + (index % 7 - 3) * .1])
        return item
    selection = DatasetSelection("synthetic.json", "2" * 64, development)
    recipe = FamilyRecipe(family, "reference", 42, digest, selection, 256,
        2 if development else 0, 4, .001, .01, ScheduleSpec("cosine_step", 4),
        "best-development-ema" if development else "final-live", "torch-pass-randperm", 1,
        target_mean_eV=3., target_std_eV=2., ema_decay=.999 if development else None,
        torch_version=str(torch.__version__).split("+", 1)[0])
    if family == "neural_atom_k1" and development:
        from molgap.family_training_state import CalibrationPolicy, source_order_sha256
        recipe = replace(recipe, selection="best-development-clean-ema", development_reports=("live",),
            clean_bn=CalibrationPolicy(selection.manifest_sha256, "3" * 64,
                source_order_sha256(torch.arange(128)), 1))
    data = TrainingData([graph(i) for i in range(256)],
        [graph(256 + i) for i in range(2)] if development else None,
        selection.manifest_sha256, 256, 2 if development else 0,
        0, 256 if development else None, 3., 2.)
    def actual_session():
        configure_fp32_determinism(42)
        return FamilyTrainingSession(FamilyKernel(recipe, initial, device="cpu"),
            FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0, development_workers=0)),
            checkpoint=lambda *args: None, selected=lambda *args: None, pass_finished=lambda *args: None)

    # Reuse the server's real family/session/output recovery journey with the
    # desktop-supported K1/GPTrans factories and no model or writer replacement.
    baseline = actual_session()
    endpoint = baseline.run()
    expected_state = deepcopy(baseline.state_dict())
    session = actual_session()
    path, record = contract_file(tmp_path, session)
    from molgap.v4_runtime import normalized_source_sha256
    # Source contracts use the package's LF identity even on Windows. The
    # retained contract is a normalized artifact with its own exact byte hash.
    path.write_bytes(path.read_bytes() + b"\r\n")
    recipe_source_sha = normalized_source_sha256(path)
    arm = _arm(family, INTERFACES[family][1], "reference", recipe_sha256=recipe_source_sha, state_sha256=digest)
    bindings = family_arm_bindings(recipe)
    arm["base"] = bindings["base"]
    arm["training"].update({key: bindings[key] for key in ("sampler", "transform", "objective")})
    arm["data"]["dataset"]["sha256"] = selection.manifest_sha256
    arm["data"]["roles"] = [dict(arm["data"]["roles"][0], role=role,
        usage_sha256=record[role + "_role_identity"])
        for role in (["train", "development"] if development else ["train"])]
    declaration = _spec_payload([arm])
    declaration["prospective"]["arms"][0]["trajectory_id"] = "synthetic-trajectory"
    spec = ExperimentSpec(declaration)
    from molgap import experiment_launch as launch
    from molgap.constants import REPO_ROOT
    from molgap.experiment_package import build_experiment_source_package
    package_dir = tmp_path / "source-package"
    package = build_experiment_source_package(spec, REPO_ROOT, ["README.md"], package_dir)
    package_identity = package["package_identity"]
    platform_reference = "synthetic-account/synthetic-pair-run"

    def observed_receipt(version):
        version = str(version)
        binding = launch.build_launch_receipt(spec, package_dir,
            expected_package_identity=package_identity)["binding"]
        known = lambda value: {"value": value, "missing_reason": None}
        unknown = {"value": None, "missing_reason": "not_observed"}
        response = {"format": launch.RESPONSE_FORMAT, "version": launch.VERSION,
            "mode": "observed", "outcome": "accepted", "conflict_kind": None,
            "binding": binding,
            "canonical_platform_reference": known(platform_reference),
            "platform_version": known(version),
            "physical_runs": known([{"run_identity": "synthetic-physical-run-v" + version,
                "canonical_reference": platform_reference,
                "platform_version": known(version), "arm_ids": ["reference"]}]),
            "timestamp": unknown, "monitor_paths": unknown}
        receipt = launch.reconcile_platform_response(spec, package_dir,
            launch.canonical_json(response), expected_package_identity=package_identity)
        directory = tmp_path / ("receipt-v" + version)
        directory.mkdir()
        return launch.write_launch_receipt(launch.canonical_json(receipt), directory,
            spec, package_dir, expected_package_identity=package_identity)

    origin_receipt = observed_receipt("1")
    continued_receipt = observed_receipt("2")
    origin_ctx = RunContext.from_launch(spec, origin_receipt, package_dir,
        expected_package_identity=package_identity, arm_id="reference")
    # The worker writes before it can observe a platform version. Preserve
    # that unknown producer field throughout checkpointing and disk restore;
    # the later receipt is carried separately by terminal staging.
    ctx = replace(origin_ctx, platform_version=None)
    outputs = SessionOutputs(tmp_path / "output", ctx, contract=path, adapter=PROFILES[family],
        trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"])
    assert b"\r" not in (outputs.root / "training_contract.json").read_bytes()
    assert sha256_file(outputs.root / "training_contract.json") == recipe_source_sha
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    session.batch_finished = outputs.batch_finished
    outputs.logger.start(session.progress)
    paused = session.run(should_pause=lambda: session.progress.optimizer_step >= 2)
    assert paused.status == "paused" and paused.optimizer_step == 2
    wire = torch.load(outputs.root / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    assert ("ema" in wire) is development
    if recipe.clean_bn is not None:
        assert wire["cursor"]["weight_state"]["clean"] is not None
    assert wire["context"]["platform_version"] is None
    from molgap.experiment_family_workflow import terminal_attempt_id
    trace_run_id = ctx.logical_run_id + ":" + ctx.arm_id + ":downstream"
    origin_attempt_id = terminal_attempt_id(spec, ctx.arm_id, "1")
    from test_run_acceptance_integration import _trajectory
    binding = declaration["prospective"]["arms"][0]
    trajectory = _trajectory(binding, arm, ctx, "experiments/synthetic/training_contract.json")
    trajectory["actions"][0]["run_ids"] = [trace_run_id]
    trajectory["actions"][0]["attempt_ids"] = [origin_attempt_id]
    trajectory_path = tmp_path / binding["output"] / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(launch.canonical_json(trajectory), encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_resume_bundle(outputs.root, ctx, trajectory, bundle)
    verified = validate_resume_bundle(bundle, spec, context=ctx, trajectory=trajectory)
    assert verified["status"] == "RESUME_VERIFIED"
    assert "weight_state" not in verified["manifest"]["cursor"]
    assert "pending_absolute" not in verified["manifest"]["cursor"]
    assert verified["owner"]["optimizer_steps"] == 2
    assert len(verified["sidecars"]["context"]) == 1
    assert json.loads(verified["sidecars"]["context"][0].read_text(encoding="utf-8"))["context"]["platform_version"] is None
    from molgap.family_training_runner import validate_screen_resume
    damaged = deepcopy(wire)
    damaged.pop("ema" if development else "scheduler")
    damaged_path = tmp_path / "incomplete-checkpoint.pt"
    torch.save(damaged, damaged_path)
    with pytest.raises(ValueError, match="EMA policy" if development else "Incomplete family resume state"):
        validate_screen_resume(spec=spec, arm_id="reference", manifest=verified["manifest"],
            bundle_root=bundle, artifacts={**verified["artifacts"], "checkpoint": damaged_path},
            sidecars=verified["sidecars"], trajectory=trajectory, context=ctx.to_dict())
    restored = tmp_path / "restored"
    restore_resume_bundle(bundle, restored, spec, context=ctx, trajectory=trajectory)
    outputs = SessionOutputs(restored, ctx, contract=path, adapter=PROFILES[family],
        trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"])
    retained = outputs.retained_state()
    before = deepcopy(retained)
    from molgap.family_training_runner import _actual_resume_proofs
    proofs_root = tmp_path / "scratch-proofs"
    proofs_root.mkdir()
    proofs = _actual_resume_proofs(recipe, data,
        ExecutionTuning(training_workers=0, development_workers=0), initial,
        retained, proofs_root, device="cpu")
    assert len(proofs) == 2 and proofs[0]["state_fingerprint"] == proofs[1]["state_fingerprint"]
    from molgap.native_proof_storage import compact_certificate, retire_proof_sources, load_proof, RESUME_ARCHIVE
    original_proofs = [torch.load(proofs_root / proof["path"], weights_only=True) for proof in proofs]
    certificate = {"proofs": proofs}
    compact_certificate(proofs_root, certificate, actual_resume=True)
    # An interrupted publication must leave the original proofs available.
    with pytest.raises(FileNotFoundError):
        retire_proof_sources(proofs_root, certificate)
    assert all((proofs_root / proof["path"]).is_file() for proof in proofs)
    from molgap.research_memory.trace import atomic_write, json_bytes
    atomic_write(proofs_root / "actual_resume_certificate.json", json_bytes(certificate))
    archive_path = proofs_root / RESUME_ARCHIVE
    archive_bytes = archive_path.read_bytes()
    archive_path.write_bytes(b"interrupted archive")
    with pytest.raises(ValueError, match="intact published storage"):
        retire_proof_sources(proofs_root, certificate)
    with pytest.raises(ValueError, match="Malformed native proof storage"):
        load_proof(proofs_root, {**proofs[0], "path": "actual-resume-absent.pt"})
    assert all((proofs_root / proof["path"]).is_file() for proof in proofs)
    archive_path.write_bytes(archive_bytes)
    retire_proof_sources(proofs_root, certificate)
    assert (proofs_root / RESUME_ARCHIVE).is_file()
    assert not any((proofs_root / proof["path"]).exists() for proof in proofs)
    for proof, original in zip(proofs, original_proofs):
        assert_equal(load_proof(proofs_root, proof), original)
    assert_equal(before, retained)
    session = actual_session()
    session.checkpoint_builder = outputs.checkpoint_from_builder
    session.selected, session.pass_finished = outputs.selected, outputs.pass_finished
    session.batch_finished = outputs.batch_finished
    session.load_state_dict(deepcopy(retained))
    outputs.logger.start(session.progress)
    progress = session.run()
    assert progress == endpoint
    assert_equal(expected_state, session.state_dict())
    assert_equal(before, retained)
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
    telemetry = json.loads((outputs.root / "training_progress.json").read_text(encoding="utf-8"))
    assert telemetry["optimizer_steps"] == telemetry["checkpoint_optimizer_steps"] == progress.optimizer_step
    assert telemetry["status"] == progress.status
    if development:
        assert telemetry["best_development_mae_eV"] == pytest.approx(report["observed"]["development_mae_eV"])
        if family == "neural_atom_k1":
            trace = json.loads((outputs.root / "canonical_trace.json").read_text(encoding="utf-8"))
            assert trace["metric_semantics"]["live_dev_metric"]["weights"] == "live"
            assert trace["metric_semantics"]["ema_dev_metric"]["weights"] == "clean-ema"
            assert all(point["live_dev_metric"] is not None and point["ema_dev_metric"] is not None
                       for point in trace["observations"])
    else:
        assert telemetry["development_mae_eV"] is telemetry["best_development_mae_eV"] is None

    # Exercise the real continuation serializer and validator using the actual
    # source checkpoint bundle and restored SessionOutputs curve. The receipt,
    # source package and bundle owners run for real; CPU fixtures do not qualify
    # a native runtime or strict replay.
    from molgap.experiment_terminal_staging import stage_verified_terminal_packages
    from molgap.research_memory.paired import validate_terminal_continuation
    from molgap.research_memory.trace import file_digest, load_canonical_trace
    source_trace = load_canonical_trace(tmp_path / "output" / "canonical_trace.json")
    continued_trace = load_canonical_trace(restored / "canonical_trace.json")
    assert continued_trace["observations"][:len(source_trace["observations"])] == source_trace["observations"]
    authority = tmp_path / "synthetic-recovery-authority.json"
    authority.write_text(json.dumps({"format": "synthetic-fixture-authority-v1",
        "scope": "local CPU fixture only; no submission or research authority"}), encoding="utf-8")
    native_root = tmp_path / "native-not-qualified"
    native_root.mkdir()
    (native_root / "runtime_certificate.json").write_text(json.dumps({
        "status": "NOT_QUALIFIED", "reason": "This synthetic CPU fixture did not run native preflight."}),
        encoding="utf-8")
    actual_resume_root = tmp_path / "resume-not-native-qualified"
    actual_resume_root.mkdir()
    (actual_resume_root / "actual_resume_certificate.json").write_text(json.dumps({
        "status": "NOT_QUALIFIED",
        "reason": "CPU mechanical restore equivalence is not native resume qualification."}),
        encoding="utf-8")
    continued_attempt_id = terminal_attempt_id(spec, ctx.arm_id, "2")
    continued_ctx = RunContext.from_launch(spec, continued_receipt, package_dir,
        expected_package_identity=package_identity, arm_id="reference")
    staging_report = inspect_output(restored, context=continued_ctx,
        expected=record["acceptance_requirements"])
    assert staging_report["status"] == "MECHANICALLY_VERIFIED", staging_report
    assert staging_report["observed"]["producer_platform_version"] is None
    assert json.loads((restored / "output_manifest.json").read_text(encoding="utf-8"))["context"]["platform_version"] is None
    trace_locator = (restored / "canonical_trace.json").relative_to(tmp_path).as_posix()
    terminal_pointer = binding["output"] + "/terminal.json"
    staging_resolved = {"spec": spec,
        "options": {"repo_root": tmp_path, "package_dir": package_dir,
            "expected_package_identity": package_identity, "receipt_path": continued_receipt},
        "contexts": {"reference": continued_ctx},
        "outputs": {"reference": {"output_dir": restored,
            "expected": record["acceptance_requirements"], "context": continued_ctx}},
        "locations": {"reference": {"trajectory_id": binding["trajectory_id"],
            "run_id": trace_run_id, "trajectory": binding["output"] + "/trajectory.json",
            "terminal": terminal_pointer, "trace": trace_locator}},
        "continuations": {"reference": {"package_dir": package_dir,
            "origin_receipt": origin_receipt, "continued_receipt": continued_receipt,
            "resume_bundle": bundle, "continued_output": restored, "authority": authority,
            "native_root": native_root, "actual_resume_root": actual_resume_root}}}
    staged = stage_verified_terminal_packages(None, tmp_path, trace_run_id, continued_attempt_id,
        resolved=staging_resolved, reports={"reference": staging_report})
    assert staged["executed"] is False
    terminal = json.loads(Path(staged["paths"]["reference"]["terminal"]).read_text(encoding="utf-8"))
    continuation_path = tmp_path / terminal["continuation_ref"]
    continuation_record = json.loads(continuation_path.read_text(encoding="utf-8"))
    assert terminal["artifact_hashes"][terminal["continuation_ref"]] == file_digest(continuation_path)
    assert continuation_record["origin"]["attempt_id"] == origin_attempt_id
    assert terminal["same_run_observation"]["attempt_id"] == continued_attempt_id
    assert validate_terminal_continuation(tmp_path, terminal) == continuation_record["origin"]

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
        # Exercise the actual worker entry before run(), using this already
        # constructed CPU fixture as its admission boundary. Source/native
        # admission is supplied here, not qualified by this cleanup check.
        import signal
        import molgap.family_training_kernel as kernels
        import molgap.family_training_session as sessions
        from molgap import family_training_runner as worker
        failed_root = tmp_path / "startup-failure"
        failed_root.mkdir()
        worker._begin_cost_interval(failed_root, ctx, had_checkpoint=False)
        created, kernel_closes = [], []
        def cpu_kernel(configuration, state_path, *, device):
            assert device == "cuda"
            kernel = FamilyKernel(configuration, state_path, device="cpu")
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
        original_signal, previous = signal.signal, signal.getsignal(signal.SIGTERM)
        def interrupt_startup(number, handler):
            result = original_signal(number, handler)
            if handler is not previous:
                raise OSError("startup interrupted before session.run")
            return result
        with monkeypatch.context() as boundary:
            boundary.setattr(worker, "_bootstrap", lambda **_: (
                record, recipe, data, ExecutionTuning(training_workers=0, development_workers=0),
                ctx, {"accelerator": "synthetic-cpu"}, tmp_path))
            boundary.setattr(worker, "_verify_preflight", lambda *a, **k: {})
            boundary.setattr(worker, "_loaded_source", lambda _: tmp_path)
            boundary.setattr(kernels, "FamilyKernel", cpu_kernel)
            boundary.setattr(sessions, "FamilyTrainingSession", retained_session)
            boundary.setattr(signal, "signal", interrupt_startup)
            with pytest.raises(OSError, match="startup interrupted"):
                worker.run_screen_arm(output=failed_root, recipe_path=path,
                    initial_state_path=initial, package_dir=package_dir,
                    trajectory_id="synthetic-trajectory")
        assert signal.getsignal(signal.SIGTERM) is previous
        assert len(created) == len(kernel_closes) == 1
        failed = created[0]
        assert failed.closed and failed.loaders.closed
        assert failed.progress.optimizer_step == failed.progress.sample_presentations == 0
        assert not failed.kernel.optimizer.state
        assert not (failed_root / "last_checkpoint.pt").exists()
        assert not (failed_root / "output_manifest.json").exists()

        # Reuse the actual CPU kernel/worker boundary for the retained predictor
        # guard. Native/source admission remains the explicit substitute above.
        probe = actual_session()
        try:
            reference = probe.kernel.evaluate(probe.loaders, weights="live")
        finally:
            probe.close()
        guarded_initial = tmp_path / "guarded-initial.pt"
        initial_payload = torch.load(initial, map_location="cpu", weights_only=True)
        initial_payload["reference_development"] = reference
        torch.save(initial_payload, guarded_initial)
        guard_record = {**record, "initial_development_check": {
            "prediction_tolerance_eV": 1e-4, "mae_tolerance_eV": 1e-6}}
        for damage in ("valid", "mismatch", "nan_prediction", "wrong_shape", "nan_mae", "inconsistent_mae"):
            matches = damage == "valid"
            guard_root = tmp_path / ("guard-" + damage)
            guard_root.mkdir()
            worker._begin_cost_interval(guard_root, ctx, had_checkpoint=False)
            guarded_reference = deepcopy(reference)
            if damage == "mismatch":
                guarded_reference["prediction_eV"] += 1
                guarded_reference["mae_eV"] = float((guarded_reference["prediction_eV"] - guarded_reference["target_eV"]).abs().mean())
            elif damage == "nan_prediction":
                guarded_reference["prediction_eV"][0] = float("nan")
            elif damage == "wrong_shape":
                guarded_reference["prediction_eV"] = guarded_reference["prediction_eV"].reshape(-1, 1)
            elif damage == "nan_mae":
                guarded_reference["mae_eV"] = float("nan")
            elif damage == "inconsistent_mae":
                guarded_reference["mae_eV"] += .1
            initial_payload["reference_development"] = guarded_reference
            torch.save(initial_payload, guarded_initial)
            with monkeypatch.context() as boundary:
                boundary.setattr(worker, "_bootstrap", lambda **_: (
                    guard_record, recipe, data, ExecutionTuning(training_workers=0, development_workers=0),
                    ctx, {"accelerator": "synthetic-cpu"}, tmp_path))
                boundary.setattr(worker, "_verify_preflight", lambda *a, **k: {})
                boundary.setattr(worker, "_loaded_source", lambda _: tmp_path)
                boundary.setattr(kernels, "FamilyKernel", cpu_kernel)
                boundary.setattr(sessions, "FamilyTrainingSession", retained_session)
                options = dict(output=guard_root, recipe_path=path, initial_state_path=guarded_initial,
                    package_dir=package_dir, trajectory_id="synthetic-trajectory", pause_after_step=1)
                if matches:
                    assert worker.run_screen_arm(**options)["status"] == "PAUSED"
                    accepted = json.loads((guard_root / "initial_development_check.json").read_text())
                    assert accepted["status"] == "VERIFIED" and accepted["optimizer_steps"] == 0
                    assert created[-1].progress.optimizer_step == 1
                else:
                    with pytest.raises(ValueError, match="Initial predictor"):
                        worker.run_screen_arm(**options)
                    assert created[-1].progress.optimizer_step == 0
                    assert not (guard_root / "last_checkpoint.pt").exists()
                assert created[-1].closed and created[-1].loaders.closed


def test_contract_rejects_counter_changes_and_unknown_fields(tmp_path):
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    try:
        path, original = contract_file(tmp_path, session)
        assert read_recipe(path)[1] == session.recipe
        retained = deepcopy(original)
        retained["initial_development_check"] = {"prediction_tolerance_eV": 1e-4, "mae_tolerance_eV": 1e-6}
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
