"""CPU addon recovery through scientific declarations and real family owners.

These synthetic graphs exercise the CPU owner/session/recovery output path, not
accepted dataset bytes, native preflight, scientific acceptance or Replay. The
test writes selected outputs and traces through SessionOutputs; no platform
operation, cloud bootstrap, real dataset ingestion or training entry is used.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from molgap import experiment_input
from molgap.experiment_family_workflow import RunContext, tensor_digest
from molgap.experiment_package import read_packaged_text, verify_experiment_source_package
from molgap.experiment_resume import (
    RESUME_BUNDLE_FILE, restore_resume_bundle, validate_resume_bundle,
)
from molgap.experiment_workflow_resume import (
    RESUME_PLAN_FORMAT, build_workflow_resume, prepare_resumed_workflow,
)
from molgap.family_training_data import TrainingData
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_runner import PROFILES, _verify_preflight, read_recipe
from molgap.family_training_session import FamilyTrainingSession
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.research_memory.trace import atomic_write, file_digest
from molgap.training_reproducibility import (
    atomic_torch_save, capture_rng_state, configure_fp32_determinism, restore_rng_state,
)
from test_experiment_lifecycle_integration import _launch_receipt
from test_experiment_workflow import _git, _write_json
from test_family_training_kernel import synthetic_batch
from test_family_training_session import assert_equal
from test_scientific_input import scientific  # Reuse the actual human-input fixture.


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    assert not torch.cuda.is_initialized(), "Use a fresh CPU-only test worker"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        patch.setenv("OMP_NUM_THREADS", "1")
        patch.setenv("MKL_NUM_THREADS", "1")
        patch.setenv("PYTHONHASHSEED", "42")
        patch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        rng, threads = capture_rng_state(), torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        precision = torch.get_float32_matmul_precision()
        patch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        patch.setattr(torch.backends.cudnn, "allow_tf32", False)
        patch.setattr(torch.backends.cudnn, "benchmark", False)
        patch.setattr(torch.backends.cudnn, "deterministic", True)
        torch.set_num_threads(1)
        try:
            yield
        finally:
            restore_rng_state(rng)
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.utils.deterministic.fill_uninitialized_memory = fill
            torch.set_float32_matmul_precision(precision)


def _synthetic_data(recipe):
    template = synthetic_batch().to_data_list()[0]

    def graph(index):
        item = template.clone()
        item.source_idx = torch.tensor([index])
        item.y = torch.tensor([3. + (index % 7 - 3) * .1])
        return item

    return TrainingData(
        [graph(i) for i in range(recipe.train_rows)],
        [graph(recipe.train_rows + i) for i in range(recipe.development_rows)],
        recipe.dataset.manifest_sha256, recipe.train_rows, recipe.development_rows,
        0, recipe.train_rows, recipe.target_mean_eV, recipe.target_std_eV,
        synthetic=True,
    )


def _session(case, output=None):
    """Bind existing owners to tiny in-memory graphs; no replacement update loop."""
    configure_fp32_determinism(case.recipe.seed)
    kernel = FamilyKernel(case.recipe, case.initial, device="cpu")
    pool = FamilyLoaders(case.recipe, case.data, ExecutionTuning(training_workers=0))
    checkpoint = pass_finished = lambda *args: None
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        contract = case.tmp / (output.name + "-training-contract.json")
        atomic_write(contract, case.contract)
        outputs = SessionOutputs(output, case.context, contract=contract,
            adapter=PROFILES[case.recipe.family], trajectory_id=case.trajectory["trajectory_id"],
            train_role_identity=case.record["train_role_identity"])
        session = FamilyTrainingSession(kernel, pool,
            checkpoint=checkpoint,
            checkpoint_builder=outputs.checkpoint_from_builder,
            selected=outputs.selected, pass_finished=outputs.pass_finished,
            batch_finished=outputs.batch_finished)
        session._test_outputs = outputs
        return session
    return FamilyTrainingSession(kernel, pool, checkpoint=checkpoint,
        selected=lambda *args: None, pass_finished=pass_finished)


def _recovery_case(scientific):
    case = scientific
    value = deepcopy(case.value)
    value.update(experiment_id="addon-recovery", logical_run_id="addon-recovery-run")
    arm = value["arms"][0]
    addon = ("k1_readout_scale" if arm["model"]["family"] == "neural_atom_k1"
             else "gptrans_readout_scale")
    arm.update(training={}, purpose="architecture_comparison")
    arm["model"]["addons"] = [{"name": addon, "version": "1", "config": None}]
    value["question"].update(changed_mechanism="readout-scale",
        question="Does the ordinary addon survive exact-run recovery?")
    path = case.repo / "experiments/addon-recovery/experiment.json"
    _write_json(path, value)
    # Match the existing synthetic reconciled-receipt helper, without an SDK.
    platform_path = case.repo / value["platform_ref"]
    platform = json.loads(platform_path.read_bytes())
    platform["kaggle"]["kernel"] = "synthetic-account/synthetic-pair-run"
    platform["kaggle"]["title"] = "Synthetic Pair Run"
    _write_json(platform_path, platform)
    contract_path = case.repo / "experiments/reference/contract.json"
    contract_record, contract_recipe = read_recipe(contract_path)
    synthetic = _synthetic_data(contract_recipe)
    source_idx = torch.cat([item.source_idx for item in synthetic.development])
    target = torch.cat([item.y for item in synthetic.development])
    contract_record["acceptance_requirements"].update(
        source_idx_sha256=tensor_digest(source_idx, role="source_idx"),
        target_sha256=tensor_digest(target, role="target"))
    _write_json(contract_path, contract_record)
    contract_sha256 = file_digest(contract_path)
    source_spec_path = case.repo / "experiments/reference/experiment_spec.json"
    source_spec = json.loads(source_spec_path.read_bytes())
    source_spec["arms"][0]["training"]["recipe"]["sha256"] = contract_sha256
    _write_json(source_spec_path, source_spec)
    from molgap.experiment_spec import ExperimentSpec
    source_spec_identity = ExperimentSpec(source_spec).identity
    plan_path = case.repo / value["arms"][0]["reference_ref"]["plan_ref"]
    plan_value = json.loads(plan_path.read_bytes())
    plan_value["spec_identity"] = source_spec_identity
    plan_value["arms"][0]["contract"] = {
        "path": contract_path.relative_to(case.repo).as_posix(), "sha256": contract_sha256}
    plan_value["arms"][0]["expected"] = contract_record["acceptance_requirements"]
    from test_comparison_readiness import _prelaunch
    from molgap.comparison_readiness import reference_bundle_digest
    bundle_path = case.repo / plan_value["arms"][0]["reference_bundle"]["path"]
    bundle = json.loads(bundle_path.read_bytes())
    prediction_manifest_path = case.repo / "experiments/reference/prediction_manifest.json"
    prediction_manifest = json.loads(prediction_manifest_path.read_bytes())
    prediction_manifest.update(source_idx_sha256=contract_record["acceptance_requirements"]["source_idx_sha256"],
                               target_sha256=contract_record["acceptance_requirements"]["target_sha256"])
    predictions_path = case.repo / "experiments/reference/predictions.json"
    predictions = json.loads(predictions_path.read_bytes())
    predictions.update(source_idx_sha256=prediction_manifest["source_idx_sha256"],
                       target_sha256=prediction_manifest["target_sha256"])
    _write_json(predictions_path, predictions)
    prediction_manifest["prediction_sha256"] = file_digest(predictions_path)
    _write_json(prediction_manifest_path, prediction_manifest)
    bundle["prediction_manifest"] = prediction_manifest
    _write_json(bundle_path, bundle)
    prelaunch_path = case.repo / plan_value["arms"][0]["comparison_prelaunch"]["path"]
    _write_json(prelaunch_path, _prelaunch(bundle))
    plan_value["arms"][0]["reference_bundle"]["sha256"] = file_digest(bundle_path)
    plan_value["arms"][0]["comparison_prelaunch"]["sha256"] = file_digest(prelaunch_path)
    plan_value["arms"][0]["reference_artifacts"]["prediction_manifest"] = {
        "path": prediction_manifest_path.relative_to(case.repo).as_posix(),
        "sha256": file_digest(prediction_manifest_path)}
    plan_value["arms"][0]["reference_artifacts"]["predictions"] = {
        "path": predictions_path.relative_to(case.repo).as_posix(),
        "sha256": file_digest(predictions_path)}
    assert reference_bundle_digest(bundle) == json.loads(prelaunch_path.read_bytes())["reference_bundle_sha256"]
    _write_json(plan_path, plan_value)
    _git(case.repo, "add", ".")
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic addon recovery input")

    prepared_report = experiment_input.prepare_experiment_input(path, case.repo)
    assert prepared_report["status"] == "PREPARED_FOR_PLATFORM", prepared_report
    assert prepared_report["submitted"] is False
    _, spec, plan, paths = experiment_input.resolve_experiment_input(path, case.repo)
    prepared = Path(prepared_report["prepared"])
    package = prepared / "package"
    manifest = verify_experiment_source_package(package, case.repo)
    arm = spec.to_dict()["arms"][0]
    contract = read_packaged_text(package, plan["arms"][0]["recipe"]).encode()
    record, recipe = read_recipe(case.repo / plan["arms"][0]["recipe"])
    assert recipe.mode == "composed" and list(recipe.addons) == arm["addons"]
    assert arm["addons"][0]["name"] == addon
    assert recipe.batch_size == 128 and recipe.optimizer_steps == 4
    assert json.loads(contract) == record
    launch = json.loads((prepared / "source_dataset/experiment_launch.json").read_bytes())
    receipt = _launch_receipt(spec, package, manifest["package_identity"], case.repo)
    context = RunContext.for_training(spec, package,
        expected_package_identity=manifest["package_identity"], arm_id=arm["arm_id"],
        account=launch["account"], run_reference=launch["run_reference"])
    trajectory_path = prepared / "source_dataset/prospective" / arm["arm_id"] / "trajectory.json"
    result = SimpleNamespace(repo=case.repo, path=path, value=value, spec=spec, plan=plan,
        run=paths.run_directory(value["logical_run_id"], value["attempt_id"]),
        prepared=prepared, package=package, manifest=manifest, receipt=receipt,
        recipe=recipe, record=record, contract=contract, context=context,
        initial=prepared / "source_dataset/initial_states" / (arm["arm_id"] + ".pt"),
        trajectory=json.loads(trajectory_path.read_bytes()), trajectory_raw=trajectory_path.read_bytes(),
        tmp=case.tmp, data=_synthetic_data(recipe), source=case.tmp / "synthetic-session")
    session = _session(result, result.source)
    result.outputs = session._test_outputs
    addon_parameters = dict(session.kernel.model._molgap_substrate.named_addon_parameters())
    assert addon_parameters
    named = dict(session.kernel.model.named_parameters())
    result.addon_keys = [key for key, param in named.items()
                         if any(param is p for p in addon_parameters.values())]
    result.addon_ids = [i for i, param in enumerate(session.kernel.optimizer.param_groups[0]["params"])
                        if any(param is p for p in addon_parameters.values())]
    initial_gains = {key: named[key].detach().clone() for key in result.addon_keys}
    pauses = iter([False, True])
    result.outputs.logger.start(session.progress)
    progress = session.run(should_pause=lambda: next(pauses))
    assert progress.status == "paused" and progress.optimizer_step == 2
    raw_saved = torch.load(result.source / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    assert raw_saved.pop("context") == context.to_dict()
    result.saved = result.outputs.retained_state()
    assert result.saved["rng_state"]["cuda"] == []
    assert result.saved["cursor"]["pass_index"] == 1
    assert result.saved["cursor"]["next_batch"] == 0
    assert result.saved["sample_presentations"] == 256
    for key, parameter_id in zip(result.addon_keys, result.addon_ids):
        assert not torch.equal(result.saved["model"][key], initial_gains[key])
        moments = result.saved["optimizer"]["state"][parameter_id]
        assert int(moments["step"]) == 2
        assert torch.count_nonzero(moments["exp_avg"])
    trace = json.loads((result.source / "canonical_trace.json").read_bytes())
    rows = [row for row in trace["observations"] if row["event"] == "observation"]
    assert [(row["optimizer_step"], row["sample_presentations"]) for row in rows] == [(2, 256)]
    selected = torch.load(result.source / "selected_model.pt", map_location="cpu", weights_only=True)
    assert selected["optimizer_step"] == 2 and selected["weights"] == "live"
    result.bundle = case.tmp / "resume-bundle"
    result.export = build_workflow_resume(spec, case.repo, prepared, package,
        manifest["package_identity"], receipt, result.bundle,
        arm_id=context.arm_id, source_output=result.source)
    assert result.export["status"] == "RESUME_VERIFIED", result.export
    return result


def _checked(case, bundle=None):
    return validate_resume_bundle(bundle or case.bundle, case.spec,
        arm_id=case.context.arm_id, context=case.context, trajectory=case.trajectory_raw)


def _assert_real_family_addon_bundle_restores_exact_session_and_next_updates(recovery_case, record_property):
    case = recovery_case
    checked = _checked(case)
    assert checked["owner"]["status"] == "SHARED_RESUME_VERIFIED"
    assert checked["owner"]["recipe_identity"] == case.recipe.identity
    assert checked["manifest"]["source"] == {
        "commit": case.manifest["source_commit"],
        "archive_sha256": case.manifest["archive_sha256"],
        "package_identity": case.manifest["package_identity"],
    }
    assert checked["manifest"]["run"]["logical_run_id"] == case.context.logical_run_id
    restored = case.tmp / "restored-session"
    restore_resume_bundle(case.bundle, restored, case.spec,
        arm_id=case.context.arm_id, context=case.context, trajectory=case.trajectory_raw)
    assert file_digest(restored / "last_checkpoint.pt") == file_digest(case.source / "last_checkpoint.pt")
    saved = torch.load(restored / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved.pop("context") == case.context.to_dict()
    resumed = _session(case, restored)
    resumed_outputs = resumed._test_outputs
    saved = resumed_outputs.retained_state()
    resumed.load_state_dict(saved)
    resumed_outputs.logger.start(resumed.progress)
    assert_equal(case.saved, resumed.state_dict())
    progress = resumed.run()
    final = deepcopy(resumed.state_dict())
    runtime = {"platform": case.context.platform, "account": case.context.account,
        "precision": "fp32", "source_commit": case.context.source_commit,
        "source_archive_sha256": case.context.source_archive_sha256}
    costs = [{"metric": "wall_seconds", "unit": "seconds", "value": 1.,
        "status": "measured", "semantics": "process_wall", "hardware": "synthetic-cpu"},
        {"metric": "device_seconds", "unit": "seconds", "value": None,
        "status": "missing", "semantics": "allocated_device", "hardware": "synthetic-cpu"}]
    resumed_report = resumed_outputs.complete(progress,
        runtime=runtime, costs=costs)
    assert resumed_report["status"] == "MECHANICALLY_VERIFIED"
    trace = json.loads((restored / "canonical_trace.json").read_bytes())
    rows = [row for row in trace["observations"] if row["event"] == "observation"]
    assert [(row["optimizer_step"], row["sample_presentations"]) for row in rows] == [(2, 256), (4, 512)]
    control = _session(case, case.tmp / "control-session")
    control_outputs = control._test_outputs
    control_outputs.logger.start(control.progress)
    assert control.run() == progress
    control_report = control_outputs.complete(progress,
        runtime=runtime, costs=costs)
    assert control_report["status"] == "MECHANICALLY_VERIFIED"
    assert progress.optimizer_step == 4 and progress.sample_presentations == 512
    assert_equal(final, control.state_dict())
    assert (restored / "output_manifest.json").is_file()
    assert (case.tmp / "control-session/output_manifest.json").is_file()
    assert not (case.run / "operations").exists()
    record_property("addon_recovery", json.dumps({
        "family": case.context.family_name, "addon_keys": case.addon_keys,
        "spec_identity": case.spec.identity, "package_identity": case.manifest["package_identity"],
        "owner": checked["owner"], "exact_state_and_next_updates": True,
        "native_qualification": "NOT_EVALUATED", "scientific_acceptance": "NOT_EVALUATED"}))


def _assert_completed_addon_preparation_continues_original_run_across_head_without_inputs(
        recovery_case, monkeypatch):
    case = recovery_case
    frozen = {str(p): file_digest(p) for base in (case.run / "inputs", case.prepared)
              for p in base.rglob("*") if p.is_file()}
    note = case.repo / "ADDON_RECOVERY_NOTES.md"
    atomic_write(note, b"Unrelated synthetic documentation change.\n")
    _git(case.repo, "add", "--", note.name)
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Unrelated fixture documentation")
    assert _git(case.repo, "rev-parse", "HEAD") != case.manifest["source_commit"]

    def compiler_is_forbidden(*args, **kwargs):
        raise AssertionError("Continuation must use the original frozen addon run")

    monkeypatch.setattr(experiment_input, "_compile_scientific_input", compiler_is_forbidden)
    moved = []
    try:
        for source in (case.repo / "experiments/reference", case.repo / "experiments/addon-recovery/generated",
                       Path(case.value["arms"][0]["initial_state_ref"])):
            target = source.with_name(source.name + "-unavailable")
            assert source.resolve().is_relative_to(case.tmp.resolve())
            assert target.resolve().is_relative_to(case.tmp.resolve()) and not target.exists()
            source.rename(target)
            moved.append((source, target))
        continued = experiment_input.prepare_experiment_input(case.path, case.repo)
        assert continued["status"] == "PREPARED_FOR_PLATFORM" and continued["reused_preparation"]
        assert continued["package_identity"] == case.manifest["package_identity"]
        assert continued["source_commit"] == case.manifest["source_commit"]
        _, spec, _, _ = experiment_input.resolve_experiment_input(case.path, case.repo)
        assert spec.identity == case.spec.identity
        resumed_root = case.tmp / "resumed-preparation"
        resumed = prepare_resumed_workflow(spec, case.repo, case.prepared, case.package,
            case.manifest["package_identity"], case.receipt, resumed_root,
            plan={"format": RESUME_PLAN_FORMAT, "spec_identity": spec.identity,
                  "arms": {case.context.arm_id: case.export["manifest"]}})
        assert resumed["status"] == "PREPARED_FOR_PLATFORM", resumed
        assert not resumed["submitted"] and not resumed["prospective_published"]
        assert resumed["source_commit"] == case.manifest["source_commit"]
        assert resumed["package_identity"] == case.manifest["package_identity"]
        staged = resumed_root / "source_dataset"
        assert (staged / "prospective" / case.context.arm_id / "trajectory.json").read_bytes() == case.trajectory_raw
        bundle = staged / "resume" / case.context.arm_id / RESUME_BUNDLE_FILE
        assert _checked(case, bundle)["owner"]["status"] == "SHARED_RESUME_VERIFIED"
        assert frozen == {str(p): file_digest(p) for base in (case.run / "inputs", case.prepared)
                          for p in base.rglob("*") if p.is_file()}
        assert not (case.run / "operations").exists()
    finally:
        for source, target in reversed(moved):
            target.rename(source)


def _assert_altered_addon_checkpoint_and_wrong_run_context_are_rejected(recovery_case):
    case = recovery_case
    pointer = case.bundle / "files/last_checkpoint.pt"
    original = pointer.read_bytes()
    altered = torch.load(pointer, map_location="cpu", weights_only=True)
    altered["model"][case.addon_keys[0]].add_(1.)
    try:
        atomic_torch_save(pointer, altered)
        with pytest.raises(ValueError, match="Resume artifact hash mismatch: checkpoint"):
            _checked(case)
    finally:
        atomic_write(pointer, original)
    with pytest.raises(ValueError, match="context identity mismatch"):
        validate_resume_bundle(case.bundle, case.spec,
            context=replace(case.context, logical_run_id="unrelated-run"))
    fresh = _session(case)
    before = deepcopy(fresh.state_dict())
    missing_moment = deepcopy(case.saved)
    del missing_moment["optimizer"]["state"][case.addon_ids[0]]
    try:
        with pytest.raises(ValueError, match="moments"):
            fresh.load_state_dict(missing_moment)
        assert_equal(before, fresh.state_dict())
    finally:
        fresh.close()
    assert _checked(case)["status"] == "RESUME_VERIFIED"


def _assert_cpu_transport_does_not_supply_native_release_evidence(recovery_case):
    case = recovery_case
    assert not (case.source / "runtime_provenance.json").exists()
    assert not (case.source / "runtime_certificate.json").exists()
    with pytest.raises((ValueError, FileNotFoundError), match="runtime_provenance"):
        _verify_preflight(case.source, case.recipe, {})


def test_scientific_addon_recovery_uses_its_own_frozen_checkout(scientific, record_property):
    # Readiness correctly rejects a validator loaded from the host checkout.
    # Select the independent fixture's code in a fresh interpreter, just as the
    # existing scientific-input and extracted-distribution tests do. Keep the
    # source-origin guard intact rather than patching it for this fixture.
    program = r'''
import importlib.util, json, pathlib, sys
from types import SimpleNamespace
repo, config, scratch, test_file = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(repo / "src"))
sys.path.insert(1, str(test_file.parent))
import pytest
loaded = importlib.util.spec_from_file_location("_addon_recovery_driver", test_file)
driver = importlib.util.module_from_spec(loaded)
loaded.loader.exec_module(driver)
# In this dependency profile, first-use Rich style initialization consumes one
# global Python getrandbits(24) call during the first model construction. Resolve
# that presentation-only dependency before _session seeds either comparison.
# Otherwise a cold interrupted group and warm control start from different RNGs.
# Keep the full Python/NumPy/Torch RNG equality assertions below unchanged.
if importlib.util.find_spec("rich") is not None:
    import rich.style
case = SimpleNamespace(repo=repo, path=config, value=json.loads(config.read_bytes()), tmp=scratch)
case = driver._recovery_case(case)
observations, checks = [], []
driver._assert_real_family_addon_bundle_restores_exact_session_and_next_updates(
    case, lambda name, value: observations.append({"name": name, "value": json.loads(value)}))
checks.append("actual_family_state_and_next_updates")
with pytest.MonkeyPatch.context() as patch:
    driver._assert_completed_addon_preparation_continues_original_run_across_head_without_inputs(case, patch)
checks.append("addon_cross_head_frozen_reuse_and_resumed_preparation")
driver._assert_altered_addon_checkpoint_and_wrong_run_context_are_rejected(case)
checks.append("tampered_checkpoint_context_and_missing_optimizer_moment_rejected")
driver._assert_cpu_transport_does_not_supply_native_release_evidence(case)
checks.append("native_release_gate_remains_closed")
for name, module in tuple(sys.modules.items()):
    if name == "molgap" or name.startswith("molgap."):
        file = getattr(module, "__file__", None)
        if file is not None:
            assert pathlib.Path(file).resolve().is_relative_to((repo / "src").resolve()), (name, file)
print(json.dumps({"status": "CPU_ADDON_RECOVERY_ACCEPTED", "family": case.context.family_name,
                  "checks": checks, "observations": observations, "platform_requests": 0,
                  "native_qualification": "NOT_EVALUATED", "scientific_acceptance": "NOT_EVALUATED"}))
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(scientific.repo), str(scientific.path),
         str(scientific.tmp), str(Path(__file__).resolve())], cwd=scientific.tmp,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["status"] == "CPU_ADDON_RECOVERY_ACCEPTED"
    assert len(report["checks"]) == 4
    record_property("isolated_addon_recovery", json.dumps(report))
