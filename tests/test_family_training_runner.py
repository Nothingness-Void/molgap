"""Synthetic CPU contracts/output fixtures; not native or scientific qualification."""
from copy import deepcopy
from dataclasses import replace
import json

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
def test_actual_session_output_translation_and_cpu_mechanical_inspection(tmp_path, cpu_replay, initial_states, family, development):
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
    ctx = replace(context(family, path), experiment_id=declaration["experiment_id"],
        logical_run_id=declaration["logical_run_id"], spec_identity=spec.identity,
        arm_identity=canonical_fingerprint(arm), training_recipe_sha256=recipe_source_sha)
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
    trajectory = {"trajectory_id": "synthetic-trajectory", "record_mode": "prospective",
        "state_at_start": {"source_commit": ctx.source_commit, "source_config_identity": ctx.arm_identity},
        "actions": [{"action_id": "A1", "run_ids": [ctx.logical_run_id + ":reference"], "attempt_ids": ["v1"]}],
        "decision": {"outcome": "ACTIVE"}}
    bundle = tmp_path / "bundle"
    create_resume_bundle(outputs.root, ctx, trajectory, bundle)
    verified = validate_resume_bundle(bundle, spec, context=ctx, trajectory=trajectory)
    assert verified["status"] == "RESUME_VERIFIED"
    assert verified["owner"]["optimizer_steps"] == 2
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
    else:
        assert telemetry["development_mae_eV"] is telemetry["best_development_mae_eV"] is None
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


def test_contract_rejects_counter_changes_and_unknown_fields(tmp_path):
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    try:
        path, original = contract_file(tmp_path, session)
        assert read_recipe(path)[1] == session.recipe
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
