"""Synthetic CPU contracts/output fixtures; not native or scientific qualification."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.experiment_family_workflow import RunContext, inspect_output, tensor_digest
from molgap.experiment_package import _name
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_runner import INTERFACES, PROFILES, family_arm_bindings, read_recipe
from molgap.training_reproducibility import sha256_file, validate_accelerator_request
from test_family_training_session import events, make_session
from test_experiment_spec import payload



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


@pytest.mark.parametrize("platform", ["local", "kaggle", "ims", "scnet", "colab"])
def test_platform_declaration_does_not_change_the_training_owner(payload, platform):
    payload["platform"]["name"] = platform
    assert ExperimentSpec(payload).to_dict()["platform"]["name"] == platform
    for family in INTERFACES:
        adapter = training_adapter({"family": {"name": family, "version": INTERFACES[family][1]}, "addons": []})
        assert adapter.module == "molgap.family_training_runner"


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
def test_actual_session_output_translation_and_cpu_mechanical_inspection(tmp_path, family, development):
    session = make_session(family, development=development, schedule_kind="cosine_step", events=events())
    path, record = contract_file(tmp_path, session)
    ctx = context(family, path)
    outputs = SessionOutputs(tmp_path / "output", ctx, contract=path, adapter=PROFILES[family],
        trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"])
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    session.batch_finished = outputs.batch_finished
    outputs.logger.start(session.progress)
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


def test_contract_accepts_only_valid_frozen_trace_semantics(tmp_path):
    from molgap.research_memory.schemas import TRACE_MANIFEST_PLAN_FIELDS
    from molgap.research_memory.terminal_wiring import build_default_trace_manifest

    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    try:
        path, record = contract_file(tmp_path, session)
        manifest = build_default_trace_manifest(tmp_path,
            {"trajectory_id": "T-plan", "state_at_start": {}}, {"run_id": "run"}, {"observations": []})
        manifest["presentation_semantics_ref"] = "synthetic-presentation-contract"
        record["trace_manifest_plan"] = {key: manifest[key] for key in TRACE_MANIFEST_PLAN_FIELDS}
        path.write_text(json.dumps(record), encoding="utf-8")
        assert read_recipe(path)[1] == session.recipe
        record["trace_manifest_plan"]["reference_id"] = "invented-reference"
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="exactly"):
            read_recipe(path)
    finally:
        session.close()


def test_reviewed_python_io_source_does_not_allow_artifact_directories():
    assert _name("src/molgap/family_training_outputs.py") == "src/molgap/family_training_outputs.py"
    assert _name("src/molgap/other_outputs.py") == "src/molgap/other_outputs.py"
    for path in ("experiments/example/outputs/model.py", "data/family_training_outputs.py"):
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
