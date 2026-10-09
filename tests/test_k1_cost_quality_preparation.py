"""Synthetic preparation/pair lifecycle only; no model execution or remote calls."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from molgap.experiment_execution import training_adapter
from molgap.experiment_family_workflow import check_acceptance_plan
from molgap.experiment_prospective import plan_prospective
from molgap.experiment_spec import ExperimentSpec, _canonical


ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.util.spec_from_file_location("cost_quality_preparation",
    ROOT / "experiments/pcqm_k1_t4_cost_quality/prepare_training.py")
prep = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(prep)


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setattr(prep, "inspect_frozen_state_artifact", lambda *a, **k: {
        "state_sha256": prep.STATE_SHA, "file_sha256": "f" * 64, "device": "cpu"})
    return prep.build_inputs(ROOT, source_commit="1" * 40, initial_state=prep.STATE)


def test_roles_modes_frozen_exposure_budget_and_identity(records):
    spec = ExperimentSpec(records["experiment_spec.json"])
    reference, candidate = spec.to_dict()["arms"]
    assert (reference["scientific_role"], candidate["scientific_role"]) == ("reference", "candidate")
    assert training_adapter(reference).mode(reference) == "mean2"
    assert training_adapter(candidate).mode(candidate) == "reference"
    assert reference["addons"][0]["config"] == {}
    assert candidate["addons"] == []
    for key in ("base", "data", "initialization"):
        assert reference[key] == candidate[key]
    for key in ("sampler", "transform"):
        assert reference["training"][key] == candidate["training"][key]
    assert reference["training"]["objective"]["sha256"] == prep.canonical_fingerprint({
        "base_loss": "normalized-gap-l1", "forward_passes": 2,
        "reduction": "arithmetic-mean-of-two-dropout-bearing-L1-losses",
        "optimizer_updates_per_batch": 1, "consistency_penalty": False,
        "batchnorm_updates_per_batch": 2})
    assert reference["training"]["objective"]["sha256"] != candidate["training"]["objective"]["sha256"]
    assert records["training_recipe_mean2.json"]["training_recipe"] == records["training_recipe_single.json"]["training_recipe"]
    for arm in ("mean2", "single"):
        recipe = records[f"training_recipe_{arm}.json"]
        assert recipe["allocation_wall_limit_seconds"] == 14400
        assert "allocation_wall_limit_seconds" not in recipe["training_recipe"]
        assert recipe["acceptance_requirements"]["optimizer_steps"] == 31240
        assert recipe["acceptance_requirements"]["sample_presentations"] == 3998720
        assert not recipe["training_recipe"]["ema"]
        plan = records[f"training_plan_{arm}.json"]
        assert plan["trajectory"]["state_at_start"]["reference_ids"] == []
        assert plan["trajectory"]["decision"]["next_allowed_actions"] == []
        assert plan["costs"][0]["measurement"]["device_hours"] == {"status": "estimated", "value": 4.0}
    assert records["preparation_report.json"]["training_authorized"] is False


def test_prospective_planner_freezes_same_run_without_accepting_reference(tmp_path, monkeypatch, records):
    import molgap.experiment_prospective as owner
    spec = ExperimentSpec(records["experiment_spec.json"])
    for binding in spec.to_dict()["prospective"]["arms"]:
        path = tmp_path / binding["plan_spec_ref"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(records[f"training_plan_{binding['arm_id']}.json"]).encode("utf-8"))
    captured = []
    def plan_many(root, plans):
        captured.extend(copy.deepcopy(plans))
        return {"status": "PLANNED", "batch": {"size": 2}, "results": [
            {"status": "PLANNED", "trajectory_id": p["spec"]["trajectory"]["trajectory_id"], "path": p["output"]} for p in plans]}
    monkeypatch.setattr(owner, "plan_many", plan_many)
    monkeypatch.setattr(owner, "rebuild_research_memory", lambda root: {})
    report, code = plan_prospective(spec, tmp_path)
    assert code == 0 and report["training_authorized"] is False
    pair = [p["spec"]["trajectory"]["state_at_start"]["same_run_replay"] for p in captured]
    assert [p["comparison_role"] for p in pair] == ["reference", "candidate"]
    assert all(p["reference_arm_id"] == "mean2" for p in pair)
    assert pair[0]["reference_trajectory_ref"] == pair[1]["reference_trajectory_ref"]


def test_acceptance_schema_has_only_prospective_pair_and_target_context(records):
    plan = records["family_acceptance_plan.json"]
    assert plan["format"] == "molgap-family-same-run-acceptance-plan-v1"
    assert set(plan) == {"format", "spec_identity", "arms"}
    for entry in plan["arms"]:
        assert set(entry) == {"arm_id", "adapter", "expected", "contract", "target_manifest"}
        target = ROOT / entry["target_manifest"]["path"]
        assert prep.file_digest(target) == entry["target_manifest"]["sha256"]
        assert prep.read(target)["development_target_sha256"] == entry["expected"]["target_sha256"]
    assert records["preparation_report.json"]["strict_ready"] is False


def _stage_acceptance(records, tmp_path):
    for arm in ("mean2", "single"):
        path = tmp_path / prep.REL / f"training_recipe_{arm}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(records[f"training_recipe_{arm}.json"]).encode("utf-8"))
    pointer = records["family_acceptance_plan.json"]["arms"][0]["target_manifest"]
    target = tmp_path / pointer["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / pointer["path"]).read_bytes())
    return target


def test_shared_same_run_acceptance_integration(records, tmp_path):
    _stage_acceptance(records, tmp_path)
    report = check_acceptance_plan(ExperimentSpec(records["experiment_spec.json"]), tmp_path,
                                  records["family_acceptance_plan.json"])
    assert report["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    assert report["reference_authority"] == "prospective_same_run_only; terminal reference not yet accepted"
    assert report["trainer_execution"] == "NOT_VERIFIED"


def test_missing_pair_declaration_rejected(records, tmp_path):
    payload = copy.deepcopy(records["experiment_spec.json"])
    del payload["prospective"]["same_run_replay"]
    spec = ExperimentSpec(payload)
    plan = copy.deepcopy(records["family_acceptance_plan.json"])
    plan["spec_identity"] = spec.identity
    with pytest.raises(ValueError, match="requires a prospective Spec pair"):
        check_acceptance_plan(spec, tmp_path, plan)


@pytest.mark.parametrize("field", ["initialization", "rows", "target_sha", "target_bytes"])
def test_same_run_identity_and_target_drift_rejected(records, tmp_path, field):
    payload = copy.deepcopy(records["experiment_spec.json"])
    plan = copy.deepcopy(records["family_acceptance_plan.json"])
    target = _stage_acceptance(records, tmp_path)
    if field == "initialization":
        payload["arms"][1]["initialization"]["state_sha256"] = "f" * 64
    elif field == "rows":
        payload["arms"][1]["data"]["roles"][1]["row_order_sha256"] = "f" * 64
    elif field == "target_sha":
        plan["arms"][1]["target_manifest"]["sha256"] = "f" * 64
    else:
        target.write_bytes(target.read_bytes() + b" ")
    spec = ExperimentSpec(payload)
    plan["spec_identity"] = spec.identity
    report = check_acceptance_plan(spec, tmp_path, plan)
    assert report["status"] == "BLOCKED"
    assert any(a["blockers"] for a in report["arms"])


@pytest.mark.parametrize("field", ["exposure", "optimizer", "target_identity"])
def test_repinned_contract_drift_still_rejected(records, tmp_path, field):
    payload = copy.deepcopy(records["experiment_spec.json"])
    plan = copy.deepcopy(records["family_acceptance_plan.json"])
    _stage_acceptance(records, tmp_path)
    recipe = copy.deepcopy(records["training_recipe_single.json"])
    if field == "exposure":
        recipe["acceptance_requirements"]["development_rows"] -= 1
    elif field == "optimizer":
        recipe["training_recipe"]["learning_rate"] *= 2
    else:
        recipe["acceptance_requirements"]["target_sha256"] = "f" * 64
    plan["arms"][1]["expected"] = copy.deepcopy(recipe["acceptance_requirements"])
    digest = prep.canonical_fingerprint(recipe)
    payload["arms"][1]["training"]["recipe"]["sha256"] = digest
    plan["arms"][1]["contract"]["sha256"] = digest
    (tmp_path / plan["arms"][1]["contract"]["path"]).write_bytes(_canonical(recipe).encode("utf-8"))
    spec = ExperimentSpec(payload)
    plan["spec_identity"] = spec.identity
    assert check_acceptance_plan(spec, tmp_path, plan)["status"] == "BLOCKED"


def test_local_stage_is_fresh_unpublished_draft(records, tmp_path):
    output = tmp_path / "draft"
    prep.stage_inputs(records, output)
    assert (output / "experiment_spec.json").read_bytes() == _canonical(records["experiment_spec.json"]).encode("utf-8")
    assert not list(output.rglob("trajectory.json"))
    assert not (tmp_path / "research_memory").exists()
    with pytest.raises(FileExistsError):
        prep.stage_inputs(records, output)


def test_initial_state_mismatch_stops_before_staging(monkeypatch):
    def reject(*args, **kwargs):
        assert kwargs["expected_state_sha256"] == prep.STATE_SHA
        raise ValueError("Frozen initialization tensor SHA differs from Spec")
    monkeypatch.setattr(prep, "inspect_frozen_state_artifact", reject)
    with pytest.raises(ValueError, match="tensor SHA"):
        prep.build_inputs(ROOT, source_commit="1" * 40, initial_state=prep.STATE)


@pytest.fixture
def release_inputs(tmp_path, monkeypatch):
    def write(ref, value):
        path = tmp_path / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(value).encode("utf-8"))
        return {"path": ref, "sha256": prep.file_digest(path)}

    decision = write(prep.CLEAN + "/terminal_decision.md", "NO_TRAIN descriptive only")
    acceptance = write(prep.CLEAN + "/acceptance.json", {
        "format": "molgap-local-frozen-diagnostic-acceptance-v1", "outcome": {
            "execution_status": "complete_no_training", "artifact_status": "local_hash_verified",
            "scientific_status": "NO_TRAIN"}})
    hashes = {p["path"]: p["sha256"] for p in (decision, acceptance)}
    terminal = write(prep.CLEAN + "/terminal.json", {
        "format": "molgap-rml-terminal-package-v1", "trajectory_id": "clean-fit",
        "acceptance_ref": acceptance["path"], "artifact_hashes": hashes,
        "decision": {"outcome": "NO_TRAIN", "decision_ref": decision["path"]}})
    finalization = write(prep.CLEAN + "/closure_receipt.json", {
        "format": "molgap-rml-finalization-v1", "trajectory_id": "clean-fit",
        "finalization_id": "finalize-test", "finalized_at": "2026-10-09", "outcome": "NO_TRAIN",
        "input_artifact_hashes": hashes, "published_hashes": {"terminal_input.json":
            prep.hashlib.sha256(prep.json_bytes(prep.read(tmp_path / terminal["path"]))).hexdigest()}})
    profile = write(prep.REL + "/profile/acceptance.json", {"fixture": "synthetic owner receipt"})
    release = {"format": prep.RELEASE_FORMAT, "controller": "human-controller",
        "approved_by": "test-human", "approved_at": "2026-10-09", "source_commit": "1" * 40,
        "action": "TRAIN_PAIR_100K", "allowed_actions": ["TRAIN_PAIR_100K"],
        "budget": {"allocated_t4_device_hours": 8, "wall_seconds": 14400},
        "profile_acceptance": profile, "clean_fit": {"terminal": terminal, "acceptance": acceptance,
            "decision": decision, "finalization": finalization,
            "assessment": "NO_URGENT_FITTING_FAILURE",
            "rationale": "Human reviewed the bounded descriptive NO_TRAIN decision; no urgent fit repair precedes the pair."}}
    release_ref = prep.REL + "/parent_release.json"
    write(release_ref, release)
    calls = []
    def owner(root, path):
        calls.append((root, path))
        return {"status": "ACCEPTED", "native_step_reduction": .3}
    monkeypatch.setattr(prep, "_validate_profile_release", owner)
    return tmp_path, release_ref, release, write, calls


def test_release_bound_fresh_plans_and_rehashed_spec(release_inputs, monkeypatch):
    root, ref, release, write, calls = release_inputs
    for name in ("experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json",
                 "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json",
                 "experiments/pcqm_k1_fusion_distillation/training_recipe_distill_weak.json",
                 prep.REL + "/protocol.md", "src/molgap/k1_screen_training.py",
                 "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(prep, "inspect_frozen_state_artifact", lambda *a, **k: {
        "state_sha256": prep.STATE_SHA, "file_sha256": "f" * 64, "device": "cpu"})
    records = prep.build_inputs(root, source_commit="1" * 40, initial_state=prep.STATE,
                               parent_release_file=Path(ref))
    assert len(calls) == 1
    spec = ExperimentSpec(records["experiment_spec.json"])
    for arm in spec.to_dict()["prospective"]["arms"]:
        plan = records[f"training_plan_{arm['arm_id']}.json"]
        assert plan["decision_state"]["chosen_action"] == "TRAIN_PAIR_100K"
        assert plan["trajectory"]["decision"]["next_allowed_actions"] == ["A001"]
        assert plan["trajectory"]["actions"][0]["type"] == "paired100k_after_parentrelease"
        assert ref in plan["trajectory"]["state_at_start"]["contract_refs"]
        assert plan["parent_release"]["sha256"] == prep.file_digest(root / ref)
        assert arm["plan_spec_sha256"] == prep.canonical_fingerprint(plan)
    assert records["workflow_plan.json"]["spec_identity"] == spec.identity
    assert records["family_acceptance_plan.json"]["spec_identity"] == spec.identity
    assert records["policy.json"]["approval"]["authority_ref"] == ref
    assert records["preparation_report.json"]["training_authorized"] is False
    assert records["preparation_report.json"]["prospective_published"] is False
    assert not list(root.rglob("trajectory.json"))
    # Nearest planner preparation only: verify binding without any publication.
    from molgap.research_memory.plan import _PlanningSnapshot, _prepare_plan, _source_pointers
    from molgap.research_memory.discovery import DiscoveredRecords
    for name in ("role_plan.json", "training_recipe_mean2.json", "training_recipe_single.json"):
        write(prep.REL + "/" + name, records[name])
    write(prep.REL + "/evidence_review.md", "synthetic evidence review")
    discovered = DiscoveredRecords((), (), (), (), (), ())
    plan = records["training_plan_mean2.json"]
    pointers = _source_pointers(root, plan["trajectory"], discovered)
    snapshot = _PlanningSnapshot(discovered, "1" * 40, [], [prep.PRIOR], frozenset(),
        frozenset(), [records["policy.json"]], {p: prep.file_digest(root / p) for p in pointers})
    prepared = _prepare_plan(root, plan, prep.REL + "/test-unpublished", _snapshot=snapshot)
    frozen = json.loads(prepared["files"]["trajectory.json"])
    assert frozen["decision_state"]["source_hashes"][ref] == plan["parent_release"]["sha256"]
    assert ref in frozen["state_at_start"]["contract_refs"]
    assert frozen["decision_state"]["chosen_action"] == "TRAIN_PAIR_100K"
    assert not prepared["destination"].exists()
    write(prep.REL + "/training_plan_mean2.json", {})
    with pytest.raises(ValueError, match="unpublished fresh"):
        prep.build_inputs(root, source_commit="1" * 40, initial_state=prep.STATE, parent_release_file=Path(ref))


@pytest.mark.parametrize("failure", ["tampered_sha", "tampered_profile", "raw_completion", "true_boolean", "wrong_budget",
    "wrong_action", "extra_action", "missing_profile", "missing_decision", "missing_terminal",
    "missing_finalization", "missing_acceptance", "missing_assessment", "boolean_assessment",
    "empty_rationale", "nonhuman", "source_commit", "unfinalized", "receipt_pin", "terminal_pin", "remote_pin"])
def test_release_fail_closed_independently(release_inputs, failure):
    root, ref, release, write, calls = release_inputs
    clean = release["clean_fit"]
    if failure == "tampered_sha":
        (root / clean["decision"]["path"]).write_bytes(b"tampered")
    elif failure == "tampered_profile":
        (root / release["profile_acceptance"]["path"]).write_bytes(b"tampered")
    elif failure == "raw_completion":
        release = {"status": "complete", "accepted": True}
    elif failure == "true_boolean":
        release = True
    elif failure == "wrong_budget":
        release["budget"]["allocated_t4_device_hours"] = 9
    elif failure == "wrong_action":
        release["action"] = "TRAIN_FULL"
    elif failure == "extra_action":
        release["allowed_actions"].append("TRAIN_FULL")
    elif failure == "missing_profile":
        del release["profile_acceptance"]
    elif failure.startswith("missing_"):
        del clean[failure.removeprefix("missing_")]
    elif failure == "boolean_assessment":
        clean["assessment"] = True
    elif failure == "empty_rationale":
        clean["rationale"] = " "
    elif failure == "nonhuman":
        release["controller"] = "agent"
    elif failure == "source_commit":
        release["source_commit"] = "2" * 40
    elif failure in ("unfinalized", "receipt_pin", "terminal_pin"):
        receipt = prep.read(root / clean["finalization"]["path"])
        if failure == "unfinalized":
            del receipt["finalized_at"]
        elif failure == "receipt_pin":
            receipt["input_artifact_hashes"][clean["decision"]["path"]] = "f" * 64
        else:
            receipt["published_hashes"]["terminal_input.json"] = "f" * 64
        clean["finalization"] = write(clean["finalization"]["path"], receipt)
    elif failure == "remote_pin":
        release["profile_acceptance"]["path"] = "https://example.org/acceptance.json"
    write(ref, release)
    with pytest.raises(ValueError):
        prep.validate_parent_release(root, Path(ref), source_commit="1" * 40)
    assert calls == []


@pytest.mark.parametrize("report", [True, {"accepted": True}, {"status": "complete"},
    {"status": "ACCEPTED", "native_step_reduction": True},
    {"status": "ACCEPTED", "native_step_reduction": .249},
    {"status": "ACCEPTED", "native_step_reduction": float("nan")}])
def test_profile_owner_return_is_not_boolean_authority(release_inputs, monkeypatch, report):
    root, ref, release, write, calls = release_inputs
    monkeypatch.setattr(prep, "_validate_profile_release", lambda *args: report)
    with pytest.raises(ValueError):
        prep.validate_parent_release(root, Path(ref), source_commit="1" * 40)


def test_missing_owner_callable_blocks_release(release_inputs, monkeypatch):
    root, ref, release, write, calls = release_inputs
    monkeypatch.undo()
    with pytest.raises(ValueError, match="callable is unavailable"):
        prep.validate_parent_release(root, Path(ref), source_commit="1" * 40)


@pytest.mark.parametrize("drift", [False, True])
def test_profile_accept_callable_contract_is_read_only(tmp_path, drift):
    directory = tmp_path / prep.REL / "profile"
    directory.mkdir(parents=True)
    # Synthetic owner tests only the adapter API; native validation is owner-tested.
    (directory / "close.py").write_text(
        "def accept(root):\n"
        "    return {'analysis': {'single_step_saving_fraction': 0.3}}\n"
        "def close(*args, **kwargs):\n"
        "    raise AssertionError('publication must not be called')\n", encoding="utf-8")
    acceptance = directory / "acceptance.json"
    acceptance.write_text(_canonical({
        "analysis": {"single_step_saving_fraction": .2 if drift else .3},
        "outcome": {"artifact_status": "local_hash_verified", "scientific_status": "NO_TRAIN"}}),
        encoding="utf-8")
    if drift:
        with pytest.raises(ValueError, match="differs from actual"):
            prep._validate_profile_release(tmp_path, acceptance)
    else:
        assert prep._validate_profile_release(tmp_path, acceptance) == {
            "status": "ACCEPTED", "native_step_reduction": .3}
    with pytest.raises(ValueError, match="not raw completion"):
        prep._validate_profile_release(tmp_path, directory / "completion.json")


@pytest.mark.parametrize("field", ["outcome", "analysis"])
def test_profile_nested_boolean_rejected_as_value_error(tmp_path, field):
    directory = tmp_path / prep.REL / "profile"
    directory.mkdir(parents=True)
    (directory / "close.py").write_text(
        "def accept(root):\n"
        "    return {'analysis': {'single_step_saving_fraction': 0.3}}\n", encoding="utf-8")
    record = {"analysis": {"single_step_saving_fraction": .3},
        "outcome": {"artifact_status": "local_hash_verified", "scientific_status": "NO_TRAIN"}}
    record[field] = True
    path = directory / "acceptance.json"
    path.write_text(_canonical(record), encoding="utf-8")
    with pytest.raises(ValueError, match="not booleans"):
        prep._validate_profile_release(tmp_path, path)


@pytest.mark.parametrize("record_key,field", [("terminal", "decision"), ("terminal", "artifact_hashes"),
    ("acceptance", "outcome"), ("finalization", "published_hashes"), ("finalization", "input_artifact_hashes")])
def test_clean_fit_nested_boolean_rejected_as_value_error(release_inputs, record_key, field):
    root, ref, release, write, calls = release_inputs
    binding = release["clean_fit"][record_key]
    record = prep.read(root / binding["path"])
    record[field] = True
    release["clean_fit"][record_key] = write(binding["path"], record)
    write(ref, release)
    with pytest.raises(ValueError, match="not booleans"):
        prep.validate_parent_release(root, Path(ref), source_commit="1" * 40)
    assert not calls


def test_active_wrapper_uses_package_root():
    from molgap.constants import REPO_ROOT
    assert prep.ROOT == REPO_ROOT


def test_integrated_clean_fit_receipt_binding_compatibility(release_inputs):
    root, ref, release, write, calls = release_inputs
    for key, binding in release["clean_fit"].items():
        if not isinstance(binding, dict):
            continue
        path = root / binding["path"]
        path.write_bytes((ROOT / binding["path"]).read_bytes())
        binding["sha256"] = prep.file_digest(path)
    write(ref, release)
    validated = prep.validate_parent_release(root, Path(ref), source_commit="1" * 40)
    assert validated["binding"]["sha256"] == prep.file_digest(root / ref)
    assert len(calls) == 1
