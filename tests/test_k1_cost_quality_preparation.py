"""Synthetic preparation/pair lifecycle only; no model execution or remote calls."""
import copy
import importlib.util
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
