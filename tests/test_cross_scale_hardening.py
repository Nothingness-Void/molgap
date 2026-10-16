"""Original review counterexamples; no real model, role or remote operation."""
from copy import deepcopy
import json

import pytest

from cross_scale_fixtures import cohorts, curve, identity, sha
from molgap.cross_scale_analysis import align_curves, comparison_report, write_comparison_report
from molgap.cross_scale_inputs import CohortProof, cohort_comparison_identity, verify_comparison_cohorts
from test_cross_scale_analysis import decision


def align(a, b, **kwargs):
    return align_curves(a, b, left_identity=identity(), right_identity=identity(), **kwargs)


def test_f1_original_missing_mae_cannot_hide_lr_conflict_or_invalidate_prior_point():
    a, b = curve([.3, None, .1]), curve([.4, .25, .09])
    b[1]["learning_rate"] = .002
    report = align(a, b)
    assert report["status"] == "NOT_STRICTLY_COMPARABLE"
    assert report["aligned"][0]["left_minus_right_eV"] == pytest.approx(-.1)
    assert report["aligned"][1]["left_minus_right_eV"] is None
    assert any(i["coordinate"] == 2 and "actual_lr" in i["reason"] for i in report["history_issues"])
    assert any("@2" in reason for reason in report["aligned"][1]["reasons"])


def test_f2_original_exposure_conflict_not_inferred_from_pass_or_horizon():
    a, b = curve([.3, .2, .1]), curve([.4, .25, .09])
    for r in b:
        r["sample_presentations"] += 128
    report = align(a, b)
    assert report["status"] == "NOT_STRICTLY_COMPARABLE"
    assert all(r["left_minus_right_eV"] is None for r in report["aligned"])
    assert any("sample_presentations" in r for r in report["reasons"])


@pytest.mark.parametrize("field,value", [("physical_batch_per_device", 64),
    ("gradient_accumulation_steps", 2), ("tail_batch_policy", "retain"), ("drop_last", False)])
def test_actual_batch_accumulation_tail_contracts_fail_closed(field, value):
    other = identity(); other[field] = value
    report = align_curves(curve([.3, .2, .1]), curve([.4, .25, .09]),
        left_identity=identity(), right_identity=other)
    assert report["status"] == "NOT_STRICTLY_COMPARABLE"
    assert "MISMATCH:"+field in report["reasons"]


def test_resume_duplicate_exposure_and_boolean_counter_are_rejected():
    for v in (128, True):
        a = curve([.3, .2, .1]); a[1]["sample_presentations"] = v
        assert align(a, curve([.4, .25, .09]))["status"] == "NOT_STRICTLY_COMPARABLE"
    a = curve([.3, .2, .1]); a[1]["optimizer_step"] = 1
    with pytest.raises(ValueError, match="monotonic"):
        align(a, curve([.4, .25, .09]))


def test_bare_tail_verification_flag_cannot_substitute_for_session_evidence():
    i=identity();i.update(drop_last=False,tail_batch_policy="retain")
    a,b=curve([.3,.2,.1]),curve([.4,.25,.09])
    for row in a+b: row["tail_exposure_verified"]=True
    report=align_curves(a,b,left_identity=i,right_identity=i)
    assert report["status"]=="NOT_STRICTLY_COMPARABLE"
    assert any("retained_tail_requires_bound_session_evidence" in r for r in report["reasons"])


def test_f3_original_truthy_flag_and_bare_true_cannot_forge_membership():
    other = identity(); other.update(row_membership_identity="500K", data_role_identity="500K")
    args = dict(left_identity=identity(), right_identity=other, lane="DATA_DIVERSITY")
    with pytest.raises(TypeError, match="exact bool"):
        align_curves(curve([.3,.2,.1]), curve([.4,.25,.09]), nested_cohorts_verified="False", **args)
    assert align_curves(curve([.3,.2,.1]), curve([.4,.25,.09]), nested_cohorts_verified=True, **args)["status"] == "NOT_STRICTLY_COMPARABLE"
    with pytest.raises(TypeError):
        CohortProof(())


@pytest.mark.parametrize("field", ["row_membership_identity", "row_order_identity", "evaluation_manifest_sha256",
    "train_authority_sha256", "evaluation_role_history_sha256", "evaluation_target_sha256", "dataset_identity"])
def test_membership_proof_is_bound_to_exact_data_authority_and_history(field):
    c, members = cohorts(); proof = verify_comparison_cohorts(*c, memberships=members)
    a, b = identity(), identity()
    a.update(cohort_comparison_identity(c[0], c[2])); b.update(cohort_comparison_identity(c[1], c[2]))
    b[field] = sha("tamper")
    report = align_curves(curve([.3,.2,.1]), curve([.4,.25,.09]), left_identity=a, right_identity=b,
        comparison_type="MATCHED_STEP_SCALE", cohort_proof=proof)
    assert report["status"] == "NOT_STRICTLY_COMPARABLE"
    assert any("bound_cohort_proof" in r for r in report["reasons"])


@pytest.mark.parametrize("field", ["terminal", "strictly_comparable", "seed_policy_approved"])
def test_f4_original_running_string_and_all_scientific_boolean_inputs(field):
    with pytest.raises(TypeError, match="exact bool"):
        decision(**{field: "RUNNING" if field == "terminal" else "False"})


@pytest.mark.parametrize("status", ["RUNNING", "COMPLETE", "STOP_FOR_COST", "FAILED", "INCONCLUSIVE"])
def test_scheduler_status_or_naked_complete_is_not_accepted_scientific_terminal(status):
    report = decision(execution_status=status)
    assert report["scientific_status"] == "INCONCLUSIVE"
    assert report["observed_progress_only"] and not report["veto_500k"]


def test_checkpoint_provenance_and_missing_cost_are_visible_in_both_reports(tmp_path):
    a, b = curve([.3,.2,.1]), curve([.4,.25,.09]); del b[-1]["checkpoint_identity"]
    result = align(a, b)
    assert result["aligned"][-1]["right_observation"].get("checkpoint_identity") is None
    assert result["aligned"][-1]["left_minus_right_eV"] is None
    report = comparison_report({"identity": identity(), "curve": a},
        {"identity": identity(), "curve": b}, comparison_type="SAME_SCALE_MODEL")
    files = write_comparison_report(tmp_path, report)
    assert set(files) == {"comparison.json", "comparison.md"}
    assert json.loads((tmp_path/"comparison.json").read_text())["runs"]["left"]["cost"] is None
    assert "right:checkpoint_sha256" in report["missing_evidence"]
    assert report["early_stop_status"] == "NOT_RELEASED"
    assert "UNKNOWN" not in (tmp_path/"comparison.json").read_text()  # Null, never invented zero.


def test_tuned_recipe_system_is_explicitly_not_architecture_causality():
    a, b = identity(), identity()
    for i in (a, b):
        i.update(tuning_budget_identity="symmetric", tuning_budget_per_system=2,
                 tuning_protocol_sha256=sha("frozen tuning"))
    b["schedule_identity"] = "independently-tuned"
    report = align_curves(curve([.3,.2,.1]),curve([.4,.25,.09]),left_identity=a,right_identity=b,
        comparison_type="TUNED_RECIPE_SYSTEM")
    assert report["strictness"] == "SYSTEM_COMPARISON"
    assert not report["automatic_training_action"]


def test_v1_v2_never_become_same_contract_and_context_never_acquires_strictness():
    a, b = identity(), identity(); b["dataset_identity"] = "Unified-V2"
    for kind in ("SAME_SCALE_MODEL", "CONTEXT_OR_PORTABILITY"):
        report = align_curves(curve([.3,.2,.1]),curve([.4,.25,.09]),left_identity=a,right_identity=b,comparison_type=kind)
        assert report["strictness"] == "CONTEXT_ONLY"
        assert all(r["left_minus_right_eV"] is None for r in report["aligned"])


def accepted_pair(tmp_path, monkeypatch, seed):
    """Reuse the real owner inspector and existing synthetic output producer."""
    import torch
    import test_experiment_family_workflow as fixture
    from molgap.experiment_family_workflow import RunContext, capture_output_inspection, write_output_manifest
    from molgap.research_memory.trace import file_digest, json_bytes
    original=fixture._recipe
    def recipe(family):
        return {**original(family), "seed": seed}
    # Context hashes are frozen BEFORE producing these synthetic outputs.
    with monkeypatch.context() as patch:
        patch.setattr(fixture,"_recipe",recipe)
        result=[]
        for name in ("candidate", "reference"):
            contract=recipe("neural_atom_k1")
            context=RunContext(experiment_id="synthetic-cross-scale",logical_run_id="synthetic-pair",
                arm_id=f"{name}{seed}",arm_identity=sha(name+str(seed)),spec_identity=sha("spec"),
                family_name="neural_atom_k1",family_version="1",source_commit="f"*40,
                source_archive_sha256=sha("source"),package_identity=sha("package"),
                training_recipe_sha256=__import__("hashlib").sha256(json_bytes(contract)).hexdigest(),
                platform="kaggle",account="synthetic-account",run_reference="synthetic-account/workload-1",
                platform_version="synthetic-platform-v1")
            root,expected,artifacts,progress=fixture._write_output(tmp_path,context,"k1-v1",publish=False)
            if name=="candidate":
                p=root/"predictions.pt";value=torch.load(p,weights_only=True);value["prediction_eV"]*=.99;torch.save(value,p)
                p=root/"selected_model.pt";value=torch.load(p,weights_only=True);value["development_mae_eV"]=.99;torch.save(value,p)
                p=root/"trace.json";value=json.loads(p.read_text())
                [r for r in value["observations"] if r["event"]=="observation"][-1]["live_dev_metric"]=.99
                p.write_bytes(json_bytes(value))
            write_output_manifest(root,context,adapter="k1-v1",artifacts=artifacts,progress=progress,
                runtime=fixture._runtime(context),costs=fixture._costs())
            result.append(capture_output_inspection(root,context=context,expected=expected))
        return (seed,*result)


def test_positive_requires_actual_accepted_paired_outputs_and_seed_effect_binding(tmp_path,monkeypatch):
    pairs=[accepted_pair(tmp_path,monkeypatch,seed) for seed in (42,43)]
    report=decision(execution_status="COMPLETE",terminal_evidence=pairs,seed_deltas=[(42,.01),(43,.01)])
    assert report["terminal_acceptance"]=="BOUND_OUTPUTS_VERIFIED"
    assert report["scientific_status"]=="EXPLORATORY_PRIMARY_POSITIVE"
    assert report["promotion"]=="NOT_AUTHORIZED" and not report["automatic_training_action"]
    with pytest.raises(ValueError,match="effect differs"):
        decision(execution_status="COMPLETE",terminal_evidence=pairs,seed_deltas=[(42,.1),(43,.1)])
    # Accepted bytes changed AFTER capture: stale proof cannot be reused.
    (pairs[0][1].output_dir/"trace.json").write_text("{}")
    with pytest.raises(ValueError,match="stale"):
        decision(execution_status="COMPLETE",terminal_evidence=pairs,seed_deltas=[(42,.01),(43,.01)])
