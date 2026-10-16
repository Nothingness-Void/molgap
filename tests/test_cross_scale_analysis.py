from copy import deepcopy
import json

import pytest

from cross_scale_fixtures import curve, identity, sha
from molgap.cross_scale_analysis import align_curves, interpret_run, summarize_costs, allocation_view, profile_view


def test_equal_steps_different_passes_align_and_no_unobserved_point_is_filled():
    a, b = curve([.3,.2,.1]), curve([.4,.3,.2], steps=(1,3,5))
    for row in b: row["complete_passes"] = 7
    result = align_curves(a,b,left_identity=identity(),right_identity=identity())
    assert result["status"] == "MATCHED_PREFIX"
    assert [r["optimizer_step"] for r in result["aligned"]] == [1,3]
    assert not result["extrapolated"]
    with pytest.raises(ValueError,match="Epoch"):
        align_curves(a,b,left_identity=identity(),right_identity=identity(),axis="epoch")


@pytest.mark.parametrize("field", ["schedule_identity", "weight_semantics", "precision", "target_transform_identity",
                                 "evaluation_role_identity", "sampler_identity", "feature_identity", "seed",
                                 "checkpoint_selection_identity", "evaluation_row_sha256", "evaluation_target_sha256",
                                 "ema_update_frequency", "runtime_certificate_scope", "target_unit", "bn_calibration_identity", "evaluation_row_count"])
def test_same_step_confounds_never_rank_as_strict(field):
    right = identity(); right[field] = "different"
    result = align_curves(curve([.2,.1,.09]),curve([.1,.09,.08]),left_identity=identity(),right_identity=right)
    assert result["status"] == "NOT_STRICTLY_COMPARABLE"
    assert "MISMATCH:"+field in result["reasons"]
    assert all(row["left_minus_right_eV"] is None for row in result["aligned"])


def test_missing_and_actual_lr_and_order_are_not_filled_from_metadata():
    right = identity(); del right["schedule_identity"]
    a,b=curve([.3,.2,.1]),curve([.4,.3,.2]); b[1]["learning_rate"] = .002
    result=align_curves(a,b,left_identity=identity(),right_identity=right)
    assert "MISSING:schedule_identity" in result["reasons"]
    assert "MISMATCH:actual_lr_at_2" in result["reasons"]
    with pytest.raises(ValueError,match="monotonic"):
        align_curves(a,list(reversed(b)),left_identity=identity(),right_identity=identity())


def test_explicit_no_ema_is_not_missing_but_unknown_metric_state_is():
    value=identity(); value.update(ema_enabled=False,ema_decay=None,ema_update_frequency=None)
    a,b=curve([.2,.1,.09]),curve([.21,.11,.1])
    assert align_curves(a,b,left_identity=value,right_identity=value)["status"]=="MATCHED_PREFIX"
    del b[0]["weights"]
    report=align_curves(a,b,left_identity=value,right_identity=value)
    assert "MISSING:metric_weights" in report["reasons"]


def test_data_diversity_lane_requires_shared_disjoint_cohort_proof_and_step_schedule():
    from cross_scale_fixtures import cohorts
    from molgap.cross_scale_inputs import verify_comparison_cohorts, cohort_comparison_identity
    c, members = cohorts()
    first, other = identity(), identity()
    first.update(cohort_comparison_identity(c[0], c[2]))
    other.update(cohort_comparison_identity(c[1], c[2]))
    args=dict(left_identity=first,right_identity=other,lane="DATA_DIVERSITY")
    assert align_curves(curve([.2,.15,.1]),curve([.21,.13,.09]),**args)["status"] == "NOT_STRICTLY_COMPARABLE"
    assert align_curves(curve([.2,.15,.1]),curve([.21,.13,.09]),nested_cohorts_verified=True,**args)["status"] == "NOT_STRICTLY_COMPARABLE"
    proof=verify_comparison_cohorts(*c,memberships=members)
    assert align_curves(curve([.2,.15,.1]),curve([.21,.13,.09]),cohort_proof=proof,**args)["status"] == "MATCHED_PREFIX"


def decision(**changes):
    options=dict(stage="PRIMARY_500K",fidelity="MODEL_FIDELITY_ACCEPTED",terminal=True,
        runtime="MECHANICALLY_VERIFIED",budget="MEASURED_WITHIN_CEILING",roles="ROLE_VERIFIED",
        strictly_comparable=True,seed_deltas=[(42,.005),(43,.006)],material_gain_eV=.003,
        seed_policy_approved=True)
    options.update(changes)
    return interpret_run(**options)


def test_slow_starter_100k_cannot_veto_later_500k_winner():
    early=decision(stage="DIAGNOSTIC_100K",seed_deltas=[(42,-.05)])
    late=decision()
    assert not early["veto_500k"] and early["scientific_status"] == "DIAGNOSTIC_ONLY"
    # A formerly accepted naked terminal=True is no longer endpoint evidence.
    assert late["scientific_status"] == "INCONCLUSIVE"
    assert late["terminal_acceptance"] == "MISSING"
    assert late["promotion"] == "NOT_AUTHORIZED" and not late["automatic_training_action"]


@pytest.mark.parametrize("changes,expected", [({"fidelity":"missing"},"IMPLEMENTATION_UNVERIFIED"),
    ({"terminal":False},"INCONCLUSIVE"),({"roles":"missing"},"INCONCLUSIVE"),
    ({"budget":"STOP_FOR_COST"},"INCONCLUSIVE"),({"runtime":"CPU_ONLY"},"INCONCLUSIVE"),
    ({"strictly_comparable":False},"INCONCLUSIVE")])
def test_incomplete_primary_or_fidelity_is_not_scientific_negative(changes,expected):
    assert decision(**changes)["scientific_status"] == expected


def test_bootstrap_does_not_replace_seed_replication_and_sign_disagreement_is_explicit():
    one=decision(seed_deltas=[(42,.005)])
    assert one["scientific_status"] == "INCONCLUSIVE"
    assert any("SEED_VARIANCE_UNKNOWN" in w for w in one["warnings"])
    two=decision(seed_deltas=[(42,.005),(43,-.001)])
    assert "SEED_SIGN_DISAGREEMENT" in two["warnings"]
    with pytest.raises(ValueError,match="Duplicate seed"):
        decision(seed_deltas=[(42,.005),(42,.006)])


@pytest.mark.parametrize("failure", ["INVALID_CODE","NONFINITE","CORRUPT_INPUT","UNAUTHORIZED_ROLE","CHECKPOINT_FAILURE","DECLARED_COST_CEILING"])
def test_implementation_or_budget_stop_does_not_condemn_architecture_family(failure):
    report=decision(stage="DIAGNOSTIC_100K",failure=failure)
    assert report["scientific_status"] == "INCONCLUSIVE" and not report["veto_500k"]
    assert report["budget_decision"] == ("STOP_FOR_COST" if failure=="DECLARED_COST_CEILING" else "BLOCKED")


def test_quality_and_stqs_are_not_stop_events():
    for event in ("LOW_100K_MAE","STQS","FUTILITY"):
        with pytest.raises(ValueError,match="not an admissible"):
            decision(failure=event)


def cost_event(attempt="attempt1",hardware="Tesla_T4",value=.2,status="measured"):
    def m(v=None,s="measurement_missing"): return {"value":v,"status":s}
    return {"schema":"molgap-cost-event-v1","cost_event_id":"cost-"+attempt,"trajectory_id":"T-synthetic",
        "action_id":"A-synthetic","run_id":"R-synthetic","attempt_id":attempt,"platform":"CPU-fixture-only",
        "hardware":hardware,"evidence_ref":"synthetic/proof.json","category":"training",
        "measurement":{"device_hours":m(value,status),"cpu_hours":m(),"wall_hours":m(),"queue_hours":m()}}


def test_missing_previous_invocation_and_allocated_cost_remain_unknown():
    event=cost_event()
    result=summarize_costs([event,event],expected_invocations=["attempt0","attempt1"])
    assert result["missing_invocations"] == ["attempt0"]
    assert result["allocation_coverage"] == "MISSING_OR_UNKNOWN"
    assert result["whole_run_allocated_device_hours"] is None
    assert result["ledger"]["event_count"] == 1
    unknown=summarize_costs([cost_event(value=None,status="measurement_missing")],expected_invocations=["attempt1"])
    assert unknown["ledger"]["totals_by_hardware"]["Tesla_T4"]["device_hours"]["measured"]["known_total"] is None


def test_hardware_and_estimated_cost_never_mix_or_become_measured():
    result=summarize_costs([cost_event(),cost_event("attempt2","A100",.1,"estimated")],expected_invocations=["attempt1","attempt2"])
    assert result["allocation_coverage"] == "MISSING_OR_UNKNOWN"
    assert result["cross_hardware_total"] is None
    assert len(result["ledger"]["totals_by_hardware"]) == 2
    with pytest.raises(ValueError,match="Conflicting"):
        summarize_costs([cost_event(),cost_event(value=.5)],expected_invocations=["attempt1"])


def test_existing_allocation_ledger_counts_idle_devices_without_double_counting():
    from molgap.experiment_allocation import AllocationLedger
    first=AllocationLedger(spec_identity=sha("spec"),hardware=["Tesla T4","Tesla T4"],assignments={"arm":0},clock=lambda:0.)
    record=first.snapshot("complete"); record["wall_seconds"]=36.; record["allocated_device_seconds"]=72.
    record["unassigned_device_seconds"]=36.
    for device in record["devices"]: device["allocated_seconds"]=36.
    result=allocation_view([record,record],spec_identity=sha("spec"),arm_ids=["arm"])
    assert result["measured_window_hours_by_hardware"]=={"Tesla T4":.02}
    assert result["status"]=="MISSING_OR_UNKNOWN" and result["whole_run_hours_by_hardware"] is None
    corrupt=deepcopy(record); corrupt["allocated_device_seconds"]=36.
    with pytest.raises(ValueError,match="disagrees"):
        allocation_view([corrupt],spec_identity=sha("spec"),arm_ids=["arm"])
    different_observation=deepcopy(record); different_observation["status"]="failed"
    with pytest.raises(ValueError,match="Overlapping"):
        allocation_view([record,different_observation],spec_identity=sha("spec"),arm_ids=["arm"])


def test_profiler_projection_is_not_a_second_timer_or_gpu_cost_estimate():
    profile={"format":"molgap-family-execution-profile-v1","runtime":{"device":"cpu","torch":"synthetic"},
        "phases_by_host_self_time":[{"name":"forward","host_self_seconds":.5}],
        "device_events_observed":False,"context_peak_allocated_bytes":None,"context_peak_reserved_bytes":None}
    result=profile_view(profile)
    assert result["phase_host_self_seconds"]["forward"]==.5
    assert result["phase_host_self_seconds"]["transfer"] is None
    assert result["allocated_device_hours"] is None and result["peak_allocated_bytes"] is None
