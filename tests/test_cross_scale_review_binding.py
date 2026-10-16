"""R1/R2 constructive counterexamples; frozen declarations are independent."""
from copy import deepcopy
import pytest

from cross_scale_fixtures import curve, identity, sha
from molgap.cross_scale_analysis import align_curves, comparison_report, write_comparison_report


def declared_identity(weights="live"):
    i = identity()
    i.update(target_identity="pcqm4mv2-gap-eV-direct", evaluation_role_identity="internal_development",
        selection_role_identity="internal_development", data_role_identity="internal_train",
        weight_semantics=weights, evaluation_weight_source=weights,
        trace_sha256=sha("trace"), trace_run_id="synthetic-run",
        source_sha256=sha("model-source"),selected_optimizer_step=3,
        checkpoint_identities={str(s): sha("observed-checkpoint-"+str(s)) for s in range(1, 6)})
    def definition(role, state, unit="eV"):
        return dict(metric="MAE",unit=unit,target="Gap",role_identity=role,weights=state,direction="minimize")
    i["metric_semantics"] = {"live_train_metric":definition("internal_train","live"),
        "live_dev_metric":definition("internal_development","live"),
        "ema_dev_metric":definition("internal_development","ema") if weights == "ema" else None}
    if weights == "ema": i.update(ema_enabled=True,ema_decay=.999,ema_update_frequency=1)
    return i


def bound_curve(values=(.3,.2,.1)):
    return [{**r,"target":"Gap","target_identity":"pcqm4mv2-gap-eV-direct",
             "source_trace_sha256":sha("trace"),"trace_run_id":"synthetic-run"} for r in curve(values)]


def compare(a=None,b=None,**kw):
    return align_curves(a or bound_curve(), b or bound_curve(),
        left_identity=kw.pop("left_identity",declared_identity()),
        right_identity=kw.pop("right_identity",declared_identity()), **kw)


def test_r1_correct_declared_observations_positive_control():
    assert compare()["strictness"] == "STRICT_SAME_SCALE"


@pytest.mark.parametrize("field,value", [("evaluation_role","internal_development_other"),
    ("weights","raw_ema"),("metric_kind","online-train"),("target","HOMO"),
    ("target_identity",sha("another target")),("source_trace_sha256",sha("other trace")),
    ("trace_run_id","another-run"),("checkpoint_identity",sha("another checkpoint")),("unit","Hartree")])
def test_r1_two_sides_agreeing_on_wrong_observations_never_become_strict(field,value):
    a,b=bound_curve(),bound_curve()
    for row in a+b: row[field]=value
    report=compare(a,b)
    assert report["strictness"] == "CONTEXT_ONLY"
    assert all(row["left_minus_right_eV"] is None for row in report["aligned"])
    assert not report["automatic_training_action"]


def test_r2_nonselected_metric_cannot_hide_lr_history_conflict():
    a,b=bound_curve(),bound_curve()
    for row in (a[1],b[1]):
        row.update(metric="live_train_metric",metric_kind="online-train",evaluation_role="internal_train")
    b[1]["learning_rate"] = .002
    report=compare(a,b)
    assert report["aligned"][0]["comparison_class"] == "MATCHED_PREFIX"
    assert report["aligned"][-1]["left_minus_right_eV"] is None
    assert any("actual_lr" in x["reason"] and x["coordinate"] == 2 for x in report["history_issues"])


def test_r2_duplicate_step_metrics_are_merged_and_contradictions_are_visible():
    a,b=bound_curve(),bound_curve()
    other={**b[1],"metric":"live_train_metric","metric_kind":"online-train","evaluation_role":"internal_train"}
    b.insert(2,other)
    assert compare(a,b)["strictness"] == "STRICT_SAME_SCALE"
    other["learning_rate"] = .002
    report=compare(a,b)
    assert report["strictness"] == "CONTEXT_ONLY"
    assert any("conflicting_step_learning_rate" in x["reason"] for x in report["history_issues"])


def test_system_comparison_binds_each_selected_metric_to_its_own_rule():
    a,b=bound_curve(),bound_curve()
    for row in b: row.update(metric="ema_dev_metric",weights="ema")
    args=dict(left_identity=declared_identity(),right_identity=declared_identity("ema"),
        metric=("live_dev_metric","ema_dev_metric"),comparison_type="TUNED_RECIPE_SYSTEM")
    for i in (args["left_identity"],args["right_identity"]):
        i.update(tuning_budget_identity="symmetric",tuning_budget_per_system=2,tuning_protocol_sha256=sha("tuning"))
    assert compare(a,b,**args)["strictness"] == "SYSTEM_COMPARISON"
    for row in b: row["weights"]="live"
    assert compare(a,b,**args)["strictness"] == "CONTEXT_ONLY"


def test_report_envelope_cannot_disagree_with_trace_or_frozen_identity(tmp_path):
    i=declared_identity();a=bound_curve()
    run=dict(identity=i,curve=a,model="synthetic",source_sha256=i["source_sha256"],trace_sha256=i["trace_sha256"],
        checkpoint_sha256=a[-1]["checkpoint_identity"],weights="live",bn_state="not_applicable",seed=42,
        cost={"allocation_coverage":"MEASURED"})
    assert comparison_report(run,run,comparison_type="SAME_SCALE_MODEL")["strictness"]=="STRICT_SAME_SCALE"
    for field,value in (("weights","ema"),("trace_sha256",sha("wrong trace")),
        ("checkpoint_sha256",sha("wrong checkpoint")),("source_sha256",sha("wrong source"))):
        changed=deepcopy(run);changed[field]=value
        report=comparison_report(changed,changed,comparison_type="SAME_SCALE_MODEL")
        assert report["strictness"]=="CONTEXT_ONLY"
        assert report["curve_comparison"]["strictness"]=="STRICT_SAME_SCALE"
    write_comparison_report(tmp_path,report)
    assert 'Grade: CONTEXT_ONLY' in (tmp_path/'comparison.md').read_text()


@pytest.mark.parametrize("field", ["metric_semantics", "trace_sha256", "trace_run_id", "checkpoint_identities"])
def test_missing_independent_declaration_cannot_be_reconstructed_from_agreeing_rows(field):
    i=declared_identity();i.pop(field)
    assert compare(left_identity=i,right_identity=deepcopy(i))["strictness"] == "CONTEXT_ONLY"


def test_auxiliary_live_dev_is_valid_with_ema_selected_but_cannot_lie_about_its_role():
    i=declared_identity("ema")
    rows=bound_curve()
    for row in rows: row.update(metric="ema_dev_metric",weights="ema")
    aux={**rows[1],"metric":"live_dev_metric","weights":"live"}
    rows.insert(2,aux)
    args=dict(left_identity=i,right_identity=deepcopy(i),metric="ema_dev_metric")
    assert compare(rows,deepcopy(rows),**args)["strictness"] == "STRICT_SAME_SCALE"
    aux["evaluation_role"]="internal_development_wrong"
    report=compare(rows,deepcopy(rows),**args)
    assert report["strictness"] == "CONTEXT_ONLY"
    assert report["aligned"][0]["comparison_class"] == "MATCHED_PREFIX"


@pytest.mark.parametrize("field,value", [("learning_rate",None),("sample_presentations",999),("trace_run_id","wrong")])
def test_auxiliary_missing_or_contradictory_training_history_is_not_hidden(field,value):
    rows=bound_curve()
    rows[1].update(metric="live_train_metric",metric_kind="online-train",evaluation_role="internal_train")
    rows[1][field]=value
    assert compare(rows,deepcopy(rows))["aligned"][-1]["left_minus_right_eV"] is None


def test_selected_checkpoint_is_bound_at_selected_step_not_last_observation():
    i=declared_identity();i["selected_optimizer_step"]=1
    rows=bound_curve()
    run=dict(identity=i,curve=rows,model="synthetic",source_sha256=i["source_sha256"],trace_sha256=i["trace_sha256"],
        checkpoint_sha256=rows[0]["checkpoint_identity"],weights="live",bn_state="not_applicable",seed=42,
        cost={"allocation_coverage":"MEASURED"})
    assert comparison_report(run,run,comparison_type="SAME_SCALE_MODEL")["strictness"]=="STRICT_SAME_SCALE"
