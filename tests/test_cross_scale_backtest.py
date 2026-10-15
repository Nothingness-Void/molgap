import hashlib
import json
from pathlib import Path

import pytest

from molgap.cross_scale_analysis import import_accepted_trace
from molgap.cross_scale_backtest import historical_pair_class, reassessment_regression, backtest_decisions
from molgap.research_memory.trace import file_digest


FIXTURE=Path(__file__).with_name("fixtures")/"cross_scale_reassessment_v1.json"


def install_historical_fixture(root):
    fixture=json.loads(FIXTURE.read_text(encoding="utf-8"))
    for relative,record in fixture["files"].items():
        payload=record["text"].encode()
        assert hashlib.sha256(payload).hexdigest()==record["sha256"]
        path=root/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(payload)
    return fixture


def bind(root,relative):
    return {"path":relative,"sha256":file_digest(root/relative)}


def test_golden_september_reassessment_contraction_not_projection(tmp_path):
    fixture=install_historical_fixture(tmp_path)
    p="experiments/pcqm_scale_transfer_reassessment/analysis.json"
    result=reassessment_regression(tmp_path,bind(tmp_path,p))
    assert fixture["source_commit"]=="ea0c400c34551afe428fda58de97e368f61dfbcf"
    assert result["pattern"]=="EARLY_ADVANTAGE_CONTRACTS"
    assert result["first_10_gain_eV"]==pytest.approx(.013116039335727692)
    assert result["last_10_gain_eV"]==pytest.approx(.0020235709846019743)
    assert result["100k_vs_500k_class"]=="CONTEXT_ONLY"
    assert result["observed_snapshots"][-1]["optimizer_step"]==234360
    assert not result["projection_accepted"] and not result["policy_activated"]


def test_actual_accepted_canonical_trace_import_preserves_missingness_and_groups(tmp_path):
    fixture=install_historical_fixture(tmp_path)
    dirname="experiments/pcqm_gptrans_noisy_pair_norm_500k/"
    trace=dirname+"results/canonical_trace.json"; manifest=dirname+"trace_manifest.json"
    # New adapter's CPU fixture acceptance binding; it does not rewrite or upgrade
    # the original historical acceptance. Independent original byte pins are above.
    acceptance=tmp_path/"accepted_binding.json"
    acceptance.write_text(json.dumps({"status":"accepted","trace_sha256":file_digest(tmp_path/trace),
                                     "manifest_sha256":file_digest(tmp_path/manifest)}))
    result=import_accepted_trace(tmp_path,trace_binding=bind(tmp_path,trace),manifest_binding=bind(tmp_path,manifest),
                                 acceptance_binding=bind(tmp_path,acceptance.name))
    assert result["rows"] and not result["rml_written"]
    assert all(r["unique_members_observed"] is None and r["allocated_device_hours"] is None for r in result["rows"])
    assert result["comparability_identity"]["scientific_contract"]=="pcqm-fixed500k-dev50k-matched60-v4"
    original=bind(tmp_path,trace); (tmp_path/trace).write_text("{}")
    with pytest.raises(ValueError,match="SHA"):
        import_accepted_trace(tmp_path,trace_binding=original,manifest_binding=bind(tmp_path,manifest),
                              acceptance_binding=bind(tmp_path,acceptance.name))


@pytest.mark.parametrize("field", ["evaluation_role_identity","lr_schedule_identity","ema_semantics"])
def test_historical_cross_role_recipe_state_is_context_only(field):
    base={k:"same" for k in ("scientific_contract","dataset_identity","row_split_identity","optimizer_identity",
        "lr_schedule_identity","precision_identity","ema_semantics","target_transform_identity","evaluation_role_identity",
        "selection_role_identity","x_axis_semantics")}
    other=dict(base); other[field]="different"
    report=historical_pair_class(base,other)
    assert report["comparison_class"]=="CONTEXT_ONLY" and field in report["reasons"]
    assert not report["paired_cross_scale"]


def case(name,positive,prediction,cost=None):
    return {"case_id":name,"group_id":name,"policy_training_groups":["independent-training-fold"],
        "prediction":prediction,"terminal_positive":positive,"terminal_accepted":True,"comparable":True,
        "cost_saved_native":cost,"hardware":"T4_device_hours"}


def test_leave_group_out_false_reject_false_promote_and_missing_cost():
    report=backtest_decisions([case("slow-start",True,"REJECT"),case("early-lead-late-loss",False,"PROMOTE")])
    assert report["false_reject"]["count"]==1 and report["false_promote"]["count"]==1
    assert report["false_reject"]["wilson95"] is not None
    assert report["status"]=="INSUFFICIENT_EVIDENCE" and report["observed_cost_saved_by_hardware"] is None
    assert not report["policy_activated"]
    bad=case("leaky",True,"PROMOTE"); bad["policy_training_groups"]=["leaky"]
    with pytest.raises(ValueError,match="leakage"):
        backtest_decisions([bad])


def test_unknown_terminal_or_context_cases_never_calibrate_quality_policy():
    report=backtest_decisions([case("missing-terminal",None,"REJECT"),case("known",True,"ABSTAIN",1.)])
    assert report["unscored_cases"]==["missing-terminal"] and not report["automatic_training_action"]
