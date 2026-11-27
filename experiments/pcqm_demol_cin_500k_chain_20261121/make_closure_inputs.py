"""One-off: write the three legacy-terminal input files for one arm (finalized_at = real time, +09:00).
Usage: python make_closure_inputs.py <arm>   arms: j2 j5 j1 j3 j4 old_j1 old_j2
Inputs land in closure_in/ (outside evidence/, so they do not change the frozen evidence hashes)."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = "experiments/pcqm_demol_cin_500k_chain_20261121"
DECISION = f"{EXP}/decision.md"
ANALYSIS = f"{EXP}/evidence/closure_analysis_20261128.json"
RUN = "demol500k-chain-20261121"
ID = "pcqm4mv2-ogb-official-train"
DEV100 = "6f35739b8eb4602eae9695d64f99ae540ce797ababf83e083af552352ddb6191"
DEV500 = "c4658eaee738ddbdf25e62363e06fb1ad616f3cb98e05f8b3f557c3af124a565"
TAIL = "f111ebbbc96f602c3733208f964fa1806fe59f1a2905ae7026b2feded346fe89"
WALL = {"j1": 11245.1, "j2": 7684.3, "j3": 47451.5, "j4": 45007.2, "j5": 18100.1 + 14127.7}
ATTEMPT = {"j1": f"{RUN}-j1-v1", "j2": f"{RUN}-j2-v1", "j3": f"{RUN}-j3-demol-500k-s42-v1",
           "j4": f"{RUN}-j4-demol-500k-s43-v1", "j5": f"{RUN}-j5-gptt-500k-s44-v1"}
NEXT_NEW = []
REOPEN_NEW = ["Any new DeMol-CIN run (including follow-up on the post-pass-40 overfitting) is a new question: query RML first, "
              "write a prospective trajectory, and pass the Opus pre-submit review. A full-scale DeMol run also needs explicit user authority."]
OK = "complete"

ARMS = {
    "j2": dict(outcome="CLOSED", dev=("dev_100k", DEV100), facts_src=f"{EXP}/evidence/score/j2.json", effect=None,
               v5=dict(execution_status=OK, artifact_status="retrieved_hash_verified",
                       comparison_status="same_job_reference_for_j1_paired_dev_100k_non_strict",
                       scientific_status="reference_arm_no_candidate_conclusion", transfer_status="not_assessed_reference_arm",
                       budget_decision="no_further_runs_from_this_arm", full_handoff_status="not_applicable"),
               terms=["gpt-t", "pair update norm", "100k", "seed 43", "ims a100", "reference arm", "demol replication", "dev rows"],
               reopen=["Terminal evidence only."]),
    "j5": dict(outcome="CLOSED", dev=("dev_500k", DEV500), facts_src=f"{EXP}/evidence/score/j5.json", effect=None,
               v5=dict(execution_status=OK, artifact_status="retrieved_hash_verified",
                       comparison_status="same_platform_reference_for_j3_j4_paired_tail_500k_non_strict",
                       scientific_status="platform_reference_shift_plus_0.035_meV_accepted",
                       transfer_status="cross_platform_reference_accepted_G_107.940_kept",
                       budget_decision="no_further_runs_from_this_arm", full_handoff_status="not_applicable"),
               terms=["gpt-t", "platform control", "500k", "init s44", "ims a100", "tail shift", "cross platform", "reference arm"],
               reopen=["Terminal evidence only."]),
    "j1": dict(outcome="POSITIVE_UNDER_CONTRACT", dev=("dev_100k", DEV100), facts_src=ANALYSIS, effect=0,
               v5=dict(execution_status=OK, artifact_status="retrieved_hash_verified",
                       comparison_status="paired_dev_rows_non_strict_same_job_reference_j2",
                       scientific_status="POSITIVE_UNDER_CONTRACT_100K_DEV_REPLICATION_RULE_A_ONLY",
                       transfer_status="not_assessed_here_see_j3_j4_inconclusive",
                       budget_decision="no_full_scale_nomination", full_handoff_status="not_nominated"),
               terms=["demol-cin", "cellular complex", "100k replication", "seed 43", "ims a100", "dev bootstrap", "rule a", "hetero ensemble"],
               reopen=REOPEN_NEW, next=NEXT_NEW),
    "j3": dict(outcome="INCONCLUSIVE", dev=("dev_500k", DEV500), facts_src=ANALYSIS, effect=1, failure_mode="below_gate",
               v5=dict(execution_status=OK, artifact_status="retrieved_hash_verified",
                       comparison_status="paired_tail_rows_vs_j5_same_platform_non_strict",
                       scientific_status="INCONCLUSIVE_BELOW_GATE_MIGRATION_DELTA_MINUS_2.882_MEV",
                       transfer_status="inconclusive_delta_in_band_minus_3.0_to_minus_1.5",
                       budget_decision="no_full_scale_nomination", full_handoff_status="not_nominated"),
               terms=["demol-cin", "cellular complex", "500k", "tail holdout", "ims a100", "below gate", "seed 42", "migration rule"],
               reopen=REOPEN_NEW, next=NEXT_NEW),
    "j4": dict(outcome="INCONCLUSIVE", dev=("dev_500k", DEV500), facts_src=ANALYSIS, effect=2, failure_mode="below_gate",
               v5=dict(execution_status=OK, artifact_status="retrieved_hash_verified",
                       comparison_status="paired_tail_rows_vs_j5_same_platform_non_strict",
                       scientific_status="INCONCLUSIVE_BELOW_GATE_SHARED_MIGRATION_VERDICT_WITH_J3",
                       transfer_status="inconclusive_delta_in_band_minus_3.0_to_minus_1.5",
                       budget_decision="no_full_scale_nomination", full_handoff_status="not_nominated"),
               terms=["demol-cin", "cellular complex", "500k", "tail holdout", "ims a100", "below gate", "seed 43", "migration rule"],
               reopen=REOPEN_NEW, next=NEXT_NEW),
}
CANCEL = {
    "old_j1": "pcqm-demol-cin-100k-ims-20261120-s43",
    "old_j2": "pcqm-gptrans-pun-100k-ims-20261120-s43",
}


def build(arm):
    now = datetime.now(timezone(timedelta(hours=9))).replace(microsecond=0).isoformat()
    out = HERE / "closure_in"
    out.mkdir(exist_ok=True)
    if arm in CANCEL:
        dec = {
            "evidence_id": f"pcqm-demol-cin-500k-chain-20261128-{arm.replace('_', '-')}",
            "scope": "Stand-alone IMS job cancelled by the user before training; replaced by the serial chain item. No training, no scoring.",
            "legacy_contract": "pcqm-demol-cin-500k-chain-v1",
            "finalized_at": now,
            "v5_outcome": dict(execution_status="not_run_cancelled_before_training", artifact_status="no_artifacts",
                               comparison_status="not_applicable", scientific_status="NO_TRAIN_superseded_by_serial_chain_item",
                               transfer_status="not_applicable", budget_decision="cost_measurement_missing_not_zero",
                               full_handoff_status="not_applicable"),
            "outcome": "NO_TRAIN", "decision_ref": DECISION, "next_allowed_actions": [],
            "reopen_conditions": ["Terminal evidence only. The result of this recipe lives in the chain trajectory."],
            "facts": {"effect_meV": None, "ci_meV": None, "failure_mode": None,
                      "search_terms": ["demol-cin", "ims", "cancelled", "no train", "superseded", "serial chain"], "source": None},
        }
        roles = {"role_use": {"official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
                              "tail_holdout_v1": "untouched"}, "events": []}
        (out / f"{arm}_decision_in.json").write_text(json.dumps(dec, indent=2) + "\n")
        (out / f"{arm}_roles_in.json").write_text(json.dumps(roles, indent=2) + "\n")
        return [f"{arm}_decision_in.json", f"{arm}_roles_in.json"]
    a = ARMS[arm]
    pair_key = {"j1": 0, "j3": 1, "j4": 2}
    if a["effect"] is None:
        eff, ci = None, None
    else:
        p = json.loads((HERE / "evidence" / "closure_analysis_20261128.json").read_text())["pairs"][a["effect"]]
        eff, ci = p["delta"], p["ci95"]
    dec = {
        "evidence_id": f"pcqm-demol-cin-500k-chain-20261128-{arm}",
        "scope": "DeMol-CIN 500K IMS serial chain item; offline dev and tail scoring only; official valid and test not read; non-strict.",
        "legacy_contract": "pcqm-demol-cin-500k-chain-v1",
        "finalized_at": now, "v5_outcome": a["v5"], "outcome": a["outcome"], "decision_ref": DECISION,
        "next_allowed_actions": a.get("next", []), "reopen_conditions": a["reopen"],
        "facts": {"effect_meV": eff, "ci_meV": ci, "failure_mode": a.get("failure_mode"), "search_terms": a["terms"],
                  "source": a["facts_src"]},
    }
    rn, rh = a["dev"]
    ev = lambda name, kind, sel, h: {"role_name": name, "access_kind": kind, "selection_used": sel, "dataset_identity": ID,
                                     "row_manifest_hash": h}
    events = [ev(rn, k, True, rh) for k in ("prediction_input", "labels_read", "metric_computed", "selection_used")]
    events += [ev("tail_holdout_v1", k, False, TAIL) for k in ("prediction_input", "labels_read", "metric_computed")]
    roles = {"role_use": {"official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
                          "tail_holdout_v1": "consumed", rn: "consumed"}, "events": events}
    h = WALL[arm] / 3600
    costs = {"events": [{"attempt_id": ATTEMPT[arm], "platform": "ims", "hardware": "NVIDIA A100 FP32", "category": "training",
                         "measurement": {"cpu_hours": {"status": "measurement_missing", "value": None},
                                         "device_hours": {"status": "measured", "value": h},
                                         "queue_hours": {"status": "measurement_missing", "value": None},
                                         "wall_hours": {"status": "measured", "value": h}}}]}
    (out / f"{arm}_decision_in.json").write_text(json.dumps(dec, indent=2) + "\n")
    (out / f"{arm}_roles_in.json").write_text(json.dumps(roles, indent=2) + "\n")
    (out / f"{arm}_costs_in.json").write_text(json.dumps(costs, indent=2) + "\n")
    return [f"{arm}_decision_in.json", f"{arm}_roles_in.json", f"{arm}_costs_in.json"]


if __name__ == "__main__":
    print(build(sys.argv[1]))
