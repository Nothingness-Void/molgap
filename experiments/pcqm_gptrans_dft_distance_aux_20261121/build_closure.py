"""Build results.json, acceptance.json and terminal.json for the closure (run after decision.md is final).

Inputs: the dev analysis JSON from score_dev_and_distance.py and the retrieved serial_summary.json.
Usage: build_closure.py --analysis dev_analysis.json --retrieved DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = "experiments/pcqm_gptrans_dft_distance_aux_20261121"
TRAJ, RUN, ACTION = "T-pcqm-gptrans-dft-distance-aux-20261121", "gptdft-chain-20261121", "A002"
NOW = "2026-11-23T12:00:00+00:00"
J2_SELECTED_SHA = "3840b378426762786676ca9d685881f785434a35e0765d1009ca8faf1985bccb"  # read-only remote cat of job 2097505


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_hash(start, stop):
    return hashlib.sha256(json.dumps(list(range(start, stop))).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--retrieved", required=True)
    args = parser.parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    retrieved = Path(args.retrieved)
    summary = json.loads((retrieved / "training" / "serial_summary.json").read_text(encoding="utf-8"))
    wall = {k: v["attempts"][-1]["wall_s"] for k, v in summary["items"].items()}
    c100, d100, c500, d500 = ("c100-gptt-100k-s43", "d100-dft-100k-s43", "c500-gptt-500k-s44", "d500-dft-500k-s44")
    b100, b500 = analysis["bootstrap"]["d100_minus_c100"], analysis["bootstrap"]["d500_minus_c500"]
    results = dict(analysis)
    results["verdict"] = {"100k": "POSITIVE_UNDER_CONTRACT", "500k": "INCONCLUSIVE", "tail_scoring": "not released",
                          "c1_a_aux_delta_500k_meV": b500["delta_mev"], "c1_a_needs_meV": -1.0}
    results["attribution"] = {
        "retention_500k_over_100k": b500["delta_mev"] / b100["delta_mev"],
        "wall_s": wall, "wall_ratio_d_over_c": {"100k": wall[d100] / wall[c100], "500k": wall[d500] / wall[c500]},
        "total_item_wall_s": sum(wall.values()),
        "determinism_c100_vs_j2_2097505": {
            "selected_model_sha256": J2_SELECTED_SHA,
            "equal_to_c100": results["items"][c100]["checkpoint_sha256"] == J2_SELECTED_SHA,
            "dev_mae_eV_equal": 0.13918505609035492},
        "determinism_c500_vs_j5": "not comparable: j5 of job 2097505 is paused (step 124992, dev 0.10276388376951218)"}
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    files = ["protocol.md", "decision.md", "results.json", "retrieve_legacy_outputs.py", "score_dev_and_distance.py"]
    roles = []
    for name, start, stop in (("dev_100k", 100_000, 150_000), ("dev_500k", 500_000, 550_000)):
        for kind in ("prediction_input", "labels_read", "metric_computed", "selection_used"):
            roles.append({"access_kind": kind, "action_id": ACTION, "dataset_identity": "pcqm4mv2-ogb-official-train",
                          "evidence_ref": f"{REL}/acceptance.json", "role_event_id": f"role-gptdft-{name}-{kind}",
                          "role_name": name, "row_manifest_hash": manifest_hash(start, stop), "run_id": RUN,
                          "schema": "molgap-role-event-v1", "selection_used": True, "trajectory_id": TRAJ})
    hours = sum(wall.values()) / 3600
    costs = [{"action_id": ACTION, "attempt_id": "gptdft-chain-20261121-v1", "category": "training",
              "cost_event_id": "cost-gptdft-chain-training", "evidence_ref": f"{REL}/acceptance.json",
              "hardware": "NVIDIA A100 FP32", "measurement": {
                  "cpu_hours": {"status": "measurement_missing", "value": None},
                  "device_hours": {"status": "measured", "value": hours},
                  "queue_hours": {"status": "measurement_missing", "value": None},
                  "wall_hours": {"status": "measured", "value": hours}},
              "platform": "ims", "run_id": RUN, "schema": "molgap-cost-event-v1", "trajectory_id": TRAJ}]
    role_use = {"common": "untouched", "dev_100k": "consumed", "dev_500k": "consumed", "official_validation": "untouched",
                "ood": "untouched", "tail_holdout_v1": "untouched", "test_challenge": "untouched", "test_dev": "untouched"}
    outcome = {"artifact_status": "local_hash_verified", "budget_decision": "no_tail_no_adoption",
               "comparison_status": "paired_dev_rows_3d_supervised_separate_contract", "execution_status": "complete",
               "full_handoff_status": "not_applicable", "scientific_status": "INCONCLUSIVE_AT_500K_100K_GAIN_NOT_RETAINED",
               "transfer_status": "100k_gain_not_retained_at_500k"}
    facts = {"ci_meV": b500["ci95_mev"], "effect_meV": b500["delta_mev"], "failure_mode": None,
             "format": "molgap-closure-facts-v1", "source": f"{REL}/results.json",
             "search_terms": ["dft distance", "3d supervised", "auxiliary head", "gptrans pair norm", "500k", "noisy nodes pattern"]}
    decision = {"decision_ref": f"{REL}/decision.md", "facts": facts, "next_allowed_actions": [], "outcome": "INCONCLUSIVE",
                "reopen_conditions": ["New user authority for another distance-head arm or for coordinates beyond row 499,999."]}
    acceptance = {"analysis_ref": f"{REL}/results.json", "comparison_class": "CONTEXT_ONLY",
                  "cost_scope": "Sum of per-item A100 wall from serial_summary.json; queue and CPU hours not measured.",
                  "costs": costs, "evidence_id": "pcqm-gptrans-dft-distance-aux-20261123",
                  "format": "molgap-local-saved-prediction-acceptance-v1", "outcome": outcome,
                  "role_counts": {"dev_100k": 50000, "dev_500k": 50000}, "role_use": role_use, "roles": roles,
                  "run_id": RUN, "scope": "GPT-T plus train-only DFT distance head, 100K and 500K, dev rows only; tail not scored.",
                  "trajectory_decision": decision}
    (HERE / "acceptance.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {f"{REL}/acceptance.json": sha(HERE / "acceptance.json")}
    artifacts = []
    for name in files:
        hashes[f"{REL}/{name}"] = sha(HERE / name)
        artifacts.append({"availability": "locally_retained_hash_verified", "locator": f"{REL}/{name}", "name": name,
                          "sha256": hashes[f"{REL}/{name}"]})
    artifacts.append({"availability": "locally_retained_hash_verified", "locator": f"{REL}/acceptance.json",
                      "name": "acceptance.json", "sha256": hashes[f"{REL}/acceptance.json"]})
    external = retrieved / "analysis" / "final" / "dev_analysis.json"
    uri = "external:///" + str(external).replace("\\", "/")
    hashes[uri] = sha(external)
    artifacts.append({"availability": "machine_local_verified", "locator": uri, "name": "dev_analysis.json", "sha256": hashes[uri]})
    terminal = {"acceptance_ref": f"{REL}/acceptance.json", "action_id": ACTION, "artifact_hashes": hashes, "costs": costs,
                "decision": decision,
                "evidence": {"artifacts": artifacts, "authority": {"pointers": [f"{REL}/protocol.md", f"{REL}/results.json",
                             f"{REL}/acceptance.json", f"{REL}/decision.md"]},
                             "contract": "MOLGAP-COMMON-V5-FINAL", "evidence_id": acceptance["evidence_id"],
                             "format": "molgap-v5-evidence-envelope-v1", "legacy_contract": "pcqm-gptrans-dft-distance-aux-v1",
                             "migration": {"inference_executed": False, "migrated_at": "2026-11-23", "scientific_reinterpretation": False,
                                           "training_executed": False,
                                           "verification_scope": "Metadata closure; the executions are recorded under observed_execution."},
                             "observed_execution": {"backbone_training": True, "execution_ref": f"{REL}/results.json",
                                                    "inference_executed": True, "postprocessor_fit": False},
                             "outcome": outcome, "role_use": role_use, "scope": "pcqm-gptrans-dft-distance-aux", "track": "B"},
                "finalized_at": NOW, "format": "molgap-rml-terminal-package-v1", "roles": roles, "run_id": RUN, "trajectory_id": TRAJ}
    (HERE / "terminal.json").write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
