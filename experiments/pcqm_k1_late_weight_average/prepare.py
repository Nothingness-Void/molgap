"""Bind existing artifacts and publish through the existing RML planner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
RUN = "local-k1-late-weight-average-20261008"
TID = "TB-k1-late-weight-average-20261008"
POLICY = "pcqm-k1-late-weight-average"


def main():
    if (HERE / "rml").exists():
        raise FileExistsError("Prospective already exists")
    prior = json.loads((ROOT / "experiments/pcqm_k1_500k_bottleneck_diagnostic/inputs.json").read_text())
    bn = json.loads((ROOT / "experiments/pcqm_k1_bn_mechanism_a100/payload_manifest.json").read_text())
    payload = Path("D:/w/k1-bn-mechanism/platforms/_records/colab/staging/k1-bn-mechanism-a100-20261008/payload")
    selected = prior["checkpoints"]["best_model.pt"]
    final = prior["checkpoints"]["last_checkpoint.pt"]
    cpu_results = Path("D:/文档/molgap/experiments/pcqm_k1_500k_bn_calibration/results")
    gp = Path(selected["path"]).parent.parent / "gptrans_g1_bond_local_ema999/best_predictions.pt"
    artifacts = {"selected_checkpoint": selected, "final_checkpoint": final}
    for key, path in {"train_probe": payload / "train_probe.pt", "development_probe": payload / "development_probe.pt",
                      "selected_original": cpu_results / "original.pt", "selected_clean_bn": cpu_results / "calibrated.pt",
                      "gptrans_predictions": gp}.items():
        artifacts[key] = {"path": str(path), "sha256": sha256_file(path)}
    for item in artifacts.values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Accepted input bytes changed")
    if any(artifacts[k]["sha256"] != bn["files"][n] for k,n in
           [("train_probe", "train_probe.pt"), ("development_probe", "development_probe.pt")]):
        raise ValueError("Accepted graph binding changed")
    cpu = json.loads((cpu_results / "analysis.json").read_text())
    if cpu["status"] != "complete" or cpu["runtime"]["torch"].split("+")[0] != "2.7.1":
        raise ValueError("Accepted CPU comparator runtime differs")
    source_root = ROOT / "experiments/pcqm_k1_bn_mechanism_a100/frozen_source/src"
    source_files = {n:h for n,h in bn["files"].items() if n.startswith("src/molgap/")}
    for n,h in source_files.items():
        if sha256_file(source_root.parent / n) != h:
            raise ValueError("Accepted frozen model source changed")
    executed = {n:sha256_file(ROOT/n) for n in [f"{REL}/run.py", f"{REL}/prepare.py",
                "src/molgap/k1_weight_average_diagnostic.py", "src/molgap/k1_bn_calibration.py", "src/molgap/k1_frozen_inference.py"]}
    inputs = {"artifacts": artifacts, "frozen_source_root": str(source_root), "source_files": source_files,
              "source_sha256": "0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a",
              "final_source_sha256": prior["archive"]["sha256"],
              "train_source_idx": bn["train_source_idx"], "development_source_idx": bn["development_source_idx"],
              "cpu_threads": 4, "ceiling_seconds": 600, "executed_source_files": executed,
              "parameter_mean": [0.5, 0.5], "true_stepwise_ema": False,
              "accepted_cpu_result_sha256": sha256_file(cpu_results/"analysis.json")}
    atomic_json(HERE/"inputs.json", inputs)
    atomic_json(HERE/"roles.json", {"train_features": "16384 retained training graphs; labels decoded, no objective",
                "internal_development": "50000 previously selection-used rows; labels/predictions/metrics",
                "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
                "common": "untouched", "ood": "untouched"})
    policy = json.loads((ROOT/"research_memory/policies/pcqm-k1-bn-mechanism-a100.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY+"-v1"},
                  created_from_source_digest=sha256_file(HERE/"protocol.md"))
    atomic_json(ROOT/f"research_memory/policies/{POLICY}.1.json", policy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    refs = ["pcqm-k1-500k-bn-calibration-20261007", "pcqm-k1-bn-mechanism-a100-20261008", "pcqm-k1-500k-bottleneck-diagnostic-20261007"]
    cost_id = "cost-k1-late-weight-average-expected"
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID, "record_mode": "prospective", "owner": "desktop", "track": "B",
       "family_id": "k1-late-weight-average", "question": "Does fixed epoch49/60 parameter averaging improve K1 after identical clean BN calibration?",
       "hypothesis": {"hypothesis_id": "H-k1-late-weight-average-20261008", "observed_deficiency": "Clean BN K1 remains1.771meV above contextual GPTrans EMA; no stepwise K1 EMA retained.",
          "supporting_evidence_ids": refs, "alternative_explanations": ["Late weights have no useful averaging gain", "BN drift explains late raw deterioration", "Stepwise EMA differs from sparse endpoint averaging"],
          "changed_mechanism": "Equal learned-parameter mean of two retained endpoints, controlled with identical clean BN",
          "cheapest_falsifier": "Four local CPU frozen-state inferences plus reused accepted selected comparators, no training",
          "decision_changed_if_positive": "Nominate this exact late averaging test if >=1meV positive95% row lower; close NO_TRAIN",
          "decision_changed_if_negative": "Deprioritize this exact sparse average; true EMA remains unresolved; close NO_TRAIN",
          "expected_native_cost_ref": cost_id, "related_closed_family_ids": ["k1-500k-bn-calibration", "k1-bn-mechanism-a100"]},
       "state_at_start": {"source_commit": commit, "source_config_identity": sha256_file(HERE/"inputs.json"),
          "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json"], "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": refs,
          "role_snapshot_refs": [f"{REL}/roles.json"], "budget_snapshot_ref": f"{REL}/protocol.md"},
       "actions": [{"action_id": "A001", "type": "local_frozen_parameter_averaging", "run_ids": [RUN], "attempt_ids": ["attempt-001"], "source_commit": commit,
                    "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}],
       "result": {"evidence_ids": [], "evidence_refs": []},
       "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md", "next_allowed_actions": ["One bounded local CPU averaging diagnostic"], "reopen_conditions": []},
       "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md", "comparison_blockers": ["Consumed checkpoint-selection cohort", "Sparse endpoint mean is not stepwise EMA"], "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": TID, "action_id": "A001", "run_id": RUN,
            "attempt_id": "attempt-001", "platform": "local-windows", "hardware": "CPU four intra-op threads; no accelerator", "category": "inference", "evidence_ref": f"{REL}/protocol.md",
            "measurement": {"wall_hours": {"value": 600/3600, "status": "estimated"}, "cpu_hours": {"value": None, "status": "measurement_missing"},
                            "device_hours": {"value": None, "status": "not_applicable"}, "queue_hours": {"value": None, "status": "not_applicable"}}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {"available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
            "policy_id": POLICY, "policy_version": "1", "state_timestamp": datetime.now(timezone.utc).isoformat(),
            "budget_snapshot_ref": f"{REL}/protocol.md", "role_snapshot_refs": [f"{REL}/roles.json"], "source_commit": commit}}
    atomic_json(HERE/"plan_input.json", spec)
    atomic_json(HERE/"plan_receipt.json", plan(ROOT,spec,f"{REL}/rml"))
    print("PROSPECTIVE_PUBLISHED_NO_MODEL_EXECUTED")


if __name__ == "__main__":
    main()
