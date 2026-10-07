"""Publish the distinct complete-audit question using shared RML planning."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
RUN = "local-k1-complete-module-audit-20261008"
TID = "TB-k1-complete-module-audit-20261008"
POLICY = "pcqm-k1-complete-module-audit"


def main():
    if (HERE / "rml").exists():
        raise FileExistsError("Prospective already exists")
    prior = json.loads((ROOT / "experiments/pcqm_k1_late_weight_average/inputs.json").read_text())
    artifacts = {k:v for k,v in prior["artifacts"].items() if k in
                 ["selected_checkpoint", "train_probe", "development_probe", "gptrans_predictions"]}
    cpu = Path("D:/文档/molgap/experiments/pcqm_k1_500k_bn_calibration/results")
    for key,name in [("selected_clean_buffers","calibrated_buffers.pt"),("selected_clean_predictions","calibrated.pt")]:
        artifacts[key] = {"path": str(cpu/name), "sha256": sha256_file(cpu/name)}
    for item in artifacts.values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Accepted input changed")
    files = [f"{REL}/prepare.py", f"{REL}/run.py", "src/molgap/k1_component_diagnostic.py",
             "src/molgap/k1_frozen_inference.py"]
    inputs = {"artifacts": artifacts, "frozen_source_root": prior["frozen_source_root"],
              "source_files": prior["source_files"], "source_sha256": prior["source_sha256"],
              "train_source_idx": prior["train_source_idx"], "development_source_idx": prior["development_source_idx"],
              "cpu_threads": 4, "ceiling_seconds": 600, "dev_sample_seed": 20261009,
              "dev_sample_rows": 4096, "prefix_train_rows": 1024, "extension_train_rows": 1024,
              "executed_source_files": {n:sha256_file(ROOT/n) for n in files}}
    atomic_json(HERE/"inputs.json", inputs)
    atomic_json(HERE/"roles.json", {"train_graph_labels": "16384 decoded retained members; no loss",
              "train_features": "1024 prefix and1024 extension descriptive input",
              "internal_development_decoded": "50000 retained labels and baseline predictions decoded",
              "internal_development": "4096 stratified diagnostic input/metrics; historically selection-used",
              "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched", "common": "untouched", "ood": "untouched"})
    policy = json.loads((ROOT/"research_memory/policies/pcqm-k1-bn-mechanism-a100.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY+"-v1"}, created_from_source_digest=sha256_file(HERE/"protocol.md"))
    atomic_json(ROOT/f"research_memory/policies/{POLICY}.1.json", policy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,text=True).strip()
    refs = ["pcqm-k1-500k-bn-calibration-20261007", "pcqm-k1-bn-mechanism-a100-20261008", "pcqm-k1-500k-bottleneck-diagnostic-20261007", "pcqm-k1-colab-execution-profile-20261007"]
    cost_id = "cost-k1-complete-module-audit-expected"
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID, "record_mode": "prospective", "owner": "desktop", "track": "B", "family_id": "k1-complete-module-audit",
       "question": "Which frozen K1 paths remain useful after clean BN, and which retained mechanisms justify a next route?",
       "hypothesis": {"hypothesis_id": "H-k1-complete-module-audit-20261008", "observed_deficiency": "K1 remains slower and above contextual GPTrans EMA after BN repair; isolated histories do not form a complete causal explanation.",
          "supporting_evidence_ids": refs, "alternative_explanations": ["Every path remains coadapted and useful", "State/selection effects dominate", "Single-slot dilution is structural but not empirically limiting"],
          "changed_mechanism": "Fixed half-scale component sensitivities and edge-memory reset, passive complete-module observations; no weights learned",
          "cheapest_falsifier": "One stratified6K baseline and14development4K controls with retained historical evidence",
          "decision_changed_if_positive": "Rank a precise training discriminator from supported dependency/state evidence; NO_TRAIN",
          "decision_changed_if_negative": "Reject unsupported module narratives and list unresolved causal controls; NO_TRAIN",
          "expected_native_cost_ref": cost_id, "related_closed_family_ids": ["k1-500k-bn-calibration", "k1-bn-mechanism-a100", "k1-500k-frozen-bottleneck-diagnostic"]},
       "state_at_start": {"source_commit": commit, "source_config_identity": sha256_file(HERE/"inputs.json"), "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json"],
          "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": refs, "role_snapshot_refs": [f"{REL}/roles.json"], "budget_snapshot_ref": f"{REL}/protocol.md"},
       "actions": [{"action_id": "A001", "type": "local_frozen_module_audit", "run_ids": [RUN], "attempt_ids": ["attempt-001"], "source_commit": commit,
          "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}], "result": {"evidence_ids": [], "evidence_refs": []},
       "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md", "next_allowed_actions": ["One bounded CPU module audit"], "reopen_conditions": []},
       "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md", "comparison_blockers": ["Coadapted frozen perturbations do not estimate retraining benefit", "Consumed selection cohort and exploratory controls"], "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001", "platform": "local-windows",
       "hardware": "CPU four threads; no accelerator", "category": "inference", "evidence_ref": f"{REL}/protocol.md", "measurement": {"wall_hours": {"value": 600/3600, "status": "estimated"},
       "cpu_hours": {"value": None, "status": "measurement_missing"}, "device_hours": {"value": None, "status": "not_applicable"}, "queue_hours": {"value": None, "status": "not_applicable"}}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {"available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC", "policy_id": POLICY,
       "policy_version": "1", "state_timestamp": datetime.now(timezone.utc).isoformat(), "source_commit": commit, "budget_snapshot_ref": f"{REL}/protocol.md", "role_snapshot_refs": [f"{REL}/roles.json"]}}
    atomic_json(HERE/"plan_input.json", spec)
    atomic_json(HERE/"plan_receipt.json", plan(ROOT,spec,f"{REL}/rml"))
    print("PROSPECTIVE_PUBLISHED_NO_MODEL_EXECUTED")


if __name__ == "__main__":
    main()
