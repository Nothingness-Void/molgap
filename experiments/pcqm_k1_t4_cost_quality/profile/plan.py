"""Freeze the new T4-only question through the existing RML planner."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-native-t4-cost-profile-20261009"
RUN = "molgap-k1-native-t4-profile-s42-v1"
POLICY = "pcqm-k1-native-t4-cost-profile"


def main():
    old = ROOT / "experiments/pcqm_k1_colab_execution_profile"
    source = json.loads((old / "plan_input.json").read_text())
    spec = copy.deepcopy(source)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    inputs = {"source_commit": commit, "accepted_profile_manifest": {
        "path": "D:/w/k1-colab-profile/platforms/_records/colab/staging/k1-profile-a100-20261007/payload/payload_manifest.json",
        "sha256": sha256_file(old / "payload_manifest_attempt002.json")},
        "source_overlays": {name: sha256_file(ROOT / name) for name in (
            "src/molgap/k1_execution_profile.py", "src/molgap/evidence_pointers.py")},
        "worker_ceiling_seconds": 900, "allocation_ceiling_seconds": 1200,
        "allocated_device_count": 2, "active_device_count": 1,
        "cost_quality_release": "separate parent review required; no automatic training"}
    path = Path(inputs["accepted_profile_manifest"]["path"])
    if sha256_file(path) != inputs["accepted_profile_manifest"]["sha256"]:
        raise ValueError("Accepted retained sample manifest differs")
    atomic_json(HERE / "inputs.json", inputs)
    atomic_json(HERE / "role_plan.json", {"training": "same accepted4096 drawn rows; timing only",
        "sample_manifest": inputs["accepted_profile_manifest"], "development": "untouched",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"})
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-colab-execution-profile.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY + "-v1"},
        created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    refs = ["pcqm-k1-colab-execution-profile-20261007", "pcqm-k1-complete-module-audit-20261008"]
    trajectory = spec["trajectory"]
    trajectory.update(trajectory_id=TID, family_id="k1-native-t4-cost-profile",
        question="How much native T4 optimizer-step time does single-forward save versus unpenalized mean2?",
        comparison_readiness_ref=f"{REL}/protocol.md",
        comparison_blockers=["execution diagnostic only; no accuracy or full-epoch cost qualification"])
    trajectory["hypothesis"].update(hypothesis_id="H-" + TID, supporting_evidence_ids=refs,
        observed_deficiency="A100 selected-state profiling cannot establish native T4 cost or single-forward quality.",
        alternative_explanations=["T4 operator mix changes step savings", "Evaluation/setup/idle allocations reduce whole-run savings", "Two dropout/BN updates affect quality"],
        changed_mechanism="Native T4 single L1 versus mean of two L1 scratch steps; consistency coefficient0",
        cheapest_falsifier="Same accepted4096 train rows and selected state; two29-step cases,512-row eval and localFS write;900s worker/1200s allocation ceiling",
        decision_changed_if_positive="If >=25% matched step savings, review separately bounded100K cost-quality pair; no automatic release",
        decision_changed_if_negative="Close cost route without100K training; no blind hardware extrapolation",
        expected_native_cost_ref="cost-k1-native-t4-profile-planned", related_closed_family_ids=["k1-execution-profile-a100"])
    state = trajectory["state_at_start"]
    state.update(source_commit=commit, source_config_identity=sha256_file(HERE / "inputs.json"),
        contract_refs=[f"{REL}/protocol.md", f"{REL}/inputs.json"], prior_evidence_ids=refs,
        role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/protocol.md")
    trajectory["actions"] = [{"action_id": "A001", "type": "native_t4_scratch_execution_profile",
        "run_ids": [RUN], "attempt_ids": ["attempt-001"], "source_commit": commit,
        "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": ["cost-k1-native-t4-profile-planned"]}]
    trajectory["decision"].update(decision_ref=f"{REL}/protocol.md",
        next_allowed_actions=["One bounded native T4 diagnostic only"], reopen_conditions=[])
    cost = spec["costs"][0]
    cost.update(cost_event_id="cost-k1-native-t4-profile-planned", trajectory_id=TID, run_id=RUN,
        platform="kaggle", hardware="Two allocated native T4 devices; cuda0 scratch workload",
        evidence_ref=f"{REL}/protocol.md")
    cost["measurement"]["device_hours"].update(value=1200 * 2 / 3600, status="estimated")
    cost["measurement"]["wall_hours"].update(value=1200 / 3600, status="estimated")
    spec["decision_state"].update(policy_id=POLICY, source_commit=commit,
        role_snapshot_refs=state["role_snapshot_refs"], budget_snapshot_ref=state["budget_snapshot_ref"],
        state_timestamp=datetime.now(timezone.utc).isoformat())
    atomic_json(HERE / "plan_input.json", spec)
    atomic_json(HERE / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print(TID)


if __name__ == "__main__":
    main()
