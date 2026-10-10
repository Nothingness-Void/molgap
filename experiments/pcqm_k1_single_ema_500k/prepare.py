"""Bind this question with the existing RML planner and source-bundle owner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
RUN = "k1-single-ema-500k-a100-20261010"
TID = "TB-k1-single-ema-500k-a100-20261010"
POLICY = "pcqm-k1-single-ema-500k-a100"
MANIFEST = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"


def main():
    from molgap.research_memory.plan import plan
    from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file
    from molgap.k1_screen_training import build_initial_state
    from molgap.v4_runtime import state_dict_sha256
    from molgap.v4_bundle import build_v4_source_bundle

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    staging = ROOT / "platforms/_records/colab/staging" / RUN
    if staging.exists() or (HERE / "rml").exists():
        raise ValueError("Reconcile existing preparation; never overwrite frozen records")
    staging.mkdir(parents=True)
    # Initialization construction is CPU packaging, not an inference diagnostic.
    initial = build_initial_state(42)
    initial_digest = state_dict_sha256(initial)
    atomic_torch_save(staging / "initial_state.pt", {
        "format": "molgap-k1-single-ema-initial-v1", "model_state": initial,
        "state_sha256": initial_digest})
    inputs = {"source_commit": commit, "run_id": RUN, "manifest_sha256": MANIFEST,
              "dataset": "nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1",
              "initial_file_sha256": sha256_file(staging / "initial_state.pt"),
              "initial_tensor_sha256": initial_digest,
              "seed": 42, "allocation_wall_ceiling_seconds": 14400,
              "arms": ["reference", "ema999"], "ema_decay": 0.999,
              "full_schedule_epochs": 60, "batch_size": 128,
              "bn_calibration_members": {"start": 0, "stop": 16384}}
    atomic_json(HERE / "inputs.json", inputs)
    atomic_json(HERE / "role_plan.json", {
        "train": {"bounds": [0, 500000], "labels_read": 500000},
        "internal_development": {"bounds": [500000, 550000], "previously_selection_used": True},
        "bn_calibration": {"bounds": [0, 16384], "labels_used_for_calibration": False},
        "official_validation": "prohibited", "test_dev": "prohibited",
        "test_challenge": "prohibited", "common_ood_p8hard": "prohibited"})
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-colab-execution-profile.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY + "-v1"},
                  created_from_source_digest=sha256_file(HERE / "protocol.md"))
    policy["action_rule"]["action"] = "RUN_BOUNDED_500K"
    atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    refs = ["pcqm-k1-bn-mechanism-a100-20261008", "pcqm-k1-colab-execution-profile-20261007"]
    cost_id = "cost-k1-single-ema-500k-a100-planned"
    trajectory = {
        "schema": "molgap-trajectory-v1", "trajectory_id": TID, "record_mode": "prospective",
        "track": "B", "owner": "desktop", "family_id": "k1-single-ema-clean-bn-500k",
        "question": "Does stepwise EMA improve matched500K single-forward K1 with clean BN within4 A100 hours?",
        "hypothesis": {
            "hypothesis_id": "H-k1-single-ema-500k-a100-20261010", "supporting_evidence_ids": refs,
            "observed_deficiency": "Single saves work but failed100K quality noninferiority; frozen BN state has recoverable500K error.",
            "alternative_explanations": ["EMA does not help", "EMA/BN mismatch", "partial-horizon slow startup", "seed uncertainty"],
            "changed_mechanism": "Stepwise parameter EMA0.999; identical single-forward training and clean BN on both arms",
            "cheapest_falsifier": "One matched-prefix500K pair under4 total A100 hours",
            "decision_changed_if_positive": "Nominate separate reviewed qualification, never automatically adopt or scale",
            "decision_changed_if_negative": "Retain failure or partial ambiguity; no automatic retry",
            "expected_native_cost_ref": cost_id,
            "related_closed_family_ids": ["k1-bn-mechanism-a100", "k1-late-weight-average", "k1-mean2-clean-second"]},
        "state_at_start": {"source_commit": commit, "source_config_identity": sha256_file(HERE / "inputs.json"),
            "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json", f"{REL}/evidence_review.md"],
            "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": refs,
            "role_snapshot_refs": [f"{REL}/role_plan.json"], "budget_snapshot_ref": f"{REL}/protocol.md"},
        "actions": [{"action_id": "A001", "type": "bounded_paired_500k_training",
            "run_ids": [RUN + ":reference", RUN + ":ema999"], "attempt_ids": ["attempt-001"],
            "source_commit": commit, "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md",
                     "next_allowed_actions": ["One four-hour Colab A100 pair"], "reopen_conditions": []},
        "comparison_class": "NO_COMPARISON", "comparison_readiness_ref": f"{REL}/protocol.md",
        "comparison_blockers": ["Native qualification/reference not yet observed", "Partial screen is not completed endpoint", "Consumed development"],
        "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": TID,
            "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001", "platform": "colab",
            "hardware": "NVIDIA A100", "category": "training", "evidence_ref": f"{REL}/protocol.md",
            "measurement": {k: {"value": 4.0 if k in ("device_hours", "wall_hours") else None,
                                 "status": "estimated" if k in ("device_hours", "wall_hours") else "measurement_missing"}
                            for k in ("device_hours", "wall_hours", "cpu_hours", "queue_hours")}}
    specification = {"trajectory": trajectory, "costs": [cost], "decision_state": {
        "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
        "available_actions": ["RUN_BOUNDED_500K", "NO_TRAIN"], "chosen_action": "RUN_BOUNDED_500K",
        "policy_id": POLICY, "policy_version": "1", "budget_snapshot_ref": f"{REL}/protocol.md",
        "role_snapshot_refs": [f"{REL}/role_plan.json"], "source_commit": commit,
        "state_timestamp": datetime.now(timezone.utc).isoformat()}}
    atomic_json(HERE / "plan_input.json", specification)
    atomic_json(HERE / "plan_receipt.json", plan(ROOT, specification, f"{REL}/rml"))
    files = json.loads((HERE / "source_allowlist.json").read_text())
    bundle = build_v4_source_bundle(repo_root=ROOT, relative_paths=files,
                                   output_dir=staging / "source", source_commit=commit)
    for name in ("inputs.json", "protocol.md", "role_plan.json", "evidence_review.md"):
        shutil.copy2(HERE / name, staging / name)
    shutil.copytree(HERE / "rml", staging / "prospective")
    atomic_json(HERE / "submission/package_binding.json", {
        "run_id": RUN, "source": bundle, "inputs_sha256": sha256_file(HERE / "inputs.json"),
        "manifest_sha256": MANIFEST,
        "initial_file_sha256": inputs["initial_file_sha256"],
        "initial_tensor_sha256": inputs["initial_tensor_sha256"]})
    shutil.copy2(HERE / "submission/package_binding.json", staging / "package_binding.json")
    print(json.dumps({"prepared": True, "staging": str(staging), "source": bundle}))


if __name__ == "__main__":
    main()
