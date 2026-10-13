"""Freeze prospective synthetic profiling with the existing RML planner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file

REL = "experiments/family_checkpoint_efficiency"
TID = "TB-family-checkpoint-efficiency-20261013"
RUN = "family-checkpoint-efficiency-20261013"


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / REL
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = subprocess.check_output(["git", "ls-files", "src/molgap", REL], cwd=root, text=True).splitlines()
    config = {"source_commit": commit,
        "source_sha256": {name: sha256_file(root / name) for name in paths if name.endswith(".py")},
        "devices": ["cpu", "cuda"], "families": ["neural_atom_k1", "gptrans_t"],
        "seed": 42, "cpu_threads": 2, "repeats": 5,
        "cursor_lengths": [0, 512, 2048, 3905], "publish_cursor_lengths": [512, 3905],
        "accelerator": "NVIDIA GeForce RTX 5060", "total_wall_seconds": 1200,
        "gpu_wall_seconds": 600, "trajectory_ref": f"{REL}/rml/trajectory.json"}
    atomic_json(folder / "config.json", config)
    prior = ["pcqm-input-pipeline-local-20261013", "pcqm-k1-colab-execution-profile-20261007"]
    trajectory = {"trajectory_id": TID, "track": "B", "owner": "desktop",
        "family_id": "k1-gptrans-checkpoint-execution", "question": "Can finite validation stop synchronizing per checkpoint tensor without changing state?",
        "hypothesis": {"hypothesis_id": "H-family-checkpoint-efficiency-20261013",
            "supporting_evidence_ids": prior,
            "observed_deficiency": "Nested GPU checkpoint checks synchronize per tensor; CPU scalar checks repeat individual kernels",
            "changed_mechanism": "Shared per-device flag transfer and same-dtype CPU scalar checks",
            "cheapest_falsifier": "CPU continuation tests plus matched synthetic checkpoint validation/publication timing and byte identity",
            "alternative_explanations": ["Atomic serialization dominates", "Core GPU work dominates epochs",
                "Shared model-only checks already batch transfers", "CPU grouping overhead offsets benefit"],
            "decision_changed_if_positive": "Integrate only the shared finite checker, no scientific/runtime release",
            "decision_changed_if_negative": "Retain original behavior; record no material checkpoint gain",
            "expected_native_cost_ref": "cost-family-checkpoint-estimate-20261013",
            "related_closed_family_ids": ["k1-gps9-input-pipeline", "k1-execution-profile-a100"]},
        "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(config),
            "contract_refs": [f"{REL}/protocol.md", f"{REL}/config.json"], "reference_ids": [],
            "parent_trajectory_ids": [], "prior_evidence_ids": prior,
            "role_snapshot_refs": [f"{REL}/role_plan.json"], "budget_snapshot_ref": f"{REL}/protocol.md"},
        "actions": [{"action_id": "A001", "type": "execution_profile", "run_ids": [RUN],
            "attempt_ids": ["local-001"], "source_commit": commit,
            "evidence_refs": [f"{REL}/config.json"], "cost_event_ids": ["cost-family-checkpoint-estimate-20261013"]}],
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md",
            "next_allowed_actions": ["Authorized bounded synthetic local checkpoint diagnostic"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md",
        "comparison_blockers": ["No molecular quality endpoint", "Synthetic local checkpoint timing only"],
        "reference_bundle_id": None}
    spec = {"trajectory": trajectory, "decision_state": {"available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"],
        "chosen_action": "RUN_DIAGNOSTIC", "policy_id": "family-checkpoint-efficiency", "policy_version": "1",
        "state_timestamp": datetime.now(timezone.utc).isoformat()},
        "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": "cost-family-checkpoint-estimate-20261013",
            "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "local-001",
            "platform": "local-windows", "hardware": "CPU and RTX5060", "category": "audit",
            "evidence_ref": f"{REL}/protocol.md", "measurement": {
                "wall_hours": {"value": 1200 / 3600, "status": "estimated"},
                "device_hours": {"value": 600 / 3600, "status": "estimated"},
                "cpu_hours": {"value": None, "status": "measurement_missing"},
                "queue_hours": {"value": None, "status": "not_applicable"}}}]}
    receipt = plan(root, spec, f"{REL}/rml")
    atomic_json(folder / "plan_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
