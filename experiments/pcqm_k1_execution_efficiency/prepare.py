"""Freeze the existing train-only inputs and prospective before any execution."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-execution-efficiency-20261010"
RUN = "k1-execution-efficiency-local-20261010"
POLICY = "pcqm-k1-execution-efficiency"


def main():
    if (HERE / "rml").exists():
        raise FileExistsError("Prospective already frozen; do not replace it")
    prior = ROOT / "experiments/pcqm_k1_colab_execution_profile"
    manifest = json.loads((prior / "payload_manifest.json").read_text())
    retained = Path("D:/w/k1-colab-profile/platforms/_records/colab/staging/k1-profile-a100-20261007/payload")
    initial = Path("D:/w/k1-colab-500k-efficient/platforms/_records/colab/staging/k1-single-ema-500k-a100-20261010/initial_state.pt")
    bindings = {
        "train_probe": {"path": str(retained / "train_probe.pt"), "sha256": manifest["files"]["train_probe.pt"]},
        "normalization_checkpoint": {"path": str(retained / "selected.pt"), "sha256": manifest["files"]["selected.pt"]},
        "initial": {"path": str(initial), "sha256": "0ce2cb3206a7a59a059b5efb09922a5d6e4478058ce2dbfec1434f3086fb5a80",
                    "tensor_sha256": "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"},
    }
    for entry in bindings.values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Retained input hash differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = ("src/molgap/k1_execution_efficiency.py", "src/molgap/k1_execution_layout.py",
             "src/molgap/qm9_neural_atom.py", "src/molgap/gps.py",
             "src/molgap/pcqm_gap_architecture.py", "src/molgap/edge_state_gps.py",
             "src/molgap/k1_screen_training.py", "src/molgap/pcqm_gptrans_v4.py",
             "src/molgap/v4_runtime.py", "src/molgap/pcqm_k1_scale.py",
             "src/molgap/pcqm_500k_v4_evidence.py", "src/molgap/training_reproducibility.py",
             f"{REL}/run.py", f"{REL}/protocol.md")
    inputs = {"source_commit": head, "files": {f: sha256_file(ROOT / f) for f in files},
              "bindings": bindings, "source_idx": manifest["sample_source_idx"],
              "manifest_sha256": "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751",
              "worker_wall_limit_seconds": 1800, "seed": 42, "batch_size": 128,
              "timing": {"warmup": 3, "steps": 12, "repeats": 4},
              "equivalence_steps": 3}
    atomic_json(HERE / "inputs.json", inputs)
    atomic_json(HERE / "role_plan.json", {"training_sample": {"rows": 4096,
        "source_idx_range": [0, 500000], "selection_used": False},
        "development": "untouched", "official_validation": "untouched",
        "test_dev": "untouched", "test_challenge": "untouched"})
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-colab-execution-profile.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY + "-v1"},
                  created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    refs = ["pcqm-k1-colab-execution-profile-20261007"]
    cost_id = "cost-k1-execution-efficiency-local-estimate"
    state = {"source_commit": head, "source_config_identity": sha256_file(HERE / "inputs.json"),
             "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json", f"{REL}/evidence_review.md"],
             "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": refs,
             "role_snapshot_refs": [f"{REL}/role_plan.json"], "budget_snapshot_ref": f"{REL}/protocol.md"}
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID,
        "record_mode": "prospective", "track": "B", "owner": "desktop", "family_id": "k1-execution-efficiency",
        "question": "Which K1 execution changes reduce scratch-step cost without changing tensors or RNG?",
        "hypothesis": {"hypothesis_id": "H-k1-execution-efficiency-20261010", "supporting_evidence_ids": refs,
            "observed_deficiency": "Small operators and repeated host/device shape synchronizations may limit the current FP32 K1 loop.",
            "alternative_explanations": ["Local edge computation dominates", "Existing guards still synchronize", "Windows local runtime is not T4"],
            "changed_mechanism": "Opt-in CPU dense layout, nonblocking transfer and redundant-barrier removal; optimizer paths diagnosed separately",
            "cheapest_falsifier": "Three exact scratch steps then repeated short BS128 train-only timing windows",
            "decision_changed_if_positive": "Nominate an exactly equivalent execution addon for separate native runtime qualification",
            "decision_changed_if_negative": "Keep default execution and record the missing bottleneck discriminator",
            "expected_native_cost_ref": cost_id, "related_closed_family_ids": ["k1-execution-profile-a100"]},
        "state_at_start": state, "actions": [{"action_id": "A001", "type": "execution_profile",
            "run_ids": [RUN], "attempt_ids": ["attempt-001"], "source_commit": head,
            "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md",
                     "next_allowed_actions": ["Bounded local execution-only diagnostic when GPU idle"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md",
        "comparison_blockers": ["Execution diagnostic is not scientific endpoint comparison", "RTX5060 is not T4/A100"],
        "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": TID,
        "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001", "platform": "local",
        "hardware": "NVIDIA GeForce RTX 5060", "category": "preflight", "evidence_ref": f"{REL}/protocol.md",
        "measurement": {key: {"value": 0.5 if key in ("wall_hours", "device_hours") else None,
            "status": "estimated" if key in ("wall_hours", "device_hours") else "measurement_missing"}
            for key in ("device_hours", "wall_hours", "cpu_hours", "queue_hours")}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {
        "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
        "available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
        "policy_id": POLICY, "policy_version": "1", "budget_snapshot_ref": f"{REL}/protocol.md",
        "role_snapshot_refs": [f"{REL}/role_plan.json"], "source_commit": head,
        "state_timestamp": datetime.now(timezone.utc).isoformat()}}
    atomic_json(HERE / "plan_input.json", spec)
    atomic_json(HERE / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print(TID + " prospective frozen")


if __name__ == "__main__":
    main()
