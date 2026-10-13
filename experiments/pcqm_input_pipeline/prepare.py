"""Publish the bounded prospective diagnostic using the existing RML owner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file

REL = "experiments/pcqm_input_pipeline"


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / REL
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    names = subprocess.check_output(["git", "ls-files", "src/molgap"], cwd=root, text=True).splitlines()
    config = {
        "source_commit": commit,
        "source_sha256": {name: sha256_file(root / name) for name in names if name.endswith(".py")},
        "input_root": "D:/\u6587\u6863/molgap/data/pcqm_fixed_100k_v1",
        "manifest": "manifest.json",
        "manifest_sha256": "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
        "train_rows": 100000,
        "initial_path": "D:/w/k1-colab-500k-efficient/platforms/_records/colab/staging/k1-single-ema-500k-a100-20261010/initial_state.pt",
        "initial_file_sha256": "0ce2cb3206a7a59a059b5efb09922a5d6e4478058ce2dbfec1434f3086fb5a80",
        "initialization_sha256": "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd",
        "target_mean_eV": 5.3383002281188965,
        "target_std_eV": 1.275090217590332,
        "seed": 42, "window_batches": 20, "cpu_threads": 2,
        "cpu_workers": [0, 1], "cpu_repeats": 2,
        "gpu_families": ["k1", "gps9"], "gpu_workers": 1,
        "gpu_rows": 8192, "gpu_warmup_steps": 8, "gpu_repeats": 3,
        "accelerator": "NVIDIA GeForce RTX 5060",
        "total_wall_seconds": 1200, "gpu_wall_seconds": 600,
        "trajectory_ref": f"{REL}/rml/trajectory.json",
    }
    if sha256_file(Path(config["input_root"]) / config["manifest"]) != config["manifest_sha256"]:
        raise ValueError("Pinned local manifest differs")
    if sha256_file(Path(config["initial_path"])) != config["initial_file_sha256"]:
        raise ValueError("Pinned local initialization differs")
    atomic_json(folder / "config.json", config)
    trajectory = {
        "trajectory_id": "TB-input-pipeline-local-20261013", "track": "B", "owner": "desktop",
        "family_id": "k1-gps9-input-pipeline", "question": "Does block/bucket sampling improve the existing fixed-index input pipeline?",
        "hypothesis": {
            "hypothesis_id": "H-input-pipeline-local-20261013",
            "supporting_evidence_ids": ["pcqm-k1-colab-execution-profile-20261007"],
            "observed_deficiency": "No full100K input traversal or matched bucket throughput observation",
            "changed_mechanism": "Diagnostic-only block and random-window size buckets; existing fixed index",
            "cheapest_falsifier": "TRAIN-only CPU traversal plus bounded scratch RTX5060 update timing",
            "alternative_explanations": ["Input is already negligible", "Lazy graph cache dominates first pass",
                "K1 is not dense atom-atom attention", "Grouping changes BN and training trajectory"],
            "decision_changed_if_positive": "Nominate a separately authorized paired quality test, not adoption",
            "decision_changed_if_negative": "Do not spend full training budget on unmeasured input optimization",
            "expected_native_cost_ref": "cost-input-pipeline-estimate-20261013",
            "related_closed_family_ids": ["k1-execution-profile-a100"],
        },
        "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(config),
            "contract_refs": [f"{REL}/protocol.md", f"{REL}/config.json"], "reference_ids": [],
            "parent_trajectory_ids": [], "prior_evidence_ids": ["pcqm-k1-colab-execution-profile-20261007"],
            "role_snapshot_refs": [f"{REL}/role_plan.json"], "budget_snapshot_ref": f"{REL}/protocol.md"},
        "actions": [{"action_id": "A001", "type": "execution_profile", "run_ids": ["input-pipeline-local-20261013"],
            "attempt_ids": ["local-001"], "source_commit": commit, "evidence_refs": [f"{REL}/config.json"],
            "cost_event_ids": ["cost-input-pipeline-estimate-20261013"]}],
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md",
            "next_allowed_actions": ["Authorized bounded TRAIN-only local diagnostic"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md",
        "comparison_blockers": ["Execution-only scratch probes, no scientific endpoint", "Local RTX5060 only"],
        "reference_bundle_id": None,
    }
    spec = {"trajectory": trajectory, "decision_state": {
        "available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
        "policy_id": "pcqm-input-pipeline-local", "policy_version": "1",
        "state_timestamp": datetime.now(timezone.utc).isoformat()},
        "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": "cost-input-pipeline-estimate-20261013",
            "trajectory_id": trajectory["trajectory_id"], "action_id": "A001", "run_id": "input-pipeline-local-20261013",
            "attempt_id": "local-001", "platform": "local", "hardware": "CPU and optional RTX5060",
            "category": "audit", "evidence_ref": f"{REL}/protocol.md",
            "measurement": {"wall_hours": {"value": 1200 / 3600, "status": "estimated"},
                "device_hours": {"value": 600 / 3600, "status": "estimated"},
                "cpu_hours": {"value": None, "status": "measurement_missing"},
                "queue_hours": {"value": None, "status": "not_applicable"}}}]}
    receipt = plan(root, spec, f"{REL}/rml")
    atomic_json(folder / "plan_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
