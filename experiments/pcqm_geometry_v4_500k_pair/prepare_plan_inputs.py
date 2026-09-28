"""Write per-arm inputs for the existing RML prospective planner."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from molgap.constants import REPO_ROOT


HERE = Path(__file__).resolve().parent
CONTRACT = "experiments/pcqm_geometry_v4_500k_pair/training_contract.json"
PROTOCOL = "experiments/pcqm_geometry_v4_500k_pair/protocol.md"
DECISION = "experiments/pcqm_geometry_v4_500k_pair/launch_decision.md"
REFERENCE = "pcqm-matched-500k-v4-three-arm"
DATASET_EVIDENCE = "platforms/_records/kaggle/pcqm_fixed_datasets_v1/acceptance.json"
ROLE_REFS = [
    "experiments/pcqm_500k_v4_evidence/roles/train.json",
    "experiments/pcqm_500k_v4_evidence/roles/development.json",
    DATASET_EVIDENCE,
]
KERNEL = "nothingnessvoid/molgap-geometry-v4-500k-paired-s42-v1"


def make_plan(arm: str, source_commit: str, source_sha: str, timestamp: str) -> dict:
    trajectory_id = f"TB-geometry-v4-500k-{arm}-s42"
    cost_id = f"cost-{trajectory_id}-expected-training"
    run_id = f"{KERNEL}/{arm}/attempt-001"
    if arm == "gptrans_distance_only":
        question = "Does bond-distance input improve matched-V4 GPTrans-T by at least 3 meV at 500K?"
        deficiency = "The 100K distance-only gain against a separate pure-2D job is contextual; a matched V4 500K geometry effect remains untested."
        mechanism = "Add zero-start ETKDGv3+MMFF94s real-bond distance projection to GPTrans pair state, with the angle path disabled."
        prior = [REFERENCE, "pcqm-gptrans-geometry-distance-100k-s42-a2"]
        parent = ["TB-gptrans-geometry-distance-100k-s42-a2"]
        positive = "Nominate distance-only GPTrans geometry for a separately frozen scale decision."
        negative = "Close this matched V4 distance-only 500K route without changing seed or schedule."
        threshold = "0.1038675369 eV"
    elif arm == "k1_distance_angle":
        question = "Does matched-V4 K1 distance-angle geometry add a material 500K gain or useful fixed-blend complementarity?"
        deficiency = "The historical geometry K1/GPTrans blend was positive, but K1 geometry lacks a matching V4 reference and runtime certificate."
        mechanism = "Use the existing zero-start distance-angle sparse-wedge K1 path under the matched 60-epoch V4 recipe."
        prior = [REFERENCE, "pcqm-geometry-transfer-500k-contextual"]
        parent = ["TB-geometry-transfer-500k-context"]
        positive = "Nominate K1 geometry or the preregistered fixed 50:50 geometry blend for separate scale review."
        negative = "Close the K1 geometry route if neither its own matched gate nor blend complementarity is material."
        threshold = "0.1018598662 eV"
    else:
        raise ValueError(f"unknown arm {arm}")
    trajectory = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": trajectory_id,
        "record_mode": "prospective", "track": "B", "owner": "desktop",
        "family_id": f"geometry-v4-500k-{arm}",
        "question": question,
        "hypothesis": {
            "hypothesis_id": f"H-{trajectory_id}",
            "observed_deficiency": deficiency,
            "supporting_evidence_ids": prior,
            "alternative_explanations": [
                "Historical geometry gains may reflect optimizer, schedule, exposure or selection differences.",
                "Single-seed development-row bootstrap does not measure training stochasticity.",
            ],
            "changed_mechanism": mechanism,
            "cheapest_falsifier": f"A single matched 500K run with 60 observed epochs missing the 3.0 meV gate ({threshold}) or paired sign gate.",
            "related_closed_family_ids": [
                "gptrans-pair-norm-500k", "gptrans-noisy-pair-norm-500k",
                "gptrans-geometry-angle-increment-100k",
            ],
            "expected_native_cost_ref": cost_id,
            "decision_changed_if_positive": positive,
            "decision_changed_if_negative": negative,
            "historical_unknowns": ["Kaggle1 T4 wall time for this exact geometry arm is unknown."],
        },
        "state_at_start": {
            "source_commit": source_commit,
            "contract_refs": [PROTOCOL, CONTRACT],
            "reference_ids": [REFERENCE],
            "parent_trajectory_ids": parent,
            "prior_evidence_ids": prior,
            "role_snapshot_refs": ROLE_REFS,
            "budget_snapshot_ref": CONTRACT,
            "source_config_identity": source_sha,
        },
        "actions": [{
            "action_id": "A001", "type": "remote_scale_transfer_training",
            "run_ids": [run_id], "attempt_ids": ["attempt-001"],
            "source_commit": source_commit, "evidence_refs": [],
            "cost_event_ids": [cost_id],
        }],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {
            "decision_ref": DECISION, "outcome": "ACTIVE",
            "next_allowed_actions": ["one_kaggle1_two_arm_attempt_after_preflight"],
            "reopen_conditions": [],
        },
    }
    decision_state = {
        "known_trajectory_ids": [], "known_evidence_ids": [],
        "active_reference_ids": [REFERENCE],
        "available_actions": ["RUN_500K", "NO_TRAIN"],
        "chosen_action": "RUN_500K",
        "policy_id": "pcqm-geometry-v4-500k-pair-launch", "policy_version": "1",
        "role_snapshot_refs": ROLE_REFS,
        "budget_snapshot_ref": CONTRACT,
        "state_timestamp": timestamp,
        "source_commit": source_commit,
    }
    cost = {
        "schema": "molgap-cost-event-v1", "cost_event_id": cost_id,
        "trajectory_id": trajectory_id, "action_id": "A001",
        "attempt_id": "attempt-001", "run_id": run_id,
        "category": "training", "platform": "kaggle",
        "hardware": "NvidiaTeslaT4_requested", "evidence_ref": CONTRACT,
        "measurement": {
            name: {"status": "measurement_missing", "value": None}
            for name in ("device_hours", "wall_hours", "queue_hours", "cpu_hours")
        },
    }
    return {"trajectory": trajectory, "decision_state": decision_state, "costs": [cost]}


def main() -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_dir = (REPO_ROOT / "platforms/_records/kaggle/staging/"
                  "pcqm_geometry_v4_500k_pair/source_bundle")
    source_commit = (source_dir / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    source_sha = (source_dir / "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    if source_commit != commit:
        raise RuntimeError("Planner HEAD differs from frozen packaged source commit")
    timestamp = datetime.now(timezone.utc).isoformat()
    for arm in ("gptrans_distance_only", "k1_distance_angle"):
        target = HERE / f"plan_{arm}.json"
        if target.exists():
            raise FileExistsError(target)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(make_plan(arm, commit, source_sha, timestamp), indent=2) + "\n")
        print(target)


if __name__ == "__main__":
    main()
