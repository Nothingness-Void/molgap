"""Freeze the prospective saved-prediction diagnostic in RML."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = REPO_ROOT
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-consistency-residual-screen-20261004"
RUN = "local-k1-consistency-residual-screen-20261004"
POLICY_ID = "pcqm-k1-consistency-residual-screen"
POLICY_PATH = ROOT / "research_memory/policies/pcqm-k1-consistency-residual-screen.1.json"


def main() -> None:
    inputs_path = HERE / "inputs.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    protocol = f"{REL}/protocol.md"
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy["created_from_source_digest"] != sha256_file(HERE / "protocol.md"):
        raise ValueError("candidate policy source digest differs from the frozen protocol")

    required_arms = {"mean2", "consistency2"}
    if set(inputs.get("arms", {})) != required_arms:
        raise ValueError("inputs.json must bind exactly the accepted mean2 and consistency2 arms")
    if inputs.get("source_idx_range") != [100000, 150000]:
        raise ValueError("inputs.json row range differs from the prospective contract")
    if inputs.get("role") != "internal_development" or inputs.get("selection_used") is not True:
        raise ValueError("inputs.json role state differs from the prospective contract")
    for arm, binding in inputs["arms"].items():
        path = Path(binding["path"])
        if path.is_absolute() or ".." in path.parts or len(binding.get("sha256", "")) != 64:
            raise ValueError(f"invalid retained arm binding: {arm}")
        if not (ROOT / path).is_file():
            raise FileNotFoundError(ROOT / path)

    source_refs = list(inputs.get("source_hashes", {}))
    if not source_refs or protocol not in source_refs:
        raise ValueError("inputs.json must include protocol and analysis source hashes")
    for relative, expected in inputs["source_hashes"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"source binding must be repository-relative: {relative}")
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"frozen analysis source changed: {relative}")

    mean2_id = "pcqm-k1-dropout-mean2-kaggle3-100k-s42-v1-terminal"
    consistency2_id = "pcqm-k1-dropout-consistency2-kaggle3-100k-s42-v1-terminal"
    parent_ids = [
        "TB-k1-dropout-mean2-kaggle3-100k-s42-v1",
        "TB-k1-dropout-consistency2-kaggle3-100k-s42-v1",
    ]
    cost_id = "cost-TB-k1-consistency-residual-screen-20261004-expected-cpu-audit"
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    contract_refs = sorted(set([
        protocol,
        f"{REL}/inputs.json",
        f"{REL}/prepare_rml.py",
        *source_refs,
        "experiments/pcqm_k1_dropout_consistency/terminal_acceptance/dropout_mean2/acceptance.json",
        "experiments/pcqm_k1_dropout_consistency/terminal_acceptance/dropout_consistency2/acceptance.json",
    ]))
    role_refs = [
        protocol,
        "experiments/pcqm_k1_dropout_consistency/terminal_acceptance/dropout_mean2/acceptance.json",
        "experiments/pcqm_k1_dropout_consistency/terminal_acceptance/dropout_consistency2/acceptance.json",
    ]
    spec = {
        "trajectory": {
            "schema": "molgap-trajectory-v1",
            "trajectory_id": TID,
            "record_mode": "prospective",
            "track": "B",
            "owner": "desktop",
            "family_id": "k1-consistency-residual-screen",
            "question": "Does the accepted K1 consistency improvement reflect removable bias or complementary residual errors, and can one frozen postprocessor improve held-out rows?",
            "hypothesis": {
                "hypothesis_id": "H-k1-consistency-residual-screen-20261004",
                "observed_deficiency": "The accepted consistency2 arm improved paired Gap MAE by 1.5287825 meV, below the original 3 meV training gate; retained predictions do not identify whether this came from bias shift or complementary errors.",
                "supporting_evidence_ids": [mean2_id, consistency2_id],
                "alternative_explanations": [
                    "The paired gain is primarily a constant residual bias shift.",
                    "The gain is distributed across rows or complements the mean2 error pattern.",
                    "The apparent subgroup and postprocessor gains reflect one seed and the already-consumed development role.",
                ],
                "changed_mechanism": "Analyze only accepted selected-epoch-37 saved predictions; fit per-arm median offsets and one convex blend on source_idx modulo 5 equals zero, then evaluate on the remaining rows.",
                "cheapest_falsifier": "Reproduce accepted paired endpoints and compute residual, bias, and fixed postprocessor comparisons with 1000 paired-row bootstrap draws on the modulo holdout.",
                "related_closed_family_ids": ["k1-dropout-consistency", "k1-flag", "k1-slot-width96"],
                "expected_native_cost_ref": cost_id,
                "decision_changed_if_positive": "Nominate only the measured postprocessor for separate independent-role qualification when its additional gain is at least 1 meV and the 95 percent row-bootstrap lower bound is above zero; this does not release training or promotion.",
                "decision_changed_if_negative": "Close this cheap postprocessing route and preserve the accepted training verdicts without a seed or schedule retry.",
            },
            "state_at_start": {
                "source_commit": source_commit,
                "contract_refs": contract_refs,
                "reference_ids": [mean2_id, consistency2_id],
                "parent_trajectory_ids": parent_ids,
                "prior_evidence_ids": [mean2_id, consistency2_id],
                "role_snapshot_refs": role_refs,
                "budget_snapshot_ref": protocol,
                "source_config_identity": sha256_file(inputs_path),
            },
            "actions": [{
                "action_id": "A001",
                "type": "local_saved_prediction_residual_and_postprocessor_diagnostic",
                "source_commit": source_commit,
                "run_ids": [RUN],
                "attempt_ids": ["local-attempt-001"],
                "evidence_refs": [],
                "cost_event_ids": [cost_id],
            }],
            "decision": {
                "decision_ref": protocol,
                "outcome": "ACTIVE",
                "next_allowed_actions": ["READ_SAVED_PREDICTIONS"],
                "reopen_conditions": [],
            },
            "result": {"evidence_ids": [], "evidence_refs": []},
        },
        "decision_state": {
            "known_trajectory_ids": [],
            "known_evidence_ids": [],
            "active_reference_ids": [mean2_id, consistency2_id],
            "available_actions": ["READ_SAVED_PREDICTIONS", "NO_TRAIN"],
            "role_snapshot_refs": role_refs,
            "chosen_action": "READ_SAVED_PREDICTIONS",
            "policy_id": POLICY_ID,
            "policy_version": "1",
            "budget_snapshot_ref": protocol,
            "state_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_commit": source_commit,
        },
        "costs": [{
            "schema": "molgap-cost-event-v1",
            "cost_event_id": cost_id,
            "trajectory_id": TID,
            "action_id": "A001",
            "run_id": RUN,
            "attempt_id": "local-attempt-001",
            "category": "audit",
            "platform": "local-windows",
            "hardware": "CPU; saved-prediction analysis only; accelerator not applicable",
            "evidence_ref": protocol,
            "measurement": {
                "device_hours": {"value": None, "status": "not_applicable"},
                "cpu_hours": {"value": 0.05, "status": "estimated"},
                "wall_hours": {"value": 0.05, "status": "estimated"},
                "queue_hours": {"value": None, "status": "not_applicable"},
            },
        }],
    }
    # Freeze the exact planner payload itself, along with every source listed by inputs.json.
    spec["trajectory"]["state_at_start"]["contract_refs"].append(f"{REL}/plan_input.json")
    spec["trajectory"]["state_at_start"]["contract_refs"] = sorted(set(spec["trajectory"]["state_at_start"]["contract_refs"]))
    atomic_json(HERE / "plan_input.json", spec)
    print(json.dumps(plan(ROOT, spec, f"{REL}/rml"), indent=2))


if __name__ == "__main__":
    main()
