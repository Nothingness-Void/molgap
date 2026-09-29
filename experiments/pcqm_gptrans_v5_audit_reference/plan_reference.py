"""Freeze the one user-authorized GPTrans 100K audit-reference RML plan."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT
from molgap.research_memory.plan import plan


ROOT = Path(REPO_ROOT)
BASE = "experiments/pcqm_gptrans_v5_audit_reference"
TRAJECTORY = "TC-gptrans-v5-audit-reference-100k"
COST = "cost-TC-gptrans-v5-audit-reference-100k"
OLD_EVIDENCE = "pcqm-gptrans-t-v4-100k-reference-s42"


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    contract = json.loads((ROOT / BASE / "contract.json").read_text(encoding="utf-8"))
    if contract["manifest_sha256"] != "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d":
        raise RuntimeError("Fixed dataset identity changed")
    from molgap.screen_policy import canonical_fingerprint

    now = datetime.now(timezone.utc).isoformat()
    spec = {
        "trajectory": {
            "trajectory_id": TRAJECTORY,
            "record_mode": "prospective", "owner": "server", "track": "C",
            "family_id": "gptrans-v5-audit-reference",
            "question": "Can one matched GPTrans 100K rerun produce the missing V5 live/EMA causal trace?",
            "hypothesis": {
                "hypothesis_id": "H-" + TRAJECTORY,
                "observed_deficiency": "The retained V4 reference lacks separate live-development MAE observations across 60 epochs.",
                "supporting_evidence_ids": [OLD_EVIDENCE],
                "alternative_explanations": ["Adding live evaluation may alter runtime or reveal nondeterministic divergence; a T4 runtime may fail calibration."],
                "changed_mechanism": "add-only live development observation with frozen V4 training and EMA selection",
                "cheapest_falsifier": "one T4 preflight and first SHA-verifiable 10-epoch segment",
                "related_closed_family_ids": ["gptrans-t-v4-reference"],
                "expected_native_cost_ref": COST,
                "decision_changed_if_positive": "allow independent 60-epoch acceptance and V5 reference-bundle construction",
                "decision_changed_if_negative": "retain historical endpoint-only reference; stop candidate causal release",
                "historical_unknowns": ["old V4 live-development trajectory cannot be reconstructed"],
            },
            "state_at_start": {
                "source_commit": commit,
                "source_config_identity": canonical_fingerprint(contract),
                "contract_refs": [f"{BASE}/contract.json"],
                "reference_ids": [OLD_EVIDENCE],
                "parent_trajectory_ids": ["TC-gptrans-t-v4-100k-reference-s42"],
                "prior_trajectory_ids": [],
                "prior_evidence_ids": [OLD_EVIDENCE],
                "role_snapshot_refs": [f"{BASE}/role_plan.json"],
                "budget_snapshot_ref": f"{BASE}/budget_snapshot.json",
            },
            "actions": [{
                "action_id": "A001", "type": "matched_reference_rerun",
                "source_commit": commit,
                "run_ids": ["kaseichou/molgap-gptrans-v5-audit-reference-s42-v1"],
                "attempt_ids": ["v1"],
                "evidence_refs": [f"{BASE}/contract.json"],
                "cost_event_ids": [COST],
            }],
            "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {
                "decision_ref": f"{BASE}/protocol.md", "outcome": "ACTIVE",
                "next_allowed_actions": ["one preflight and six accepted 10-epoch segments"],
                "reopen_conditions": ["terminal evidence or a changed scientific/runtime contract"],
            },
        },
        "decision_state": {
            "known_trajectory_ids": [], "known_evidence_ids": [],
            "active_reference_ids": [OLD_EVIDENCE],
            "available_actions": ["RUN_MATCHED_GPTRANS_REFERENCE", "DEFER"],
            "chosen_action": "RUN_MATCHED_GPTRANS_REFERENCE",
            "policy_id": "gptrans-v5-audit-reference", "policy_version": "v1",
            "budget_snapshot_ref": f"{BASE}/budget_snapshot.json",
            "role_snapshot_refs": [f"{BASE}/role_plan.json"],
            "state_timestamp": now, "source_commit": commit,
        },
        "costs": [{
            "schema": "molgap-cost-event-v1", "cost_event_id": COST,
            "trajectory_id": TRAJECTORY, "action_id": "A001",
            "run_id": "kaseichou/molgap-gptrans-v5-audit-reference-s42-v1",
            "attempt_id": "v1", "category": "training",
            "platform": "kaggle2", "hardware": "NvidiaTeslaT4-allocated-shape",
            "evidence_ref": f"{BASE}/budget_snapshot.json",
            "measurement": {
                "device_hours": {"status": "estimated", "value": 20.0},
                "wall_hours": {"status": "estimated", "value": 10.0},
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "queue_hours": {"status": "measurement_missing", "value": None},
            },
        }],
    }
    result = plan(ROOT, spec, f"{BASE}/rml_plan")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
