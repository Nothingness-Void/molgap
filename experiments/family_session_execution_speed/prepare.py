"""Freeze both actual Git source trees, then publish a prospective RML plan."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import zipfile

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file

REL = "experiments/family_session_execution_speed"
TID = "TB-family-session-speed-20261014"
RUN = "family-session-speed-20261014"


def main():
    root = Path(__file__).resolve().parents[2]
    here = root / REL
    if (here / "config.json").exists() or (here / "rml").exists():
        raise FileExistsError("Never overwrite a prospective diagnostic")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    reference = subprocess.check_output(["git", "rev-parse", "6b516b10"], cwd=root, text=True).strip()
    sources = {}
    (here / "source_snapshots").mkdir()
    import hashlib
    for name, revision in (("reference", reference), ("candidate", commit)):
        archive = here / "source_snapshots" / (name + ".zip")
        subprocess.run(["git", "archive", "--format=zip", "--output=" + str(archive), revision, "src/molgap"],
            cwd=root, check=True)
        with zipfile.ZipFile(archive) as package:
            files = {p: hashlib.sha256(package.read(p)).hexdigest() for p in package.namelist() if p.endswith(".py")}
        sources[name] = {"commit": revision, "archive": archive.relative_to(root).as_posix(),
            "archive_sha256": sha256_file(archive), "files": files}
    paths = subprocess.check_output(["git", "ls-files", "src/molgap", REL], cwd=root, text=True).splitlines()
    config = {"source_commit": commit, "sources": sources,
        "source_sha256": {p: sha256_file(root / p) for p in paths if p.endswith(".py")},
        "families": ["neural_atom_k1", "gptrans_t"], "seed": 42, "cpu_threads": 2, "repeats": 3,
        "steps": 64, "checkpoint_every": 16, "warmup_steps": 4,
        "synthetic": {"rows": 4096, "minimum_nodes": 8, "maximum_nodes": 24},
        "accelerator": "NVIDIA GeForce RTX 5060", "maximum_process_wall_seconds": 600,
        "maximum_trial_wall_seconds": 120, "trajectory_ref": REL + "/rml/trajectory.json",
        "trajectory_id": TID, "experiment_id": RUN}
    atomic_json(here / "config.json", config)
    prior = ["family-checkpoint-efficiency-20261013", "pcqm-input-pipeline-local-20261013"]
    trajectory = {"trajectory_id": TID, "track": "B", "owner": "desktop", "family_id": "shared-family-execution",
        "question": "Do recovery-safe shared sessions preserve endpoints and improve local execution?",
        "hypothesis": {"hypothesis_id": "H-family-session-speed-20261014", "supporting_evidence_ids": prior,
            "observed_deficiency": "Incremental metrics and removed operations were not measured with recoverable session IO",
            "changed_mechanism": "Reviewed metric prefix, layout transport, distance expansion and output recovery",
            "cheapest_falsifier": "Matched synthetic BS128 updates including actual checkpoints and endpoint hashes",
            "alternative_explanations": ["Journal hash IO offsets saved operations", "Core compute dominates",
                "Short-window device variance hides a small gain"],
            "decision_changed_if_positive": "Record bounded local cost evidence, not model/native platform promotion",
            "decision_changed_if_negative": "Keep necessary recovery guards and avoid advertising speed gains",
            "expected_native_cost_ref": "cost-family-session-speed-estimate-20261014",
            "related_closed_family_ids": ["k1-gptrans-checkpoint-execution", "k1-gps9-input-pipeline"]},
        "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(config),
            "contract_refs": [REL + "/protocol.md", REL + "/config.json"], "reference_ids": [],
            "parent_trajectory_ids": [], "prior_evidence_ids": prior, "role_snapshot_refs": [REL + "/role_plan.json"],
            "budget_snapshot_ref": REL + "/protocol.md"},
        "actions": [{"action_id": "A001", "type": "execution_profile", "run_ids": [RUN],
            "attempt_ids": ["local-001"], "source_commit": commit, "evidence_refs": [REL + "/config.json"],
            "cost_event_ids": ["cost-family-session-speed-estimate-20261014"]}],
        "decision": {"outcome": "ACTIVE", "decision_ref": REL + "/decision.md",
            "next_allowed_actions": ["Authorized bounded synthetic session diagnostic"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": REL + "/protocol.md",
        "comparison_blockers": ["Synthetic execution only; no molecular quality or platform qualification"],
        "reference_bundle_id": None}
    spec = {"trajectory": trajectory, "decision_state": {"available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"],
        "chosen_action": "RUN_DIAGNOSTIC", "policy_id": "family-session-speed", "policy_version": "1",
        "state_timestamp": datetime.now(timezone.utc).isoformat()}, "costs": [{"schema": "molgap-cost-event-v1",
        "cost_event_id": "cost-family-session-speed-estimate-20261014", "trajectory_id": TID, "action_id": "A001",
        "run_id": RUN, "attempt_id": "local-001", "platform": "local-windows", "hardware": "RTX5060",
        "category": "audit", "evidence_ref": REL + "/protocol.md", "measurement": {
            "wall_hours": {"value": 600 / 3600, "status": "estimated"},
            "device_hours": {"value": 600 / 3600, "status": "estimated"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}]}
    atomic_json(here / "plan_receipt.json", plan(root, spec, REL + "/rml"))
    print(json.dumps({"trajectory_id": TID, "source_commit": commit, "status": "PLANNED"}))


if __name__ == "__main__":
    main()
