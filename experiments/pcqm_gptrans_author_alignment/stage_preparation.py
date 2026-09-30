"""Thin CPU-stage planning/package adapter, without remote submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT
from molgap.experiment_package import _name
from molgap.experiment_staging import UploadArtifact, stage_release_inputs
from molgap.experiment_spec import ExperimentSpec
from molgap.research_memory.plan import plan
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_runtime import normalized_source_sha256

BASE = "experiments/pcqm_gptrans_author_alignment"
REFERENCE = "pcqm-gptrans-v5-audit-reference-s42"
TRAJECTORY = "TC-gptrans-author-input-preparation"
RUN = "kaseichou/molgap-gptrans-author-inputs-preparation-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verification-recovery", action="store_true",
                        help="Reverify the pinned failed CPU output under a separate prospective attempt")
    parser.add_argument("--recovery-attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    root, output = Path(REPO_ROOT), args.output.absolute()
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    recovery = args.verification_recovery
    fallback = recovery and args.recovery_attempt == 2
    trajectory = TRAJECTORY + "-verification-recovery" + ("-v2" if fallback else "") if recovery else TRAJECTORY
    run = "kaseichou/molgap-gptrans-author-inputs-verification-recovery" if recovery else RUN
    attempt = f"recovery-v{args.recovery_attempt}" if recovery else "v1"
    plan_ref = f"{BASE}/verification_recovery/rml_plan_v2" if fallback else (f"{BASE}/verification_recovery/rml_plan" if recovery else f"{BASE}/preparation_rml")
    contract_ref = f"{BASE}/verification_recovery/fixed_cache_contract.json" if fallback else (f"{BASE}/verification_recovery/contract.json" if recovery else f"{BASE}/preparation_contract.json")
    metadata_ref = f"{BASE}/verification_recovery/fixed_cache_kernel-metadata.json" if fallback else (f"{BASE}/verification_recovery/kernel-metadata.json" if recovery else f"{BASE}/kaggle_prepare/kernel-metadata.json")
    contract_path = root / contract_ref
    contract = json.loads(contract_path.read_text())
    budget_ref, role_ref = f"{BASE}/preparation_budget.json", f"{BASE}/preparation_role_plan.json"
    cost_id = "cost-" + trajectory
    proposal = {
        "trajectory": {
            "trajectory_id": trajectory, "record_mode": "prospective", "owner": "server", "track": "C",
            "family_id": "gptrans-author-input-preparation",
            "question": "Are the complete path inputs and degree-only initial state valid before G1/G2 compute release?",
            "hypothesis": {
                "hypothesis_id": "H-" + trajectory,
                "observed_deficiency": "Kaggle rejected the failed-kernel output mount; independently rebuild paths from the unchanged fixed graph cache." if fallback else "Only sampled input diagnostics exist; the complete path sidecar and scaled state were absent.",
                "supporting_evidence_ids": [REFERENCE],
                "alternative_explanations": ["BFS tie or row/serialization errors may invalidate the intended input comparison."],
                "changed_mechanism": "CPU-only deterministic path derivation and two-table initial-state transform",
                "cheapest_falsifier": "Independent source-row/path/state validation without model execution",
                "related_closed_family_ids": ["gptrans-shortest-path-execution-profile"],
                "expected_native_cost_ref": cost_id,
                "decision_changed_if_positive": "permit separately gated dual-arm preparation, not scientific promotion",
                "decision_changed_if_negative": "retain CPU evidence and block GPU release",
                "historical_unknowns": [],
            },
            "state_at_start": {
                "source_commit": commit, "source_config_identity": canonical_fingerprint(contract),
                "contract_refs": [contract_ref, f"{BASE}/dual_arm_protocol.md"],
                "reference_ids": [REFERENCE], "parent_trajectory_ids": ["TC-gptrans-v5-audit-reference-100k"],
                "prior_trajectory_ids": [TRAJECTORY] if recovery else [], "prior_evidence_ids": [REFERENCE],
                "role_snapshot_refs": [role_ref], "budget_snapshot_ref": budget_ref,
            },
            "actions": [{"action_id": "A001", "type": "cpu_input_preparation", "source_commit": commit,
                         "run_ids": [run], "attempt_ids": [attempt],
                         "evidence_refs": [contract_ref], "cost_event_ids": [cost_id]}],
            "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {"decision_ref": f"{BASE}/dual_arm_protocol.md", "outcome": "ACTIVE",
                         "next_allowed_actions": ["one CPU preparation and independent acceptance"],
                         "reopen_conditions": ["terminal CPU evidence"]},
        },
        "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [REFERENCE],
                           "available_actions": ["PREPARE_AUTHOR_INPUTS", "DEFER"], "chosen_action": "PREPARE_AUTHOR_INPUTS",
                           "policy_id": "gptrans-author-input-preparation", "policy_version": "v1",
                           "budget_snapshot_ref": budget_ref, "role_snapshot_refs": [role_ref],
                           "state_timestamp": datetime.now(timezone.utc).isoformat(), "source_commit": commit},
        "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": trajectory,
                   "action_id": "A001", "run_id": run, "attempt_id": attempt, "category": "cache_build",
                   "platform": "kaggle2", "hardware": "CPU-only", "evidence_ref": budget_ref,
                   "measurement": {"device_hours": {"status": "not_applicable", "value": None},
                                   "cpu_hours": {"status": "measurement_missing", "value": None},
                                   "wall_hours": {"status": "estimated", "value": 3},
                                   "queue_hours": {"status": "measurement_missing", "value": None}}}],
    }
    plan_dir = root / plan_ref
    if plan_dir.exists():
        retained = json.loads((plan_dir / "trajectory.json").read_text())
        if (retained["trajectory_id"] != trajectory or retained["decision"]["outcome"] != "ACTIVE"
                or retained["state_at_start"]["source_commit"] != commit
                or retained["state_at_start"]["source_config_identity"] != canonical_fingerprint(contract)):
            raise RuntimeError("Existing CPU plan cannot be reused for this package")
        planned = {"trajectory_id": trajectory, "status": "PLANNED", "path": plan_ref}
    else:
        planned = plan(root, proposal, plan_ref)
    ref_bundle = json.loads((root / "experiments/pcqm_gptrans_v5_audit_reference/results/terminal/reference_bundle.json").read_text())
    reference_contract = json.loads((root / "experiments/pcqm_gptrans_v5_audit_reference/contract.json").read_text())
    def reference(name, value):
        return {"name": name, "version": "1", "sha256": value}
    def hashed(name, value):
        return reference(name, canonical_fingerprint({"value": value}))
    spec = ExperimentSpec({
        "schema_version": "molgap-experiment-spec-v1", "experiment_id": "gptrans-author-input-preparation",
        "logical_run_id": "gptrans-author-input-verification-recovery" if recovery else "gptrans-author-input-preparation-v1",
        "arms": [{"arm_id": "input-preparation", "scientific_role": "ablation", "family": {"name": "gptrans_t", "version": "1"},
                  "base": reference("gptrans_t_frozen_core", reference_contract["architecture_sha256"]),
                  "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": contract["base_initial_tensor_sha256"]},
                  "data": {"dataset": reference("pcqm4mv2", contract["dataset_manifest_sha256"]),
                           "split": reference("fixed100k-internal50k", contract["dataset_manifest_sha256"]),
                           "roles": [{"role": role, "membership_sha256": canonical_fingerprint({"range": bounds}),
                                      "row_order_sha256": canonical_fingerprint({"range": bounds, "order": "ascending"}),
                                      "usage_sha256": sha256_file(root / role_ref)}
                                     for role, bounds in (("train", [0, 100000]), ("development", [100000, 150000]))],
                           "feature_schema": "ogb-atom9-bond3-shortest-path-cap20",
                           "feature_sha256": ref_bundle["comparison_identity"]["feature_identity"],
                           "target": "pcqm4mv2-gap-eV-direct"},
                  "training": {"recipe": reference("pcqm_gptrans_v4", normalized_source_sha256(contract_path)), "overrides": {},
                               "objective": hashed("normalized-gap-l1", reference_contract["loss"]),
                               "sampler": hashed("seed-plus-epoch-global-randperm-v1", {"seed": 42, "epochs": 60}),
                               "transform": reference("fixed-train-100k-mean-sample-std", ref_bundle["comparison_identity"]["target_transform_asset_sha256"])},
                  "addons": [], "addon_semantics": "baseline"}],
        "platform": {"name": "kaggle", "accelerator": "CPU-only", "device_count": 1, "cpu_cores": 4,
                     "memory_gib": 16, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"trajectory_id": trajectory, "hypothesis": proposal["trajectory"]["question"],
                        "cheapest_falsifier": "CPU-only independent full input verification", "stop_rule": "No GPU release from this CPU stage alone",
                        "budget_sha256": sha256_file(root / budget_ref)},
        "evidence": {"policy": reference("molgap-v5", sha256_file(root / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")),
                     "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": "molgap-experiment-terminal-descriptor-v1",
    })
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "src/molgap"], cwd=root).split(b"\0")
    paths = []
    for name in tracked:
        if not name.endswith(b".py") or b"/archive/" in name:
            continue
        try:
            paths.append(_name(name.decode()))
        except ValueError:
            # Package policy remains unchanged; selected dependency checks below fail closed.
            continue
    paths.append(contract_ref)
    paths.extend((f"{BASE}/kaggle_prepare/run.py", metadata_ref))
    initial = root / "platforms/_records/kaggle/packages/gptrans_t_v4_source_814d104/initial_state.pt"
    staged = stage_release_inputs(spec, root, paths, output,
        artifacts={"initial_state.pt": UploadArtifact(initial, contract["initial_file_sha256"])},
        recipe_files={"input-preparation": contract_ref},
        initial_states={"input-preparation": "initial_state.pt"},
        required_modules=["molgap.gptrans_author_inputs", "molgap.pcqm_gptrans_v4"],
        entry_template=root / BASE / "kaggle_prepare/run.py",
        kernel_metadata=root / metadata_ref,
        dataset_metadata={"title": "MolGap GPTrans Author Input Verification Source" if recovery else "MolGap GPTrans Author Inputs Source V1",
                          "id": "kaseichou/molgap-gptrans-author-inputs-verification-source" if recovery else "kaseichou/molgap-gptrans-author-inputs-source-v1",
                          "licenses": [{"name": "other"}], "isPrivate": True})
    atomic_json(output / "plan_binding.json", planned)
    print(json.dumps({"plan": planned, "release_status": staged["release_status"], "errors": staged["errors"]}))
    if staged["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
