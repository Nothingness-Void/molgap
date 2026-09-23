"""Prospectively freeze both RML arms against the actual reference evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.constants import REPO_ROOT
from molgap.k1_conjugated_hyperedge import MODES
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS
from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import write_server_comparison_prelaunch


REL = "experiments/pcqm_k1_conjugated_hyperedge_100k"
ROOT = REPO_ROOT / REL
TEMPLATE = REPO_ROOT / "experiments/pcqm_k1_sparse_triplet_100k"
REFERENCE = REPO_ROOT / (
    "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/"
    "reference_bundle.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return path.relative_to(REPO_ROOT).as_posix()


def main():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", REL],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Commit the frozen source/protocol before release")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    bundle = load(REFERENCE)
    role = load(TEMPLATE / "role_plan.json")
    trace = load(TEMPLATE / "trace_plan.json")
    save(ROOT / "role_plan.json", role)
    save(ROOT / "trace_plan.json", trace)
    config_ids = {
        mode: canonical_fingerprint(ARCHITECTURE_CONFIGS[mode]) for mode in MODES
    }
    save(ROOT / "source_config.json", {
        "format": "molgap-frozen-source-config-v1",
        "candidate_ids": list(MODES),
        "source_commit": commit,
        "architecture_config_identities": config_ids,
        "training_contract_ref": f"{REL}/training_contract.json",
        "training_contract_sha256": digest(ROOT / "training_contract.json"),
        "implementation_refs": [
            "src/molgap/k1_conjugated_hyperedge.py",
            "src/molgap/k1_conjugated_sidecar.py",
            "src/molgap/pcqm_k1_variants.py",
            "src/molgap/pcqm_k1_variants_runner.py",
        ],
        "single_changed_mechanism": "conjugated-component-hyperedge-exchange",
    })
    save(ROOT / "budget_snapshot.json", {
        "format": "molgap-budget-snapshot-v1",
        "available_pool": "Kaggle2-weekly-budget",
        "available_device_hours_user_reported": None,
        "reserved_for_this_action_device_hours": 6.0,
        "kaggle1_reserved_for_desktop": True,
        "successor_compute_pre_authorized": False,
    })
    today = datetime.now(timezone.utc).date().isoformat()
    for mode in MODES:
        label = "oneshot" if mode.endswith("oneshot") else "persistent"
        arm = ROOT / "arms" / label
        arm.mkdir(parents=True, exist_ok=True)
        trajectory_id = f"TC-k1-conjugated-{label}-100k-s42"
        evidence_id = f"pcqm-k1-conjugated-{label}-100k-s42"
        run_id = "kaseichou/molgap-k1-conjugated-dual-s42:v1-planned"
        identity = dict(bundle["comparison_identity"])
        identity["architecture_config_identity"] = config_ids[mode]
        prelaunch = assess_comparison_prelaunch(
            candidate_id=mode,
            candidate_plan={
                "comparison_identity": identity,
                "source_config_status": "frozen",
                "source_commit_or_archive": commit,
            },
            reference_id=bundle["reference_id"],
            reference_bundle=bundle,
            experiment_purpose="architecture_comparison",
            intervention_group_id="architecture",
            declared_intervention_fields=["architecture_config_identity"],
            role_applicability_plan={key: role[key] for key in (
                "training_membership", "prediction_input", "labels_read",
                "metric_computed", "selection_used", "external_submission",
            )},
            trace_plan={key: trace[key] for key in (
                "optimizer_step", "sample_presentations", "epoch_or_pass",
                "learning_rate", "live_train_metric", "live_dev_metric",
                "ema_dev_metric", "checkpoint_identity",
            )},
            runtime_qualification_plan={
                "status": "declared", "runtime_certificate_required": True,
                "qualification_scope":
                    "single-device-fp32-no-tf32-bs128-optimizer-inclusive",
            },
        )
        prelaunch_path = arm / "comparison_readiness_prelaunch.json"
        write_server_comparison_prelaunch(
            prelaunch_path,
            comparison_prelaunch=prelaunch,
            experiment_purpose="architecture_comparison",
            reference_bundle=bundle,
            repo_root=REPO_ROOT,
            reference_bundle_path=REFERENCE,
        )
        cost_id = f"cost-{trajectory_id}-training"
        cost = copy.deepcopy(load(TEMPLATE / "costs/expected_training.json"))
        cost.update({
            "cost_event_id": cost_id,
            "trajectory_id": trajectory_id,
            "run_id": run_id,
            "attempt_id": f"dual-v1-{label}-planned",
            "platform": "kaggle2",
            "evidence_ref": ref(arm / "v5_evidence.json"),
        })
        cost["measurement"]["device_hours"] = {"value": 3.0, "status": "estimated"}
        cost["measurement"]["wall_hours"] = {"value": 3.0, "status": "estimated"}
        save(arm / "costs/expected_training.json", cost)
        trajectory = copy.deepcopy(load(TEMPLATE / "trajectory.json"))
        trajectory.update({
            "trajectory_id": trajectory_id,
            "family_id": "k1-conjugated-hyperedge",
            "question": "Does conjugated-component hyperedge exchange improve frozen K1?",
            "comparison_readiness_ref": ref(prelaunch_path),
        })
        hypothesis = trajectory["hypothesis"]
        hypothesis.update({
            "hypothesis_id": f"H-{trajectory_id}",
            "observed_deficiency": "K1's low-rank global advantage survived 500K whereas generic PairToken reversed; a prior separate GraphState experiment benefited from conjugated component communication.",
            "supporting_evidence_ids": [bundle["reference_id"]],
            "alternative_explanations": [
                "GraphState component benefit may not transplant to K1",
                "new capacity may overfit 100K as PairToken did",
            ],
            "changed_mechanism": ARCHITECTURE_CONFIGS[mode]["change"],
            "cheapest_falsifier": "one seed42 fixed PCQM-100K arm against immutable K1",
            "related_closed_family_ids": [
                "k1-functional-group-token", "k1-sparse-triplet",
                "k1-gpspp-local", "k1-induced-pair-token",
            ],
            "expected_native_cost_ref": cost_id,
            "decision_changed_if_positive": "retain shortlist; require independent matched 500K release",
            "decision_changed_if_negative": "close exact K1 conjugated hyperedge arm",
            "historical_unknowns": ["training stochasticity unavailable; not inferred"],
        })
        trajectory["state_at_start"] = {
            "source_commit": commit,
            "source_config_identity": config_ids[mode],
            "contract_refs": [f"{REL}/training_contract.json"],
            "reference_ids": [bundle["reference_id"]],
            "parent_trajectory_ids": ["TC-k1-v4-100k-reference-s42"],
            "prior_trajectory_ids": [],
            "prior_evidence_ids": [bundle["reference_id"]],
            "role_snapshot_refs": [f"{REL}/role_plan.json"],
            "budget_snapshot_ref": f"{REL}/budget_snapshot.json",
        }
        trajectory["actions"] = [{
            "action_id": "A001",
            "type": "planned_candidate_only_seed42_100k_screen",
            "run_ids": [run_id],
            "attempt_ids": [f"dual-v1-{label}-planned"],
            "source_commit": commit,
            "evidence_refs": [ref(prelaunch_path), f"{REL}/source_config.json"],
            "cost_event_ids": [cost_id],
        }]
        trajectory["result"] = {
            "evidence_ids": [evidence_id],
            "evidence_refs": [ref(arm / "v5_evidence.json")],
        }
        trajectory["decision"] = {
            "decision_ref": f"{REL}/STATUS.md",
            "outcome": "ACTIVE",
            "next_allowed_actions": ["submit one frozen T4x2 dual-arm screen"],
            "reopen_conditions": ["terminal attribution before scale or successor"],
        }
        save(arm / "trajectory.json", trajectory)
        artifacts = [
            {
                "name": name,
                "locator": f"repo://{ref(path)}",
                "sha256": digest(path),
                "availability": "committed_prelaunch_metadata",
            }
            for name, path in (
                ("source-config", ROOT / "source_config.json"),
                ("training-contract", ROOT / "training_contract.json"),
                ("comparison-prelaunch", prelaunch_path),
                ("role-plan", ROOT / "role_plan.json"),
                ("trace-plan", ROOT / "trace_plan.json"),
                ("budget-snapshot", ROOT / "budget_snapshot.json"),
            )
        ]
        evidence = copy.deepcopy(load(TEMPLATE / "v5_evidence.json"))
        evidence.update({
            "evidence_id": evidence_id,
            "scope": "prospective_fixed_100k_architecture_screen",
            "authority": {"pointers": [
                f"{REL}/protocol.md", f"{REL}/training_contract.json",
                f"{REL}/source_config.json", ref(prelaunch_path),
                ref(arm / "trajectory.json"),
            ]},
            "artifacts": artifacts,
        })
        evidence["outcome"]["budget_decision"] = "two_independent_arms_one_job_authorized"
        evidence["migration"]["migrated_at"] = today
        save(arm / "v5_evidence.json", evidence)
        serialized = json.dumps({"trajectory": trajectory, "evidence": evidence, "cost": cost})
        if "sparse-triplet" in serialized or "nvoid912" in serialized:
            raise RuntimeError("Template scientific identity leaked into new arm")
    print(json.dumps({"source_commit": commit, "arms": list(MODES)}))


if __name__ == "__main__":
    main()
