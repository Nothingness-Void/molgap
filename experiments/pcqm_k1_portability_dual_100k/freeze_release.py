"""Freeze two causal RML plans and a separate conditional NO_TRAIN audit plan."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.constants import REPO_ROOT
from molgap.k1_portability_dual import MODES
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS
from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import write_server_comparison_prelaunch


REL = "experiments/pcqm_k1_portability_dual_100k"
ROOT = REPO_ROOT / REL
TEMPLATE = REPO_ROOT / "experiments/pcqm_k1_sparse_triplet_100k"
REFERENCE = REPO_ROOT / (
    "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
)
RUN = "nothingnessvoid/molgap-k1-topology-portability-dual-s42:v1-planned"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _evidence(template, *, evidence_id, scope, pointers, artifacts, budget):
    evidence = copy.deepcopy(template)
    evidence.update({
        "evidence_id": evidence_id,
        "scope": scope,
        "authority": {"pointers": pointers},
        "artifacts": artifacts,
    })
    evidence["outcome"]["budget_decision"] = budget
    evidence["migration"]["migrated_at"] = datetime.now(timezone.utc).date().isoformat()
    return evidence


def _artifact(name, path):
    return {
        "name": name, "locator": f"repo://{rel(path)}",
        "sha256": digest(path), "availability": "committed_prelaunch_metadata",
    }


def main():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", f"{REL}/protocol.md",
         f"{REL}/training_contract.json", f"{REL}/freeze_release.py",
         f"{REL}/package_source.py", f"{REL}/kaggle_gpu", f"{REL}/test_static.py"],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Commit the frozen source and protocol before release")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
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
            "src/molgap/k1_portability_dual.py",
            "src/molgap/k1_portability_audit.py",
            "src/molgap/pcqm_k1_variants.py",
            "src/molgap/pcqm_k1_variants_runner.py",
        ],
        "separate_changed_mechanisms": [
            ARCHITECTURE_CONFIGS[mode]["change"] for mode in MODES
        ],
    })
    save(ROOT / "budget_snapshot.json", {
        "format": "molgap-budget-snapshot-v1",
        "available_pool": "Kaggle1-shared-with-desktop",
        "available_device_hours_user_reported": None,
        "estimated_training_device_hours": 6.0,
        "estimated_no_train_audit_device_hours": 1.0,
        "desktop_jobs_must_not_be_interrupted": True,
        "successor_compute_pre_authorized": False,
    })
    evidence_template = load(TEMPLATE / "v5_evidence.json")
    cost_template = load(TEMPLATE / "costs/expected_training.json")
    trajectory_template = load(TEMPLATE / "trajectory.json")
    for label, mode in (("rwse_refresh", MODES[0]), ("degree_balance", MODES[1])):
        arm = ROOT / "arms" / label
        arm.mkdir(parents=True, exist_ok=True)
        tid = f"TC-k1-{label.replace('_', '-')}-100k-s42"
        evidence_id = f"pcqm-k1-{label.replace('_', '-')}-100k-s42"
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
                "qualification_scope": "single-device-fp32-no-tf32-bs128-optimizer-inclusive",
            },
        )
        prelaunch_path = arm / "comparison_readiness_prelaunch.json"
        write_server_comparison_prelaunch(
            prelaunch_path, comparison_prelaunch=prelaunch,
            experiment_purpose="architecture_comparison", reference_bundle=bundle,
            repo_root=REPO_ROOT, reference_bundle_path=REFERENCE,
        )
        cost_id = f"cost-{tid}-training"
        cost = copy.deepcopy(cost_template)
        cost.update({
            "cost_event_id": cost_id, "trajectory_id": tid, "run_id": RUN,
            "attempt_id": f"dual-v1-{label}-planned", "platform": "kaggle1",
            "evidence_ref": rel(arm / "v5_evidence.json"),
        })
        cost["measurement"]["device_hours"] = {"value": 3.0, "status": "estimated"}
        cost["measurement"]["wall_hours"] = {"value": 3.0, "status": "estimated"}
        save(arm / "costs/expected_training.json", cost)
        trajectory = copy.deepcopy(trajectory_template)
        trajectory.update({
            "trajectory_id": tid,
            "family_id": "k1-topology-portability",
            "question": f"Does {mode} improve and transfer beyond frozen K1?",
            "comparison_readiness_ref": rel(prelaunch_path),
        })
        trajectory["hypothesis"].update({
            "hypothesis_id": f"H-{tid}",
            "observed_deficiency": "K1 relation additions often improve the selected 100K role without preserving their gain on disjoint 500K development molecules; test a low-capacity topology/degree invariance instead.",
            "supporting_evidence_ids": [bundle["reference_id"]],
            "alternative_explanations": [
                "original development role is reused for model selection",
                "new calibration may not explain within-bin transfer attenuation",
            ],
            "changed_mechanism": ARCHITECTURE_CONFIGS[mode]["change"],
            "cheapest_falsifier": "one isolated fixed-100K seed42 arm plus conditional frozen post-100K audit",
            "related_closed_family_ids": [
                "k1-pair-token", "k1-global-gate", "k1-gpspp-local",
                "k1-slot-readout", "k1-conjugated-hyperedge",
            ],
            "expected_native_cost_ref": cost_id,
            "decision_changed_if_positive": "retain only a prospectively gated shortlist; no automatic scale training",
            "decision_changed_if_negative": f"close exact {mode} intervention",
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
            "action_id": "A001", "type": "planned_candidate_only_seed42_100k_screen",
            "run_ids": [RUN], "attempt_ids": [f"dual-v1-{label}-planned"],
            "source_commit": commit,
            "evidence_refs": [rel(prelaunch_path), f"{REL}/source_config.json"],
            "cost_event_ids": [cost_id],
        }]
        trajectory["result"] = {
            "evidence_ids": [evidence_id],
            "evidence_refs": [rel(arm / "v5_evidence.json")],
        }
        trajectory["decision"] = {
            "decision_ref": f"{REL}/STATUS.md", "outcome": "ACTIVE",
            "next_allowed_actions": ["submit one frozen T4x2 dual-arm screen"],
            "reopen_conditions": ["terminal evidence and attribution before scale"],
        }
        save(arm / "trajectory.json", trajectory)
        artifacts = [_artifact(name, path) for name, path in (
            ("source-config", ROOT / "source_config.json"),
            ("training-contract", ROOT / "training_contract.json"),
            ("comparison-prelaunch", prelaunch_path),
            ("role-plan", ROOT / "role_plan.json"),
            ("trace-plan", ROOT / "trace_plan.json"),
            ("budget-snapshot", ROOT / "budget_snapshot.json"),
        )]
        evidence = _evidence(
            evidence_template, evidence_id=evidence_id,
            scope="prospective_fixed_100k_architecture_screen",
            pointers=[f"{REL}/protocol.md", f"{REL}/training_contract.json",
                      f"{REL}/source_config.json", rel(prelaunch_path), rel(arm / "trajectory.json")],
            artifacts=artifacts, budget="two_independent_arms_one_job_authorized",
        )
        save(arm / "v5_evidence.json", evidence)
    audit = ROOT / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    audit_role = dict(role)
    audit_role["training_membership"] = "not_applicable"
    audit_role["selection_used"] = "not_applicable"
    save(audit / "role_plan.json", audit_role)
    audit_tid = "TC-k1-topology-portability-post100k-audit-s42"
    audit_evidence_id = "pcqm-k1-topology-portability-post100k-audit-s42"
    audit_cost_id = f"cost-{audit_tid}-inference"
    audit_cost = copy.deepcopy(cost_template)
    audit_cost.update({
        "category": "inference", "cost_event_id": audit_cost_id,
        "trajectory_id": audit_tid, "run_id": RUN,
        "attempt_id": "dual-v1-post100k-audit-planned", "platform": "kaggle1",
        "evidence_ref": rel(audit / "v5_evidence.json"),
    })
    audit_cost["measurement"]["device_hours"] = {"value": 1.0, "status": "estimated"}
    audit_cost["measurement"]["wall_hours"] = {"value": 1.0, "status": "estimated"}
    save(audit / "costs/expected_inference.json", audit_cost)
    audit_trajectory = copy.deepcopy(trajectory_template)
    audit_trajectory.update({
        "trajectory_id": audit_tid,
        "family_id": "k1-topology-portability-no-train-audit",
        "question": "Does a frozen 100K advantage survive on disjoint fixed500K internal development molecules?",
        "comparison_class": "NO_COMPARISON",
        "reference_bundle_id": bundle["reference_bundle_id"],
    })
    audit_trajectory.pop("comparison_readiness_ref", None)
    audit_trajectory["hypothesis"].update({
        "hypothesis_id": f"H-{audit_tid}",
        "observed_deficiency": "Original PairToken lost most of its selected 100K advantage on disjoint fixed500K development rows before retraining; a prospectively gated frozen audit is needed for future 100K screens.",
        "supporting_evidence_ids": [bundle["reference_id"]],
        "alternative_explanations": [
            "100K and 500K development distributions differ",
            "original 100K development was repeatedly reused",
        ],
        "changed_mechanism": "NO_TRAIN frozen checkpoint portability measurement only",
        "cheapest_falsifier": "reproduce original 50K predictions; then infer fixed500K development indices500000-549999 once",
        "related_closed_family_ids": ["k1-pair-token-scale-bridge"],
        "expected_native_cost_ref": audit_cost_id,
        "decision_changed_if_positive": "retain portability evidence, not a full-scale training release",
        "decision_changed_if_negative": "close the nonportable 100K advantage without retraining",
        "historical_unknowns": ["500K outcome unknown before frozen role read"],
    })
    audit_trajectory["state_at_start"] = {
        "source_commit": commit,
        "source_config_identity": canonical_fingerprint({"audit": "post100k-v1", "arms": config_ids}),
        "contract_refs": [f"{REL}/training_contract.json"],
        "reference_ids": [bundle["reference_id"]],
        "parent_trajectory_ids": [
            "TC-k1-rwse-refresh-100k-s42",
            "TC-k1-degree-balance-100k-s42",
        ],
        "prior_trajectory_ids": [],
        "prior_evidence_ids": [bundle["reference_id"]],
        "role_snapshot_refs": [rel(audit / "role_plan.json")],
        "budget_snapshot_ref": f"{REL}/budget_snapshot.json",
    }
    audit_trajectory["actions"] = [{
        "action_id": "A001",
        "type": "conditional_no_train_post100k_frozen_audit",
        "run_ids": [RUN],
        "attempt_ids": ["dual-v1-post100k-audit-planned"],
        "source_commit": commit,
        "evidence_refs": [f"{REL}/source_config.json", rel(audit / "role_plan.json")],
        "cost_event_ids": [audit_cost_id],
    }]
    audit_trajectory["result"] = {
        "evidence_ids": [audit_evidence_id],
        "evidence_refs": [rel(audit / "v5_evidence.json")],
    }
    audit_trajectory["decision"] = {
        "decision_ref": f"{REL}/STATUS.md", "outcome": "ACTIVE",
        "next_allowed_actions": ["only after both accepted 100K training terminals: frozen NO_TRAIN inference"],
        "reopen_conditions": ["failed reproduction or role/artefact mismatch stops audit"],
    }
    save(audit / "trajectory.json", audit_trajectory)
    audit_evidence = _evidence(
        evidence_template, evidence_id=audit_evidence_id,
        scope="prospective_fixed_500k_post100k_no_train_audit",
        pointers=[f"{REL}/protocol.md", f"{REL}/training_contract.json",
                  f"{REL}/source_config.json", rel(audit / "role_plan.json"),
                  rel(audit / "trajectory.json")],
        artifacts=[_artifact(name, path) for name, path in (
            ("source-config", ROOT / "source_config.json"),
            ("training-contract", ROOT / "training_contract.json"),
            ("audit-role-plan", audit / "role_plan.json"),
            ("budget-snapshot", ROOT / "budget_snapshot.json"),
        )],
        budget="conditional_no_train_audit_authorized_after_training_acceptance",
    )
    save(audit / "v5_evidence.json", audit_evidence)
    print(json.dumps({"source_commit": commit, "arms": list(MODES)}))


if __name__ == "__main__":
    main()
