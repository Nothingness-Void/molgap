"""Verify saved-prediction metadata and publish the immutable RML terminal package."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = REPO_ROOT
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-consistency-residual-screen-20261004"
RUN = "local-k1-consistency-residual-screen-20261004"
EID = "pcqm-k1-consistency-residual-screen-20261004"
MEAN2_SHA256 = "603fc5eba19e676ff3b5bcbb96b27d26221194bd641eb2bbce554bafa5458076"
CONSISTENCY2_SHA256 = "be88294358a33592064ddfb6ad2b3d77c95350351bb92adb6559bf77de409e8a"
ACCEPTED_COHORT_SHA256 = "3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    destination = HERE / "rml/rml_finalized"
    if destination.exists():
        receipt = verified_receipt(destination)
        print(json.dumps({"status": "ALREADY_FINALIZED", "finalization_id": receipt["finalization_id"]}))
        return

    inputs_path = HERE / "inputs.json"
    analysis_path = HERE / "results/analysis.json"
    trajectory_path = HERE / "rml/trajectory.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    report = json.loads(analysis_path.read_text(encoding="utf-8"))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    plan_hashes = trajectory["decision_state"]["source_hashes"]

    require(report.get("status") == "EXECUTED_SAVED_PREDICTION_DIAGNOSTIC", "analysis is not a completed saved-prediction diagnostic")
    require(report.get("rows") == 50000, "analysis row count differs from the frozen 50K cohort")
    require(report.get("inputs_sha256") == sha256_file(inputs_path), "analysis inputs manifest hash mismatch")
    require(report.get("prospective_sha256") == sha256_file(trajectory_path), "analysis prospective trajectory hash mismatch")
    require(trajectory.get("trajectory_id") == TID, "prospective trajectory identity mismatch")
    require(trajectory.get("state_at_start", {}).get("source_config_identity") == report["inputs_sha256"], "planned source configuration differs from analysis")
    require(report.get("source_idx_sha256") and len(report["source_idx_sha256"]) == 64, "analysis lacks the exact source index identity")
    require(report.get("split") == {"fit_rows": 10000, "holdout_rows": 40000, "rule": "source_idx modulo5 ==0 fits"}, "fit/holdout partition differs from the protocol")

    accepted_hashes = {"mean2": MEAN2_SHA256, "consistency2": CONSISTENCY2_SHA256}
    require(set(inputs.get("arms", {})) == set(accepted_hashes), "input arms differ from the accepted pair")
    expected_artifacts = {}
    for arm, expected in accepted_hashes.items():
        binding = inputs["arms"][arm]
        require(binding.get("sha256") == expected, f"{arm} is not the accepted prediction artifact")
        require(sha256_file(ROOT / binding["path"]) == expected, f"{arm} raw prediction bytes changed")
        expected_artifacts[binding["path"]] = expected
    require(report.get("artifacts_sha256") == expected_artifacts, "analysis prediction artifact hashes differ from inputs.json")

    source_hashes = inputs.get("source_hashes", {})
    for relative, expected in source_hashes.items():
        require(sha256_file(ROOT / relative) == expected, f"analysis source changed: {relative}")
        require(plan_hashes.get(relative) == expected, f"prospective plan did not freeze analysis source: {relative}")
    for relative in (f"{REL}/inputs.json", f"{REL}/prepare_rml.py", f"{REL}/plan_input.json"):
        require(plan_hashes.get(relative) == sha256_file(ROOT / relative), f"prospective plan source changed: {relative}")

    checks = report.get("checks", {})
    for key in ("aligned_rows", "finite_predictions", "exact_targets", "reconstructed_accepted_endpoints", "no_model_execution", "protected_roles_untouched"):
        require(checks.get(key) is True, f"analysis check did not pass: {key}")
    accepted_path = ROOT / inputs["accepted_authority"]
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_primary = accepted["primary"]
    for key in ("reference_mae_eV", "candidate_mae_eV", "paired_gain_eV"):
        require(math.isclose(float(report["accepted_primary"][key]), float(accepted_primary[key]), rel_tol=0, abs_tol=1e-12), f"accepted endpoint differs: {key}")
    require(report["accepted_primary"].get("rows") == 50000, "accepted paired endpoint row count mismatch")
    require(report["heldout_against_consistency2"]["fixed_equal_blend"].get("nomination_passed") is True, "fixed 50:50 blend does not satisfy its frozen nomination rule")
    require(inputs.get("source_idx_range") == [100000, 150000] and inputs.get("role") == "internal_development" and inputs.get("selection_used") is True, "input role state differs from the prospective plan")

    cohort_sources = [
        "experiments/pcqm_k1_dropout_consistency/terminal_acceptance/dropout_mean2/acceptance.json",
        "experiments/pcqm_k1_dropout_consistency/terminal_acceptance/dropout_consistency2/acceptance.json",
    ]
    cohort_roles = []
    for relative in cohort_sources:
        prior = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        matches = [r for r in prior["roles"] if r["access_kind"] == "prediction_input" and r["selection_used"] is True]
        require(len(matches) == 1, f"accepted role authority is ambiguous: {relative}")
        cohort_roles.append(matches[0])
    require(len({(r["dataset_identity"], r["row_manifest_hash"]) for r in cohort_roles}) == 1, "accepted pair role identities do not agree")
    cohort_dataset, cohort_hash = cohort_roles[0]["dataset_identity"], cohort_roles[0]["row_manifest_hash"]
    require(cohort_hash == ACCEPTED_COHORT_SHA256, "accepted development cohort manifest hash changed")

    decision_ref = f"{REL}/terminal_decision.md"
    decision_path = ROOT / decision_ref
    require(decision_path.is_file(), "terminal_decision.md is required before RML finalization")
    decision_text = decision_path.read_text(encoding="utf-8")
    require("NO_TRAIN" in decision_text and "fixed-blend" in decision_text.lower(), "terminal decision does not record the NO_TRAIN fixed-blend nomination")

    decision = {
        "decision_ref": decision_ref,
        "outcome": "NO_TRAIN",
        "next_allowed_actions": [],
        "reopen_conditions": ["Any further analysis or training requires its own prospective authority and role review."],
    }
    outcome = {
        "execution_status": "complete_no_training",
        "artifact_status": "hash_verified_saved_predictions",
        "comparison_status": "posthoc_internal_development_modulo_holdout",
        "scientific_status": "NO_TRAIN",
        "transfer_status": "not_v5_qualified",
        "budget_decision": "no_training_release",
        "full_handoff_status": "not_applicable",
    }
    role_use = {
        "internal_development": "selection_used",
        "official_validation": "untouched",
        "test_dev": "untouched",
        "test_challenge": "untouched",
    }
    acceptance_ref = f"{REL}/acceptance.json"
    cost = {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": "cost-TB-k1-consistency-residual-screen-20261004-measured-audit",
        "trajectory_id": TID,
        "action_id": "A001",
        "run_id": RUN,
        "attempt_id": "local-attempt-001",
        "category": "audit",
        "platform": "local-windows",
        "hardware": report["hardware"],
        "evidence_ref": acceptance_ref,
        "measurement": {
            "device_hours": {"value": None, "status": "not_applicable"},
            "cpu_hours": {"value": float(report["cpu_seconds"]) / 3600, "status": "measured"},
            "wall_hours": {"value": float(report["wall_seconds"]) / 3600, "status": "measured"},
            "queue_hours": {"value": None, "status": "not_applicable"},
        },
    }
    role_events = [
        {
            "schema": "molgap-role-event-v1",
            "role_event_id": f"role-{TID}-{kind}",
            "trajectory_id": TID,
            "action_id": "A001",
            "run_id": RUN,
            "dataset_identity": cohort_dataset,
            "row_manifest_hash": cohort_hash,
            "role_name": "internal_development",
            "access_kind": kind,
            "selection_used": True,
            "evidence_ref": acceptance_ref,
        }
        for kind in ("prediction_input", "labels_read", "metric_computed", "selection_used")
    ]
    fixed = report["heldout_against_consistency2"]["fixed_equal_blend"]
    acceptance = {
        "format": "molgap-k1-consistency-residual-screen-acceptance-v1",
        "evidence_id": EID,
        "run_id": RUN,
        "outcome": outcome,
        "trajectory_decision": decision,
        "role_use": role_use,
        "checks": {
            "inputs_manifest_hash_verified": True,
            "source_code_hashes_verified": True,
            "planned_source_hashes_verified": True,
            "raw_prediction_hashes_verified": True,
            "accepted_pair_endpoints_reproduced": True,
            "source_rows_100000_150000_aligned": True,
            "fixed_blend_nomination_rule_passed": True,
            "no_checkpoint_or_model_execution": True,
            "no_training_or_accelerator_use": True,
            "protected_roles_untouched": True,
        },
        "observed_execution": {
            "analysis_ref": f"{REL}/results/analysis.json",
            "inputs_sha256": report["inputs_sha256"],
            "prospective_sha256": report["prospective_sha256"],
            "source_idx_sha256": report["source_idx_sha256"],
            "source_idx_state_dict_sha256": report["source_idx_sha256"],
            "accepted_cohort_binding": {
                "role_name": "internal_development",
                "dataset_identity": cohort_dataset,
                "row_manifest_hash": cohort_hash,
                "source_acceptances": cohort_sources,
                "selection_used": True,
            },
            "fixed_equal_blend_gain_meV": fixed["gain_meV"],
            "fixed_equal_blend_row_bootstrap_95pct_meV": fixed["row_bootstrap_95pct_meV"],
            "other_postprocessor_results": report["heldout_against_consistency2"],
            "analysis_timer_scope": "analysis_body_excludes_interpreter_import_startup",
            "cost_scope": "Process CPU and wall timers begin in main after imports; interpreter/import startup, planning, summary, and Git overhead are outside these measurements.",
        },
        "limitations": report["limitations"],
        "costs": [cost],
        "roles": role_events,
    }
    atomic_json(ROOT / acceptance_ref, acceptance)

    hash_refs = {
        f"{REL}/results/analysis.json",
        f"{REL}/acceptance.json",
        decision_ref,
        f"{REL}/protocol.md",
        f"{REL}/inputs.json",
        f"{REL}/run_analysis.py",
        f"{REL}/prepare_rml.py",
        f"{REL}/finalize_rml.py",
        inputs["accepted_authority"],
        *cohort_sources,
        *source_hashes.keys(),
        *(binding["path"] for binding in inputs["arms"].values()),
    }
    verification_path = HERE / "verification.json"
    if verification_path.is_file():
        hash_refs.add(f"{REL}/verification.json")
    hashes = {relative: sha256_file(ROOT / relative) for relative in sorted(hash_refs)}
    artifacts = [
        {"name": "results/analysis.json", "locator": f"{REL}/results/analysis.json", "sha256": hashes[f"{REL}/results/analysis.json"], "availability": "locally_retained_hash_verified"},
        {"name": "acceptance.json", "locator": acceptance_ref, "sha256": hashes[acceptance_ref], "availability": "locally_retained_hash_verified"},
        {"name": "terminal_decision.md", "locator": decision_ref, "sha256": hashes[decision_ref], "availability": "locally_retained_hash_verified"},
        {"name": "protocol.md", "locator": f"{REL}/protocol.md", "sha256": hashes[f"{REL}/protocol.md"], "availability": "locally_retained_hash_verified"},
        {"name": "inputs.json", "locator": f"{REL}/inputs.json", "sha256": hashes[f"{REL}/inputs.json"], "availability": "locally_retained_hash_verified"},
        {"name": "run_analysis.py", "locator": f"{REL}/run_analysis.py", "sha256": hashes[f"{REL}/run_analysis.py"], "availability": "locally_retained_hash_verified"},
        {"name": "inputs/mean2.pt", "locator": inputs["arms"]["mean2"]["path"], "sha256": hashes[inputs["arms"]["mean2"]["path"]], "availability": "locally_retained_hash_verified"},
        {"name": "inputs/consistency2.pt", "locator": inputs["arms"]["consistency2"]["path"], "sha256": hashes[inputs["arms"]["consistency2"]["path"]], "availability": "locally_retained_hash_verified"},
    ]
    if verification_path.is_file():
        relative = f"{REL}/verification.json"
        artifacts.append({"name": "verification.json", "locator": relative, "sha256": hashes[relative], "availability": "locally_retained_hash_verified"})

    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "evidence_id": EID,
        "track": "B",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "legacy_contract": "pcqm-k1-consistency-residual-screen-v1",
        "scope": "desktop_k1_consistency_saved_prediction_residual_and_fixed_blend_screen",
        "authority": {"pointers": [f"{REL}/protocol.md", decision_ref, acceptance_ref, f"{REL}/results/analysis.json", f"{REL}/inputs.json", *cohort_sources]},
        "artifacts": artifacts,
        "outcome": outcome,
        "role_use": role_use,
        "migration": {
            "migrated_at": "2026-10-04",
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
            "verification_scope": "Prospective local CPU analysis of hash-bound accepted saved predictions; no checkpoint inference, training, or V5 promotion.",
        },
    }
    terminal = {
        "format": "molgap-rml-terminal-package-v1",
        "trajectory_id": TID,
        "run_id": RUN,
        "action_id": "A001",
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_ref": acceptance_ref,
        "artifact_hashes": hashes,
        "evidence": evidence,
        "decision": decision,
        "costs": [cost],
        "roles": role_events,
    }
    terminal_path = HERE / "terminal.json"
    atomic_json(terminal_path, terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json"), indent=2))


if __name__ == "__main__":
    main()
