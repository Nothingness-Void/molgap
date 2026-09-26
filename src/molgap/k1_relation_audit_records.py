"""Saved-prediction analysis and separate NO_TRAIN terminal publication inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

from .constants import REPO_ROOT
from .k1_relation_audit import MODES, REFERENCE, accept_audit
from .k1_relation_study_records import REL, ROOT, load, save
from .research_memory.trace import file_digest, json_bytes

RECORDS = REPO_ROOT / "platforms/_records/kaggle/training/k1_relation_resolution_audit_v1"
AUDIT = RECORDS / "pcqm_k1_relation_audit"
RESULTS = ROOT / "audit/results"


def _adapter():
    path = REPO_ROOT / "experiments/pcqm_k1_linear_attention_100k/accept.py"
    spec = importlib.util.spec_from_file_location("relation_saved_chunk_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_summary(candidate, reference):
    """Rows are paired; these intervals never estimate training-seed variance."""
    import numpy as np
    from .pcqm_k1_shadow_audit import paired_bootstrap
    candidate, reference = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError("Paired error arrays must have identical shapes")
    delta = candidate - reference
    if delta.ndim != 1 or not delta.size or not np.isfinite(delta).all():
        raise ValueError("Finite aligned row errors required")
    low, high = paired_bootstrap(delta, replicates=2000)
    return {"candidate_minus_reference_eV": float(delta.mean()),
        "paired_row_bootstrap_95_eV": [low, high],
        "candidate_win_fraction": float((delta < 0).mean()),
        "winning_margin_mean_eV": float(-delta[delta < 0].mean()) if (delta < 0).any() else None,
        "losing_margin_mean_eV": float(delta[delta > 0].mean()) if (delta > 0).any() else None}


def analyze():
    """Load already accepted prediction tensors; never instantiate a network."""
    import numpy as np
    from .pcqm_k1_shadow_audit import BOOTSTRAP_SEED
    accepted = accept_audit(AUDIT)
    terminal = load(AUDIT / "post100k_audit/terminal.json")
    adapter = _adapter()
    analysis = {"format": "molgap-relation-portability-analysis-v1",
        "run_id": accepted["run_id"], "training_executed": False,
        "model_inference_executed_locally": False,
        "statistical_scope": "post-hoc paired row bootstrap; 2000 replicates; not seed uncertainty, not multiplicity adjusted",
        "bootstrap_seed": BOOTSTRAP_SEED, "roles": {}, "retention": {}}
    manifests = {}
    for role, key, start in (("original_100k", "reproduction", 100000),
                              ("unseen_500k", "unseen_500k", 500000)):
        errors = {}
        for mode in (REFERENCE, *MODES):
            payload = adapter._joined(AUDIT / "post100k_audit" / role / mode,
                terminal[key][mode]["chunks"], start)
            errors[mode] = (payload["prediction_eV"].double() - payload["target_eV"].double()).abs().numpy()
            if mode == REFERENCE:
                manifests[role] = {"cache_manifest_sha256": terminal["manifest_sha256"][role],
                    "rows": len(errors[mode]), "start_inclusive": start, "end_exclusive": start + 50000,
                    "row_ids_sha256_le_i64": hashlib.sha256(payload["source_idx"].numpy().astype("<i8").tobytes()).hexdigest(),
                    "targets_sha256_le_f32": hashlib.sha256(payload["target_eV"].numpy().astype("<f4").tobytes()).hexdigest()}
        comparisons = {mode: paired_summary(errors[mode], errors[REFERENCE]) for mode in MODES}
        parent = MODES[0]
        incremental = {mode: paired_summary(errors[mode], errors[parent]) for mode in MODES[1:]}
        # Ordered row blocks are descriptive checks, not independent random splits.
        for mode in MODES:
            delta = errors[mode] - errors[REFERENCE]
            comparisons[mode]["ordered_5k_block_delta_eV"] = [float(part.mean()) for part in np.split(delta, 10)]
        analysis["roles"][role] = {"mae_eV": {mode: float(error.mean()) for mode, error in errors.items()},
            "versus_k1": comparisons, "versus_receiver_pair": incremental}
    for mode in MODES:
        original = -analysis["roles"]["original_100k"]["versus_k1"][mode]["candidate_minus_reference_eV"]
        unseen = -analysis["roles"]["unseen_500k"]["versus_k1"][mode]["candidate_minus_reference_eV"]
        analysis["retention"][mode] = {"original_gain_eV": original, "frozen500k_gain_eV": unseen,
            "gain_retention_ratio": unseen / original if original > 0 else None,
            "favorable_ordered_5k_blocks": sum(x < 0 for x in analysis["roles"]["unseen_500k"]["versus_k1"][mode]["ordered_5k_block_delta_eV"])}
    save(RESULTS / "portability_analysis.json", analysis)
    save(RESULTS / "role_row_manifests.json", manifests)
    save(RESULTS / "acceptance_summary.json", {**accepted,
        "source_acceptance_sha256": file_digest(RECORDS / "acceptance.json")})
    save(RESULTS / "audit_terminal.json", terminal)
    save(RESULTS / "execution.json", load(AUDIT / "execution.json"))
    return analysis


def measured_cost(execution):
    devices = execution["allocated_devices"]
    wall, device = execution["wall_seconds"], execution["allocated_device_seconds"]
    if (not devices or not all(name == "Tesla T4" for name in devices)
            or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (wall, device))
            or not math.isclose(device, wall * len(devices), rel_tol=1e-9)
            or device > execution["spec"]["max_allocated_device_seconds"]):
        raise ValueError("Observed allocation or budget identity invalid")
    return {"wall_hours": {"status": "measured", "value": wall / 3600},
        "device_hours": {"status": "measured", "value": device / 3600},
        "cpu_hours": {"status": "measurement_missing", "value": None},
        "queue_hours": {"status": "measurement_missing", "value": None}}


def prepare(finalized_at):
    """Bind a controller-approved negative-portability decision, not a new gate."""
    accepted = accept_audit(AUDIT)
    analysis = load(RESULTS / "portability_analysis.json")
    frozen = load(ROOT / "audit/rml_plan/trajectory.json")
    tid, run_id = frozen["trajectory_id"], accepted["run_id"]
    if frozen["actions"][0]["run_ids"] != [run_id] or analysis["run_id"] != run_id:
        raise ValueError("NO_TRAIN physical run differs from the prospective plan")
    if any(row["frozen500k_gain_eV"] >= 0 for row in analysis["retention"].values()):
        raise ValueError("A different scientific outcome needs an explicit controller interpretation")
    prefix = f"{REL}/audit"
    source_ref = f"{prefix}/results/terminal_acceptance.json"
    role_use = {key: "untouched" for key in ("official_validation", "test_dev", "test_challenge")}
    roles = []
    manifests = load(RESULTS / "role_row_manifests.json")
    for key, dataset, role in (
            ("original_100k", "pcqm4mv2-ogb-fixed-100k-v1", "internal_development_100000_150000"),
            ("unseen_500k", "pcqm4mv2-ogb-fixed-500k-scnet-v1", "internal_development_500000_550000")):
        role_use[role] = "used"
        for access in ("prediction_input", "labels_read", "metric_computed"):
            roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{key}-{access}",
                "trajectory_id": tid, "action_id": "A001", "run_id": run_id,
                "dataset_identity": dataset,
                "row_manifest_hash": hashlib.sha256(json_bytes(manifests[key])).hexdigest(),
                "role_name": role, "access_kind": access, "selection_used": False,
                "evidence_ref": source_ref})
    execution = load(RESULTS / "execution.json")
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": frozen["actions"][0]["cost_event_ids"][0],
        "trajectory_id": tid, "action_id": "A001", "run_id": run_id, "attempt_id": "v1",
        "category": "audit", "platform": "kaggle2", "hardware": "Tesla_T4_16GB",
        "measurement": measured_cost(execution), "evidence_ref": source_ref}
    save(RESULTS / "role_history.json", {"events": roles, "protected_roles_read": False,
        "observed_scope": "accepted remote prediction chunks and local saved-prediction metric computation; no training or checkpoint selection"})
    save(RESULTS / "cost_records.json", {"costs": [cost],
        "semantics": "sum over all allocated devices, including idle second T4 and setup; not claimed Kaggle billing"})
    outcome = {"execution_status": "complete", "artifact_status": "accepted",
        "comparison_status": "paired_endpoint_diagnostic", "scientific_status": "no_train_negative_transfer",
        "transfer_status": "frozen_portability_regressed", "budget_decision": "stop_under_contract",
        "full_handoff_status": "not_authorized"}
    decision = {"decision_ref": f"{prefix}/decision.md", "outcome": "NO_TRAIN", "final": True,
        "next_allowed_actions": [], "reopen_conditions": ["new mechanism evidence and explicit prospective authority"]}
    artifact_refs = [f"{prefix}/results/{name}.json" for name in (
        "acceptance_summary", "audit_terminal", "execution", "portability_analysis", "role_row_manifests", "role_history", "cost_records")]
    artifact_refs += [f"{REL}/audit_submission_receipt_v1.json", f"{REL}/audit_release.json"]
    authority = [f"{REL}/protocol.md", f"{REL}/training_contract.json", f"{REL}/audit_role_plan.json",
        f"{prefix}/rml_plan/trajectory.json", decision["decision_ref"]]
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "legacy_contract": "none-prospective-v5", "evidence_id": "pcqm-k1-relation-resolution-post100k-audit-s42",
        "track": "C", "scope": "frozen_checkpoint_NO_TRAIN_portability_not_scale_training",
        "outcome": outcome, "role_use": role_use, "authority": {"pointers": authority},
        "artifacts": [{"name": Path(ref).stem, "locator": ref, "availability": "repository_retained",
                       "sha256": file_digest(REPO_ROOT / ref)} for ref in artifact_refs],
        "migration": {"training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False, "migrated_at": finalized_at,
            "verification_scope": "saved remote artifacts; no local model inference; distinct NO_TRAIN evidence"}}
    save(RESULTS / "terminal_evidence.json", evidence)
    save(REPO_ROOT / source_ref, {"format": "molgap-rml-terminal-acceptance-adapter-v1",
        "evidence_id": evidence["evidence_id"], "run_id": run_id, "outcome": outcome,
        "trajectory_decision": decision, "role_use": role_use, "costs": [cost], "roles": roles,
        "source_acceptance_ref": f"{prefix}/results/acceptance_summary.json",
        "source_acceptance_sha256": file_digest(RESULTS / "acceptance_summary.json")})
    bound = sorted(set(artifact_refs + authority + [source_ref]))
    save(RESULTS / "terminal.json", {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid,
        "run_id": run_id, "action_id": "A001", "finalized_at": finalized_at, "acceptance_ref": source_ref,
        "artifact_hashes": {ref: file_digest(REPO_ROOT / ref) for ref in bound},
        "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles, "role_use": role_use})
    return {"trajectory": f"{prefix}/rml_plan/trajectory.json", "terminal": f"{prefix}/results/terminal.json"}
