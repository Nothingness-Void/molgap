"""NO_TRAIN diagnostic packaging, planning and saved-artifact analysis."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile

from .constants import REPO_ROOT
from .k1_relation_intervention import VARIANTS, ROLES, REFERENCE, MAX_REPRODUCTION_EV, joined_payload, group_masks
from .research_memory.trace import atomic_write, file_digest, json_bytes

BASE = "experiments/pcqm_k1_relation_resolution_100k"
REL = f"{BASE}/diagnostic"
ROOT = REPO_ROOT / REL
PRIOR = REPO_ROOT / "platforms/_records/kaggle/training/k1_relation_resolution_audit_v1/pcqm_k1_relation_audit"
RECORDS = REPO_ROOT / "platforms/_records/kaggle/training/k1_relation_diagnostic_v1"
REMOTE = RECORDS / "pcqm_k1_relation_diagnostic"
KERNEL = "kaseichou/molgap-k1-relation-diagnostic-s42"
RUN = KERNEL + ":v1"
TID = "TC-k1-relation-frozen-intervention-s42"
EID = "pcqm-k1-relation-frozen-intervention-s42"
POLICY = "k1-relation-frozen-intervention"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    atomic_write(path, json_bytes(value))


def package_and_plan(destination):
    """Creates a new private input package; never submits or loads a model."""
    from .k1_relation_audit import accept_audit
    from .research_memory.plan import plan
    source = "src/molgap/k1_relation_intervention.py"
    paths_to_freeze = [source, "src/molgap/k1_relation_intervention_records.py", REL, f"{BASE}/kaggle_diagnostic"]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *paths_to_freeze], cwd=REPO_ROOT, text=True).strip():
        raise ValueError("Commit diagnostic source/protocol before packaging")
    destination = Path(destination)
    if destination.exists() or (ROOT / "release.json").exists() or (ROOT / "rml_plan").exists():
        raise FileExistsError("Reconcile existing immutable diagnostic release")
    accepted = accept_audit(PRIOR)
    if not accepted["accepted"]:
        raise ValueError("Prior NO_TRAIN audit not accepted")
    prior = load(PRIOR / "post100k_audit/terminal.json")
    old_release = load(REPO_ROOT / BASE / "audit_release.json")
    paths = {"prior_terminal.json": PRIOR / "post100k_audit/terminal.json",
        "target_transform.json": REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/target_transform.json"}
    for mode in VARIANTS:
        slot = "rrwp" if mode.endswith("rrwp_pair") else "dual"
        path = REPO_ROOT / f"platforms/_records/kaggle/training/k1_relation_resolution_{slot}_v1/pcqm_k1_relation_resolution/{mode}/best_model.pt"
        if file_digest(path) != prior["checkpoint_sha256"][mode]:
            raise ValueError("Accepted candidate checkpoint changed")
        paths[f"candidates/{mode}/best_model.pt"] = path
    for role in ROLES:
        key = "reproduction" if role == "original_100k" else "unseen_500k"
        for mode in (REFERENCE, *VARIANTS):
            for chunk in prior[key][mode]["chunks"]:
                path = PRIOR / f"post100k_audit/{role}/{mode}" / chunk["file"]
                if file_digest(path) != chunk["sha256"]:
                    raise ValueError("Prior accepted prediction changed")
                paths[f"prior/{role}/{mode}/{chunk['file']}"] = path
    destination.mkdir(parents=True)
    with tarfile.open(destination / "diagnostic_inputs.bin", "w:gz") as archive:
        for name, path in sorted(paths.items()):
            archive.add(path, arcname=name, recursive=False)
    shutil.copyfile(REPO_ROOT / source, destination / "diagnostic_impl.py")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    release = {"format": "molgap-relation-intervention-release-v1", "run_id": RUN,
        "helper_commit": commit, "helper_sha256": file_digest(destination / "diagnostic_impl.py"),
        "entry_sha256": file_digest(REPO_ROOT / BASE / "kaggle_diagnostic/run.py"),
        "model_source_commit": old_release["source_commit"],
        "source_archive_sha256": old_release["source_archive_sha256"],
        "input_archive_sha256": file_digest(destination / "diagnostic_inputs.bin"),
        "input_files": {name: file_digest(path) for name,path in sorted(paths.items())},
        "manifest_sha256": prior["manifest_sha256"],
        "prior_terminal_sha256": file_digest(paths["prior_terminal.json"]),
        "prior_acceptance_sha256": file_digest(PRIOR.parent / "acceptance.json"),
        "variants": {key: list(value) for key,value in VARIANTS.items()},
        "training_authorized": False, "reference_inference_reused": True,
        "max_allocated_device_seconds": 5400, "max_wall_seconds": 5400}
    save(destination / "DIAGNOSTIC_RELEASE.json", release)
    save(ROOT / "release.json", release)
    save(destination / "dataset-metadata.json", {"id": "kaseichou/molgap-k1-relation-diagnostic-inputs",
        "title": "MolGap K1 Relation Diagnostic Inputs", "isPrivate": True, "licenses": [{"name": "other"}]})
    save(ROOT / "budget_snapshot.json", {"authorization": "user-requested frozen-checkpoint diagnosis; no new training",
        "available_account_device_hours": None, "max_allocated_device_seconds": 5400,
        "max_wall_seconds": 5400, "expected_device_hours": 1.0, "expected_wall_hours": 0.5,
        "paid_overage_authorized": False, "automatic_training_successor_authorized": False})
    save(ROOT / "role_plan.json", load(REPO_ROOT / BASE / "audit_role_plan.json"))
    policy = load(REPO_ROOT / f"research_memory/policies/k1-relation-resolution-bounded-research.v1.json")
    policy.update(policy_id=POLICY, created_from_source_digest=file_digest(ROOT / "protocol.md"),
        required_observable_fields=["accepted_frozen_models_with_unresolved_interference"],
        approval={"authority_ref": f"{REL}/protocol.md", "activation": "one-user-authorized-NO_TRAIN-diagnostic"},
        action_rule={"field": "accepted_frozen_models_with_unresolved_interference", "operator": "eq",
                     "threshold": True, "action": "RUN_FROZEN_INTERVENTIONS"})
    save(REPO_ROOT / f"research_memory/policies/{POLICY}.v1.json", policy)
    template = load(REPO_ROOT / BASE / "audit/rml_plan/trajectory.json")
    trajectory = {key: copy.deepcopy(value) for key,value in template.items()
                  if key not in ("decision_state",)}
    trajectory.update(trajectory_id=TID, family_id="k1-frozen-relation-dependency",
        question="Which relation computations help, harm, or merely support co-adapted weights on both fixed internal roles?")
    evidence = ["pcqm-k1-relation-resolution-post100k-audit-s42", "pcqm-k1-v4-100k-reference-s42"]
    trajectory["hypothesis"].update(hypothesis_id=f"H-{TID}", expected_native_cost_ref=f"cost-{TID}",
        observed_deficiency="Original-role benefits reverse on fixed500K internal dev, especially weakly conjugated rows.",
        supporting_evidence_ids=evidence, changed_mechanism="frozen checkpoint interventions; no optimizer or retraining",
        alternative_explanations=["co-adaptation", "receiver redistribution interference", "topology/triplet redundancy", "coverage and selection optimism"],
        cheapest_falsifier="predeclared full/half/off/targeted frozen inference on both roles",
        decision_changed_if_positive="identify a narrowly supported computation; no automatic training or promotion",
        decision_changed_if_negative="close unsupported attribution; do not blindly extend epochs")
    trajectory["state_at_start"] = {"source_commit": commit, "source_config_identity": file_digest(ROOT / "release.json"),
        "contract_refs": [f"{REL}/protocol.md", f"{REL}/release.json"],
        "reference_ids": ["pcqm-k1-v4-100k-reference-s42"],
        "parent_trajectory_ids": ["TC-k1-relation-resolution-post100k-audit-s42"],
        "prior_trajectory_ids": template["state_at_start"]["parent_trajectory_ids"],
        "prior_evidence_ids": evidence, "role_snapshot_refs": [f"{REL}/role_plan.json"],
        "budget_snapshot_ref": f"{REL}/budget_snapshot.json"}
    trajectory["actions"] = [{"action_id": "A001", "type": "NO_TRAIN_frozen_intervention",
        "run_ids": [RUN], "attempt_ids": ["v1"], "source_commit": commit,
        "evidence_refs": [f"{REL}/release.json"], "cost_event_ids": [f"cost-{TID}"]}]
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
    trajectory["decision"] = {"decision_ref": f"{REL}/protocol.md", "outcome": "ACTIVE",
        "next_allowed_actions": ["one frozen diagnostic; accept and interpret; no training"],
        "reopen_conditions": ["explicit new scientific authority"]}
    cost = load(REPO_ROOT / BASE / "audit/rml_plan/costs/cost-TC-k1-relation-resolution-post100k-audit-s42.json")
    cost.update(cost_event_id=f"cost-{TID}", trajectory_id=TID, run_id=RUN,
                evidence_ref=f"{REL}/budget_snapshot.json")
    cost["measurement"]["wall_hours"] = {"status": "estimated", "value": 0.5}
    result = plan(REPO_ROOT, {"trajectory": trajectory, "costs": [cost], "decision_state": {
        "available_actions": ["RUN_FROZEN_INTERVENTIONS", "DEFER"], "chosen_action": "RUN_FROZEN_INTERVENTIONS",
        "policy_id": POLICY, "policy_version": "v1", "state_timestamp": datetime.now(timezone.utc).isoformat()}}, f"{REL}/rml_plan")
    return {"release": str(ROOT / "release.json"), "package": str(destination), "plan": result}


def accept_and_analyze(remote=REMOTE):
    """Independent saved-tensor acceptance, never model construction/inference."""
    import numpy as np
    import torch
    from .k1_relation_audit_records import paired_summary
    remote = Path(remote)
    release, execution = load(ROOT / "release.json"), load(remote / "execution.json")
    if (execution["spec"] != release or execution["complete"] is not True
        or execution["training_executed"] is not False or execution["optimizer_steps"] != 0
        or execution["run_id"] != RUN or execution["cublas_workspace_config"] != ":4096:8"
        or set(execution["worker_terminal_sha256"]) != set(VARIANTS)):
        raise ValueError("Execution/release/inventory mismatch")
    wall, device = execution["wall_seconds"], execution["allocated_device_seconds"]
    if (not execution["allocated_devices"] or not all(math.isfinite(x) and x > 0 for x in (wall,device))
        or not math.isclose(device, wall*len(execution["allocated_devices"]), rel_tol=1e-9)
        or device > release["max_allocated_device_seconds"] or wall > release["max_wall_seconds"]):
        raise ValueError("Native cost or budget invalid")
    prior = load(PRIOR / "post100k_audit/terminal.json")
    if file_digest(PRIOR / "post100k_audit/terminal.json") != release["prior_terminal_sha256"]:
        raise ValueError("Prior authority changed")
    summary = {"format": "molgap-relation-intervention-analysis-v1", "run_id": RUN,
        "model_inference_executed_locally": False, "training_executed": False,
        "statistical_scope": "paired row bootstrap 2000; descriptive, not multiplicity adjusted or training-seed uncertainty",
        "decision_scope": "checkpoint dependence/interference only; no retrained ablation, no new candidate promotion",
        "models": {}}
    row_manifests, observed_roles = {}, []
    for mode, variants in VARIANTS.items():
        terminal_path = remote / f"workers/{mode}/terminal.json"
        if file_digest(terminal_path) != execution["worker_terminal_sha256"][mode]:
            raise ValueError("Worker terminal hash changed")
        terminal = load(terminal_path)
        if (terminal["complete"] is not True or terminal["state_unchanged"] is not True
            or terminal["training_executed"] is not False or terminal["optimizer_steps"] != 0
            or terminal["model_inference_executed"] is not True
            or terminal["physical_batch"] != 128 or terminal["precision"] != "FP32" or terminal["tf32"] is not False
            or terminal["run_id"] != RUN or terminal["mode"] != mode
            or terminal["checkpoint_sha256"] != release["input_files"][f"candidates/{mode}/best_model.pt"]
            or terminal["runtime"]["torch"] != "2.4.1+cu121"
            or terminal["runtime"]["gpu"] not in execution["allocated_devices"]
            or any(terminal[x] is not False for x in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))
            or set(terminal["records"]) != set(ROLES)):
            raise ValueError("Worker scientific/runtime identity changed")
        expected_events = [{"role": role, "access_kind": access} for role in ROLES
                           for access in ("prediction_input", "labels_read", "metric_computed")]
        if terminal["role_events"] != expected_events:
            raise ValueError("Observed role events missing/undeclared")
        observed_roles.extend(dict(event, mode=mode) for event in terminal["role_events"])
        model_result = {}
        for role, start in ROLES.items():
            if set(terminal["records"][role]) != set(variants):
                raise ValueError("Intervention inventory changed")
            key = "reproduction" if role == "original_100k" else "unseen_500k"
            original = joined_payload(PRIOR / f"post100k_audit/{role}/{mode}", prior[key][mode]["chunks"], start)
            reference = joined_payload(PRIOR / f"post100k_audit/{role}/{REFERENCE}", prior[key][REFERENCE]["chunks"], start)
            payloads = {variant: joined_payload(remote / f"workers/{mode}/{role}/{variant}",
                terminal["records"][role][variant]["chunks"], start) for variant in variants}
            full = payloads["full"]
            diff = float((full["prediction_eV"]-original["prediction_eV"]).abs().max())
            if (diff > MAX_REPRODUCTION_EV or abs(diff-terminal["records"][role]["full"]["reproduction_max_abs_eV"]) > 1e-8
                or not torch.equal(full["target_eV"], original["target_eV"])
                or not torch.equal(full["target_eV"], reference["target_eV"])):
                raise ValueError("Full model failed independent reproduction/alignment")
            target = full["target_eV"].double().numpy()
            ref_error = np.abs(reference["prediction_eV"].double().numpy()-target)
            errors = {}
            for variant, payload in payloads.items():
                if not torch.equal(payload["target_eV"], full["target_eV"]):
                    raise ValueError("Intervention target misalignment")
                errors[variant] = np.abs(payload["prediction_eV"].double().numpy()-target)
                if abs(float(errors[variant].mean())-terminal["records"][role][variant]["mae_eV"]) > 1e-8:
                    raise ValueError("Stored MAE differs from recomputation")
            groups = group_masks(full["descriptors"]["conjugated_bond_fraction"].numpy())
            result = {"full_reproduction_max_abs_eV": diff, "groups": {}}
            for group, mask in groups.items():
                result["groups"][group] = {"rows": int(mask.sum()), "k1_mae_eV": float(ref_error[mask].mean()),
                    "conditions": {variant: {"mae_eV": float(error[mask].mean()),
                        "minus_full": paired_summary(error[mask], errors["full"][mask]),
                        "minus_k1": paired_summary(error[mask], ref_error[mask])} for variant,error in errors.items()},
                    "mean_diagnostics": {k: float(v.double().numpy()[mask].mean()) for k,v in full["diagnostics"].items()},
                    "mean_descriptors": {k: float(v.double().numpy()[mask].mean()) for k,v in full["descriptors"].items()}}
            model_result[role] = result
            row_manifests[role] = {"rows": len(target), "start_inclusive": start, "end_exclusive": start+len(target),
                "cache_manifest_sha256": release["manifest_sha256"][role],
                "row_ids_sha256_le_i64": hashlib.sha256(full["source_idx"].numpy().astype("<i8").tobytes()).hexdigest(),
                "targets_sha256_le_f32": hashlib.sha256(full["target_eV"].numpy().astype("<f4").tobytes()).hexdigest()}
        summary["models"][mode] = model_result
    results = ROOT / "results"
    save(results / "analysis.json", summary)
    save(results / "role_row_manifests.json", row_manifests)
    save(results / "execution.json", execution)
    for mode in VARIANTS:
        save(results / f"{mode}_terminal.json", load(remote / f"workers/{mode}/terminal.json"))
    accepted = {"accepted": True, "run_id": RUN, "model_inference_executed_by_acceptance": False,
        "training_executed": False, "protected_roles_read": False, "observed_role_events": observed_roles,
        "execution_sha256": file_digest(remote / "execution.json"),
        "allocated_device_seconds": device, "wall_seconds": wall,
        "no_new_model_or_coefficient_selected": True, "training_prefix_replay_claim": False}
    save(results / "acceptance.json", accepted)
    return accepted
