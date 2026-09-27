"""Translate accepted joint-objective artifacts through existing V5/RML wheels."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from .constants import REPO_ROOT
from .k1_joint_acceptance import _chunks, _paired, MODE, AUDIT_RELEASE
from .k1_joint_study_records import REL, RELEASE_REL, load, save
from .k1_joint_study_runtime import ATTEMPT, RUN_ID, RECIPES, TRAJECTORIES
from .k1_relation_audit_records import prepare_no_train_terminal
from .k1_relation_terminal import allocated_arm_cost
from .k1_terminal_analysis import accepted_native_trace
from .research_memory.trace import file_digest

RECORDS = REPO_ROOT / f"platforms/_records/kaggle/training/k1_joint_atom_s42_v{ATTEMPT}"
CANDIDATE = RECORDS / "pcqm_k1_joint_atom_reconstruction"
RESULTS = REPO_ROOT / RELEASE_REL / "results"
REFERENCE_ROOT = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference" / MODE
AUDIT_REFERENCE = REPO_ROOT / "platforms/_records/kaggle/training/k1_relation_resolution_audit_v1/pcqm_k1_relation_audit/inputs"


def split_stage_cost(execution, costs):
    """Count each allocated second once; audit stage wall is not a parallel sum."""
    audit_seconds = 0.0
    training = {}
    for recipe in RECIPES:
        native = costs[recipe]
        allocation = allocated_arm_cost(execution, recipe)
        audit = native["audit_wall_seconds"]
        if not (native["complete"] and native["device_count"] == 1 and
                0 < audit < native["wall_seconds"] <= allocation["wall_seconds"]):
            raise ValueError("Incomplete native stage allocation")
        training[recipe] = {**allocation, "device_seconds": allocation["device_seconds"] - audit,
            "wall_seconds": allocation["wall_seconds"] - audit,
            "audit_device_seconds_excluded": audit}
        audit_seconds += audit
    total = sum(row["device_seconds"] for row in training.values()) + audit_seconds
    if abs(total - execution["total_allocated_device_seconds"]) > 1e-6:
        raise ValueError("Training/audit costs do not conserve notebook allocation")
    return {"training": training, "audit_device_seconds": audit_seconds,
        "total_allocated_device_seconds": total,
        "semantics": "Observed per-device audit intervals excluded from training; notebook bootstrap/idle assigned once to first training arm. No account-billing claim."}


def analyze():
    import numpy as np
    import torch
    accepted = load(RECORDS / "acceptance.json")
    receipt = load(REPO_ROOT / REL / f"submission_receipt_v{ATTEMPT}.json")
    execution = load(CANDIDATE / "execution_summary.json")
    if (accepted["accepted"] is not True or accepted["model_inference_executed"] is not False
            or accepted["audit"]["accepted"] is not True or accepted["run_id"] != RUN_ID
            or execution["run_id"] != RUN_ID or execution["complete"] is not True
            or execution["automatic_successor_submitted"] is not False
            or accepted["source_commit"] != receipt["source_commit"]
            or accepted["source_archive_sha256"] != receipt["source_archive_sha256"]):
        raise ValueError("Accepted artifacts and actual receipt required")
    costs = {r: load(CANDIDATE / r / "native_cost.json") for r in RECIPES}
    allocation = split_stage_cost(execution, costs)
    release = load(AUDIT_RELEASE)
    report = {"format": "molgap-joint-objective-analysis-v1", "run_id": RUN_ID,
        "training_executed_locally": False, "model_inference_executed_locally": False,
        "frozen_500k_model_training_executed": False,
        "role_caveat": "Both are reused internal development roles; neither is a sealed test or a 500K-trained model.",
        "uncertainty_caveat": "Paired row bootstrap does not measure training-seed variance; post-hoc row blocks are not independent splits.",
        "roles": {}, "matched_trajectory": {}, "native_cost_allocation": allocation}
    manifests = {}
    for role, start in (("original_100k", 100000), ("unseen_500k", 500000)):
        prefix = f"reference/{role}/{MODE}/"
        rows = [{"file": k.removeprefix(prefix), "rows": 5000, "sha256": v}
                for k, v in sorted(release["input_files"].items()) if k.startswith(prefix)]
        ref = _chunks(AUDIT_REFERENCE / "reference" / role / MODE, rows, start)
        errors = {MODE: (ref["prediction_eV"].double() - ref["target_eV"].double()).abs().numpy()}
        for recipe in RECIPES:
            terminal = load(CANDIDATE / recipe / "post100k_audit/terminal.json")
            item = _chunks(CANDIDATE / recipe / "post100k_audit" / role, terminal[role]["chunks"], start)
            if not torch.equal(item["target_eV"], ref["target_eV"]):
                raise ValueError("Audit comparison target mismatch")
            errors[recipe] = (item["prediction_eV"].double() - item["target_eV"].double()).abs().numpy()
        report["roles"][role] = {"mae_eV": {k: float(v.mean()) for k, v in errors.items()},
            "B_vs_A": _paired(errors[RECIPES[1]], errors[RECIPES[0]]),
            "versus_K1": {r: {**_paired(errors[r], errors[MODE]),
                "ordered_5k_block_delta_eV": [float(a.mean()) for a in np.split(errors[r] - errors[MODE], 10)]}
                for r in RECIPES}}
        manifests[role] = {"rows": 50000, "start_inclusive": start, "end_exclusive": start + 50000,
            "row_ids_sha256_le_i64": hashlib.sha256(ref["source_idx"].numpy().astype("<i8").tobytes()).hexdigest(),
            "targets_sha256_le_f32": hashlib.sha256(ref["target_eV"].numpy().astype("<f4").tobytes()).hexdigest()}
    baseline = load(REFERENCE_ROOT / "trace.json")["epochs"]
    for recipe in RECIPES:
        rows = load(CANDIDATE / recipe / "trace.json")["epochs"]
        report["matched_trajectory"][recipe] = []
        for observed, base in zip(rows, baseline, strict=True):
            if observed["optimizer_steps"] != base["optimizer_steps"]:
                raise ValueError("Trajectory exposure mismatch")
            report["matched_trajectory"][recipe].append({"epoch": observed["epoch"],
                "optimizer_steps": observed["optimizer_steps"], "candidate_dev_eV": observed["development_gap_mae_eV"],
                "reference_dev_eV": base["development_gap_mae_eV"],
                "delta_eV": observed["development_gap_mae_eV"] - base["development_gap_mae_eV"],
                "candidate_corrupted_train_gap_normalized": observed["gap_loss"],
                "auxiliary_CE": observed["auxiliary_loss"], "seconds": observed["seconds"]})
    save(RESULTS / "acceptance.json", accepted)
    save(RESULTS / "analysis.json", report)
    save(RESULTS / "role_row_manifests.json", manifests)
    save(RESULTS / "execution_summary.json", execution)
    return report


def prepare_training(recipe, finalized_at):
    accepted = load(RESULTS / "acceptance.json")
    candidate = accepted["candidates"][recipe]
    if candidate["gate"]["passed"]:
        raise ValueError("A passing gate requires a separate explicit controller decision")
    positive = candidate["gate"]["gain_eV"] > 0 and candidate["paired_vs_k1"]["row_interval_favorable"]
    prefix = f"{RELEASE_REL}/arms/{recipe}"
    arm, results = REPO_ROOT / prefix, REPO_ROOT / prefix / "results"
    frozen = load(arm / "rml_plan/trajectory.json")
    if frozen["actions"][0]["run_ids"] != [RUN_ID]:
        raise ValueError("Wrong prospective physical attempt")
    allocation = load(RESULTS / "analysis.json")["native_cost_allocation"]["training"][recipe]
    spec = importlib.util.spec_from_file_location("joint_existing_terminal", REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py")
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    adapter.ROOT, adapter.ROOT_REL = arm, prefix
    adapter.SHARED_ROOT, adapter.SHARED_REL = REPO_ROOT / RELEASE_REL, RELEASE_REL
    adapter.RESULTS, adapter.RECORD_ROOT, adapter.CANDIDATE_ROOT = results, RECORDS, CANDIDATE / recipe
    adapter.RAW_ACCEPTANCE = results / "raw_acceptance.json"
    adapter.SOURCE_ACCEPTANCE_REL = "results/raw_acceptance.json"
    adapter.TRAJECTORY_ID, adapter.RUN_ID, adapter.MODE = TRAJECTORIES[recipe], RUN_ID, recipe
    adapter.ACTION_ID, adapter.EVIDENCE_ID = "A001", "pcqm-" + recipe.replace("_", "-") + f"-100k-s42-v{ATTEMPT}"
    adapter.COST_EVENT_ID = frozen["actions"][0]["cost_event_ids"][0]
    adapter.LOCAL_RECORD_URI = "external://local-platform-record/" + (CANDIDATE / recipe).relative_to(REPO_ROOT).as_posix()
    adapter.DECISION_OUTCOME = adapter.SCIENTIFIC_STATUS = "POSITIVE_BELOW_GATE" if positive else "NEGATIVE_UNDER_CONTRACT"
    adapter.NEXT_ALLOWED_ACTIONS = []
    adapter.REOPEN_CONDITIONS = ["new mechanism and explicit prospective authorization; no scale-up or extra seed released"]
    adapter.FINALIZED_AT, adapter.MIGRATED_AT = finalized_at, finalized_at
    adapter.PLATFORM, adapter.HARDWARE = "kaggle2", "Tesla_T4_16GB"
    adapter.rel = lambda name: f"{prefix}/rml_plan/trajectory.json" if name == "trajectory.json" else f"{prefix}/{name}"
    adapter.shared_rel = lambda name: f"{REL}/{name}" if name in ("protocol.md", "training_contract.json") else f"{RELEASE_REL}/{name}"
    adapter.make_trace = lambda raw, checkpoint: accepted_native_trace(CANDIDATE / recipe / "canonical_trace.json", raw, checkpoint)
    names = ("native_cost.json", "observed_role_history.json", "arm_record.json", "completion_manifest.json", "objective_config.json", "target_transform_asset.json")
    for name in names:
        save(results / name, load(CANDIDATE / recipe / name))
    save(results / "cost_allocation.json", allocation)
    adapter.EXTRA_ARTIFACTS = tuple((Path(n).stem, f"{prefix}/results/{n}") for n in (*names, "cost_allocation.json")) + (
        ("submission_receipt", f"{REL}/submission_receipt_v{ATTEMPT}.json"),
        ("study_analysis", f"{RELEASE_REL}/results/analysis.json"))
    original_write = adapter.write

    def write(path, value):
        if Path(path) == results / "cost_records.json":
            event = value["costs"][0]
            event["attempt_id"] = f"v{ATTEMPT}"
            for kind in ("device", "wall"):
                event["measurement"][kind + "_hours"] = {"status": "measured", "value": allocation[kind + "_seconds"] / 3600}
        original_write(path, value)

    adapter.write = write
    adapted = json.loads(json.dumps(accepted))
    for row in adapted["candidates"].values():
        row["gate"].update(training_stochasticity_accounted=False, row_bootstrap_is_sufficient_alone=False)
    adapter.write(adapter.RAW_ACCEPTANCE, adapted)
    adapter.main(observed_comparison_identity=candidate["comparison_identity"],
        experiment_purpose="training_objective_comparison", declared_intervention_fields=("loss_identity",),
        intervention_group_id="joint-training-objective", evidence_scope="terminal_fixed_100k_training_objective_screen")
    if file_digest(results / "canonical_trace.json") != file_digest(CANDIDATE / recipe / "canonical_trace.json"):
        raise ValueError("Native trace bytes changed")
    return {"trajectory": f"{prefix}/rml_plan/trajectory.json", "terminal": f"{prefix}/results/terminal.json",
            "trace": f"{prefix}/results/canonical_trace.json"}


def prepare_audit(finalized_at):
    prefix = f"{RELEASE_REL}/audit"
    results = REPO_ROOT / prefix / "results"
    accepted, analysis = load(RESULTS / "acceptance.json"), load(RESULTS / "analysis.json")
    if not accepted["audit"]["accepted"] or any(r["shortlist_gate"] for r in accepted["audit"]["arms"].values()):
        raise ValueError("This closure requires accepted, non-promoted audit outcomes")
    save(results / "acceptance_summary.json", accepted["audit"])
    save(results / "role_row_manifests.json", load(RESULTS / "role_row_manifests.json"))
    execution = {"native_audit_terminals": {}, "native_cost": {}, "allocation": analysis["native_cost_allocation"]}
    for recipe in RECIPES:
        terminal = load(CANDIDATE / recipe / "post100k_audit/terminal.json")
        execution["native_audit_terminals"][recipe] = terminal
        execution["native_cost"][recipe] = load(CANDIDATE / recipe / "native_cost.json")
    save(results / "execution.json", execution)
    missing = {"status": "measurement_missing", "value": None}
    costs = {name: dict(missing) for name in ("wall_hours", "cpu_hours", "queue_hours")}
    costs["device_hours"] = {"status": "measured", "value": analysis["native_cost_allocation"]["audit_device_seconds"] / 3600}
    artifact_refs = [f"{prefix}/results/{name}.json" for name in ("acceptance_summary", "role_row_manifests", "execution", "role_history", "cost_records")]
    artifact_refs += [f"{RELEASE_REL}/results/analysis.json", f"{REL}/submission_receipt_v{ATTEMPT}.json"]
    outcome = {"execution_status": "complete", "artifact_status": "accepted", "comparison_status": "paired_endpoint_diagnostic",
        "scientific_status": "NO_TRAIN", "transfer_status": "neither_arm_passed_frozen_portability_gate",
        "budget_decision": "stop_under_contract", "full_handoff_status": "not_authorized"}
    return prepare_no_train_terminal(prefix=prefix, frozen=load(REPO_ROOT / prefix / "rml_plan/trajectory.json"),
        run_id=RUN_ID, evidence_id=f"pcqm-k1-joint-atom-post100k-audit-s42-v{ATTEMPT}", outcome=outcome,
        scope="frozen_checkpoint_portability_not_500K_training", finalized_at=finalized_at,
        artifact_refs=artifact_refs, authority=[f"{REL}/protocol.md", f"{REL}/training_contract.json",
            f"{RELEASE_REL}/audit_role_plan.json", f"{prefix}/rml_plan/trajectory.json", f"{prefix}/decision.md"],
        acceptance_name="acceptance_summary", repo_root=REPO_ROOT, cost_measurement=costs,
        attempt_id=f"v{ATTEMPT}", cost_semantics=analysis["native_cost_allocation"]["semantics"])
