"""Exact-schema saved-artifact acceptance; no model construction or execution."""
from __future__ import annotations
import argparse
import importlib.util
import math
from pathlib import Path

from .constants import REPO_ROOT
from .k1_joint_objective import ATOM_FEATURE_DIMS, AugmentationCounter, objective_config, objective_fingerprint
from .k1_joint_study_runtime import AUDIT_TRAJECTORY, RECIPES, RUN_ID, TRAJECTORIES
from .k1_joint_study_records import REL, REFERENCE, TRANSFORM, TRANSFORM_FILE_SHA, load, save
from .research_memory.trace import file_digest, load_canonical_trace
from .screen_policy import REFERENCE_MATCH_FIELDS, validate_runtime_certificate, validate_screen_arm

MODE = "neural_atom_k1_v4"
AUDIT_RELEASE = REPO_ROOT / "experiments/pcqm_k1_relation_resolution_100k/audit_release.json"


def _shared():
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("joint_saved_k1_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify_objective_preflight(preflight, recipe):
    checks = preflight["objective_training"]
    for key in ("accepted", "corruption_replayable", "selected_rows_are_other_categories",
        "finite_component_losses", "finite_gradients", "finite_optimizer", "clean_eval_finite",
        "clean_eval_no_corruption", "resume_counter_replayable", "head_initialization_rng_unchanged",
        "same_encoder_initialization", "auxiliary_head_gradient_ok"):
        _require(checks.get(key) is True, f"Missing objective preflight: {key}")
    _require(checks["objective_fingerprint"] == objective_fingerprint(recipe), "Preflight objective changed")
    _require(checks["objective_config"] == objective_config(recipe), "Preflight recipe changed")
    _require(checks["physical_batch_size"] == 128, "Preflight batch changed")


def verify_observed_recipe(record, root, recipe, reference):
    """Qualify the declared delta from actual contracts and transform bytes."""
    from .comparison_readiness import validate_target_transform_asset
    root = Path(root)
    _require(record["objective"]["config"] == load(root / "objective_config.json") == objective_config(recipe), "Objective configuration changed")
    contract, baseline = record["contract"], reference["contract"]
    _require(contract["loss_fingerprint"] == objective_fingerprint(recipe), "Loss fingerprint mismatch")
    _require(contract["loss_fingerprint"] != baseline["loss_fingerprint"], "No actual objective intervention")
    for field in REFERENCE_MATCH_FIELDS:
        if field not in {"loss_fingerprint", "target_transform_fingerprint"}:
            _require(contract[field] == baseline[field], f"Undeclared contract delta: {field}")
    _require(contract["architecture_fingerprint"] == baseline["architecture_fingerprint"], "Inference architecture changed")
    _require(record["preflight"]["shared_k1_initial_state_sha256"] == reference["preflight"]["shared_k1_initial_state_sha256"], "Encoder initialization changed")
    _require(file_digest(REPO_ROOT / TRANSFORM) == TRANSFORM_FILE_SHA, "Frozen transform file changed")
    asset = validate_target_transform_asset(load(REPO_ROOT / TRANSFORM))
    observed = load(root / "target_transform_asset.json")
    _require(all(observed.get(k) == v for k, v in asset.items()), "Observed transform asset differs")
    _require(observed["file_sha256"] == TRANSFORM_FILE_SHA, "Observed transform file hash differs")
    _require(observed["train_target_sha256"] == asset["target_sha256"] and observed["train_rows"] == 100000 and observed["train_source_indices_verified"] is True, "Observed training target identity differs")
    _require(observed["applied_mean_eV"] == asset["mean"] and observed["applied_sample_std_eV"] == asset["std"], "Applied frozen transform differs")
    _require(observed["transform_source"] == "immutable_asset_exact_values" and observed["computed_statistics_usage"] == "diagnostic_only_not_transform", "Target transform was recomputed")
    for key, expected in (("target_transform_fingerprint", asset["asset_id"]),
        ("target_transform_asset_sha256", asset["asset_sha256"]), ("target_transform_file_sha256", TRANSFORM_FILE_SHA)):
        _require(contract[key] == expected, f"Transform contract mismatch: {key}")
    validate_runtime_certificate(record["runtime_certificate"], contract)
    validate_screen_arm(physical_batch_per_device=contract["physical_batch_per_device"],
        device_count=contract["device_count"], gradient_accumulation_steps=contract["gradient_accumulation_steps"])
    # Legacy fingerprints encode a formula; the recovered asset binds exact
    # statistics. All other raw controls were checked above, not inferred.
    return dict(load(REPO_ROOT / REFERENCE)["comparison_identity"], loss_identity=contract["loss_fingerprint"])


def _checkpoint_and_trace(root, record, recipe):
    import torch
    checkpoint = torch.load(root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    for key, expected in (("trajectory_id", TRAJECTORIES[recipe]), ("physical_run_id", RUN_ID),
        ("source_commit", record["source_commit"]), ("source_archive_sha256", record["contract"]["source_archive_sha256"]),
        ("mode", MODE), ("epoch", 39)):
        _require(checkpoint[key] == expected, f"Checkpoint identity mismatch: {key}")
    state = checkpoint["objective_state"]
    _require(state["objective_fingerprint"] == objective_fingerprint(recipe), "Checkpoint objective changed")
    _require(state["objective_config"] == objective_config(recipe), "Checkpoint config changed")
    _require(state["head_initialization_rng_unchanged"] is True, "Head initialization altered RNG")
    counter = AugmentationCounter()
    counter.load_state_dict(state["augmentation_counter_state"])
    _require(counter.optimizer_steps == 31240 and counter.next_epoch == 40, "Incomplete augmentation counter")
    heads = checkpoint["objective_heads"]
    _require(set(heads) == {f"heads.{i}.{s}" for i in range(9) for s in ("weight", "bias")}, "Incomplete heads")
    for i, size in enumerate(ATOM_FEATURE_DIMS):
        _require(tuple(heads[f"heads.{i}.weight"].shape) == (size, 192), "Head dimensions changed")
        _require(tuple(heads[f"heads.{i}.bias"].shape) == (size,), "Head bias changed")
    _require(all(torch.isfinite(v).all().item() for v in heads.values()), "Nonfinite head weights")
    _require(checkpoint["rng_state"] and checkpoint["optimizer"]["state"] and checkpoint["scheduler"], "Incomplete continuation state")
    best = torch.load(root / "best_model.pt", map_location="cpu", weights_only=False)
    _require(set(best) == set(checkpoint["model"]), "Inference export changed state keys")
    _require(not any("objective_heads" in k for k in best), "Auxiliary heads leaked into inference")
    trace = load_canonical_trace(root / "canonical_trace.json")
    _require((trace["trajectory_id"], trace["run_id"]) == (TRAJECTORIES[recipe], RUN_ID), "Trace identity changed")
    rows = load(root / "trace.json")["epochs"]
    _require(len(rows) == len(trace["observations"]) == 40 and rows == checkpoint["trace"], "Incomplete/different traces")
    weight = objective_config(recipe)["auxiliary_loss"]["weight"]
    for i, (row, native) in enumerate(zip(rows, trace["observations"])):
        _require(row["epoch"] == i and native["optimizer_step"] == (i + 1) * 781 and native["sample_presentations"] == (i + 1) * 99968, "Exposure changed")
        _require(0 <= row["corrupted_atom_rows"] <= row["total_atom_rows"] and row["total_atom_rows"] > 0, "Invalid corruption counts")
        _require(row["corruption_rate_observed"] == row["corrupted_atom_rows"] / row["total_atom_rows"], "Wrong corruption denominator")
        for key in ("gap_loss", "auxiliary_loss", "total_loss", "backbone_gradient_norm_max", "auxiliary_head_gradient_norm_max"):
            _require(math.isfinite(row[key]), f"Nonfinite telemetry: {key}")
        _require(abs(row["total_loss"] - row["gap_loss"] - weight * row["auxiliary_loss"]) < 1e-5, "Loss terms do not add up")
        _require(native["checkpoint_identity"] and native["live_dev_metric"] == row["development_gap_mae_eV"], "Native trace lost checkpoint or metric")
    _require(trace["observations"][-1]["checkpoint_identity"] == file_digest(root / "last_checkpoint.pt"), "Terminal checkpoint mismatch")
    return rows


def _paired(candidate, reference):
    import numpy as np
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    _require(delta.ndim == 1 and delta.size == 50000 and np.isfinite(delta).all(), "Invalid paired errors")
    upper, interval = _shared()._bootstrap_upper(delta)
    return {"gain_eV": -float(delta.mean()), "paired_error_delta_bootstrap_95_eV": interval,
        "row_interval_favorable": upper < 0, "candidate_win_fraction": float((delta < 0).mean()),
        "training_stochasticity_measured": False}


def _verify_cost_roles(root, recipe):
    cost, roles = load(root / "native_cost.json"), load(root / "observed_role_history.json")
    for item in (cost, roles):
        _require(item["trajectory_id"] == TRAJECTORIES[recipe] and item["run_id"] == RUN_ID, "Cost/role identity changed")
    _require(cost["training_completed"] is True and cost["device_count"] == 1, "Training cost incomplete")
    _require(math.isfinite(cost["wall_seconds"]) and cost["wall_seconds"] > 0 and cost["allocated_device_seconds"] == cost["wall_seconds"], "Invalid native device cost")
    _require(roles["training_membership"] == "official_train_prefix_0_100000" and roles["development_prediction_input"] == "internal_development_100000_150000", "Observed roles changed")
    for field in ("training_labels_read", "development_labels_read", "development_metric_computed", "development_selection_used"):
        _require(roles[field] is True, f"Missing observed role: {field}")
    for field in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        _require(roles[field] is False, "Protected role was accessed")
    return cost


def accept(reference_root, candidate_root, *, source_commit, source_archive_sha256, include_audit=False, audit_reference_root=None):
    import torch
    shared, root = _shared(), Path(candidate_root)
    reference, payload_ref = shared._load_arm(Path(reference_root), MODE)
    validate_runtime_certificate(reference["runtime_certificate"], reference["contract"])
    bundle = load(REPO_ROOT / REFERENCE)
    _require(reference["training"]["best_model_sha256"] == bundle["checkpoint_identity"], "Wrong frozen reference model")
    _require(reference["contract"]["frozen_reference"] is True, "Unfrozen reference")
    errors_ref = (payload_ref["prediction"].double() - payload_ref["target"].double()).abs().numpy()
    result = {"format": "molgap-k1-joint-objective-acceptance-v1", "accepted": True,
        "model_inference_executed": False, "training_executed": False,
        "experiment_purpose": "training_objective_comparison", "declared_intervention_fields": ["loss_identity"],
        "source_commit": source_commit, "source_archive_sha256": source_archive_sha256,
        "run_id": RUN_ID, "reference_development_gap_mae_eV": float(errors_ref.mean()), "candidates": {},
        "replay_status": "pending_controller_terminal_closure", "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False}
    errors, counts = {}, {}
    for recipe in RECIPES:
        arm = root / recipe
        record, payload = shared._load_arm(root, MODE, arm_directory=arm)
        required = {"canonical_trace.json", "observed_role_history.json", "objective_config.json", "target_transform_asset.json", "native_cost.json", *[f"recovery_epoch_{e:02d}.tar" for e in (10, 20, 30, 40)]}
        _require(required <= set(load(arm / "completion_manifest.json")["artifact_sha256"]), "Required artifacts are unbound")
        _require(record["source_commit"] == source_commit and record["contract"]["source_archive_sha256"] == source_archive_sha256, "Frozen source changed")
        _require(record["trajectory_id"] == TRAJECTORIES[recipe] and record["physical_run_id"] == RUN_ID, "Unbound physical run")
        _require(torch.equal(payload["target"], payload_ref["target"]) and torch.equal(payload["source_idx"], payload_ref["source_idx"]), "Candidate/reference rows or targets differ")
        verify_objective_preflight(record["preflight"], recipe)
        identity = verify_observed_recipe(record, arm, recipe, reference)
        rows = _checkpoint_and_trace(arm, record, recipe)
        cost = _verify_cost_roles(arm, recipe)
        counts[recipe] = [(r["corrupted_atom_rows"], r["total_atom_rows"]) for r in rows]
        errors[recipe] = (payload["prediction"].double() - payload["target"].double()).abs().numpy()
        paired = _paired(errors[recipe], errors_ref)
        reserve = 1 - record["training"]["peak_reserved_mib"] / record["training"]["total_memory_mib"]
        _require(math.isfinite(reserve), "Nonfinite memory cost")
        result["candidates"][recipe] = {"record": record, "comparison_identity": identity,
            "recomputed_development_gap_mae_eV": float(errors[recipe].mean()),
            "paired_error_delta_bootstrap_95_eV": paired["paired_error_delta_bootstrap_95_eV"],
            "paired_vs_k1": paired, "native_cost": cost,
            "gate": {"gain_eV": paired["gain_eV"], "required_gain_eV": .003,
                "passed": paired["gain_eV"] >= .003 and paired["row_interval_favorable"] and reserve >= .15,
                "memory_reserve_fraction": reserve, "training_stochasticity_measured": False}}
    _require(counts[RECIPES[0]] == counts[RECIPES[1]], "Arm corruption streams differ")
    result["paired_b_minus_a"] = _paired(errors[RECIPES[1]], errors[RECIPES[0]])
    if include_audit:
        _require(audit_reference_root is not None, "Explicit retained reference audit directory required")
        result["audit"] = accept_audit(root, Path(audit_reference_root), result)
    return result


def _chunks(directory, rows, start):
    import torch
    parts = []
    _require(len(rows) == 10, "Expected ten durable 5K-row audit chunks")
    for i, row in enumerate(rows):
        _require(row["file"] == f"chunk_{i:02d}.pt" and row["rows"] == 5000, "Audit chunk layout changed")
        path = directory / row["file"]
        _require(file_digest(path) == row["sha256"], "Audit chunk hash changed")
        part = torch.load(path, map_location="cpu", weights_only=False)
        _require(torch.equal(part["source_idx"].view(-1).long(), torch.arange(start+i*5000, start+(i+1)*5000)), "Audit rows changed")
        _require(all(torch.isfinite(part[k]).all().item() for k in ("target_eV", "prediction_eV")), "Nonfinite audit chunk")
        parts.append(part)
    return {k: torch.cat([p[k] for p in parts]) for k in ("source_idx", "target_eV", "prediction_eV")}


def accept_audit(candidate_root, reference_root, training_acceptance):
    import torch
    from .pcqm_k1_cross_scale_diagnostic import MANIFESTS
    release = load(AUDIT_RELEASE)
    reference = {}
    for role, start in (("original_100k", 100000), ("unseen_500k", 500000)):
        prefix = f"reference/{role}/{MODE}/"
        rows = [{"file": k.removeprefix(prefix), "sha256": v, "rows": 5000}
                for k, v in sorted(release["input_files"].items()) if k.startswith(prefix)]
        reference[role] = _chunks(reference_root / "reference" / role / MODE, rows, start)
    result = {"accepted": True, "experiment_purpose": "NO_TRAIN", "training_executed": False,
        "model_inference_executed_by_acceptance": False, "trajectory_id": AUDIT_TRAJECTORY,
        "arms": {}, "reference_inference_reused": True, "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False}
    for recipe in RECIPES:
        root = Path(candidate_root) / recipe
        terminal = load(root / "post100k_audit/terminal.json")
        for key, expected in (("complete", True), ("experiment_purpose", "NO_TRAIN"), ("recipe", recipe),
            ("trajectory_id", AUDIT_TRAJECTORY), ("run_id", RUN_ID), ("training_executed_in_audit_stage", False),
            ("optimizer_steps_in_audit_stage", 0), ("selection_used_in_audit_stage", False),
            ("source_commit", training_acceptance["source_commit"]), ("source_archive_sha256", training_acceptance["source_archive_sha256"]),
            ("target_transform_sha256", TRANSFORM_FILE_SHA), ("manifest_sha256", MANIFESTS),
            ("checkpoint_sha256", file_digest(root / "best_model.pt")),
            ("official_validation_role_read", False), ("test_dev_role_read", False), ("test_challenge_role_read", False)):
            _require(terminal[key] == expected, f"Audit identity changed: {key}")
        saved = torch.load(root / "best_development_payload.pt", map_location="cpu", weights_only=False)
        report = {}
        for role, start in (("original_100k", 100000), ("unseen_500k", 500000)):
            payload = _chunks(root / "post100k_audit" / role, terminal[role]["chunks"], start)
            _require(torch.equal(payload["target_eV"], reference[role]["target_eV"]), "Audit targets differ from reference")
            if role == "original_100k":
                difference = float((payload["prediction_eV"].view(-1) - saved["prediction_eV"].view(-1)).abs().max())
                _require(difference <= .001 and difference == terminal["original_reproduction_max_abs_eV"], "Checkpoint reproduction failed")
            errors = (payload["prediction_eV"].double() - payload["target_eV"].double()).abs().numpy()
            errors_ref = (reference[role]["prediction_eV"].double() - reference[role]["target_eV"].double()).abs().numpy()
            report[role] = {"mae_eV": float(errors.mean()), **_paired(errors, errors_ref)}
        original_gain = training_acceptance["candidates"][recipe]["gate"]["gain_eV"]
        audit_gain = report["unseen_500k"]["gain_eV"]
        report["shortlist_gate"] = (training_acceptance["candidates"][recipe]["gate"]["passed"]
            and audit_gain >= .001 and audit_gain >= .5 * original_gain and report["unseen_500k"]["row_interval_favorable"])
        result["arms"][recipe] = report
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--include-audit", action="store_true")
    parser.add_argument("--audit-reference-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, accept(args.reference_root, args.candidate_root, source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256, include_audit=args.include_audit,
        audit_reference_root=args.audit_reference_root))


if __name__ == "__main__":
    main()
