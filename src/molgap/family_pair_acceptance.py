"""Saved-output acceptance for a prospective shared-contract reference/candidate.

No model construction, inference, platform submission or automatic RML promotion.
The existing terminal workflow consumes these independently checked artifacts.
"""
from pathlib import Path
import math

from .experiment_family_workflow import RunContext, _artifact_path, _check_context, _json, inspect_output
from .experiment_package import verify_experiment_source_package
from .experiment_retention import validate_execution_retention
from .family_training_runner import read_recipe, validate_screen_recipe, _verify_preflight
from .family_pair_release import bounded_cycle_pair, WALL_CEILING_SECONDS
from .training_reproducibility import atomic_json, sha256_file


def accept_pair(spec, *, package_dir, receipt_path, output_root, contracts, destination):
    """Accept B first, then compare A to B's verified saved predictions."""
    import torch
    from .k1_terminal_analysis import paired_saved_errors
    from .research_memory.trace import validate_canonical_trace
    if not bounded_cycle_pair(spec):
        raise ValueError("Unsupported same-run training pair")
    package = verify_experiment_source_package(Path(package_dir))
    root = Path(output_root).absolute()
    retained = validate_execution_retention(root, spec, package_identity=package["package_identity"],
        source_commit=package["source_commit"], source_archive_sha256=package["archive_sha256"])
    ledger = retained["ledgers"]["root"]
    segments = [ledger, *ledger["prior_segments"]]
    wall = sum(s["wall_seconds"] for s in segments)
    device = sum(s["allocated_device_seconds"] for s in segments)
    if (ledger["status"] != "complete" or ledger["prior_unobserved_intervals"] or
            wall > WALL_CEILING_SECONDS or device > 16 * 3600 or
            any(s["status"] != "complete" for s in segments)):
        raise ValueError("Incomplete, uncertain or over-budget whole allocation cost")
    pair = spec.to_dict()["prospective"]["same_run_replay"]
    order = [pair["reference_arm_id"], *pair["candidate_arm_ids"]]
    reports, payloads = {}, {}
    for arm_id in order:
        context = RunContext.from_launch(spec, Path(receipt_path), Path(package_dir),
            expected_package_identity=package["package_identity"], arm_id=arm_id)
        record, recipe = read_recipe(Path(contracts[arm_id]))
        validate_screen_recipe(spec, arm_id, record)
        arm = root / arm_id
        report = inspect_output(arm, context=context, expected=record["acceptance_requirements"])
        if report["status"] != "MECHANICALLY_VERIFIED":
            raise ValueError("Independent arm acceptance failed: " + str(report))
        # Restore and validate native proof payloads, not just their status JSON.
        native = arm if not (root / "_resume_preflight" / arm_id).exists() else root / "_resume_preflight" / arm_id
        provenance = _json(native / "runtime_provenance.json")
        _check_context(provenance["context"], context)
        if (provenance["recipe_identity"] != recipe.identity or
                provenance["dataset_manifest_sha256"] != recipe.dataset.manifest_sha256 or
                "T4" not in provenance["accelerator"]):
            raise ValueError("Native runtime identity differs from frozen arm")
        _verify_preflight(native, recipe, provenance)
        resource = _json(native / "resource_qualification.json")
        if recipe.mode == "degree_local_cycle_common_ema999" and resource.get("zero_start", {}).get("accepted") is not True:
            raise ValueError("Missing actual native zero-start equivalence")
        trace = validate_canonical_trace(_json(arm / "canonical_trace.json"))
        observations = [r for r in trace["observations"] if r["event"] == "observation"]
        for name in ("live_train_metric", "live_dev_metric", "ema_dev_metric"):
            if trace["metric_semantics"][name] is None or any(
                    type(r[name]) not in (float, int) or not math.isfinite(r[name]) for r in observations):
                raise ValueError("Missing finite complete live/EMA trajectory")
        if len(observations) != 60:
            raise ValueError("A ten-pass physical segment is not a completed scientific screen")
        artifacts = report["observed"]["artifacts"]
        payloads[arm_id] = torch.load(_artifact_path(arm, artifacts["predictions"]["path"]),
                                    map_location="cpu", weights_only=True)
        reports[arm_id] = {"accepted": True, "role": "reference" if arm_id == order[0] else "candidate",
            "recipe_identity": recipe.identity, "context": context.to_dict(),
            "checkpoint": {"path": str(_artifact_path(arm, artifacts["selected_model"]["path"])),
                "sha256": artifacts["selected_model"]["sha256"]},
            "predictions": {"path": str(_artifact_path(arm, artifacts["predictions"]["path"])),
                "sha256": artifacts["predictions"]["sha256"]},
            "trace_sha256": artifacts["trace"]["sha256"], "inspection": report}
        # B becomes an accepted saved-output reference only here, never at release.
        atomic_json(Path(destination) / (arm_id + "_acceptance.json"), reports[arm_id])
    analysis = paired_saved_errors(payloads[order[0]], payloads[order[1]])
    result = {"format": "molgap-family-pair-acceptance-v1", "accepted": True,
        "spec_identity": spec.identity, "package_identity": package["package_identity"],
        "source_commit": package["source_commit"], "source_archive_sha256": package["archive_sha256"],
        "acceptance_order": order, "arms": reports, "paired_analysis": analysis,
        "material_gate_eV": .001, "material_gate_passed": (
            -analysis["candidate_minus_reference_eV"] > .001 and analysis["paired_row_bootstrap"]["ci95"][1] < 0),
        "full_allocation": {"wall_seconds": wall, "allocated_device_seconds": device},
        "rml_finalization": "NOT_EXECUTED_USE_EXISTING_TERMINAL_WORKFLOW",
        "training_executed": False, "model_inference_executed": False,
        "caveat": "Paired single-seed internal-development evidence, not seed stability, scale or protected-test superiority"}
    atomic_json(Path(destination) / "pair_acceptance.json", result)
    return result


def bind_deferred_audit(*, acceptance_path, recipe_path, arm_id, maximum_wall_seconds=1800):
    """Bind real accepted bytes after training; this does not execute the audit."""
    accepted = _json(Path(acceptance_path))
    if (accepted.get("format") != "molgap-family-pair-acceptance-v1" or accepted.get("accepted") is not True or
            type(maximum_wall_seconds) is not int or not 0 < maximum_wall_seconds <= 1800):
        raise ValueError("Audit requires independently accepted training")
    arm = accepted["arms"][arm_id]
    _, recipe = read_recipe(Path(recipe_path))
    if arm["recipe_identity"] != recipe.identity:
        raise ValueError("Audit recipe differs from accepted training")
    remaining = (16 * 3600 - accepted["full_allocation"]["allocated_device_seconds"]) / 2
    limit = min(maximum_wall_seconds, remaining)
    if limit <= 120:
        raise ValueError("No whole-allocation budget remains for the deferred audit")
    for pointer in (arm["checkpoint"], arm["predictions"]):
        if sha256_file(Path(pointer["path"])) != pointer["sha256"]:
            raise ValueError("Accepted parent changed before deferred binding")
    from .gptrans_cycle_inputs import FIXED_100K_SHA
    from .gptrans_cycle_release import BASE
    # The sidecar is the same accepted CPU asset, including the separately
    # predeclared500Kdevelopment role. The baseline never consumes these features.
    root = Path(recipe_path).resolve().parents[4]
    cache = _json(root / BASE / "cpu/acceptance.json")
    if recipe.dataset.manifest_sha256 != FIXED_100K_SHA or cache.get("accepted") is not True:
        raise ValueError("Deferred audit input identity changed")
    return {"training_independently_accepted": True, "training_acceptance_sha256": sha256_file(Path(acceptance_path)),
        "mode": recipe.mode, "audit_kind": "fixed500k-full50k-NO_TRAIN", "batch_size": 128, "tf32": False,
        "checkpoint": arm["checkpoint"]["path"], "checkpoint_sha256": arm["checkpoint"]["sha256"],
        "original_predictions": arm["predictions"]["path"], "original_predictions_sha256": arm["predictions"]["sha256"],
        "sidecar_manifest": "cycles/manifest.json", "sidecar_manifest_sha256": cache["manifest_sha256"],
        "target_mean_eV": recipe.target_mean_eV, "target_std_eV": recipe.target_std_eV,
        "maximum_wall_seconds": limit, "selection_used": False}
