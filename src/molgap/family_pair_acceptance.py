"""Saved-output acceptance for a prospective shared-contract reference/candidate.

No model construction, inference, platform submission or automatic RML promotion.
The existing terminal workflow consumes these independently checked artifacts.
"""
from pathlib import Path
import math

from .experiment_family_workflow import RunContext, _artifact_path, _check_context, _json, inspect_output
from .experiment_package import verify_experiment_source_package
from .experiment_retention import validate_execution_retention
from .experiment_allocation import _prior_segments
from .family_training_runner import read_recipe, validate_screen_recipe, _verify_preflight
from .family_pair_release import bounded_cycle_pair, WALL_CEILING_SECONDS
from .training_reproducibility import atomic_json, sha256_file


def _validate_campaign_reference(previous, *, package_dir, package, reference_package_dir):
    """Bind slot2's retained comparator to the frozen slot1 arm and budget."""
    import json
    import hashlib
    from .experiment_family_workflow import _digest
    from .experiment_package import read_packaged_text
    from .experiment_spec import ExperimentSpec
    from .gptrans_campaign_release import BASE
    from .screen_policy import canonical_fingerprint

    campaign = json.loads(read_packaged_text(Path(package_dir), BASE + "/campaign.json"))
    first_pin = campaign["slots"][0]
    first_text = read_packaged_text(Path(package_dir), first_pin["path"])
    if hashlib.sha256(first_text.encode()).hexdigest() != first_pin["sha256"]:
        raise ValueError("Frozen slot1 Spec source pin changed")
    first = ExperimentSpec(json.loads(first_text))
    frozen = campaign["arms"]["reference"]
    from .family_training_runner import _recipe_configuration
    contract_text = read_packaged_text(Path(package_dir), frozen["contract"]["path"])
    if hashlib.sha256(contract_text.encode()).hexdigest() != frozen["contract"]["sha256"]:
        raise ValueError("Frozen campaign reference recipe source pin changed")
    record = json.loads(contract_text)
    recipe = _recipe_configuration(record)
    if reference_package_dir is None:
        raise ValueError("Slot2 requires the independently accepted slot1 source package")
    original = verify_experiment_source_package(Path(reference_package_dir))
    if (read_packaged_text(Path(reference_package_dir), first_pin["path"]) != first_text
            or read_packaged_text(Path(reference_package_dir), frozen["contract"]["path"]) != contract_text):
        raise ValueError("Original reference package differs from frozen slot1 Spec/recipe")
    if (previous.get("format") != "molgap-family-pair-acceptance-v1" or previous.get("accepted") is not True
            or previous.get("acceptance_order") != ["reference", "cycle"]
            or previous.get("spec_identity") != first.identity
            or previous.get("source_commit") != package["source_commit"]
            or original["spec_identity"] != first.identity
            or any(previous.get(key) != original[pack_key] for key, pack_key in (
                ("source_commit", "source_commit"), ("source_archive_sha256", "archive_sha256"),
                ("package_identity", "package_identity")))):
        raise ValueError("Missing/mismatching actual accepted campaign reference")
    for key in ("source_archive_sha256", "package_identity"):
        _digest(previous.get(key), "accepted_reference." + key)
    reference = previous.get("arms", {}).get("reference", {})
    context = reference.get("context", {})
    validated_context = RunContext(**context)
    arm = first.to_dict()["arms"][0]
    if (reference.get("accepted") is not True or reference.get("role") != "reference"
            or reference.get("recipe_identity") != recipe.identity
            or context.get("spec_identity") != first.identity or context.get("arm_id") != "reference"
            or context.get("arm_identity") != canonical_fingerprint(arm)
            or context.get("training_recipe_sha256") != frozen["contract"]["sha256"]
            or validated_context.experiment_id != first.to_dict()["experiment_id"]
            or validated_context.logical_run_id != first.to_dict()["logical_run_id"]
            or validated_context.platform != first.to_dict()["platform"]["name"]
            or validated_context.platform_version is None
            or (validated_context.family_name, validated_context.family_version) != (
                arm["family"]["name"], arm["family"]["version"])
            or any(context.get(k) != previous[k] for k in ("source_commit", "source_archive_sha256", "package_identity"))):
        raise ValueError("Accepted reference is not the exact frozen campaign baseline")
    inspection = reference.get("inspection", {})
    progress = inspection.get("observed", {}).get("progress", {})
    if (inspection.get("status") != "MECHANICALLY_VERIFIED" or progress.get("status") != "complete"
            or progress.get("optimizer_steps") != recipe.optimizer_steps
            or progress.get("sample_presentations") != record["acceptance_requirements"]["sample_presentations"]
            or progress.get("completed_passes") != 60 or progress.get("next_batch") != 0):
        raise ValueError("Accepted reference lacks the complete frozen scientific endpoint")
    full = previous.get("full_allocation", {})
    wall, device = full.get("wall_seconds"), full.get("allocated_device_seconds")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (wall, device))
            or wall > WALL_CEILING_SECONDS or device > 16 * 3600
            or not math.isclose(device, 2 * wall, rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError("Accepted reference full allocated T4 cost is invalid")
    return reference, device


def accept_pair(spec, *, package_dir, receipt_path, output_root, contracts, destination,
                accepted_reference=None, reference_package_dir=None):
    """Accept B first, then compare A to B's verified saved predictions."""
    import torch
    from .k1_terminal_analysis import paired_saved_errors
    from .research_memory.trace import validate_canonical_trace
    campaign = spec.to_dict()["experiment_id"] in {"gptrans-research-slot1", "gptrans-research-slot2"}
    if not bounded_cycle_pair(spec) and not campaign:
        raise ValueError("Unsupported same-run training pair")
    package = verify_experiment_source_package(Path(package_dir))
    root = Path(output_root).absolute()
    retained = validate_execution_retention(root, spec, package_identity=package["package_identity"],
        source_commit=package["source_commit"], source_archive_sha256=package["archive_sha256"])
    ledger = retained["ledgers"]["root"]
    segments = _prior_segments([ledger], spec.identity)
    wall = sum(s["wall_seconds"] for s in segments)
    device = sum(s["allocated_device_seconds"] for s in segments)
    retired_cost = {"wall_seconds": 0.0, "allocated_device_seconds": 0.0}
    if campaign:
        from .gptrans_campaign_release import BASE, retired_bootstrap_cost
        from .experiment_package import read_packaged_text
        number = 1 if spec.to_dict()["experiment_id"] == "gptrans-research-slot1" else 2
        retired_cost = retired_bootstrap_cost(None, number, source_text=read_packaged_text(
            Path(package_dir), f"{BASE}/infrastructure_costs/slot{number}.json"))
        wall += retired_cost["wall_seconds"]
        device += retired_cost["allocated_device_seconds"]
    if (ledger["status"] != "complete" or ledger["prior_unobserved_intervals"] or
            wall > WALL_CEILING_SECONDS or device > 16 * 3600 or
            any(s["status"] != "complete" for s in segments)):
        raise ValueError("Incomplete, uncertain or over-budget whole allocation cost")
    if campaign:
        order = [a["arm_id"] for a in spec.to_dict()["arms"]]
        reference_id = "reference"
        if order not in (["reference", "cycle"], ["carry", "correspondence"]):
            raise ValueError("Campaign physical arm coverage changed")
    else:
        pair = spec.to_dict()["prospective"]["same_run_replay"]
        order = [pair["reference_arm_id"], *pair["candidate_arm_ids"]]
        reference_id = order[0]
    reports, payloads = {}, {}
    prior_device_seconds = 0
    if campaign and order[0] != reference_id:
        if accepted_reference is None:
            raise ValueError("Slot2 comparison waits for independently accepted slot1 reference")
        previous = _json(Path(accepted_reference))
        reference, prior_device_seconds = _validate_campaign_reference(previous,
            package_dir=package_dir, package=package, reference_package_dir=reference_package_dir)
        for pointer in (reference["checkpoint"], reference["predictions"]):
            if sha256_file(Path(pointer["path"])) != pointer["sha256"]:
                raise ValueError("Actual accepted campaign reference changed")
        if prior_device_seconds + device > 32 * 3600:
            raise ValueError("Whole campaign exceeded32allocatedT4hours")
        reports[reference_id] = reference
        payloads[reference_id] = torch.load(reference["predictions"]["path"], map_location="cpu", weights_only=True)
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
        if (recipe.mode == "degree_local_cycle_common_ema999" or campaign) and resource.get("zero_start", {}).get("accepted") is not True:
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
        reports[arm_id] = {"accepted": True, "role": "reference" if arm_id == reference_id else "candidate",
            "recipe_identity": recipe.identity, "context": context.to_dict(),
            "checkpoint": {"path": str(_artifact_path(arm, artifacts["selected_model"]["path"])),
                "sha256": artifacts["selected_model"]["sha256"]},
            "predictions": {"path": str(_artifact_path(arm, artifacts["predictions"]["path"])),
                "sha256": artifacts["predictions"]["sha256"]},
            "trace_sha256": artifacts["trace"]["sha256"], "inspection": report}
        # B becomes an accepted saved-output reference only here, never at release.
        atomic_json(Path(destination) / (arm_id + "_acceptance.json"), reports[arm_id])
    analyses = {a: paired_saved_errors(payloads[reference_id], payloads[a]) for a in order if a != reference_id}
    analysis = next(iter(analyses.values()))
    result = {"format": "molgap-family-pair-acceptance-v1", "accepted": True,
        "spec_identity": spec.identity, "package_identity": package["package_identity"],
        "source_commit": package["source_commit"], "source_archive_sha256": package["archive_sha256"],
        "acceptance_order": order, "arms": reports, "paired_analysis": analysis,
        "candidate_analyses": analyses,
        "candidate_material_gates": {name: -value["candidate_minus_reference_eV"] > .001 and
            value["paired_row_bootstrap"]["ci95"][1] < 0 for name, value in analyses.items()},
        "material_gate_eV": .001, "material_gate_passed": (
            -analysis["candidate_minus_reference_eV"] > .001 and analysis["paired_row_bootstrap"]["ci95"][1] < 0),
        "full_allocation": {"wall_seconds": wall, "allocated_device_seconds": device},
        "retired_bootstrap_cost": retired_cost,
        "campaign_allocated_device_seconds": prior_device_seconds + device,
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
    campaign = accepted.get("campaign_allocated_device_seconds") is not None
    spent = accepted["campaign_allocated_device_seconds"] if campaign else accepted["full_allocation"]["allocated_device_seconds"]
    remaining = ((32 if campaign else 16) * 3600 - spent) / 2
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
    root = Path(recipe_path).resolve().parents[3 if recipe.mode.startswith("degree_aggregate_") else 4]
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
