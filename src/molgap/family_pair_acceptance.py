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


def verify_session_reference(root, bundle):
    """Reuse a complete finalized Session control without another training run."""
    from .comparison_readiness import validate_reference_bundle
    from .research_memory.finalize import verified_receipt
    from .research_memory.paths import resolve_repo_pointer, verify_bound_artifact
    from .research_memory.schemas import validate_trace_manifest
    root = Path(root).resolve()
    bundle = validate_reference_bundle(bundle)
    trajectory_path = resolve_repo_pointer(root, bundle["finalized_reference_ref"])
    directory = trajectory_path.parent
    verified_receipt(directory)
    trajectory, evidence = _json(trajectory_path), _json(directory / "v5_evidence.json")
    trace = validate_trace_manifest(_json(directory / "trace_manifest.json"))
    if (trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] == "ACTIVE"
            or evidence["outcome"]["execution_status"] != "complete"
            or evidence["outcome"]["artifact_status"] != "accepted"
            or evidence["evidence_id"] != bundle["reference_id"]
            or evidence["evidence_id"] not in trajectory["result"]["evidence_ids"]
            or trace["comparison_role"] != "reference" or trace["reference_id"] != bundle["reference_id"]
            or trace["backtest_eligibility"] != {"eligible": True, "exclusion_reasons": []}
            or resolve_repo_pointer(root, bundle["trace_manifest_ref"]) != directory / "trace_manifest.json"):
        raise ValueError("Session reference is not its own complete accepted control")
    terminal = _json(directory / "terminal_input.json")
    original_ref = terminal["observed_reference_ref"]
    verify_bound_artifact(root, original_ref, terminal["artifact_hashes"][original_ref])
    original = _json(resolve_repo_pointer(root, original_ref))
    mutable = {"reference_bundle_id", "trace_manifest_ref", "finalized_reference_ref"}
    if {key: value for key, value in bundle.items() if key not in mutable} != {
            key: value for key, value in original.items() if key not in mutable}:
        raise ValueError("Session reference changed its accepted observed comparison identity")
    for field in ("runtime_certificate_ref", "row_manifest_ref", "target_manifest_ref",
                  "role_history_ref", "target_transform_asset_ref", "cost_records_ref", "acceptance_ref", "decision_ref",
                  "checkpoint_ref", "prediction_manifest_ref", "source_config_ref"):
        pointer = bundle[field]
        digest = terminal["artifact_hashes"].get(pointer)
        if digest is None:
            raise ValueError("Session reference pointer is not bound by its accepted transaction: " + field)
        verify_bound_artifact(root, pointer, digest)
    prediction = bundle["prediction_manifest"]
    if (terminal["artifact_hashes"].get(bundle["checkpoint_ref"]) != bundle["checkpoint_identity"]
            or terminal["artifact_hashes"].get(prediction["artifact_locator"]) != prediction["prediction_sha256"]):
        raise ValueError("Session reference checkpoint/prediction differs from its accepted output")
    verify_bound_artifact(root, prediction["artifact_locator"], prediction["prediction_sha256"])
    return bundle


def register_session_reference(root, spec, staged):
    """Expose the accepted baseline's standard bundle; no new evidence or cost."""
    import copy
    from .common.io import publish_immutable_bytes
    from .research_memory.trace import json_bytes
    root = Path(root).resolve()
    pair = spec.to_dict()["prospective"].get("same_run_replay")
    if pair is None:
        return None
    arm_id = pair["reference_arm_id"]
    source = Path(staged["paths"][arm_id]["terminal"]).parent / "reference_bundle.json"
    if not source.is_file():
        return None
    directory = (root / staged["locations"][arm_id]["trajectory"]).parent
    bundle = copy.deepcopy(_json(source))
    bundle.update(reference_bundle_id="accepted-" + bundle["reference_id"],
        finalized_reference_ref=(directory / "rml_finalized/trajectory.json").relative_to(root).as_posix(),
        trace_manifest_ref=(directory / "rml_finalized/trace_manifest.json").relative_to(root).as_posix())
    verify_session_reference(root, bundle)
    path = directory / "reference/reference_bundle.json"
    publish_immutable_bytes(path, json_bytes(bundle))
    return {"reference_id": bundle["reference_id"], "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path)}


def complete_pair_terminal_packages(root, spec, *, comparison, outputs, staged):
    """Bind a frozen same-run comparison to observed Session outputs.

    This adapter consumes saved bytes only. It leaves the existing finalizer,
    reference-first transaction and replay admission in charge of publication.
    """
    import copy
    import torch
    from .common.io import publish_immutable_bytes
    from .comparison_readiness import (
        REQUIRED_OBSERVED_BINDINGS,
        assess_comparison_readiness, reference_bundle_digest,
        validate_comparison_readiness, validate_reference_bundle,
        validate_target_transform_asset,
    )
    from .research_memory.paths import artifact_locator, bound_artifact_path, verify_bound_artifact
    from .research_memory.trace import json_bytes, load_canonical_trace
    from .screen_policy import canonical_fingerprint

    root = Path(root).resolve()
    pair = spec.to_dict()["prospective"].get("same_run_replay")
    order = ([pair["reference_arm_id"], *pair["candidate_arm_ids"]] if pair is not None
             else [arm["arm_id"] for arm in spec.to_dict()["arms"]])
    if staged["blockers"]:
        return staged
    transform = validate_target_transform_asset(comparison["target_transform_asset"])
    transform_path = next((path for path in root.glob("experiments/**/target_transform.json")
                           if _json(path) == transform), None)
    purpose = comparison["experiment_purpose"]
    packages, paths, sides, payloads = {}, {}, {}, {}
    if pair is not None:
        reference_id = "ev-terminal-" + next(b["trajectory_id"] for b in
            spec.to_dict()["prospective"]["arms"] if b["arm_id"] == order[0])
    else:
        pointer = comparison["reference_bundle"]
        bundle_path = bound_artifact_path(root, pointer["path"])
        verify_bound_artifact(root, pointer["path"], pointer["sha256"])
        bundle = verify_session_reference(root, _json(bundle_path))
        reference_id = bundle["reference_id"]

    for arm_id in order:
        package_path = Path(staged["paths"][arm_id]["terminal"])
        package = copy.deepcopy(_json(package_path))
        item, identity = outputs[arm_id], comparison["identities"][arm_id]
        output = Path(item["output_dir"]).absolute()
        context = item["context"]
        manifest = _json(_artifact_path(output, "output_manifest.json"))
        _check_context(manifest["context"], context)
        artifacts = manifest["artifacts"]
        record, recipe = read_recipe(_artifact_path(output, artifacts["contract"]["path"]))
        validate_screen_recipe(spec, arm_id, record)
        # The exact worker gate verified repeat/resume before training. Retain
        # its certificate and provenance rather than infer native acceptance
        # from successful training or require another download of model states.
        certificate_path = _artifact_path(output, artifacts["runtime_certificate"]["path"])
        provenance_path = _artifact_path(output, artifacts["runtime_provenance"]["path"])
        certificate, provenance = _json(certificate_path), _json(provenance_path)
        _check_context(provenance["context"], context)
        from .family_training_runner import _probe_recipe
        if (certificate.get("format") != "molgap-runtime-certificate-v1"
                or certificate.get("status") != "accepted"
                or certificate.get("provenance_sha256") != canonical_fingerprint(provenance)
                or certificate.get("probe_recipe_identity") != _probe_recipe(recipe).identity
                or certificate.get("formal_training_exposure") != 0
                or provenance.get("recipe_identity") != recipe.identity
                or manifest["runtime"].get("runtime_certificate_id") != canonical_fingerprint(certificate)):
            raise ValueError("Saved runtime certificate is not bound to this scientific Session")
        trace = load_canonical_trace(_artifact_path(output, artifacts["trace"]["path"]))
        rows = [row for row in trace["observations"] if row["event"] == "observation"]
        if (not rows or rows[-1]["optimizer_step"] != recipe.optimizer_steps
                or rows[-1]["sample_presentations"] != record["acceptance_requirements"]["sample_presentations"]):
            raise ValueError("Scientific trace does not reach its frozen endpoint")
        fields = {field: all(row.get(field) is not None for row in rows)
                  for field in comparison["trace_plan"]}
        payload = torch.load(_artifact_path(output, artifacts["predictions"]["path"]),
                             map_location="cpu", weights_only=True)
        payloads[arm_id] = payload
        directory = package_path.parent / "scientific"
        def save(name, value):
            path = directory / name
            publish_immutable_bytes(path, json_bytes(value))
            return path.relative_to(root).as_posix()
        def bind(path):
            return {"ref": artifact_locator(root, path), "sha256": sha256_file(path)}
        source_config = directory / "source_config.json"
        publish_immutable_bytes(source_config, _artifact_path(output, artifacts["contract"]["path"]).read_bytes())
        if transform_path is None:
            transform_path = directory / "target_transform.json"
            publish_immutable_bytes(transform_path, json_bytes(transform))
        pathmap = {"checkpoint": _artifact_path(output, artifacts["selected_model"]["path"]),
            "source_config": source_config,
            "target_transform_asset": transform_path,
            "decision": bound_artifact_path(root, package["decision"]["decision_ref"])}
        accepted = copy.deepcopy(_json(bound_artifact_path(root, package["acceptance_ref"])))
        dimensions = {**package["evidence"]["outcome"],
            "artifact_status": "accepted", "comparison_status": "strict_causal"}
        package["evidence"]["outcome"] = accepted["outcome"] = dimensions
        accepted_ref = save("scientific_acceptance.json", accepted)
        package["acceptance_ref"] = accepted_ref
        pathmap["acceptance"] = root / accepted_ref
        prediction_manifest = {"prediction_sha256": artifacts["predictions"]["sha256"],
            "source_idx_sha256": record["acceptance_requirements"]["source_idx_sha256"],
            "target_sha256": record["acceptance_requirements"]["target_sha256"],
            "row_count": int(payload["source_idx"].numel()),
            "unique_source_idx": int(payload["source_idx"].unique().numel()),
            "ordering_semantics": "frozen-source-index-order",
            "evaluation_role_identity": record["development_role_identity"],
            "artifact_locator": artifact_locator(root, _artifact_path(output, artifacts["predictions"]["path"]))}
        docs = {"prediction_manifest": prediction_manifest,
            "runtime_certificate": certificate,
            "row_manifest": {"source_idx_sha256": prediction_manifest["source_idx_sha256"],
                "rows": prediction_manifest["row_count"]},
            "target_manifest": {"target_sha256": prediction_manifest["target_sha256"],
                "rows": prediction_manifest["row_count"], "unit": "eV"},
            "role_history": {"roles": package["roles"], "role_use": package["role_use"]},
            "cost_records": {"costs": package["costs"]}}
        manifest_trace = package["trace_manifest"]
        manifest_trace.update(reference_id=reference_id, model_identity=identity["architecture_config_identity"],
            evaluation_role_identity=identity["evaluation_role_identity"],
            selection_semantics=identity["checkpoint_selection_identity"],
            weight_semantics=identity["weight_semantics"],
            trace_fields=fields, backtest_eligibility={"eligible": True, "exclusion_reasons": []})
        manifest_trace["comparability_identity"] = {
            "scientific_contract": comparison["contract_identity"],
            "dataset_identity": identity["dataset_identity"],
            "row_split_identity": canonical_fingerprint({k: identity[k] for k in
                ("data_role_identity", "row_membership_identity", "row_order_identity")}),
            "architecture_identity": identity["architecture_config_identity"],
            "optimizer_identity": identity["optimizer_identity"],
            "lr_schedule_identity": identity["schedule_identity"],
            "target_transform_identity": identity["target_transform_identity"],
            "precision_identity": identity["precision"],
            "ema_semantics": canonical_fingerprint({k: identity[k] for k in
                ("ema_enabled", "ema_decay", "ema_update_frequency", "evaluation_weight_source")}),
            "evaluation_role_identity": identity["evaluation_role_identity"],
            "selection_role_identity": identity["selection_role_identity"],
            "x_axis_semantics": manifest_trace["x_axis"],
            "terminal_endpoint_identity": canonical_fingerprint({k: identity[k] for k in
                ("sample_presentations", "optimizer_steps", "checkpoint_selection_identity")}),
            "matched_architecture_required": False}
        docs["trace_manifest"] = manifest_trace
        for name, value in docs.items():
            # Staging is not another canonical RML record. The finalizer alone
            # publishes the discoverable trace_manifest.json transaction.
            filename = "observed_trace_manifest.json" if name == "trace_manifest" else name + ".json"
            pathmap[name] = root / save(filename, value)
        paths[arm_id] = pathmap
        sides[arm_id] = {"comparison_identity": identity,
            "artifacts": {name: "complete" for name in pathmap},
            "artifact_bindings": {name: bind(pathmap[name]) for name in REQUIRED_OBSERVED_BINDINGS},
            "terminal_complete": True, "prediction_status": "complete", "row_alignment_status": "aligned",
            "runtime_certificate_status": "accepted", "role_applicability_plan": comparison["role_applicability_plan"],
            "observed_role_event_kinds": sorted({event["access_kind"] for event in package["roles"]}),
            "trace_field_availability": fields, "trace_status": "complete", "stochasticity_status": "unavailable"}
        packages[arm_id] = (package, directory, prediction_manifest)

    if pair is not None:
        ref_package, ref_directory, prediction_manifest = packages[order[0]]
        bundle = {"format": "molgap-reference-bundle-v1", "reference_id": reference_id,
        "reference_bundle_id": "bundle-" + reference_id,
        "contract_ref": artifact_locator(root, paths[order[0]]["source_config"]),
        "architecture_config_identity": comparison["identities"][order[0]]["architecture_config_identity"],
        "source_commit_or_archive": outputs[order[0]]["context"].source_commit,
        "checkpoint_identity": sha256_file(paths[order[0]]["checkpoint"]),
        "prediction_manifest": prediction_manifest, "comparison_identity": comparison["identities"][order[0]],
        "comparison_plan": comparison,
        "stochasticity": {"row_bootstrap_uncertainty": {"status": "not_requested"},
            "training_stochasticity": {"status": "unavailable", "estimation_method": "single paired seed",
                "same_contract_repeat_ids": [], "n_repeats": 0, "stochasticity_floor_eV": None}},
        **{name + "_ref": artifact_locator(root, paths[order[0]][name]) for name in
           ("runtime_certificate", "row_manifest", "target_manifest", "trace_manifest", "role_history",
            "target_transform_asset", "cost_records", "acceptance", "decision",
            "checkpoint", "prediction_manifest", "source_config")}}
        validate_reference_bundle(bundle)
        # Discovery needs the declared bundle before either terminal can
        # validate its cross-links. Reuse qualification still requires the
        # baseline's later atomic finalization receipt.
        observed_bundle = ref_directory / "reference_bundle.json"
        publish_immutable_bytes(observed_bundle, json_bytes(bundle))
        observed_ref = observed_bundle.relative_to(root).as_posix()
        ref_package["observed_reference_ref"] = observed_ref
        ref_package["artifact_hashes"][observed_ref] = sha256_file(observed_bundle)
        reference_side = dict(sides[order[0]])
        reference_payload = payloads[order[0]]
    else:
        reference_trace = _json(bound_artifact_path(root, bundle["trace_manifest_ref"]))
        reference_roles = _json(bound_artifact_path(root, bundle["role_history_ref"]))["roles"]
        reference_side = {"comparison_identity": bundle["comparison_identity"],
            "artifacts": {name: "complete" for name in REQUIRED_OBSERVED_BINDINGS},
            "artifact_bindings": {name: {"ref": bundle[name + "_ref"],
                "sha256": sha256_file(bound_artifact_path(root, bundle[name + "_ref"]))}
                for name in REQUIRED_OBSERVED_BINDINGS},
            "terminal_complete": True, "prediction_status": "complete", "row_alignment_status": "aligned",
            "runtime_certificate_status": "accepted", "role_applicability_plan": comparison["role_applicability_plan"],
            "observed_role_event_kinds": sorted({event["access_kind"] for event in reference_roles}),
            "trace_field_availability": reference_trace["trace_fields"], "trace_status": "complete",
            "stochasticity_status": "unavailable"}
        reference_payload = torch.load(bound_artifact_path(root, bundle["prediction_manifest"]["artifact_locator"]),
                                       map_location="cpu", weights_only=True)
    reference_side.update(reference_bundle_id=bundle["reference_bundle_id"],
                          reference_bundle_sha256=reference_bundle_digest(bundle))
    for arm_id in (order[1:] if pair is not None else order):
        package, directory, _ = packages[arm_id]
        ref, candidate = reference_payload, payloads[arm_id]
        if any(not torch.equal(ref[key].view(-1), candidate[key].view(-1)) for key in ("source_idx", "target_eV")):
            raise ValueError("Scientific pair saved rows or targets differ")
        # Recompute endpoint errors from saved predictions, without inference.
        errors = {name: float((payload["prediction_eV"].view(-1).double() - payload["target_eV"].view(-1).double()).abs().mean())
                  for name, payload in (("reference", ref), ("candidate", candidate))}
        outcome = package["decision"]["outcome"]
        gain = errors["reference"] - errors["candidate"]
        if outcome in {"POSITIVE_UNDER_CONTRACT", "POSITIVE_BELOW_GATE"} and gain <= 0:
            raise ValueError("Positive scientific decision has no observed saved-prediction gain")
        if outcome == "POSITIVE_UNDER_CONTRACT":
            gate = comparison.get("minimum_material_gain_eV")
            if gate is None or gain < gate:
                raise ValueError("Positive scientific decision requires its frozen material-gain gate")
        analysis_path = directory / "paired_analysis.json"
        publish_immutable_bytes(analysis_path, json_bytes({"rows": int(ref["source_idx"].numel()),
            "reference_mae_eV": errors["reference"], "candidate_mae_eV": errors["candidate"],
            "candidate_minus_reference_eV": errors["candidate"] - errors["reference"],
            "training_stochasticity": "unavailable; one paired seed"}))
        side = copy.deepcopy(sides[arm_id])
        side["artifact_bindings"]["paired_analysis"] = {"ref": analysis_path.relative_to(root).as_posix(),
            "sha256": sha256_file(analysis_path)}
        side["artifacts"]["paired_analysis"] = "complete"
        readiness = assess_comparison_readiness(candidate_id=package["evidence"]["evidence_id"], candidate=side,
            reference_id=reference_id, reference=reference_side, experiment_purpose=purpose,
            intervention_group_id=comparison["intervention_group_id"],
            declared_intervention_fields=comparison["declared_intervention_fields"])
        validate_comparison_readiness(readiness, evidence_verifier=lambda p, s: verify_bound_artifact(root, p, s))
        if not readiness["strict_ready"]:
            raise ValueError("Saved pair scientific comparison is incomplete: " + str(readiness["blocker_codes"]))
        ready_path = directory / "comparison_readiness.json"
        publish_immutable_bytes(ready_path, json_bytes(readiness))
        pointer = ready_path.relative_to(root).as_posix()
        package["comparison_readiness_ref"] = pointer
        package["artifact_hashes"][pointer] = sha256_file(ready_path)
        package["evidence"]["artifacts"].append({"name": "comparison_readiness", "locator": pointer,
            "sha256": sha256_file(ready_path), "availability": "durable_local_verified"})
    for arm_id in order:
        package, directory, _ = packages[arm_id]
        for binding in sides[arm_id]["artifact_bindings"].values():
            package["artifact_hashes"][binding["ref"]] = binding["sha256"]
        path = directory / "terminal.json"
        publish_immutable_bytes(path, json_bytes(package))
        staged["locations"][arm_id]["terminal"] = path.relative_to(root).as_posix()
        staged["paths"][arm_id]["terminal"] = str(path)
    return staged


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
