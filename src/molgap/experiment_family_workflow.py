"""Shared producer/inspector hooks; existing Spec and RML remain authoritative.

No training, model construction, platform request or scientific decision occurs
here. A verified output is a mechanical result, never a replay-ready claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

from .experiment_family_artifacts import artifact_adapter
from .experiment_launch import _safe_local, read_launch_receipt, publish_immutable_bytes, build_launch_receipt
from .experiment_spec import ExperimentSpec, _digest, _identifier, _text
from .research_memory.paths import repo_local_path
from .research_memory.trace import (
    RMLTraceRecorder, file_digest, json_bytes, validate_canonical_trace,
)
from .screen_policy import canonical_fingerprint


OUTPUT_FORMAT = "molgap-family-output-v1"
ROLES = frozenset({"predictions", "selected_model", "resume", "trace", "contract"})
# Optional sixth role: files the frozen worker opened from the dataset (see run_screen_arm).
DATA_OPEN_FILE = "data_open_provenance.json"
DATA_OPEN_FORMAT = "molgap-data-open-record-v1"
EXPECTED = frozenset({"epochs", "optimizer_steps", "sample_presentations",
                      "development_rows", "source_idx_sha256", "target_sha256", "precision"})
EXPECTED_FAMILY = frozenset({"completed_passes", "next_batch", "optimizer_steps", "sample_presentations",
                            "development_rows", "source_idx_sha256", "target_sha256", "precision"})


def _json(path: Path) -> dict:
    return _json_bytes(path.read_bytes())


def _verify_canonical_output(root, spec, context, package, observed, output_dir):
    """Bind accepted evidence to this observed run without relying on artifact names."""
    from .research_memory.paths import bound_artifact_path
    expected = {"schema": "molgap-same-run-observation-v1", "spec_identity": spec.identity,
        "logical_run_id": context.logical_run_id, "platform_name": context.platform,
        "platform_run_reference": context.run_reference,
        "attempt_id": terminal_attempt_id(spec, context.arm_id, context.platform_version),
        "source_commit": context.source_commit, "source_package_sha256": context.source_archive_sha256}
    if package.get("same_run_observation") != expected:
        raise ValueError("Canonical qualification differs from the observed run context")
    artifacts = {(bound_artifact_path(root, item["locator"]).resolve(), item["sha256"])
                 for item in package["evidence"]["artifacts"]}
    for item in observed["artifacts"].values():
        path = (bound_artifact_path(root, item["locator"]) if "locator" in item
                else _artifact_path(Path(output_dir), item["path"]))
        if (path.resolve(), item["sha256"]) not in artifacts:
            raise ValueError("Canonical qualification differs from the observed output artifacts")


def same_run_comparison_scope(configuration, arm_id, reference_id, *, declared_fields=None):
    """Read the intervention group declared before training, never infer it from results."""
    from .comparison_readiness import _validate_intervention_scope
    shared = configuration.get("shared", {})
    arms = configuration.get("arms")
    if (type(shared) is not dict or type(arms) is not list
            or any(type(item) is not dict for item in arms)):
        raise ValueError("Same-run comparison requires frozen scientific arm declarations")
    selected = [item for item in arms if item.get("arm_id") == arm_id]
    references = [item for item in arms if item.get("arm_id") == reference_id]
    if len(selected) != 1 or len(references) != 1:
        raise ValueError("Same-run comparison requires exact frozen arm identities")
    candidate, reference = ({**shared, **selected[0]}, {**shared, **references[0]})
    purpose = candidate.get("purpose", "architecture_comparison")
    if purpose != reference.get("purpose", "architecture_comparison"):
        raise ValueError("Same-run arms declare different comparison purposes")
    # Frozen single-Addon screens used this name before the common taxonomy.
    if purpose == "single_module_ablation":
        purpose = "mechanism_comparison"
    allowed = {"architecture_comparison": ["architecture_config_identity"],
               "mechanism_comparison": ["architecture_config_identity"],
               "optimizer_comparison": ["optimizer_identity"],
               "initialization_comparison": ["initialization_identity"]}
    if purpose not in allowed:
        raise ValueError("Unsupported same-run scientific comparison purpose: " + str(purpose))
    question = configuration.get("question", {})
    group = question.get("changed_mechanism") if type(question) is dict else None
    if group is None:
        group = "architecture-config" if purpose == "architecture_comparison" else None
    scope = {"experiment_purpose": purpose, "intervention_group_id": group,
             "mechanism_id": group if purpose == "mechanism_comparison" else None,
             "declared_intervention_fields": allowed[purpose] if declared_fields is None else declared_fields}
    _validate_intervention_scope(**scope, require_nonempty=True)
    if set(scope["declared_intervention_fields"]) - set(allowed[purpose]):
        raise ValueError("Same-run declaration exceeds supported scientific changes")
    return scope


def _frozen_same_run_scopes(resolved):
    """Use the archived human input; the mutable checkout is not scope authority."""
    declaration = resolved["spec"].to_dict()
    pair = declaration["prospective"]["same_run_replay"]
    reference = pair["reference_arm_id"]
    binding = resolved["target_identities"][reference]
    if binding.package_dir is None:
        raise ValueError("Same-run strict scope requires the frozen source package")
    manifest = _json(binding.package_dir / "package_manifest.json")
    source = manifest.get("configuration_source")
    if source is None:
        # Original hand-authored packages had only architecture comparison.
        return {arm["arm_id"]: {"experiment_purpose": "architecture_comparison",
            "intervention_group_id": "architecture-config", "mechanism_id": None,
            "declared_intervention_fields": ["architecture_config_identity"]}
            for arm in declaration["arms"] if arm["arm_id"] != reference}
    if source.get("compiler") != "molgap-scientific-experiment-v1":
        raise ValueError("Unknown frozen scientific comparison compiler")
    if source.get("path") not in manifest["relative_allowlist"]:
        raise ValueError("Scientific comparison source is absent from the frozen package")
    configuration = binding._pinned_json(source["path"], source["sha256"], label="scientific scope")
    if configuration.get("format") != source["compiler"]:
        raise ValueError("Frozen scientific comparison format differs")
    return {arm["arm_id"]: same_run_comparison_scope(configuration, arm["arm_id"], reference)
            for arm in declaration["arms"] if arm["arm_id"] != reference}


def validate_initialization_peer(arm, reference, contract, reference_contract):
    """A declared warm-start comparison changes only its frozen initial state."""
    for key in ("family", "base", "addons", "addon_semantics", "data"):
        if arm[key] != reference[key]:
            raise ValueError("Initialization peers differ outside initial state: " + key)
    for key in ("sampler", "transform", "objective", "overrides"):
        if arm["training"][key] != reference["training"][key]:
            raise ValueError("Initialization peers differ outside initial state: " + key)
    if ({k: v for k, v in arm["initialization"].items() if k != "state_sha256"}
            != {k: v for k, v in reference["initialization"].items() if k != "state_sha256"}):
        raise ValueError("Initialization comparison must retain initialization kind and seed")
    if arm["initialization"]["state_sha256"] == reference["initialization"]["state_sha256"]:
        raise ValueError("Initialization comparison requires distinct initial states")
    fixed = []
    for side, recipe in ((arm, contract), (reference, reference_contract)):
        if recipe.get("initialization_sha256") != side["initialization"]["state_sha256"]:
            raise ValueError("Initialization contract differs from frozen Spec")
        fixed.append({k: v for k, v in recipe.items()
                      if k not in {"initialization_sha256", "initial_development_check"}})
    if fixed[0] != fixed[1]:
        raise ValueError("Initialization peers differ outside initial state: recipe")


def resolve_strict_target_transforms(repo_root, resolved, strict_inputs=None):
    """Resolve retained target proof without reading training graphs or models."""
    from .comparison_readiness import target_transform_asset_digest, validate_target_transform_asset
    from .family_training_runner import read_recipe
    from .research_memory.paths import artifact_locator, bound_artifact_path
    root = Path(repo_root).resolve()
    declaration = resolved["spec"].to_dict()
    requirements, blockers = {}, []
    if strict_inputs is None:
        strict_inputs = {}
    if type(strict_inputs) is not dict or set(strict_inputs) - set(resolved["outputs"]):
        raise ValueError("Strict inputs must name only frozen run arms")
    # Resolve every transform before publishing any qualification metadata.
    for arm_id, item in sorted(resolved["outputs"].items()):
        try:
            arm = next(arm for arm in declaration["arms"] if arm["arm_id"] == arm_id)
            _, recipe = read_recipe(Path(item["output_dir"]) / "training_contract.json", _frozen_read=True)
            role_by_name = {role["role"]: role for role in arm["data"]["roles"]}
            target_binding = resolved["target_identities"][arm_id]
            target_plan = target_binding._pinned_json(target_binding.plan_path, target_binding.plan_sha256)
            entry = next(entry for entry in target_plan["arms"] if entry["arm_id"] == arm_id)
            pin = (entry["target_manifest"] if target_plan["format"] == "molgap-family-same-run-acceptance-plan-v1"
                   else entry["reference_artifacts"]["target_manifest"])
            targets = target_binding._pinned_json(pin["path"], pin["sha256"])
            supplied = strict_inputs.get(arm_id, {})
            if type(supplied) is not dict:
                raise ValueError("Strict arm inputs must be an object")
            asset = supplied.get("target_transform_asset")
            expected_transform = {"format": "molgap-target-transform-asset-v1",
                "asset_id": arm["training"]["transform"]["sha256"], "target_identity": arm["data"]["target"],
                "mean": recipe.target_mean_eV, "std": recipe.target_std_eV, "ddof": 1,
                "variance_convention": "sample", "source_row_manifest_sha256": role_by_name["train"]["membership_sha256"]}
            if asset is not None:
                if type(asset) is not dict or set(asset) != {"path", "sha256"}:
                    raise ValueError("Target transform evidence requires path and sha256")
                _digest(asset["sha256"], "target transform evidence digest")
                path = bound_artifact_path(root, asset["path"])
                _safe_local(path)
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                    raise ValueError("Target transform evidence file hash mismatch")
                transform = validate_target_transform_asset(_json_bytes(raw))
                if any(transform.get(key) != value for key, value in expected_transform.items()):
                    raise ValueError("Target transform evidence differs from frozen normalization/train identity")
            else:
                if "training_float32_target_sha256" not in targets:
                    raise ValueError("Frozen training_float32_target_sha256 is missing; supply a hash-bound "
                        "target_transform_asset from retained ordered training-label evidence")
                _digest(targets.get("training_float32_target_sha256"), "training_float32_target_sha256")
                transform = {**expected_transform, "target_sha256": targets["training_float32_target_sha256"]}
                transform["asset_sha256"] = target_transform_asset_digest(transform)
                validate_target_transform_asset(transform)
            if (not math.isfinite(transform["mean"]) or not math.isfinite(transform["std"])
                    or transform["std"] <= 0):
                raise ValueError("Target transform normalization must be finite with positive std")
            if "training_float32_target_sha256" in targets:
                _digest(targets["training_float32_target_sha256"], "training_float32_target_sha256")
                if transform["target_sha256"] != targets["training_float32_target_sha256"]:
                    raise ValueError("Target transform evidence differs from frozen training targets")
            requirements[arm_id] = {"transform": transform, "targets": targets}
            if asset is not None:
                requirements[arm_id]["supplemental_binding"] = {
                    "ref": artifact_locator(root, path), "sha256": asset["sha256"]}
        except (ValueError, TypeError, KeyError, OSError, StopIteration) as exc:
            blockers.append(arm_id + ": " + str(exc))
            requirements[arm_id] = {"required_evidence": ["target_transform_asset"], "blockers": [str(exc)]}
    return {"status": "BLOCKED" if blockers else "VERIFIED", "blockers": blockers,
            "arms": requirements}


def qualify_same_run_terminal_packages(repo_root, resolved, locations, native_reports, *, reports, strict_inputs=None,
                                       _target_transforms=None, _paired_cache=None):
    """Translate verified family artifacts into the existing V5 comparison forms.

    Publishes new metadata and terminal inputs. Original terminal packages and
    scientific decisions are retained byte-for-byte; finalization is separate.
    """
    import copy
    import numpy as np
    from .comparison_readiness import (CAUSAL_REQUIRED_ROLE_KINDS, ROLE_EVENT_KINDS,
        assess_comparison_readiness, reference_bundle_digest,
        validate_comparison_readiness, validate_reference_bundle)
    from .experiment_attribution import _load_arm_artifacts, _compare_paired_errors
    from .research_memory.paths import artifact_locator, bound_artifact_path, verify_bound_artifact
    from .research_memory.roles import validate_observed_role_truth
    from .family_training_runner import read_recipe, comparison_objective_identity
    from .research_memory.finalize import (current_finalized_directory, load_terminal_input,
        verified_receipt, prospective_bytes, _qualification_amendment, _bindings)
    root = Path(repo_root).resolve()
    declaration = resolved["spec"].to_dict()
    pair = declaration["prospective"].get("same_run_replay")
    if pair is None:
        raise ValueError("Automatic strict comparison requires the frozen same-run declaration")
    reference_id = pair["reference_arm_id"]
    if any(native_reports[arm]["status"] != "VERIFIED" for arm in locations):
        raise ValueError("Every same-run arm requires independently reverified native proof")
    packages, finalized = {}, {}
    for arm_id, loc in locations.items():
        current = current_finalized_directory(root, loc["trajectory"])
        terminal_path = repo_local_path(root, loc["terminal"])
        incoming = _json(terminal_path) if terminal_path.is_file() else None
        if current.is_dir():
            receipt = verified_receipt(current)
            accepted = load_terminal_input(current)
            if incoming is not None:
                if incoming.get("format") == "molgap-rml-terminal-input-v2":
                    incoming = load_terminal_input(terminal_path.parent)
                # Preserve qualification bindings absent from an older staged
                # input, while the existing owner rejects altered science.
                candidate = copy.deepcopy(incoming)
                candidate["artifact_hashes"] = {**accepted["artifact_hashes"], **candidate["artifact_hashes"]}
                candidate["evidence"]["artifacts"] = accepted["evidence"]["artifacts"] + [
                    item for item in candidate["evidence"]["artifacts"]
                    if item not in accepted["evidence"]["artifacts"]]
                trace = bound_artifact_path(root, candidate["trace_manifest"]["trace_artifact_ref"])
                _qualification_amendment(root, current, prospective_bytes(current, receipt),
                    candidate, trace.read_bytes())
                accepted.update({key: value for key, value in candidate.items() if key not in accepted})
                accepted["artifact_hashes"] = candidate["artifact_hashes"]
                accepted["evidence"]["artifacts"] = candidate["evidence"]["artifacts"]
            packages[arm_id], finalized[arm_id] = accepted, current
        elif incoming is not None:
            packages[arm_id] = incoming
        else:
            raise ValueError("Strict qualification requires staged or canonical terminal evidence")
    if len(finalized) == len(locations) and all(package["trace_manifest"]["backtest_eligibility"]["eligible"]
            for package in packages.values()):
        comparisons, reused = {}, {}
        for arm_id, package in packages.items():
            ctx = resolved["contexts"][arm_id]
            _verify_canonical_output(root, resolved["spec"], ctx, package,
                reports[arm_id]["observed"], resolved["outputs"][arm_id]["output_dir"])
            _bindings(root, package["artifact_hashes"])
            if arm_id != reference_id:
                pointer = package.get("comparison_readiness_ref")
                if pointer not in package["artifact_hashes"]:
                    raise ValueError("Canonical qualification has no bound strict comparison")
                comparison = _json(repo_local_path(root, pointer))
                validate_comparison_readiness(comparison,
                    evidence_verifier=lambda ref, sha: verify_bound_artifact(root, ref, sha))
                if (not comparison["strict_ready"] or comparison["candidate_id"] != package["evidence"]["evidence_id"]
                        or comparison["reference_id"] != packages[reference_id]["evidence"]["evidence_id"]):
                    raise ValueError("Canonical strict comparison differs from the accepted pair")
                comparisons[arm_id] = comparison
            reused[arm_id] = {**locations[arm_id],
                "terminal": (finalized[arm_id] / "terminal_input.json").relative_to(root).as_posix(),
                "trace": package["trace_manifest"]["trace_artifact_ref"]}
        return {"locations": reused, "comparisons": comparisons, "canonical_reused": True,
            "reference_bundle_ref": None, "directory": None,
            "qualification_adapter_sha256": file_digest(Path(__file__))}
    target_report = (_target_transforms if _target_transforms is not None else
                     resolve_strict_target_transforms(root, resolved, strict_inputs))
    if target_report["status"] != "VERIFIED":
        return {**target_report, "executed": False}
    scopes = _frozen_same_run_scopes(resolved)
    retained_recipes = {arm_id: read_recipe(Path(item["output_dir"]) / "training_contract.json", _frozen_read=True)
                        for arm_id, item in resolved["outputs"].items()}
    by_arm = {arm["arm_id"]: arm for arm in declaration["arms"]}
    for arm_id, scope in scopes.items():
        if scope["experiment_purpose"] == "initialization_comparison":
            validate_initialization_peer(by_arm[arm_id], by_arm[reference_id],
                retained_recipes[arm_id][0], retained_recipes[reference_id][0])
    transforms = {arm: row["transform"] for arm, row in target_report["arms"].items()}
    target_manifests = {arm: row["targets"] for arm, row in target_report["arms"].items()}
    appended = {arm: row["supplemental_binding"] for arm, row in target_report["arms"].items()
                if "supplemental_binding" in row}
    binding_blockers, binding_requirements = [], {}
    for arm_id, asset in appended.items():
        try:
            verify_bound_artifact(root, asset["ref"], asset["sha256"])
            accepted_hash = packages[arm_id]["artifact_hashes"].get(asset["ref"])
            if accepted_hash is not None and accepted_hash != asset["sha256"]:
                raise ValueError("Target transform evidence cannot replace an accepted binding")
        except (ValueError, OSError) as exc:
            binding_blockers.append(arm_id + ": " + str(exc))
            binding_requirements[arm_id] = {"required_evidence": ["target_transform_asset"], "blockers": [str(exc)]}
    if binding_blockers:
        return {"status": "BLOCKED", "arms": binding_requirements, "blockers": binding_blockers, "executed": False}
    adapter_sha256 = file_digest(Path(__file__))
    identity = canonical_fingerprint({"packages": packages, "native": native_reports,
        "target_transforms": transforms, "target_transform_evidence": appended,
        "qualification_adapter_sha256": adapter_sha256})
    directory = repo_local_path(root, Path(locations[reference_id]["trajectory"]).parent.parent
        / "strict_qualification" / identity[:20])

    def publish(arm, name, value):
        path = directory / arm / name
        publish_immutable_bytes(path, json_bytes(value))
        return path.relative_to(root).as_posix()

    def binding(pointer):
        path = bound_artifact_path(root, pointer)
        return {"ref": pointer, "sha256": file_digest(path)}

    loaded, sides, bundles = {}, {}, {}
    for arm_id, item in sorted(resolved["outputs"].items()):
        arm = next(arm for arm in declaration["arms"] if arm["arm_id"] == arm_id)
        package, ctx = packages[arm_id], resolved["contexts"][arm_id]
        for pointer, sha in package["artifact_hashes"].items():
            verify_bound_artifact(root, pointer, sha)
        validate_observed_role_truth([package["role_use"]], package["roles"])
        if not any(event["measurement"]["device_hours"]["status"] == "measured"
                   for event in package["costs"]):
            raise ValueError("Strict family comparison requires observed native device cost")
        data = _load_arm_artifacts(Path(item["output_dir"]), context=ctx, expected=item["expected"],
            target_identity=resolved.get("target_identities", {}).get(arm_id), _inspection=reports[arm_id])
        loaded[arm_id] = data
        contract, recipe = retained_recipes[arm_id]
        role_by_name = {role["role"]: role for role in arm["data"]["roles"]}
        targets, transform = target_manifests[arm_id], transforms[arm_id]
        transform_path = repo_local_path(root, Path(locations[arm_id]["trajectory"]).parent.parent
            / "comparison_assets" / transform["asset_sha256"] / "target_transform.json")
        publish_immutable_bytes(transform_path, json_bytes(transform))
        transform_ref = transform_path.relative_to(root).as_posix()
        policy = contract["execution_policy"]
        provenance = _json(Path(item["output_dir"]) / "runtime_provenance.json")
        comparison_identity = {
            "benchmark_identity": arm["data"]["dataset"]["name"], "dataset_identity": arm["data"]["dataset"]["sha256"],
            "data_role_identity": canonical_fingerprint({"roles": arm["data"]["roles"]}),
            "row_membership_identity": canonical_fingerprint({k: v["membership_sha256"] for k, v in role_by_name.items()}),
            "row_order_identity": canonical_fingerprint({k: v["row_order_sha256"] for k, v in role_by_name.items()}),
            "feature_identity": arm["data"]["feature_sha256"], "target_identity": arm["data"]["target"], "seed": recipe.seed,
            "precision": policy["precision"], "tf32_enabled": policy["tf32"], "deterministic_algorithms": True,
            "physical_batch_per_device": recipe.batch_size, "gradient_accumulation_steps": 1,
            "tail_batch_policy": "drop_last" if recipe.drop_last else "keep_last",
            "optimizer_identity": canonical_fingerprint({"recipe": {k: contract[k] for k in ("learning_rate", "weight_decay", "gradient_clip")},
                "policy": {k: policy[k] for k in ("adamw_betas", "adamw_epsilon")}}),
            "optimizer_mode": policy["optimizer"], "optimizer_fused": policy["optimizer"] == "fused-adamw",
            "schedule_identity": canonical_fingerprint(contract["schedule"]), "loss_identity": comparison_objective_identity(arm, recipe),
            "target_transform_identity": arm["training"]["transform"]["sha256"], "target_transform_asset_sha256": transform["asset_sha256"],
            "sample_presentations": data["manifest"]["progress"]["sample_presentations"],
            "optimizer_steps": data["manifest"]["progress"]["optimizer_steps"], "checkpoint_selection_identity": recipe.selection,
            "evaluation_role_identity": contract["development_role_identity"], "selection_role_identity": contract["development_role_identity"],
            "architecture_config_identity": canonical_fingerprint({k: arm[k] for k in ("base", "addons")}),
            "runtime_certificate_scope": provenance["runtime_fingerprint"], "weight_semantics": data["selection_weights"],
            "ema_enabled": recipe.ema_decay is not None, "ema_decay": recipe.ema_decay,
            "ema_update_frequency": 1 if recipe.ema_decay is not None else None,
            "evaluation_weight_source": data["selection_weights"]}
        if any(scope["experiment_purpose"] == "initialization_comparison" for scope in scopes.values()):
            if recipe.initialization_sha256 != arm["initialization"]["state_sha256"]:
                raise ValueError("Observed recipe initialization differs from frozen Spec")
            comparison_identity["initialization_identity"] = canonical_fingerprint(arm["initialization"])
        certificates = [entry["local_path"] for entry in native_reports[arm_id]["dependencies"]
                        if entry["path"] == "runtime_certificate.json"]
        if len(certificates) != 1:
            raise ValueError("Native qualification requires one exact retained certificate")
        bindings = {"checkpoint": binding(artifact_locator(root, Path(item["output_dir"]) / "selected_model.pt")),
            "runtime_certificate": binding(artifact_locator(root, Path(certificates[0]))),
            "prediction_manifest": binding(publish(arm_id, "prediction_manifest.json", {"prediction_sha256": data["hashes"]["development_predictions_sha256"],
                "source_idx_sha256": item["expected"]["source_idx_sha256"], "target_sha256": item["expected"]["target_sha256"],
                "ordering_semantics": "frozen source_idx order", "evaluation_role_identity": contract["development_role_identity"],
                "row_count": len(data["source_idx"]), "unique_source_idx": int(np.unique(data["source_idx"]).size)})),
            "row_manifest": binding(publish(arm_id, "row_manifest.json", {"roles": arm["data"]["roles"]})),
            "target_manifest": binding(publish(arm_id, "target_manifest.json", targets)),
            "trace_manifest": binding(publish(arm_id, "bound_trace_manifest.json", package["trace_manifest"])),
            "role_history": binding(publish(arm_id, "role_history.json", {"roles": package["roles"], "role_use": package["role_use"]})),
            "cost_records": binding(publish(arm_id, "cost_records.json", {"costs": package["costs"]})),
            "target_transform_asset": binding(transform_ref), "acceptance": binding(package["acceptance_ref"]),
            "decision": binding(package["decision"]["decision_ref"]),
            "source_config": binding(artifact_locator(root, Path(item["output_dir"]) / "training_contract.json"))}
        bundle = {"format": "molgap-reference-bundle-v1", "reference_bundle_id": "ref-" + package["evidence"]["evidence_id"] + "-" + identity[:20],
            "reference_id": package["evidence"]["evidence_id"], "contract_ref": bindings["source_config"]["ref"],
            "architecture_config_identity": comparison_identity["architecture_config_identity"], "source_commit_or_archive": ctx.source_archive_sha256,
            "checkpoint_identity": bindings["checkpoint"]["sha256"], "prediction_manifest": _json(root / bindings["prediction_manifest"]["ref"]),
            "comparison_identity": comparison_identity, "stochasticity": {"row_bootstrap_uncertainty": {"status": "not_requested"},
                "training_stochasticity": {"status": "unavailable", "estimation_method": "unavailable", "source": "unavailable",
                    "same_contract_repeat_ids": [], "n_repeats": 0, "stochasticity_floor_eV": None}}}
        for name in ("runtime_certificate", "row_manifest", "target_manifest", "trace_manifest", "role_history", "target_transform_asset", "cost_records", "acceptance", "decision"):
            bundle[name + "_ref"] = bindings[name]["ref"]
        validate_reference_bundle(bundle)
        bundles[arm_id] = bundle
        sides[arm_id] = {"comparison_identity": comparison_identity, "artifacts": {k: "complete" for k in bindings if k != "source_config"},
            "prediction_status": "complete", "row_alignment_status": "aligned", "runtime_certificate_status": "accepted",
            "role_status": "complete", "role_applicability_plan": {kind: "applicable" if kind in CAUSAL_REQUIRED_ROLE_KINDS else "not_applicable" for kind in ROLE_EVENT_KINDS},
            "observed_role_event_kinds": sorted({event["access_kind"] for event in package["roles"]}),
            "trace_status": "complete", "trace_field_availability": package["trace_manifest"]["trace_fields"],
            "stochasticity_status": "unavailable", "terminal_complete": data["manifest"]["progress"]["status"] == "complete", "artifact_bindings": bindings}
    reference_bundle = bundles[reference_id]
    bundle_ref = publish(reference_id, "reference_bundle.json", reference_bundle)
    sides[reference_id].update(reference_bundle_id=reference_bundle["reference_bundle_id"], reference_bundle_sha256=reference_bundle_digest(reference_bundle))
    comparisons, qualified = {}, {}
    for arm_id, package in packages.items():
        package = copy.deepcopy(package)
        if arm_id in appended:
            asset = appended[arm_id]
            verify_bound_artifact(root, asset["ref"], asset["sha256"])
            if asset["ref"] in package["artifact_hashes"] and package["artifact_hashes"][asset["ref"]] != asset["sha256"]:
                raise ValueError("Target transform evidence cannot replace an accepted binding")
            package["artifact_hashes"][asset["ref"]] = asset["sha256"]
            artifact = {"name": "target_transform_evidence", "locator": asset["ref"], "sha256": asset["sha256"],
                "availability": "machine_local_verified" if asset["ref"].startswith("external://") else "durable_local_verified"}
            if artifact not in package["evidence"]["artifacts"]:
                package["evidence"]["artifacts"].append(artifact)
        if arm_id != reference_id:
            left, right = loaded[reference_id], loaded[arm_id]
            if not np.array_equal(left["source_idx"], right["source_idx"]) or not np.array_equal(left["target_eV"], right["target_eV"]):
                raise ValueError("Strict comparison predictions/targets are not aligned")
            cached = (_paired_cache or {}).get(arm_id, {})
            paired = cached.get("paired", {})
            if not (cached.get("reference_id") == reference_id
                    and cached.get("hashes") == (left["hashes"], right["hashes"])
                    and paired.get("n_bootstrap") == 2000 and paired.get("seed") == 42
                    and paired.get("threshold_eV") == 0.003
                    and paired.get("noninferiority_margin_eV") == 0.0005):
                paired = _compare_paired_errors(np.abs(left["prediction_eV"] - left["target_eV"]),
                    np.abs(right["prediction_eV"] - right["target_eV"]), n_bootstrap=2000, seed=42)
            paired_ref = publish(arm_id, "paired_analysis.json", paired)
            sides[arm_id]["artifacts"]["paired_analysis"] = "complete"
            sides[arm_id]["artifact_bindings"]["paired_analysis"] = binding(paired_ref)
            comparison = assess_comparison_readiness(candidate_id=package["evidence"]["evidence_id"], candidate=sides[arm_id],
                reference_id=packages[reference_id]["evidence"]["evidence_id"], reference=sides[reference_id],
                **scopes[arm_id])
            validate_comparison_readiness(comparison, evidence_verifier=lambda pointer, sha: verify_bound_artifact(root, pointer, sha))
            comparisons[arm_id] = comparison
            comparison_ref = publish(arm_id, "comparison_readiness.json", comparison)
            if not comparison["strict_ready"]:
                raise ValueError("Strict comparison remains blocked: " + repr(comparison["blocker_codes"]))
            package["comparison_readiness_ref"] = comparison_ref
            package["artifact_hashes"][comparison_ref] = file_digest(root / comparison_ref)
        package["artifact_hashes"][bundle_ref] = file_digest(root / bundle_ref)
        package["trace_manifest"]["backtest_eligibility"] = {"eligible": True, "exclusion_reasons": []}
        if arm_id in finalized:
            previous = (finalized[arm_id] / "finalization.json").relative_to(root).as_posix()
            package["supersedes_finalization_ref"] = previous
            package["artifact_hashes"][previous] = file_digest(root / previous)
        terminal_ref = publish(arm_id, "qualified_terminal.json", package)
        qualified[arm_id] = {**locations[arm_id], "terminal": terminal_ref,
            "trace": package["trace_manifest"]["trace_artifact_ref"]}
    return {"locations": qualified, "comparisons": comparisons, "reference_bundle_ref": bundle_ref,
        "directory": str(directory), "qualification_adapter_sha256": adapter_sha256}


def _json_bytes(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("Expected JSON object")
    return value


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}: expected positive integer")


def _artifact_path(root: Path, value: str) -> Path:
    if type(value) is not str or "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError("Expected relative POSIX artifact path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Invalid artifact path")
    _safe_local((root / value).absolute())
    path = repo_local_path(root, value)
    if not path.is_file():
        raise ValueError(f"Missing artifact: {value}")
    return path


@dataclass(frozen=True)
class RunContext:
    experiment_id: str
    logical_run_id: str
    arm_id: str
    arm_identity: str
    spec_identity: str
    family_name: str
    family_version: str
    source_commit: str
    source_archive_sha256: str
    package_identity: str
    training_recipe_sha256: str
    platform: str
    account: str
    run_reference: str
    platform_version: str | None

    def __post_init__(self):
        for field in ("experiment_id", "logical_run_id", "arm_id"):
            _identifier(getattr(self, field), field)
        for field in ("arm_identity", "spec_identity", "source_archive_sha256", "package_identity", "training_recipe_sha256"):
            _digest(getattr(self, field), field)
        for field in ("family_name", "family_version", "platform", "account", "run_reference"):
            _text(getattr(self, field), field)
        if self.platform_version is not None:
            _text(self.platform_version, "platform_version")
        import re
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.source_commit):
            raise ValueError("Expected full source commit")
        if any(c in self.run_reference for c in "?@#") or self.account != self.run_reference.split("/")[0]:
            raise ValueError("Run reference/account mismatch or credential-bearing reference")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_training(cls, spec: ExperimentSpec, package_dir: Path, *,
                     expected_package_identity: str, arm_id: str,
                     account: str, run_reference: str, _verification=None) -> RunContext:
        """Producer context before a submitter returns a version.

        The account/slug are frozen declarations, not a platform observation.
        Unknown version remains null; acceptance binds the real receipt later.
        """
        receipt = build_launch_receipt(spec, Path(package_dir), expected_package_identity=expected_package_identity,
            **({"_verification": _verification} if _verification is not None else {}))
        declaration = spec.to_dict()
        arm = next((a for a in declaration["arms"] if a["arm_id"] == arm_id), None)
        if arm is None:
            raise ValueError("Unknown producer arm")
        binding = receipt["binding"]
        return cls(declaration["experiment_id"], declaration["logical_run_id"], arm_id,
                   canonical_fingerprint(arm), spec.identity, arm["family"]["name"],
                   arm["family"]["version"], binding["source_commit"], binding["source_archive_sha256"],
                   binding["package_identity"], arm["training"]["recipe"]["sha256"], declaration["platform"]["name"], account,
                   run_reference, None)
    @classmethod
    def from_launch(cls, spec: ExperimentSpec, receipt_path: Path, package_dir: Path,
                    *, expected_package_identity: str, arm_id: str, _verification=None) -> RunContext:
        receipt = read_launch_receipt(Path(receipt_path).absolute(), spec, Path(package_dir),
                                      expected_package_identity=expected_package_identity,
                                      **({"_verification": _verification} if _verification is not None else {}))
        if receipt["submission_state"] not in {"ACCEPTED", "RECONCILED_EXISTING"}:
            raise ValueError("RunContext requires a reconciled observed launch")
        runs = receipt["physical_runs"]["value"]
        if runs is None:
            raise ValueError("Missing physical arm mapping")
        run = next((item for item in runs if arm_id in item["arm_ids"]), None)
        if run is None or run["platform_version"]["value"] is None:
            raise ValueError("Missing exact physical run/version")
        declaration = spec.to_dict()
        arm = next(item for item in declaration["arms"] if item["arm_id"] == arm_id)
        binding = receipt["binding"]
        reference = run["canonical_reference"]
        return cls(declaration["experiment_id"], declaration["logical_run_id"], arm_id,
                   canonical_fingerprint(arm), spec.identity, arm["family"]["name"],
                   arm["family"]["version"], binding["source_commit"],
                   binding["source_archive_sha256"], binding["package_identity"], arm["training"]["recipe"]["sha256"],
                   declaration["platform"]["name"], reference.split("/")[0], reference,
                   run["platform_version"]["value"])


def _check_context(observed: dict, expected: RunContext) -> None:
    if type(observed) is not dict or set(observed) != set(expected.to_dict()):
        raise ValueError("Output context missing identity fields")
    values = expected.to_dict()
    for key, value in observed.items():
        if key == "platform_version" and value is None:
            continue  # Missing producer observation is retained, never backfilled.
        if value != values[key]:
            raise ValueError("Output context/source/arm mismatch: " + key)


class StageRecorder:
    """Use one trace per phase; counters are acknowledged processed work.

    Explicit stage/run identities prevent pretraining and downstream metrics or
    optimizer resets from being silently combined into one curve.
    """
    def __init__(self, path: Path, context: RunContext, stage_id: str,
                 metric_semantics: dict, *, trajectory_id: str | None = None,
                 device_time_semantics: str | None = None):
        _identifier(stage_id, "stage_id")
        _safe_local(Path(path).absolute())
        self.context = context
        _publish(Path(path).with_suffix(Path(path).suffix + ".context.json"),
                 {"context": context.to_dict(), "stage_id": stage_id})
        self.recorder = RMLTraceRecorder(path,
            trajectory_id=trajectory_id or context.experiment_id + "-" + context.arm_id,
            run_id=context.logical_run_id + ":" + context.arm_id + ":" + stage_id,
            metric_semantics=metric_semantics, device_time_semantics=device_time_semantics)
        # The companion is immutable and disallows reuse with another source/arm.

    def observe(self, *, optimizer_step: int, sample_presentations: int,
                epoch_or_pass: int | float, **fields) -> None:
        _positive(optimizer_step, "optimizer_step")
        _positive(sample_presentations, "sample_presentations")
        if isinstance(epoch_or_pass, bool) or not isinstance(epoch_or_pass, (int, float)) or not math.isfinite(epoch_or_pass) or epoch_or_pass < 0:
            raise ValueError("Invalid epoch_or_pass")
        self.recorder.append_observation(optimizer_step=optimizer_step,
            sample_presentations=sample_presentations, epoch_or_pass=epoch_or_pass, **fields)


def _publish(path: Path, value: dict) -> Path:
    """Immutable metadata publication, with identical retries allowed."""
    _safe_local(path.absolute())
    publish_immutable_bytes(path.absolute(), json_bytes(value))
    return path


def write_output_manifest(output_dir: Path, context: RunContext, *, adapter: str,
                          artifacts: dict[str, str], progress: dict,
                          runtime: dict, costs: list[dict]) -> Path:
    root = Path(output_dir).absolute()
    _safe_local(root)
    profile = artifact_adapter(adapter, (context.family_name, context.family_version))
    roles = ROLES
    if profile.protocol == "family-session" and "predictions" not in artifacts:
        roles = ROLES - {"predictions"}
    if profile.protocol == "family-session" and "trace_context" in artifacts:
        roles = roles | {"trace_context"}
        if artifacts["trace_context"] != artifacts["trace"] + ".context.json":
            raise ValueError("Trace companion must bind the declared trace")
    if profile.protocol == "family-session" and "data_open" in artifacts:
        roles = roles | {"data_open"}
    if set(artifacts) != roles:
        raise ValueError("Output requires exactly predictions/selected_model/resume/trace/contract")
    entries = {role: {"path": name, "sha256": file_digest(_artifact_path(root, name))}
               for role, name in artifacts.items()}
    if len({item["path"].casefold() for item in entries.values()}) != len(roles):
        raise ValueError("Artifact roles must bind distinct files")
    return _publish(root / "output_manifest.json", {
        "format": OUTPUT_FORMAT, "context": context.to_dict(), "adapter": adapter,
        "artifacts": entries, "progress": progress, "runtime": runtime, "costs": costs,
    })


def write_selected_state(path: Path, context: RunContext, *, model_state: dict,
                         epoch: int, optimizer_step: int, weights: str,
                         development_mae_eV: float) -> Path:
    """Producer hook; the owning trainer selects the endpoint and its weights."""
    from .training_reproducibility import atomic_torch_save, assert_finite_state_dict
    _safe_local(Path(path).absolute())
    _positive(epoch, "epoch")
    _positive(optimizer_step, "optimizer_step")
    if weights not in {"live", "ema"} or isinstance(development_mae_eV, bool) or not isinstance(development_mae_eV, (float, int)) or not math.isfinite(development_mae_eV) or development_mae_eV < 0:
        raise ValueError("Invalid selected weight/metric semantics")
    assert_finite_state_dict(model_state, label="selected_model")
    atomic_torch_save(Path(path), {"context": context.to_dict(), "model": model_state,
        "epoch": epoch, "optimizer_step": optimizer_step, "weights": weights,
        "development_mae_eV": development_mae_eV})
    return Path(path)


def write_resume_state(path: Path, context: RunContext, *, adapter: str, model_state: dict,
                       optimizer_state: dict, rng_state: dict, cursor: dict,
                       optimizer_step: int, sample_presentations: int,
                       scheduler_state: dict | None = None, ema_state: dict | None = None,
                       owner_state: dict | None = None) -> Path:
    """Atomic output translation; resume loading stays with the owning trainer."""
    from .training_reproducibility import atomic_torch_save
    _safe_local(Path(path).absolute())
    profile = artifact_adapter(adapter, (context.family_name, context.family_version))
    _positive(optimizer_step, "optimizer_step")
    _positive(sample_presentations, "sample_presentations")
    state = {"context": context.to_dict(), "model": model_state, "optimizer": optimizer_state,
             "rng_state": tensor_safe_rng_state(rng_state), "cursor": cursor, "optimizer_step": optimizer_step,
             "sample_presentations": sample_presentations}
    if scheduler_state is not None:
        state["scheduler"] = scheduler_state
    if ema_state is not None:
        state["ema"] = ema_state
    if owner_state is not None:
        if type(owner_state) is not dict or not owner_state:
            raise ValueError("Owner checkpoint state must be a nonempty mapping")
        state["owner_state"] = owner_state
    if any(not isinstance(state.get(key), dict) or not state[key] for key in profile.resume_keys):
        raise ValueError("Incomplete checkpoint/resume state")
    _finite_tree(state, "resume")
    atomic_torch_save(Path(path), state)
    return Path(path)


def tensor_safe_rng_state(state: dict) -> dict:
    """Translate the existing RNG helper's NumPy array to safe primitive data."""
    import copy
    result = copy.deepcopy(state)
    numpy_state = result.get("numpy")
    if isinstance(numpy_state, tuple) and len(numpy_state) == 5 and hasattr(numpy_state[1], "tolist"):
        result["numpy"] = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    _validate_rng_state(result)
    return result


def _validate_rng_state(state):
    import torch
    if not isinstance(state, dict) or not {"python", "numpy", "torch", "cuda"} <= set(state):
        raise ValueError("Incomplete Python/NumPy/Torch/CUDA RNG resume state")
    for value in (state["torch"], *state["cuda"]):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not len(value):
            raise ValueError("Invalid torch/CUDA RNG state")
    python = state["python"]
    numpy = state["numpy"]
    if not isinstance(python, tuple) or len(python) != 3 or not isinstance(python[1], tuple):
        raise ValueError("Invalid Python RNG state")
    if not isinstance(numpy, tuple) or len(numpy) != 5 or not isinstance(numpy[1], list) or not numpy[1] or any(type(n) is not int or not 0 <= n < 2**32 for n in numpy[1]):
        raise ValueError("Invalid tensor-safe NumPy RNG state")


def _finite_tree(value, label):
    from .training_reproducibility import finite_validation_observation
    with finite_validation_observation():
        return _finite_tree_impl(value, label)


def _finite_tree_impl(value, label):
    import torch
    from .training_reproducibility import nonfinite_tensor_keys

    tensors, paths, host_failure = {}, [], []

    def visit(child, path):
        if host_failure:
            return
        if isinstance(child, torch.Tensor):
            # Ordinals keep distinct tensors even when their display paths collide.
            tensors[len(paths)] = child
            paths.append(path)
        elif isinstance(child, dict):
            for key, item in child.items():
                visit(item, path + "." + str(key))
        elif isinstance(child, (list, tuple)):
            for item in child:
                visit(item, path)
        elif isinstance(child, float) and not math.isfinite(child):
            host_failure.append(path)

    visit(value, label)
    failures = nonfinite_tensor_keys(tensors)
    if failures or host_failure:
        path = paths[failures[0]] if failures else host_failure[0]
        raise ValueError("Nonfinite checkpoint state: " + path)


def _development_role(contract: dict) -> str:
    """The retained recipe owns the exact development-role identity."""
    identity = contract.get("development_role_identity")
    _text(identity, "contract.development_role_identity")
    return identity


def _validate_development_semantics(contract: dict, semantics: dict) -> None:
    role = _development_role(contract)
    for field, weights in (("live_dev_metric", "live"), ("ema_dev_metric", "ema")):
        definition = semantics.get(field)
        if definition is None:
            continue  # The selected metric still requires an observation below.
        expected = {"metric": "MAE", "unit": "eV", "target": "Gap",
                    "role_identity": role, "weights": weights, "direction": "minimize"}
        if definition != expected:
            raise ValueError("Development metric semantics disagree with the frozen Gap MAE contract: " + field)


class FamilyOutputSession:
    """Small event adapter for an existing trainer; no replacement training loop.

    The trainer provides acknowledged counters, selected weights, predictions,
    sampler cursor and actual runtime observations. This session handles their
    persistence and translation consistently for both registered families.
    """
    def __init__(self, output_dir: Path, context: RunContext, *, adapter: str,
                 contract: Path, trajectory_id: str, metric_semantics: dict):
        import time
        from .v4_bundle import _payload
        from .v4_runtime import normalized_source_sha256
        self.started = time.perf_counter()
        self.root, self.context, self.adapter = Path(output_dir).absolute(), context, adapter
        _safe_local(self.root)
        _safe_local(Path(contract).absolute())
        artifact_adapter(adapter, (context.family_name, context.family_version))
        if normalized_source_sha256(Path(contract)) != context.training_recipe_sha256:
            raise ValueError("Session contract/Spec recipe mismatch")
        contract_record = _json(Path(contract))
        self.expected = contract_record.get("acceptance_requirements")
        if not isinstance(self.expected, dict) or set(self.expected) != EXPECTED:
            raise ValueError("Owning recipe lacks frozen acceptance_requirements")
        _validate_development_semantics(contract_record, metric_semantics)
        self.root.mkdir(parents=True, exist_ok=True)
        publish_immutable_bytes(self.root / "training_contract.json", _payload(Path(contract)))
        self.stage = StageRecorder(self.root / "canonical_trace.json", context, "downstream",
                                   metric_semantics, trajectory_id=trajectory_id)

    def epoch_finished(self, *, epoch: int, optimizer_step: int, sample_presentations: int, **metrics):
        self.stage.observe(epoch_or_pass=epoch, optimizer_step=optimizer_step,
                           sample_presentations=sample_presentations, **metrics)

    def selected(self, *, model_state: dict, epoch: int, optimizer_step: int, weights: str,
                 prediction_eV, target_eV, source_idx):
        from .training_reproducibility import atomic_torch_save
        mae = (prediction_eV.detach().cpu().double() - target_eV.detach().cpu().double()).abs().mean().item()
        write_selected_state(self.root / "selected_model.pt", self.context, model_state=model_state,
            epoch=epoch, optimizer_step=optimizer_step, weights=weights, development_mae_eV=mae)
        atomic_torch_save(self.root / "development_predictions.pt", {
            "context": self.context.to_dict(),
            "prediction_eV": prediction_eV.detach().cpu(), "target_eV": target_eV.detach().cpu(),
            "source_idx": source_idx.detach().cpu()})

    def checkpoint(self, *, model_state: dict, optimizer_state: dict, cursor: dict,
                   optimizer_step: int, sample_presentations: int, rng_state: dict,
                   scheduler_state: dict | None = None, ema_state: dict | None = None,
                   owner_state: dict | None = None):
        write_resume_state(self.root / "last_checkpoint.pt", self.context, adapter=self.adapter,
            model_state=model_state, optimizer_state=optimizer_state, rng_state=rng_state,
            cursor=cursor, optimizer_step=optimizer_step, sample_presentations=sample_presentations,
            scheduler_state=scheduler_state, ema_state=ema_state, owner_state=owner_state)

    def complete(self, *, runtime: dict, hardware: str, observed_costs: list | None = None,
                 target_identity: TargetIdentityBinding | None = None) -> dict:
        """Inspect retained outputs with the owning prelaunch target binding.

        An omitted binding retains canonical float64 identity. Historical target
        encodings require an explicit pinned binding and the same strict checks
        used during terminal closure.
        """
        import time
        rows = [row for row in self.stage.recorder.record["observations"]
                if row["event"] == "observation"]
        if not rows:
            raise ValueError("Cannot complete a session without observed training work")
        progress = {"epochs": len(rows), "optimizer_steps": rows[-1]["optimizer_step"],
                    "sample_presentations": rows[-1]["sample_presentations"]}
        costs = observed_costs if observed_costs is not None else [
            {"metric": "wall_seconds", "unit": "seconds", "value": time.perf_counter() - self.started,
             "status": "measured", "semantics": "process_wall", "hardware": hardware},
            {"metric": "device_seconds", "unit": "seconds", "value": None,
             "status": "missing", "semantics": "allocated_device", "hardware": hardware},
        ]
        _costs(costs)
        write_output_manifest(self.root, self.context, adapter=self.adapter,
            artifacts={"predictions": "development_predictions.pt", "selected_model": "selected_model.pt",
                       "resume": "last_checkpoint.pt", "trace": "canonical_trace.json", "contract": "training_contract.json"},
            progress=progress, runtime=runtime, costs=costs)
        return inspect_output(self.root, context=self.context, expected=self.expected,
                              target_identity=target_identity)


def tensor_digest(tensor, *, role: str) -> str:
    """Canonical row/target identity: little-endian int64 or float64 CPU bytes."""
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
        raise ValueError("Expected one-dimensional tensor")
    if role == "source_idx":
        if tensor.dtype != torch.int64:
            raise ValueError("source_idx must be int64")
        array = tensor.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    elif role == "target":
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError("target must be finite floating point")
        array = tensor.detach().cpu().double().contiguous().numpy().astype("<f8", copy=False)
    else:
        raise ValueError("Unknown digest role")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _runtime(runtime: dict, context: RunContext, precision: str):
    required = {"platform", "account", "precision", "source_commit", "source_archive_sha256"}
    if type(runtime) is not dict or not required <= set(runtime) or set(runtime) - required - {"runtime_certificate_id"}:
        raise ValueError("Runtime requires explicit platform/account/precision/source identity")
    for key in required - {"precision"}:
        if runtime[key] != getattr(context, key):
            raise ValueError(f"Runtime identity mismatch: {key}")
    if runtime["precision"] != precision:
        raise ValueError("Runtime precision mismatch")
    if "runtime_certificate_id" in runtime:
        _digest(runtime["runtime_certificate_id"], "runtime.runtime_certificate_id")


def _costs(costs):
    if type(costs) is not list or not costs:
        raise ValueError("Missing native cost observations")
    seen = set()
    for cost in costs:
        required = {"metric", "unit", "value", "status", "semantics", "hardware"}
        if type(cost) is not dict or not required <= set(cost) or set(cost) - required - {"scope", "reason"}:
            raise ValueError("Cost requires native metric/unit/status/semantics/hardware")
        if cost["metric"] in seen:
            raise ValueError("Duplicate native cost metric")
        seen.add(cost["metric"])
        if cost["status"] == "missing":
            if cost["value"] is not None:
                raise ValueError("Missing cost must remain null")
        elif cost["status"] in {"measured", "estimated"}:
            if isinstance(cost["value"], bool) or not isinstance(cost["value"], (int, float)) or not math.isfinite(cost["value"]) or cost["value"] < 0:
                raise ValueError("Invalid native cost")
        else:
            raise ValueError("Unknown cost status")
        allowed = {("wall_seconds", "seconds", "process_wall"),
                   ("device_seconds", "seconds", "allocated_device"),
                   ("device_seconds", "seconds", "device_busy")}
        if (cost["metric"], cost["unit"], cost["semantics"]) not in allowed:
            raise ValueError("Unsupported or conflated native cost semantics")
        _text(cost["hardware"], "cost.hardware")
        if "scope" in cost:
            _text(cost["scope"], "cost.scope")
        if "reason" in cost:
            if cost["status"] == "measured" and cost["reason"] is not None:
                raise ValueError("Measured cost cannot have a missing/estimate reason")
            if cost["status"] != "measured":
                _text(cost["reason"], "cost.reason")


@dataclass(frozen=True)
class TargetIdentityBinding:
    """An explicit prelaunch encoding binding; never infer it from outputs."""

    repo_root: Path
    plan_path: str
    plan_sha256: str
    package_dir: Path | None = None
    _verification: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_acceptance_plan(cls, spec: ExperimentSpec, repo_root: Path,
                             plan_path: str, *, plan_sha256: str):
        root = Path(repo_root).absolute()
        path = _artifact_path(root, plan_path)
        _digest(plan_sha256, "acceptance plan digest")
        if file_digest(path) != plan_sha256:
            raise ValueError("Target identity acceptance plan hash mismatch")
        report = check_acceptance_plan(spec, root, _json(path))
        if report["status"] != "ACCEPTANCE_INPUTS_AVAILABLE":
            raise ValueError("Target identity acceptance plan is blocked")
        return cls(root, plan_path, plan_sha256)

    @classmethod
    def from_frozen_package(cls, spec: ExperimentSpec, repo_root: Path,
                            package_dir: Path, plan_path: str, *, _verification=None):
        """Bind encoding metadata already verified by the original run resolver."""
        from .experiment_package import read_packaged_text
        from .experiment_spec import _unique_object, _canonical
        raw = read_packaged_text(package_dir, plan_path, _verification=_verification)
        plan = json.loads(raw, object_pairs_hook=_unique_object)
        _canonical(plan)
        if (plan.get("format") not in {"molgap-family-acceptance-plan-v1", "molgap-family-same-run-acceptance-plan-v1"}
                or plan.get("spec_identity") != spec.identity
                or [entry.get("arm_id") for entry in plan.get("arms", [])] !=
                   [arm["arm_id"] for arm in spec.to_dict()["arms"]]):
            raise ValueError("Frozen target identity plan/Spec mismatch")
        return cls(Path(repo_root).absolute(), plan_path,
                   hashlib.sha256(raw.encode()).hexdigest(), Path(package_dir).absolute(), _verification)

    def _pinned_json(self, name, digest, *, label="metadata"):
        _digest(digest, "target identity metadata digest")
        if self.package_dir is not None:
            from .experiment_package import read_packaged_text
            from .experiment_spec import _unique_object, _canonical
            inventory = _json(self.package_dir / "package_manifest.json")
            if name in inventory["relative_allowlist"]:
                raw = read_packaged_text(self.package_dir, name, _verification=self._verification)
                if hashlib.sha256(raw.encode()).hexdigest() != digest:
                    raise ValueError("Target identity " + label + " hash mismatch")
                result = json.loads(raw, object_pairs_hook=_unique_object)
                _canonical(result)
                return result
        path = _artifact_path(self.repo_root, name)
        if file_digest(path) != digest:
            raise ValueError("Target identity " + label + " hash mismatch")
        return _json(path)

    def digest(self, target, *, context: RunContext, expected: dict) -> tuple[str, dict]:
        import torch

        _digest(self.plan_sha256, "acceptance plan digest")
        plan = self._pinned_json(self.plan_path, self.plan_sha256, label="acceptance plan")
        if plan.get("format") not in {"molgap-family-acceptance-plan-v1", "molgap-family-same-run-acceptance-plan-v1"} or plan.get("spec_identity") != context.spec_identity:
            raise ValueError("Target identity plan/Spec mismatch")
        entries = [entry for entry in plan["arms"] if entry["arm_id"] == context.arm_id]
        if len(entries) != 1:
            raise ValueError("Target identity plan requires one matching arm")
        entry = entries[0]
        if entry["expected"] != expected or entry["contract"]["sha256"] != context.training_recipe_sha256:
            raise ValueError("Target identity plan/frozen expectations mismatch")
        pointer = (entry["target_manifest"] if plan["format"] == "molgap-family-same-run-acceptance-plan-v1"
                   else entry["reference_artifacts"]["target_manifest"])
        if set(pointer) != {"path", "sha256"}:
            raise ValueError("Target identity requires a pinned manifest")
        _digest(pointer["sha256"], "target manifest digest")
        manifest = self._pinned_json(pointer["path"], pointer["sha256"], label="manifest")
        if manifest.get("development_target_sha256") != expected["target_sha256"]:
            raise ValueError("Target identity manifest/frozen target mismatch")
        encoding = manifest.get("target_encoding")
        if encoding == "little-endian-float32-contiguous-raw-bytes":
            # Requiring the original dtype prevents a lossy cast hiding changes.
            if target.dtype != torch.float32 or not torch.isfinite(target).all():
                raise ValueError("Pinned float32 targets require finite original float32 tensors")
            data = target.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
            digest = hashlib.sha256(data).hexdigest()
        elif encoding == "little-endian-float64-contiguous-raw-bytes":
            digest = tensor_digest(target, role="target")
        else:
            raise ValueError("Unsupported pinned target encoding")
        return digest, {"encoding": encoding, "manifest": pointer,
                        "acceptance_plan_sha256": self.plan_sha256}


def _inspect_predictions(path, *, context, expected, target_identity):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or not {"source_idx", "target_eV", "prediction_eV", "context"} <= payload.keys():
        raise ValueError("Unsupported predictions payload")
    _check_context(payload["context"], context)
    if any(value is True for key, value in payload.items() if key in {"official_validation_used", "test_dev_used", "test_challenge_used"}):
        raise ValueError("Protected role consumption declared")
    tensors = [payload[key] for key in ("source_idx", "target_eV", "prediction_eV")]
    if any(not isinstance(t, torch.Tensor) or t.ndim != 1 or len(t) != expected["development_rows"] for t in tensors):
        raise ValueError("Prediction/target/row shape mismatch")
    idx, target, prediction = tensors
    if idx.dtype != torch.int64 or len(torch.unique(idx)) != len(idx):
        raise ValueError("Invalid or duplicate source rows")
    if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite or nonfloating predictions")
    if tensor_digest(idx, role="source_idx") != expected["source_idx_sha256"]:
        raise ValueError("Development row identity mismatch")
    observed = {}
    if target_identity is None:
        target_sha256 = tensor_digest(target, role="target")
    else:
        if not isinstance(target_identity, TargetIdentityBinding):
            raise ValueError("Expected explicit pinned target identity binding")
        target_sha256, observed["target_identity"] = target_identity.digest(
            target, context=context, expected=expected)
    if target_sha256 != expected["target_sha256"]:
        raise ValueError("Development target identity mismatch")
    observed["development_mae_eV"] = (prediction.double() - target.double()).abs().mean().item()
    return observed


def _inspect_selected_state(state, *, rows, trace, profile, observed):
    step, epoch = state.get("optimizer_step"), state.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= len(rows) or type(step) is not int or step not in {r["optimizer_step"] for r in rows}:
        raise ValueError("Selected endpoint is outside the retained trace")
    if state.get("weights") not in {"live", "ema"} or (state["weights"] == "ema" and not profile.ema_required):
        raise ValueError("Unsupported selected weight semantics")
    selected = rows[epoch - 1]
    if selected["optimizer_step"] != step:
        raise ValueError("Selected epoch/step disagree")
    metric = "ema_dev_metric" if state["weights"] == "ema" else "live_dev_metric"
    semantics = trace["metric_semantics"][metric]
    if not semantics or semantics.get("unit") != "eV" or semantics.get("weights") != state["weights"]:
        raise ValueError("Selected development metric semantics missing or mismatched")
    for value in (state.get("development_mae_eV"), selected[metric]):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not math.isclose(value, observed["development_mae_eV"], rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Selected trace/checkpoint/prediction metric mismatch")
    observed.update(selected_epoch=epoch, selected_optimizer_step=step, selected_weights=state["weights"])


def validate_retained_selection(output_dir: Path, *, context: RunContext, adapter: str,
                                contract: dict, trace: dict, target_identity=None) -> dict:
    """Check existing selection before resume; not terminal or replay acceptance."""
    import torch
    from .training_reproducibility import assert_finite_state_dict
    root = Path(output_dir).absolute()
    _safe_local(root)
    trace = validate_canonical_trace(trace)
    _validate_development_semantics(contract, trace["metric_semantics"])
    if trace["run_id"] != context.logical_run_id + ":" + context.arm_id + ":downstream":
        raise ValueError("Trace arm/stage identity mismatch")
    rows = [row for row in trace["observations"] if row["event"] == "observation"]
    if not rows or [row["epoch_or_pass"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Selection requires contiguous acknowledged epochs")
    observed = _inspect_predictions(_artifact_path(root, "development_predictions.pt"),
        context=context, expected=contract["acceptance_requirements"], target_identity=target_identity)
    state = torch.load(_artifact_path(root, "selected_model.pt"), map_location="cpu", weights_only=True)
    if type(state) is not dict:
        raise ValueError("selected_model context identity mismatch")
    _check_context(state.get("context"), context)
    if not isinstance(state.get("model"), dict) or not state["model"] or not all(isinstance(t, torch.Tensor) for t in state["model"].values()):
        raise ValueError("selected_model: missing tensor model state")
    assert_finite_state_dict(state["model"], label="selected_model")
    _inspect_selected_state(state, rows=rows, trace=trace,
        profile=artifact_adapter(adapter, (context.family_name, context.family_version)), observed=observed)
    return observed


def inspect_terminal_output(item: dict) -> dict:
    """Carry an explicit encoding binding through every closure reinspection."""
    kwargs = {"context": item["context"], "expected": item["expected"]}
    if item.get("target_identity") is not None:
        kwargs["target_identity"] = item["target_identity"]
    return inspect_output(item["output_dir"], **kwargs)


def inspect_output(output_dir: Path, *, context: RunContext, expected: dict,
                   target_identity: TargetIdentityBinding | None = None) -> dict:
    """Inspect pinned retained files on CPU, using safe tensor-only loading.

    Expected row/target hashes and exposure must come from the frozen contract,
    not from these outputs. No reference, role, runtime performance or causal
    qualification is inferred from this mechanical inspection.
    """
    import torch
    from .training_reproducibility import assert_finite_state_dict
    root = Path(output_dir).absolute()
    _safe_local(root)
    if (context.family_name, context.family_version) in {("neural_atom_k1", "3"), ("gptrans_t", "2")}:
        from .family_training_outputs import inspect_session_output
        return inspect_session_output(root, context=context, expected=expected, target_identity=target_identity)
    blockers, observed = [], {}
    try:
        if type(expected) is not dict or set(expected) != EXPECTED:
            raise ValueError("Missing or unknown acceptance expectations")
        for key in ("epochs", "optimizer_steps", "sample_presentations", "development_rows"):
            _positive(expected[key], key)
        for key in ("source_idx_sha256", "target_sha256"):
            _digest(expected[key], key)
        if expected["precision"] not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported precision")
        manifest = _json(_artifact_path(root, "output_manifest.json"))
        if set(manifest) != {"format", "context", "adapter", "artifacts", "progress", "runtime", "costs"} or manifest["format"] != OUTPUT_FORMAT:
            raise ValueError("Unsupported output manifest")
        _check_context(manifest["context"], context)
        observed["producer_platform_version"] = manifest["context"]["platform_version"]
        profile = artifact_adapter(manifest["adapter"], (context.family_name, context.family_version))
        artifacts = manifest["artifacts"]
        if type(artifacts) is not dict or set(artifacts) != ROLES:
            raise ValueError("Missing or unknown artifact roles")
        paths = {}
        for role, entry in artifacts.items():
            if type(entry) is not dict or set(entry) != {"path", "sha256"}:
                raise ValueError("Invalid artifact binding")
            _digest(entry["sha256"], role + ".sha256")
            paths[role] = _artifact_path(root, entry["path"])
            if file_digest(paths[role]) != entry["sha256"]:
                raise ValueError(f"Artifact hash mismatch: {role}")
        if len(set(paths.values())) != len(ROLES):
            raise ValueError("Artifact roles alias each other")
        if artifacts["contract"]["sha256"] != context.training_recipe_sha256:
            raise ValueError("Acceptance contract is not pinned by the Spec recipe")
        contract = _json(paths["contract"])
        if contract.get("acceptance_requirements") != expected:
            raise ValueError("Supplied expectations disagree with the frozen recipe contract")
        _development_role(contract)
        _runtime(manifest["runtime"], context, expected["precision"])
        _costs(manifest["costs"])
        progress = manifest["progress"]
        if type(progress) is not dict or set(progress) != {"epochs", "optimizer_steps", "sample_presentations"}:
            raise ValueError("Missing explicit progress counters")
        for key, value in progress.items():
            if type(value) is not int or value != expected[key]:
                raise ValueError(f"Progress mismatch: {key}")
        observed.update(_inspect_predictions(paths["predictions"], context=context,
            expected=expected, target_identity=target_identity))
        trace = validate_canonical_trace(_json(paths["trace"]))
        _validate_development_semantics(contract, trace["metric_semantics"])
        rows = [r for r in trace["observations"] if r["event"] == "observation"]
        if not rows or any(r["optimizer_step"] is None or r["sample_presentations"] is None or r["epoch_or_pass"] is None for r in rows):
            raise ValueError("Trace missing observed exposure/epoch counters")
        last = rows[-1]
        if last["optimizer_step"] != progress["optimizer_steps"] or last["sample_presentations"] != progress["sample_presentations"]:
            raise ValueError("Trace/terminal exposure mismatch")
        if len(rows) != progress["epochs"]:
            raise ValueError("Protocol requires one observation per completed epoch")
        if [r["epoch_or_pass"] for r in rows] != list(range(1, progress["epochs"] + 1)):
            raise ValueError("Trace epochs must be one-based contiguous completed epochs")
        expected_run = context.logical_run_id + ":" + context.arm_id + ":downstream"
        if trace["run_id"] != expected_run:
            raise ValueError("Trace arm/stage identity mismatch")
        for role in ("selected_model", "resume"):
            state = torch.load(paths[role], map_location="cpu", weights_only=True)
            if type(state) is not dict:
                raise ValueError(f"{role} context identity mismatch")
            _check_context(state.get("context"), context)
            if not isinstance(state.get("model"), dict) or not state["model"] or not all(isinstance(t, torch.Tensor) for t in state["model"].values()):
                raise ValueError(f"{role}: missing tensor model state")
            assert_finite_state_dict(state["model"], label=role)
            if role == "selected_model":
                _inspect_selected_state(state, rows=rows, trace=trace, profile=profile, observed=observed)
            else:
                if any(not isinstance(state.get(key), dict) or not state[key] for key in profile.resume_keys):
                    raise ValueError("Incomplete checkpoint/resume state")
                if profile.ema_required:
                    assert_finite_state_dict(state["ema"], label="resume.ema")
                optimizer = state["optimizer"]
                if not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]:
                    raise ValueError("Incomplete optimizer resume state")
                _validate_rng_state(state["rng_state"])
                _finite_tree(state, "resume")
                cursor = state.get("cursor")
                if type(cursor) is not dict or set(cursor) != {"epoch", "next_batch", "sampler_order_sha256"}:
                    raise ValueError("Missing acknowledged sampler cursor")
                if type(cursor["epoch"]) is not int or cursor["epoch"] != progress["epochs"] or type(cursor["next_batch"]) is not int or cursor["next_batch"] < 0:
                    raise ValueError("Resume cursor/epoch mismatch")
                _digest(cursor["sampler_order_sha256"], "resume.sampler_order_sha256")
                _positive(state.get("optimizer_step"), "resume.optimizer_step")
                _positive(state.get("sample_presentations"), "resume.sample_presentations")
                if state.get("optimizer_step") != progress["optimizer_steps"] or state.get("sample_presentations") != progress["sample_presentations"]:
                    raise ValueError("Resume/terminal exposure mismatch")
        observed.update(progress=progress, costs=manifest["costs"], artifacts=artifacts)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, EOFError) as exc:
        blockers.append(str(exc))
    return {"status": "BLOCKED" if blockers else "MECHANICALLY_VERIFIED", "blockers": blockers,
            "context": context.to_dict(), "observed": observed,
            "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED",
            "runtime_qualification": "NOT_EVALUATED"}


def _check_same_run_acceptance_plan(spec: ExperimentSpec, repo_root: Path, plan: dict, *,
                                    require_training_targets=False) -> dict:
    """Check frozen peer/target availability, never certify a future reference."""
    declaration = spec.to_dict()
    pair = declaration["prospective"].get("same_run_replay")
    if pair is None:
        raise ValueError("Same-run acceptance requires a prospective Spec pair")
    arms = {arm["arm_id"]: arm for arm in declaration["arms"]}
    if (type(plan) is not dict or set(plan) not in (
            {"format", "spec_identity", "arms"}, {"format", "spec_identity", "arms", "comparison_source"})
            or plan["spec_identity"] != spec.identity):
        raise ValueError("Acceptance plan/Spec mismatch")
    entries = plan["arms"]
    if type(entries) is not list or [entry.get("arm_id") for entry in entries] != list(arms):
        raise ValueError("Acceptance plan requires exactly all Spec arms in order")
    if set(arms) != {pair["reference_arm_id"], *pair["candidate_arm_ids"]}:
        raise ValueError("Same-run acceptance cannot include unbound arms")
    reference = arms[pair["reference_arm_id"]]
    configuration = None
    if "comparison_source" in plan:
        source = plan["comparison_source"]
        if type(source) is not dict or set(source) != {"path", "sha256"}:
            raise ValueError("Expected pinned scientific comparison source")
        _digest(source["sha256"], "comparison source digest")
        path = _artifact_path(Path(repo_root).absolute(), source["path"])
        if file_digest(path) != source["sha256"]:
            raise ValueError("Scientific comparison source hash mismatch")
        configuration = _json(path)
        if (configuration.get("format") != "molgap-scientific-experiment-v1"
                or configuration.get("logical_run_id") != declaration["logical_run_id"]
                or configuration.get("same_run", {}).get("reference_arm_id") != pair["reference_arm_id"]):
            raise ValueError("Scientific comparison source differs from frozen run")
    initialization_pair = configuration is not None and all(
        same_run_comparison_scope(configuration, candidate, pair["reference_arm_id"])["experiment_purpose"]
        == "initialization_comparison" for candidate in pair["candidate_arm_ids"])
    if initialization_pair and len(arms) != 2:
        raise ValueError("Initialization comparison requires one exact same-run pair")
    results, contracts, expectations = [], [], []
    for entry in entries:
        blockers = []
        try:
            if set(entry) != {"arm_id", "adapter", "expected", "contract", "target_manifest"}:
                raise ValueError("Missing or unknown same-run acceptance fields")
            arm = arms[entry["arm_id"]]
            artifact_adapter(entry["adapter"], (arm["family"]["name"], arm["family"]["version"]))
            for key in ("family", "base", "data", *(("initialization",) if not initialization_pair else ())):
                if arm[key] != reference[key]:
                    raise ValueError("Same-run peer identity differs: " + key)
            if initialization_pair:
                if arm["addons"] != reference["addons"] or arm["training"]["objective"] != reference["training"]["objective"]:
                    raise ValueError("Initialization comparison must retain architecture and loss")
                if arm is not reference and arm["initialization"] == reference["initialization"]:
                    raise ValueError("Initialization comparison requires distinct initial states")
            for key in ("sampler", "transform", "overrides"):
                if arm["training"][key] != reference["training"][key]:
                    raise ValueError("Same-run peer training identity differs: " + key)
            expected = entry["expected"]
            if type(expected) is not dict or set(expected) not in (EXPECTED, EXPECTED_FAMILY):
                raise ValueError("Incomplete terminal expectations")
            for key in ("optimizer_steps", "sample_presentations", "development_rows"):
                _positive(expected[key], key)
            if "epochs" in expected:
                _positive(expected["epochs"], "epochs")
            else:
                for key in ("completed_passes", "next_batch"):
                    if type(expected[key]) is not int or expected[key] < 0:
                        raise ValueError("Invalid shared terminal counter: " + key)
            for key in ("source_idx_sha256", "target_sha256"):
                _digest(expected[key], key)
            if expected["precision"] not in {"fp32", "fp16", "bf16"}:
                raise ValueError("Unsupported precision")
            def pinned(pointer):
                if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
                    raise ValueError("Expected pinned repository artifact")
                _digest(pointer["sha256"], "plan artifact digest")
                path = _artifact_path(Path(repo_root).absolute(), pointer["path"])
                if file_digest(path) != pointer["sha256"]:
                    raise ValueError("Plan artifact hash mismatch")
                return _json(path)
            contract = pinned(entry["contract"])
            if initialization_pair and contract.get("initialization_sha256") != arm["initialization"]["state_sha256"]:
                raise ValueError("Initialization contract differs from frozen Spec")
            if entry["contract"]["sha256"] != arm["training"]["recipe"]["sha256"] or contract.get("acceptance_requirements") != expected:
                raise ValueError("Acceptance expectations are not pinned by the owning Spec recipe")
            _development_role(contract)
            target = pinned(entry["target_manifest"])
            if require_training_targets:
                _digest(target.get("training_float32_target_sha256"), "training_float32_target_sha256")
            if target.get("development_target_sha256") != expected["target_sha256"] or target.get("target_encoding") not in {
                "little-endian-float32-contiguous-raw-bytes", "little-endian-float64-contiguous-raw-bytes"
            }:
                raise ValueError("Target encoding/identity differs from frozen expectations")
            contracts.append(contract)
            expectations.append(expected)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            blockers.append(str(exc))
        results.append({"arm_id": entry["arm_id"], "status": "BLOCKED" if blockers else "AVAILABLE", "blockers": blockers})
    if len(contracts) == len(entries) and (
        any(expected != expectations[0] for expected in expectations)
        or any(contract.get("training_recipe") != contracts[0].get("training_recipe") for contract in contracts)
        or any(_development_role(contract) != _development_role(contracts[0]) for contract in contracts)
    ):
        for row in results:
            row.update(status="BLOCKED", blockers=["Same-run peers differ in exposure, optimization or development identity"])
    if initialization_pair and len(contracts) == len(entries):
        by_id = dict(zip(arms, contracts))
        try:
            for candidate in pair["candidate_arm_ids"]:
                validate_initialization_peer(arms[candidate], reference, by_id[candidate], by_id[pair["reference_arm_id"]])
        except ValueError as exc:
            for row in results:
                row.update(status="BLOCKED", blockers=[str(exc)])
    return {"status": "BLOCKED" if any(row["blockers"] for row in results) else "ACCEPTANCE_INPUTS_AVAILABLE",
        "spec_identity": spec.identity, "arms": results, "trainer_execution": "NOT_VERIFIED",
        "reference_authority": "prospective_same_run_only; terminal reference not yet accepted"}


def acceptance_dependency_pins(plan: dict) -> dict[str, str]:
    """Project typed artifact references once for transport and recovery.

    Scientific/schema admission remains with check_acceptance_plan. Traversing
    the existing pointer type keeps optional dependencies in the same inventory.
    """
    from .common.paths import validate_relative_path
    pins = {}

    def collect(value):
        if type(value) is dict:
            if set(value) == {"path", "sha256"}:
                name = value["path"]
                validate_relative_path(name)
                _digest(value["sha256"], "acceptance dependency digest")
                if pins.setdefault(name, value["sha256"]) != value["sha256"]:
                    raise ValueError("Conflicting acceptance artifact pins")
            else:
                for child in value.values():
                    collect(child)
        elif type(value) is list:
            for child in value:
                collect(child)

    collect(plan)
    return pins


def check_acceptance_plan(spec: ExperimentSpec, repo_root: Path, plan: dict, *,
                          require_training_targets=False) -> dict:
    """Prelaunch capability and retained-reference check, without execution.

    The owning contract still provides comparison_prelaunch and reference_bundle.
    A passing plan only checks availability and schemas, not trainer execution.
    """
    from .comparison_readiness import validate_comparison_prelaunch, validate_reference_bundle, reference_bundle_digest
    if type(plan) is dict and plan.get("format") == "molgap-family-same-run-acceptance-plan-v1":
        return _check_same_run_acceptance_plan(spec, repo_root, plan,
                                               require_training_targets=require_training_targets)
    declaration = spec.to_dict()
    expected_arms = {a["arm_id"]: a for a in declaration["arms"]}
    results = []
    if type(plan) is not dict or set(plan) != {"format", "spec_identity", "arms"} or plan["format"] != "molgap-family-acceptance-plan-v1" or plan["spec_identity"] != spec.identity:
        raise ValueError("Acceptance plan/spec mismatch")
    entries = plan["arms"]
    if type(entries) is not list or [e.get("arm_id") for e in entries] != list(expected_arms):
        raise ValueError("Acceptance plan requires exactly all Spec arms in order")
    for entry in entries:
        blockers = []
        try:
            if set(entry) != {"arm_id", "adapter", "expected", "contract", "comparison_prelaunch", "reference_bundle", "reference_artifacts"}:
                raise ValueError("Missing or unknown acceptance plan fields")
            arm = expected_arms[entry["arm_id"]]
            artifact_adapter(entry["adapter"], (arm["family"]["name"], arm["family"]["version"]))
            if type(entry["expected"]) is not dict or set(entry["expected"]) not in (EXPECTED, EXPECTED_FAMILY):
                raise ValueError("Incomplete terminal expectations")
            for key in ("optimizer_steps", "sample_presentations", "development_rows"):
                _positive(entry["expected"][key], key)
            if "epochs" in entry["expected"]:
                _positive(entry["expected"]["epochs"], "epochs")
            else:
                for key in ("completed_passes", "next_batch"):
                    if type(entry["expected"][key]) is not int or entry["expected"][key] < 0:
                        raise ValueError("Invalid shared terminal counter: " + key)
            for key in ("source_idx_sha256", "target_sha256"):
                _digest(entry["expected"][key], key)
            if entry["expected"]["precision"] not in {"fp32", "fp16", "bf16"}:
                raise ValueError("Unsupported precision")
            def pinned(pointer):
                if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
                    raise ValueError("Expected pinned repository artifact")
                _digest(pointer["sha256"], "plan artifact digest")
                path = _artifact_path(Path(repo_root).absolute(), pointer["path"])
                if file_digest(path) != pointer["sha256"]:
                    raise ValueError("Plan artifact hash mismatch")
                return path
            contract_path = pinned(entry["contract"])
            contract_record = _json(contract_path)
            if entry["contract"]["sha256"] != arm["training"]["recipe"]["sha256"] or contract_record.get("acceptance_requirements") != entry["expected"]:
                raise ValueError("Acceptance expectations are not pinned by the owning Spec recipe")
            _development_role(contract_record)
            comparison = validate_comparison_prelaunch(_json(pinned(entry["comparison_prelaunch"])))
            if not comparison["prelaunch_ready"]:
                raise ValueError("Owning comparison prelaunch is blocked")
            bundle = validate_reference_bundle(_json(pinned(entry["reference_bundle"])))
            if comparison["reference_bundle_id"] != bundle["reference_bundle_id"] or comparison["reference_bundle_sha256"] != reference_bundle_digest(bundle):
                raise ValueError("Comparison/reference bundle mismatch")
            predictions = bundle["prediction_manifest"]
            if predictions["source_idx_sha256"] != entry["expected"]["source_idx_sha256"] or predictions["target_sha256"] != entry["expected"]["target_sha256"] or predictions["row_count"] != entry["expected"]["development_rows"]:
                raise ValueError("Reference/expected development identity mismatch")
            bindings = entry["reference_artifacts"]
            from .comparison_readiness import REQUIRED_REFERENCE_ARTIFACTS
            if type(bindings) is not dict or set(bindings) != REQUIRED_REFERENCE_ARTIFACTS | {"predictions"}:
                raise ValueError("Incomplete retained reference artifacts")
            for pointer in bindings.values():
                pinned(pointer)
            if bindings["predictions"]["sha256"] != predictions["prediction_sha256"]:
                raise ValueError("Retained reference prediction hash mismatch")
            owners = {"runtime_certificate": "runtime_certificate_ref", "row_manifest": "row_manifest_ref",
                      "target_manifest": "target_manifest_ref", "trace_manifest": "trace_manifest_ref",
                      "role_history": "role_history_ref", "target_transform_asset": "target_transform_asset_ref",
                      "cost_records": "cost_records_ref", "acceptance": "acceptance_ref", "decision": "decision_ref"}
            for artifact, field in owners.items():
                if repo_local_path(repo_root, bindings[artifact]["path"]) != repo_local_path(repo_root, bundle[field]):
                    raise ValueError("Reference artifact/owner pointer mismatch: " + artifact)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            blockers.append(str(exc))
        results.append({"arm_id": entry["arm_id"], "status": "BLOCKED" if blockers else "AVAILABLE", "blockers": blockers})
    return {"status": "BLOCKED" if any(r["blockers"] for r in results) else "ACCEPTANCE_INPUTS_AVAILABLE",
            "spec_identity": spec.identity, "arms": results, "trainer_execution": "NOT_VERIFIED"}


def close_verified_outputs(repo_root: Path, spec: ExperimentSpec, descriptor, *, outputs: dict,
                           execute: bool = False, _reports: dict | None = None) -> dict:
    """Reinspect every arm, then delegate the unchanged terminal/RML transaction.

    outputs maps arm_id to {context, output_dir, expected}. Descriptor artifact
    paths and SHA must agree with the inspected raw files before any RML write.
    """
    from .experiment_terminal import execute_terminal_descriptor, translate_terminal_descriptor
    if type(execute) is not bool:
        raise ValueError("Expected explicit boolean execution switch")
    arms = spec.to_dict()["arms"]
    if set(outputs) != {a["arm_id"] for a in arms}:
        raise ValueError("Closure requires independent output inputs for every arm")
    # Execution translates immediately before the RML transaction. A dry run
    # needs the same binding checks here, without a second full artifact scan.
    if not execute:
        translate_terminal_descriptor(repo_root, spec, descriptor)
    reports = {}
    for arm in descriptor.to_dict()["arms"]:
        item = outputs[arm["arm_id"]]
        report = inspect_terminal_output(item) if _reports is None else _reports[arm["arm_id"]]
        reports[arm["arm_id"]] = report
        if report["status"] == "BLOCKED":
            continue
        ctx = item["context"]
        expected_arm = next(a for a in arms if a["arm_id"] == ctx.arm_id)
        if ctx.spec_identity != spec.identity or ctx.arm_identity != canonical_fingerprint(expected_arm):
            raise ValueError("Verified output/Spec mismatch")
        identity = arm["observed"]["identity"]
        for key, value in (("source_commit", ctx.source_commit), ("source_package_sha256", ctx.source_archive_sha256)):
            if identity[key]["value"] != value:
                raise ValueError("Descriptor/verified source identity mismatch")
        if identity["platform"]["run_reference"]["value"] != ctx.run_reference:
            raise ValueError("Descriptor/verified physical run mismatch")
        for role, output_role in (("predictions", "predictions"), ("checkpoint", "selected_model"), ("trace", "trace")):
            binding = arm["observed"]["artifacts"][role]
            checked = report["observed"]["artifacts"][output_role]
            if binding["sha256"] != checked["sha256"]:
                raise ValueError("Descriptor/verified artifact mismatch: " + role)
            inspected_path = _artifact_path(Path(item["output_dir"]).absolute(), checked["path"])
            from .research_memory.paths import bound_artifact_path
            if bound_artifact_path(repo_root, binding["locator"]) != inspected_path:
                raise ValueError("Descriptor/verified artifact path mismatch: " + role)
            if role == "trace" and ("trace" not in arm or bound_artifact_path(repo_root, arm["trace"]) != inspected_path):
                raise ValueError("Descriptor closure trace differs from the inspected trace")
        pass_counter = "completed_passes" if "completed_passes" in report["observed"]["progress"] else "epochs"
        for field, counter in (("epoch", pass_counter), ("step", "optimizer_steps"), ("samples", "sample_presentations")):
            if arm["observed"]["progress"][field]["value"] != report["observed"]["progress"][counter]:
                raise ValueError("Descriptor/verified progress mismatch")
    if any(r["status"] == "BLOCKED" for r in reports.values()):
        return {"status": "BLOCKED", "arms": reports, "executed": False}
    if not execute:
        return {"status": "MECHANICALLY_VERIFIED", "arms": reports, "executed": False}
    results = execute_terminal_descriptor(repo_root, spec, descriptor)
    complete = all(r.get("pipeline_status") == "COMPLETE" for r in results)
    return {"status": "COMPLETE" if complete else "INCOMPLETE",
            "arms": reports, "results": results, "executed": True, "closure_complete": complete}


def prepare_terminal_outputs(spec: ExperimentSpec, repo_root: Path, supplied: dict, *,
                             receipt_path: Path, package_dir: Path,
                             expected_package_identity: str, _verification=None) -> dict:
    """Confine CLI output inputs in the owning core, keeping CLI dispatch thin."""
    if type(supplied) is not dict or set(supplied) != {a["arm_id"] for a in spec.to_dict()["arms"]}:
        raise ValueError("Expected every independent Spec arm")
    outputs = {}
    for arm_id, item in supplied.items():
        if type(item) is not dict or set(item) != {"output_dir", "expected"}:
            raise ValueError("Expected per-arm output_dir/expected inputs")
        output_dir = Path(item["output_dir"])
        output_dir = output_dir if output_dir.is_absolute() else Path(repo_root) / output_dir
        _safe_local(output_dir.absolute())
        outputs[arm_id] = {
            "context": RunContext.from_launch(spec, receipt_path, package_dir,
                expected_package_identity=expected_package_identity, arm_id=arm_id,
                **({"_verification": _verification} if _verification is not None else {})),
            "expected": item["expected"], "output_dir": output_dir.absolute()}
    return outputs


def build_verified_terminal_descriptor(repo_root: Path, spec: ExperimentSpec, *, outputs: dict,
                                       locations: dict, _reports: dict | None = None):
    """Translate mechanical observations without per-experiment evidence glue.

    locations supplies existing trajectory, terminal, canonical trace and run ID.
    Scientific acceptance/V5/cost/role records remain the owning pipeline inputs;
    this helper neither invents nor rewrites their decisions.
    """
    from .experiment_terminal import TerminalDescriptor, translate_terminal_descriptor
    from .experiment_spec import TERMINAL_PROTOCOL
    root = Path(repo_root).resolve()
    declaration = spec.to_dict()
    ids = {a["arm_id"] for a in declaration["arms"]}
    if set(outputs) != ids or set(locations) != ids:
        raise ValueError("Descriptor requires every independently inspected arm")
    entries = []
    fact = lambda value: {"value": value, "missing_reason": None}
    for arm in declaration["arms"]:
        arm_id = arm["arm_id"]
        item = outputs[arm_id]
        ctx = item["context"]
        if ctx.spec_identity != spec.identity or ctx.arm_identity != canonical_fingerprint(arm):
            raise ValueError("Output/Spec arm binding mismatch")
        report = inspect_terminal_output(item) if _reports is None else _reports[arm_id]
        if report["status"] != "MECHANICALLY_VERIFIED":
            raise ValueError("Output inspection blocked: " + "; ".join(report["blockers"]))
        where = locations[arm_id]
        if set(where) != {"trajectory_id", "run_id", "trajectory", "terminal", "trace"}:
            raise ValueError("Expected existing per-arm trajectory/terminal/trace locations")
        observation = report["observed"]
        def binding(role):
            entry = observation["artifacts"][role]
            path = _artifact_path(Path(item["output_dir"]).absolute(), entry["path"])
            from .research_memory.paths import artifact_locator
            relative = artifact_locator(root, path)
            return {"status": "available", "locator": relative, "sha256": entry["sha256"], "missing_reason": None}
        progress = observation["progress"]
        identity = _terminal_identity(spec, ctx)
        # A contract binding is a declaration. Completion has independent file,
        # shape and exposure checks above; causal validity is still unevaluated.
        costs = [{"metric": c["metric"], "unit": c["unit"], "value": c["value"],
                  "status": "measurement_missing" if c["status"] == "missing" else c["status"],
                  "reason": None if c["status"] == "measured" else c.get("reason",
                      "Producer retained no native " + c["hardware"] + " " + c["metric"] + " measurement (" + c["semantics"] + ")")}
                 for c in observation["costs"]]
        missing = ["runtime_qualification_not_evaluated", "scientific_comparison_not_evaluated"]
        if observation.get("producer_platform_version") is None:
            missing.append("producer_platform_version_not_observed")
        metrics = {"status": "missing", "locator": None, "sha256": None,
                   "missing_reason": "Owning scientific metric record is separate from mechanical inspection"}
        entry = {"arm_id": arm_id, "arm_identity": ctx.arm_identity, **where,
                 "observed": {"identity": identity,
                    "terminal": {"status": fact("complete"), "exit_reason": fact("verified_outputs_complete")},
                    "artifacts": {"metrics": metrics, "predictions": binding("predictions"),
                                  "checkpoint": binding("selected_model"), "trace": binding("trace")},
                    "progress": {"epoch": fact(progress["completed_passes"] if "completed_passes" in progress else progress["epochs"]), "step": fact(progress["optimizer_steps"]),
                                 "samples": fact(progress["sample_presentations"])},
                    "costs": costs, "missing_evidence": missing}}
        from .research_memory.paths import bound_artifact_path
        if bound_artifact_path(root, where["trace"]) != bound_artifact_path(root, entry["observed"]["artifacts"]["trace"]["locator"]):
            raise ValueError("Descriptor closure trace differs from the inspected trace")
        entries.append(entry)
    descriptor = TerminalDescriptor(spec, {"schema_version": TERMINAL_PROTOCOL,
        "spec_identity": spec.identity, "experiment_id": declaration["experiment_id"],
        "logical_run_id": declaration["logical_run_id"], "arms": entries})
    translate_terminal_descriptor(root, spec, descriptor)
    return descriptor


def terminal_attempt_id(spec, arm_id, platform_version):
    """A same-job pair shares an attempt; arm identity remains separate."""
    if not platform_version:
        return None
    declaration = spec.to_dict()
    prefix = declaration["logical_run_id"] if declaration["prospective"].get("same_run_replay") else arm_id
    return prefix + "-v" + platform_version


def _terminal_identity(spec, context):
    """One translator owns identity fields for complete and incomplete runs."""
    arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == context.arm_id)
    if context.spec_identity != spec.identity or context.arm_identity != canonical_fingerprint(arm):
        raise ValueError("Terminal context/Spec mismatch")
    fact = lambda value: {"value": value, "missing_reason": None}
    return {
        "experiment_id": context.experiment_id, "logical_run_id": context.logical_run_id,
        "arm_id": context.arm_id, "spec_identity": context.spec_identity,
        "family": fact(arm["family"]),
        "recipe_identity": fact(canonical_fingerprint(arm["training"]["recipe"])),
        "data_identity": fact(canonical_fingerprint(arm["data"])),
        "split_identity": fact(canonical_fingerprint({"split": arm["data"]["split"], "roles": arm["data"]["roles"]})),
        "feature_identity": fact(arm["data"]["feature_sha256"]), "target": fact(arm["data"]["target"]),
        "initialization_identity": fact(canonical_fingerprint(arm["initialization"])),
        "source_commit": fact(context.source_commit), "source_package_sha256": fact(context.source_archive_sha256),
        "attempt_id": fact(terminal_attempt_id(spec, context.arm_id, context.platform_version)) if context.platform_version else
            {"value": None, "missing_reason": "Platform attempt version not observed"},
        "platform": {"name": context.platform, "run_reference": fact(context.run_reference)},
    }


def build_incomplete_terminal_descriptor(spec: ExperimentSpec, *, contexts: dict,
                                         locations: dict, observations: dict):
    """Build the existing descriptor for truthful failures/cancellation/interrupts.

    The caller supplies scheduler observations and existing canonical terminal
    inputs. Missing progress/cost/artifacts stay unknown; zero requires an
    explicit observation. This never fabricates acceptance or finalizes RML.
    """
    from .experiment_terminal import TerminalDescriptor
    from .experiment_spec import TERMINAL_PROTOCOL
    ids = {a["arm_id"] for a in spec.to_dict()["arms"]}
    if any(set(mapping) != ids for mapping in (contexts, locations, observations)):
        raise ValueError("Incomplete closure requires every independent arm")
    entries = []
    fact = lambda value, reason: {"value": value, "missing_reason": reason if value is None else None}
    for arm in spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        observed, where = observations[arm_id], locations[arm_id]
        if type(observed) is not dict or not {"status", "exit_reason"} <= set(observed) or set(observed) - {"status", "exit_reason", "progress", "costs", "artifacts", "missing_evidence"}:
            raise ValueError("Expected terminal status/exit_reason and optional observed progress/cost/artifacts")
        if observed["status"] not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("Complete runs require independently verified family outputs")
        if set(where) not in ({"trajectory_id", "run_id", "trajectory", "terminal"},
                              {"trajectory_id", "run_id", "trajectory", "terminal", "trace"}):
            raise ValueError("Expected existing canonical terminal locations")
        progress = observed.get("progress", {})
        if type(progress) is not dict or set(progress) - {"epoch", "step", "samples"}:
            raise ValueError("Unsupported observed progress counters")
        artifacts = observed.get("artifacts", {})
        if type(artifacts) is not dict or set(artifacts) - {"metrics", "predictions", "checkpoint", "trace"}:
            raise ValueError("Unsupported terminal artifact roles")
        missing_artifact = {"status": "missing", "locator": None, "sha256": None,
                            "missing_reason": "Not retained for this incomplete attempt"}
        costs = observed.get("costs", [{"metric": "wall_seconds", "unit": "seconds", "value": None,
            "status": "measurement_missing", "reason": "No retained native cost measurement for this attempt"}])
        entries.append({"arm_id": arm_id, "arm_identity": contexts[arm_id].arm_identity, **where,
            "observed": {"identity": _terminal_identity(spec, contexts[arm_id]),
                "terminal": {"status": fact(observed["status"], "Terminal state not observed"),
                             "exit_reason": fact(observed["exit_reason"], "Exit reason not retained")},
                "artifacts": {role: artifacts.get(role, dict(missing_artifact)) for role in
                              ("metrics", "predictions", "checkpoint", "trace")},
                "progress": {key: fact(progress.get(key), "Progress counter not retained")
                             for key in ("epoch", "step", "samples")},
                "costs": costs, "missing_evidence": observed.get("missing_evidence", ["incomplete_attempt"])} })
    declaration = spec.to_dict()
    return TerminalDescriptor(spec, {"schema_version": TERMINAL_PROTOCOL,
        "spec_identity": spec.identity, "experiment_id": declaration["experiment_id"],
        "logical_run_id": declaration["logical_run_id"], "arms": entries})


def incomplete_observations_from_execution(spec: ExperimentSpec, state: dict) -> dict:
    """Translate retained failed orchestration facts; no scientific disposition."""
    ids = {a["arm_id"] for a in spec.to_dict()["arms"]}
    if (state.get("format") != "molgap-kaggle-two-phase-pair-v2" or
        state.get("spec_identity") != spec.identity or state.get("status") != "failed" or
        type(state.get("arms")) is not dict or set(state["arms"]) != ids):
        raise ValueError("Expected the exact failed execution state for every arm")
    declaration = spec.to_dict()
    hardware = state.get("hardware")
    count = declaration["platform"]["device_count"]
    if (declaration["platform"]["name"] != "kaggle" or type(hardware) is not list or
        len(hardware) != count or any(type(name) is not str or "T4" not in name for name in hardware)):
        raise ValueError("Missing matching observed T4 allocation")
    devices = [arm.get("device") for arm in state["arms"].values()]
    if (any(type(device) is not int or not 0 <= device < count for device in devices) or
        len(set(devices)) != len(devices)):
        raise ValueError("Missing distinct observed arm device assignment")
    observations = {}
    for arm_id, arm in state["arms"].items():
        if type(arm.get("training_started")) is not bool:
            raise ValueError("Training-start state is unknown")
        status = arm.get("terminal_status")
        if status not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("Completed/unknown arms require their owning artifact inspection")
        seconds = arm.get("worker_wall_seconds")
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Missing retained worker allocation window")
        observations[arm_id] = {"status": status, "exit_reason": arm["exit_reason"],
            "progress": {key: None if arm["training_started"] else 0 for key in ("epoch", "step", "samples")},
            "costs": [{"metric": "wall_seconds", "unit": "seconds", "value": seconds,
                       "status": "measured", "reason": None},
                      {"metric": "device_seconds", "unit": "seconds", "value": seconds,
                       "status": "measured", "reason": None}],
            "missing_evidence": ["training_progress_not_retained"] if arm["training_started"] else ["formal_training_not_started"]}
    return observations
