"""Observed-only curve projections and conservative, non-executing decisions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .comparison_readiness import STRICT_IDENTITY_FIELDS
from .cross_scale_plan import digest, exact, number
from .research_memory.paths import repo_local_path
from .research_memory.trace import file_digest, load_canonical_trace, validate_manifest_trace


def read_bound_json(root, binding):
    import json
    from .common.spec import _unique_object
    exact(binding, {"path", "sha256"}, "artifact binding")
    digest(binding["sha256"])
    path = repo_local_path(Path(root), binding["path"])
    if file_digest(path) != binding["sha256"]:
        raise ValueError("Bound artifact SHA mismatch: " + binding["path"])
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def import_accepted_trace(root, *, trace_binding, manifest_binding, acceptance_binding):
    """No data/model IO; all three independent artifact pins are caller supplied.

    This verifies a retained acceptance binding, not its scientific authority.
    Historical manifest backtest exclusions and missing fields stay explicit.
    """
    manifest = read_bound_json(root, manifest_binding)
    acceptance = read_bound_json(root, acceptance_binding)
    from .research_memory.schemas import validate_trace_manifest
    validate_trace_manifest(manifest)
    if (acceptance.get("status") != "accepted" or
            acceptance.get("trace_sha256") != trace_binding["sha256"] or
            acceptance.get("manifest_sha256") != manifest_binding["sha256"] or
            manifest.get("trace_artifact_sha256") != trace_binding["sha256"]):
        raise ValueError("Trace lacks matching hash-bound acceptance")
    if manifest["trace_artifact_ref"] != trace_binding["path"]:
        raise ValueError("Trace pointer differs from manifest")
    read_bound_json(root, trace_binding)
    trace = load_canonical_trace(repo_local_path(Path(root), trace_binding["path"]))
    validate_manifest_trace(manifest, trace)
    rows = []
    for observed in trace["observations"]:
        if observed["event"] != "observation":
            continue
        for name, definition in trace["metric_semantics"].items():
            value = observed[name]
            if definition is None:
                continue
            if definition["target"] not in {"Gap", "gap", "PCQM4Mv2 Gap"} or definition["direction"] != "minimize":
                raise ValueError("Unknown target/direction")
            if definition["unit"] not in {"eV", "normalized-Gap"}:
                raise ValueError("Unknown metric unit")
            role = definition["role_identity"]
            if (not (role.startswith("internal_") or role.endswith(":train") or role.endswith(":development")) or
                    any(term in role.lower() for term in ("audit", "official", "test", "ood", "p8", "external", "shadow", "protected"))):
                raise ValueError("ROLE_BLOCKED: trace role is not an internal role")
            rows.append({"optimizer_step": observed["optimizer_step"],
                "sample_presentations": observed["sample_presentations"],
                "unique_members_observed": None,
                "complete_passes": observed["epoch_or_pass"] if type(observed["epoch_or_pass"]) is int else None,
                "epoch_or_pass_observed": observed["epoch_or_pass"],
                "cumulative_measured_wall_seconds": observed["cumulative_wall_time_seconds"],
                "allocated_device_hours": None, "learning_rate": observed["learning_rate"],
                "checkpoint_identity": observed["checkpoint_identity"],
                "metric": name, "metric_kind": "online-train" if name == "live_train_metric" else "development",
                "value": value, "target": definition["target"], "unit": definition["unit"], "weights": definition["weights"],
                "evaluation_role": role, "target_identity":manifest["comparability_identity"].get("target_identity"),
                "source_trace_sha256":trace_binding["sha256"], "trace_run_id":trace["run_id"]})
    return {"rows": rows, "comparability_identity": deepcopy(manifest["comparability_identity"]),
            "backtest_eligibility": deepcopy(manifest["backtest_eligibility"]),
            "trace_sha256": trace_binding["sha256"], "scientific_acceptance": "NOT_REASSESSED",
            "observation_identity": {"trace_sha256":trace_binding["sha256"], "trace_run_id":trace["run_id"],
                "metric_semantics":deepcopy(trace["metric_semantics"]),
                "checkpoint_identities":{str(r["optimizer_step"]):r["checkpoint_identity"]
                    for r in trace["observations"] if r["event"] == "observation" and r["checkpoint_identity"] is not None}},
            "rml_written": False}


def align_curves(left, right, *, left_identity, right_identity, metric="live_dev_metric",
                 axis="optimizer_step", lane="MECHANISM", nested_cohorts_verified=False,
                 cohort_proof=None, comparison_type=None):
    """Intersect actually observed coordinates; no interpolation or extrapolation.

    Explicit identity uses existing V5 names. Data-diversity matching relaxes
    only train membership/exposure, and requires an independent nested-role proof.
    It never grants V5 STRICT_CAUSAL or prospective admission.
    """
    from .cross_scale_inputs import CohortProof
    kinds = {"MECHANISM": "SAME_SCALE_MODEL", "DATA_DIVERSITY": "MATCHED_STEP_SCALE",
             "TUNED_RECIPE": "TUNED_RECIPE_SYSTEM"}
    if comparison_type is None:
        comparison_type = kinds.get(lane)
    if comparison_type not in {*kinds.values(), "CONTEXT_OR_PORTABILITY"}:
        raise ValueError("Unknown comparison type")
    lane = {v: k for k, v in kinds.items()}.get(comparison_type, "CONTEXT")
    if type(nested_cohorts_verified) is not bool:
        raise TypeError("Legacy cohort flag must be an exact bool; use a bound cohort proof")
    if axis not in {"optimizer_step", "sample_presentations"}:
        raise ValueError("Epoch/pass-only alignment is forbidden")
    fields = set(STRICT_IDENTITY_FIELDS) | {"sampler_identity", "schedule_axis", "target_unit", "evaluation_row_sha256",
        "evaluation_target_sha256", "evaluation_row_count", "bn_calibration_identity", "drop_last"}
    ignored = {"optimizer_steps", "sample_presentations"}
    if lane == "MECHANISM":
        ignored.add("architecture_config_identity")
    if lane == "TUNED_RECIPE":
        ignored |= {"architecture_config_identity", "optimizer_identity", "optimizer_mode", "optimizer_fused",
                    "schedule_identity", "schedule_axis", "loss_identity", "sampler_identity",
                    "target_transform_identity", "target_transform_asset_sha256", "weight_semantics",
                    "ema_enabled", "ema_decay", "ema_update_frequency", "evaluation_weight_source",
                    "bn_calibration_identity", "checkpoint_selection_identity"}
    if lane == "DATA_DIVERSITY":
        ignored |= {"data_role_identity", "row_membership_identity", "row_order_identity"}
    reasons = []
    for field in sorted(fields - ignored):
        if (field in {"ema_decay", "ema_update_frequency"} and
                left_identity.get("ema_enabled") is False and right_identity.get("ema_enabled") is False and
                field in left_identity and field in right_identity and left_identity[field] is None and right_identity[field] is None):
            continue  # Explicitly disabled EMA is not missing evidence.
        if left_identity.get(field) is None or right_identity.get(field) is None:
            reasons.append("MISSING:" + field)
        elif type(left_identity[field]) is not type(right_identity[field]) or left_identity[field] != right_identity[field]:
            reasons.append("MISMATCH:" + field)
    for name in ("tf32_enabled", "deterministic_algorithms", "ema_enabled", "drop_last"):
        if any(type(i.get(name)) is not bool for i in (left_identity, right_identity)):
            reasons.append("INVALID_TYPE:" + name)
    if lane == "DATA_DIVERSITY":
        if type(cohort_proof) is not CohortProof:
            reasons.append("MISSING:bound_cohort_proof")
        else:
            try:
                cohort_proof.verify(left_identity, right_identity)
            except ValueError as exc:
                reasons.append("MISMATCH:bound_cohort_proof:" + str(exc))
    if lane == "DATA_DIVERSITY" and (left_identity.get("schedule_axis") != "optimizer_step" or right_identity.get("schedule_axis") != "optimizer_step"):
        reasons.append("MISMATCH:step_indexed_schedule_required")
    if lane == "CONTEXT":
        reasons.append("DECLARED:context_or_portability")
    if lane == "TUNED_RECIPE":
        for field in ("tuning_budget_identity", "tuning_budget_per_system", "tuning_protocol_sha256"):
            if any(i.get(field) is None for i in (left_identity, right_identity)):
                reasons.append("MISSING:" + field)
            elif left_identity[field] != right_identity[field]:
                reasons.append("MISMATCH:" + field)

    from .comparison_readiness import metric_observation_contract, canonical_weight_state
    from .research_memory.trace import METRICS
    metrics = (metric,metric) if isinstance(metric,str) else metric
    if type(metrics) is not tuple or len(metrics) != 2 or any(m not in METRICS for m in metrics):
        raise ValueError("Expected one canonical metric or a two-side metric tuple")

    def panel(rows, identity, selected, side):
        result, states, last = {}, {}, -1
        expected, semantic_issues = metric_observation_contract(identity, selected)
        reasons.extend(f"{r}[{side}]" for r in semantic_issues)
        for row in rows:
            point = row.get(axis)
            if point is None:
                raise ValueError("Missing observed coordinate")
            number(point, axis, integer=True)
            if point < last:
                raise ValueError("Duplicate or non-monotonic observed coordinate")
            last = point
            value = row.get("value", row.get(row.get("metric")))
            if value is not None:
                number(value, "MAE")
            observed = {k:row.get(k) for k in ("optimizer_step","sample_presentations","learning_rate",
                "source_trace_sha256","trace_run_id")}
            if point in states:
                for key,value in observed.items():
                    prior = states[point].get(key)
                    if prior is not None and value is not None and (type(prior) is not type(value) or prior != value):
                        history_issues.append({"coordinate":point,"side":side,"reason":"MISMATCH:conflicting_step_"+key})
                    elif prior is None and value is not None:
                        states[point][key] = value
            else:
                states[point] = observed
            for key in ("source_trace_sha256","trace_run_id"):
                if row.get(key) != expected.get(key) or row.get(key) is None:
                    history_issues.append({"coordinate":point,"side":side,"reason":"MISMATCH:observed_"+key})
            if row.get("metric") not in METRICS:
                history_issues.append({"coordinate":point,"side":side,"reason":"MISSING_OR_INVALID:metric_slot"})
            else:
                row_expected, row_issues = metric_observation_contract(
                    identity, row["metric"], selected=row["metric"] == selected)
                for field, declared in row_expected.items():
                    observed = row.get(field)
                    if field == "weights":
                        observed = canonical_weight_state(observed)
                    if field == "target" and observed in {"gap", "PCQM4Mv2 Gap"}:
                        observed = "Gap"
                    if observed is None or declared is None or observed != declared:
                        row_issues.append("MISMATCH:observed_" + field)
                history_issues.extend({"coordinate":point,"side":side,"reason":r} for r in row_issues)
            if row.get("metric") != selected:
                continue
            if point in result:
                raise ValueError("Duplicate or non-monotonic observed coordinate for metric")
            result[point] = row
        return result,states,expected

    history_issues = []
    a, astates, aexpected = panel(left,left_identity,metrics[0],"left")
    b, bstates, bexpected = panel(right,right_identity,metrics[1],"right")
    # Exposure is observed independently on each side, including metric-less
    # rows. No epoch arithmetic and no planned horizon substituted for work.
    for side, rows, identity in (("left", astates, left_identity), ("right", bstates, right_identity)):
        previous_step = previous_samples = -1
        for point, row in rows.items():
            step, samples = row.get("optimizer_step"), row.get("sample_presentations")
            local = []
            if type(step) is not int or type(samples) is not int or step < 0 or samples < 0:
                local.append("MISSING_OR_INVALID:observed_exposure")
            else:
                if step <= previous_step or samples <= previous_samples:
                    local.append("MISMATCH:resume_duplicate_or_nonmonotonic_exposure")
                previous_step, previous_samples = step, samples
                batch, accumulation = identity.get("physical_batch_per_device"), identity.get("gradient_accumulation_steps")
                if type(batch) is not int or type(accumulation) is not int or batch < 1 or accumulation < 1:
                    local.append("INVALID:batch_or_accumulation")
                elif identity.get("drop_last") is True:
                    if samples != step * batch * accumulation:
                        local.append("MISMATCH:observed_exposure_vs_full_batches")
                else:
                    # Existing released screens drop tails. A caller's boolean
                    # cannot replace the owning session's actual batch ledger.
                    local.append("UNSUPPORTED:retained_tail_requires_bound_session_evidence")
                for key, observed in (("optimizer_steps", step), ("sample_presentations", samples)):
                    horizon = identity.get(key)
                    if type(horizon) is not int or observed > horizon:
                        local.append("MISMATCH:observed_exceeds_planned_" + key)
            if row.get("learning_rate") is None:
                local.append("MISSING:actual_lr")
            else:
                number(row["learning_rate"], "actual LR")
            for reason in local:
                history_issues.append({"coordinate": point, "side": side, "reason": reason})
    for point in sorted(astates.keys() | bstates.keys()):
        if point not in astates or point not in bstates:
            history_issues.append({"coordinate":point,"side":"pair","reason":"MISSING:paired_training_state"})
            continue
        l, r = astates[point], bstates[point]
        for field, code in (("learning_rate", "actual_lr"), ("optimizer_step", "optimizer_step"),
                            ("sample_presentations", "sample_presentations")):
            if lane == "TUNED_RECIPE" and field == "learning_rate":
                continue
            if l.get(field) != r.get(field):
                history_issues.append({"coordinate": point, "side": "pair", "reason": f"MISMATCH:{code}_at_{point}"})
    aligned = []
    global_reasons = list(reasons)
    for point in sorted(a.keys() & b.keys()):
        l, r = a[point], b[point]
        lv, rv = l.get("value", l.get(metrics[0])), r.get("value", r.get(metrics[1]))
        if lv is None or rv is None:
            continue
        local = global_reasons + [f"{issue['reason']}[{issue['side']}@{issue['coordinate']}]"
            for issue in history_issues if issue["coordinate"] <= point]
        if l.get("unit") is None or r.get("unit") is None:
            local.append("MISSING:metric_unit")
        elif l["unit"] != r["unit"] or l["unit"] != "eV":
            local.append("MISMATCH:metric_unit")
        for field in ("weights", "evaluation_role", "metric_kind"):
            if l.get(field) is None or r.get(field) is None:
                local.append("MISSING:metric_" + field)
            elif ((canonical_weight_state(l[field]) != canonical_weight_state(r[field]))
                  if field == "weights" else l[field] != r[field]):
                if not (lane == "TUNED_RECIPE" and field == "weights"):
                    local.append("MISMATCH:metric_" + field)
        for side, row, identity, expected in (("left",l,left_identity,aexpected),("right",r,right_identity,bexpected)):
            for field,value in expected.items():
                observed=row.get(field)
                if field == "weights": observed=canonical_weight_state(observed)
                if field == "target" and observed in {"gap","PCQM4Mv2 Gap"}: observed="Gap"
                if observed is None or value is None or observed != value:
                    local.append("MISMATCH:"+side+"_observed_"+field)
            try:
                digest(row.get("checkpoint_identity"), "checkpoint identity")
            except ValueError:
                local.append("MISSING_OR_INVALID:" + side + "_checkpoint_identity")
            checkpoints=identity.get("checkpoint_identities")
            frozen=checkpoints.get(str(row.get("optimizer_step"))) if type(checkpoints) is dict else None
            if frozen is None or frozen != row.get("checkpoint_identity"):
                local.append("MISMATCH:"+side+"_declared_checkpoint_identity")
        reasons.extend(local)
        aligned.append({axis: point, "left": lv, "right": rv,
            "left_observation": deepcopy(l), "right_observation": deepcopy(r),
            "left_minus_right_eV": None if local else lv - rv,
            "comparison_class": "CONTEXT_ONLY" if local else "MATCHED_PREFIX",
            "reasons": sorted(set(local))})
    if not aligned:
        reasons.append("MISSING:common_observed_coordinates")
    reasons.extend(issue["reason"] for issue in history_issues)
    return {"status": "NOT_STRICTLY_COMPARABLE" if reasons else "MATCHED_PREFIX",
            "comparison_class": "CONTEXT_ONLY" if reasons else "MATCHED_PREFIX",
            "comparison_type": comparison_type, "history_issues": history_issues,
            "metric_slots": list(metrics), "training_history_filter":"ALL_OBSERVED_METRICS",
            "strictness": ("INCOMPARABLE" if not aligned else "CONTEXT_ONLY") if reasons else
                {"DATA_DIVERSITY": "STRICT_WITH_DECLARED_SCALE_INTERVENTION", "MECHANISM": "STRICT_SAME_SCALE",
                 "TUNED_RECIPE": "SYSTEM_COMPARISON"}[lane],
            "reasons": sorted(set(reasons)), "axis": axis, "aligned": aligned,
            "extrapolated": False, "automatic_training_action": False}


def summarize_costs(events, *, expected_invocations):
    """Reuse the RML ledger; never infer missing allocation cost from loop wall."""
    from .research_memory.cost import build_cost_ledger
    from .research_memory.schemas import validate_cost_event
    seen, clean = {}, []
    for event in events:
        validate_cost_event(event)
        for measurement in event["measurement"].values():
            if measurement["value"] is not None:
                number(measurement["value"], "measured cost")
        prior = seen.get(event["cost_event_id"])
        if prior is not None:
            if prior != event:
                raise ValueError("Conflicting duplicate cost event")
            continue
        seen[event["cost_event_id"]] = event
        clean.append(event)
    observed = {event["attempt_id"] for event in clean}
    missing = sorted(set(expected_invocations) - observed)
    unplanned = sorted(observed - set(expected_invocations))
    if unplanned:
        raise ValueError("Unbound cost invocation: " + ",".join(unplanned))
    complete = bool(expected_invocations) and not missing and bool(clean) and all(
        event["measurement"]["device_hours"]["status"] == "measured" for event in clean)
    return {"ledger": build_cost_ledger(clean), "missing_invocations": missing,
            "allocation_coverage": "MEASURED" if complete else "MISSING_OR_UNKNOWN",
            "whole_run_allocated_device_hours": None,
            "cross_hardware_total": None, "rml_written": False}


def allocation_view(records, *, spec_identity, arm_ids):
    """Project existing validated allocation ledgers, including idle devices.

    Do not add a process wall interval on top of its enclosing allocation.
    Unknown provisioning remains outside the measured Python-window subtotal.
    """
    from .experiment_allocation import _prior_segments
    from .experiment_retention import _validate_ledger
    if not records:
        return {"status": "MISSING_OR_UNKNOWN", "measured_window_hours_by_hardware": None,
                "whole_run_hours_by_hardware": None, "cross_hardware_total": None}
    for index, record in enumerate(records):
        _validate_ledger(record, spec_identity, arm_ids, f"allocation[{index}]")
    segments = _prior_segments(records, spec_identity)
    known, unknown, windows = {}, False, set()
    for segment in segments:
        window = (segment.get("started_at"), tuple((d["device"], d["hardware"]) for d in segment.get("devices", [])))
        if window in windows:
            raise ValueError("Overlapping observations of one allocation invocation require reconciliation")
        windows.add(window)
        if segment.get("status") not in {"complete", "failed"} or segment.get("prior_unobserved_intervals"):
            unknown = True
        if segment.get("provisioning_before_python_seconds", {}).get("status") != "measured":
            unknown = True
        for device in segment.get("devices", []):
            key = device["hardware"]
            known[key] = known.get(key, 0.) + device["allocated_seconds"] / 3600
    return {"status": "MISSING_OR_UNKNOWN" if unknown else "MEASURED_WINDOWS",
            "measured_window_hours_by_hardware": known or None,
            "whole_run_hours_by_hardware": None, "cross_hardware_total": None,
            "scope": "deduplicated Python allocation windows, idle devices included; not total billed cost"}


def profile_view(profile):
    """No timer/profiler implementation: preserve existing measured phase scopes."""
    if profile.get("format") != "molgap-family-execution-profile-v1":
        raise ValueError("Expected owning shared profiler output")
    expected = ("loader_wait", "transfer", "layout_and_transfer", "forward", "backward", "optimizer",
                "development", "checkpoint_state", "checkpoint_publish")
    rows = {}
    for row in profile.get("phases_by_host_self_time", []):
        if row["name"] in rows:
            raise ValueError("Duplicate measured phase")
        number(row["host_self_seconds"], "phase wall")
        rows[row["name"]] = row["host_self_seconds"]
    return {"runtime": deepcopy(profile.get("runtime")),
            "phase_host_self_seconds": {name: rows.get(name) for name in expected},
            "device_events_observed": profile.get("device_events_observed"),
            "peak_allocated_bytes": profile.get("context_peak_allocated_bytes"),
            "peak_reserved_bytes": profile.get("context_peak_reserved_bytes"),
            "preprocessing_seconds": None, "startup_seconds": None,
            "inference_input_preparation_seconds": None,
            "allocated_device_hours": None,
            "scope": "instrumented host self durations, not native device cost or unprofiled throughput"}


def interpret_run(*, stage, fidelity, terminal, runtime, budget, roles,
                  strictly_comparable, seed_deltas=(), material_gain_eV=None,
                  seed_policy_approved=False, failure=None, execution_status=None,
                  terminal_evidence=()):
    """Review output only. No MAE argument on the 100K veto path by design."""
    if stage not in {"DIAGNOSTIC_100K", "PRIMARY_500K"}:
        raise ValueError("Unknown scale role")
    for name, value in (("terminal", terminal), ("strictly_comparable", strictly_comparable),
                        ("seed_policy_approved", seed_policy_approved)):
        if type(value) is not bool:
            raise TypeError(name + " must be an exact bool, not a status string")
    status = execution_status if execution_status is not None else "UNKNOWN"
    if status not in {"UNKNOWN", "RUNNING", "COMPLETE", "STOP_FOR_COST", "FAILED", "INCONCLUSIVE"}:
        raise ValueError("Unknown execution status")
    result = {"implementation_fidelity": fidelity, "runtime_mechanical": runtime,
              "scientific_status": "NOT_EVALUATED", "scale_transfer": "NOT_EVALUATED",
              "budget_decision": budget, "replay": "NOT_EVALUATED", "role_usage": roles,
              "promotion": "NOT_AUTHORIZED", "automatic_training_action": False,
              "veto_500k": False, "warnings": [], "execution_status": status,
              "terminal_acceptance": "MISSING", "observed_progress_only": True}
    verified_seeds, observed_gains = set(), {}
    if terminal_evidence:
        from .experiment_inspection import InspectionSnapshot, validate_snapshot
        from .experiment_family_workflow import _json, _safe_local
        for seed, candidate, reference in terminal_evidence:
            number(seed, "terminal paired seed", integer=True)
            if seed in verified_seeds:
                raise ValueError("Duplicate terminal seed")
            paired_reports = []
            for snapshot in (candidate, reference):
                if type(snapshot) is not InspectionSnapshot:
                    raise TypeError("Terminal evidence must use the owning InspectionSnapshot")
                metadata = snapshot.metadata()
                report = validate_snapshot(snapshot, output_dir=snapshot.output_dir,
                    context=metadata["context"], expected=metadata["expected"],
                    file_digest=file_digest, load_json=_json, safe_local=_safe_local)
                progress = report["observed"].get("progress", {})
                if (progress.get("optimizer_steps") != metadata["expected"]["optimizer_steps"] or
                        progress.get("sample_presentations") != metadata["expected"]["sample_presentations"] or
                        not {"trace", "selected_model", "resume", "contract"} <= set(snapshot.artifact_bindings())):
                    raise ValueError("Incomplete accepted endpoint/trace/checkpoint")
                contract = _json(snapshot.output_dir / snapshot.artifact_bindings()["contract"]["path"])
                if type(contract.get("seed")) is not int or contract["seed"] != seed:
                    raise ValueError("Terminal seed is not bound to the frozen output recipe")
                paired_reports.append(report)
            if candidate.digest == reference.digest:
                raise ValueError("Candidate cannot be its own reference")
            for key in ("source_idx_sha256", "target_sha256", "development_rows", "precision",
                        "optimizer_steps", "sample_presentations"):
                if candidate.metadata()["expected"].get(key) != reference.metadata()["expected"].get(key):
                    raise ValueError("Paired terminal evaluation/exposure identity differs: " + key)
            if paired_reports[0]["observed"].get("selected_weights") != paired_reports[1]["observed"].get("selected_weights"):
                raise ValueError("Paired terminal weight state differs")
            verified_seeds.add(seed)
            observed_gains[seed] = (paired_reports[1]["observed"]["development_mae_eV"] -
                                    paired_reports[0]["observed"]["development_mae_eV"])
        result["terminal_acceptance"] = "BOUND_OUTPUTS_VERIFIED"
    deltas = dict(seed_deltas)
    if len(deltas) != len(seed_deltas):
        raise ValueError("Duplicate seed is not independent evidence")
    for seed, value in deltas.items():
        number(seed, "paired seed", integer=True)
        if type(value) not in (int, float):
            raise TypeError("Seed effect must be a finite number")
        number(abs(value), "per-seed effect")
    values = list(deltas.values())
    if len(values) < 2:
        result["warnings"].append("SEED_VARIANCE_UNKNOWN; row bootstrap is conditional on frozen models")
    elif min(values) <= 0 <= max(values):
        result["warnings"].append("SEED_SIGN_DISAGREEMENT")
    if not verified_seeds:
        result["warnings"].append("MISSING_BOUND_TERMINAL_OUTPUTS; supplied effects are unverified")
    failures = {"INVALID_CODE", "NONFINITE", "CORRUPT_INPUT", "UNAUTHORIZED_ROLE", "CHECKPOINT_FAILURE", "DECLARED_COST_CEILING"}
    if failure is not None:
        if failure not in failures:
            raise ValueError("Quality/STQS/futility is not an admissible hard stop")
        result["scientific_status"] = "INCONCLUSIVE"
        result["budget_decision"] = "STOP_FOR_COST" if failure == "DECLARED_COST_CEILING" else "BLOCKED"
        return result
    if fidelity != "MODEL_FIDELITY_ACCEPTED":
        result["scientific_status"] = "IMPLEMENTATION_UNVERIFIED"
    elif stage == "DIAGNOSTIC_100K":
        result["scientific_status"] = "DIAGNOSTIC_ONLY"
        result["warnings"].append("SLOW_START_POSSIBLE; 100K MAE is not a 500K veto")
    elif (not terminal or status != "COMPLETE" or not verified_seeds or
          runtime != "MECHANICALLY_VERIFIED" or roles != "ROLE_VERIFIED" or
          budget != "MEASURED_WITHIN_CEILING" or not strictly_comparable):
        result["scientific_status"] = "INCONCLUSIVE"
    else:
        if set(deltas) != verified_seeds:
            raise ValueError("Seed effects are not bound to all accepted paired outputs")
        from math import isclose
        if any(not isclose(deltas[s], observed_gains[s], rel_tol=1e-8, abs_tol=1e-10) for s in deltas):
            raise ValueError("Seed effect differs from accepted predictions/selected endpoint")
        result["observed_progress_only"] = False
        result["per_seed_gain_eV"] = deltas
        result["across_seed_spread_eV"] = max(values) - min(values) if len(values) > 1 else None
        result["scientific_status"] = "INCONCLUSIVE"
        if len(values) < 2 or min(values) <= 0 <= max(values):
            pass
        elif material_gain_eV is None or not seed_policy_approved:
            result["warnings"].append("MISSING_PROSPECTIVE_MATERIAL_OR_SEED_POLICY")
        else:
            number(material_gain_eV, "material effect", minimum=0.000001)
            result["scientific_status"] = "EXPLORATORY_PRIMARY_POSITIVE" if min(values) > material_gain_eV else "INCONCLUSIVE"
            result["warnings"].append("Two seeds are exploratory, not a precise variance estimate; human review required")
    return result


def comparison_report(left, right, *, comparison_type, cohort_proof=None, metric="live_dev_metric"):
    """One observed report over existing owners, never a release or ranking gate.

    Runs carry identity/curve/source/checkpoint/cost/seed metadata. Unknowns
    remain null. Prefix eligibility is separate from complete report evidence.
    """
    aligned = align_curves(left.get("curve", []), right.get("curve", []),
        left_identity=left.get("identity", {}), right_identity=right.get("identity", {}),
        comparison_type=comparison_type, cohort_proof=cohort_proof, metric=metric)
    missing, runs = [], {}
    for side, run in (("left", left), ("right", right)):
        view = {key: deepcopy(run.get(key)) for key in
            ("model", "source_sha256", "trace_sha256", "identity", "checkpoint_sha256", "weights", "bn_state",
             "seed", "uncertainty", "curve", "cost")}
        for key in ("model", "source_sha256", "trace_sha256", "checkpoint_sha256", "weights", "bn_state", "seed", "cost"):
            if view[key] is None:
                missing.append(side + ":" + key)
        for key in ("source_sha256", "trace_sha256", "checkpoint_sha256"):
            if view[key] is not None:
                digest(view[key], key)
        if view["cost"] is not None and view["cost"].get("allocation_coverage") != "MEASURED":
            missing.append(side + ":complete_native_cost")
        from .comparison_readiness import canonical_weight_state
        identity = run.get("identity",{})
        for key in ("source_sha256","trace_sha256","seed"):
            if identity.get(key) is None or view[key] != identity[key]:
                missing.append(side+":envelope_"+key+"_binding")
        selected=identity.get("selected_optimizer_step")
        checkpoints=identity.get("checkpoint_identities",{})
        if type(selected) is not int or type(checkpoints) is not dict or checkpoints.get(str(selected)) != view["checkpoint_sha256"]:
            missing.append(side+":selected_checkpoint_binding")
        weight=canonical_weight_state(view["weights"])
        if weight is None or weight != canonical_weight_state(identity.get("evaluation_weight_source")):
            missing.append(side+":selected_weight_binding")
        if view["bn_state"] != identity.get("bn_calibration_identity"):
            missing.append(side+":bn_state_binding")
        runs[side] = view
    grade = aligned["strictness"]
    if missing and grade.startswith("STRICT") or missing and grade == "SYSTEM_COMPARISON":
        grade = "CONTEXT_ONLY"
    report = {"format": "molgap-cross-scale-comparison-report-v1", "comparison_type": comparison_type,
        "strictness": grade, "curve_comparison": aligned, "runs": runs,
        "missing_evidence": missing, "cohort_proof": cohort_proof.summary() if cohort_proof is not None else None,
        "limitations": ["Observed points only; no terminal projection", "Row bootstrap is not training variance",
                        "Report grades do not grant V5 STRICT_CAUSAL or role/training authority"],
        "next_gates": ["Bound complete outputs, role history, native cost and separate prospective authority"],
        "early_stop_status": "NOT_RELEASED", "automatic_training_action": False,
        "training_authority": "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING"}
    return report


def write_comparison_report(output_dir, report):
    import json
    from .common.io import publish_immutable_bytes
    from .research_memory.trace import json_bytes
    output = Path(output_dir).absolute()
    publish_immutable_bytes(output / "comparison.json", json_bytes(report))
    lines = ["# Cross-scale comparison", "", f"Question: {report['comparison_type']}", "",
             f"Grade: {report['strictness']}", "", "Automatic early stop: NOT_RELEASED.", "",
             "Missing evidence: " + (", ".join(report["missing_evidence"]) or "none declared"), "",
             "## Observed evidence", "", "```json", json.dumps(report, indent=2, allow_nan=False), "```", ""]
    publish_immutable_bytes(output / "comparison.md", "\n".join(lines).encode())
    return {name: {"path": str(output / name), "sha256": file_digest(output / name)}
            for name in ("comparison.json", "comparison.md")}
