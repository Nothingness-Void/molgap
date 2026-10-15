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
            if value is None:
                continue
            if definition["target"] not in {"Gap", "gap", "PCQM4Mv2 Gap"} or definition["direction"] != "minimize":
                raise ValueError("Unknown target/direction")
            if definition["unit"] not in {"eV", "normalized-Gap"}:
                raise ValueError("Unknown metric unit")
            role = definition["role_identity"]
            if (not (role.startswith("internal_") or role.endswith(":train") or role.endswith(":development")) or
                    any(term in role.lower() for term in ("official", "test", "ood", "p8", "external", "shadow", "protected"))):
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
                "evaluation_role": role})
    return {"rows": rows, "comparability_identity": deepcopy(manifest["comparability_identity"]),
            "backtest_eligibility": deepcopy(manifest["backtest_eligibility"]),
            "trace_sha256": trace_binding["sha256"], "scientific_acceptance": "NOT_REASSESSED",
            "rml_written": False}


def align_curves(left, right, *, left_identity, right_identity, metric="live_dev_metric",
                 axis="optimizer_step", lane="MECHANISM", nested_cohorts_verified=False):
    """Intersect actually observed coordinates; no interpolation or extrapolation.

    Explicit identity uses existing V5 names. Data-diversity matching relaxes
    only train membership/exposure, and requires an independent nested-role proof.
    It never grants V5 STRICT_CAUSAL or prospective admission.
    """
    if axis not in {"optimizer_step", "sample_presentations"}:
        raise ValueError("Epoch/pass-only alignment is forbidden")
    if lane not in {"MECHANISM", "DATA_DIVERSITY", "TUNED_RECIPE"}:
        raise ValueError("Unknown comparison lane")
    fields = set(STRICT_IDENTITY_FIELDS) | {"sampler_identity", "schedule_axis", "target_unit", "evaluation_row_sha256",
        "evaluation_target_sha256", "evaluation_row_count", "bn_calibration_identity"}
    ignored = {"optimizer_steps", "sample_presentations"}
    if lane == "MECHANISM":
        ignored.add("architecture_config_identity")
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
        elif left_identity[field] != right_identity[field]:
            reasons.append("MISMATCH:" + field)
    if lane == "DATA_DIVERSITY" and not nested_cohorts_verified:
        reasons.append("MISSING:nested_cohorts_and_shared_disjoint_role")
    if lane == "DATA_DIVERSITY" and (left_identity.get("schedule_axis") != "optimizer_step" or right_identity.get("schedule_axis") != "optimizer_step"):
        reasons.append("MISMATCH:step_indexed_schedule_required")
    if lane == "TUNED_RECIPE":
        reasons.append("TUNED_SYSTEM_NOT_ARCHITECTURE_CAUSAL")

    def panel(rows):
        result, last = {}, -1
        for row in rows:
            if row.get("metric", metric) != metric:
                continue
            point = row.get(axis)
            if point is None:
                continue
            number(point, axis, integer=True)
            if point <= last:
                raise ValueError("Duplicate or non-monotonic observed coordinate")
            last = point
            value = row.get("value", row.get(metric))
            if value is None:
                continue
            number(value, "MAE")
            result[point] = row
        return result

    a, b = panel(left), panel(right)
    aligned = []
    for point in sorted(a.keys() & b.keys()):
        l, r = a[point], b[point]
        if l.get("learning_rate") is None or r.get("learning_rate") is None:
            reasons.append(f"MISSING:actual_lr_at_{point}")
        elif l["learning_rate"] != r["learning_rate"]:
            reasons.append(f"MISMATCH:actual_lr_at_{point}")
        if l.get("unit") is None or r.get("unit") is None:
            reasons.append("MISSING:metric_unit")
        elif l["unit"] != r["unit"] or l["unit"] != "eV":
            reasons.append("MISMATCH:metric_unit")
        for field in ("weights", "evaluation_role", "metric_kind"):
            if l.get(field) is None or r.get(field) is None:
                reasons.append("MISSING:metric_" + field)
            elif l[field] != r[field]:
                reasons.append("MISMATCH:metric_" + field)
        lv, rv = l.get("value", l.get(metric)), r.get("value", r.get(metric))
        aligned.append({axis: point, "left": lv, "right": rv, "left_minus_right_eV": lv - rv})
    if not aligned:
        reasons.append("MISSING:common_observed_coordinates")
    if reasons:
        for row in aligned:
            row["left_minus_right_eV"] = None  # Context is displayed, never a cross-role causal delta.
    return {"status": "NOT_STRICTLY_COMPARABLE" if reasons else "MATCHED_PREFIX",
            "comparison_class": "CONTEXT_ONLY" if reasons else "MATCHED_PREFIX",
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
                  seed_policy_approved=False, failure=None):
    """Review output only. No MAE argument on the 100K veto path by design."""
    if stage not in {"DIAGNOSTIC_100K", "PRIMARY_500K"}:
        raise ValueError("Unknown scale role")
    result = {"implementation_fidelity": fidelity, "runtime_mechanical": runtime,
              "scientific_status": "NOT_EVALUATED", "scale_transfer": "NOT_EVALUATED",
              "budget_decision": budget, "replay": "NOT_EVALUATED", "role_usage": roles,
              "promotion": "NOT_AUTHORIZED", "automatic_training_action": False,
              "veto_500k": False, "warnings": []}
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
    elif (not terminal or runtime != "MECHANICALLY_VERIFIED" or roles != "ROLE_VERIFIED" or
          budget != "MEASURED_WITHIN_CEILING" or not strictly_comparable):
        result["scientific_status"] = "INCONCLUSIVE"
    else:
        deltas = dict(seed_deltas)
        if len(deltas) != len(seed_deltas):
            raise ValueError("Duplicate seed is not independent evidence")
        for value in deltas.values():
            number(abs(value), "per-seed effect")
        values = list(deltas.values())
        for seed in deltas:
            number(seed, "paired seed", integer=True)
        result["per_seed_gain_eV"] = deltas
        result["across_seed_spread_eV"] = max(values) - min(values) if len(values) > 1 else None
        result["scientific_status"] = "INCONCLUSIVE"
        if len(values) < 2:
            result["warnings"].append("SEED_VARIANCE_UNKNOWN; row bootstrap is conditional on frozen models")
        elif min(values) <= 0 <= max(values):
            result["warnings"].append("SEED_SIGN_DISAGREEMENT")
        elif material_gain_eV is None or not seed_policy_approved:
            result["warnings"].append("MISSING_PROSPECTIVE_MATERIAL_OR_SEED_POLICY")
        else:
            number(material_gain_eV, "material effect", minimum=0.000001)
            result["scientific_status"] = "EXPLORATORY_PRIMARY_POSITIVE" if min(values) > material_gain_eV else "INCONCLUSIVE"
            result["warnings"].append("Two seeds are exploratory, not a precise variance estimate; human review required")
    return result
