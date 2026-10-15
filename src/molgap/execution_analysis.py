"""Shared saved-output execution comparison; family owners retain native gates."""
from collections import Counter
from pathlib import Path
import json
import math
import statistics


def speed(old, new):
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in (old, new)):
        raise ValueError("Matched positive finite durations are required")
    return {"speedup_factor": old / new, "wall_reduction_percent": 100 * (old - new) / old}


def paired_spans(original, optimized, *, total_passes=40):
    if type(total_passes) is not int or total_passes < 2:
        raise ValueError("A complete multi-pass horizon is required")
    def keyed(rows):
        result = {}
        for row in rows:
            key = row["start_relative_update"]
            if (type(key) is not int or not 781 <= key <= (total_passes - 1) * 781 or key % 781 or key in result
                    or row["optimizer_updates"] != 781 or row["sample_presentations"] != 99968):
                raise ValueError("Expected unique complete fixed100K training spans")
            speed(row["wall_seconds"], row["wall_seconds"])
            result[key] = row
        return result
    old, new = keyed(original), keyed(optimized)
    if set(old) != set(new) or not old:
        raise ValueError("Capture-off spans must cover the same complete passes")
    rows = [{"pass": key // 781 + 1, "original_seconds": old[key]["wall_seconds"],
        "optimized_seconds": new[key]["wall_seconds"],
        **speed(old[key]["wall_seconds"], new[key]["wall_seconds"])} for key in sorted(old)]
    totals = [sum(item["wall_seconds"] for item in arm.values()) for arm in (old, new)]
    return {"scope": "matched passes after Kineto capture; passive observer wrappers remain; validation/publication excluded",
        "passes": rows, "original_graphs_per_second": len(old) * 99968 / totals[0],
        "optimized_graphs_per_second": len(new) * 99968 / totals[1],
        "median_paired_speedup_factor": statistics.median(row["speedup_factor"] for row in rows),
        "aggregate": speed(*totals)}


def trace_observations(path):
    events = json.loads(Path(path).read_bytes())["traceEvents"]
    calls, runtime_calls, intervals = Counter(), Counter(), []
    for event in events:
        if event.get("ph") != "X":
            continue
        categories = set(event.get("cat", "").split(","))
        if "cpu_op" in categories and event.get("name", "").startswith("aten::"):
            calls[event["name"]] += 1
        if "cuda_runtime" in categories:
            runtime_calls[event["name"]] += 1
        if categories & {"kernel", "gpu_memcpy", "gpu_memset"}:
            start, duration = event["ts"], event["dur"]
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in (start, duration)) or duration < 0:
                raise ValueError("Invalid device trace duration")
            intervals.append((start, start + duration))
    union, end = 0., None
    for start, stop in sorted(intervals):
        union += stop - start if end is None or start > end else max(0., stop - end)
        end = stop if end is None else max(end, stop)
    return {"aten_dispatch_calls": dict(sorted(calls.items())),
        "cuda_runtime_calls": dict(sorted(runtime_calls.items())),
        "device_activity_union_us": union if intervals else None,
        "device_activity_events": len(intervals),
        "scope": "single-worker capture only; overlapping GPU intervals merged; not allocation cost or whole-run utilization"}


def prediction_comparison(original, optimized):
    import torch
    if (not torch.equal(original["source_idx"], optimized["source_idx"]) or
            not torch.equal(original["target_eV"], optimized["target_eV"])):
        raise ValueError("Comparison row/target alignment differs")
    target, old, new = (value.detach().cpu().double() for value in
        (original["target_eV"], original["prediction_eV"], optimized["prediction_eV"]))
    if target.shape != old.shape or target.shape != new.shape or target.ndim != 1 or not target.numel():
        raise ValueError("Comparison prediction shape differs")
    if any(not bool(torch.isfinite(value).all()) for value in (target, old, new)):
        raise ValueError("Nonfinite comparison payload")
    first, second = float((old - target).abs().mean()), float((new - target).abs().mean())
    return {"rows": target.numel(), "original_mae_eV_float64": first, "optimized_mae_eV_float64": second,
        "optimized_minus_original_mae_eV": second - first,
        "maximum_prediction_delta_eV": float((new - old).abs().max()),
        "mean_prediction_delta_eV": float((new - old).abs().mean()),
        "prediction_values_equal": torch.equal(original["prediction_eV"], optimized["prediction_eV"])}


def paired_expectations(original, optimized):
    """Compare scientific axes across native epoch and shared cursor schemas."""
    def normalize(record):
        common = {key: record[key] for key in ("optimizer_steps", "sample_presentations",
            "development_rows", "precision", "source_idx_sha256", "target_sha256")}
        if "epochs" in record:
            common["passes"] = record["epochs"]
        else:
            if record["next_batch"] != 0:
                raise ValueError("Comparison requires complete training passes")
            common["passes"] = record["completed_passes"]
        return common
    old, new = normalize(original), normalize(optimized)
    if old != new:
        raise ValueError("Paired exposure/target expectations differ")
    return old


def analyze_execution(*, package, receipt, output_root, comparison, native_validator,
                      report_format, scientific_subdirectories=None):
    """Reuse both owning inspectors and the sealed execution/allocation manifest."""
    import torch
    from .experiment_spec import ExperimentSpec
    from .experiment_package import verify_experiment_source_package, read_packaged_text
    from .experiment_family_workflow import RunContext, inspect_output, _artifact_path, _json, _check_context
    from .experiment_retention import validate_execution_retention
    from .screen_policy import canonical_fingerprint
    from .training_reproducibility import sha256_file
    package, output_root = Path(package).absolute(), Path(output_root).absolute()
    manifest = verify_experiment_source_package(package)
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    base = comparison.comparison_base(spec)
    scientific_subdirectories = scientific_subdirectories or {}
    retention = validate_execution_retention(output_root, spec, package_identity=manifest["package_identity"],
        source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"])
    retention["manifest_path"] = retention["manifest_path"].relative_to(output_root).as_posix()
    execution = _json(_artifact_path(output_root, "execution_report.json"))
    if execution["status"] != "complete":
        raise ValueError("Incomplete execution cannot produce a terminal speed claim")
    phases = {row["phase"]: row for row in execution["phases"]}
    if set(phases) != {"preflight", "train"} or any(row["status"] != "complete" for row in phases.values()):
        raise ValueError("Both native barriers and formal workers must complete")
    if any(len(row["workers"]) != 2 or
           {worker["device"] for worker in row["workers"]} != {0, 1} or
           any(worker["returncode"] != 0 for worker in row["workers"])
           for row in phases.values()):
        raise ValueError("Expected exactly two independent workers per phase")
    workers = {phase: {row["arm_id"]: row["worker_wall_seconds"] for row in record["workers"]}
        for phase, record in phases.items()}
    if any(set(rows) != {"original", "optimized"} for rows in workers.values()):
        raise ValueError("Missing matched phase worker wall measurements")
    profiles, payloads, arms, requirements = {}, {}, {}, {}
    for arm_id in ("original", "optimized"):
        context = RunContext.from_launch(spec, Path(receipt), package,
            expected_package_identity=manifest["package_identity"], arm_id=arm_id)
        native_root = _artifact_path(output_root, arm_id + "/execution_profile_receipt.json").parent
        relative = scientific_subdirectories.get(arm_id, "")
        prefix = arm_id + ("/" + relative if relative else "")
        root = _artifact_path(output_root, prefix + "/output_manifest.json").parent
        record = json.loads(read_packaged_text(package, base + "/" + arm_id + "/contract.json"))
        requirements[arm_id] = record["acceptance_requirements"]
        inspected = inspect_output(root, context=context, expected=record["acceptance_requirements"])
        if inspected["status"] != "MECHANICALLY_VERIFIED":
            raise ValueError("Owning inspection failed: " + arm_id + ": " + repr(inspected.get("blockers")))
        observed = _json(_artifact_path(native_root, "execution_profile_receipt.json"))
        _check_context(observed["context"], context)
        profile_path = _artifact_path(native_root, observed["report_path"])
        if sha256_file(profile_path) != observed["report_sha256"]:
            raise ValueError("Profile report changed")
        profile = _json(profile_path)
        trace = _artifact_path(native_root, (profile_path.parent.relative_to(native_root) / "trace.json").as_posix())
        if (observed.get("capture_status") != "COMPLETE" or
                profile["status"] != "COMPLETE" or profile["captured_updates"] != 16 or
                profile["device_events_observed"] is not True or profile["observed_updates"] != record["acceptance_requirements"]["optimizer_steps"] or
                profile["config"] != record["profiling"] or
                sha256_file(trace) != profile["trace_sha256"] or profile["trace_sha256"] != observed["trace_sha256"]):
            raise ValueError("Incomplete or changed native operator capture")
        if arm_id == "original":
            identity = canonical_fingerprint(record)
        else:
            from .family_training_runner import _recipe_configuration
            identity = _recipe_configuration(record).identity
        if profile["recipe_identity"] != identity:
            raise ValueError("Profile recipe identity differs")
        certificate = _json(_artifact_path(native_root, "runtime_certificate.json"))
        native_validator(arm_id=arm_id, root=native_root, context=context, record=record,
                         certificate=certificate, identity=identity)
        profiles[arm_id] = profile
        payloads[arm_id] = torch.load(_artifact_path(root, "development_predictions.pt"), map_location="cpu", weights_only=True)
        selected = torch.load(_artifact_path(root, "selected_model.pt"), map_location="cpu", weights_only=True)
        selected_pass = selected["epoch"] if arm_id == "original" else selected["epoch_or_pass"]
        arms[arm_id] = {"mechanical_inspection": inspected, "selected_pass": selected_pass,
            "native_certificate_sha256": sha256_file(native_root / "runtime_certificate.json"),
            "preflight_worker_wall_seconds": workers["preflight"][arm_id],
            "train_worker_wall_seconds": workers["train"][arm_id],
            "profile_report_sha256": observed["report_sha256"], "profile": profile,
            "trace_observations": trace_observations(trace)}
    for key in ("torch", "cuda", "hip", "device_name"):
        if profiles["original"]["runtime"][key] != profiles["optimized"]["runtime"][key]:
            raise ValueError("Paired native runtime differs: " + key)
    matched = paired_expectations(requirements["original"], requirements["optimized"])
    return {"format": report_format, "status": "EXECUTION_COMPARISON_VERIFIED",
        "spec_identity": spec.identity, "source_commit": manifest["source_commit"],
        "logical_run_id": spec.to_dict()["logical_run_id"], "comparison_base": base,
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC",
        "parent_timed_train_workers": speed(workers["train"]["original"], workers["train"]["optimized"]),
        "capture_off_spans": paired_spans(profiles["original"]["capture_off_training_spans"], profiles["optimized"]["capture_off_training_spans"], total_passes=matched["passes"]),
        "numerical_comparison": prediction_comparison(payloads["original"], payloads["optimized"]),
        "execution_retention": retention, "arms": arms,
        "limitations": ["One seed and one device assignment; correlated pass timings are not independent replicates.",
            "Worker wall includes loading, evaluation, persistence and inspection; internal trainer cost windows differ.",
            "Capture-off spans retain passive wrappers; they are not a pristine profiler-free control.",
            "Host/self-device operator timings and the capture activity union do not measure allocation billing.",
            "Train-update range boundaries differ between legacy and shared owners; compare matching subphases and trace operators.",
            "Legacy CPU RNG ownership and metric reduction precision differ; saved aligned endpoints determine observed drift."]}
