"""Saved-output K1 execution comparison; no model construction or inference."""
from collections import Counter
from pathlib import Path
import json
import math
import statistics


def speed(old, new):
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in (old, new)):
        raise ValueError("Matched positive finite durations are required")
    return {"speedup_factor": old / new, "wall_reduction_percent": 100 * (old - new) / old}


def paired_spans(original, optimized):
    def keyed(rows):
        result = {}
        for row in rows:
            key = row["start_relative_update"]
            if (type(key) is not int or not 781 <= key <= 39 * 781 or key % 781 or key in result
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
    calls, intervals = Counter(), []
    for event in events:
        if event.get("ph") != "X":
            continue
        categories = set(event.get("cat", "").split(","))
        if "cpu_op" in categories and event.get("name", "").startswith("aten::"):
            calls[event["name"]] += 1
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


def analyze(*, package, receipt, output_root):
    """Reuse both owning inspectors and the sealed execution/allocation manifest."""
    import torch
    from .experiment_spec import ExperimentSpec
    from .experiment_package import verify_experiment_source_package, read_packaged_text
    from .experiment_family_workflow import RunContext, inspect_output, _artifact_path, _json, _check_context
    from .experiment_retention import validate_execution_retention
    from .k1_infrastructure_comparison import comparison_base, is_comparison
    from .screen_policy import canonical_fingerprint
    from .training_reproducibility import sha256_file
    package, output_root = Path(package).absolute(), Path(output_root).absolute()
    manifest = verify_experiment_source_package(package)
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    if not is_comparison(spec):
        raise ValueError("Not the frozen K1 execution comparison")
    base = comparison_base(spec)
    retention = validate_execution_retention(output_root, spec, package_identity=manifest["package_identity"],
        source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"])
    retention["manifest_path"] = retention["manifest_path"].relative_to(output_root).as_posix()
    execution = _json(_artifact_path(output_root, "execution_report.json"))
    if execution["status"] != "complete":
        raise ValueError("Incomplete execution cannot produce a terminal speed claim")
    phases = {row["phase"]: row for row in execution["phases"]}
    if set(phases) != {"preflight", "train"} or any(row["status"] != "complete" for row in phases.values()):
        raise ValueError("Both native barriers and formal workers must complete")
    if any(len(row["workers"]) != 2 for row in phases.values()):
        raise ValueError("Expected exactly two independent workers per phase")
    workers = {phase: {row["arm_id"]: row["worker_wall_seconds"] for row in record["workers"]}
        for phase, record in phases.items()}
    if any(set(rows) != {"original", "optimized"} for rows in workers.values()):
        raise ValueError("Missing matched phase worker wall measurements")
    profiles, payloads, arms = {}, {}, {}
    for arm_id in ("original", "optimized"):
        context = RunContext.from_launch(spec, Path(receipt), package,
            expected_package_identity=manifest["package_identity"], arm_id=arm_id)
        root = _artifact_path(output_root, arm_id + "/output_manifest.json").parent
        record = json.loads(read_packaged_text(package, base + "/" + arm_id + "/contract.json"))
        inspected = inspect_output(root, context=context, expected=record["acceptance_requirements"])
        if inspected["status"] != "MECHANICALLY_VERIFIED":
            raise ValueError("Owning inspection failed: " + arm_id + ": " + repr(inspected.get("blockers")))
        observed = _json(_artifact_path(root, "execution_profile_receipt.json"))
        _check_context(observed["context"], context)
        profile_path = _artifact_path(root, observed["report_path"])
        if sha256_file(profile_path) != observed["report_sha256"]:
            raise ValueError("Profile report changed")
        profile = _json(profile_path)
        trace = _artifact_path(root, (profile_path.parent.relative_to(root) / "trace.json").as_posix())
        if (profile["status"] != "COMPLETE" or profile["captured_updates"] != 16 or
                profile["device_events_observed"] is not True or profile["observed_updates"] != 31240 or
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
        certificate = _json(_artifact_path(root, "runtime_certificate.json"))
        provenance = _json(_artifact_path(root, "runtime_provenance.json"))
        _check_context(provenance["context"], context)
        from .family_training_runner import _verify_profile_proof
        if arm_id == "original":
            from types import SimpleNamespace
            from .k1_screen_training import validate_runtime_preflight
            validate_runtime_preflight(root, provenance)
            fingerprints = certificate.get("profiled_plain_state_fingerprints", [])
            if len(fingerprints) != 2 or len(set(fingerprints)) != 1:
                raise ValueError("Legacy profiled/plain state qualification differs")
            _verify_profile_proof(root, SimpleNamespace(identity=identity), certificate.get("profile_proof"), required=True)
        else:
            from .family_training_runner import _verify_preflight, _recipe_configuration
            _verify_preflight(root, _recipe_configuration(record), provenance, profiling_required=True)
        profiles[arm_id] = profile
        payloads[arm_id] = torch.load(_artifact_path(root, "development_predictions.pt"), map_location="cpu", weights_only=True)
        selected = torch.load(_artifact_path(root, "selected_model.pt"), map_location="cpu", weights_only=True)
        selected_pass = selected["epoch"] if arm_id == "original" else selected["epoch_or_pass"]
        arms[arm_id] = {"mechanical_inspection": inspected, "selected_pass": selected_pass,
            "native_certificate_sha256": sha256_file(root / "runtime_certificate.json"),
            "preflight_worker_wall_seconds": workers["preflight"][arm_id],
            "train_worker_wall_seconds": workers["train"][arm_id],
            "profile_report_sha256": observed["report_sha256"], "profile": profile,
            "trace_observations": trace_observations(trace)}
    for key in ("torch", "cuda", "hip", "device_name"):
        if profiles["original"]["runtime"][key] != profiles["optimized"]["runtime"][key]:
            raise ValueError("Paired native runtime differs: " + key)
    return {"format": "molgap-k1-execution-comparison-v1", "status": "EXECUTION_COMPARISON_VERIFIED",
        "spec_identity": spec.identity, "source_commit": manifest["source_commit"],
        "logical_run_id": spec.to_dict()["logical_run_id"], "comparison_base": base,
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC",
        "parent_timed_train_workers": speed(workers["train"]["original"], workers["train"]["optimized"]),
        "capture_off_spans": paired_spans(profiles["original"]["capture_off_training_spans"], profiles["optimized"]["capture_off_training_spans"]),
        "numerical_comparison": prediction_comparison(payloads["original"], payloads["optimized"]),
        "execution_retention": retention, "arms": arms,
        "limitations": ["One seed and one device assignment; correlated pass timings are not independent replicates.",
            "Worker wall includes loading, evaluation, persistence and inspection; internal trainer cost windows differ.",
            "Capture-off spans retain passive wrappers; they are not a pristine profiler-free control.",
            "Host/self-device operator timings and the capture activity union do not measure allocation billing.",
            "Train-update range boundaries differ between legacy and shared owners; compare matching subphases and trace operators.",
            "Legacy CPU RNG ownership and metric reduction precision differ; saved aligned endpoints determine observed drift."]}


def main():
    import argparse
    from .research_memory.trace import atomic_write, json_bytes
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for field in ("package", "receipt", "output-root", "report"):
        parser.add_argument("--" + field, required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Use a fresh retained analysis report")
    result = analyze(package=args.package, receipt=args.receipt, output_root=args.output_root)
    atomic_write(args.report, json_bytes(result))
    print(json.dumps({key: result[key] for key in ("status", "parent_timed_train_workers", "numerical_comparison")}))


if __name__ == "__main__":
    main()
