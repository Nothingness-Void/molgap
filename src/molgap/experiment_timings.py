"""Read-only reform timing evidence; no SDK, credentials, or journal writes.

``report_experiment_timings(run_directory)`` reads only the named run's local
preparation observations and allowlisted metadata. Journal ``timings`` are
active invocation accumulators, not elapsed wall time across invocations.
Quantiles describe retained samples, never population latency or qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import stat

from .common.io import _safe_local
from .common.spec import _unique_object

__all__ = ["report_experiment_timings", "main"]

_FORMAT = "molgap-local-preparation-observation-v1"
MAX_JSON_BYTES = 1024 * 1024
MAX_OBSERVATION_ENTRIES = 1000
_PHASES = {
    "NEW": ("submission_prechecks_active", "Local admission/quota/status work during active invocations."),
    "SOURCE_SUBMITTING": ("source_publication_request_active", "Publication request including upload/API overhead; not isolated transfer time."),
    "SOURCE_PUBLISHED": ("source_ready_checks_active", "Active readiness/verification work; excludes offline waiting."),
    "SOURCE_PENDING": ("source_ready_checks_active", "Active resumed readiness, download, release and quota checks; excludes offline waiting."),
    "SOURCE_UNKNOWN": ("source_reconciliation_active", "Active reconciliation after an uncertain publication outcome."),
    "KERNEL_SUBMITTING": ("kernel_post_active", "Push wrapper, before-POST checks and response persistence; not isolated HTTP POST time."),
    "SUBMIT_UNKNOWN": ("kernel_reconciliation_active", "Active reconciliation; not the original lost response latency."),
    "KERNEL_ACCEPTED": ("receipt_binding_active", "Source/receipt binding plus scheduler/startup observations; not isolated receipt writing."),
    "BOUND": ("bound_reobservation_active", "Repeated bound-run checks; not a new submission or queue sample."),
}
_UNMEASURED = {
    "upload_transfer": "Publication request intervals do not isolate upload transfer time.",
    "source_ready_wait": "No retained publication-to-ready endpoints; active phase timers exclude offline waiting.",
    "receipt_end_to_end": "No retained code-ready-to-receipt endpoints; phase sums are not end-to-end latency.",
    "training_queue": "A scheduler status snapshot does not measure queue entry/start endpoints.",
}


def _float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("nonfinite_json_number")
    return value


def _constant(_raw):
    raise ValueError("nonfinite_json_number")


def _read(path, diagnostics):
    # Check every ancestor before even testing existence; no recursive discovery
    # or links embedded in metadata can extend this reader's allowlist.
    _safe_local(path)
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not_regular_file")
        if info.st_size > MAX_JSON_BYTES:
            diagnostics.append({"source": str(path), "reason": "json_size_limit_exceeded"})
            return None, "json_size_limit_exceeded"
        with path.open("rb") as stream:
            payload = stream.read(MAX_JSON_BYTES + 1)
        if len(payload) > MAX_JSON_BYTES:
            diagnostics.append({"source": str(path), "reason": "json_size_limit_exceeded"})
            return None, "json_size_limit_exceeded"
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_float=_float, parse_constant=_constant)
        if type(value) is not dict:
            raise ValueError("not_json_object")
        return value, None
    except FileNotFoundError:
        reason = "missing_file"
    except (OSError, UnicodeError):
        reason = "unreadable_file"
    except (ValueError, RecursionError):
        reason = "invalid_json_or_object"
    diagnostics.append({"source": str(path), "reason": reason})
    return None, reason


def _digest(value, lengths=(64,)):
    return type(value) is str and len(value) in lengths and re.fullmatch(r"[0-9a-f]+", value) is not None


def _token(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_]{1,64}", value) is not None


def _timestamp(value):
    if type(value) is not str:
        return False
    try:
        stamp = datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
        return stamp.utcoffset() == timedelta(0)
    except (ValueError, OverflowError):
        return False


def _seconds(value):
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _identities(value):
    return {key: value.get(key) if _digest(value.get(key), (40, 64) if key == "source_commit" else (64,)) else None
            for key in ("spec_identity", "package_identity", "source_commit")}


def _sample(metric, source, *, seconds=None, missing_reason=None, qualification,
            identity=None, reasons=(), **fields):
    return {"metric": metric, "source": str(source), "seconds": seconds,
            "missing_reason": missing_reason, "qualification": qualification,
            "qualification_reasons": list(reasons),
            "identity_matches_retained_metadata": False,
            "identity": identity or _identities({}), **fields}


def _observation(path, value, expected):
    identity = _identities(value)
    reasons = [key + "_unavailable" for key, item in identity.items() if item is None]
    reasons.extend(key + "_retained_binding_unavailable" for key, item in expected.items() if item is None)
    reasons.extend(key + "_binding_mismatch" for key, item in identity.items()
                   if item is not None and expected.get(key) is not None and item != expected[key])
    required = {"format", "observed_at", "status", "spec_identity", "package_identity",
                "reused_preparation", "local_call_seconds"}
    valid = (required <= value.keys() <= required | {"source_commit"}
             and value.get("format") == _FORMAT and _timestamp(value.get("observed_at"))
             and _digest(value.get("spec_identity"))
             and (value.get("package_identity") is None or _digest(value["package_identity"]))
             and (value.get("source_commit") is None or _digest(value["source_commit"], (40, 64)))
             and type(value.get("reused_preparation")) is bool)
    seconds = _seconds(value.get("local_call_seconds"))
    missing = ("invalid_preparation_observation" if not valid else
               "not_successful_preparation" if value.get("status") != "PREPARED_FOR_PLATFORM" else
               "invalid_duration" if seconds is None else None)
    return _sample("local_preparation", path, seconds=seconds if missing is None else None,
                   missing_reason=missing, qualification="Local processing up to observation write; excludes observation publication and platform work. Warm environment/data/permission applicability is not established.",
                   identity=identity, reasons=reasons,
                   identity_matches_retained_metadata=not reasons and missing is None,
                   sample_kind="local_call", status=value.get("status") if _token(value.get("status")) else None,
                   observed_at=value.get("observed_at") if _timestamp(value.get("observed_at")) else None,
                   reused_preparation=value.get("reused_preparation") if type(value.get("reused_preparation")) is bool else None)


def _summaries(samples):
    groups = {}
    for sample in samples:
        # Distinct phases and frozen identities are not interchangeable trials.
        key = (sample["metric"], sample.get("phase"), sample.get("reused_preparation"),
               *sample["identity"].values())
        groups.setdefault(key, []).append(sample)
    result = []
    for group in groups.values():
        first = group[0]
        values = sorted(s["seconds"] for s in group if s["seconds"] is not None)
        def quantile(q):
            if not values:
                return None
            index = (len(values) - 1) * q
            lower = int(index)
            return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)
        result.append({"metric": first["metric"], "phase": first.get("phase"),
                       "reused_preparation": first.get("reused_preparation"), "identity": first["identity"],
                       "sample_count": len(values), "missing_sample_count": len(group) - len(values),
                       "metadata_matching_sample_count": sum(s.get("identity_matches_retained_metadata", False) for s in group if s["seconds"] is not None),
                       "p50_seconds": quantile(.5), "p95_seconds": quantile(.95),
                       "interpretation": "Descriptive linear interpolation of retained samples only; insufficient evidence for stable population p50/p95 or target qualification. Phase accumulators are not individual invocation samples."})
    return result


def report_experiment_timings(run_directory):
    """Return JSON-compatible raw samples, descriptive summaries and evidence gaps.

    Missing/corrupt metadata produces explicit null durations and diagnostics.
    Unsafe paths (including any symlink/reparse ancestor) raise ``ValueError``.
    No inferred totals, timestamps from mtimes, credentials, or remote requests.
    """
    if ".." in Path(run_directory).parts:
        raise ValueError("Run directory traversal is forbidden")
    run = Path(run_directory).absolute()
    _safe_local(run)
    if not run.is_dir() or run == Path(run.anchor):
        raise ValueError("Expected an existing explicit run directory")
    diagnostics, samples = [], []
    binding, _ = _read(run / "inputs/input_binding.json", diagnostics)
    manifest, _ = _read(run / "prepared/package/package_manifest.json", diagnostics)
    expected = _identities(binding or {})
    manifest_identity = _identities(manifest or {})
    conflicts = []
    for key, item in manifest_identity.items():
        if item is not None and expected[key] is not None and item != expected[key]:
            conflicts.append(key + "_binding_mismatch")
        elif item is not None:
            expected[key] = item
    preparation, _ = _read(run / "prepared/workflow_report.json", diagnostics)
    journal_path = run / "operations/submission_state.json"
    journal, journal_reason = _read(journal_path, diagnostics)
    scheduler, _ = _read(run / "operations/scheduler_observation.json", diagnostics)
    receipt, _ = _read(run / "operations/launch_receipt.json", diagnostics)

    directory = run / "local_observations"
    _safe_local(directory)
    if directory.exists():
        if not directory.is_dir():
            raise ValueError("Expected a local observations directory")
        # Only immutable observation names are inputs. Never inspect credentials,
        # arbitrary JSON, old staging trees, model tensors, or metadata links.
        paths = []
        with os.scandir(directory) as entries:
            for index, entry in enumerate(entries):
                if index == MAX_OBSERVATION_ENTRIES:
                    diagnostics.append({"source": str(directory), "reason": "observation_enumeration_truncated",
                                        "entry_limit": MAX_OBSERVATION_ENTRIES})
                    break
                if re.fullmatch(r"preparation-(?:[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\.json", entry.name):
                    paths.append(directory / entry.name)
        for path in sorted(paths):
            value, reason = _read(path, diagnostics)
            if value is None:
                samples.append(_sample("local_preparation", path, missing_reason=reason,
                                       qualification="Corrupt/unavailable retained local observation."))
            else:
                sample = _observation(path, value, expected)
                sample["qualification_reasons"].extend(conflicts)
                sample["identity_matches_retained_metadata"] = sample["identity_matches_retained_metadata"] and not conflicts
                samples.append(sample)
    if not samples:
        samples.append(_sample("local_preparation", directory,
                               missing_reason="no_retained_successful_preparation_observations",
                               qualification="Success-only observation retention; failures and missing index writes have no measured latency here."))

    timings = journal.get("timings") if journal is not None else None
    journal_binding = (journal or {}).get("binding")
    journal_binding = journal_binding if type(journal_binding) is dict else {}
    identity = _identities(journal_binding)
    # Journal binding normally owns only package identity; other provenance is
    # read from the frozen run, never from the reporting checkout's current HEAD.
    journal_conflicts = list(conflicts)
    for key, item in expected.items():
        if identity[key] is not None and item is not None and identity[key] != item:
            journal_conflicts.append(key + "_binding_mismatch")
        elif item is not None:
            identity[key] = item
    journal_reasons = journal_conflicts + [key + "_unavailable" for key, item in identity.items() if item is None]
    journal_reasons.extend(key + "_retained_binding_unavailable" for key, item in expected.items() if item is None)
    if not _digest(journal_binding.get("package_identity")):
        journal_reasons.append("journal_package_binding_unavailable")
    if type(timings) is not dict:
        timings = {}
        journal_reason = journal_reason or "missing_or_invalid_timings"
    for phase in dict.fromkeys((*_PHASES, *timings)):
        known = phase in _PHASES
        metric, meaning = _PHASES.get(phase, ("unclassified_phase_active", "Unknown phase semantics; no upload/wait/POST/receipt attribution."))
        seconds = _seconds(timings.get(phase))
        missing = journal_reason or ("phase_not_recorded" if phase not in timings else "invalid_duration" if seconds is None else None)
        samples.append(_sample(metric, journal_path, seconds=seconds if missing is None else None,
                               missing_reason=missing, identity=identity, reasons=journal_reasons,
                               phase=phase if _token(phase) else None, sample_kind="active_phase_accumulator",
                               identity_matches_retained_metadata=not journal_reasons and missing is None and known,
                               qualification=meaning + " One aggregate over recorded active invocations; excludes unrecorded/crashed intervals and time between invocations."))
    for metric, reason in _UNMEASURED.items():
        samples.append(_sample(metric, journal_path, missing_reason="not_measured", qualification=reason))

    receipt_binding = (receipt or {}).get("binding")
    receipt_identity = _identities(receipt_binding if type(receipt_binding) is dict else {})
    receipt_conflicts = [key + "_binding_mismatch" for key, item in receipt_identity.items()
                        if item is not None and expected[key] is not None and item != expected[key]]
    return {"format": "molgap-experiment-timings-report-v1", "status": "OBSERVATIONS_ONLY",
            "run_directory": str(run), "samples": samples, "summaries": _summaries(samples),
            "preparation_status": (preparation or {}).get("status") if _token((preparation or {}).get("status")) else None,
            "journal_phase": (journal or {}).get("phase") if _token((journal or {}).get("phase")) else None,
            "scheduler_observation": {"status": (scheduler or {}).get("status") if _token((scheduler or {}).get("status")) else None,
                "observed_at": (scheduler or {}).get("observed_at") if _timestamp((scheduler or {}).get("observed_at")) else None,
                "queue_seconds": None, "missing_reason": "queue_endpoints_not_recorded"},
            "receipt_observation": {"present": receipt is not None, "identity": receipt_identity,
                "submission_state": (receipt or {}).get("submission_state") if _token((receipt or {}).get("submission_state")) else None,
                "qualification_reasons": receipt_conflicts,
                "qualification": "Metadata observation only; package/source/receipt validity is owned by existing launch acceptance. Receipt presence establishes no latency."},
            "compute_cost": {"value": None, "missing_reason": "not_measured_by_timing_metadata"},
            "failure_observations": [item for item in (
                {"source": str(run / "prepared/workflow_report.json"), "status": preparation["status"],
                 "reason": "preparation_did_not_complete"}
                    if preparation and _token(preparation.get("status")) and preparation["status"] != "PREPARED_FOR_PLATFORM" else None,
                {"source": str(journal_path), "status": journal["phase"],
                 "reason": journal.get("error_type") if _token(journal.get("error_type")) else "submission_outcome_requires_reconciliation"}
                    if journal and _token(journal.get("phase")) and journal["phase"] in {"SOURCE_UNKNOWN", "SUBMIT_UNKNOWN"} else None) if item],
            "targets": {"local_preparation_90_seconds": "NOT_ESTABLISHED", "code_ready_to_receipt_5_minutes": "NOT_ESTABLISHED"},
            "read_limits": {"json_bytes": MAX_JSON_BYTES, "observation_directory_entries": MAX_OBSERVATION_ENTRIES},
            "diagnostics": diagnostics,
            "limitations": ["Success-only retained local observations are not a failure-rate denominator; unsuccessful prepares retain their owning workflow report/journal.",
                "No warm-environment, data-acceptance, permission, native, live-five-minute or scientific qualification is inferred.",
                "No phase sum, file mtime or receipt timestamp is used as wall latency, queue duration or compute cost.",
                "Identity matches compare metadata strings only; no source/archive/package verification or launch acceptance is performed.",
                "Directory scanning is capped at 1000 entries in filesystem order; truncated reports describe only the retained subset, not all observations.",
                "Reads are independent snapshots; concurrent writers may change metadata between reads."]}


def main(argv=None):
    """Print a report without creating directories or persisting another journal."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-directory", required=True)
    args = parser.parse_args(argv)
    try:
        report = report_experiment_timings(args.run_directory)
    except (OSError, ValueError, TypeError):
        print(json.dumps({"status": "ERROR", "reason": "invalid_or_unsafe_run_directory"}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
