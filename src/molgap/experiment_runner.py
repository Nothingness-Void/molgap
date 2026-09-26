"""Spawn-isolated adapter diagnostics, without training or evidence authority."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import stat
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from types import MappingProxyType

from .experiment_spec import ExperimentSpec
from .screen_policy import canonical_fingerprint

__all__ = ["run_experiment"]

ARM_FORMAT = "molgap-experiment-arm-result-v1"
RUN_FORMAT = "molgap-experiment-run-summary-v1"
_FAMILIES = MappingProxyType({
    ("gptrans_t", "1"): "gptrans",
    ("neural_atom_k1", "1"): "k1",
    ("edge_state_gps", "1"): "edge_state",
})
_WORKERS = frozenset({"adapter_probe", "construct"})
_LIMITATIONS = (
    "Orchestration diagnostic only; no forward, training, checkpoint or state loading. "
    "Declared source/data/state hashes are not authenticated. "
    "No runtime certification, role admission, terminal descriptor or replay authority."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _atomic(path: Path, value: dict) -> None:
    payload = _json(value).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".arm-write-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validated(spec: ExperimentSpec) -> ExperimentSpec:
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    if set(vars(spec)) != {"_canonical_json"} or type(spec._canonical_json) is not str:
        raise ValueError("Invalid ExperimentSpec snapshot")
    rebuilt = ExperimentSpec.from_json(spec._canonical_json)
    if rebuilt.to_json() != spec._canonical_json or rebuilt.identity != spec.identity:
        raise ValueError("Invalid ExperimentSpec canonical identity")
    return rebuilt


def _safe_name(name: str) -> None:
    if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}", name)
            or name.endswith(".")
            or name.split(".")[0].upper() in {
                "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                *(f"LPT{i}" for i in range(10)),
            }):
        raise ValueError(f"Unsafe path component: {name!r}")


def _output(path: Path) -> Path:
    if type(path) is not type(Path()):
        raise TypeError("output_root must be a concrete local Path")
    if not path.is_absolute() or ".." in path.parts or path == Path(path.anchor):
        raise ValueError("output_root must be an absolute non-root path without traversal")
    _safe_name(path.name)
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/junction/reparse output paths are forbidden")
    if path.exists():
        raise ValueError("output_root already exists; outputs cannot be overwritten")
    if not path.parent.is_dir():
        raise ValueError("output_root must have an existing directory parent")
    return path


def _devices(device_ids, count: int) -> list[str]:
    if type(device_ids) is not list or not device_ids or len(device_ids) != count:
        raise ValueError("device_ids must be a nonempty primitive list matching arms")
    visible = []
    for value in device_ids:
        if value is None or (type(value) is str and value in {"cpu", "CPU"}):
            visible.append("")
        elif type(value) is int and value >= 0:
            visible.append(str(value))
        elif type(value) is str and re.fullmatch(r"0|[1-9][0-9]*", value):
            visible.append(value)
        else:
            raise ValueError("Invalid device ID; use a nonnegative index, None or cpu")
    if len(set(visible)) != len(visible):
        raise ValueError("device_ids must be unique after CPU/index normalization")
    return visible


def _adapter(arm: dict):
    token = _FAMILIES[(arm["family"]["name"], arm["family"]["version"])]
    if token == "gptrans":
        from .gptrans_adapter import gptrans_metadata, build_gptrans_model
        return gptrans_metadata, build_gptrans_model
    if token == "edge_state":
        from .edge_state_adapter import edge_state_metadata, build_edge_state_model
        return edge_state_metadata, build_edge_state_model
    from .k1_adapter import k1_metadata, build_k1_model
    return k1_metadata, build_k1_model


def _metadata(spec: ExperimentSpec, arm: dict, metadata_fn) -> dict:
    metadata = asdict(metadata_fn(spec, arm["arm_id"]))
    metadata.update(
        arm_identity=canonical_fingerprint(arm), scientific_role=arm["scientific_role"],
        addons=arm["addons"], addon_semantics=arm["addon_semantics"],
        runner_limitations=_LIMITATIONS,
    )
    metadata.setdefault("source_module", metadata["family"]["source_module"])
    # JSON round-trip detaches tuples and preserves every adapter metadata field.
    return json.loads(_json(metadata))


def _empty_result(identity, arm, worker, requested, visible) -> dict:
    return {
        "format": ARM_FORMAT, "spec_identity": identity, "arm_id": arm["arm_id"],
        "arm_identity": canonical_fingerprint(arm), "role": arm["scientific_role"],
        "worker": worker, "requested_device": requested, "visible_device": visible,
        "status": "RUNNING", "pid": None, "observed_start": None,
        "observed_end": None, "observed_duration_seconds": None,
        "metadata": None, "parameter_count": None, "error": None,
        "recorded_by": "child",
    }


def _child(spec_json, arm_id, worker, requested, visible, output_dir):
    # All runner-owned adapter/model imports follow this boundary. Imports by
    # a launcher are not evidence that this runner initialized a model early.
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    started = time.monotonic()
    spec = ExperimentSpec.from_json(spec_json)
    arm = next(item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id)
    result = _empty_result(spec.identity, arm, worker, requested, os.environ["CUDA_VISIBLE_DEVICES"])
    result.update(pid=os.getpid(), observed_start=_now())
    path = Path(output_dir) / "arm_result.json"
    _atomic(path, result)
    try:
        if worker not in _WORKERS:
            raise ValueError("Unknown worker")
        metadata_fn, build_fn = _adapter(arm)
        result["metadata"] = _metadata(spec, arm, metadata_fn)
        if worker == "construct":
            if arm["initialization"]["kind"] != "random":
                raise ValueError("construct refuses frozen_state: runner does not implement state loading")
            seed = arm["initialization"]["seed"]
            random.seed(seed)
            import numpy as np
            import torch
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = build_fn(spec, arm_id)
            result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
            del model
        result["status"] = "SUCCEEDED"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result.update(observed_end=_now(), observed_duration_seconds=time.monotonic() - started)
    _atomic(path, result)
    if result["status"] != "SUCCEEDED":
        raise SystemExit(1)


def _check_result(loaded, expected, spec, arm, worker, pid):
    if (type(loaded) is not dict or set(loaded) != set(expected)
            or any(type(loaded[key]) is not type(expected[key]) or loaded[key] != expected[key]
                   for key in (
                       "format", "spec_identity", "arm_id", "arm_identity", "role", "worker",
                       "requested_device", "visible_device", "recorded_by",
                   ))
            or type(loaded["status"]) is not str
            or loaded["status"] not in {"RUNNING", "SUCCEEDED", "FAILED"}):
        raise ValueError("Child result identity/schema mismatch")
    if type(loaded["pid"]) is not int or loaded["pid"] <= 0 or loaded["pid"] != pid:
        raise ValueError("Child result PID mismatch")
    for field in ("observed_start", "observed_end"):
        value = loaded[field]
        if field == "observed_end" and loaded["status"] == "RUNNING" and value is None:
            continue
        if type(value) is not str or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("Missing/invalid child observation time")
    duration = loaded["observed_duration_seconds"]
    if loaded["status"] == "RUNNING":
        if loaded["observed_end"] is not None or duration is not None:
            raise ValueError("RUNNING cannot contain completed observations")
    elif (type(duration) not in {int, float} or not math.isfinite(duration) or duration < 0):
        raise ValueError("Invalid observed child duration")
    metadata = loaded["metadata"]
    if metadata is not None or loaded["status"] == "SUCCEEDED":
        metadata_fn, _ = _adapter(arm)
        if _json(metadata) != _json(_metadata(spec, arm, metadata_fn)):
            raise ValueError("Child metadata mismatch")
    error = loaded["error"]
    if loaded["status"] == "FAILED":
        if (type(error) is not dict or set(error) != {"type", "message"}
                or any(type(value) is not str for value in error.values())):
            raise ValueError("Invalid child error")
    elif error is not None:
        raise ValueError("Non-failed child cannot contain an error")
    count = loaded["parameter_count"]
    if worker == "construct" and loaded["status"] == "SUCCEEDED":
        if arm["initialization"]["kind"] != "random":
            raise ValueError("frozen_state construction cannot succeed")
        if type(count) is not int or count < 0:
            raise ValueError("Missing/invalid parameter count")
    elif count is not None:
        raise ValueError("Parameter count is only observed for successful construction")


def _finish(record, spec, worker):
    path = record["path"]
    arm = record["arm"]
    result = _empty_result(spec.identity, arm, worker, record["requested"], record["visible"])
    error = record.get("error")
    process = record["process"]
    try:
        raw = path.read_bytes()
        loaded = json.loads(raw)
        if raw != _json(loaded).encode("utf-8"):
            raise ValueError("Child result is not canonical JSON")
        _check_result(loaded, result, spec, arm, worker,
                      process.pid if process is not None else None)
        result = loaded
    except (OSError, ValueError, TypeError) as exc:
        error = error or {"type": type(exc).__name__, "message": str(exc)}
    exitcode = process.exitcode if process is not None else None
    if record.get("timed_out"):
        status = "TIMED_OUT"
        error = {"type": "TimeoutError", "message": "Arm exceeded its parent-observed deadline"}
    elif error or exitcode != 0 or result["status"] != "SUCCEEDED":
        status = "FAILED"
        error = error or result["error"] or {
            "type": "ChildProcessError", "message": f"Child exited with code {exitcode} without success",
        }
    else:
        status = "SUCCEEDED"
    if status != result["status"] or error:
        result.update(status=status, error=error, recorded_by="parent")
        # Missing child timing stays null; termination is not an observed worker end.
        _atomic(path, result)
    return {
        "arm_id": arm["arm_id"], "role": arm["scientific_role"], "status": status,
        "result_path": path.relative_to(record["root"]).as_posix(), "exit_code": exitcode,
        "pid": process.pid if process is not None else None,
        "parent_observed_launch": record.get("launch"),
        "parent_observed_exit": record.get("exit"),
        "error": error or result["error"],
    }


def run_experiment(
    spec: ExperimentSpec, output_root: Path, device_ids: list,
    *, worker: str = "adapter_probe", timeout_seconds: float = 60.0,
) -> dict:
    """Run every declared arm, preserving spec order and independent failures.

    Device IDs are unique normalized visibility tokens, not hardware admission.
    Timeout is per process from launch, including spawn/import overhead.
    The returned dictionary is also atomically persisted as run_summary.json.
    """
    spec = _validated(spec)
    if type(worker) is not str or worker not in _WORKERS:
        raise ValueError("Unknown worker; expected adapter_probe or construct")
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    declaration = spec.to_dict()
    arms = declaration["arms"]
    visible = _devices(device_ids, len(arms))
    if declaration["platform"]["device_count"] != len(arms):
        raise ValueError("platform.device_count must match the arm/device count")
    for arm in arms:
        _safe_name(arm["arm_id"])
        if (arm["family"]["name"], arm["family"]["version"]) not in _FAMILIES:
            raise ValueError("Unsupported runner family/version")
    if len({arm["arm_id"].casefold() for arm in arms}) != len(arms):
        raise ValueError("Case-colliding arm IDs are unsafe")
    root = _output(output_root)
    context = multiprocessing.get_context("spawn")
    root.mkdir(exist_ok=False)
    (root / "arms").mkdir()
    records = []
    for arm, requested, device in zip(arms, device_ids, visible):
        directory = root / "arms" / arm["arm_id"]
        directory.mkdir()
        records.append({"arm": arm, "requested": requested, "visible": device,
                        "path": directory / "arm_result.json", "root": root, "process": None})
    start = _now()
    tick = time.monotonic()
    try:
        for record in records:
            try:
                process = context.Process(target=_child, args=(
                    spec.to_json(), record["arm"]["arm_id"], worker, record["requested"],
                    record["visible"], str(record["path"].parent),
                ))
                record.update(process=process, launch=_now(), deadline=time.monotonic() + timeout_seconds)
                process.start()
            except Exception as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                record["exit"] = _now()
        pending = [record for record in records if not record.get("error")]
        while pending:
            for record in pending[:]:
                process = record["process"]
                if process.is_alive() and time.monotonic() >= record["deadline"]:
                    record["timed_out"] = True
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join()
                if not process.is_alive():
                    process.join()
                    record["exit"] = _now()
                    pending.remove(record)
            if pending:
                time.sleep(0.01)
        results = [_finish(record, spec, worker) for record in records]
        statuses = [result["status"] for result in results]
        status = ("TIMED_OUT" if "TIMED_OUT" in statuses else
                  "SUCCEEDED" if all(item == "SUCCEEDED" for item in statuses) else
                  "PARTIAL_FAILURE" if "SUCCEEDED" in statuses else "FAILED")
        summary = {
            "format": RUN_FORMAT, "spec_identity": spec.identity,
            "experiment_id": declaration["experiment_id"], "logical_run_id": declaration["logical_run_id"],
            "worker": worker, "status": status, "start_method": "spawn",
            "timeout_seconds": timeout_seconds, "observed_start": start, "observed_end": _now(),
            "observed_duration_seconds": time.monotonic() - tick, "arms": results,
            "limitations": _LIMITATIONS,
        }
        _atomic(root / "run_summary.json", summary)
        return summary
    finally:
        for record in records:
            process = record["process"]
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join()
                else:
                    process.join()
