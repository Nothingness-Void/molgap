"""Isolated adapter execution envelopes, without training or evidence authority."""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal

from .experiment_spec import ExperimentSpec
from .screen_policy import canonical_fingerprint

_ADAPTERS = MappingProxyType({
    ("gptrans_t", "1"): ("molgap.gptrans_adapter", "gptrans_metadata", "build_gptrans_model"),
    ("neural_atom_k1", "1"): ("molgap.k1_adapter", "k1_metadata", "build_k1_model"),
})
_WORKERS = ("adapter_probe", "construct")
_FORMAT = "molgap-arm-envelope"
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.I)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("Metadata keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, int, bool, float):
        _canonical(value)
        return value
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def _atomic_json(path, value):
    payload = _canonical(value).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validated(spec):
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec; providers are unsupported")
    if set(vars(spec)) != {"_canonical_json"} or type(spec._canonical_json) is not str:
        raise ValueError("Invalid ExperimentSpec snapshot")
    rebuilt = ExperimentSpec.from_json(spec._canonical_json)
    if rebuilt.to_json() != spec._canonical_json or rebuilt.identity != spec.identity:
        raise ValueError("Invalid canonical spec identity")
    return rebuilt


def _binding(value):
    if value is None or (type(value) is str and value.lower() == "cpu"):
        return {"requested": value, "cuda_visible_devices": ""}
    if type(value) is not str or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError("Device must be a canonical nonnegative index, CPU, or None")
    return {"requested": value, "cuda_visible_devices": value}


def _safe_component(name):
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name)
            or name.endswith(".") or _RESERVED.match(name)):
        raise ValueError(f"Unsafe arm directory: {name}")
    return name


def _output_path(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("output_root must be a nonempty path")
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise ValueError("Parent traversal is forbidden")
    root = raw.absolute()
    blocked = {"src", "data", "dataset", "datasets", ".git", "source", "source_package", "data_package"}
    for part in root.parts[1:]:
        if (part.lower() in blocked or part.endswith((".", " "))
                or _RESERVED.match(part) or any(c in part for c in '<>:"|?*')
                or any(ord(c) < 32 for c in part)):
            raise ValueError("Unsafe or source/data output path")
    for parent in (root, *root.parents):
        if parent.is_symlink() or (parent.exists() and
                getattr(parent.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Symlink/reparse output paths are forbidden")
        if any((parent / marker).exists() for marker in (
                "package_manifest.json", "SOURCE_FILES.json", "source.tar.gz", "dataset_manifest.json")):
            raise ValueError("Output must not be inside a source/data package")
    if root == root.parent or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise ValueError("Output root must be new or empty")
    return root


def _now():
    return datetime.now(timezone.utc).isoformat()


def _error(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _execute(spec, arm, worker):
    if worker not in _WORKERS:
        raise ValueError("Unknown worker")
    module, metadata_name, build_name = _ADAPTERS[(arm["family"]["name"], arm["family"]["version"])]
    adapter = importlib.import_module(module)
    metadata = _json_value(getattr(adapter, metadata_name)(spec, arm["arm_id"]))
    metadata.update(arm_identity=canonical_fingerprint(arm), addons=arm["addons"], adapter_module=module)
    if worker == "adapter_probe":
        return metadata, None
    if arm["initialization"]["kind"] != "random":
        raise ValueError("construct rejects frozen_state: state loading is not implemented by this runner")
    seed = arm["initialization"]["seed"]
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = getattr(adapter, build_name)(spec, arm["arm_id"])
    return metadata, sum(parameter.numel() for parameter in model.parameters())


def _child_main(snapshot, arm_id, worker, binding, directory):
    # Popen already installed this environment before interpreter/module imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = binding["cuda_visible_devices"]
    started, clock = _now(), time.monotonic()
    result = {
        "format": _FORMAT, "version": 1, "spec_identity": None, "arm_id": arm_id,
        "scientific_role": None, "worker": worker, "device_binding": binding,
        "observed_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "observed_pid": os.getpid(), "observed_started": started,
        "status": "FAILED", "metadata": None,
    }
    if worker == "construct":
        result["parameter_count"] = None
    try:
        spec = _validated(ExperimentSpec.from_json(snapshot))
        arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == arm_id)
        result.update(spec_identity=spec.identity, scientific_role=arm["scientific_role"])
        random.seed(arm["initialization"]["seed"])
        metadata, count = _execute(spec, arm, worker)
        result.update(metadata=metadata, status="SUCCEEDED")
        if worker == "construct":
            result["parameter_count"] = count
    except BaseException as exc:
        result["error"] = _error(exc)
    result.update(observed_finished=_now(), observed_duration_seconds=time.monotonic() - clock)
    _atomic_json(Path(directory) / "arm_result.json", result)
    return 0 if result["status"] == "SUCCEEDED" else 1


def _launch(snapshot, arm_id, worker, binding, directory):
    # Unlike multiprocessing spawn, a fresh interpreter does not import the
    # caller's __main__ (which may import torch before a spawn target runs).
    bootstrap = (
        "import sys,json; sys.path.insert(0,sys.argv[1]); "
        "from molgap.experiment_runner import _child_main; "
        "sys.exit(_child_main(**json.loads(sys.stdin.read())))"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = binding["cuda_visible_devices"]
    env["PYTHONHASHSEED"] = "0"
    # A file-backed stdin avoids blocking the parent on pipe capacity if a
    # large declaration meets a child stalled during interpreter startup.
    with tempfile.TemporaryFile() as stream:
        stream.write(_canonical(dict(snapshot=snapshot, arm_id=arm_id, worker=worker,
                                     binding=binding, directory=str(directory))).encode("utf-8"))
        stream.seek(0)
        return subprocess.Popen(
            [sys.executable, "-c", bootstrap, str(Path(__file__).resolve().parents[1])],
            stdin=stream, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )


def _stop(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_experiment_arms(
    spec: ExperimentSpec, output_root: Path | str, device_ids: list[str | None],
    worker: Literal["adapter_probe", "construct"] = "adapter_probe",
    timeout_seconds: float = 300.0,
) -> dict:
    """Run isolated static workers; SUCCEEDED describes the envelope only.

    CPU/None both bind the empty CUDA mask; duplicate masks are rejected.
    The timeout is per child, measured from launch, including interpreter startup.
    Missing/killed child results are represented only in the parent summary.
    """
    spec = _validated(spec)
    if type(worker) is not str or worker not in _WORKERS:
        raise ValueError("Unknown worker")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    arms = spec.to_dict()["arms"]
    if type(device_ids) is not list or len(device_ids) != len(arms):
        raise ValueError("device_ids must be a list with one binding per arm")
    if spec.to_dict()["platform"]["device_count"] != len(arms):
        raise ValueError("platform.device_count must equal the number of arms")
    bindings = [_binding(device) for device in device_ids]
    if len({b["cuda_visible_devices"] for b in bindings}) != len(bindings):
        raise ValueError("Duplicate device binding")
    names = [_safe_component(arm["arm_id"]) for arm in arms]
    if len(set(name.casefold() for name in names)) != len(names):
        raise ValueError("Case-colliding arm directories")
    for arm in arms:
        if (arm["family"]["name"], arm["family"]["version"]) not in _ADAPTERS:
            raise ValueError("Unsupported static adapter")
    root = _output_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive directory creation claims an empty output root across callers.
    (root / "arms").mkdir()
    for name in names:
        (root / "arms" / name).mkdir()
    records, processes = [], []
    try:
        for arm, binding, name in zip(arms, bindings, names):
            path = root / "arms" / name / "arm_result.json"
            record = dict(arm_id=arm["arm_id"], scientific_role=arm["scientific_role"],
                          device_binding=binding, result_path=None,
                          expected_result_path=str(path), exit_code=None, status="FAILED",
                          observed_started=None, observed_finished=None, observed_duration_seconds=None)
            records.append(record)
            launched = time.monotonic()
            try:
                process = _launch(spec.to_json(), arm["arm_id"], worker, binding, path.parent)
            except Exception as exc:
                record["error"] = _error(exc)
                continue
            processes.append((process, launched, record, path))
        pending = list(processes)
        while pending:
            for item in pending[:]:
                process, launched, record, path = item
                if process.poll() is None:
                    if time.monotonic() - launched < timeout_seconds:
                        continue
                    _stop(process)
                    record.update(status="TIMED_OUT", error={"type": "Timeout", "message": "Child exceeded timeout"})
                process.wait()
                record["exit_code"] = process.returncode
                if path.is_file():
                    record["result_path"] = str(path)
                if record["status"] != "TIMED_OUT":
                    try:
                        result = json.loads(path.read_text(encoding="utf-8"))
                        if (result["format"] != _FORMAT or result["version"] != 1
                                or result["spec_identity"] != spec.identity or result["arm_id"] != record["arm_id"]
                                or result["worker"] != worker or result["device_binding"] != record["device_binding"]
                                or result["scientific_role"] != record["scientific_role"]
                                or result["status"] not in ("SUCCEEDED", "FAILED")):
                            raise ValueError("Child envelope identity mismatch")
                        record["status"] = result["status"] if process.returncode == 0 else "FAILED"
                        for field in ("observed_started", "observed_finished", "observed_duration_seconds"):
                            record[field] = result.get(field)
                        if "error" in result:
                            record["error"] = result["error"]
                        elif record["status"] == "FAILED":
                            record["error"] = {"type": "ChildExit", "message": f"Child exit code {process.returncode}"}
                    except Exception as exc:
                        record.update(status="FAILED", error=_error(exc))
                pending.remove(item)
            if pending:
                time.sleep(0.01)
    finally:
        for process, _, _, _ in processes:
            if process.poll() is None:
                _stop(process)
    statuses = [record["status"] for record in records]
    aggregate = ("TIMED_OUT" if "TIMED_OUT" in statuses else "SUCCEEDED" if all(s == "SUCCEEDED" for s in statuses)
                 else "PARTIAL_FAILURE" if "SUCCEEDED" in statuses else "FAILED")
    summary = dict(format="molgap-run-envelope", version=1, spec_identity=spec.identity,
                   arm_ids=names, worker=worker, device_map=[dict(arm_id=n, **b) for n, b in zip(names, bindings)],
                   arms=records, status=aggregate)
    _atomic_json(root / "run_summary.json", summary)
    return summary
