"""Bound, sequential clean-BN diagnostics; no training, selection or RML writes.

Paths and evidence identities are caller-owned. Each arm is isolated so the
parent can enforce a wall ceiling even during a blocked loader/forward call.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ARMS = ("k1_pretrained_mean2", "k1_pretrained_consistency")
SETTINGS = {"cpu_threads": 4, "precision": "fp32", "tf32_enabled": False,
            "sample_seed": 20261008, "train_rows": 500000,
            "calibration_rows": 16384, "development_rows": 50000,
            "batch_size": 128, "epoch_zero_based": 48,
            "prediction_tolerance_eV": 1e-4, "arm_wall_seconds": 600,
            "pair_wall_seconds": 1200}


def _keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"Expected exactly these keys: {sorted(required)}")


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def _bound(binding, *, normalized=False):
    from .training_reproducibility import sha256_file
    from .v4_runtime import normalized_source_sha256
    _keys(binding, ("path", "sha256"))
    path = Path(binding["path"])
    digest = binding["sha256"]
    if not path.is_absolute() or not isinstance(digest, str) or len(digest) != 64 or \
            any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Expected absolute path and SHA256 binding")
    if (normalized_source_sha256(path) if normalized else sha256_file(path)) != digest:
        raise ValueError(f"Bound bytes differ: {path}")
    return path


def validate_inputs(inputs):
    """Verify both original trajectories and all pins before state adaptation."""
    from .training_reproducibility import sha256_file
    from .v4_runtime import normalized_source_sha256
    from .experiment_package import verify_experiment_source_package
    _keys(inputs, ("format", "settings", "protocol", "source_archive",
                   "source_inventory", "frozen_source_root", "cache_root",
                   "manifest_sha256", "executed_source_files", "arms"))
    if inputs["format"] != "molgap-k1-clean-bn-pair-v1" or inputs["settings"] != SETTINGS:
        raise ValueError("Clean-BN diagnostic contract differs")
    protocol = _bound(inputs["protocol"], normalized=True)
    archive_path = _bound(inputs["source_archive"])
    inventory_path = _bound(inputs["source_inventory"])
    if archive_path.name != "source.tar.gz" or inventory_path != archive_path.parent / "SOURCE_FILES.json":
        raise ValueError("Source pins must reference the original package sidecars")
    verify_experiment_source_package(archive_path.parent)
    inventory = _json(inventory_path)
    if inventory.get("format") != "molgap-v4-source-inventory-v1":
        raise ValueError("Unknown frozen source inventory")
    root = Path(inputs["frozen_source_root"])
    if not root.is_absolute() or not Path(inputs["cache_root"]).is_absolute():
        raise ValueError("Expected absolute source/cache roots")
    entries = {item["path"]: item for item in inventory["files"]}
    if len(entries) != len(inventory["files"]):
        raise ValueError("Duplicate inventory entry")
    for name, entry in entries.items():
        if name.startswith("src/molgap/") and normalized_source_sha256(root / name) != entry["sha256"]:
            raise ValueError(f"Frozen source bytes differ: {name}")
    protocol_names = [name for name in entries if name.endswith("/protocol.md") and
                      entries[name]["sha256"] == inputs["protocol"]["sha256"]]
    if len(protocol_names) != 1 or protocol.name != "protocol.md":
        raise ValueError("Protocol does not match frozen inventory")
    _keys(inputs["arms"], ARMS)
    refs = []
    for arm in ARMS:
        bound = inputs["arms"][arm]
        _keys(bound, ("trajectory", "trajectory_id", "checkpoint", "saved_predictions"))
        path = _bound(bound["trajectory"])
        trajectory = _json(path)
        if trajectory.get("record_mode") != "prospective" or \
                trajectory.get("trajectory_id") != bound["trajectory_id"] or \
                arm not in bound["trajectory_id"] or \
                trajectory.get("decision_state", {}).get("source_hashes", {}).get(protocol_names[0]) != \
                inputs["protocol"]["sha256"]:
            raise ValueError("Original prospective trajectory differs")
        refs.append((path.resolve(), bound["trajectory_id"]))
        _bound(bound["checkpoint"])
        _bound(bound["saved_predictions"])
    if refs[0][0] == refs[1][0] or refs[0][1] == refs[1][1]:
        raise ValueError("Expected two distinct original trajectories")
    if not inputs["executed_source_files"]:
        raise ValueError("Missing executed diagnostic source pins")
    for path, digest in inputs["executed_source_files"].items():
        _bound({"path": path, "sha256": digest})
    required = tuple(Path(__file__).with_name(name).resolve() for name in
                     ("k1_bn_diagnostic.py", "k1_frozen_inference.py", "k1_bn_calibration.py"))
    if any(str(path) not in {str(Path(p).resolve()) for p in inputs["executed_source_files"]}
           for path in required):
        raise ValueError("Missing driver/loader/calibration source pins")
    return inputs


def _frozen_imports(inputs):
    import molgap
    root = Path(inputs["frozen_source_root"]) / "src/molgap"
    local = Path(__file__).resolve().parent
    molgap.__path__[:] = [str(root), str(local)]
    for name in ("k1_screen_training", "pcqm_k1_scale_runner", "v4_runtime",
                 "training_reproducibility", "pcqm_wedge", "pcqm_gap_architecture"):
        importlib.import_module(f"molgap.{name}")
    # The diagnostic loader is the reviewed adapter, not the packaged old loader.
    for name in ("k1_frozen_inference", "k1_bn_calibration"):
        fullname = f"molgap.{name}"
        if fullname not in sys.modules:
            spec = importlib.util.spec_from_file_location(fullname, local / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            spec.loader.exec_module(module)
        if Path(sys.modules[fullname].__file__).resolve() != local / f"{name}.py":
            raise ValueError("Wrong diagnostic adapter imported")
    _verify_frozen_modules(inputs)


def _verify_frozen_modules(inputs):
    """Recheck lazy model/serialized-graph imports before any model forward."""
    inventory = _json(inputs["source_inventory"]["path"])
    pins = {item["path"]: item["sha256"] for item in inventory["files"]}
    from .v4_runtime import normalized_source_sha256
    for name, module in tuple(sys.modules.items()):
        if name.startswith("molgap.") and name not in (
                "molgap.k1_bn_diagnostic", "molgap.k1_frozen_inference", "molgap.k1_bn_calibration"):
            relative = "src/" + name.replace(".", "/") + ".py"
            if relative not in pins or Path(module.__file__).resolve() != \
                    (Path(inputs["frozen_source_root"]) / relative).resolve() or \
                    normalized_source_sha256(Path(module.__file__)) != pins[relative]:
                raise ValueError(f"Nonfrozen dependency imported: {name}")


def _load_graphs(inputs, indices, deadline):
    """Select members through the owning packed reader; never rebuild a cache."""
    import torch
    from .k1_screen_training import _PackedGraphDatasetFactory
    from .pcqm_k1_scale_runner import find_cache
    from .training_reproducibility import sha256_file
    os.environ["MOLGAP_PCQM_500K_V4_ROOT"] = inputs["cache_root"]
    root, manifest = find_cache(inputs["manifest_sha256"])
    selected, development, offset = {}, None, 0
    for item in manifest["geometry_shards"]:
        _deadline(deadline)
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != item["sha256"]:
            raise ValueError("Graph shard path/hash differs")
        data = _PackedGraphDatasetFactory.load(path)
        count = item["rows"]
        if len(data) != count or not torch.equal(data._data.source_idx.view(-1).long(),
                                                torch.arange(offset, offset + count)):
            raise ValueError("Graph row identity differs")
        if offset < SETTINGS["train_rows"]:
            if offset + count > SETTINGS["train_rows"]:
                raise ValueError("Shard crosses role boundary")
            for index in indices[(indices >= offset) & (indices < offset + count)]:
                selected[int(index)] = data[int(index - offset)].clone()
        elif development is None and offset == 500000 and count == 50000:
            development = data
        else:
            raise ValueError("Unexpected development shard")
        offset += count
    if offset != 550000 or development is None or len(selected) != len(indices):
        raise ValueError("Incomplete calibration/development members")
    return [selected[int(index)] for index in indices], development


def _deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Clean-BN wall ceiling exhausted")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _execute_arm(inputs, arm, output, deadline, progress):
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from .k1_frozen_inference import load_native500k_k1, predict_clean
    from .k1_bn_calibration import recalibrated_batch_norm
    from .training_reproducibility import (
        atomic_torch_save, sha256_file, configure_fp32_determinism, build_runtime_manifest)
    from .v4_runtime import state_dict_sha256
    settings = configure_fp32_determinism(SETTINGS["sample_seed"])
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(4)
    progress("runtime_configured", runtime=build_runtime_manifest(settings),
             cpu_threads=torch.get_num_threads(), device="cpu", autocast_enabled=False,
             cpu_processor=__import__("platform").processor(), cpu_logical_count=os.cpu_count(),
             torch_build_config=torch.__config__.show())
    indices = np.random.default_rng(20261008).choice(500000, 16384, replace=False)
    progress("sample_bound", calibration_source_idx_sha256=state_dict_sha256(
        {"source_idx": torch.as_tensor(indices, dtype=torch.long)}),
        development_source_idx_sha256=state_dict_sha256(
        {"source_idx": torch.arange(500000, 550000)}),
        row_hash_scope="v4_runtime.state_dict_sha256({'source_idx': int64 ordered rows})")
    calibration, development = _load_graphs(inputs, indices, deadline)
    _verify_frozen_modules(inputs)
    progress("graphs_verified")
    bound = inputs["arms"][arm]
    model, metadata = load_native500k_k1(Path(bound["checkpoint"]["path"]),
        expected_sha256=bound["checkpoint"]["sha256"],
        expected_source_sha256=inputs["source_archive"]["sha256"],
        expected_epoch=48, checkpoint_kind="selected", expected_arm=arm)
    _verify_frozen_modules(inputs)
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in model.state_dict().values()):
        raise ValueError("Expected FP32 model state")
    params = state_dict_sha256(dict(model.named_parameters()))
    buffers = state_dict_sha256(dict(model.named_buffers()))
    loader = DataLoader(development, batch_size=128, shuffle=False, num_workers=0)
    def predict():
        result, timing = predict_clean(model, loader, mean=metadata["mean"], std=metadata["std"],
                                        device="cpu", deadline=deadline)
        _deadline(deadline)
        if any(value.ndim != 1 or len(value) != 50000 for value in result.values()) or \
                not torch.equal(result["source_idx"], torch.arange(500000, 550000)):
            raise ValueError("Expected full ordered 50K prediction")
        return result, timing
    original, timing = predict()
    atomic_torch_save(output / "original.pt", original)
    saved = torch.load(_bound(bound["saved_predictions"]), map_location="cpu", weights_only=True)
    if not torch.equal(original["source_idx"], saved["source_idx"].view(-1).long()) or \
            not torch.equal(original["target_eV"], saved["target"].view(-1)):
        raise ValueError("Saved row/target alignment differs")
    retained = saved["prediction"].view(-1)
    if retained.shape != original["prediction_eV"].shape or not torch.isfinite(retained).all():
        raise ValueError("Invalid saved prediction")
    error = float((original["prediction_eV"] - retained).abs().max())
    if error > 1e-4:
        raise ValueError("Saved predictions do not reconstruct within 1e-4 eV")
    progress("original_verified", reconstruction_max_abs_eV=error, original_timing=timing,
             parameter_sha256_before=params, buffer_sha256_before=buffers)
    calibration_loader = DataLoader(calibration, batch_size=128, shuffle=False, num_workers=0)
    with recalibrated_batch_norm(model, calibration_loader, source_idx=indices,
            source_bounds=(0, 500000), device="cpu", deadline=deadline) as report:
        atomic_torch_save(output / "calibrated_buffers.pt",
                          {name: value.detach().cpu().clone() for name, value in model.named_buffers()})
        progress("bn_calibrated", calibration=dict(report),
                 parameter_sha256_calibrated=state_dict_sha256(dict(model.named_parameters())))
        calibrated, timing = predict()
        if not torch.equal(original["target_eV"], calibrated["target_eV"]):
            raise ValueError("Calibrated targets differ")
        atomic_torch_save(output / "calibrated.pt", calibrated)
        progress("calibrated_saved", calibrated_timing=timing)
    restored, timing = predict()
    atomic_torch_save(output / "restored.pt", restored)
    if any(not torch.equal(value, restored[name]) for name, value in original.items()):
        raise ValueError("Restored full50K prediction is not EXACT")
    if params != state_dict_sha256(dict(model.named_parameters())) or \
            buffers != state_dict_sha256(dict(model.named_buffers())) or not report["buffers_restored"]:
        raise ValueError("Original state not restored")
    _deadline(deadline)
    progress("complete", calibration=dict(report), restored_timing=timing,
             restored_prediction_exact=True, parameter_sha256_restored=params,
             buffer_sha256_restored=buffers,
             artifacts={p.name: sha256_file(p) for p in output.glob("*.pt")})


def _arm_worker(inputs, arm, output):
    # Import only the source-pinned dependency tree before model/data access.
    started, cpu_started = time.perf_counter(), time.process_time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = {"arm": arm, "status": "running", "started_at_utc": _utc()}
    def progress(phase, **fields):
        from .training_reproducibility import atomic_json
        report.update(fields)
        report.update(phase=phase, updated_at_utc=_utc(),
                      wall_seconds=time.perf_counter() - started,
                      process_cpu_seconds=time.process_time() - cpu_started)
        atomic_json(Path(output) / "report.json", report)
    try:
        # Spawned processes have not imported local model owners.
        _frozen_imports(inputs)
        progress("started")
        _execute_arm(inputs, arm, Path(output), started + 600, progress)
        report["status"] = "complete"
        progress("complete", ended_at_utc=_utc())
    except BaseException as exc:
        report["status"] = "failed"
        progress("failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_utc())
        raise


def run_clean_bn_pair(inputs, output):
    """Run validated bound inputs into a fresh output directory, without retry."""
    # Validation imports IO helpers only, never model constructors or cache readers.
    from .training_reproducibility import atomic_json
    validate_inputs(inputs)
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "started_at_utc": _utc(), "arms": {}, "inputs": inputs}
    atomic_json(output / "pair_report.json", report)
    context = multiprocessing.get_context("spawn")
    for arm in ARMS:
        arm_dir = output / arm
        arm_dir.mkdir()
        arm_started = time.perf_counter()
        worker = context.Process(target=_arm_worker, args=(inputs, arm, str(arm_dir)))
        try:
            worker.start()
            worker.join(max(0, min(600 - (time.perf_counter() - arm_started),
                                   1200 - (time.perf_counter() - started))))
            timed_out = worker.is_alive()
            if timed_out:
                worker.terminate()
                worker.join()
            path = arm_dir / "report.json"
            observation = _json(path) if path.exists() else {"process_cpu_seconds": None}
            observation.update(worker_exitcode=worker.exitcode,
                               parent_observed_wall_seconds=time.perf_counter() - arm_started)
            if timed_out or worker.exitcode != 0 or observation.get("status") != "complete":
                observation.update(status="failed", ended_at_utc=_utc(),
                                   stop_reason="wall_ceiling" if timed_out else "worker_failure")
                atomic_json(path, observation)
                report["status"] = "failed"
            report["arms"][arm] = observation
        except BaseException as exc:
            if worker.pid is not None and worker.is_alive():
                worker.terminate()
                worker.join()
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report.update(updated_at_utc=_utc(), wall_seconds=time.perf_counter() - started)
            atomic_json(output / "pair_report.json", report)
        if report["status"] == "failed":
            break
    if report["status"] != "failed":
        report["status"] = "complete"
    report.update(ended_at_utc=_utc(), wall_seconds=time.perf_counter() - started)
    atomic_json(output / "pair_report.json", report)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = _json(args.inputs)
    entry = Path(sys.modules["__main__"].__file__).resolve()
    if str(entry) not in {str(Path(p).resolve()) for p in inputs["executed_source_files"]}:
        raise ValueError("Missing CLI entry point source pin")
    return 0 if run_clean_bn_pair(inputs, args.output)["status"] == "complete" else 1
