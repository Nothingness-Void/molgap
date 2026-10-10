"""Bounded paired K1/parameter-EMA adapter; no submission or role discovery.

Reuses pcqm_500k_v4_evidence's K1 factory, loader, optimizer, step and
cosine60 schedule (reviewed against retained owner 031a890b). Neither its
Kaggle launch loop nor its metric-based futility gates apply here.
Run/preflight require CPU-accepted fixed500K input and a parent-frozen source
archive/config. The parent must enforce a hard allocation watchdog, including
setup, before 14400 seconds; cooperative deadlines cannot interrupt a kernel.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from . import pcqm_500k_v4_evidence as owner
from .edge_state_training_core import EpochPermutationBatchSampler
from .k1_bn_calibration import recalibrated_batch_norm
from .k1_screen_training import FORBIDDEN_MODEL_FIELDS, _batch_sha256, _forward
from .k1_weight_ema import make_ema, update_ema
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .training_reproducibility import (
    assert_finite_state_dict, atomic_json, atomic_torch_save,
    build_runtime_manifest, capture_rng_state, configure_fp32_determinism,
    restore_rng_state, sha256_file,
)
from .v4_runtime import load_frozen_initial_state, state_dict_sha256, torch_load_compat

ARMS = ("reference", "ema999")
TRAIN_ROWS = 500_000
DEV_ROWS = 50_000
CALIBRATION_ROWS = 16_384
FORMAT = "molgap-colab-k1-paired500k-v1"


def _check(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("K1 allocation wall budget exhausted")


def _retain(source, target, expected_sha256):
    if target.exists():
        if sha256_file(target) != expected_sha256:
            raise ValueError("Retained input hash mismatch")
        return
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        raise ValueError("Input changed during atomic retention")
    os.replace(temporary, target)


@contextmanager
def _preserve_rng():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def _cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _exact(left, right):
    import numpy as np
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def _sampler(graphs, epoch, offset=0):
    return EpochPermutationBatchSampler(len(graphs), owner.BS, seed=42,
                                       epoch=epoch, start_batch=offset)


def _train_loader(graphs, epoch, offset=0):
    from torch.utils.data import Subset
    sampler = _sampler(graphs, epoch, offset)
    # Keep the owning PyG loader; resume indexes use processed, not prefetched, batches.
    return owner.loader(Subset(graphs, sampler.indices[offset * owner.BS:].tolist()))


def _data(dataset_root):
    import os
    import torch
    from .pcqm_k1_scale_runner import find_cache, load_roles, _targets
    previous = os.environ.get("MOLGAP_PCQM_500K_V4_ROOT")
    try:
        os.environ["MOLGAP_PCQM_500K_V4_ROOT"] = str(dataset_root)
        root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    finally:
        if previous is None:
            os.environ.pop("MOLGAP_PCQM_500K_V4_ROOT", None)
        else:
            os.environ["MOLGAP_PCQM_500K_V4_ROOT"] = previous
    # Validate path/roles before the existing owner deserializes trusted CPU shards.
    for shard in manifest["geometry_shards"]:
        path = (root / shard["file"]).resolve()
        if root.resolve() not in path.parents or shard["role"] not in {"train", "development"}:
            raise ValueError("Shard outside accepted root or internal roles")
    roles = load_roles(root, manifest)
    for role, start, rows in (("train", 0, TRAIN_ROWS), ("validation", TRAIN_ROWS, DEV_ROWS)):
        cursor = start
        for shard in roles[role].datasets:
            observed = shard._data.source_idx.view(-1).cpu()
            if not torch.equal(observed, torch.arange(cursor, cursor + len(shard))):
                raise ValueError("Fixed source membership/order changed")
            for field in FORBIDDEN_MODEL_FIELDS:
                if field in shard._data:
                    del shard._data[field]
                    shard.slices.pop(field, None)
            cursor += len(shard)
        if cursor != start + rows:
            raise ValueError("Internal role row count changed")
    target = _targets(roles["train"]).double()
    if target.numel() != TRAIN_ROWS or not torch.isfinite(target).all():
        raise ValueError("Invalid train-only target statistics")
    return roles, float(target.mean()), float(target.std(unbiased=True).clamp_min(1e-6)), manifest


def _arm(model):
    return {"model": model, "optimizer": owner.optimizer_for(model),
            "ema": None, "rng": capture_rng_state(), "epoch": 0, "offset": 0,
            "steps": 0, "samples": 0, "best": None, "best_epoch": None,
            "trace": [], "step_trace": [], "train_loss_sum": 0.0,
            "best_payload": None}


def _snapshot(arm):
    return _cpu({**{key: value for key, value in arm.items()
                   if key not in {"model", "optimizer", "ema"}},
                 "model": arm["model"].state_dict(),
                 "optimizer": arm["optimizer"].state_dict(),
                 "ema": None if arm["ema"] is None else arm["ema"].state_dict()})


def _restore(arm, saved):
    arm["model"].load_state_dict(saved["model"], strict=True)
    arm["optimizer"].load_state_dict(saved["optimizer"])
    if (arm["ema"] is None) != (saved["ema"] is None):
        raise ValueError("EMA checkpoint arm changed")
    if arm["ema"] is not None:
        arm["ema"].load_state_dict(saved["ema"], strict=True)
    arm.update({key: value for key, value in saved.items()
                if key not in {"model", "optimizer", "ema"}})


def _step(arm, batch, mean, std, device, deadline):
    import torch
    _check(deadline)
    batch = batch.to(device)
    if any(field in batch for field in FORBIDDEN_MODEL_FIELDS):
        raise ValueError("Geometry reached pure-2D optimizer step")
    if batch.y.dtype != torch.float32 or batch.random_walk_pe.dtype != torch.float32 or any(
            p.dtype != torch.float32 for p in arm["model"].parameters()):
        raise ValueError("K1 screen requires FP32")
    arm["model"].train()
    # Check the actual training forward before the reused owner performs backward.
    def finite_output(module, args, prediction):
        if prediction.numel() != owner.BS or not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite or malformed training forward")
    hook = arm["model"].register_forward_hook(finite_output)
    try:
        loss = owner.step(arm["model"], arm["optimizer"], batch, mean, std)
    finally:
        hook.remove()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite optimizer loss")
    if arm["ema"] is not None:
        update_ema(arm["ema"], arm["model"])
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    arm["offset"] += 1
    arm["steps"] += 1
    arm["samples"] += owner.BS
    arm["train_loss_sum"] += float(loss)
    return float(loss)


def _evaluate(model, graphs, mean, std, device, deadline, expected):
    import torch
    result = {"source_idx": [], "target_eV": [], "prediction_eV": []}
    model.eval()
    with torch.no_grad():
        for batch in owner.loader(graphs):
            _check(deadline)
            if any(field in batch for field in FORBIDDEN_MODEL_FIELDS):
                raise ValueError("Geometry reached pure-2D evaluation")
            batch = batch.to(device)
            result["source_idx"].append(batch.source_idx.view(-1).long().cpu())
            result["target_eV"].append(batch.y.view(-1).cpu())
            result["prediction_eV"].append((_forward(model, batch) * std + mean).cpu())
    result = {key: torch.cat(values) for key, values in result.items()}
    if not torch.equal(result["source_idx"], expected) or not all(
            torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Development identity or finite output failed")
    result["mae_eV"] = float((result["prediction_eV"] - result["target_eV"]).abs().mean())
    return result


def _endpoint(arm, roles, mean, std, device, deadline):
    import torch
    from torch.utils.data import Subset
    # A frozen clone avoids clearing live gradients or changing live mode/buffers.
    selected = make_ema(arm["model"] if arm["ema"] is None else arm["ema"])
    with _preserve_rng():
        expected = torch.arange(TRAIN_ROWS, TRAIN_ROWS + len(roles["validation"]))
        raw = _evaluate(selected, roles["validation"], mean, std, device, deadline, expected)
        members = torch.arange(CALIBRATION_ROWS)
        with recalibrated_batch_norm(
                selected, owner.loader(Subset(roles["train"], members.tolist())),
                source_idx=members, source_bounds=(0, TRAIN_ROWS), device=device,
                deadline=deadline, dropout_enabled=False, passes_per_batch=1) as report:
            calibrated = _evaluate(selected, roles["validation"], mean, std,
                                   device, deadline, expected)
            # Must copy while calibrated buffers are still installed.
            calibrated_state = _cpu(selected.state_dict())
        calibration = copy.deepcopy(report)
    return raw, calibrated, calibrated_state, calibration


def _save(output, arms, binding, runtime, train_rows, status, started):
    snapshots = {name: _snapshot(arm) for name, arm in arms.items()}
    for name, arm in arms.items():
        sampler = _sampler(range(train_rows), arm["epoch"], arm["offset"])
        snapshots[name]["sampler"] = sampler.state_for(arm["offset"])
        assert_finite_state_dict(snapshots[name]["model"], label=name)
    payload = {"format": FORMAT, "binding": binding, "runtime": runtime,
               "arms": snapshots, "status": status}
    atomic_torch_save(output / "last.pt", payload)
    atomic_json(output / "trace.json", {name: {"epochs": a["trace"], "steps": a["step_trace"]}
                                           for name, a in arms.items()})
    matched = min(a["epoch"] for a in arms.values())
    eligible = {}
    for name, arm in arms.items():
        rows = [row for row in arm["trace"] if row["epoch"] < matched]
        best = min(rows, key=lambda row: row["calibrated_mae_eV"]) if rows else None
        eligible[name] = None if best is None else {
            "epoch": best["epoch"], "calibrated_mae_eV": best["calibrated_mae_eV"],
            "snapshot": f"{name}/epoch_{best['epoch']:02d}.pt"}
    receipt = {"sha256": sha256_file(output / "last.pt"), "status": status,
               "trace_sha256": sha256_file(output / "trace.json"),
               "elapsed_allocation_seconds": time.time() - started,
               "matched_completed_epochs": matched,
               "matched_prefix_selection": eligible,
               "complete": status == "COMPLETE",
               "arms": {name: {key: arm[key] for key in
                         ("epoch", "offset", "steps", "samples", "best", "best_epoch")}
                        for name, arm in arms.items()}}
    atomic_json(output / "last.json", receipt)
    print(json.dumps({"checkpoint": str(output / "last.pt"),
                      "status": status, "matched_completed_epochs": matched,
                      "exposure": receipt["arms"]}), flush=True)
    return receipt


def _resume(path, expected_sha256, arms, binding, runtime, train_rows):
    if sha256_file(path) != expected_sha256:
        raise ValueError("Resume checkpoint hash mismatch")
    saved = torch_load_compat(path, map_location="cpu", weights_only=False)
    if saved["format"] != FORMAT or saved["binding"] != binding or saved["runtime"] != runtime:
        raise ValueError("Resume source/config/runtime identity changed")
    if set(saved["arms"]) != set(ARMS):
        raise ValueError("Resume arm identities changed")
    for name, state in saved["arms"].items():
        epoch, offset = state["epoch"], state["offset"]
        if not 0 <= epoch <= owner.EPOCHS or (epoch == owner.EPOCHS and offset != 0):
            raise ValueError("Resume epoch outside full horizon")
        EpochPermutationBatchSampler.from_state(state["sampler"], dataset_size=train_rows,
            batch_size=owner.BS, seed=42, epoch=epoch)
        if state["sampler"]["next_batch"] != offset or state["steps"] != (
                epoch * (train_rows // owner.BS) + offset) or state["samples"] != state["steps"] * owner.BS:
            raise ValueError("Resume exposure/cursor changed")
        if len(state["trace"]) != epoch or len(state["step_trace"]) != state["steps"]:
            raise ValueError("Resume trace/cursor changed")
        _restore(arms[name], {key: value for key, value in state.items() if key != "sampler"})
    ref, ema = (arms[name] for name in ARMS)
    if not (ref["epoch"] == ema["epoch"] or ref["epoch"] == ema["epoch"] + 1) or (
            ref["epoch"] == ema["epoch"] and ema["offset"] != 0) or (
            ref["epoch"] > ema["epoch"] and ref["offset"] != 0):
        raise ValueError("Resume is not an alternating matched-prefix pair")


def _qualify(arms, graphs, mean, std, output, device, deadline):
    import torch
    from torch.utils.data import Subset
    order = _sampler(graphs, 0).indices
    fixtures = [next(iter(owner.loader(Subset(graphs, indices.tolist()))))
                for indices in (order[:owner.BS], order[-owner.BS:])]
    report = {}
    with _preserve_rng():
        for name, arm in arms.items():
            initial = _snapshot(arm)
            results = []
            for repetition in range(2):
                _restore(arm, copy.deepcopy(initial))
                restore_rng_state(arm["rng"])
                first = _step(arm, fixtures[0], mean, std, device, deadline)
                arm["rng"] = capture_rng_state()
                prefix = _snapshot(arm)
                path = output / f"{name}_preflight_resume.pt"
                atomic_torch_save(path, prefix)
                prefix_sha = sha256_file(path)
                last = _step(arm, fixtures[1], mean, std, device, deadline)
                continuous = _snapshot(arm)
                continuous["rng"] = capture_rng_state()
                assert_finite_state_dict(continuous["model"], label="preflight live")
                if continuous["ema"] is not None:
                    assert_finite_state_dict(continuous["ema"], label="preflight EMA")
                if sha256_file(path) != prefix_sha:
                    raise ValueError("Preflight checkpoint changed")
                _restore(arm, torch_load_compat(path, map_location="cpu", weights_only=False))
                restore_rng_state(arm["rng"])
                replay = _step(arm, fixtures[1], mean, std, device, deadline)
                resumed = _snapshot(arm)
                resumed["rng"] = capture_rng_state()
                if last != replay or not _exact(continuous, resumed):
                    raise ValueError("Optimizer/EMA/RNG exact resume failed")
                results.append({"losses": [first, last],
                                "model": state_dict_sha256(arm["model"].state_dict()),
                                "ema": None if arm["ema"] is None else
                                state_dict_sha256(arm["ema"].state_dict())})
            if results[0] != results[1]:
                raise ValueError("Deterministic step repeat failed")
            _restore(arm, initial)
            report[name] = {"repeat": results[0], "exact_resume": True,
                            "initial_state_sha256": state_dict_sha256(initial["model"]),
                            "fixture_sha256": [_batch_sha256(b) for b in fixtures],
                            "formal_samples": 0}
    if report["reference"]["initial_state_sha256"] != report["ema999"]["initial_state_sha256"] or (
            report["reference"]["repeat"]["model"] != report["ema999"]["repeat"]["model"] or
            report["reference"]["repeat"]["losses"] != report["ema999"]["repeat"]["losses"]):
        raise ValueError("Paired single-forward qualification differs")
    if str(device).startswith("cuda"):
        peak = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        if peak > total * 0.85:
            raise ValueError("Optimizer-inclusive BS128 memory reserve below 15%")
        report["memory"] = {"peak_reserved_bytes": peak, "total_bytes": total}
    atomic_json(output / "preflight.json", report)
    return report


def _paired(arms, roles, mean, std, output, binding, runtime, device, deadline,
            started, checkpoint_steps=128):
    output.mkdir(parents=True, exist_ok=True)
    status = "RUNNING"
    try:
        _check(deadline)
        for name, arm in arms.items():
            # best.pt is a mirror: last.pt owns selection if publication was interrupted.
            if arm["best_payload"] is not None:
                atomic_torch_save(output / name / "best.pt", arm["best_payload"])
        _save(output, arms, binding, runtime, len(roles["train"]), status, started)
        last_save = time.perf_counter()
        while min(a["epoch"] for a in arms.values()) < owner.EPOCHS:
            # Reference leads by at most one epoch; never run either arm's successor alone.
            name = "reference" if arms["reference"]["epoch"] == arms["ema999"]["epoch"] else "ema999"
            arm = arms[name]
            restore_rng_state(arm["rng"])
            epoch = arm["epoch"]
            endpoint_started = time.perf_counter()
            for group in arm["optimizer"].param_groups:
                group["lr"] = owner.schedule(epoch)
            for batch in _train_loader(roles["train"], epoch, arm["offset"]):
                tick = time.perf_counter()
                loss = _step(arm, batch, mean, std, device, deadline)
                arm["rng"] = capture_rng_state()
                arm["step_trace"].append({"step": arm["steps"], "epoch": epoch,
                    "next_batch": arm["offset"], "samples": arm["samples"],
                    "loss": loss, "seconds": time.perf_counter() - tick,
                    "allocation_seconds": time.time() - started})
                if arm["steps"] % checkpoint_steps == 0 or time.perf_counter() - last_save >= 120:
                    _save(output, arms, binding, runtime, len(roles["train"]), status, started)
                    last_save = time.perf_counter()
            raw, calibrated, selected, calibration = _endpoint(arm, roles, mean, std, device, deadline)
            _check(deadline)
            selection = "live-calibrated" if arm["ema"] is None else "ema-calibrated"
            selected_payload = {"model": selected, "epoch": epoch, "payload": calibrated,
                "calibration": calibration, "binding": binding, "mean": mean, "std": std,
                "selection": selection}
            atomic_torch_save(output / name / f"epoch_{epoch:02d}.pt",
                {"epoch": epoch, "raw": raw, "calibrated": calibrated,
                 "calibration": calibration, "binding": binding, "selected": selected_payload})
            if arm["best"] is None or calibrated["mae_eV"] < arm["best"]:
                arm["best_payload"] = selected_payload
                atomic_torch_save(output / name / "best.pt", selected_payload)
                arm["best"], arm["best_epoch"] = calibrated["mae_eV"], epoch
            arm["trace"].append({"epoch": epoch, "steps": arm["steps"],
                "samples": arm["samples"], "lr": owner.schedule(epoch),
                "train_normalized_l1": arm["train_loss_sum"] / (len(roles["train"]) // owner.BS),
                "raw_mae_eV": raw["mae_eV"], "calibrated_mae_eV": calibrated["mae_eV"],
                "allocation_seconds": time.time() - started,
                "round_seconds": time.perf_counter() - endpoint_started,
                "order_sha256": _sampler(roles["train"], epoch).order_sha256})
            arm["epoch"] += 1
            arm["offset"], arm["train_loss_sum"] = 0, 0.0
            _save(output, arms, binding, runtime, len(roles["train"]), status, started)
        status = "COMPLETE"
    except TimeoutError:
        status = "STOP_FOR_COST"
    except Exception:
        _save(output, arms, binding, runtime, len(roles["train"]), "INCOMPLETE_ERROR", started)
        raise
    return _save(output, arms, binding, runtime, len(roles["train"]), status, started)


def execute(*, mode, dataset_root, initial_path, initial_sha256, runconfig_path,
            runconfig_sha256, source_archive, output, deadline, resume=None,
            resume_sha256=None):
    """Preflight or run on one A100; deadline is absolute Unix time, not a duration.

    Config keys: format, source_commit, source_package_sha256, job_id,
    cpu_accepted_dataset_manifest_sha256, initial_format, initial_state_sha256,
    allocation_started_unix. Its file bytes are pinned by runconfig_sha256.
    Resume uses the same config/allocation window and verified last.pt SHA.
    Output must be the parent's durable Drive directory, not worker-only storage.
    """
    import torch
    if mode not in {"run", "preflight"}:
        raise ValueError("Unknown mode")
    config_path, output = Path(runconfig_path), Path(output)
    if sha256_file(config_path) != runconfig_sha256:
        raise ValueError("Runconfig hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"format", "source_commit", "source_package_sha256", "job_id",
                "cpu_accepted_dataset_manifest_sha256", "initial_format",
                "initial_state_sha256", "allocation_started_unix"}
    if set(config) != required or config["format"] != FORMAT or not config["job_id"]:
        raise ValueError("Runconfig schema/identity changed")
    if type(config["source_commit"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]):
        raise ValueError("Source commit must be a full Git SHA")
    for digest in (config["source_package_sha256"], config["initial_state_sha256"],
                   initial_sha256, runconfig_sha256):
        if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Expected lowercase SHA256 identities")
    if config["cpu_accepted_dataset_manifest_sha256"] != FIXED_500K_MANIFEST_SHA256:
        raise ValueError("CPU acceptance must bind the fixed500K manifest")
    started = config["allocation_started_unix"]
    if not all(type(v) in (float, int) and math.isfinite(v) for v in (started, deadline)) or not (
            started <= time.time() and 0 < deadline - started <= 14400):
        raise ValueError("Deadline must remain within the original four-hour total allocation")
    cutoff = time.perf_counter() + deadline - time.time() - 120
    publication_deadline = cutoff + 105
    if sha256_file(Path(source_archive)) != config["source_package_sha256"]:
        raise ValueError("Source archive hash mismatch")
    if sha256_file(Path(initial_path)) != initial_sha256:
        raise ValueError("Pinned initial file hash mismatch")
    if (resume is None) != (resume_sha256 is None) or (mode == "preflight" and resume is not None):
        raise ValueError("Resume requires run and a checkpoint hash")
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError("Fresh execution requires empty durable output")
    output.mkdir(parents=True, exist_ok=True)
    binding = {"config": config, "config_sha256": runconfig_sha256,
               "initial_file_sha256": initial_sha256, "deadline_unix": deadline}
    if resume is not None and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume must use its existing durable output directory")
    for source, target, digest in (
            (Path(source_archive), output / "source.archive", config["source_package_sha256"]),
            (config_path, output / "runconfig.json", runconfig_sha256),
            (Path(initial_path), output / "initial.pt", initial_sha256)):
        _retain(source, target, digest)
    atomic_json(output / "binding.json", binding)
    try:
        _check(cutoff)
        roles, mean, std, manifest = _data(Path(dataset_root))
        atomic_json(output / "sample_manifest.json", manifest)
        _check(cutoff)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("Exactly one visible A100 required")
        settings = configure_fp32_determinism(42)
        runtime = build_runtime_manifest(settings)
        atomic_json(output / "runtime.json", runtime)
        binding["target_statistics"] = {"mean": mean, "std": std}
        arms = {}
        for name in ARMS:
            _check(cutoff)
            model = owner.make_model("neural_atom_k1")
            load_frozen_initial_state(model, Path(initial_path), expected_file_sha256=initial_sha256,
                expected_state_sha256=config["initial_state_sha256"], expected_format=config["initial_format"])
            configure_fp32_determinism(42)
            arms[name] = _arm(model.to("cuda"))
            if name == "ema999":
                arms[name]["ema"] = make_ema(arms[name]["model"])
        torch.cuda.reset_peak_memory_stats()
        _qualify(arms, roles["train"], mean, std, output, "cuda", cutoff)
        print("A100_PAIRED_K1_QUALIFICATION_PASSED", flush=True)
        if mode == "preflight":
            result = {"status": "PREFLIGHT_COMPLETE", "complete": False, "formal_samples": 0}
        else:
            if resume is not None:
                _resume(Path(resume), resume_sha256, arms, binding, runtime["runtime_fingerprint"], TRAIN_ROWS)
            result = _paired(arms, roles, mean, std, output, binding,
                             runtime["runtime_fingerprint"], "cuda", cutoff, started)
    except TimeoutError:
        result = {"status": "STOP_FOR_COST", "complete": False}
    except Exception as error:
        atomic_json(output / "terminal.json", {"status": "INCOMPLETE_ERROR" if
                    (output / "last.pt").exists() else "NO_TRAIN", "complete": False,
                    "error": str(error), "type": type(error).__name__})
        raise
    result.update({"official_validation_role_read": False, "test_dev_role_read": False,
                   "test_challenge_role_read": False, "elapsed_allocation_seconds": time.time() - started})
    atomic_json(output / "terminal.json", result)
    artifacts = {}
    manifest_complete = True
    for path in output.rglob("*"):
        if path.is_file() and path.name != "artifacts.json" and not path.name.endswith(".tmp"):
            if time.perf_counter() >= publication_deadline:
                manifest_complete = False
                break
            artifacts[str(path.relative_to(output))] = sha256_file(path)
    atomic_json(output / "artifacts.json", {"status": result["status"],
        "manifest_complete": manifest_complete, "artifacts": artifacts})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    for option in ("dataset-root", "initial-path", "runconfig-path", "source-archive", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("initial-sha256", "runconfig-sha256"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--deadline", type=float, required=True, help="Absolute Unix allocation deadline")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = vars(parser.parse_args())
    execute(**args)


if __name__ == "__main__":
    main()
