"""Equal-update 500K EMA study: one live optimizer, two observable filters.

This fills the qualified profile's missing scale sampler/resume owner. Model,
optimizer, schedule, loss, EMA, evaluation, atomic IO and canonical recording
are reused. It is a transfer study, not a claim against a historical 500K scalar.
"""
from pathlib import Path
import json
import time

from . import pcqm_gptrans_v4 as native
from .pcqm_k1_scale_runner import find_cache, load_roles
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .training_reproducibility import (atomic_json, atomic_torch_save, configure_fp32_determinism,
    capture_rng_state, restore_rng_state, sha256_file, assert_finite_state_dict)
from .research_memory.trace import RMLTraceRecorder

TOTAL_STEPS, RUNG_STEPS, BATCH = 46860, 781, 128
FILTERS = {"ema9999": .9999, "ema999": .999}


def rung_indices(rung):
    import torch
    if type(rung) is not int or not 0 <= rung < 60:
        raise ValueError("Scale rung must lie in0..59")
    cycle_batches = 500000 // BATCH
    cursor, remaining = rung * RUNG_STEPS, RUNG_STEPS
    chunks = []
    while remaining:
        cycle, offset = divmod(cursor, cycle_batches)
        take = min(remaining, cycle_batches - offset)
        permutation = torch.randperm(500000, generator=torch.Generator().manual_seed(42 + cycle))
        chunks.append(permutation[offset * BATCH:(offset + take) * BATCH])
        cursor, remaining = cursor + take, remaining - take
    return torch.cat(chunks).tolist()


def _recorder(path, view, trajectory_id, run_id):
    def metric(weights, role):
        return {"metric": "MAE", "unit": "eV", "target": "Gap", "role_identity": role,
                "weights": weights, "direction": "minimize"}
    return RMLTraceRecorder(path, trajectory_id=trajectory_id, run_id=run_id,
        metric_semantics={"live_train_metric": metric("live", "fixed500k-train-0-500000"),
            "live_dev_metric": metric("live", "fixed500k-development-500000-550000"),
            "ema_dev_metric": metric("ema", "fixed500k-development-500000-550000")})


def train(inputs, output, *, config, source_identity):
    import torch
    from torch_geometric.loader import DataLoader
    from .gptrans_scale_profile import model_binding, profile
    variant, initial_file, parameters = model_binding(config)
    started = time.perf_counter()
    initial, transform = inputs / initial_file, inputs / "target_transform.json"
    if sha256_file(initial) != config["initial_file_sha256"]:
        raise ValueError("Scale initialization differs from the frozen plan")
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("One visible T4 per physical study")
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion_manifest.json"
    if completion.exists():
        result = json.loads(completion.read_bytes())
        if result["source_identity"] != source_identity or result["configuration"] != config:
            raise ValueError("Completed study binding changed")
        return result
    qualification_path = output / "qualification.json"
    if not qualification_path.exists():
        qualification = profile(inputs, output, config, source_identity=source_identity)
    else:
        qualification = json.loads(qualification_path.read_bytes())
    if (not qualification["qualification_passed"] or qualification["source_identity"] != source_identity
            or qualification["parameters"] != parameters):
        raise ValueError("Qualified scale execution/budget binding failed")
    root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    roles = load_roles(root, manifest)
    configure_fp32_determinism(42)
    mean_value, std_value = native._run_target_stats([], variant, transform)
    mean, std = torch.tensor(mean_value, device="cuda"), torch.tensor(std_value, device="cuda")
    model, optimizer, scheduler, unused = native._make_training_state(initial, variant)
    del unused
    averages = {name: native.ExponentialMovingAverage(model, decay) for name, decay in FILTERS.items()}
    records = {view: [] for view in FILTERS}
    best = {view: {"mae_eV": float("inf"), "rung": -1} for view in FILTERS}
    checkpoint = output / "last_checkpoint.pt"
    start, retained_wall_seconds = 0, 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
        if saved["configuration"] != config or saved["source_identity"] != source_identity:
            raise ValueError("Resume contract/source identity changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        for view, average in averages.items():
            average.load_state_dict(saved["ema"][view])
        records, best, start = saved["records"], saved["best"], saved["next_rung"]
        if start:
            retained_wall_seconds = max(rows[-1]["canonical"]["cumulative_wall_time_seconds"] for rows in records.values())
        restore_rng_state(saved["rng_state"])
    recorders = {view: _recorder(output / view / "canonical_trace.json", view,
        config["view_trajectories"][view], config["logical_run_id"]) for view in FILTERS}
    # Recover an interrupted canonical publication from the retained checkpoint.
    for view, recorder in recorders.items():
        missing = len(records[view]) - len(recorder.record["observations"])
        if missing not in (0, 1):
            raise ValueError("Scale trace/resume cursor diverged")
        if missing:
            row = records[view][-1]
            recorder.checkpoint_event(**{**row["canonical"], "checkpoint_identity": "sha256:" + sha256_file(checkpoint)})
    for rung in range(start, 60):
        began = time.perf_counter()
        scheduler.step(rung)
        loader = DataLoader(roles["train"], batch_size=BATCH, sampler=rung_indices(rung),
            num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(42 + rung))
        model.train()
        total = torch.zeros((), device="cuda")
        count = 0
        profile_steps = config.get("phase_profiling_steps", 0) if rung == 0 and start == 0 else 0
        if profile_steps not in (0, 4):
            raise ValueError("Phase timing is limited to the prospectively declared first four updates")
        profiler = None
        if profile_steps:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
            profiler.start()
        for batch in loader:
            if int(batch.num_graphs) != BATCH:
                raise ValueError("Physical batch/tail identity changed")
            batch = batch.to("cuda", non_blocking=True)
            loss = native._optimizer_step(model, optimizer, averages["ema9999"], batch, mean, std, check_finite=True)
            averages["ema999"].update(model)
            total.add_(loss * BATCH)
            count += BATCH
            if profiler is not None and count == profile_steps * BATCH:
                profiler.stop()
                profiler.export_chrome_trace(str(output / "optimizer_phase_profile.json"))
                profiler = None
        training_seconds = time.perf_counter() - began
        if count != RUNG_STEPS * BATCH:
            raise ValueError("Scale exposure changed")
        live = native._evaluate(model, averages["ema999"], roles["validation"], mean, std,
                                weights="live", source_idx_start=500000)
        scores = {view: native._evaluate(model, avg, roles["validation"], mean, std,
                            source_idx_start=500000) for view, avg in averages.items()}
        evaluation_seconds = time.perf_counter() - began - training_seconds
        atomic_json(output / "timing" / f"rung_{rung:02d}.json", {
            "training_including_loader_h2d_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds, "profiling_updates": profile_steps,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "timing_scope": "native operator profile separates forward/autograd/AdamW; aggregate train includes loader/H2D",
            "optimizer_steps": (rung + 1) * RUNG_STEPS})
        for view, score in scores.items():
            if score["mae_eV"] < best[view]["mae_eV"]:
                best[view] = {"mae_eV": score["mae_eV"], "rung": rung}
                atomic_torch_save(output / view / "best_model.pt", {"model": averages[view].state_dict(),
                    "target_stats": {"mean_eV": mean_value, "sample_std_eV": std_value}, "source_identity": source_identity,
                    "configuration": config, "rung": rung, "ema_decay": FILTERS[view]})
                atomic_torch_save(output / view / "development_predictions.pt", {"prediction_eV": score["prediction_eV"],
                    "target_eV": score["target_eV"], "source_idx": score["source_idx"], "official_validation_role_read": False,
                    "test_dev_role_read": False, "test_challenge_role_read": False})
            canonical = {"optimizer_step": (rung + 1) * RUNG_STEPS,
                "sample_presentations": (rung + 1) * RUNG_STEPS * BATCH,
                "epoch_or_pass": (rung + 1) * RUNG_STEPS / (500000 // BATCH),
                "learning_rate": scheduler.learning_rate(rung), "live_train_metric": float(total.cpu()) * std_value / count,
                "live_dev_metric": live["mae_eV"], "ema_dev_metric": score["mae_eV"],
                "cumulative_wall_time_seconds": retained_wall_seconds + time.perf_counter() - started}
            records[view].append({"rung": rung, "canonical": canonical, "ema_decay": FILTERS[view],
                                  "elapsed_seconds": time.perf_counter() - began})
            atomic_json(output / view / "trace.json", {"rows": records[view]})
        assert_finite_state_dict(model.state_dict(), label="scale live")
        for average in averages.values():
            assert_finite_state_dict(average.state_dict(), label="scale EMA")
        atomic_torch_save(checkpoint, {"format": "molgap-scale-ema-checkpoint-v1", "source_identity": source_identity,
            "configuration": config, "next_rung": rung + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "ema": {v: avg.state_dict() for v, avg in averages.items()},
            "best": best, "records": records, "rng_state": capture_rng_state()})
        digest = sha256_file(checkpoint)
        for view, recorder in recorders.items():
            recorder.checkpoint_event(**records[view][-1]["canonical"], checkpoint_identity="sha256:" + digest)
        if rung % 10 == 9:
            import shutil
            chunk = output / f"checkpoint_rung_{rung:02d}.pt"
            temporary = chunk.with_suffix(".tmp")
            shutil.copyfile(checkpoint, temporary)
            temporary.replace(chunk)
        print(f"scale-ema rung={rung:02d} live={live['mae_eV']:.8f} slow={scores['ema9999']['mae_eV']:.8f} fast={scores['ema999']['mae_eV']:.8f}", flush=True)
    result = {"format": "molgap-scale-ema-completion-v1", "complete": True, "configuration": config,
        "source_identity": source_identity, "optimizer_steps": TOTAL_STEPS, "sample_presentations": TOTAL_STEPS * BATCH,
        "live_optimizer_streams": 1, "parameters": parameters, "manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "best": best, "training_executed": True, "development_role_read": True,
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False}
    result["files"] = {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*")
                       if p.is_file() and p != completion}
    atomic_json(completion, result)
    return result
