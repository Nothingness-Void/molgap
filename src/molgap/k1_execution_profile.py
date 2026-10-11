"""Bounded execution diagnostics; never a trainer or a model acceptance gate."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

CASES = (("single_w2", 1, 2), ("double_w0", 2, 0),
         ("double_w2", 2, 2), ("double_w4", 2, 4))
WARMUP, MEASURE, BATCH_SIZE = 5, 24, 128


def gradient_relation(supervised, regularizer):
    """Report scaled regularizer norm and alignment, including zero norms."""
    import torch
    a = torch.cat([v.detach().reshape(-1) for v in supervised])
    b = torch.cat([v.detach().reshape(-1) for v in regularizer])
    an, bn = float(a.norm()), float(b.norm())
    return {"supervised_norm": an, "weighted_consistency_norm": bn,
            "norm_ratio": bn / an if an else None,
            "cosine": float(torch.dot(a, b) / (an * bn)) if an and bn else None}


def run(root: Path, output: Path):
    import torch
    from torch_geometric.loader import DataLoader
    from .k1_frozen_inference import load_native500k_k1
    from .k1_screen_training import _forward
    from .k1_pretrained_combo import objective
    from .training_reproducibility import atomic_json, configure_fp32_determinism, sha256_file
    from .v4_runtime import state_dict_sha256

    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + 1200
    manifest = json.loads((root / "payload_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Payload bytes differ: {name}")
    trajectory = json.loads((root / "prospective/trajectory.json").read_text())
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Completed diagnostic must not be overwritten")
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("Actual A100 required")
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "determinism": settings,
               "payload_sha256": sha256_file(root / "payload_manifest.json"),
               "started_at_unix": time.time(), "scientific_training": False}
    atomic_json(output / "runtime.json", runtime)
    graphs = torch.load(root / "train_probe.pt", map_location="cpu", weights_only=False)
    expected_rows = manifest["sample_source_idx"]
    if len(graphs) != 4096 or [int(g.source_idx) for g in graphs] != expected_rows:
        raise ValueError("Fixed training sample order differs")
    if any(not 0 <= int(g.source_idx) < 500000 or "pos" in g for g in graphs):
        raise ValueError("Only pure2D training members permitted")
    state_meta = manifest["checkpoint"]

    def fresh():
        configure_fp32_determinism(42)
        model, meta = load_native500k_k1(root / "selected.pt",
            expected_sha256=state_meta["sha256"], expected_source_sha256=state_meta["source_sha256"],
            expected_epoch=48, checkpoint_kind="selected")
        model.requires_grad_(True).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5,
                                      foreach=False, fused=False)
        return model, optimizer, meta

    def loss_for(model, batch, passes, meta):
        target = (batch.y.view(-1) - meta["mean"]) / meta["std"]
        first = _forward(model, batch)
        if passes == 1:
            return torch.nn.functional.l1_loss(first, target)
        second = _forward(model, batch)
        return objective(first, second, target, mode="pretrained_consistency")[0]

    def loader(workers):
        args = dict(batch_size=128, shuffle=False, drop_last=True, num_workers=workers,
                    pin_memory=True, persistent_workers=bool(workers))
        if workers:
            args["prefetch_factor"] = 2
        return DataLoader(graphs, **args)

    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Bounded profile ceiling exhausted")

    results = []
    for name, passes, workers in CASES:
        guard()
        model, optimizer, meta = fresh()
        iterator = iter(loader(workers))
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(WARMUP + MEASURE):
            guard()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = next(iterator)
            t1 = time.perf_counter()
            batch = batch.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, batch, passes, meta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite probe")
            if step >= WARMUP:
                samples.append({"step_s": t2-t0, "loader_wait_s": t1-t0,
                                "transfer_compute_s": t2-t1})
            if (step + 1) % 8 == 0:
                print(f"{name}: {step+1}/{WARMUP+MEASURE}", flush=True)
        row = {"case": name, "passes": passes, "workers": workers, "steps": samples,
               "median_step_s": statistics.median(x["step_s"] for x in samples),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        results.append(row)
        atomic_json(output / f"{name}.json", row)
        print(json.dumps({k:v for k,v in row.items() if k != "steps"}), flush=True)
        del iterator, model, optimizer, batch, loss
        gc.collect()
        torch.cuda.empty_cache()

    # Synchronization intentionally separates phases; these are not throughput measurements.
    model, optimizer, meta = fresh()
    iterator = iter(loader(2))
    phases = []
    for step in range(8):
        guard()
        stamps = [time.perf_counter()]
        batch = next(iterator); stamps.append(time.perf_counter())
        batch = batch.cuda(non_blocking=True); torch.cuda.synchronize(); stamps.append(time.perf_counter())
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, 2, meta); torch.cuda.synchronize(); stamps.append(time.perf_counter())
        loss.backward(); torch.cuda.synchronize(); stamps.append(time.perf_counter())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize(); stamps.append(time.perf_counter())
        optimizer.step(); torch.cuda.synchronize(); stamps.append(time.perf_counter())
        if step >= 2:
            phases.append(dict(zip(("loader", "h2d", "forward_loss", "backward", "clip", "optimizer"),
                                   (b-a for a,b in zip(stamps, stamps[1:])))))
    atomic_json(output / "phase_timings.json", {"synchronized_instrumentation": True, "samples": phases})
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(3):
            guard()
            batch = next(iterator).cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, batch, 2, meta)
            loss.backward(); optimizer.step(); torch.cuda.synchronize(); prof.step()
    (output / "operator_table.txt").write_text(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=35))
    del iterator, model, optimizer, batch, loss
    gc.collect(); torch.cuda.empty_cache()

    model, optimizer, meta = fresh()
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    before = state_dict_sha256(model.state_dict())
    gradient_rows = []
    for index, batch in enumerate(loader(0)):
        if index == 4:
            break
        guard()
        batch = batch.cuda()
        target = (batch.y.view(-1) - meta["mean"]) / meta["std"]
        first, second = _forward(model, batch), _forward(model, batch)
        _, pieces = objective(first, second, target, mode="pretrained_consistency")
        a = torch.autograd.grad(pieces["supervised_l1"], parameters, retain_graph=True, allow_unused=True)
        b = torch.autograd.grad(0.1*pieces["disagreement"], parameters, allow_unused=True)
        a = [torch.zeros_like(p) if g is None else g for p,g in zip(parameters,a)]
        b = [torch.zeros_like(p) if g is None else g for p,g in zip(parameters,b)]
        gradient_rows.append({"batch": index, **gradient_relation(a,b)})
    # Probe BN updates are discarded with the entire scratch model. No accepted state is overwritten.
    gradient_info = {"selected_checkpoint_posthoc": True, "optimizer_updates": 0,
        "bn_buffers_updated_by_train_forward": before != state_dict_sha256(model.state_dict()),
        "batches": gradient_rows, "causal_accuracy_claim": False}
    atomic_json(output / "gradient_relation.json", gradient_info)
    elapsed = time.perf_counter()-started
    result = {"format": "molgap-k1-execution-profile-v1", "status": "complete",
        "cases": results, "phase_samples": phases, "gradient_relation": gradient_info,
        "cost": {"allocated_A100_worker_seconds": elapsed,
                 "parent_cpu_seconds": time.process_time()-cpu_started,
                 "cpu_children_seconds": None, "gpu_busy_seconds": None},
        "training_rows": expected_rows, "development_rows": [], "protected_roles": "untouched",
        "accuracy_acceptance": False, "training_replay_ready": False,
        "limits": ["A100 != historical T4 runtime", "bounded train subset",
                   "single-pass changes objective and BN updates", "selected epoch49 posthoc gradients",
                   "no validation or checkpoint-publication timing", "no full epoch cost extrapolation"]}
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", {"status": "complete",
        # The notebook owns the live log and process receipt; they finish after this worker exits.
        "artifacts": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()
                      and p.name not in {"completion.json", "worker.log", "worker_process_observation.json"}}})
    print("K1_PROFILE_COMPLETE", flush=True)


def _profile_synthetic_operators(output, modes, fresh, step):
    """Instrument scratch steps separately; profiled time is not throughput."""
    import cProfile
    import io
    import pstats
    import torch
    from .training_reproducibility import atomic_json

    rows = {}
    for mode in modes:
        model, optimizer = fresh()
        for _ in range(3):
            step(model, optimizer, mode)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            for _ in range(3):
                step(model, optimizer, mode)
            torch.cuda.synchronize()
        events = profile.key_averages()
        rows[mode] = [{"name": event.key, "calls": event.count,
            "self_cpu_us": event.self_cpu_time_total,
            "total_cpu_us": event.cpu_time_total,
            "self_device_us": event.self_device_time_total,
            "total_device_us": event.device_time_total} for event in events]
        for column in ("self_cpu_time_total", "self_device_time_total"):
            (output / f"{mode}_{column}.txt").write_text(
                events.table(sort_by=column, row_limit=35), encoding="utf-8")
        profile.export_chrome_trace(str(output / f"{mode}_trace.json"))
        python_profile = cProfile.Profile()
        python_profile.enable()
        for _ in range(3):
            step(model, optimizer, mode)
        torch.cuda.synchronize()
        python_profile.disable()
        stream = io.StringIO()
        pstats.Stats(python_profile, stream=stream).sort_stats("tottime").print_stats(45)
        (output / f"{mode}_python.txt").write_text(stream.getvalue(), encoding="utf-8")
        del model, optimizer
    device_observed = any(event["self_device_us"] > 0 for case in rows.values() for event in case)
    atomic_json(output / "operators.json", {"steps_per_case": 3,
        "instrumentation_not_throughput": True, "device_activity_observed": device_observed,
        "device_activity_limit": None if device_observed else
            "CUDA profiler activity unavailable; CPU records are not GPU kernel times",
        "cases": rows})


def run_synthetic_local(output: Path, *, maximum_seconds: int = 180, deep_profile=False):
    """Compare scratch execution paths without datasets, checkpoints or role access."""
    import copy
    from contextlib import nullcontext
    import torch
    from torch_geometric.data import Batch, Data
    from .qm9_neural_atom import make_encoder
    from .k1_execution_layout import cpu_layout_context
    from .k1_screen_training import (_evaluate, _model_batch_to_device, _optimizer_step,
        _screen_step, qualify_initialized_layout)
    from .training_reproducibility import (atomic_json, configure_fp32_determinism,
        deterministic_memory_fill_context, capture_rng_state, restore_rng_state, sha256_file)

    if type(maximum_seconds) is not int or not 30 <= maximum_seconds <= 180:
        raise ValueError("Synthetic local ceiling must be 30..180 seconds")
    if type(deep_profile) is not bool:
        raise ValueError("deep_profile must be an exact bool")
    if output.exists():
        raise FileExistsError("Use a fresh synthetic profile output directory")
    started = time.perf_counter()
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5060":
        raise ValueError("This bounded synthetic profile is qualified only for local RTX5060")
    output.mkdir(parents=True)
    graphs = []
    for index in range(BATCH_SIZE):
        n = 8 + (index * 13) % 33
        chain = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
        ring = torch.tensor([[0, 2], [n - 1, n - 3]])
        forward = torch.cat((chain, ring), dim=1)
        edges = torch.cat((forward, forward.flip(0)), dim=1)
        graphs.append(Data(x=torch.zeros(n, 9, dtype=torch.long), edge_index=edges,
            edge_attr=torch.zeros(edges.shape[1], 3, dtype=torch.long),
            random_walk_pe=torch.rand(n, 16), y=torch.randn(1),
            source_idx=torch.tensor([index])))
    cpu_batch = Batch.from_data_list(graphs)
    initial = make_encoder("neural_atom_k1").train()
    for mixer in initial.neural_atom_mixers.values():
        torch.nn.init.normal_(mixer.return_projection.weight, std=0.01)
    modes = ("original", "cpu_layout", "initialized_layout")

    def fresh():
        configure_fp32_determinism(42)
        model = copy.deepcopy(initial).cuda().train()
        return model, torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)

    def step(model, optimizer, mode):
        if time.perf_counter() - started >= maximum_seconds:
            raise TimeoutError("Synthetic profile ceiling exhausted; no automatic continuation")
        with deterministic_memory_fill_context(enabled=True):
            batch = cpu_batch.clone()
            if mode == "initialized_layout":
                return _screen_step(model, optimizer, batch, 0., 1., mode)
            scope = (nullcontext() if mode == "original" else cpu_layout_context(
                model, batch, "cuda"))
            with scope:
                _model_batch_to_device(batch, "cuda", training=True)
                return _optimizer_step(model, optimizer, batch, 0., 1.)

    def compare(first, second):
        different, maximum = 0, 0.
        for a, b in zip(first, second):
            if (a is None) != (b is None):
                raise ValueError("Gradient participation changed")
            if a is not None:
                if not bool(torch.isfinite(b).all()):
                    raise ValueError("Nonfinite synthetic state")
                different += int(not torch.equal(a, b))
                maximum = max(maximum, float((a.double() - b.double()).abs().max()))
        return {"different_tensors": different, "max_absolute_difference": maximum}

    if deep_profile:
        _profile_synthetic_operators(output, modes, fresh, step)
    parity = {}
    snapshots = {}
    for mode in modes:
        model, optimizer = fresh()
        loss, absolute, count = step(model, optimizer, mode)
        snapshots[mode] = {
            "state": [v.detach().cpu().clone() for v in model.state_dict().values()],
            "grad": [None if p.grad is None else p.grad.detach().cpu().clone()
                     for p in model.parameters()],
            "loss": float(loss), "absolute": float(absolute), "count": count,
            "rng": torch.cuda.get_rng_state().clone(), "cpu_rng": torch.get_rng_state().clone()}
        del model, optimizer
    reference = snapshots["original"]
    for mode in modes[1:]:
        row = snapshots[mode]
        parity[mode] = {"state": compare(reference["state"], row["state"]),
            "grad": compare(reference["grad"], row["grad"]),
            "loss_equal": reference["loss"] == row["loss"],
            "absolute_equal": reference["absolute"] == row["absolute"],
            "rng_equal": torch.equal(reference["rng"], row["rng"])
                and torch.equal(reference["cpu_rng"], row["cpu_rng"])}
    del snapshots, reference
    # Include tiny/ragged graphs and an edgeless batch, not only one fixed shape.
    edgeless = Batch.from_data_list([Data(x=torch.zeros(n, 9, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, 3, dtype=torch.long), random_walk_pe=torch.rand(n, 16),
        y=torch.randn(1)) for n in (1, 2, 5)])
    fixtures = [cpu_batch, Batch.from_data_list(graphs[:3]), edgeless,
                Batch.from_data_list(graphs[10:26])] * 2
    multi_step = qualify_initialized_layout(initial.state_dict(), fixtures, 0., 1.)
    atomic_json(output / "multi_step_equivalence.json", multi_step)
    # Exercise actual serialization on scratch states; never publish a trained model.
    import io
    from .experiment_family_workflow import tensor_safe_rng_state
    from .k1_screen_training import _step_identity
    model, optimizer = fresh()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    step(model, optimizer, "initialized_layout")
    scheduler.step()
    checkpoint = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": tensor_safe_rng_state(capture_rng_state())}, checkpoint)
    expected_step = _step_identity(model, optimizer, *step(model, optimizer, "initialized_layout"))
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    restore_rng_state(restored["rng"])
    resumed_step = _step_identity(model, optimizer, *step(model, optimizer, "initialized_layout"))
    if expected_step != resumed_step:
        raise ValueError("Synthetic initialized-layout resume changed exact step identity")
    resume_equivalence = {"accepted": True, "uninterrupted": expected_step, "resumed": resumed_step,
                          "scope": "scratch model/AdamW/cosine/RNG BytesIO serialization"}
    atomic_json(output / "resume_equivalence.json", resume_equivalence)
    del model, optimizer, scheduler, checkpoint, restored
    timings = []
    # Counterbalance mode order; synchronize block boundaries, not every step.
    for round_index, order in enumerate((modes, modes[::-1], modes[1:] + modes[:1])):
        for mode in order:
            model, optimizer = fresh()
            for _ in range(WARMUP):
                step(model, optimizer, mode)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            metrics = [step(model, optimizer, mode)[1] for _ in range(MEASURE)]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if not bool(torch.isfinite(torch.stack(metrics)).all()):
                raise ValueError("Nonfinite synthetic metric")
            row = {"round": round_index, "mode": mode, "steps": MEASURE,
                   "seconds": elapsed, "seconds_per_step": elapsed / MEASURE,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            timings.append(row)
            atomic_json(output / "timings.json", timings)
            print(json.dumps(row), flush=True)
            del metrics, model, optimizer
    medians = {mode: statistics.median(r["seconds_per_step"] for r in timings
                                      if r["mode"] == mode) for mode in modes}
    evaluation_rows = []
    expected = None
    for order in ((None, 512, 4096), (4096, 512, None), (512, None, 4096)):
        for chunk in order:
            model, optimizer = fresh()

            def batches():
                for index in range(32):
                    if time.perf_counter() - started >= maximum_seconds:
                        raise TimeoutError("Synthetic profile ceiling exhausted")
                    batch = cpu_batch.clone()
                    batch.source_idx += index * BATCH_SIZE
                    yield batch

            _evaluate(model, [cpu_batch.clone()], 0.17, 1.23, prediction_chunk_rows=chunk)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            observed = _evaluate(model, batches(), 0.17, 1.23, prediction_chunk_rows=chunk)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if expected is None:
                expected = observed
            if (expected[0] != observed[0] or not all(torch.equal(a, b)
                    for a, b in zip(expected[1:], observed[1:]))):
                raise ValueError("Prediction copy chunk changed evaluation output")
            evaluation_rows.append({"prediction_chunk_rows": chunk, "seconds": elapsed,
                                    "rows": 32 * BATCH_SIZE, "output_bitwise_equal": True})
            del model, optimizer
    evaluation_medians = {str(chunk): statistics.median(r["seconds"] for r in evaluation_rows
        if r["prediction_chunk_rows"] == chunk) for chunk in (None, 512, 4096)}
    result = {"format": "molgap-k1-synthetic-execution-profile-v1", "status": "complete",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "determinism": settings},
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
            ("k1_execution_profile.py", "k1_execution_layout.py", "k1_screen_training.py",
             "gps.py", "qm9_neural_atom.py", "training_reproducibility.py")},
        "shape": {"graphs": BATCH_SIZE, "nodes": cpu_batch.num_nodes,
                  "directed_edges": cpu_batch.num_edges, "max_nodes": 40},
        "optimizer": "AdamW native defaults; lr=4e-4 weight_decay=1e-5",
        "memory_fill_policy": {mode: mode != "initialized_layout" for mode in modes},
        "multi_step_equivalence": multi_step,
        "resume_equivalence": resume_equivalence,
        "parity": parity, "timings": timings, "median_seconds_per_step": medians,
        "time_reduction_vs_original": {mode: 1 - medians[mode] / medians["original"]
                                       for mode in modes[1:]},
        "synthetic_evaluation": {"timings": evaluation_rows, "median_seconds": evaluation_medians,
            "time_reduction_vs_batch_copy": {str(chunk): 1 - evaluation_medians[str(chunk)] /
                evaluation_medians["None"] for chunk in (512, 4096)}},
        "local_gpu_wall_seconds": time.perf_counter() - started,
        "quality_claim": False, "rml_record": False, "molecular_roles_consumed": [],
        "deep_profile": deep_profile,
        "limits": ["synthetic single batch including clone/H2D/layout, no data loading",
                   "nonexclusive desktop GPU; not full training throughput",
                   "numerical parity is not MAE or seed variance"]}
    atomic_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-local", action="store_true")
    parser.add_argument("--deep-profile", action="store_true")
    args = parser.parse_args()
    if args.synthetic_local == (args.root is not None):
        parser.error("Choose exactly one of --root or --synthetic-local")
    if args.deep_profile and not args.synthetic_local:
        parser.error("--deep-profile requires --synthetic-local")
    if args.synthetic_local and args.output.exists():
        parser.error("Use a fresh synthetic profile output directory")
    try:
        if args.synthetic_local:
            run_synthetic_local(args.output, deep_profile=args.deep_profile)
        else:
            run(args.root, args.output)
    except Exception as error:
        from .training_reproducibility import atomic_json
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
