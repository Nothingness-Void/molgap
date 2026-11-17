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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.output)
    except Exception as error:
        from .training_reproducibility import atomic_json
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
