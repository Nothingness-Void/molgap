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
T4_CASES = (("single_w2", 1, 2), ("double_mean2_w2", 2, 2))


def verify_t4_payload(root, expected_manifest_sha256):
    """Verify the caller-pinned executable inventory before any pickle load."""
    from .training_reproducibility import sha256_file
    from .evidence_pointers import resolve_repo_pointer
    root = Path(root).resolve()
    if not expected_manifest_sha256 or sha256_file(root / "payload_manifest.json") != expected_manifest_sha256:
        raise ValueError("Pinned payload manifest required")
    manifest = json.loads((root / "payload_manifest.json").read_text())
    if manifest.get("format") != "molgap-k1-native-t4-profile-payload-v1":
        raise ValueError("Expected native-T4 payload format")
    sources = manifest["source_files"]
    if (not isinstance(sources, list) or not sources or len(set(sources)) != len(sources)
            or len({name.casefold() for name in sources}) != len(sources)
            or any(not name.startswith("src/molgap/") or not name.endswith(".py")
                   or "\\" in name or ":" in name or any(part in {"", ".", ".."}
                   for part in name.split("/")) for name in sources)):
        raise ValueError("Expected explicit unique POSIX Python source inventory")
    required = {"src/molgap/k1_execution_profile.py", "src/molgap/k1_frozen_inference.py",
                "src/molgap/pcqm_wedge.py", "src/molgap/k1_screen_training.py",
                "src/molgap/k1_pretrained_combo.py"}
    if not required <= set(sources):
        raise ValueError("Missing executable or WedgeData dependency")
    files = manifest["files"]
    if not set(sources) | {"selected.pt", "train_probe.pt", "prospective/trajectory.json",
                          "bootstrap.py", "run.sh", "setup.sh", "kaggle_entry.py", "protocol.md"} <= set(files):
        raise ValueError("Incomplete payload inventory")
    for name, digest in files.items():
        path = resolve_repo_pointer(root, name)
        if path is None or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"Payload bytes differ: {name}")
    # Reject an unpinned Python module that could shadow a reviewed dependency.
    if {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")} != set(sources):
        raise ValueError("Source inventory differs")
    if manifest["checkpoint"]["sha256"] != files["selected.pt"]:
        raise ValueError("Checkpoint pin differs")
    rows = manifest["sample_source_idx"]
    if len(rows) != 4096 or len(set(rows)) != 4096 or any(type(i) is not int or not 0 <= i < 500000 for i in rows):
        raise ValueError("Expected fixed unique train4096 sample")
    trajectory = json.loads((root / "prospective/trajectory.json").read_text())
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    return manifest


def mean2_loss(first, second, target):
    """Two independently stochastic forwards, mean supervised L1, coefficient0."""
    from .k1_pretrained_combo import objective
    # The frozen owner has no coefficient parameter. Never backprop its combined
    # default0.1 loss: reuse only its exact supervised mean-two component.
    _, pieces = objective(first, second, target, mode="pretrained_consistency")
    return pieces["supervised_l1"]


def verify_loaded_sources(root, manifest):
    """Prevent current-checkout/frozen-owner mixing before deserialization."""
    import sys
    from .training_reproducibility import sha256_file
    root = Path(root).resolve()
    for name, module in list(sys.modules.items()):
        if name == "molgap" or name.startswith("molgap."):
            path = getattr(module, "__file__", None)
            if path is None:
                continue
            path = Path(path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Imported nonpayload source: {name}")
            relative = path.relative_to(root).as_posix()
            if relative not in manifest["source_files"] or sha256_file(path) != manifest["files"].get(relative):
                raise ValueError(f"Imported unpinned source: {name}")


def t4_cost(elapsed, count, case_seconds, *, phase_seconds=0.0, eval_write_seconds=0.0):
    active_seconds = case_seconds + phase_seconds + eval_write_seconds
    return {"worker_wall_seconds": elapsed, "visible_gpu_count": count,
            "allocated_T4_device_hours": elapsed * count / 3600,
            "active_case_T4_device_hours": case_seconds / 3600,
            "active_case_definition": "one device case wall time, including loader; not GPU busy",
            "active_work_T4_device_hours": active_seconds / 3600,
            "active_work_seconds": {"cases": case_seconds, "synchronized_phases": phase_seconds,
                                    "train_eval_localFS_write": eval_write_seconds},
            "active_work_definition": "cuda0 work windows including scratch load, loader, phases, eval, localFS write and teardown; not GPU busy",
            "gpu_busy_seconds": None, "setup_device_hours": None,
            "global_cost_extrapolation": False}


def _synchronized_phases(model, optimizer, iterator, meta, *, loss_for, guard):
    """Internal shared instrumentation; two warmup steps and six recorded steps."""
    import torch
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
    return phases, batch, loss


def gradient_relation(supervised, regularizer):
    """Report scaled regularizer norm and alignment, including zero norms."""
    import torch
    a = torch.cat([v.detach().reshape(-1) for v in supervised])
    b = torch.cat([v.detach().reshape(-1) for v in regularizer])
    an, bn = float(a.norm()), float(b.norm())
    return {"supervised_norm": an, "weighted_consistency_norm": bn,
            "norm_ratio": bn / an if an else None,
            "cosine": float(torch.dot(a, b) / (an * bn)) if an and bn else None}


def run(root: Path, output: Path, *, native_t4=False, expected_manifest_sha256=None):
    started, cpu_started = time.perf_counter(), time.process_time()
    if native_t4:
        verify_t4_payload(root, expected_manifest_sha256)
    import torch
    from torch_geometric.loader import DataLoader
    from .k1_frozen_inference import load_native500k_k1
    from .k1_screen_training import _forward
    from .k1_pretrained_combo import objective
    from .training_reproducibility import atomic_json, configure_fp32_determinism, sha256_file
    from .v4_runtime import state_dict_sha256

    if not native_t4:
        started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + (900 if native_t4 else 1200)
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
    hardware = "T4" if native_t4 else "A100"
    if not torch.cuda.is_available() or hardware not in torch.cuda.get_device_name(0):
        raise ValueError(f"Actual {hardware} required")
    if native_t4 and (torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))):
        raise ValueError("Observed two native T4 devices required")
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "determinism": settings,
               "payload_sha256": sha256_file(root / "payload_manifest.json"),
               "started_at_unix": time.time(), "scientific_training": False}
    if native_t4:
        runtime.update(visible_gpu_count=torch.cuda.device_count(), active_gpu_indices=[0],
                       gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    atomic_json(output / "runtime.json", runtime)
    if native_t4:
        verify_loaded_sources(root, manifest)
    graphs = torch.load(root / "train_probe.pt", map_location="cpu", weights_only=False)
    expected_rows = manifest["sample_source_idx"]
    if len(graphs) != 4096 or [int(g.source_idx) for g in graphs] != expected_rows:
        raise ValueError("Fixed training sample order differs")
    if any(not 0 <= int(g.source_idx) < 500000 or "pos" in g for g in graphs):
        raise ValueError("Only pure2D training members permitted")
    if native_t4:
        from .k1_screen_training import FORBIDDEN_MODEL_FIELDS
        if any(any(field in g for field in FORBIDDEN_MODEL_FIELDS) for g in graphs):
            raise ValueError("Geometry fields forbidden")
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
        if native_t4:
            return mean2_loss(first, second, target)
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
    case_seconds = 0.0
    for name, passes, workers in (T4_CASES if native_t4 else CASES):
        guard()
        case_started = time.perf_counter()
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
        case_seconds += time.perf_counter() - case_started

    if native_t4:
        guard()
        phase_started = time.perf_counter()
        model, optimizer, meta = fresh()
        iterator = iter(loader(2))
        phases, batch, loss = _synchronized_phases(model, optimizer, iterator, meta,
                                                  loss_for=loss_for, guard=guard)
        atomic_json(output / "phase_timings.json", {"synchronized_instrumentation": True,
            "warmup_steps": 2, "optimizer_steps": 8, "consistency_coefficient": 0, "samples": phases})
        del iterator, model, optimizer, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        phase_seconds = time.perf_counter() - phase_started
        guard()
        eval_write_started = time.perf_counter()
        model, optimizer, meta = fresh()
        model.eval().requires_grad_(False)
        eval_samples = []
        iterator = iter(loader(2))
        with torch.no_grad():
            for _ in range(4):
                guard()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                batch = next(iterator).cuda(non_blocking=True)
                prediction = _forward(model, batch)
                torch.cuda.synchronize()
                if not bool(torch.isfinite(prediction).all()):
                    raise ValueError("Nonfinite eval timing probe")
                eval_samples.append(time.perf_counter() - t0)
        from .training_reproducibility import atomic_torch_save
        guard()
        t0 = time.perf_counter()
        atomic_torch_save(output / "scratch_publication_probe.pt", {
            "model": model.state_dict(), "scientific_training": False,
            "scope": "fresh selected scratch state; localFS publication timing only"})
        write_seconds = time.perf_counter() - t0
        sample_timings = {"eval_role": "train_members_only", "eval_rows": 4 * BATCH_SIZE,
                          "eval_step_seconds": eval_samples, "eval_optimizer_steps": 0,
                          "localFS_atomic_checkpoint_write_seconds": write_seconds,
                          "localFS_checkpoint_bytes": (output / "scratch_publication_probe.pt").stat().st_size,
                          "full50k_development_seconds": None, "remote_upload_seconds": None}
        atomic_json(output / "sample_timings.json", sample_timings)
        del iterator, model, optimizer, batch, prediction
        gc.collect()
        torch.cuda.empty_cache()
        eval_write_seconds = time.perf_counter() - eval_write_started
        guard()
        if sha256_file(root / "selected.pt") != state_meta["sha256"]:
            raise ValueError("Accepted checkpoint changed")
        result = {"format": "molgap-k1-native-t4-profile-v1", "status": "complete",
                  "cases": results, "cost": t4_cost(time.perf_counter()-started,
                      torch.cuda.device_count(), case_seconds, phase_seconds=phase_seconds,
                      eval_write_seconds=eval_write_seconds),
                  "training_rows": expected_rows, "development_rows": [],
                  "protected_roles": "untouched", "accuracy_acceptance": False,
                  "training_replay_ready": False, "consistency_coefficient": 0,
                  "scientific_outcome": "NO_TRAIN", "scratch_optimizer_steps_per_case": WARMUP + MEASURE,
                  "phase_samples": phases, "phase_optimizer_steps": 8,
                  "scratch_optimizer_steps_total": len(T4_CASES) * (WARMUP + MEASURE) + 8,
                  "gradient_norm_probe": None, "gradient_relation_probe_performed": False,
                  "sample_timings": sample_timings,
                  "limits": ["bounded selected-state scratch execution", "no quality proof",
                             "no global cost extrapolation", "active case wall != GPU busy"]}
        atomic_json(output / "result.json", result)
        atomic_json(output / "completion.json", {"status": "complete", "artifacts": {
            p.name: sha256_file(p) for p in output.iterdir() if p.is_file()
            and p.name not in {"completion.json", "worker.log", "worker_process_observation.json"}}})
        return

    # Synchronization intentionally separates phases; these are not throughput measurements.
    model, optimizer, meta = fresh()
    iterator = iter(loader(2))
    phases, batch, loss = _synchronized_phases(model, optimizer, iterator, meta,
                                              loss_for=loss_for, guard=guard)
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
    parser.add_argument("--native-t4", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--bounded-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.native_t4 and not args.bounded_child:
            import subprocess
            import sys
            from .training_reproducibility import atomic_json
            # The process ceiling includes imports, verification, load and teardown.
            process_started = time.perf_counter()
            status = "incomplete"
            try:
                subprocess.run([sys.executable, "-m", "molgap.k1_execution_profile",
                    "--native-t4", "--bounded-child", "--root", str(args.root),
                    "--output", str(args.output), "--expected-manifest-sha256",
                    args.expected_manifest_sha256 or ""], timeout=900, check=True)
                status = "complete"
            finally:
                elapsed = time.perf_counter() - process_started
                runtime_path = args.output / "runtime.json"
                count = json.loads(runtime_path.read_text()).get("visible_gpu_count") if runtime_path.exists() else None
                args.output.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output / "worker_process_observation.json", {
                    "status": status, "child_wall_seconds_including_imports_load_teardown": elapsed,
                    "visible_gpu_count": count, "allocated_T4_device_hours":
                        elapsed * count / 3600 if count is not None else None,
                    "gpu_busy_seconds": None, "worker_ceiling_seconds": 900})
        else:
            run(args.root, args.output, native_t4=args.native_t4,
                expected_manifest_sha256=args.expected_manifest_sha256)
    except Exception as error:
        from .training_reproducibility import atomic_json
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
