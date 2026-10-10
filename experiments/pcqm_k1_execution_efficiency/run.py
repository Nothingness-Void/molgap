"""Thin local diagnostic wrapper; no loader rebuild, scientific fit or submitter."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

from molgap.training_reproducibility import (
    atomic_json, atomic_torch_save, build_runtime_manifest, capture_rng_state,
    configure_fp32_determinism, restore_rng_state, sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def idle_observation():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    rows = [[int(v.strip()) for v in row.split(",")] for row in text.splitlines()]
    return {"utilization_memory_MiB": rows, "observed_at_unix": time.time(),
            "idle": len(rows) == 1 and rows[0][0] <= 10 and rows[0][2] - rows[0][1] >= 3000}


def _guard_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Whole-wrapper cooperative wall budget exhausted")


def _verify_cpu_optimizer_steps(optimizer):
    checked = 0
    for group in optimizer.param_groups:
        if group.get("fused") or group.get("capturable"):
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "step" in state:
                if state["step"].device.type != "cpu":
                    raise RuntimeError("Unfused noncapturable AdamW step state must remain on CPU")
                checked += 1
    if not checked:
        raise RuntimeError("No unfused noncapturable optimizer step state verified")
    return checked


def _checkpoint_resume(*, torch, factory, optimizer_for, step, batches, mean, std,
                       device, deadline, checkpoint_path, cpu_layout_context, compare_states):
    def guard():
        _guard_deadline(deadline)

    guard()
    model = factory().to(device).train()
    optimizer = optimizer_for(model)
    batch = batches[0].clone().to(device)
    guard()
    step(model, optimizer, batch, mean, std)
    guard()
    _verify_cpu_optimizer_steps(optimizer)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng_state()}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    guard()
    if device == "cuda":
        torch.cuda.synchronize()
    guard()
    tick = time.perf_counter()
    atomic_torch_save(checkpoint_path, checkpoint)
    save_seconds = time.perf_counter() - tick
    guard()
    publication = {"atomic_save_seconds": save_seconds, "bytes": checkpoint_path.stat().st_size,
                   "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                   "scope": "One scratch model/optimizer, not both-arm remote checkpoint publication"}
    guard()
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    guard()
    other = factory().to(device).train()
    other.load_state_dict(loaded["model"], strict=True)
    other_optimizer = optimizer_for(other)
    other_optimizer.load_state_dict(loaded["optimizer"])
    verified_steps = _verify_cpu_optimizer_steps(other_optimizer)
    guard()
    restore_rng_state(checkpoint["rng"])
    batch = batches[1].clone().to(device)
    guard()
    a = step(model, optimizer, batch, mean, std)
    guard()
    after_a = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng_state(), "loss": a}
    restore_rng_state(loaded["rng"])
    guard()
    with cpu_layout_context(other, batch := batches[1].clone(), device):
        transferred = batch.to(device)
        guard()
        b = step(other, other_optimizer, transferred, mean, std)
        guard()
    guard()
    after_b = {"model": other.state_dict(), "optimizer": other_optimizer.state_dict(), "rng": capture_rng_state(), "loss": b}
    resumed = compare_states(after_a, after_b)
    guard()
    return {"status": "complete", "publication": publication, "comparison": resumed,
            "verified_cpu_step_states": verified_steps,
            "state_keys_preserved": set(model.state_dict()) == set(other.state_dict())}


def run(device, output):
    started, cpu_started = time.perf_counter(), time.process_time()
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.k1_execution_efficiency import compare_states, probe_k1_execution_efficiency
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.pcqm_500k_v4_evidence import make_model, optimizer_for, step
    from molgap.v4_runtime import state_dict_sha256

    if output.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    inputs = json.loads((HERE / "inputs.json").read_text())
    wall_limit = inputs["worker_wall_limit_seconds"]
    if not math.isfinite(wall_limit) or not 0 < wall_limit <= 3600:
        raise ValueError("worker_wall_limit_seconds must be in (0, 3600]")
    deadline = started + wall_limit
    _guard_deadline(deadline)
    trajectory = json.loads((HERE / "rml/trajectory.json").read_text())
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required before diagnostic")
    for name, digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Source bytes changed: {name}")
    for entry in inputs["bindings"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Retained input differs")
    _guard_deadline(deadline)
    if device == "cuda":
        observations = [idle_observation()]
        time.sleep(2)
        observations.append(idle_observation())
        if not all(o["idle"] for o in observations):
            raise RuntimeError("GPU not idle; no diagnostic launched, do not monitor")
    else:
        observations = []
    output.mkdir(parents=True)
    _guard_deadline(deadline)
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    initial = torch.load(inputs["bindings"]["initial"]["path"], map_location="cpu", weights_only=True)
    initial = initial["model_state"]
    _guard_deadline(deadline)
    if state_dict_sha256(initial) != inputs["bindings"]["initial"]["tensor_sha256"]:
        raise ValueError("Initial tensor identity differs")
    selected = torch.load(inputs["bindings"]["normalization_checkpoint"]["path"], map_location="cpu", weights_only=True)
    mean, std = float(selected["mean"]), float(selected["std"])
    _guard_deadline(deadline)
    del selected
    graphs = torch.load(inputs["bindings"]["train_probe"]["path"], map_location="cpu", weights_only=False)
    _guard_deadline(deadline)
    if [int(g.source_idx) for g in graphs] != inputs["source_idx"] or len(graphs) != 4096:
        raise ValueError("Frozen training members/order differ")
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
    if any(not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
        raise ValueError("Nontraining or geometry probe input")
    timing = inputs["timing"]
    batches = list(DataLoader(graphs, batch_size=128, shuffle=False, drop_last=True, num_workers=0))
    _guard_deadline(deadline)
    batches = batches[:timing["warmup"] + timing["steps"]]
    runtime = build_runtime_manifest(settings)
    runtime.update(execution_device=device, source_commit=inputs["source_commit"],
                   prospective_sha256=sha256_file(HERE / "rml/trajectory.json"),
                   input_sha256=sha256_file(HERE / "inputs.json"), idle_observations=observations,
                   normalization={"mean": mean, "std": std},
                   observed_train_source_idx=inputs["source_idx"], scientific_training=False,
                   development_rows=[], protected_roles="untouched")
    atomic_json(output / "runtime.json", runtime)
    _guard_deadline(deadline)

    def factory():
        model = make_model("neural_atom_k1")
        model.load_state_dict(initial, strict=True)
        return model

    report = probe_k1_execution_efficiency(factory, batches, mean, std, device=device,
        **timing, equivalence_steps=inputs["equivalence_steps"],
        wall_seconds=deadline - time.perf_counter(),
        optimizer_cases=("foreach", "fused"), extra_cases={"cpu_layout": cpu_layout_context},
        atol=1e-6, rtol=1e-5)
    atomic_json(output / "result.json", report)

    # An exhausted probe never authorizes extra updates. Reporting may finish after
    # a cooperative overrun; this is not checkpoint-work grace or a hard ceiling.
    resume = {"status": "skipped", "reason": "probe budget exhausted", "comparison": None,
              "publication": None, "state_keys_preserved": None}
    if report["status"] != "budget_exhausted":
        try:
            resume = _checkpoint_resume(torch=torch, factory=factory, optimizer_for=optimizer_for,
                step=step, batches=batches, mean=mean, std=std, device=device, deadline=deadline,
                checkpoint_path=ROOT / "platforms/_records/local/staging/k1-execution-efficiency" / (device + "-scratch.pt"),
                cpu_layout_context=cpu_layout_context, compare_states=compare_states)
        except TimeoutError as error:
            resume.update(status="budget_exhausted", reason=str(error))
    atomic_json(output / "checkpoint_resume.json", resume)
    elapsed = time.perf_counter() - started
    budget_exhausted = report["status"] == "budget_exhausted" or resume["status"] == "budget_exhausted" or elapsed >= wall_limit
    status = "budget_exhausted" if budget_exhausted else report["status"]
    atomic_json(output / "cost_observation.json", {"worker_wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started, "cpu_children_seconds": 0,
        "execution_device": device, "gpu_busy_seconds": None,
        "budget_mode": "cooperative; no beyond-budget checkpoint-work grace",
        "wall_limit_seconds": wall_limit, "observed_overrun_seconds": max(0., elapsed - wall_limit),
        "cost_scope": "Whole run() including preparation/imports; module import, test suite and parent overhead excluded"})
    atomic_json(output / "completion.json", {"artifacts": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        "status": status, "checkpoint_resume_status": resume["status"],
        "missing_evidence": ["checkpoint_resume"] if resume["status"] != "complete" else [],
        "scientific_training": False, "accuracy_acceptance": False, "training_replay_ready": False})
    print(json.dumps({"device": device, "status": status,
        "resume_exact": resume["comparison"]["exact"] if resume["comparison"] is not None else None,
        "cases": {k: {f: row.get(f) for f in ("status", "median_seconds_per_step", "speedup", "execution_only_admissible")}
                  for k, row in report["cases"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.device, args.output)
