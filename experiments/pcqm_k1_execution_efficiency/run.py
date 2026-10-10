"""Thin local diagnostic wrapper; no loader rebuild, scientific fit or submitter."""
import argparse
import json
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


def run(device, output):
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.k1_execution_efficiency import compare_states, probe_k1_execution_efficiency
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.pcqm_500k_v4_evidence import make_model, optimizer_for, step
    from molgap.v4_runtime import state_dict_sha256

    if output.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    inputs = json.loads((HERE / "inputs.json").read_text())
    trajectory = json.loads((HERE / "rml/trajectory.json").read_text())
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required before diagnostic")
    for name, digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Source bytes changed: {name}")
    for entry in inputs["bindings"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Retained input differs")
    if device == "cuda":
        observations = [idle_observation()]
        time.sleep(2)
        observations.append(idle_observation())
        if not all(o["idle"] for o in observations):
            raise RuntimeError("GPU not idle; no diagnostic launched, do not monitor")
    else:
        observations = []
    output.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    initial = torch.load(inputs["bindings"]["initial"]["path"], map_location="cpu", weights_only=True)
    initial = initial["model_state"]
    if state_dict_sha256(initial) != inputs["bindings"]["initial"]["tensor_sha256"]:
        raise ValueError("Initial tensor identity differs")
    selected = torch.load(inputs["bindings"]["normalization_checkpoint"]["path"], map_location="cpu", weights_only=True)
    mean, std = float(selected["mean"]), float(selected["std"])
    del selected
    graphs = torch.load(inputs["bindings"]["train_probe"]["path"], map_location="cpu", weights_only=False)
    if [int(g.source_idx) for g in graphs] != inputs["source_idx"] or len(graphs) != 4096:
        raise ValueError("Frozen training members/order differ")
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
    if any(not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
        raise ValueError("Nontraining or geometry probe input")
    timing = inputs["timing"]
    batches = list(DataLoader(graphs, batch_size=128, shuffle=False, drop_last=True, num_workers=0))
    batches = batches[:timing["warmup"] + timing["steps"]]
    runtime = build_runtime_manifest(settings)
    runtime.update(execution_device=device, source_commit=inputs["source_commit"],
                   prospective_sha256=sha256_file(HERE / "rml/trajectory.json"),
                   input_sha256=sha256_file(HERE / "inputs.json"), idle_observations=observations,
                   normalization={"mean": mean, "std": std},
                   observed_train_source_idx=inputs["source_idx"], scientific_training=False,
                   development_rows=[], protected_roles="untouched")
    atomic_json(output / "runtime.json", runtime)

    def factory():
        model = make_model("neural_atom_k1")
        model.load_state_dict(initial, strict=True)
        return model

    report = probe_k1_execution_efficiency(factory, batches, mean, std, device=device,
        **timing, equivalence_steps=inputs["equivalence_steps"],
        wall_seconds=inputs["worker_wall_limit_seconds"] - 120,
        optimizer_cases=("foreach", "fused"), extra_cases={"cpu_layout": cpu_layout_context},
        atol=1e-6, rtol=1e-5)
    atomic_json(output / "result.json", report)

    # Verify model/optimizer/RNG serialization and continued dropout updates independently.
    model = factory().to(device).train()
    optimizer = optimizer_for(model)
    batch = batches[0].clone().to(device)
    step(model, optimizer, batch, mean, std)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng_state()}
    checkpoint_path = ROOT / "platforms/_records/local/staging/k1-execution-efficiency" / (device + "-scratch.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    atomic_torch_save(checkpoint_path, checkpoint)
    publication = {"atomic_save_seconds": time.perf_counter() - tick, "bytes": checkpoint_path.stat().st_size(),
                   "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                   "scope": "One scratch model/optimizer, not both-arm remote checkpoint publication"}
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    other = factory().to(device).train()
    other.load_state_dict(loaded["model"], strict=True)
    other_optimizer = optimizer_for(other)
    other_optimizer.load_state_dict(loaded["optimizer"])
    restore_rng_state(checkpoint["rng"])
    a = step(model, optimizer, batches[1].clone().to(device), mean, std)
    after_a = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng_state(), "loss": a}
    restore_rng_state(loaded["rng"])
    with cpu_layout_context(other, batch := batches[1].clone(), device):
        b = step(other, other_optimizer, batch.to(device), mean, std)
    after_b = {"model": other.state_dict(), "optimizer": other_optimizer.state_dict(), "rng": capture_rng_state(), "loss": b}
    resumed = compare_states(after_a, after_b)
    atomic_json(output / "checkpoint_resume.json", {"publication": publication, "comparison": resumed,
        "state_keys_preserved": set(model.state_dict()) == set(other.state_dict())})
    atomic_json(output / "cost_observation.json", {"worker_wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started, "cpu_children_seconds": 0,
        "execution_device": device, "gpu_busy_seconds": None,
        "cost_scope": "Local worker body; imports, test suite and parent overhead excluded"})
    atomic_json(output / "completion.json", {"artifacts": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        "scientific_training": False, "accuracy_acceptance": False, "training_replay_ready": False})
    print(json.dumps({"device": device, "status": report["status"], "resume_exact": resumed["exact"],
        "cases": {k: {f: row.get(f) for f in ("status", "median_seconds_per_step", "speedup", "execution_only_admissible")}
                  for k, row in report["cases"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.device, args.output)
