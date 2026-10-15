"""Synthetic comparison of frozen family sessions; never a training launcher."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile


def tree_digest(value):
    """Bind primitive structure as well as tensor bytes using the existing owner."""
    import torch
    from molgap.training_reproducibility import canonical_fingerprint
    from molgap.v4_runtime import state_dict_sha256
    tensors = {}
    def visit(child, path):
        if isinstance(child, torch.Tensor):
            tensors[path] = child
            return {"tensor": path}
        if isinstance(child, dict):
            return {"dict": [[str(key), visit(item, path + "/" + str(key))]
                for key, item in sorted(child.items(), key=lambda pair: str(pair[0]))]}
        if isinstance(child, (list, tuple)):
            return {type(child).__name__: [visit(item, path + "/" + str(i)) for i, item in enumerate(child)]}
        return child
    structure = visit(value, "root")
    return canonical_fingerprint({"structure": structure, "tensors": state_dict_sha256(tensors)})


def synthetic_graphs(settings):
    import torch
    from torch_geometric.data import Data
    graphs = []
    low, high = settings["minimum_nodes"], settings["maximum_nodes"]
    for i in range(settings["rows"]):
        n = low + i % (high - low + 1)
        start = torch.arange(n - 1)
        edges = torch.stack((torch.cat((start, start + 1)), torch.cat((start + 1, start))))
        x = torch.zeros(n, 9, dtype=torch.long)
        x[:, 0] = (torch.arange(n) + i) % 8
        graphs.append(Data(x=x, edge_index=edges, edge_attr=torch.zeros(edges.shape[1], 3, dtype=torch.long),
            random_walk_pe=torch.zeros(n, 16), y=torch.tensor([.25 + (i % 13) / 20]),
            source_idx=torch.tensor([i])))
    return graphs


def worker(config_path, source_root, implementation, family, repeat, output):
    import torch
    from molgap.experiment_family_workflow import RunContext, tensor_safe_rng_state
    from molgap.family_training_core import run_training_loop
    from molgap.family_training_data import DatasetSelection, TrainingData
    from molgap.family_training_kernel import ContractSchedule, FamilyKernel
    from molgap.family_training_loaders import FamilyLoaders
    from molgap.family_training_outputs import SessionOutputs
    from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
    from molgap.family_training_runner import INTERFACES, PROFILES, build_screen_recipe
    from molgap.family_training_session import FamilyTrainingSession
    from molgap.gptrans_execution_tuning import ExecutionTuning
    from molgap.training_reproducibility import (
        atomic_json, atomic_torch_save, canonical_fingerprint, capture_rng_state, configure_fp32_determinism, sha256_file,
    )
    from molgap.v4_runtime import state_dict_sha256

    started, cpu_started = time.perf_counter(), time.process_time()
    config_path, source_root, output = map(Path, (config_path, source_root, output))
    config = json.loads(config_path.read_text())
    source = config["sources"][implementation]
    if output.exists():
        raise FileExistsError("Fresh trial output required")
    output.mkdir(parents=True)
    for name, digest in source["files"].items():
        if sha256_file(source_root / name) != digest:
            raise ValueError("Frozen execution source changed: " + name)
    import molgap.family_training_session as session_owner
    if Path(session_owner.__file__).resolve() != (source_root / "src/molgap/family_training_session.py").resolve():
        raise ValueError("Session import escaped its frozen source")
    torch.set_num_threads(config["cpu_threads"])
    configure_fp32_determinism(config["seed"])
    graphs = synthetic_graphs(config["synthetic"])
    if family == "neural_atom_k1":
        from molgap.qm9_neural_atom import make_encoder
        model = make_encoder(family)
    else:
        from molgap.pcqm_gptrans_v4 import _make_model
        model = _make_model()
    initial = output / "initial.pt"
    initialization = state_dict_sha256(model.state_dict())
    atomic_torch_save(initial, model.state_dict())
    del model
    selection = DatasetSelection("synthetic-only.json", canonical_fingerprint(config["synthetic"]), False)
    recipe = FamilyRecipe(family, "reference", config["seed"], initialization, selection, len(graphs), 0,
        config["steps"], 4e-4, 1e-5, ScheduleSpec("cosine_step", config["steps"]), "final-ema",
        "python-epoch-shuffle" if family == "neural_atom_k1" else "torch-pass-randperm",
        config["checkpoint_every"], 0., 1., ema_decay=.999,
        torch_version=str(torch.__version__).split("+", 1)[0])
    data = TrainingData(graphs, None, selection.manifest_sha256, len(graphs), 0, 0, None, 0., 1.)
    tuning = ExecutionTuning(training_workers=0)
    record = build_screen_recipe("reference", configuration=recipe, family=INTERFACES[family],
        train_role_identity="synthetic-only-no-molecular-role")
    contract = output / "contract.json"
    atomic_json(contract, record)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != config["accelerator"]:
        raise ValueError("Pinned local accelerator required")
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "accelerator": torch.cuda.get_device_name(0), "cpu_threads": torch.get_num_threads(), "pid": os.getpid()}

    def fresh():
        configure_fp32_determinism(config["seed"])
        return FamilyKernel(recipe, initial, device="cuda:0"), FamilyLoaders(recipe, data, tuning)
    def snapshot(kernel, schedule, losses):
        return tree_digest({"model": kernel.model.state_dict(), "optimizer": kernel.optimizer.state_dict(),
            "ema": kernel.ema.state_dict(), "scheduler": schedule.state_dict(), "rng": tensor_safe_rng_state(capture_rng_state()),
            "gradient": {key: p.grad for key, p in kernel.model.named_parameters() if p.grad is not None},
            "losses": torch.stack(losses)})

    kernel, loaders = fresh()
    try:
        warm = next(iter(loaders.train(0, 0)))
        kernel.model.train()
        for _ in range(config["warmup_steps"]):
            kernel.step(warm, check_finite=True)
        torch.cuda.synchronize()
    finally:
        kernel.close()
        loaders.close()
    del kernel, loaders, warm
    gc.collect()
    result = {"family": family, "implementation": implementation, "repeat": repeat,
        "source_commit": source["commit"], "source_archive_sha256": source["archive_sha256"],
        "initialization_sha256": initialization, "config_sha256": sha256_file(config_path),
        "runtime": runtime, "steps": config["steps"], "samples": config["steps"] * 128,
        "scratch_optimizer_updates": True, "formal_training_executed": False, "molecular_roles_consumed": False}

    for mode in ("compute", "session"):
        kernel, loaders = fresh()
        losses = []
        original_step = kernel.step
        def step(batch, *, check_finite):
            value = original_step(batch, check_finite=check_finite)
            losses.append(value[0])
            return value
        kernel.step = step
        try:
            if mode == "compute":
                plan = recipe.loop_plan
                batches = {p: list(loaders.train(p, 0)) for p in range(config["steps"] // plan.steps_per_pass)}
                schedule = ContractSchedule(kernel)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                stamp = time.perf_counter()
                progress = run_training_loop(kernel.model, plan, lambda p, offset: batches[p][offset:], step,
                    lambda *_: None, before_pass=schedule.before_pass, before_step=schedule.before_step,
                    after_step=schedule.after_step, after_pass=schedule.after_pass)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - stamp
            else:
                context = RunContext(config["experiment_id"], config["experiment_id"], implementation,
                    canonical_fingerprint({"implementation": implementation, "source": source["commit"]}),
                    canonical_fingerprint(config), *INTERFACES[family], source["commit"], source["archive_sha256"],
                    canonical_fingerprint(source), sha256_file(contract), "local-windows", "local",
                    "local/" + config["experiment_id"] + "/" + str(os.getpid()), None)
                outputs = SessionOutputs(output / "session", context, contract=contract, adapter=PROFILES[family],
                    trajectory_id=config["trajectory_id"], train_role_identity=record["train_role_identity"])
                # Frozen comparison roots may predate the complete-state sink.
                hooks = {"checkpoint_builder": outputs.checkpoint_from_builder} if hasattr(
                    outputs, "checkpoint_from_builder") else {}
                session = FamilyTrainingSession(kernel, loaders, checkpoint=outputs.checkpoint,
                    selected=outputs.selected, pass_finished=outputs.pass_finished, **hooks)
                schedule = session.schedule
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                stamp = time.perf_counter()
                progress = session.run()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - stamp
                checked = outputs.complete(progress, runtime={"platform": context.platform, "account": context.account,
                    "precision": "fp32", "source_commit": context.source_commit,
                    "source_archive_sha256": context.source_archive_sha256}, costs=[{
                    "metric": "wall_seconds", "unit": "seconds", "value": elapsed, "status": "measured",
                    "semantics": "process_wall", "hardware": config["accelerator"], "scope": "synthetic session only"}])
                if checked["status"] != "MECHANICALLY_VERIFIED":
                    raise ValueError("Synthetic session inspection failed: " + repr(checked["blockers"]))
                result["mechanical_inspection"] = checked["status"]
            if progress.optimizer_step != config["steps"] or progress.status != "complete":
                raise ValueError("Incomplete synthetic exposure")
            result[mode] = {"seconds": elapsed, "molecules_per_second": config["steps"] * 128 / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "endpoint_sha256": snapshot(kernel, schedule, losses)}
            atomic_json(output / "result.json", result)
        finally:
            kernel.close()
            loaders.close()
        del kernel, loaders, losses
        gc.collect()
    result.update(status="COMPLETE", worker_wall_seconds=time.perf_counter() - started,
        process_cpu_seconds=time.process_time() - cpu_started)
    atomic_json(output / "result.json", result)
    return result


def run(root, config_path, output):
    """Execute sequential paired trials in independently frozen import roots."""
    from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file
    root, config_path, output = map(lambda p: Path(p).resolve(), (root, config_path, output))
    output.relative_to(root)
    config = json.loads(config_path.read_text())
    trajectory = json.loads((root / config["trajectory_ref"]).read_text())
    if (trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE"
            or trajectory["state_at_start"]["source_config_identity"] != canonical_fingerprint(config)):
        raise ValueError("Matching active prospective diagnostic required")
    if output.exists():
        raise FileExistsError("Fresh output required")
    for name, digest in config["source_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Diagnostic source changed: " + name)
    output.mkdir(parents=True)
    sources = {}
    for implementation, source in config["sources"].items():
        archive = root / source["archive"]
        if sha256_file(archive) != source["archive_sha256"]:
            raise ValueError("Frozen archive changed")
        target = output / (implementation + "_source")
        target.mkdir()
        with zipfile.ZipFile(archive) as package:
            for item in package.infolist():
                if not (target / item.filename).resolve().is_relative_to(target.resolve()):
                    raise ValueError("Archive path escapes frozen root")
            package.extractall(target)
        sources[implementation] = target
    began = time.perf_counter()
    deadline = began + config["maximum_process_wall_seconds"]
    result = {"status": "RUNNING", "format": "molgap-family-session-speed-v1",
        "config_sha256": sha256_file(config_path), "trials": [], "molecular_roles_consumed": False,
        "formal_training_executed": False, "quality_evaluated": False}
    driver = Path(__file__).resolve()
    for family in config["families"]:
        for repeat in range(config["repeats"]):
            order = ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")
            for implementation in order:
                trial = output / (family + "_" + str(repeat) + "_" + implementation)
                env = dict(os.environ, PYTHONPATH=str(sources[implementation] / "src"), CUDA_VISIBLE_DEVICES="0",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED=str(config["seed"]))
                env.pop("HIP_VISIBLE_DEVICES", None)
                env.pop("ROCR_VISIBLE_DEVICES", None)
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Frozen diagnostic budget exhausted")
                command = [sys.executable, str(driver), "--worker", "--config", str(config_path),
                    "--source-root", str(sources[implementation]), "--implementation", implementation,
                    "--family", family, "--repeat", str(repeat), "--output", str(trial)]
                with (output / (trial.name + ".log")).open("w", encoding="utf-8") as log:
                    stamp = time.perf_counter()
                    subprocess.run(command, cwd=sources[implementation], env=env, check=True,
                        stdout=log, stderr=subprocess.STDOUT, timeout=min(remaining, config["maximum_trial_wall_seconds"]))
                observed = json.loads((trial / "result.json").read_text())
                observed["process_wall_seconds"] = time.perf_counter() - stamp
                result["trials"].append(observed)
                atomic_json(output / "result.json", result)
                print(json.dumps({"family": family, "repeat": repeat, "implementation": implementation,
                    "compute_s": observed["compute"]["seconds"], "session_s": observed["session"]["seconds"]}), flush=True)
    result["checks"] = compare_trials(result["trials"], config)
    result.update(status="COMPLETE", process_wall_seconds=time.perf_counter() - began,
        allocated_worker_process_seconds=sum(row["process_wall_seconds"] for row in result["trials"]))
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", {"status": result["status"], "result_sha256": sha256_file(output / "result.json")})
    return result


def compare_trials(trials, config):
    expected = {(family, repeat, implementation) for family in config["families"]
        for repeat in range(config["repeats"]) for implementation in ("reference", "candidate")}
    observed = [(row["family"], row["repeat"], row["implementation"]) for row in trials]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Missing or duplicated speed trial")
    index = {key: row for key, row in zip(observed, trials)}
    if any(row["compute"]["endpoint_sha256"] != row["session"]["endpoint_sha256"] for row in trials):
        raise ValueError("Persistence changed a compute/session mathematical endpoint")
    summaries = []
    for family in config["families"]:
        for mode in ("compute", "session"):
            reference, candidate, ratios = [], [], []
            for repeat in range(config["repeats"]):
                left, right = [index[(family, repeat, name)] for name in ("reference", "candidate")]
                if (left["initialization_sha256"] != right["initialization_sha256"]
                        or left["steps"] != right["steps"] or left["samples"] != right["samples"]
                        or left["runtime"]["torch"] != right["runtime"]["torch"]
                        or left[mode]["endpoint_sha256"] != right[mode]["endpoint_sha256"]):
                    raise ValueError("Mathematical endpoint or paired identity changed")
                for row in (left, right):
                    seconds = row[mode]["seconds"]
                    if (row["status"] != "COMPLETE" or row["steps"] != config["steps"]
                            or row["samples"] != 128 * config["steps"]
                            or row["source_commit"] != config["sources"][row["implementation"]]["commit"]
                            or row["molecular_roles_consumed"] is not False or row["formal_training_executed"] is not False
                            or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
                        raise ValueError("Invalid timing, source, scope or exposure")
                reference.append(left[mode]["seconds"])
                candidate.append(right[mode]["seconds"])
                ratios.append(1 - candidate[-1] / reference[-1])
            summaries.append({"family": family, "mode": mode,
                "reference_median_seconds": statistics.median(reference),
                "candidate_median_seconds": statistics.median(candidate),
                "paired_time_reduction_median": statistics.median(ratios),
                "paired_time_reduction_range": [min(ratios), max(ratios)], "exact_endpoint_equal": True})
    return {"paired_summaries": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation", choices=("reference", "candidate"), required=True)
    parser.add_argument("--family", choices=("neural_atom_k1", "gptrans_t"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(args.config, args.source_root, args.implementation, args.family, args.repeat, args.output)
