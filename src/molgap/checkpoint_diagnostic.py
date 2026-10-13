"""Bounded synthetic checkpoint profiling; no trainer or remote dispatch."""
from __future__ import annotations

import json
from pathlib import Path
import time


def run(root, config_path, output, *, legacy_tree, legacy_state):
    import torch
    from .experiment_family_workflow import _finite_tree, tensor_safe_rng_state
    from .training_reproducibility import (
        assert_finite_state_dict, atomic_json, atomic_torch_save, canonical_fingerprint,
        capture_rng_state, configure_fp32_determinism, sha256_file,
    )
    from .v4_runtime import state_dict_sha256

    root, config_path, output = Path(root).resolve(), Path(config_path).resolve(), Path(output).resolve()
    output.relative_to(root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trajectory = json.loads((root / config["trajectory_ref"]).read_text(encoding="utf-8"))
    if (trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE"
            or trajectory["state_at_start"]["source_config_identity"] != canonical_fingerprint(config)):
        raise ValueError("Active matching prospective checkpoint diagnostic required")
    for name, digest in config["source_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    if output.exists():
        raise FileExistsError("Use a fresh diagnostic output")
    output.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + config["total_wall_seconds"]
    # RNG capture can initialize CUDA during a nominally CPU case. Count that
    # allocation from worker setup rather than pretending CPU cases use no GPU.
    gpu_started = started if "cuda" in config["devices"] else None
    gpu_stopped = None
    configure_fp32_determinism(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    result = {"format": "molgap-checkpoint-diagnostic-v1", "status": "RUNNING",
        "config_sha256": sha256_file(config_path), "source_commit": config["source_commit"],
        "training_executed": False, "inference_executed": False, "molecular_roles_consumed": False,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads()}, "cases": [], "checks": {}}

    def tensor_digest(value):
        tensors = {}
        def visit(child):
            if isinstance(child, torch.Tensor):
                tensors[str(len(tensors))] = child
            elif isinstance(child, dict):
                for item in child.values():
                    visit(item)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    visit(item)
        visit(value)
        return state_dict_sha256(tensors)

    def guard():
        now = time.perf_counter()
        if now >= deadline or (gpu_started is not None and now - gpu_started >= config["gpu_wall_seconds"]):
            raise TimeoutError("Checkpoint diagnostic wall budget exhausted")

    def synchronize(device):
        if device == "cuda":
            torch.cuda.synchronize()

    def check(state, implementation, *, publish):
        tree, flat = ((_finite_tree, assert_finite_state_dict) if implementation == "batched"
                      else (legacy_tree, legacy_state))
        tree(state, "family.checkpoint")
        if publish:
            # Match the session and writer's deliberate independent validations.
            flat(state["model"], label="resume.model")
            flat(state["ema"], label="resume.ema")
            atomic_torch_save(output / "scratch.pt", state)

    for device in config["devices"]:
        if device == "cuda":
            if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != config["accelerator"]:
                raise ValueError("Expected the pinned RTX5060 allocation")
            result["runtime"]["accelerator"] = torch.cuda.get_device_name(0)
        for family in config["families"]:
            guard()
            if family == "neural_atom_k1":
                from .qm9_neural_atom import make_encoder
                model = make_encoder("neural_atom_k1")
            elif family == "gptrans_t":
                from .pcqm_gptrans_v4 import _make_model
                model = _make_model()
            else:
                raise ValueError("Unknown checkpoint schema family")
            model.to(device)
            weights = model.state_dict()
            moments = {i: {"step": torch.tensor(1.), "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p)} for i, p in enumerate(model.parameters())}
            state = {"model": weights, "optimizer": {"state": moments,
                "param_groups": [{"params": list(moments), "lr": .0004}]},
                "scheduler": {"completed_units": 0},
                "ema": {key: value.clone() for key, value in weights.items()},
                "rng_state": tensor_safe_rng_state(capture_rng_state()), "optimizer_step": 1,
                "sample_presentations": 128, "cursor": {}}
            schema = {"model_tensors": len(weights), "optimizer_tensors": 3 * len(moments),
                "ema_tensors": len(weights), "model_elements": sum(v.numel() for v in weights.values())}
            for count in config["cursor_lengths"]:
                state["cursor"] = {"pending_absolute": list(torch.arange(count, dtype=torch.float32).unbind()),
                    "pending_rows": count * 128}
                before = tensor_digest(state)
                for publish in (False, True):
                    if publish and count not in config["publish_cursor_lengths"]:
                        continue
                    row = {"family": family, "device": device, "cursor_length": count,
                        "operation": "validate_publish" if publish else "validate", "schema": schema,
                        "original_seconds": [], "batched_seconds": []}
                    expected_sha = None
                    for repeat in range(-1, config["repeats"]):
                        order = ("original", "batched") if repeat % 2 == 0 else ("batched", "original")
                        for implementation in order:
                            guard()
                            synchronize(device)
                            t0 = time.perf_counter()
                            check(state, implementation, publish=publish)
                            synchronize(device)
                            seconds = time.perf_counter() - t0
                            if repeat >= 0:
                                row[implementation + "_seconds"].append(seconds)
                            if publish:
                                digest = sha256_file(output / "scratch.pt")
                                if expected_sha is not None and digest != expected_sha:
                                    raise ValueError("Saved checkpoint bytes changed")
                                expected_sha = digest
                    if publish:
                        row["identical_checkpoint_sha256"] = expected_sha
                    result["cases"].append(row)
                    atomic_json(output / "result.json", result)
                if tensor_digest(state) != before:
                    raise ValueError("Finite checking mutated checkpoint tensors")
            # Saving/finite checking may not draw RNG samples either.
            rng = capture_rng_state()
            saved_rng = state["rng_state"]
            if (rng["python"] != saved_rng["python"] or tensor_safe_rng_state(rng)["numpy"] != saved_rng["numpy"]
                    or not torch.equal(rng["torch"], saved_rng["torch"])
                    or len(rng["cuda"]) != len(saved_rng["cuda"]) or any(
                        not torch.equal(a, b) for a, b in zip(rng["cuda"], saved_rng["cuda"]))):
                raise ValueError("Finite checking mutated RNG state")
            del model, weights, moments, state
            if device == "cuda":
                torch.cuda.empty_cache()
        if device == "cuda":
            synchronize(device)
            gpu_stopped = time.perf_counter()
    (output / "scratch.pt").unlink(missing_ok=True)
    result.update(status="COMPLETE", wall_seconds=time.perf_counter() - started,
        process_cpu_seconds=time.process_time() - cpu_started,
        gpu_phase_seconds=0 if gpu_started is None else gpu_stopped - gpu_started,
        checks={"checkpoint_tensors_unchanged": True, "rng_unchanged": True, "checkpoint_bytes_equal": True,
            "no_molecular_role_access": True})
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", {"result_sha256": sha256_file(output / "result.json"),
        "config_sha256": result["config_sha256"], "status": result["status"]})
    return result
