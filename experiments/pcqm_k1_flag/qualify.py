"""Bounded, prospective CPU qualification on the first 256 training graphs.

This worker never submits training or reads development/protected roles.
The parent publishes the freeze and prospective record before invoking it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def run(freeze_path, output):
    from molgap.training_reproducibility import atomic_json, sha256_file

    started = time.perf_counter()
    started_process_cpu = time.process_time()
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (frozen.get("row_start") != 0 or frozen.get("row_stop") != 256 or
            frozen.get("cpu_threads") != 4 or frozen.get("max_seconds") != 300):
        raise ValueError("CPU scope differs from the frozen 256-row/four-thread/300-second contract")

    def timeout():
        atomic_json(output / "cpu_failure.json", {"accepted": False,
            "error": "CPU qualification exceeded 300 seconds", "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - started_process_cpu})
        os._exit(1)

    timer = threading.Timer(300, timeout)
    timer.daemon = True
    timer.start()
    try:
        return qualify(frozen, freeze_path, output, started, started_process_cpu)
    finally:
        timer.cancel()


def qualify(frozen, freeze_path, output, started, started_process_cpu):
    import torch
    import torch.nn.functional as functional
    from torch_geometric.data import Batch
    from molgap.k1_flag import CONFIG, embedding_perturbation, adversarial_optimizer_step
    from molgap.k1_screen_training import (
        INITIAL_STATE_SHA256, FORBIDDEN_MODEL_FIELDS, _PackedGraphDatasetFactory, _forward)
    from molgap.qm9_neural_atom import make_encoder
    from molgap.training_reproducibility import (
        atomic_json, atomic_torch_save, sha256_file, configure_fp32_determinism,
        capture_rng_state, restore_rng_state, assert_finite_state_dict)
    from molgap.v4_runtime import normalized_source_sha256, state_dict_sha256, certify_numerical_repeatability

    def require(condition, reason):
        if not condition:
            raise RuntimeError(reason)
        if time.perf_counter() - started > 300:
            raise RuntimeError("CPU qualification exceeded its wall budget")

    def path(value):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else ROOT / candidate

    def pinned(name):
        candidate = path(frozen[name + "_path"])
        require(sha256_file(candidate) == frozen[name + "_sha256"], name + " bytes differ from freeze")
        return candidate

    require(bool(frozen.get("source_hashes")), "Missing frozen source inventory")
    for relative, digest in frozen["source_hashes"].items():
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(ROOT) and normalized_source_sha256(source) == digest,
                "Frozen source differs: " + relative)
    require("experiments/pcqm_k1_flag/qualify.py" in frozen["source_hashes"], "Worker source must be frozen")
    prospective_path = pinned("prospective")
    prospective = json.loads(prospective_path.read_text(encoding="utf-8"))
    require(prospective.get("record_mode") == "prospective", "CPU record is not prospective")
    shard_path = pinned("graph_shard")
    require(shard_path.resolve() == Path("D:/文档/molgap/data/pcqm_fixed_100k_v1/train/train_shard_0000.pt").resolve(),
            "CPU graph path differs from authorized first training shard")
    selected_path = pinned("selected_checkpoint")
    require(frozen["initial_state_sha256"] == INITIAL_STATE_SHA256, "Frozen initial tensor identity changed")
    mean_value, std_value = frozen["target_mean"], frozen["target_std"]
    require(mean_value == 5.3383002281188965 and std_value == 1.275090217590332,
            "Frozen target statistics changed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    determinism = configure_fp32_determinism(42)
    graphs = _PackedGraphDatasetFactory.load(shard_path)
    prefix = [graphs[index] for index in range(256)]
    source_idx = torch.cat([graph.source_idx.view(-1).long() for graph in prefix])
    targets = torch.cat([graph.y.view(-1).float() for graph in prefix])
    require(torch.equal(source_idx, torch.arange(256)), "Training prefix row identities differ")
    require(all(not any(field in graph for field in FORBIDDEN_MODEL_FIELDS) for graph in prefix),
            "Geometry was not removed from CPU inputs")
    require(bool(torch.isfinite(targets).all()), "CPU target prefix is nonfinite")
    row_sha = hashlib.sha256(source_idx.numpy().tobytes()).hexdigest()
    target_sha = hashlib.sha256(targets.numpy().tobytes()).hexdigest()
    if "source_idx_sha256" in frozen:
        require(row_sha == frozen["source_idx_sha256"], "CPU source-index bytes changed")
    if "target_sha256" in frozen:
        require(target_sha == frozen["target_sha256"], "CPU prefix target bytes changed")
    batch = Batch.from_data_list(prefix[:128])
    mean, std = torch.tensor(mean_value), torch.tensor(std_value)
    model = make_encoder("neural_atom_k1").cpu()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    require(state_dict_sha256(initial) == INITIAL_STATE_SHA256, "Constructed original K1 initial state differs")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    require(parameters == 3658817, "Original K1 parameter count changed")
    initial_path = output / "initial_state.pt"
    atomic_torch_save(initial_path, initial)
    keys, stored_count = set(initial), sum(value.numel() for value in initial.values())
    model.eval()
    hooks_before = tuple(model.node_emb._forward_hooks)
    with torch.no_grad():
        clean = _forward(model, batch)
        with embedding_perturbation(model, torch.zeros((batch.num_nodes, 192))):
            zero = _forward(model, batch)
        cleanup = _forward(model, batch)
    require(torch.equal(clean, zero) and torch.equal(clean, cleanup), "Zero/cleanup inference differs")
    require(tuple(model.node_emb._forward_hooks) == hooks_before, "FLAG hook leaked")
    require(state_dict_sha256(model.state_dict()) == INITIAL_STATE_SHA256, "Read-only hook changed model state")

    # Independent accumulation spells out the perturbation and loss loop.
    left, right = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    left_optimizer = torch.optim.AdamW(left.parameters(), lr=4e-4, weight_decay=1e-5)
    right_optimizer = torch.optim.AdamW(right.parameters(), lr=4e-4, weight_decay=1e-5)
    rng = capture_rng_state()
    loss_left, absolute_left, count_left = adversarial_optimizer_step(
        left, left_optimizer, batch, mean, std, _forward)
    restore_rng_state(rng)
    right_optimizer.zero_grad(set_to_none=True)
    delta = torch.empty((batch.num_nodes, 192)).uniform_(-0.001, 0.001).requires_grad_()
    normalized = (batch.y.view(-1) - mean) / std
    loss_right, absolute_right = normalized.new_zeros(()), normalized.new_zeros(())
    perturbation_norms = []
    for index in range(3):
        handle = right.node_emb.register_forward_hook(lambda _m, _args, value: value + delta)
        try:
            prediction = _forward(right, batch)
            loss = functional.l1_loss(prediction, normalized)
            (loss / 3).backward()
        finally:
            handle.remove()
        require(delta.grad is not None and bool(torch.isfinite(delta.grad).all()) and
                float(delta.grad.abs().sum()) > 0, "Perturbation gradient is not finite/nonzero")
        perturbation_norms.append(float(delta.grad.abs().sum()))
        loss_right += loss.detach() / 3
        absolute_right += (prediction.detach() - normalized).abs().sum() / 3
        if index < 2:
            delta = (delta.detach() + 0.001 * delta.grad.detach().sign()).requires_grad_()
    right_gradients = [p.grad for p in right.parameters() if p.grad is not None]
    require(right_gradients and all(bool(torch.isfinite(g).all()) for g in right_gradients) and
            any(float(g.abs().sum()) > 0 for g in right_gradients), "Model gradients are not finite/nonzero")
    torch.nn.utils.clip_grad_norm_(right.parameters(), 1.0, error_if_nonfinite=True)
    gradient_delta = 0.0
    for (left_name, left_parameter), (right_name, right_parameter) in zip(left.named_parameters(), right.named_parameters()):
        require(left_name == right_name and (left_parameter.grad is None) == (right_parameter.grad is None),
                "Gradient parameter identities differ")
        if left_parameter.grad is not None:
            gradient_delta = max(gradient_delta, float((left_parameter.grad - right_parameter.grad).abs().max()))
    require(gradient_delta <= 1e-7, "FLAG gradient average differs from explicit loop")
    right_optimizer.step()
    accumulation = certify_numerical_repeatability(losses=[float(loss_left), float(loss_right)],
        states=[left.state_dict(), right.state_dict()])
    require(count_left == 128 and abs(float(absolute_left - absolute_right)) <= 1e-7,
            "FLAG metric average differs from explicit loop")
    require(set(left.state_dict()) == keys and sum(v.numel() for v in left.state_dict().values()) == stored_count and
            sum(p.numel() for p in left.parameters()) == parameters, "FLAG added model state")
    require(not left.node_emb._forward_hooks and not right.node_emb._forward_hooks, "Training hook leaked")

    # Durable primitive and trusted helper cover the next optimizer step after resume.
    resume_path = output / "cpu_resume.pt"
    atomic_torch_save(resume_path, {"model": left.state_dict(), "optimizer": left_optimizer.state_dict(),
                                  "rng": capture_rng_state()})
    uninterrupted, _, _ = adversarial_optimizer_step(left, left_optimizer, batch, mean, std, _forward)
    uninterrupted_state = {key: value.detach().clone() for key, value in left.state_dict().items()}
    saved = torch.load(resume_path, map_location="cpu", weights_only=False)
    right.load_state_dict(saved["model"], strict=True)
    right_optimizer.load_state_dict(saved["optimizer"])
    restore_rng_state(saved["rng"])
    resumed, _, _ = adversarial_optimizer_step(right, right_optimizer, batch, mean, std, _forward)
    resume = certify_numerical_repeatability(losses=[float(uninterrupted), float(resumed)],
        states=[uninterrupted_state, right.state_dict()])
    assert_finite_state_dict(right.state_dict(), label="CPU resumed model")

    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    selected = selected_payload["model"]
    require(state_dict_sha256(selected) == frozen["selected_state_sha256"], "Selected reference tensor identity changed")
    model.load_state_dict(selected, strict=True)
    model.eval().requires_grad_(False)
    sensitivity = []
    for offset in range(0, 256, 64):
        fixture = Batch.from_data_list(prefix[offset:offset + 64])
        fixture_target = (fixture.y.view(-1) - mean) / std
        with torch.no_grad():
            clean_prediction = _forward(model, fixture)
        require(bool(torch.isfinite(clean_prediction).all()), "Selected reference inference is nonfinite")
        rng = capture_rng_state()
        perturb = torch.empty((fixture.num_nodes, 192)).uniform_(-0.001, 0.001).requires_grad_()
        adversarial_losses, adversarial_shifts = [], []
        for index in range(3):
            with embedding_perturbation(model, perturb):
                prediction = _forward(model, fixture)
                loss = functional.l1_loss(prediction, fixture_target)
                gradient, = torch.autograd.grad(loss, perturb)
            require(bool(torch.isfinite(gradient).all()), "Selected-reference sensitivity gradient is nonfinite")
            adversarial_losses.append(float(loss.detach() * std))
            adversarial_shifts.append(float((prediction.detach() - clean_prediction).abs().mean() * std))
            if index < 2:
                perturb = (perturb.detach() + 0.001 * gradient.sign()).requires_grad_()
        restore_rng_state(rng)
        random_losses = []
        with torch.no_grad():
            for scale in (0.001, 0.002, 0.003):
                random_delta = torch.empty((fixture.num_nodes, 192)).uniform_(-scale, scale)
                with embedding_perturbation(model, random_delta):
                    prediction = _forward(model, fixture)
                random_losses.append(float(functional.l1_loss(prediction, fixture_target) * std))
            require(torch.equal(clean_prediction, _forward(model, fixture)), "Sensitivity hook cleanup changed inference")
        sensitivity.append({"source_start": offset, "rows": 64,
            "clean_mae_eV": float(functional.l1_loss(clean_prediction, fixture_target) * std),
            "adversarial_mae_eV": adversarial_losses, "random_mae_eV": random_losses,
            "adversarial_mean_absolute_shift_eV": adversarial_shifts})
        require(all(math.isfinite(value) for value in adversarial_losses + random_losses + adversarial_shifts +
                    [sensitivity[-1]["clean_mae_eV"]]), "Sensitivity contains nonfinite measurements")
    require(not model.node_emb._forward_hooks and state_dict_sha256(model.state_dict()) == frozen["selected_state_sha256"],
            "Sensitivity altered selected reference state or retained a hook")
    sensitivity_record = {"status": "descriptive-only", "role": "training-prefix-0-256",
        "selected_checkpoint_sha256": frozen["selected_checkpoint_sha256"], "batches": sensitivity,
        "mean_clean_mae_eV": sum(row["clean_mae_eV"] for row in sensitivity) / 4,
        "no_scientific_nomination": True}
    atomic_json(output / "sensitivity.json", sensitivity_record)
    result = {"accepted": True, "scope": "CPU engineering qualification; not formal training or scientific nomination",
        "freeze_sha256": sha256_file(freeze_path), "prospective_sha256": frozen["prospective_sha256"],
        "source_idx_sha256": row_sha, "target_sha256": target_sha, "rows": 256,
        "cpu_threads": torch.get_num_threads(), "determinism": determinism, "flag_config": CONFIG,
        "parameters": parameters, "stored_tensor_elements": stored_count, "state_keys_unchanged": True,
        "initial_state_sha256": INITIAL_STATE_SHA256, "initial_state_file_sha256": sha256_file(initial_path),
        "zero_perturbation_and_cleanup_exact": True, "finite_nonzero_perturbation_gradients": perturbation_norms,
        "finite_nonzero_model_gradients": True, "maximum_clipped_gradient_delta": gradient_delta,
        "explicit_accumulation": accumulation, "resume": resume, "hook_cleanup": True,
        "sensitivity_sha256": sha256_file(output / "sensitivity.json"), "formal_sample_presentations": 0,
        "development_role_read": False, "protected_roles_read": False,
        "wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - started_process_cpu}
    require(all(math.isfinite(float(value)) for value in perturbation_norms + [result["wall_seconds"], gradient_delta]),
            "Qualification contains nonfinite measurements")
    atomic_json(output / "qualification_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=Path(__file__).with_name("cpu_frozen.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("qualification"))
    args = parser.parse_args()
    try:
        run(args.freeze.resolve(), args.output.resolve())
    except Exception as exc:
        from molgap.training_reproducibility import atomic_json
        atomic_json(args.output.resolve() / "cpu_failure.json", {"accepted": False,
            "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
