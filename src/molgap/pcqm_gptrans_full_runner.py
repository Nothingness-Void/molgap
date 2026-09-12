"""Resumable full-train GPTrans-T run on the accepted PCQM4Mv2 cache."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from .pcqm_k1_full_runner import (
    BATCHES_PER_PASS,
    CHECKPOINT_EVERY_STEPS,
    FULL_MANIFEST_CANONICAL_SHA256,
    PHYSICAL_BATCH,
    SAMPLE_PRESENTATIONS,
    TRAIN_ROWS,
    _architecture_source_sha256,
    _load_graphs,
    _loader,
    _target_stats,
    validate_source_archive,
    validate_full_manifest,
)
from .training_reproducibility import (
    CHECKPOINT_FORMAT,
    MODEL_BUNDLE_FORMAT,
    assert_finite_state_dict,
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    canonical_fingerprint,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)


MODEL_ID = "gptrans_t_core"
SEED = 42
TOTAL_STEPS = SAMPLE_PRESENTATIONS // PHYSICAL_BATCH
WARMUP_STEPS = 10_417
LEARNING_RATE = 1.0e-3
MINIMUM_LEARNING_RATE = 1.0e-6
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 1.0
EMA_DECAY = 0.9999
CHECKPOINT_FORMAT_ID = "molgap-pcqm-gptrans-t-full-checkpoint-v1"
RESULT_FORMAT = "molgap-pcqm-gptrans-t-full-result-v1"
EXPECTED_PARAMETERS = 5_246_817
EXPECTED_INITIAL_SHA256 = "e43af04fa1bd3c9b6d48bc61d92b861d423e920d91edb70091abe8ab9d9e1045"
GPTRANS_SOURCE_SHA256 = "76020400020a475c53cf3628a820b77bf49dc3b8df7dff9ad7506e594126778e"

TRAINING_CONTRACT = {
    "format": "molgap-pcqm-gptrans-t-full-contract-v1",
    "benchmark_id": "ogb-lsc-pcqm4mv2-gap",
    "model_id": MODEL_ID,
    "architecture": {
        "source_file": "src/molgap/gptrans.py",
        "source_sha256_lf": GPTRANS_SOURCE_SHA256,
        "node_channels": 256,
        "pair_channels": 32,
        "num_layers": 12,
        "num_heads": 8,
        "shortest_path_cap": 20,
        "dropout": 0.1,
        "drop_path": 0.1,
        "layer_scale": 1.0,
        "parameter_count": EXPECTED_PARAMETERS,
        "seed42_initial_model_sha256": EXPECTED_INITIAL_SHA256,
    },
    "input_modality": "official-ogb-topology-only",
    "ogb_runtime": {
        "version": "1.3.6",
        "source_archive_sha256": "a18d4cacc6a35ad24938f52cfe197a255a5f64bb197f8d0f056c204467ec1e33",
    },
    "data_role": "official-train-full",
    "train_rows": TRAIN_ROWS,
    "full_manifest_canonical_sha256": FULL_MANIFEST_CANONICAL_SHA256,
    "seed": SEED,
    "precision": "fp32",
    "tf32_enabled": False,
    "deterministic_algorithms": True,
    "physical_batch_per_device": PHYSICAL_BATCH,
    "device_count": 1,
    "gradient_accumulation_steps": 1,
    "tail_batch_policy": "drop_last-global-pass",
    "rows_dropped_per_complete_pass": TRAIN_ROWS % PHYSICAL_BATCH,
    "row_order_policy": "seed42-global-randperm-by-pass-v1",
    "optimizer": "adamw",
    "optimizer_fused": False,
    "learning_rate": LEARNING_RATE,
    "minimum_learning_rate": MINIMUM_LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip_norm": GRADIENT_CLIP,
    "scheduler": "four-stage-warmup-cosine-per-optimizer-step",
    "warmup_steps": WARMUP_STEPS,
    "max_optimizer_steps": TOTAL_STEPS,
    "sample_presentations": SAMPLE_PRESENTATIONS,
    "ema_decay": EMA_DECAY,
    "loss": "normalized-gap-l1",
    "target_transform": "full-train-mean-sample-std",
    "selection": "fixed-final-step-ema-no-development-or-official-validation",
    "checkpoint_every_optimizer_steps": CHECKPOINT_EVERY_STEPS,
    "finite_check_every_optimizer_steps": 50,
    "loader_workers": 2,
}
TRAINING_CONTRACT_SHA256 = canonical_fingerprint(TRAINING_CONTRACT)


class ExponentialMovingAverage:
    def __init__(self, model, decay: float) -> None:
        self.decay = float(decay)
        self.state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    def update(self, model) -> None:
        for name, value in model.state_dict().items():
            target = self.state[name]
            if value.is_floating_point():
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value)

    def load_state_dict(self, state) -> None:
        if state.keys() != self.state.keys():
            raise RuntimeError("EMA state keys changed")
        self.state = {name: value.clone() for name, value in state.items()}

    def state_dict(self):
        return self.state


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _make_model():
    from .gptrans import OGBGPTransTiny

    model = OGBGPTransTiny(
        node_channels=256,
        pair_channels=32,
        num_layers=12,
        num_heads=8,
        shortest_path_cap=20,
        dropout=0.1,
        drop_path=0.1,
        layer_scale=1.0,
        n_targets=1,
    )
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("GPTrans-T parameter count changed")
    if _state_sha256(model) != EXPECTED_INITIAL_SHA256:
        raise RuntimeError("GPTrans-T seed-42 initial state changed")
    return model


def _learning_rate(step_after_update: int) -> float:
    if step_after_update <= WARMUP_STEPS:
        stage = min(4, math.ceil(step_after_update * 60 / TOTAL_STEPS))
        return LEARNING_RATE * stage / 4
    progress = (step_after_update - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return MINIMUM_LEARNING_RATE + 0.5 * (
        LEARNING_RATE - MINIMUM_LEARNING_RATE
    ) * (1.0 + math.cos(math.pi * progress))


def _set_learning_rate(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _forward(model, batch):
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)


def _optimizer_step(model, optimizer, batch, mean, std, *, finite_check: bool):
    import torch
    import torch.nn.functional as functional

    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, batch)
    target = (batch.y.view(-1).float() - mean) / std
    loss = functional.l1_loss(prediction, target)
    if finite_check and not bool(torch.isfinite(loss)):
        raise RuntimeError("GPTrans-T loss became non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if finite_check and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("GPTrans-T gradient became non-finite")
    optimizer.step()
    return loss.detach()


def _runtime_gate():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name.upper():
        raise RuntimeError(f"Expected A100, received {name}")
    return name


def run_preflight(*, dataset_root: Path, manifest_path: Path, output: Path) -> dict:
    import torch

    from .pcqm_k1_full_runner import (
        _batch_sha256,
        _gpu_utilization_percent,
        PREFLIGHT_MEASURED_STEPS,
        PREFLIGHT_WARMUP_STEPS,
    )

    determinism = configure_fp32_determinism(SEED)
    accelerator = _runtime_gate()
    source_path = Path(__file__).resolve().parents[1] / "gptrans.py"
    if _architecture_source_sha256(source_path) != GPTRANS_SOURCE_SHA256:
        raise RuntimeError("GPTrans-T source hash changed")
    runtime = build_runtime_manifest(determinism)
    _, paths = validate_full_manifest(dataset_root, manifest_path, verify_content=True)
    graphs, shards = _load_graphs(paths)
    mean_value, std_value = _target_stats(shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    batch = next(iter(_loader(graphs, pass_index=0, start_batch=0))).to(
        "cuda", non_blocking=True
    )
    if int(batch.num_graphs) != PHYSICAL_BATCH:
        raise RuntimeError("Preflight did not produce a full 128-graph batch")

    configure_fp32_determinism(SEED)
    model = _make_model().to("cuda")
    initial_model_sha256 = _state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model.train()
    torch.cuda.reset_peak_memory_stats()
    batches = iter(_loader(graphs, pass_index=0, start_batch=0))
    for index in range(PREFLIGHT_WARMUP_STEPS):
        item = batch if index == 0 else next(batches).to("cuda", non_blocking=True)
        _set_learning_rate(optimizer, _learning_rate(index + 1))
        _optimizer_step(model, optimizer, item, mean, std, finite_check=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(PREFLIGHT_MEASURED_STEPS):
        item = next(batches).to("cuda", non_blocking=True)
        _set_learning_rate(
            optimizer, _learning_rate(PREFLIGHT_WARMUP_STEPS + index + 1)
        )
        _optimizer_step(model, optimizer, item, mean, std, finite_check=index % 50 == 0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    reserve = 1.0 - peak_reserved / total_memory
    graphs_per_second = PREFLIGHT_MEASURED_STEPS * PHYSICAL_BATCH / elapsed
    estimated_hours = SAMPLE_PRESENTATIONS / graphs_per_second / 3600.0
    accepted = reserve >= 0.15 and estimated_hours <= 11.5
    if not accepted:
        raise RuntimeError(
            f"GPTrans-T preflight rejected: reserve={reserve:.3f}, "
            f"estimated_hours={estimated_hours:.2f}"
        )

    certificate = {
        "format": "molgap-gptrans-full-runtime-certificate-v1",
        "status": "accepted",
        "accelerator": accelerator,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "initial_model_sha256": initial_model_sha256,
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "estimated_training_hours": estimated_hours,
        "memory_reserve_fraction": reserve,
        "calibration_fixture_sha256": _batch_sha256(batch),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    certificate_id = canonical_fingerprint(certificate)
    result = {
        "format": "molgap-gptrans-t-full-preflight-v1",
        "accepted": True,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "initial_model_sha256": initial_model_sha256,
        "runtime_manifest": runtime,
        "runtime_certificate": certificate,
        "runtime_certificate_id": certificate_id,
        "target_stats": {"mean_eV": mean_value, "sample_std_eV": std_value},
        "optimizer_step_calibration": {
            "warmup_steps": PREFLIGHT_WARMUP_STEPS,
            "measured_steps": PREFLIGHT_MEASURED_STEPS,
            "elapsed_seconds": elapsed,
            "graphs_per_second": graphs_per_second,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_memory_bytes": total_memory,
            "memory_reserve_fraction": reserve,
            "gpu_utilization_percent": _gpu_utilization_percent(),
            "estimated_training_hours": estimated_hours,
        },
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", certificate)
    atomic_json(output / "preflight.json", result)
    return result


def _save_checkpoint(path: Path, *, model, ema, optimizer, global_step: int,
                     target_stats: dict, trace: list[dict], cache_sha256: str,
                     runtime_fingerprint: str, certificate_id: str,
                     source_commit: str, source_archive_sha256: str,
                     initial_model_sha256: str) -> None:
    pass_index, next_batch = divmod(global_step, BATCHES_PER_PASS)
    atomic_torch_save(path, {
        "format": CHECKPOINT_FORMAT_ID,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "initial_model_sha256": initial_model_sha256,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "pass_index": pass_index,
        "next_batch_in_pass": next_batch,
        "target_stats": target_stats,
        "trace": trace,
        "rng_state": capture_rng_state(),
        "cache_sha256": cache_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_certificate_id": certificate_id,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    })


def train_full(*, dataset_root: Path, manifest_path: Path, output: Path,
               preflight_path: Path, source_commit: str, source_archive: Path,
               resume: bool,
               max_wall_seconds: int | None = None) -> dict:
    import torch

    if len(source_commit) != 40:
        raise ValueError("A committed source identity is required")
    source_archive_sha256 = validate_source_archive(source_archive, source_commit)
    determinism = configure_fp32_determinism(SEED)
    accelerator = _runtime_gate()
    runtime = build_runtime_manifest(determinism)
    source_path = Path(__file__).resolve().parents[1] / "gptrans.py"
    if _architecture_source_sha256(source_path) != GPTRANS_SOURCE_SHA256:
        raise RuntimeError("GPTrans-T source changed after preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("accepted") is not True:
        raise RuntimeError("Accepted GPTrans-T preflight required")
    certificate = preflight["runtime_certificate"]
    certificate_id = canonical_fingerprint(certificate)
    if (preflight.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256
            or certificate_id != preflight.get("runtime_certificate_id")
            or certificate.get("runtime_fingerprint") != runtime["runtime_fingerprint"]
            or certificate.get("accelerator") != accelerator):
        raise RuntimeError("Preflight identity differs from the training runtime")

    completion_path = output / "completion_manifest.json"
    if completion_path.is_file():
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        if (completed.get("complete") is True
                and completed.get("training_contract_sha256") == TRAINING_CONTRACT_SHA256):
            return completed
        raise RuntimeError("Incompatible completion artifact exists")

    manifest, paths = validate_full_manifest(dataset_root, manifest_path,
                                              verify_content=True)
    graphs, shards = _load_graphs(paths)
    mean_value, std_value = _target_stats(shards)
    target_stats = {"mean_eV": mean_value, "sample_std_eV": std_value}
    if preflight.get("target_stats") != target_stats:
        raise RuntimeError("Target statistics changed since preflight")

    configure_fp32_determinism(SEED)
    model = _make_model().to("cuda")
    initial_model_sha256 = _state_sha256(model)
    if certificate.get("initial_model_sha256") != initial_model_sha256:
        raise RuntimeError("Preflight initialization hash differs from training")
    ema = ExponentialMovingAverage(model, EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "last_checkpoint.pt"
    trace_path = output / "trace.json"
    trace: list[dict] = []
    global_step = 0
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Resume requested without last_checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        identities = {
            "format": CHECKPOINT_FORMAT_ID,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "cache_sha256": manifest["aggregates"]["topology"],
            "runtime_fingerprint": runtime["runtime_fingerprint"],
            "runtime_certificate_id": certificate_id,
            "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "initial_model_sha256": initial_model_sha256,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        }
        if any(checkpoint.get(key) != value for key, value in identities.items()):
            raise RuntimeError("Resume checkpoint identity changed")
        if checkpoint.get("target_stats") != target_stats:
            raise RuntimeError("Resume target statistics changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        if divmod(global_step, BATCHES_PER_PASS) != (
                int(checkpoint["pass_index"]), int(checkpoint["next_batch_in_pass"])):
            raise RuntimeError("Resume global cursor is inconsistent")
        trace = list(checkpoint["trace"])
        restore_rng_state(checkpoint["rng_state"])
    elif checkpoint_path.exists():
        raise FileExistsError("Existing checkpoint requires --resume")

    atomic_json(output / "training_contract.json", TRAINING_CONTRACT)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", certificate)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    segment_started = time.perf_counter()
    started = segment_started
    segment_loss = torch.zeros((), dtype=torch.float64, device="cuda")
    segment_rows = 0
    pause = False
    while global_step < TOTAL_STEPS:
        pass_index, start_batch = divmod(global_step, BATCHES_PER_PASS)
        loader = _loader(graphs, pass_index=pass_index, start_batch=start_batch)
        model.train()
        for batch in loader:
            if int(batch.num_graphs) != PHYSICAL_BATCH:
                raise RuntimeError("Training encountered a partial batch")
            batch = batch.to("cuda", non_blocking=True)
            next_step = global_step + 1
            _set_learning_rate(optimizer, _learning_rate(next_step))
            loss = _optimizer_step(model, optimizer, batch, mean, std,
                                   finite_check=next_step % 50 == 0)
            ema.update(model)
            global_step = next_step
            segment_loss += loss.double() * PHYSICAL_BATCH
            segment_rows += PHYSICAL_BATCH
            wall_due = (max_wall_seconds is not None
                        and time.perf_counter() - started >= max_wall_seconds - 300)
            complete = global_step == TOTAL_STEPS
            if (global_step % CHECKPOINT_EVERY_STEPS == 0) or wall_due or complete:
                torch.cuda.synchronize()
                now = time.perf_counter()
                trace.append({
                    "global_step": global_step,
                    "sample_presentations": global_step * PHYSICAL_BATCH,
                    "train_normalized_mae": float(segment_loss.cpu()) / segment_rows,
                    "segment_seconds": now - segment_started,
                    "graphs_per_second": segment_rows / (now - segment_started),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })
                _save_checkpoint(
                    checkpoint_path, model=model, ema=ema, optimizer=optimizer,
                    global_step=global_step, target_stats=target_stats,
                    trace=trace, cache_sha256=manifest["aggregates"]["topology"],
                    runtime_fingerprint=runtime["runtime_fingerprint"],
                    certificate_id=certificate_id, source_commit=source_commit,
                    source_archive_sha256=source_archive_sha256,
                    initial_model_sha256=initial_model_sha256,
                )
                atomic_json(trace_path, {"segments": trace})
                print(f"gptrans_t_full step={global_step}/{TOTAL_STEPS} "
                      f"train_norm_mae={trace[-1]['train_normalized_mae']:.6f} "
                      f"lr={trace[-1]['learning_rate']:.3e} "
                      f"{trace[-1]['graphs_per_second']:.1f} graphs/s", flush=True)
                segment_started = now
                segment_loss = torch.zeros((), dtype=torch.float64, device="cuda")
                segment_rows = 0
            if wall_due:
                pause = True
                break
            if complete:
                break
        if pause:
            break

    complete = global_step == TOTAL_STEPS
    result = {
        "format": RESULT_FORMAT,
        "status": "complete" if complete else "paused",
        "complete": complete,
        "training_contract": TRAINING_CONTRACT,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "initial_model_sha256": initial_model_sha256,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "runtime_certificate_id": certificate_id,
        "cache_aggregate_sha256": manifest["aggregates"]["topology"],
        "train_rows": TRAIN_ROWS,
        "global_step": global_step,
        "sample_presentations": global_step * PHYSICAL_BATCH,
        "target_stats": target_stats,
        "trace_segments": len(trace),
        "training_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "resume_required": not complete,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "run_summary.json", result)
    if not complete:
        return result

    model.load_state_dict(ema.state_dict(), strict=True)
    assert_finite_state_dict(model.state_dict(), label="GPTrans-T final EMA")
    bundle = {
        "format": MODEL_BUNDLE_FORMAT,
        "model_id": MODEL_ID,
        "model_config": TRAINING_CONTRACT["architecture"],
        "seed": SEED,
        "initial_model_sha256": initial_model_sha256,
        "state_dict": model.state_dict(),
        "target_stats": target_stats,
        "training_contract": TRAINING_CONTRACT,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "training_global_step": global_step,
        "cache_aggregate_sha256": manifest["aggregates"]["topology"],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_torch_save(output / "model_bundle.pt", bundle)
    result.update({
        "model_bundle_sha256": sha256_file(output / "model_bundle.pt"),
        "model_parameters": EXPECTED_PARAMETERS,
        "final_state_sha256": _state_sha256(model),
    })
    result.pop("status")
    atomic_json(output / "run_summary.json", result)
    atomic_json(completion_path, result)
    return result
