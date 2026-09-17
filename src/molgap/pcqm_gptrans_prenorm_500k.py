"""V5 matched 500K screen for GPTrans-T reference and Pair PreNorm."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

from .pcqm_gptrans_v4 import (
    ExponentialMovingAverage,
    _batch_sha256,
    _forward,
    _make_model,
    _state_sha256,
)
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .pcqm_k1_scale_runner import _targets, find_cache, load_roles
from .screen_policy import canonical_fingerprint, validate_runtime_certificate
from .training_reproducibility import (
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)


MODES = ("reference", "pair_prenorm")
PARAMETERS = 5_246_817
SEED = 42
BATCH_SIZE = 128
TRAIN_ROWS = 500_000
DEVELOPMENT_ROWS = 50_000
STEPS_PER_EPOCH = TRAIN_ROWS // BATCH_SIZE
ROWS_PER_EPOCH = STEPS_PER_EPOCH * BATCH_SIZE
EPOCHS = 60
WARMUP_EPOCHS = 4
SAMPLE_PRESENTATIONS = ROWS_PER_EPOCH * EPOCHS
MINIMUM_GAIN_EV = 0.003
EMA_DECAY = 0.9999


def learning_rate(epoch: int) -> float:
    if not 0 <= epoch < EPOCHS:
        raise ValueError(epoch)
    if epoch < WARMUP_EPOCHS:
        return 1.0e-3 * float(epoch + 1) / WARMUP_EPOCHS
    progress = float(epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS - 1)
    return 1.0e-6 + (1.0e-3 - 1.0e-6) * (
        1.0 + math.cos(math.pi * progress)
    ) / 2.0


def scientific_contract(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    return {
        "policy": "molgap-reference-comparability-v4",
        "benchmark_id": "pcqm-fixed500k-dev50k-gptrans60-v5",
        "data_role_fingerprint": FIXED_500K_MANIFEST_SHA256,
        "row_order_fingerprint": "global-randperm-seed42-plus-epoch-drop-last32",
        "feature_fingerprint": "ogb-node9-edge3-shortest-path20-no-geometry-input",
        "target_fingerprint": "pcqm4mv2-gap-eV-direct",
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-unfused-foreachFalse-lr1e-3-wd0.05-clip1",
        "schedule_fingerprint": "warmup4-cosine60-epoch59-1e-6",
        "loss_fingerprint": "normalized-gap-l1",
        "target_transform_fingerprint": "all500k-train-only-mean-unbiased-std",
        "selection_fingerprint": "best-development-ema9999-60epochs",
        "role_access_fingerprint": (
            "official-train-derived-train-and-internal-development-only"
        ),
        "sample_exposure": SAMPLE_PRESENTATIONS,
        "tail_batch_policy": "drop_last",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "scale_rows": TRAIN_ROWS,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "mode": mode,
    }


def make_model(mode: str):
    if mode not in MODES:
        raise ValueError(mode)
    model = _make_model(None, mode)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != PARAMETERS:
        raise RuntimeError(f"GPTrans parameter count changed: {parameters}")
    return model


def optimizer_for(model):
    import torch

    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate(0), weight_decay=0.05, foreach=False
    )


def loader(graphs, epoch: int | None = None):
    import torch
    from torch_geometric.loader import DataLoader

    workers = int(os.environ.get("MOLGAP_V5_LOADER_WORKERS", "2"))
    sampler = None
    if epoch is not None:
        sampler = torch.randperm(
            len(graphs), generator=torch.Generator().manual_seed(SEED + epoch)
        )[:ROWS_PER_EPOCH].tolist()
    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(15000 + (epoch or 0)),
    )


def optimizer_step(model, optimizer, ema, batch, mean, std):
    import torch

    if batch.num_graphs != BATCH_SIZE:
        raise RuntimeError("Non-128 optimizer batch")
    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, batch)
    target = (batch.y.view(-1) - mean) / std
    loss = torch.nn.functional.l1_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    ema.update(model)
    return loss.detach()


def evaluate(model, ema, graphs, mean, std):
    import torch

    live = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema.state_dict(), strict=True)
    model.eval()
    rows = {"prediction": [], "target": [], "source_idx": []}
    with torch.no_grad():
        for batch in loader(graphs):
            batch = batch.to("cuda", non_blocking=True)
            rows["prediction"].append((_forward(model, batch) * std + mean).cpu())
            rows["target"].append(batch.y.view(-1).cpu())
            rows["source_idx"].append(batch.source_idx.view(-1).cpu().long())
    model.load_state_dict(live, strict=True)
    result = {key: torch.cat(values) for key, values in rows.items()}
    expected = torch.arange(TRAIN_ROWS, TRAIN_ROWS + DEVELOPMENT_ROWS)
    if not torch.equal(result["source_idx"], expected):
        raise RuntimeError("Development row order changed")
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("Non-finite development output")
    result["mae_eV"] = float(
        (result["prediction"] - result["target"]).abs().mean()
    )
    return result


def _runtime_certificate(
    runtime: dict, settings: dict, fixture_sha: str, state_sha: str
) -> tuple[dict, str]:
    import torch

    certificate = {
        "format": "molgap-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": "scnet-kunshan",
        "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": BATCH_SIZE,
        "tail_batch_policy": "drop_last",
        "loader_workers": int(os.environ.get("MOLGAP_V5_LOADER_WORKERS", "2")),
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(settings),
        "calibration_fixture_sha256": fixture_sha,
        "calibration_output_sha256": state_sha,
        "calibration_checks_passed": True,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
    }
    certificate_id = canonical_fingerprint(certificate)
    validate_runtime_certificate(
        certificate,
        {
            "platform_id": "scnet-kunshan",
            "accelerator": certificate["accelerator"],
            "runtime_certificate_id": certificate_id,
        },
    )
    return certificate, certificate_id


def _check_pair_prenorm() -> dict:
    import torch
    from .gptrans_variants import normalize_pair

    pair = torch.randn(2, 32, 5, 5, device="cuda", requires_grad=True)
    normalized = normalize_pair(pair)
    if not torch.allclose(
        normalized.mean(1), torch.zeros_like(normalized[:, 0]), atol=2e-6
    ):
        raise RuntimeError("Pair PreNorm channel mean check failed")
    normalized.square().mean().backward()
    if not bool(torch.isfinite(pair.grad).all()):
        raise RuntimeError("Pair PreNorm finite-gradient check failed")
    return {
        "channel_mean_zero": True,
        "finite_backward": True,
        "synthetic_data_only": True,
    }


def run(
    output: Path,
    *,
    mode: str,
    source_commit: str,
    source_archive_sha256: str,
    resume: Path | None = None,
    preflight_only: bool = False,
) -> dict:
    import torch

    if mode not in MODES:
        raise ValueError(mode)
    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise ValueError("Committed source and archive identities are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each V5 arm requires exactly one visible accelerator")
    output.mkdir(parents=True, exist_ok=True)
    settings = configure_fp32_determinism(SEED)
    runtime = build_runtime_manifest(settings)
    root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    roles = load_roles(root, manifest)
    targets = _targets(roles["train"]).double()
    mean_value = float(targets.mean())
    std_value = float(targets.std(unbiased=True).clamp_min(1e-6))
    contract = scientific_contract(mode)
    contract["target_transform_fingerprint"] = canonical_fingerprint(
        {"mean": mean_value, "std": std_value}
    )
    atomic_json(output / "runtime.json", runtime)
    atomic_json(output / "scientific_contract.json", contract)
    atomic_json(output / "data_manifest.json", manifest)

    batch = next(iter(loader(roles["train"], 0))).to("cuda")
    fixture_sha = _batch_sha256(batch)
    calibrations = []
    pair_check = _check_pair_prenorm() if mode == "pair_prenorm" else None
    initial_sha = None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model = make_model(mode).to("cuda").train()
        initial_sha = _state_sha256(model)
        optimizer = optimizer_for(model)
        ema = ExponentialMovingAverage(model)
        mean = torch.tensor(mean_value, device="cuda")
        std = torch.tensor(std_value, device="cuda")
        losses = [
            float(optimizer_step(model, optimizer, ema, batch, mean, std))
            for _ in range(3)
        ]
        calibrations.append(
            {"initial": initial_sha, "losses": losses, "state": _state_sha256(model)}
        )
        del model, optimizer, ema
    if calibrations[0] != calibrations[1]:
        raise RuntimeError("Deterministic optimizer-step calibration failed")
    peak = int(torch.cuda.max_memory_reserved())
    total = int(torch.cuda.get_device_properties(0).total_memory)
    if peak > 0.85 * total:
        raise RuntimeError("BS128 optimizer-inclusive memory reserve below 15%")
    certificate, certificate_id = _runtime_certificate(
        runtime, settings, fixture_sha, calibrations[0]["state"]
    )
    atomic_json(output / "runtime_certificate.json", certificate)
    atomic_json(
        output / "preflight.json",
        {
            "mode": mode,
            "parameter_count": PARAMETERS,
            "initial_state_sha256": initial_sha,
            "pair_prenorm_check": pair_check,
            "calibrations": calibrations,
            "peak_reserved_mib": peak / 1024**2,
            "total_memory_mib": total / 1024**2,
            "memory_reserve_fraction": 1.0 - peak / total,
        },
    )
    if preflight_only:
        result = {
            "format": "molgap-gptrans-prenorm-500k-v5-preflight-v1",
            "complete": True,
            "training_started": False,
            "mode": mode,
            "parameter_count": PARAMETERS,
            "contract": contract,
            "runtime_certificate_id": certificate_id,
            "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        }
        atomic_json(output / "preflight_complete.json", result)
        return result
    del batch
    torch.cuda.empty_cache()

    configure_fp32_determinism(SEED)
    model = make_model(mode).to("cuda")
    optimizer = optimizer_for(model)
    ema = ExponentialMovingAverage(model)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    start_epoch, trace, best, best_epoch = 0, [], math.inf, -1
    if resume is not None:
        checkpoint = torch.load(
            resume / "last_checkpoint.pt", map_location="cpu", weights_only=False
        )
        if checkpoint["contract"] != contract or checkpoint["mode"] != mode:
            raise RuntimeError("Resume scientific contract changed")
        if checkpoint["source_commit"] != source_commit:
            raise RuntimeError("Resume source commit changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng_state(checkpoint["rng"])
        start_epoch = checkpoint["next_epoch"]
        trace = checkpoint["trace"]
        best = checkpoint["best"]
        best_epoch = checkpoint["best_epoch"]
        for name in ("best_model.pt", "best_predictions.pt", "initial_state.pt"):
            if (resume / name).resolve() != (output / name).resolve():
                shutil.copy2(resume / name, output / name)
    else:
        atomic_torch_save(
            output / "initial_state.pt",
            {"model": model.state_dict(), "state_sha256": initial_sha},
        )
    print(
        f"PREFLIGHT PASS {mode} parameters={PARAMETERS} "
        f"resume_epoch={start_epoch} peak={peak}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, EPOCHS):
        started = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch)
        model.train()
        loss_sum = torch.zeros((), device="cuda")
        count = 0
        for batch_index, batch in enumerate(loader(roles["train"], epoch)):
            loss_sum += optimizer_step(
                model,
                optimizer,
                ema,
                batch.to("cuda", non_blocking=True),
                mean,
                std,
            )
            count += BATCH_SIZE
            if (batch_index + 1) % 500 == 0:
                print(
                    f"{mode} ep={epoch} batch={batch_index + 1}/{STEPS_PER_EPOCH}",
                    flush=True,
                )
        if count != ROWS_PER_EPOCH:
            raise RuntimeError("Sample exposure changed")
        evaluation = evaluate(model, ema, roles["validation"], mean, std)
        mae = evaluation["mae_eV"]
        if mae < best:
            best, best_epoch = mae, epoch
            atomic_torch_save(
                output / "best_model.pt",
                {
                    "mode": mode,
                    "model": ema.state_dict(),
                    "mean": mean_value,
                    "std": std_value,
                    "contract": contract,
                    "epoch": epoch,
                    "source_commit": source_commit,
                },
            )
            atomic_torch_save(output / "best_predictions.pt", evaluation)
        row = {
            "epoch": epoch,
            "train_mae_eV": float(loss_sum) / STEPS_PER_EPOCH * std_value,
            "development_mae_eV": mae,
            "learning_rate": learning_rate(epoch),
            "seconds": time.monotonic() - started,
            "global_step": (epoch + 1) * STEPS_PER_EPOCH,
            "sample_presentations": (epoch + 1) * ROWS_PER_EPOCH,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        trace.append(row)
        atomic_json(output / "trace.json", {"epochs": trace})
        atomic_torch_save(
            output / "last_checkpoint.pt",
            {
                "format": "molgap-gptrans-prenorm-500k-v5-checkpoint-v1",
                "mode": mode,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng_state(),
                "next_epoch": epoch + 1,
                "trace": trace,
                "best": best,
                "best_epoch": best_epoch,
                "contract": contract,
                "source_commit": source_commit,
                "source_archive_sha256": source_archive_sha256,
                "runtime_certificate_id": certificate_id,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
                "test_challenge_role_read": False,
            },
        )
        atomic_json(
            output / "progress.json",
            {"status": "RUNNING", "next_epoch": epoch + 1, "best": best},
        )
        print(
            f"{mode} ep{epoch:02d} dev={mae:.8f} best={best:.8f}@{best_epoch} "
            f"{row['seconds']:.1f}s",
            flush=True,
        )

    result = {
        "format": "molgap-gptrans-prenorm-500k-v5-result-v1",
        "complete": True,
        "mode": mode,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "parameter_count": PARAMETERS,
        "contract": contract,
        "runtime_certificate_id": certificate_id,
        "best_epoch": best_epoch,
        "development_gap_mae_eV": best,
        "epochs_completed": len(trace),
        "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "mean_epoch_seconds": sum(item["seconds"] for item in trace) / len(trace),
        "mean_graphs_per_second": SAMPLE_PRESENTATIONS
        / sum(item["seconds"] for item in trace),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "best_model_sha256": sha256_file(output / "best_model.pt"),
        "payload_sha256": sha256_file(output / "best_predictions.pt"),
        "checkpoint_sha256": sha256_file(output / "last_checkpoint.pt"),
        "model_inference_executed_by_acceptance": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "metrics.json", result)
    atomic_json(
        output / "progress.json",
        {
            "status": "COMPLETE",
            "next_epoch": EPOCHS,
            "best": best,
            "best_epoch": best_epoch,
        },
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "completion_manifest.json"
    }
    atomic_json(output / "completion_manifest.json", {**result, "artifact_sha256": artifacts})
    return result
