"""Bounded, paired K1 FP32/TF32 execution diagnostic on the fixed PCQM 100K role."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import tarfile
import time

import torch
import torch.nn.functional as F

from .pcqm_k1_variants import make_encoder
from .pcqm_k1_variants_runner import (
    BATCH_SIZE,
    DEVELOPMENT_ROWS,
    EPOCHS,
    FIXED_MANIFEST_SHA256,
    LEARNING_RATE,
    ROW_ORDER_FINGERPRINT,
    ROWS_PER_EPOCH,
    SAMPLE_EXPOSURE,
    SEED,
    STEPS_PER_EPOCH,
    TRAIN_ROWS,
    WEIGHT_DECAY,
    _development_loader,
    _forward,
    _target_stats,
    _train_loader,
    compute_row_order_fingerprint,
    find_fixed_cache,
    load_roles,
)
from .training_reproducibility import (
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)
from .research_memory.trace import RMLTraceRecorder


FORMAT = "molgap-k1-tf32-paired-runtime-v1"
MODEL_MODE = "neural_atom_k1_v4"
PARAMETERS = 3_658_817
AUTHORIZED_ROOT = Path("/lustre/home/users/sm2/chou")
ARMS = ("fp32", "tf32_matmul")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_recorder(path: Path, arm: str) -> RMLTraceRecorder:
    return RMLTraceRecorder(
        path,
        trajectory_id=f"TC-k1-tf32-a100-{arm}-s42",
        run_id=f"ims-k1-tf32-paired-v1:{arm}",
        metric_semantics={
            "live_train_metric": {
                "metric": "mean_absolute_error",
                "unit": "normalized_target_units",
                "target": "gap",
                "role_identity": "official_train_prefix_0_100000",
                "weights": "live",
                "direction": "minimize",
            },
            "live_dev_metric": {
                "metric": "mean_absolute_error",
                "unit": "eV",
                "target": "gap",
                "role_identity": "internal_development_100000_150000",
                "weights": "live",
                "direction": "minimize",
            },
            "ema_dev_metric": None,
        },
    )


def _record_epoch(recorder: RMLTraceRecorder, row: dict, checkpoint: Path, trace: list[dict]) -> None:
    elapsed = float(row["training_seconds"]) + float(row["validation_seconds"])
    recorder.checkpoint_event(
        sha256_file(checkpoint),
        optimizer_step=int(row["optimizer_steps"]),
        sample_presentations=int(row["sample_presentations"]),
        epoch_or_pass=float(row["epoch"]),
        learning_rate=float(row["learning_rate"]),
        live_train_metric=float(row["train_normalized_mae"]),
        live_dev_metric=float(row["development_gap_mae_eV"]),
        wall_time_seconds=elapsed,
        cumulative_wall_time_seconds=sum(
            float(item["training_seconds"]) + float(item["validation_seconds"])
            for item in trace
        ),
    )


def _write_role_history(output: Path, arm: str, epoch: int) -> None:
    atomic_json(
        output / "role_history.json",
        {
            "format": "molgap-k1-tf32-observed-role-history-v1",
            "arm": arm,
            "last_observed_epoch": epoch,
            "training_membership": {"role": "official_train_prefix_0_100000", "rows": TRAIN_ROWS},
            "training_labels_read": True,
            "development_prediction_and_labels_read": {
                "role": "internal_development_100000_150000", "rows": DEVELOPMENT_ROWS
            },
            "development_metric_computed_and_selection_used": True,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        },
    )


def _inside_authorized_root(path: Path) -> Path:
    boundary = AUTHORIZED_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=False)
    if resolved == boundary or boundary not in resolved.parents:
        raise ValueError(f"Path outside authorized IMS root: {path}")
    return resolved


def _configure_arm(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown precision arm: {arm}")
    determinism = configure_fp32_determinism(SEED)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA accelerator is required")
    if torch.version.cuda is None or torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("A CUDA Ampere-or-newer GPU is required for both arms")
    if arm == "tf32_matmul":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = False
    enabled = bool(torch.backends.cuda.matmul.allow_tf32)
    if enabled != (arm == "tf32_matmul"):
        raise RuntimeError("Requested TF32 setting was not applied")
    return {
        **determinism,
        "precision": "fp32-storage-tf32-matmul" if enabled else "strict-fp32",
        "tf32_enabled": enabled,
        "cudnn_tf32_enabled": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tf32_probe() -> dict:
    """Require an actual FP32/TF32 numerical difference before charging training."""
    _configure_arm("fp32")
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    left = torch.randn(1024, 1024, generator=generator, device="cuda")
    right = torch.randn(1024, 1024, generator=generator, device="cuda")
    strict = left @ right
    _configure_arm("tf32_matmul")
    reduced = left @ right
    torch.cuda.synchronize()
    max_difference = float((strict - reduced).abs().max().item())
    if not math.isfinite(max_difference) or max_difference <= 0:
        raise RuntimeError("TF32 probe did not demonstrate changed matmul arithmetic")
    return {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "strict_vs_tf32_max_abs_difference": max_difference,
    }


def _calibrate(arm: str, roles, mean_value: float, std_value: float, output: Path) -> dict:
    """Two independently initialized optimizer steps must be byte-identical."""
    determinism = _configure_arm(arm)
    runtime = build_runtime_manifest(determinism)
    batch = next(iter(_train_loader(roles["train"], 0))).to("cuda")
    if int(batch.num_graphs) != BATCH_SIZE:
        raise RuntimeError("Physical batch changed during calibration")
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    results = []
    for _ in range(2):
        _configure_arm(arm)
        model = make_encoder(MODEL_MODE).to("cuda").train()
        if sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS:
            raise RuntimeError("K1 parameter identity changed")
        initial_sha = _state_sha(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(model, batch)
        loss = F.l1_loss(prediction, (batch.y.view(-1) - mean) / std)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite calibration loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        results.append((initial_sha, _state_sha(model), float(loss.item())))
        del model, optimizer
    if results[0] != results[1]:
        raise RuntimeError(f"{arm} optimizer-step calibration is not deterministic")
    certificate = {
        "format": "molgap-k1-tf32-runtime-certificate-v1",
        "status": "accepted",
        "arm": arm,
        "initial_model_sha256": results[0][0],
        "one_step_model_sha256": results[0][1],
        "one_step_loss": results[0][2],
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "precision": determinism,
        "calibration_checks_passed": True,
    }
    for name, value in (
        ("runtime_manifest.json", runtime),
        ("runtime_certificate.json", certificate),
    ):
        path = output / name
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise RuntimeError(f"Existing {arm} runtime evidence changed: {name}")
        else:
            atomic_json(path, value)
    return certificate


def _bundle_recovery(output: Path, epoch: int) -> None:
    destination = output / f"recovery_epoch_{epoch + 1:02d}.tar"
    temporary = destination.with_suffix(".tmp")
    with tarfile.open(temporary, "w") as archive:
        for name in (
            "last_checkpoint.pt",
            "best_model.pt",
            "best_development_payload.pt",
            "trace.json",
            "runtime_certificate.json",
        ):
            archive.add(output / name, arcname=name)
    os.replace(temporary, destination)


def _train_arm(
    arm: str,
    roles,
    root: Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    target_stats: tuple[float, float],
    resume: bool,
) -> dict:
    output = root / arm
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    start_path = output / "run_start.json"
    if metrics_path.exists():
        if not resume:
            raise FileExistsError(f"Existing terminal arm requires explicit resume: {arm}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            metrics.get("complete") is not True
            or metrics.get("arm") != arm
            or metrics.get("source_commit") != source_commit
            or metrics.get("source_archive_sha256") != source_archive_sha256
            or metrics.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
        ):
            raise RuntimeError("Existing terminal arm has invalid identity")
        for name, digest in metrics["artifact_sha256"].items():
            if sha256_file(output / name) != digest:
                raise RuntimeError(f"Completed artifact changed: {name}")
        return metrics

    if not start_path.exists():
        atomic_json(start_path, {"arm": arm, "started_at_utc": _utc_now()})
    certificate = _calibrate(arm, roles, *target_stats, output)
    _configure_arm(arm)
    model = make_encoder(MODEL_MODE).to("cuda")
    initial_sha = _state_sha(model)
    if initial_sha != certificate["initial_model_sha256"]:
        raise RuntimeError("Calibration and training initial states differ")
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    mean = torch.tensor(target_stats[0], device="cuda")
    std = torch.tensor(target_stats[1], device="cuda")
    development_loader = _development_loader(roles["development"])
    canonical = _canonical_recorder(output / "canonical_trace.json", arm)
    trace: list[dict] = []
    best, best_epoch = math.inf, -1
    start_epoch = 0
    checkpoint_path = output / "last_checkpoint.pt"
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(f"Existing checkpoint requires explicit resume: {arm}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "format": FORMAT,
            "arm": arm,
            "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
            "runtime_fingerprint": certificate["runtime_fingerprint"],
            "initial_model_sha256": initial_sha,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise RuntimeError(f"Resume identity mismatch: {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        start_epoch = int(checkpoint["epoch"]) + 1
        if len(trace) != start_epoch or start_epoch > EPOCHS:
            raise RuntimeError("Resume trace cursor changed")
        best, best_epoch = float(checkpoint["best"]), int(checkpoint["best_epoch"])
        for name, digest in checkpoint["best_artifact_sha256"].items():
            if sha256_file(output / name) != digest:
                raise RuntimeError(f"Resume best artifact changed: {name}")
        observed = [
            item for item in canonical.record["observations"]
            if item["event"] == "checkpoint"
        ]
        if len(observed) == start_epoch - 1:
            _record_epoch(canonical, trace[-1], checkpoint_path, trace)
        elif len(observed) != start_epoch:
            raise RuntimeError("Canonical trace and checkpoint cursor disagree")
        if start_epoch:
            atomic_json(output / "trace.json", {"epochs": trace})
            _write_role_history(output, arm, start_epoch - 1)
        iter(development_loader)
        restore_rng_state(checkpoint["rng_state"])

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        normalized_error, rows = 0.0, 0
        for batch in _train_loader(roles["train"], epoch):
            batch = batch.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            target = (batch.y.view(-1) - mean) / std
            loss = F.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            normalized_error += float((prediction.detach() - target).abs().sum())
            rows += int(target.numel())
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - epoch_started
        if rows != ROWS_PER_EPOCH:
            raise RuntimeError("Physical optimizer exposure changed")
        model.eval()
        predictions, targets, source_indices = [], [], []
        with torch.no_grad():
            for batch in development_loader:
                batch = batch.to("cuda", non_blocking=True)
                predictions.append((_forward(model, batch) * std + mean).float().cpu())
                targets.append(batch.y.view(-1).float().cpu())
                source_indices.append(batch.source_idx.view(-1).long().cpu())
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - epoch_started - training_seconds
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        source_idx = torch.cat(source_indices)
        if prediction.numel() != DEVELOPMENT_ROWS:
            raise RuntimeError("Development coverage changed")
        development_mae = float((prediction - target).abs().mean().item())
        if not math.isfinite(development_mae):
            raise RuntimeError("Non-finite development MAE")
        if development_mae < best:
            best, best_epoch = development_mae, epoch
            atomic_torch_save(output / "best_model.pt", model.state_dict())
            atomic_torch_save(
                output / "best_development_payload.pt",
                {"target_eV": target, "prediction_eV": prediction, "source_idx": source_idx},
            )
        row = {
            "epoch": epoch,
            "optimizer_steps": (epoch + 1) * STEPS_PER_EPOCH,
            "sample_presentations": (epoch + 1) * ROWS_PER_EPOCH,
            "train_normalized_mae": normalized_error / rows,
            "development_gap_mae_eV": development_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "best_so_far": best,
        }
        trace.append(row)
        scheduler.step()
        atomic_torch_save(
            checkpoint_path,
            {
                "format": FORMAT,
                "arm": arm,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trace": trace,
                "rng_state": capture_rng_state(),
                "best": best,
                "best_epoch": best_epoch,
                "best_artifact_sha256": {
                    name: sha256_file(output / name)
                    for name in ("best_model.pt", "best_development_payload.pt")
                },
                "source_commit": source_commit,
                "source_archive_sha256": source_archive_sha256,
                "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
                "runtime_fingerprint": certificate["runtime_fingerprint"],
                "initial_model_sha256": initial_sha,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        atomic_json(output / "trace.json", {"epochs": trace})
        _record_epoch(canonical, row, checkpoint_path, trace)
        _write_role_history(output, arm, epoch)
        if (epoch + 1) % 10 == 0:
            _bundle_recovery(output, epoch)
        print(
            f"{arm} ep{epoch:02d} train={row['train_normalized_mae']:.6f} "
            f"dev={development_mae:.6f}eV train_s={training_seconds:.1f} ",
            flush=True,
        )
    if not canonical.record["observations"] or canonical.record["observations"][-1]["event"] != "terminal":
        canonical.terminal_event()
    artifacts = {
        name: sha256_file(output / name)
        for name in (
            "best_model.pt", "best_development_payload.pt", "last_checkpoint.pt",
            "trace.json", "canonical_trace.json", "role_history.json",
            "runtime_certificate.json", "runtime_manifest.json", "run_start.json",
        )
    }
    metrics = {
        "format": FORMAT,
        "complete": True,
        "arm": arm,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "runtime_fingerprint": certificate["runtime_fingerprint"],
        "model_id": MODEL_MODE,
        "parameter_count": PARAMETERS,
        "initial_model_sha256": initial_sha,
        "precision": certificate["precision"],
        "seed": SEED,
        "physical_batch": BATCH_SIZE,
        "epochs_completed": EPOCHS,
        "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
        "sample_presentations": SAMPLE_EXPOSURE,
        "best_epoch": best_epoch,
        "best_development_gap_mae_eV": best,
        "training_seconds": sum(float(row["training_seconds"]) for row in trace),
        "validation_seconds": sum(float(row["validation_seconds"]) for row in trace),
        "training_graphs_per_second": SAMPLE_EXPOSURE / sum(float(row["training_seconds"]) for row in trace),
        "steady_training_graphs_per_second": (
            (EPOCHS - 5) * ROWS_PER_EPOCH
            / sum(float(row["training_seconds"]) for row in trace[5:])
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "started_at_utc": json.loads(start_path.read_text(encoding="utf-8"))["started_at_utc"],
        "ended_at_utc": _utc_now(),
        "artifact_sha256": artifacts,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(metrics_path, metrics)
    return metrics


def run(
    *,
    cache_root: Path,
    output_root: Path,
    source_commit: str,
    source_archive_sha256: str,
    resume: bool = False,
) -> dict:
    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise ValueError("Frozen source commit and archive SHA-256 are required")
    cache_root = _inside_authorized_root(cache_root)
    output_root = _inside_authorized_root(output_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError("A populated output root requires explicit resume")
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["MOLGAP_FIXED_CACHE_ROOT"] = str(cache_root)
    os.environ["MOLGAP_PLATFORM_ID"] = "ims-a100"
    if compute_row_order_fingerprint() != ROW_ORDER_FINGERPRINT:
        raise RuntimeError("Frozen K1-v4 row-order fingerprint changed")
    hardware_probe = _tf32_probe()
    atomic_json(output_root / "hardware_probe.json", hardware_probe)
    cache_path, manifest = find_fixed_cache()
    if cache_path != cache_root:
        raise RuntimeError("Accepted cache resolved to a different root")
    roles = load_roles(cache_path, manifest)
    target_stats = _target_stats(roles["train"])
    arms = {}
    for arm in ARMS:
        arms[arm] = _train_arm(
            arm,
            roles,
            output_root,
            source_commit=source_commit,
            source_archive_sha256=source_archive_sha256,
            target_stats=target_stats,
            resume=resume,
        )
    if arms["fp32"]["initial_model_sha256"] != arms["tf32_matmul"]["initial_model_sha256"]:
        raise RuntimeError("Precision arms did not start from identical weights")
    result = {
        "format": FORMAT,
        "complete": True,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "hardware_probe": hardware_probe,
        "arms": {arm: {
            "best_development_gap_mae_eV": metrics["best_development_gap_mae_eV"],
            "best_epoch": metrics["best_epoch"],
            "training_seconds": metrics["training_seconds"],
            "validation_seconds": metrics["validation_seconds"],
            "training_graphs_per_second": metrics["training_graphs_per_second"],
            "steady_training_graphs_per_second": metrics["steady_training_graphs_per_second"],
            "artifact_sha256": metrics["artifact_sha256"],
        } for arm, metrics in arms.items()},
        "tf32_minus_fp32_mae_eV": (
            arms["tf32_matmul"]["best_development_gap_mae_eV"]
            - arms["fp32"]["best_development_gap_mae_eV"]
        ),
        "training_speedup_fp32_over_tf32": (
            arms["fp32"]["training_seconds"] / arms["tf32_matmul"]["training_seconds"]
        ),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output_root / "comparison.json", result)
    return result
