"""One bounded V5 fixed-PCQM-100K MetaGIN2D architecture screen."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np

from .pcqm_k1_variants_runner import (
    BATCH_SIZE, EPOCHS, FIXED_GEOMETRY_SHA256, FIXED_MANIFEST_SHA256,
    LEARNING_RATE, ROW_ORDER_FINGERPRINT, ROWS_PER_EPOCH, SAMPLE_EXPOSURE,
    SEED, STEPS_PER_EPOCH, WEIGHT_DECAY, _batch_sha256,
    _development_loader, _train_loader, compute_row_order_fingerprint,
    find_fixed_cache, load_roles,
)
from .screen_policy import canonical_fingerprint, validate_screen_arm
from .training_reproducibility import (
    atomic_json, atomic_torch_save, build_runtime_manifest,
    capture_rng_state, configure_fp32_determinism, sha256_file,
)
from .pcqm_metagin import DEPTH, HOPS, WIDTH, MetaGIN2D
from .pcqm_metagin_sidecar import attach_accepted_sidecar


MODEL_ID = "metagin_2d_3hop_4x256"
TRAJECTORY_ID = "TC-metagin-2d-3hop-100k-s42"
ARCHITECTURE = {
    "family": "MetaGIN-derived-independent-2d-backbone",
    "model_id": MODEL_ID,
    "width": WIDTH, "depth": DEPTH, "hops": HOPS,
    "path_attributes": "directed-simple-path-multiplicity-1-2-3plus",
    "virtual_molecular_state": "persistent-after-first-block",
    "mixing": "sequential-hop-message-plus-gated-metaformer",
    "readout": "graph-sum-direct-normalized-gap",
    "raw_features": "accepted-ogb9-bond3-rwse16",
    "geometry": False, "teacher": False, "pretraining": False,
}
MAX_WORKER_SECONDS = 6 * 3600
MIN_MEMORY_RESERVE = 0.15


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fixture_sha256(batch) -> str:
    digest = hashlib.sha256(_batch_sha256(batch).encode("ascii"))
    for name in ("hop2_edge_index", "hop2_count", "hop3_edge_index", "hop3_count"):
        value = getattr(batch, name).detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _step(model, optimizer, batch, mean, std):
    import torch
    import torch.nn.functional as functional

    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)
    target = (batch.y.view(-1).float() - mean) / std
    loss = functional.l1_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite MetaGIN training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return prediction.detach(), target.detach(), float(loss.detach().cpu())


def _evaluate(model, loader, mean, std):
    import torch

    model.eval()
    targets, predictions, source_indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda", non_blocking=True)
            predictions.append((model(batch) * std + mean).float().cpu())
            targets.append(batch.y.view(-1).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())
    target = torch.cat(targets)
    predicted = torch.cat(predictions)
    sources = torch.cat(source_indices)
    expected = torch.arange(100_000, 150_000, dtype=torch.long)
    if not torch.equal(sources, expected):
        raise RuntimeError("Development source-index order changed")
    return float((target - predicted).abs().mean()), target, predicted, sources


def _target_transform(raw_roles, transform_path: Path):
    import torch

    from .comparison_readiness import validate_target_transform_asset
    from .k1_joint_objective import verify_frozen_train_targets

    transform = validate_target_transform_asset(
        json.loads(transform_path.read_text(encoding="utf-8"))
    )
    if (
        transform["asset_id"] != "a5ebd05f82f481020f2059c77a7bc7b9dee84081fdb9806fdc18d93a6b8d89d7"
        or transform["asset_sha256"] != "1631a6842bcd0ed330eada408cc78b3abe05109a1266466eb2f2ded731857932"
        or sha256_file(transform_path) != "20e6730d57080b0bec9901a26a931034aad162848e0075940fd1e1273bf084b3"
    ):
        raise RuntimeError("Frozen K1 target-transform identity changed")
    labels = torch.cat([part._data.y.view(-1).float() for part in raw_roles["train"].datasets])
    sources = torch.cat([part._data.source_idx.view(-1).long() for part in raw_roles["train"].datasets])
    observation = verify_frozen_train_targets(labels, sources, transform)
    return transform, observation


def _preflight(roles, transform, *, output: Path, source_commit: str, sidecar_manifest: dict):
    import torch

    determinism = configure_fp32_determinism(SEED)
    runtime_manifest = build_runtime_manifest(determinism)
    first = next(iter(_train_loader(roles["train"], 0))).to("cuda", non_blocking=True)
    if first.num_graphs != BATCH_SIZE:
        raise RuntimeError("MetaGIN physical batch is not 128")
    fixture_sha = _fixture_sha256(first)
    mean = torch.tensor(float(transform["mean"]), device="cuda")
    std = torch.tensor(float(transform["std"]), device="cuda")
    results = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model = MetaGIN2D().to("cuda").train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            fused=False,
        )
        started = time.perf_counter()
        _, _, loss = _step(model, optimizer, first, mean, std)
        torch.cuda.synchronize()
        results.append((loss, _state_sha256(model), time.perf_counter() - started))
        del model, optimizer
        torch.cuda.empty_cache()
    if results[0][:2] != results[1][:2]:
        raise RuntimeError("MetaGIN optimizer step calibration is not deterministic")
    parameter_count = sum(parameter.numel() for parameter in MetaGIN2D().parameters())
    total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    reserved_mib = torch.cuda.max_memory_reserved() / 1024**2
    reserve = 1.0 - reserved_mib / total_mib
    if reserve < MIN_MEMORY_RESERVE:
        raise RuntimeError("MetaGIN preflight retains less than 15% GPU memory")
    # Conservative lower bound: one measured train step plus no evaluation.
    # A second runtime estimate after the first epoch is the stronger budget gate.
    if max(result[2] for result in results) * STEPS_PER_EPOCH * EPOCHS > MAX_WORKER_SECONDS * 0.8:
        raise RuntimeError("MetaGIN calibrated optimizer throughput exceeds bounded worker budget")
    certificate = {
        "format": "molgap-runtime-certificate-v1", "status": "accepted",
        "platform_id": "kaggle2", "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32", "tf32_enabled": False,
        "deterministic_algorithms": True, "physical_batch_per_device": BATCH_SIZE,
        "tail_batch_policy": "drop_last",
        "software_fingerprint": runtime_manifest["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(determinism),
        "calibration_fixture_sha256": fixture_sha,
        "calibration_model_id": MODEL_ID,
        "calibration_output_sha256": results[0][1],
        "runtime_fingerprint": runtime_manifest["runtime_fingerprint"],
        "calibration_checks_passed": True,
    }
    preflight = {
        "format": "molgap-metagin-2d-preflight-v1", "accepted": True,
        "source_commit": source_commit,
        "architecture": ARCHITECTURE,
        "architecture_config_identity": canonical_fingerprint(ARCHITECTURE),
        "parameter_count": parameter_count,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "sidecar_aggregate_sha256": sidecar_manifest["aggregate_sha256"],
        "runtime_certificate_id": canonical_fingerprint(certificate),
        "calibration_step_seconds": max(result[2] for result in results),
        "peak_reserved_mib": reserved_mib, "total_memory_mib": total_mib,
        "memory_reserve_fraction": reserve,
        "geometry_model_input": False, "teacher_model_input": False,
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "runtime_manifest.json", runtime_manifest)
    atomic_json(output / "runtime_certificate.json", certificate)
    atomic_json(output / "preflight.json", preflight)
    return certificate, preflight


def train_screen(
    output: Path, *, source_commit: str, source_archive_sha256: str,
    sidecar_root: Path, transform_path: Path, run_id: str,
) -> dict:
    import torch

    from .k1_screen_trace import recorder, record_epoch

    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise ValueError("Frozen executable source identity required")
    if output.exists():
        raise RuntimeError("Existing MetaGIN output may not be overwritten")
    output.mkdir(parents=True)
    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    if compute_row_order_fingerprint() != ROW_ORDER_FINGERPRINT:
        raise RuntimeError("Frozen K1 row order implementation changed")
    configure_fp32_determinism(SEED)
    root, manifest = find_fixed_cache()
    raw_roles = load_roles(root, manifest)
    transform, target_observation = _target_transform(raw_roles, transform_path)
    roles, sidecar, sidecar_acceptance = attach_accepted_sidecar(
        raw_roles, sidecar_root, expected_source_commit=source_commit,
    )
    certificate, preflight = _preflight(
        roles, transform, output=output, source_commit=source_commit,
        sidecar_manifest=sidecar,
    )
    atomic_json(output / "target_transform_observation.json", target_observation)
    atomic_json(output / "sidecar_acceptance_binding.json", sidecar_acceptance)
    configure_fp32_determinism(SEED)
    model = MetaGIN2D().to("cuda")
    if sum(value.numel() for value in model.parameters()) != preflight["parameter_count"]:
        raise RuntimeError("MetaGIN model identity changed after preflight")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )
    mean = torch.tensor(float(transform["mean"]), device="cuda")
    std = torch.tensor(float(transform["std"]), device="cuda")
    development = _development_loader(roles["development"])
    canonical = recorder(output, TRAJECTORY_ID, run_id)
    best, best_epoch, trace = math.inf, -1, []
    steps, samples = 0, 0
    torch.cuda.reset_peak_memory_stats()
    started_total = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        absolute, rows = 0.0, 0
        started = time.perf_counter()
        for batch in _train_loader(roles["train"], epoch):
            batch = batch.to("cuda", non_blocking=True)
            prediction, target, _ = _step(model, optimizer, batch, mean, std)
            absolute += float((prediction - target).abs().sum())
            rows += int(target.numel())
            steps += 1
            samples += int(target.numel())
        if rows != ROWS_PER_EPOCH:
            raise RuntimeError("MetaGIN optimizer exposure changed")
        mae, target_eV, prediction_eV, source_idx = _evaluate(
            model, development, mean, std,
        )
        if not math.isfinite(mae):
            raise RuntimeError("Non-finite development MAE")
        improved = mae < best
        if improved:
            best, best_epoch = mae, epoch
            atomic_torch_save(output / "best_model.pt", model.state_dict())
            atomic_torch_save(output / "best_development_payload.pt", {
                "target_eV": target_eV, "prediction_eV": prediction_eV,
                "source_idx": source_idx,
            })
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch, "optimizer_steps": steps,
            "sample_presentations": samples,
            "train_normalized_mae": absolute / rows,
            "development_gap_mae_eV": mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed, "improved": improved,
        }
        trace.append(row)
        scheduler.step()
        atomic_torch_save(output / "last_checkpoint.pt", {
            "format": "molgap-metagin-2d-resumable-checkpoint-v1",
            "epoch": epoch, "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "run_id": run_id, "trajectory_id": TRAJECTORY_ID,
            "runtime_certificate_id": preflight["runtime_certificate_id"],
            "sidecar_aggregate_sha256": sidecar["aggregate_sha256"],
            "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
            "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
            "target_transform_asset_id": transform["asset_id"],
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng_state": capture_rng_state(),
            "trace": trace, "best": best, "best_epoch": best_epoch,
            "best_model_sha256": sha256_file(output / "best_model.pt"),
            "best_development_payload_sha256": sha256_file(output / "best_development_payload.pt"),
        })
        record_epoch(canonical, output, row, observed_steps=steps,
                     observed_samples=samples, elapsed=elapsed)
        atomic_json(output / "trace.json", {"epochs": trace})
        if (epoch + 1) % 10 == 0:
            archive = output / f"recovery_epoch_{epoch + 1:02d}.tar"
            temporary = archive.with_suffix(".tmp")
            with tarfile.open(temporary, "w") as bundle:
                for name in (
                    "last_checkpoint.pt", "best_model.pt", "best_development_payload.pt",
                    "trace.json", "canonical_trace.json", "observed_role_history.json",
                    "preflight.json", "runtime_certificate.json",
                ):
                    bundle.add(output / name, arcname=name)
            temporary.replace(archive)
        print(
            f"{MODEL_ID} ep{epoch:02d} train={row['train_normalized_mae']:.6f} "
            f"dev={mae:.6f}eV {elapsed:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if epoch == 0 and elapsed * EPOCHS > MAX_WORKER_SECONDS * 0.9:
            raise RuntimeError("First-epoch measured runtime exceeds six-hour contract budget")
    if steps != STEPS_PER_EPOCH * EPOCHS or samples != SAMPLE_EXPOSURE:
        raise RuntimeError("MetaGIN final exposure changed")
    result = {
        "format": "molgap-metagin-2d-100k-arm-v1", "complete": True,
        "model_id": MODEL_ID, "architecture": ARCHITECTURE,
        "architecture_config_identity": canonical_fingerprint(ARCHITECTURE),
        "trajectory_id": TRAJECTORY_ID, "run_id": run_id,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "sidecar_aggregate_sha256": sidecar["aggregate_sha256"],
        "runtime_certificate_id": preflight["runtime_certificate_id"],
        "target_transform_asset_id": transform["asset_id"],
        "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
        "training": {
            "parameter_count": preflight["parameter_count"],
            "best_epoch": best_epoch, "development_gap_mae_eV": best,
            "epochs_completed": EPOCHS, "optimizer_steps": steps,
            "sample_presentations": samples,
            "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
            "mean_graphs_per_second": samples / sum(row["seconds"] for row in trace),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            "training_wall_seconds": time.perf_counter() - started_total,
            "best_model_sha256": sha256_file(output / "best_model.pt"),
            "payload_sha256": sha256_file(output / "best_development_payload.pt"),
            "checkpoint_sha256": sha256_file(output / "last_checkpoint.pt"),
        },
        "pure_2d": True, "geometry_attributes_removed_before_batching": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    }
    atomic_json(output / "arm_record.json", result)
    atomic_json(output / "completion_manifest.json", {
        "format": "molgap-metagin-2d-100k-completion-v1", "complete": True,
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir()) if path.is_file()
            and path.name != "completion_manifest.json"
        },
        "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    })
    return result
