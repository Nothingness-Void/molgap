"""V4 reference run for the OGB-rich EdgeState Structural GPS9 model."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .pcqm_gptrans_v4 import (
    BATCHES_PER_EPOCH,
    DEVELOPMENT_ROWS,
    EPOCHS,
    FINITE_CHECK_EVERY_STEPS,
    GEOMETRY_AGGREGATE_SHA256,
    MANIFEST_SHA256,
    PHYSICAL_BATCH,
    SAMPLE_PRESENTATIONS,
    SEED,
    TRAIN_ROWS,
    DeterministicEpochBatchSampler,
    ExponentialMovingAverage,
    FrozenEpochScheduler,
    _batch_sha256,
    _evaluate,
    _optimizer_step,
    _scientific_fields as _shared_scientific_fields,
    _state_sha256,
    _target_stats as _shared_target_stats,
    validate_fixed_assets,
    validate_source_archive,
)
from .pcqm_k1_variants_runner import load_roles
from .pcqm_gap_architecture import make_pcqm_gap_encoder
from .screen_policy import (
    canonical_fingerprint,
    validate_reference_screen_contract,
    validate_runtime_certificate,
    validate_screen_arm,
)
from .training_reproducibility import (
    assert_finite_state_dict,
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)


MODEL_ID = "ogb_edge_state_structural_gps9"
MODEL_PARAMETERS = 4_771_073
MODEL_CONFIG = {
    "node_encoder": "OGBAtomEncoder-9-category",
    "edge_encoder": "OGBBondEncoder-3-category",
    "hidden_channels": 192,
    "num_layers": 9,
    "num_heads": 4,
    "dropout": 0.1,
    "rwse_dim": 16,
    "edge_state_channels": 64,
    "pooling": "mean",
    "n_targets": 1,
}
PLATFORM_ID = "scnet-kunshan"
RUN_FORMAT = "molgap-pcqm-ogb-rich-edgegps9-100k-v4"
CHECKPOINT_FORMAT = f"{RUN_FORMAT}-checkpoint-v1"
GEOMETRY_FIELDS = (
    "pos",
    "edge_distance",
    "wedge_angle_cos",
    "wedge_edge_ids",
    "geometry_valid",
)
CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments/pcqm_ogb_rich_edgegps9_100k_v4/training_contract.json"
)


def _model():
    return make_pcqm_gap_encoder(MODEL_ID)


def _new_state():
    import torch

    model = _model().to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=0.05, foreach=False
    )
    scheduler = FrozenEpochScheduler(optimizer)
    ema = ExponentialMovingAverage(model)
    return model, optimizer, scheduler, ema


def _scientific_fields() -> dict:
    fields = _shared_scientific_fields()
    fields["feature_fingerprint"] = canonical_fingerprint(
        {
            "payload_manifest_sha256": MANIFEST_SHA256,
            "used": ["ogb-atom9", "ogb-bond3", "rwse16"],
            "geometry_model_input": False,
        }
    )
    fields["architecture_fingerprint"] = canonical_fingerprint(MODEL_CONFIG)
    return fields


def _validate_contract_file() -> str:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected = {
        **_scientific_fields(),
        "format": "molgap-pcqm-reference-screen-v4",
        "model_id": MODEL_ID,
        "parameter_count": MODEL_PARAMETERS,
        "model_config": MODEL_CONFIG,
        "manifest_sha256": MANIFEST_SHA256,
        "geometry_aggregate_sha256": GEOMETRY_AGGREGATE_SHA256,
        "train_rows": TRAIN_ROWS,
        "development_rows": DEVELOPMENT_ROWS,
        "seed": SEED,
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "epochs": EPOCHS,
        "optimizer_steps_per_epoch": BATCHES_PER_EPOCH,
        "sample_presentations_per_epoch": BATCHES_PER_EPOCH * PHYSICAL_BATCH,
        "total_optimizer_steps": BATCHES_PER_EPOCH * EPOCHS,
        "sample_exposure": SAMPLE_PRESENTATIONS,
        "loss": "normalized-gap-l1",
        "selection": "best-development-ema9999-60epochs",
    }
    mismatches = {
        key: (value, contract.get(key))
        for key, value in expected.items()
        if contract.get(key) != value
    }
    expected_roles = {
        "train": [0, TRAIN_ROWS],
        "development": [TRAIN_ROWS, TRAIN_ROWS + DEVELOPMENT_ROWS],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    if contract.get("role_access") != expected_roles:
        mismatches["role_access"] = (expected_roles, contract.get("role_access"))
    expected_optimizer = {
        "name": "torch-adamw",
        "foreach": False,
        "fused": False,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.05,
        "gradient_clip": 1.0,
    }
    if contract.get("optimizer") != expected_optimizer:
        mismatches["optimizer"] = (expected_optimizer, contract.get("optimizer"))
    expected_schedule = {
        "name": "epoch-warmup-cosine-v1",
        "warmup_epochs": 4,
        "minimum_learning_rate": 1.0e-6,
    }
    if contract.get("schedule") != expected_schedule:
        mismatches["schedule"] = (expected_schedule, contract.get("schedule"))
    if mismatches:
        raise RuntimeError(f"Frozen V4 contract file mismatch: {mismatches}")
    return sha256_file(CONTRACT_PATH)


def _load_v4_roles(dataset_root: Path, manifest_path: Path):
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("Fixed V4 manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = load_roles(dataset_root, manifest)
    for role, count in (("train", TRAIN_ROWS), ("development", DEVELOPMENT_ROWS)):
        if len(roles[role]) != count:
            raise RuntimeError(f"Fixed V4 {role} row count changed")
        for shard in roles[role].datasets:
            for field in GEOMETRY_FIELDS:
                if field in shard._data:
                    raise RuntimeError(f"Geometry leaked into 2D batches: {field}")
    if len(assets.train_paths) != 2 or len(assets.development_paths) != 1:
        raise RuntimeError("Unexpected fixed V4 shard layout")
    return roles


def _training_loader(graphs, epoch: int, start_batch: int = 0):
    import torch
    from torch_geometric.loader import DataLoader

    sampler = DeterministicEpochBatchSampler(len(graphs), epoch)
    batches = list(sampler)[start_batch:]
    return DataLoader(
        graphs,
        batch_sampler=batches,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(SEED + epoch),
    )


def _save_checkpoint(
    path: Path,
    *,
    next_epoch: int,
    next_batch_index: int,
    epoch_train_loss_sum: float,
    epoch_train_count: int,
    model,
    optimizer,
    scheduler,
    ema,
    trace: list[dict],
    best: float,
    best_epoch: int,
    target_stats: dict,
    certificate_id: str,
    source_commit: str,
    source_archive_sha256: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "format": CHECKPOINT_FORMAT,
            "epoch": next_epoch,
            "next_batch_index": next_batch_index,
            "epoch_train_loss_sum": epoch_train_loss_sum,
            "epoch_train_count": epoch_train_count,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "trace": trace,
            "best_development_mae_eV": best,
            "best_epoch": best_epoch,
            "target_stats": target_stats,
            "runtime_certificate_id": certificate_id,
            "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "fixed_manifest_sha256": MANIFEST_SHA256,
            "scientific_fields": _scientific_fields(),
            "rng_state": capture_rng_state(),
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        },
    )


def _runtime_certificate(roles, output: Path, platform_id: str):
    import torch

    contract_sha256 = _validate_contract_file()
    determinism = configure_fp32_determinism(SEED)
    runtime = build_runtime_manifest(determinism)
    target_mean, target_std = _shared_target_stats(
        roles["train"].datasets
    )
    mean = torch.tensor(target_mean, device="cuda")
    std = torch.tensor(target_std, device="cuda")
    batch = next(iter(_training_loader(roles["train"], 0))).to(
        "cuda", non_blocking=True
    )
    if int(batch.num_graphs) != PHYSICAL_BATCH:
        raise RuntimeError("V4 calibration did not use a physical batch of 128")
    fixture_hash = _batch_sha256(batch)
    losses, state_hashes = [], []
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model, optimizer, scheduler, ema = _new_state()
        scheduler.step(0)
        loss = _optimizer_step(
            model, optimizer, ema, batch, mean, std, check_finite=True
        )
        torch.cuda.synchronize()
        losses.append(float(loss.cpu()))
        state_hashes.append(_state_sha256(model))
        del model, optimizer, scheduler, ema
        torch.cuda.empty_cache()
    if losses[0] != losses[1] or state_hashes[0] != state_hashes[1]:
        raise RuntimeError(
            f"V4 deterministic optimizer calibration failed: {losses}, {state_hashes}"
        )

    configure_fp32_determinism(SEED)
    model, optimizer, scheduler, ema = _new_state()
    scheduler.step(0)
    batches = iter(_training_loader(roles["train"], 1))
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        _optimizer_step(
            model,
            optimizer,
            ema,
            next(batches).to("cuda", non_blocking=True),
            mean,
            std,
            check_finite=True,
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(30):
        _optimizer_step(
            model,
            optimizer,
            ema,
            next(batches).to("cuda", non_blocking=True),
            mean,
            std,
            check_finite=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    reserve = 1.0 - peak_reserved / total_memory
    if reserve < 0.15:
        raise RuntimeError(f"V4 BS128 reserve below 15%: {reserve:.4f}")
    graphs_per_second = 30 * PHYSICAL_BATCH / elapsed
    cert = {
        "format": "molgap-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": platform_id,
        "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "tail_batch_policy": "drop_last",
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(determinism),
        "calibration_fixture_sha256": fixture_hash,
        "calibration_output_sha256": state_hashes[0],
        "calibration_checks_passed": True,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
    }
    cert_id = canonical_fingerprint(cert)
    provisional = {
        **_scientific_fields(),
        "platform_id": platform_id,
        "accelerator": cert["accelerator"],
        "runtime_certificate_id": cert_id,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }
    validate_runtime_certificate(cert, provisional)
    estimate_hours = SAMPLE_PRESENTATIONS / graphs_per_second / 3600.0
    result = {
        "accepted": True,
        "runtime_certificate": cert,
        "runtime_certificate_id": cert_id,
        "runtime_manifest": runtime,
        "target_stats": {"mean_eV": target_mean, "sample_std_eV": target_std},
        "calibration": {
            "identical_optimizer_step_loss": losses[0],
            "identical_optimizer_step_state_sha256": state_hashes[0],
            "measured_steps": 30,
            "graphs_per_second": graphs_per_second,
            "estimated_training_hours": estimate_hours,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_memory_bytes": total_memory,
            "reserved_memory_headroom": reserve,
        },
        "source_commit": None,
        "source_archive_sha256": None,
        "training_contract_sha256": contract_sha256,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", cert)
    return result


def run_training(
    *,
    dataset_root: Path,
    manifest_path: Path,
    source_archive: Path,
    source_archive_sha256: str,
    source_commit: str,
    output: Path,
    platform_id: str = PLATFORM_ID,
) -> dict:
    import torch

    determinism = configure_fp32_determinism(SEED)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V4 run requires exactly one visible accelerator")
    validate_screen_arm(physical_batch_per_device=PHYSICAL_BATCH)
    contract_sha256 = _validate_contract_file()
    validate_source_archive(source_archive, source_archive_sha256, source_commit)
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion_manifest.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("complete"):
            return completion
        raise RuntimeError("An incompatible completion manifest already exists")
    roles = _load_v4_roles(dataset_root, manifest_path)
    if sum(parameter.numel() for parameter in _model().parameters()) != MODEL_PARAMETERS:
        raise RuntimeError("OGB-rich EdgeState Structural GPS9 parameter count changed")

    preflight_path = output / "preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (
            preflight.get("accepted") is not True
            or preflight.get("source_commit") != source_commit
            or preflight.get("source_archive_sha256") != source_archive_sha256
        ):
            raise RuntimeError("Existing V4 preflight identity mismatch")
        runtime = build_runtime_manifest(determinism)
        if runtime["runtime_fingerprint"] != preflight["runtime_manifest"]["runtime_fingerprint"]:
            raise RuntimeError("Runtime differs from accepted V4 preflight")
        certificate = preflight["runtime_certificate"]
        certificate_id = preflight["runtime_certificate_id"]
    else:
        preflight = _runtime_certificate(roles, output, platform_id)
        preflight["source_commit"] = source_commit
        preflight["source_archive_sha256"] = source_archive_sha256
        atomic_json(preflight_path, preflight)
        certificate = preflight["runtime_certificate"]
        certificate_id = preflight["runtime_certificate_id"]

    runtime = build_runtime_manifest(determinism)
    if runtime["runtime_fingerprint"] != certificate["runtime_fingerprint"]:
        raise RuntimeError("Training runtime differs from certified runtime")
    contract_fields = {
        **_scientific_fields(),
        "platform_id": platform_id,
        "accelerator": certificate["accelerator"],
        "runtime_certificate_id": certificate_id,
    }
    validate_runtime_certificate(certificate, contract_fields)

    mean_value, std_value = _shared_target_stats(roles["train"].datasets)
    target_stats = {"mean_eV": mean_value, "sample_std_eV": std_value}
    if target_stats != preflight["target_stats"]:
        raise RuntimeError("Target statistics changed after preflight")
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    model, optimizer, scheduler, ema = _new_state()
    checkpoint_path = output / "last_checkpoint.pt"
    trace = []
    start_epoch = 0
    next_batch = 0
    partial_loss_sum = 0.0
    partial_count = 0
    best = math.inf
    best_epoch = -1
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Checkpoint format mismatch")
        if checkpoint.get("scientific_fields") != _scientific_fields():
            raise RuntimeError("Checkpoint scientific contract changed")
        for key, expected in (
            ("source_commit", source_commit),
            ("source_archive_sha256", source_archive_sha256),
            ("runtime_certificate_id", certificate_id),
            ("fixed_manifest_sha256", MANIFEST_SHA256),
        ):
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Checkpoint {key} mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        next_batch = int(checkpoint["next_batch_index"])
        partial_loss_sum = float(checkpoint["epoch_train_loss_sum"])
        partial_count = int(checkpoint["epoch_train_count"])
        if not 0 <= next_batch <= BATCHES_PER_EPOCH:
            raise RuntimeError("Checkpoint batch cursor is invalid")
        if checkpoint.get("target_stats") != target_stats:
            raise RuntimeError("Checkpoint target statistics changed")
        trace = list(checkpoint["trace"])
        best = float(checkpoint["best_development_mae_eV"])
        best_epoch = int(checkpoint["best_epoch"])

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, EPOCHS):
        learning_rate = scheduler.step(epoch)
        model.train()
        train_loss_sum = torch.tensor(partial_loss_sum, device="cuda")
        train_count = partial_count
        steps = 0
        started = time.perf_counter()
        for local_index, batch in enumerate(
            _training_loader(roles["train"], epoch, next_batch)
        ):
            batch_index = local_index + next_batch
            if int(batch.num_graphs) != PHYSICAL_BATCH:
                raise RuntimeError("V4 optimizer received a non-128 batch")
            batch = batch.to("cuda", non_blocking=True)
            global_step = epoch * BATCHES_PER_EPOCH + batch_index + 1
            loss = _optimizer_step(
                model,
                optimizer,
                ema,
                batch,
                mean,
                std,
                check_finite=global_step % FINITE_CHECK_EVERY_STEPS == 0,
            )
            train_loss_sum.add_(loss * int(batch.num_graphs))
            train_count += int(batch.num_graphs)
            steps += 1
            next_batch = batch_index + 1
            if next_batch % 200 == 0:
                _save_checkpoint(
                    checkpoint_path,
                    next_epoch=epoch,
                    next_batch_index=next_batch,
                    epoch_train_loss_sum=float(train_loss_sum.cpu()),
                    epoch_train_count=train_count,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    trace=trace,
                    best=best,
                    best_epoch=best_epoch,
                    target_stats=target_stats,
                    certificate_id=certificate_id,
                    source_commit=source_commit,
                    source_archive_sha256=source_archive_sha256,
                )
                print(
                    f"checkpoint ep{epoch:02d} next_batch={next_batch}/{BATCHES_PER_EPOCH}",
                    flush=True,
                )
        if next_batch != BATCHES_PER_EPOCH or train_count != BATCHES_PER_EPOCH * PHYSICAL_BATCH:
            raise RuntimeError("V4 epoch exposure changed")
        _save_checkpoint(
            checkpoint_path,
            next_epoch=epoch,
            next_batch_index=BATCHES_PER_EPOCH,
            epoch_train_loss_sum=float(train_loss_sum.cpu()),
            epoch_train_count=train_count,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            trace=trace,
            best=best,
            best_epoch=best_epoch,
            target_stats=target_stats,
            certificate_id=certificate_id,
            source_commit=source_commit,
            source_archive_sha256=source_archive_sha256,
        )
        development = _evaluate(model, ema, roles["development"], mean, std)
        improved = development["mae_eV"] < best
        if improved:
            best = development["mae_eV"]
            best_epoch = epoch
            best_payload = {
                "format": RUN_FORMAT,
                "model_id": MODEL_ID,
                "model_config": MODEL_CONFIG,
                "model": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                "target_stats": target_stats,
                "epoch": epoch,
                "development_mae_eV": best,
                "parameter_count": MODEL_PARAMETERS,
                "architecture_fingerprint": _scientific_fields()["architecture_fingerprint"],
                "runtime_certificate_id": certificate_id,
                "manifest_sha256": MANIFEST_SHA256,
                "source_archive_sha256": source_archive_sha256,
                "source_commit": source_commit,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
                "test_challenge_role_read": False,
            }
            assert_finite_state_dict(best_payload["model"], label="best GPS9 model")
            atomic_torch_save(output / "best_model.pt", best_payload)
            atomic_torch_save(
                output / "development_predictions.pt",
                {
                    "prediction_eV": development["prediction_eV"],
                    "target_eV": development["target_eV"],
                    "source_idx": development["source_idx"],
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                    "test_challenge_role_read": False,
                },
            )
        row = {
            "epoch": epoch,
            "train_mae_eV": float(train_loss_sum.cpu()) * std_value / train_count,
            "development_mae_eV": development["mae_eV"],
            "best_development_mae_eV": best,
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "optimizer_steps": BATCHES_PER_EPOCH,
            "sample_presentations": train_count,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(row)
        atomic_json(output / "trace.json", {"format": RUN_FORMAT, "rows": trace})
        _save_checkpoint(
            checkpoint_path,
            next_epoch=epoch + 1,
            next_batch_index=0,
            epoch_train_loss_sum=0.0,
            epoch_train_count=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            trace=trace,
            best=best,
            best_epoch=best_epoch,
            target_stats=target_stats,
            certificate_id=certificate_id,
            source_commit=source_commit,
            source_archive_sha256=source_archive_sha256,
        )
        next_batch = 0
        partial_loss_sum = 0.0
        partial_count = 0
        print(
            f"{MODEL_ID} ep{epoch:02d} train={row['train_mae_eV']:.6f} "
            f"dev={best if improved else row['development_mae_eV']:.6f}eV "
            f"best={best:.6f}@{best_epoch} lr={learning_rate:.3e} "
            f"{row['elapsed_seconds']:.1f}s" + (" *" if improved else ""),
            flush=True,
        )

    result_sha = sha256_file(output / "best_model.pt")
    reference = {
        **_scientific_fields(),
        "run_id": "ogb-rich-edgegps9-100k-v4-seed42",
        "model_id": MODEL_ID,
        "source_archive_sha256": source_archive_sha256,
        "result_artifact_sha256": result_sha,
        "platform_id": platform_id,
        "accelerator": certificate["accelerator"],
        "runtime_certificate_id": certificate_id,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "frozen_reference": True,
        "stochasticity_floor_eV": 0.003,
        "minimum_material_gain_eV": 0.003,
        "best_development_mae_eV": best,
        "best_epoch": best_epoch,
    }
    for key in ("benchmark_id", "data_role_fingerprint", "row_order_fingerprint", "feature_fingerprint", "target_fingerprint", "seed", "precision", "optimizer_fingerprint", "schedule_fingerprint", "loss_fingerprint", "target_transform_fingerprint", "selection_fingerprint", "role_access_fingerprint", "sample_exposure", "tail_batch_policy"):
        if key not in reference:
            raise RuntimeError(f"V4 reference is missing {key}")
    validate_runtime_certificate(certificate, reference)
    atomic_json(output / "frozen_reference.json", reference)
    completion = {
        "format": RUN_FORMAT,
        "complete": True,
        "epochs": EPOCHS,
        "optimizer_steps": BATCHES_PER_EPOCH * EPOCHS,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "parameter_count": MODEL_PARAMETERS,
        "best_development_mae_eV": best,
        "best_epoch": best_epoch,
        "manifest_sha256": MANIFEST_SHA256,
        "geometry_aggregate_sha256": GEOMETRY_AGGREGATE_SHA256,
        "runtime_certificate_id": certificate_id,
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
        "training_contract_sha256": contract_sha256,
        "best_model_sha256": result_sha,
        "development_predictions_sha256": sha256_file(output / "development_predictions.pt"),
        "frozen_reference_sha256": sha256_file(output / "frozen_reference.json"),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    completion["artifact_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "completion_manifest.json"
    }
    atomic_json(completion_path, completion)
    return completion


def validate_completion(output: Path) -> dict:
    """Mechanically accept the V4 artifact without running model inference."""
    import torch

    completion_path = output / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    contract_sha256 = _validate_contract_file()
    if completion.get("format") != RUN_FORMAT or completion.get("complete") is not True:
        raise RuntimeError("V4 run is not complete")
    expected = {
        "epochs": EPOCHS,
        "optimizer_steps": BATCHES_PER_EPOCH * EPOCHS,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "parameter_count": MODEL_PARAMETERS,
        "manifest_sha256": MANIFEST_SHA256,
        "geometry_aggregate_sha256": GEOMETRY_AGGREGATE_SHA256,
        "training_contract_sha256": contract_sha256,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(f"Completion field mismatch: {key}")
    for filename, expected_sha in completion["artifact_sha256"].items():
        path = output / filename
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Artifact hash mismatch: {filename}")
    trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))["rows"]
    if len(trace) != EPOCHS or [row["epoch"] for row in trace] != list(range(EPOCHS)):
        raise RuntimeError("Trace epoch sequence mismatch")
    payload = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    if payload["parameter_count"] != MODEL_PARAMETERS or payload["model_id"] != MODEL_ID:
        raise RuntimeError("Best model identity mismatch")
    assert_finite_state_dict(payload["model"], label="accepted best model")
    pred = torch.load(output / "development_predictions.pt", map_location="cpu", weights_only=False)
    if not torch.equal(pred["source_idx"].long(), torch.arange(100_000, 150_000)):
        raise RuntimeError("Development prediction source order mismatch")
    if pred["prediction_eV"].numel() != DEVELOPMENT_ROWS:
        raise RuntimeError("Development prediction count mismatch")
    calculated = float((pred["prediction_eV"] - pred["target_eV"]).abs().mean())
    if not math.isclose(calculated, completion["best_development_mae_eV"], rel_tol=0, abs_tol=1e-7):
        raise RuntimeError("Development MAE does not match saved predictions")
    reference = json.loads((output / "frozen_reference.json").read_text(encoding="utf-8"))
    if reference.get("result_artifact_sha256") != sha256_file(output / "best_model.pt"):
        raise RuntimeError("Frozen reference artifact hash mismatch")
    certificate = json.loads(
        (output / "runtime_certificate.json").read_text(encoding="utf-8")
    )
    if reference.get("runtime_certificate_id") != canonical_fingerprint(certificate):
        raise RuntimeError("Frozen reference runtime certificate mismatch")
    validate_runtime_certificate(certificate, reference)
    if not torch.isfinite(pred["prediction_eV"]).all() or not torch.isfinite(pred["target_eV"]).all():
        raise RuntimeError("Development predictions or targets are non-finite")
    return {
        "accepted": True,
        "model_id": MODEL_ID,
        "parameter_count": MODEL_PARAMETERS,
        "epochs": EPOCHS,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "best_epoch": completion["best_epoch"],
        "development_mae_eV": calculated,
        "best_model_sha256": sha256_file(output / "best_model.pt"),
        "manifest_sha256": MANIFEST_SHA256,
        "sealed_roles_read": False,
    }
