"""Matched 500K reference evidence with bounded, resumable Kaggle stages."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from .training_reproducibility import (
    atomic_json, atomic_torch_save, build_runtime_manifest, capture_rng_state,
    configure_fp32_determinism, restore_rng_state, sha256_file,
)
from .screen_policy import canonical_fingerprint, validate_runtime_certificate
from .pcqm_k1_scale_runner import find_cache, load_roles, _targets
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .pcqm_gptrans_v4 import _state_sha256, _batch_sha256, _forward

PARAMETERS = {"full_gps": 4771073, "neural_atom_k1": 3658817, "gptrans": 5246817}
EPOCHS = 60
BS = 128
STEPS = 500000 // BS


def schedule(epoch):
    return 1e-6 + (4e-4 - 1e-6) * (1 + math.cos(math.pi * epoch / (EPOCHS - 1))) / 2


def scientific_contract():
    return {
        "benchmark_id": "pcqm-fixed500k-dev50k-matched60-v4",
        "data_role_fingerprint": FIXED_500K_MANIFEST_SHA256,
        "row_order_fingerprint": "global-randperm-seed42-plus-epoch-drop-last32",
        "feature_fingerprint": "ogb-node9-edge3-rwse16-no-geometry-input",
        "target_fingerprint": "pcqm4mv2-gap-eV",
        "seed": 42, "precision": "fp32",
        "optimizer_fingerprint": "adamw-unfused-foreachFalse-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine60-epoch0-4e-4-epoch59-1e-6",
        "loss_fingerprint": "normalized-gap-l1",
        "target_transform_fingerprint": "all500k-train-only-mean-unbiased-std",
        "selection_fingerprint": "best-development-raw-model-60epochs",
        "role_access_fingerprint": "official-train-derived-train-and-internal-development-only",
        "sample_exposure": EPOCHS * STEPS * BS,
        "tail_batch_policy": "drop_last", "physical_batch_per_device": BS,
        "device_count": 1, "gradient_accumulation_steps": 1,
        "epochs": EPOCHS, "steps_per_epoch": STEPS,
    }


def make_model(arm):
    if arm == "gptrans":
        from .pcqm_gptrans_v4 import _make_model
        model = _make_model()
    else:
        from .qm9_neural_atom import make_encoder
        model = make_encoder(arm)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm]:
        raise RuntimeError("Frozen architecture parameter count changed")
    return model


def optimizer_for(model):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=schedule(0), weight_decay=1e-5,
                            foreach=False, fused=False)


def loader(graphs, epoch=None):
    import torch
    from torch_geometric.loader import DataLoader
    sampler = None if epoch is None else torch.randperm(
        len(graphs), generator=torch.Generator().manual_seed(42 + epoch)
    )[:STEPS * BS].tolist()
    return DataLoader(graphs, batch_size=BS, sampler=sampler, shuffle=False,
                      num_workers=2, pin_memory=True, persistent_workers=True,
                      generator=torch.Generator().manual_seed(9000 + (epoch or 0)))


def step(model, optimizer, batch, mean, std):
    import torch
    if batch.num_graphs != BS:
        raise RuntimeError("Non-128 optimizer batch")
    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, batch)
    loss = torch.nn.functional.l1_loss(prediction, (batch.y.view(-1) - mean) / std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def evaluate(model, graphs, mean, std):
    import torch
    model.eval()
    rows = {"prediction": [], "target": [], "source_idx": []}
    with torch.no_grad():
        for batch in loader(graphs):
            batch = batch.to("cuda", non_blocking=True)
            rows["prediction"].append((_forward(model, batch) * std + mean).cpu())
            rows["target"].append(batch.y.view(-1).cpu())
            rows["source_idx"].append(batch.source_idx.view(-1).cpu().long())
    result = {key: torch.cat(values) for key, values in rows.items()}
    if not torch.equal(result["source_idx"], torch.arange(500000, 550000)):
        raise RuntimeError("Development row order changed")
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("Non-finite development output")
    result["mae_eV"] = float((result["prediction"] - result["target"]).abs().mean())
    return result


def run(arm, output, source_sha, stage_epochs=4, resume=None):
    import shutil
    import torch
    output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One visible accelerator per worker required")
    settings = configure_fp32_determinism(42)
    runtime = build_runtime_manifest(settings)
    root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    roles = load_roles(root, manifest)
    target = _targets(roles["train"]).double()
    mean, std = float(target.mean()), float(target.std(unbiased=True).clamp_min(1e-6))
    contract = scientific_contract()
    contract["target_transform_fingerprint"] = canonical_fingerprint({"mean": mean, "std": std})
    atomic_json(output / "runtime.json", runtime)
    atomic_json(output / "scientific_contract.json", contract)
    atomic_json(output / "data_manifest.json", manifest)
    batch = next(iter(loader(roles["train"], 0))).to("cuda")
    fixture_sha = _batch_sha256(batch)
    calibrations = []
    initial_sha = None
    torch.cuda.reset_peak_memory_stats()
    for repetition in range(2):
        configure_fp32_determinism(42)
        model = make_model(arm).to("cuda").train()
        initial_sha = _state_sha256(model)
        optimizer = optimizer_for(model)
        losses = [float(step(model, optimizer, batch, mean, std)) for _ in range(3)]
        calibrations.append({"initial": initial_sha, "losses": losses, "state": _state_sha256(model)})
        del model, optimizer
    atomic_json(output / "calibration.json", {"repeats": calibrations, "fixture": fixture_sha})
    if calibrations[0] != calibrations[1]:
        raise RuntimeError("Deterministic optimizer-step calibration failed")
    peak = torch.cuda.max_memory_reserved()
    if peak > 0.85 * torch.cuda.get_device_properties(0).total_memory:
        raise RuntimeError("BS128 optimizer-inclusive memory reserve below 15%")
    certificate = {
        "format": "molgap-runtime-certificate-v1", "status": "accepted",
        "platform_id": "kaggle1", "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32", "tf32_enabled": False, "deterministic_algorithms": True,
        "physical_batch_per_device": BS, "tail_batch_policy": "drop_last",
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(settings),
        "calibration_fixture_sha256": fixture_sha,
        "calibration_output_sha256": calibrations[0]["state"],
        "calibration_checks_passed": True, "runtime_fingerprint": runtime["runtime_fingerprint"],
    }
    certificate_id = canonical_fingerprint(certificate)
    validate_runtime_certificate(certificate, {"platform_id": "kaggle1",
        "accelerator": certificate["accelerator"], "runtime_certificate_id": certificate_id})
    atomic_json(output / "runtime_certificate.json", certificate)
    del batch
    torch.cuda.empty_cache()
    configure_fp32_determinism(42)
    model = make_model(arm).to("cuda")
    optimizer = optimizer_for(model)
    start_epoch, trace, best, best_epoch = 0, [], float("inf"), -1
    if resume is not None:
        resume = Path(resume)
        checksums = json.loads((resume / "stage_manifest.json").read_text())["artifacts"]
        for name in ("last_checkpoint.pt", "best_model.pt", "best_predictions.pt"):
            if sha256_file(resume / name) != checksums[name]:
                raise RuntimeError(f"Resume hash mismatch: {name}")
        state = torch.load(resume / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        if state["contract"] != contract or state["arm"] != arm or state["source_sha256"] != source_sha:
            raise RuntimeError("Resume scientific/source identity mismatch")
        if state["runtime_software"] != runtime["installed_distributions_sha256"] or state["accelerator"] != certificate["accelerator"]:
            raise RuntimeError("Resume runtime changed; requalification required")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state["rng"])
        start_epoch, trace, best, best_epoch = state["next_epoch"], state["trace"], state["best"], state["best_epoch"]
        for name in ("best_model.pt", "best_predictions.pt", "initial_state.pt"):
            shutil.copy2(resume / name, output / name)
    else:
        atomic_torch_save(output / "initial_state.pt", {"model": model.state_dict(), "state_sha256": initial_sha})
    print(f"PREFLIGHT PASS {arm} parameters={PARAMETERS[arm]} resume_epoch={start_epoch} peak={peak}", flush=True)
    stage_start = time.monotonic()
    for epoch in range(start_epoch, min(EPOCHS, start_epoch + stage_epochs)):
        started = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = schedule(epoch)
        model.train()
        loss_sum = torch.zeros((), device="cuda")
        count = 0
        for batch_index, batch in enumerate(loader(roles["train"], epoch)):
            loss_sum += step(model, optimizer, batch.to("cuda", non_blocking=True), mean, std)
            count += BS
            if (batch_index + 1) % 500 == 0:
                print(f"{arm} ep={epoch} batch={batch_index+1}/{STEPS}", flush=True)
        if count != STEPS * BS:
            raise RuntimeError("Sample exposure mismatch")
        evaluation = evaluate(model, roles["validation"], mean, std)
        mae = evaluation["mae_eV"]
        if mae < best:
            best, best_epoch = mae, epoch
            atomic_torch_save(output / "best_model.pt", {"arm": arm, "model": model.state_dict(),
                "mean": mean, "std": std, "contract": contract, "epoch": epoch, "source_sha256": source_sha})
            atomic_torch_save(output / "best_predictions.pt", evaluation)
        atomic_torch_save(output / f"predictions_epoch_{epoch:02d}.pt", evaluation)
        trace.append({"epoch": epoch, "train_mae_eV": float(loss_sum) / STEPS * std,
            "development_mae_eV": mae, "lr": schedule(epoch), "seconds": time.monotonic()-started,
            "global_step": (epoch+1)*STEPS, "sample_presentations": (epoch+1)*STEPS*BS,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        atomic_json(output / "trace.json", {"epochs": trace})
        atomic_torch_save(output / "last_checkpoint.pt", {"arm": arm, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": capture_rng_state(), "next_epoch": epoch+1,
            "trace": trace, "best": best, "best_epoch": best_epoch, "contract": contract,
            "source_sha256": source_sha, "runtime_software": runtime["installed_distributions_sha256"],
            "accelerator": certificate["accelerator"]})
        atomic_json(output / "progress.json", {"status": "RUNNING", "next_epoch": epoch+1, "best": best})
        print(f"{arm} ep{epoch:02d} dev={mae:.8f} best={best:.8f}@{best_epoch} {trace[-1]['seconds']:.1f}s", flush=True)
        # End at an epoch boundary well before Kaggle's session limit.
        if time.monotonic() - stage_start + 1.5 * trace[-1]["seconds"] > 3 * 3600:
            break
    status = "COMPLETE" if trace[-1]["epoch"] + 1 == EPOCHS else "STAGE_COMPLETE"
    artifacts = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "stage_manifest.json"}
    atomic_json(output / "stage_manifest.json", {"status": status, "arm": arm,
        "next_epoch": trace[-1]["epoch"]+1, "best_development_mae_eV": best, "best_epoch": best_epoch,
        "parameters": PARAMETERS[arm], "contract": contract, "source_sha256": source_sha,
        "runtime_certificate_id": certificate_id, "artifacts": artifacts,
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=PARAMETERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage-epochs", type=int, default=4)
    args = parser.parse_args()
    try:
        run(args.arm, args.output, args.source_sha, args.stage_epochs, args.resume)
    except Exception as error:
        atomic_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
