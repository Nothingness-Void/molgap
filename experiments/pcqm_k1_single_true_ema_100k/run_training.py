"""Executable training runner for K1 Single-Forward True-EMA 100K Screen.

Fully complies with MolGap RML & Research Protocol:
1. Reuses unified accelerated modules (BoundForeachEMA, ReusedWeightView, recalibrated_batch_norm).
2. Fixed physical BS=128, FP32, seed=42, CosineAnnealingLR across 40 epochs.
3. Evaluates Live & EMA MAE on fixed 50K development set after each epoch.
4. Performs post-selection Clean-BN calibration on 16,384 training molecules.
5. Saves artifacts, trace.jsonl, and terminal acceptance files.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from molgap.qm9_neural_atom import make_encoder
from molgap.gptrans_execution_hotpath import BoundForeachEMA, ReusedWeightView
from molgap.k1_screen_training import (
    _PackedGraphDatasetFactory, _forward, _optimizer_step,
    build_initial_state, epoch_order, SEED
)
from molgap.k1_bn_calibration import recalibrated_batch_norm
from molgap.training_reproducibility import atomic_torch_save, atomic_json, sha256_file
from molgap.v4_runtime import state_dict_sha256

MEAN_EV = 5.3383002281188965
STD_EV = 1.275090217590332
BATCH_SIZE = 128
LR = 4e-4
WEIGHT_DECAY = 1e-5
EMA_DECAY = 0.9999
TOTAL_EPOCHS = 40
TRAIN_ROWS = 100_000
DEV_ROWS = 50_000
REF_MEAN2_MAE_EV = 0.140257285566

EXP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = EXP_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("MolGap K1 Single-Forward True-EMA 100K Screen")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Epochs: {TOTAL_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}, EMA Decay: {EMA_DECAY}")
    print("=" * 70)

    # 1. Load Datasets
    data_dir = Path("d:/文档/molgap/data/pcqm_fixed_100k_v1/train")
    t0_data = time.perf_counter()
    shard0 = _PackedGraphDatasetFactory.load(data_dir / "train_shard_0000.pt")
    shard1 = _PackedGraphDatasetFactory.load(data_dir / "train_shard_0001.pt")
    dev_shard = _PackedGraphDatasetFactory.load(data_dir / "train_shard_0002.pt")
    print(f"Loaded datasets in {time.perf_counter() - t0_data:.2f}s: {len(shard0) + len(shard1)} train, {len(dev_shard)} dev.")

    # 2. Model Initialization
    initial_state = build_initial_state(SEED)
    model = make_encoder("neural_atom_k1").to(device)
    model.load_state_dict(initial_state, strict=True)
    initial_sha = state_dict_sha256(model.state_dict())
    print(f"Initialized K1 encoder from seed {SEED} (SHA256: {initial_sha[:16]}...).")

    # 3. Optimizer, Scheduler, EMA
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = TRAIN_ROWS // BATCH_SIZE  # 781
    total_steps = steps_per_epoch * TOTAL_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    ema = BoundForeachEMA(model, decay=EMA_DECAY)
    weight_view = ReusedWeightView()

    dev_loader = DataLoader(dev_shard, batch_size=BATCH_SIZE, shuffle=False)

    trace_records = []
    trace_path = EXP_DIR / "trace.jsonl"
    best_ema_mae = float("inf")
    best_epoch = -1
    total_training_wall = 0.0

    # 4. Training Loop
    training_start = time.perf_counter()

    for epoch in range(TOTAL_EPOCHS):
        order = epoch_order(epoch, SEED)
        all_train = [shard0[i] if i < 50000 else shard1[i - 50000] for i in order]
        train_loader = DataLoader(all_train, batch_size=BATCH_SIZE, shuffle=False)

        model.train()
        epoch_start = time.perf_counter()
        epoch_losses = []

        for step_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, absolute, rows = _optimizer_step(
                model, optimizer, batch, MEAN_EV, STD_EV, mode="reference",
                clip_norm=1.0, check_finite=True
            )
            scheduler.step()
            ema.update(model)
            epoch_losses.append(loss.item())

        torch.cuda.synchronize()
        epoch_dur = time.perf_counter() - epoch_start
        total_training_wall += epoch_dur
        mean_train_loss = float(np.mean(epoch_losses))

        # Evaluate on 50K development set
        model.eval()

        def evaluate_current():
            errors = []
            with torch.no_grad():
                for b in dev_loader:
                    b = b.to(device)
                    pred = _forward(model, b) * STD_EV + MEAN_EV
                    errors.append((pred - b.y.view(-1)).abs().cpu())
            return float(torch.cat(errors).mean().item())

        # Live evaluation
        t_eval0 = time.perf_counter()
        live_mae = evaluate_current()

        # EMA evaluation
        with weight_view.use(model, ema, "ema"):
            ema_mae = evaluate_current()
        eval_dur = time.perf_counter() - t_eval0

        # Check best EMA
        is_best = ema_mae < best_ema_mae
        if is_best:
            best_ema_mae = ema_mae
            best_epoch = epoch
            # Save best EMA checkpoint
            with weight_view.use(model, ema, "ema"):
                atomic_torch_save(ARTIFACTS_DIR / "best_ema_checkpoint.pt", {
                    "epoch": epoch,
                    "ema_mae_eV": ema_mae,
                    "live_mae_eV": live_mae,
                    "state_dict": model.state_dict(),
                    "step": (epoch + 1) * steps_per_epoch
                })

        record = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "live_dev_mae_eV": live_mae,
            "ema_dev_mae_eV": ema_mae,
            "best_ema_mae_eV": best_ema_mae,
            "is_best": is_best,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_train_seconds": epoch_dur,
            "eval_seconds": eval_dur,
            "cumulative_train_seconds": total_training_wall
        }
        trace_records.append(record)
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS:2d} ({epoch_dur:.1f}s) | "
              f"Loss: {mean_train_loss:.4f} | "
              f"Live MAE: {live_mae:.6f} eV | "
              f"EMA MAE: {ema_mae:.6f} eV"
              f"{' [*BEST*]' if is_best else ''}")
        sys.stdout.flush()

    total_wall_seconds = time.perf_counter() - training_start
    print("\n" + "=" * 70)
    print(f"Training Complete! Total wall-clock time: {total_wall_seconds/60:.2f} minutes.")
    print(f"Best EMA epoch: {best_epoch} with MAE: {best_ema_mae:.6f} eV.")
    print("=" * 70)

    # 5. Clean-BN Recalibration on the Best EMA Checkpoint
    print("\nPerforming Clean-BN calibration on 16,384 training molecules...")
    best_ckpt = torch.load(ARTIFACTS_DIR / "best_ema_checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)

    calib_graphs = [shard0[i] for i in range(16384)]
    calib_loader = DataLoader(calib_graphs, batch_size=BATCH_SIZE, shuffle=False)
    calib_indices = list(range(16384))

    optimizer.zero_grad(set_to_none=True)
    orig_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = False

    t_calib0 = time.perf_counter()
    with recalibrated_batch_norm(
        model, calib_loader, source_idx=calib_indices,
        source_bounds=(0, 100000), device=str(device),
        deadline=time.perf_counter() + 180
    ) as calib_report:
        calib_dur = time.perf_counter() - t_calib0
        print(f"Clean-BN recalibration finished in {calib_dur:.2f}s across {calib_report['bn_modules']} modules.")

    for name, p in model.named_parameters():
        p.requires_grad = orig_requires_grad[name]

    # Evaluate Clean-BN model and accumulate full 50,000 predictions
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for b in dev_loader:
            b = b.to(device)
            pred = (_forward(model, b) * STD_EV + MEAN_EV).cpu()
            all_preds.append(pred)
            all_targets.append(b.y.view(-1).cpu())

    pred_tensor = torch.cat(all_preds)
    target_tensor = torch.cat(all_targets)
    clean_bn_mae = float((pred_tensor - target_tensor).abs().mean().item())

    # Save Clean-BN checkpoint and predictions
    atomic_torch_save(ARTIFACTS_DIR / "selected_clean_bn_model.pt", {
        "epoch": best_epoch,
        "clean_bn_mae_eV": clean_bn_mae,
        "raw_ema_mae_eV": best_ema_mae,
        "state_dict": model.state_dict(),
        "calib_report": calib_report
    })
    atomic_torch_save(ARTIFACTS_DIR / "development_predictions.pt", {
        "prediction_eV": pred_tensor,
        "target_eV": target_tensor,
        "mae_eV": clean_bn_mae,
        "epoch": best_epoch
    })

    # 6. Scientific Analysis vs mean2 reference
    gain_eV = REF_MEAN2_MAE_EV - clean_bn_mae
    gain_meV = gain_eV * 1000.0

    print("\n" + "=" * 70)
    print("FINAL ACCEPTANCE EVALUATION")
    print(f"Reference mean2 MAE : {REF_MEAN2_MAE_EV:.6f} eV")
    print(f"Best Raw EMA MAE    : {best_ema_mae:.6f} eV (Epoch {best_epoch})")
    print(f"Final Clean-BN MAE  : {clean_bn_mae:.6f} eV")
    print(f"Clean-BN Recovery   : {(best_ema_mae - clean_bn_mae)*1000:+.2f} meV")
    print(f"Net Gain vs mean2   : {gain_meV:+.2f} meV ({gain_eV:+.6f} eV)")
    print("=" * 70)

    # 7. Write Structured Result & Attribution
    result_data = {
        "status": "COMPLETE",
        "question": "Does single-forward K1 with True EMA and Clean-BN match or exceed mean2 at lower cost?",
        "best_epoch": best_epoch,
        "ref_mean2_mae_eV": REF_MEAN2_MAE_EV,
        "raw_ema_mae_eV": best_ema_mae,
        "clean_bn_mae_eV": clean_bn_mae,
        "gain_vs_mean2_eV": gain_eV,
        "gain_vs_mean2_meV": gain_meV,
        "total_training_wall_seconds": total_training_wall,
        "total_wall_seconds": total_wall_seconds,
        "device": torch.cuda.get_device_name(0),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "artifacts": {
            "best_ema_checkpoint": str(ARTIFACTS_DIR / "best_ema_checkpoint.pt"),
            "selected_clean_bn_model": str(ARTIFACTS_DIR / "selected_clean_bn_model.pt"),
            "development_predictions": str(ARTIFACTS_DIR / "development_predictions.pt"),
            "trace": str(trace_path)
        }
    }
    atomic_json(EXP_DIR / "result.json", result_data)

    decision_md = f"""# Decision: K1 Single-Forward True-EMA 100K Screen

- **Date**: {time.strftime('%Y-%m-%d')}
- **Device**: {torch.cuda.get_device_name(0)}
- **Training Wall Time**: {total_training_wall/60:.2f} minutes ({total_training_wall:.1f} seconds)
- **Selected Epoch**: {best_epoch} / {TOTAL_EPOCHS}

| Metric | MAE (eV) | Gain vs mean2 (meV) |
| :--- | :---: | :---: |
| **mean2 Reference** | {REF_MEAN2_MAE_EV:.6f} | baseline (0.0) |
| **Best Raw EMA** | {best_ema_mae:.6f} | {(REF_MEAN2_MAE_EV - best_ema_mae)*1000:+.2f} |
| **Clean-BN EMA** | {clean_bn_mae:.6f} | **{gain_meV:+.2f}** |

## Outcome
- Net Gain: **{gain_meV:+.2f} meV** ({gain_eV:+.6f} eV).
- Clean-BN Effect: recovered **{(best_ema_mae - clean_bn_mae)*1000:+.2f} meV** over raw EMA.
"""
    (EXP_DIR / "decision.md").write_text(decision_md, encoding="utf-8")
    print(f"Saved results and decision to {EXP_DIR}.")


if __name__ == "__main__":
    main()
