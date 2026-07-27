"""
Phase 4: SchNet hyperparameter search with Optuna.

Strategy:
  1. Reuse cached 3D graphs from gnn_schnet_3d.py
  2. Each trial trains with early stopping (short patience for speed)
  3. Best trial is retrained with full patience + more epochs
  4. Results saved to results/phase4/schnet_optuna/

Usage:
  python scripts/phase4/schnet_optuna.py [--n-trials 30] [--fast-epochs 100] [--full-epochs 500]
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

warnings.filterwarnings("ignore")

from molgap.utils import (
    MODELS_DIR,
    RAW_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    create_split_indices,
    ensure_dirs,
    regression_metrics,
    save_json,
)

from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase4" / "schnet_optuna"
PROCESSED_GRAPHS = RESULTS_DIR / "phase4" / "pyg_3d_graphs_etkdg.pt"
PROCESSED_GRAPHS_LEGACY = RESULTS_DIR / "phase4" / "pyg_3d_graphs.pt"
RAW_CSV = RAW_DIR / "phase3_chonsfcl_mw200_1000_30k.csv"
SEED = 42


# ── Train / evaluate ──────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            charges = getattr(batch, 'charges', None)
            out = model(batch.z, batch.pos, batch.batch, charges=charges)
            loss = F.l1_loss(out, batch.y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda"):
            charges = getattr(batch, 'charges', None)
            out = model(batch.z, batch.pos, batch.batch, charges=charges)
        preds.append(out.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    return np.vstack(preds), np.vstack(trues)


def run_training(params, train_loader, valid_loader, y_mean, y_std,
                 device, max_epochs, patience, verbose=False, use_charges=False):
    """Train a SchNet model with given params, return best val MAE and metrics."""

    model = SchNetWrapper(
        hidden_channels=params["hidden_channels"],
        num_filters=params["num_filters"],
        num_interactions=params["num_interactions"],
        num_gaussians=params["num_gaussians"],
        cutoff=params["cutoff"],
        dropout=params["dropout"],
        use_charges=use_charges,
    ).to(device)

    lr = params["lr"]
    wd = params["weight_decay"]
    scheduler_type = params["scheduler"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                      patience=max(5, patience // 3), min_lr=1e-6)

    scaler = torch.amp.GradScaler("cuda")

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    log_rows = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_pred, val_true = evaluate(model, valid_loader, device)

        val_pred_real = val_pred * y_std + y_mean
        val_true_real = val_true * y_std + y_mean
        val_mae = float(np.mean(np.abs(val_pred_real - val_true_real)))

        if scheduler_type == "cosine":
            scheduler.step(epoch)
        else:
            scheduler.step(val_mae)

        elapsed = time.time() - t0

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_mae": val_mae, "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_mae:.4f} | "
                  f"best={best_val_mae:.4f}@{best_epoch} | lr={optimizer.param_groups[0]['lr']:.1e} | {elapsed:.1f}s")

        if epoch - best_epoch >= patience:
            if verbose:
                print(f"    Early stop at epoch {epoch} (best={best_epoch})")
            break

    return best_val_mae, best_epoch, best_state, log_rows


# ── Data loading ──────────────────────────────────────────────

def load_data(batch_size_default=64):
    """Load cached graphs and split into train/valid/test."""
    from torch_geometric.loader import DataLoader

    graph_path = PROCESSED_GRAPHS if PROCESSED_GRAPHS.exists() else PROCESSED_GRAPHS_LEGACY
    if graph_path.exists():
        print(f"  Loading cached graphs from {graph_path}...")
        data_list = torch.load(graph_path, weights_only=False)
        print(f"  Loaded {len(data_list)} graphs", flush=True)
    else:
        raise FileNotFoundError(
            f"No graph cache found. Run gnn_schnet_3d.py first to generate 3D graphs."
        )

    train_idx, valid_idx, test_idx = create_split_indices(
        len(data_list), random_state=SEED)

    train_data = [data_list[i] for i in train_idx]
    valid_data = [data_list[i] for i in valid_idx]
    test_data = [data_list[i] for i in test_idx]

    train_y = np.stack([d.y.squeeze(0).numpy() for d in train_data])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    for d in train_data + valid_data + test_data:
        d.y = (d.y - torch.tensor(y_mean)) / torch.tensor(y_std)

    has_charges = hasattr(data_list[0], 'charges')
    return train_data, valid_data, test_data, y_mean, y_std, has_charges


# ── Optuna objective ──────────────────────────────────────────

def create_objective(train_data, valid_data, y_mean, y_std, device, fast_epochs,
                     use_charges=False):
    import optuna
    from torch_geometric.loader import DataLoader

    def objective(trial: optuna.Trial):
        params = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [128, 192, 256]),
            "num_filters": trial.suggest_categorical("num_filters", [128, 192, 256]),
            "num_interactions": trial.suggest_int("num_interactions", 3, 6),
            "num_gaussians": trial.suggest_categorical("num_gaussians", [25, 50, 100]),
            "cutoff": trial.suggest_float("cutoff", 6.0, 12.0, step=1.0),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "scheduler": trial.suggest_categorical("scheduler", ["plateau", "cosine"]),
        }

        train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=params["batch_size"])

        try:
            best_mae, best_ep, _, _ = run_training(
                params, train_loader, valid_loader, y_mean, y_std,
                device, max_epochs=fast_epochs, patience=15,
                use_charges=use_charges,
            )
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}", flush=True)
            torch.cuda.empty_cache()
            return float("inf")

        n_params = sum(
            p.numel() for p in SchNetWrapper(
                params["hidden_channels"], params["num_filters"],
                params["num_interactions"], params["num_gaussians"],
                params["cutoff"], params["dropout"],
            ).parameters()
        )

        print(f"  Trial {trial.number:2d} | MAE={best_mae:.4f} @ ep{best_ep} | "
              f"h={params['hidden_channels']} int={params['num_interactions']} "
              f"lr={params['lr']:.1e} sched={params['scheduler']} "
              f"bs={params['batch_size']} params={n_params:,}",
              flush=True)

        torch.cuda.empty_cache()
        return best_mae

    return objective


# ── Main ──────────────────────────────────────────────────────

def main():
    import optuna

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--fast-epochs", type=int, default=100)
    parser.add_argument("--full-epochs", type=int, default=500)
    parser.add_argument("--full-patience", type=int, default=40)
    args = parser.parse_args()

    ensure_dirs(OUT_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 4: SchNet Optuna Tuning ===")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Trials: {args.n_trials}, Fast epochs: {args.fast_epochs}, "
          f"Full epochs: {args.full_epochs}", flush=True)

    train_data, valid_data, test_data, y_mean, y_std, has_charges = load_data()
    print(f"  Gasteiger charges: {has_charges}")

    # ── Phase 1: Optuna search ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1: Optuna search ({args.n_trials} trials, {args.fast_epochs} epochs each)")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.NopPruner(),
    )

    objective = create_objective(train_data, valid_data, y_mean, y_std,
                                 device, args.fast_epochs, use_charges=has_charges)
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print(f"\n  Best trial {best.number}: MAE={best.value:.4f}")
    print(f"  Params: {best.params}")

    save_json(best.params, OUT_DIR / "optuna_best_params.json")

    trial_df = study.trials_dataframe()
    trial_df.to_csv(OUT_DIR / "optuna_trials.csv", index=False)

    # ── Phase 2: Full retrain with best params ────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Full retrain (best params, {args.full_epochs} epochs, patience={args.full_patience})")
    print(f"{'='*60}\n")

    from torch_geometric.loader import DataLoader

    best_params = dict(best.params)
    train_loader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=best_params["batch_size"])
    test_loader = DataLoader(test_data, batch_size=best_params["batch_size"])

    best_mae, best_epoch, best_state, log_rows = run_training(
        best_params, train_loader, valid_loader, y_mean, y_std,
        device, max_epochs=args.full_epochs, patience=args.full_patience,
        verbose=True, use_charges=has_charges,
    )

    pd.DataFrame(log_rows).to_csv(OUT_DIR / "retrain_log.csv", index=False)

    # ── Evaluate on test ──────────────────────────────────────
    model = SchNetWrapper(
        hidden_channels=best_params["hidden_channels"],
        num_filters=best_params["num_filters"],
        num_interactions=best_params["num_interactions"],
        num_gaussians=best_params["num_gaussians"],
        cutoff=best_params["cutoff"],
        dropout=best_params["dropout"],
        use_charges=has_charges,
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_pred, test_true = evaluate(model, test_loader, device)
    test_pred_real = test_pred * y_std + y_mean
    test_true_real = test_true * y_std + y_mean

    m = regression_metrics(test_true_real, test_pred_real)

    print(f"\n{'='*60}")
    print(f"  SchNet Optuna-tuned Test Results")
    print(f"{'='*60}")
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f}  RMSE={m[t]['rmse']:.4f}  R2={m[t]['r2']:.4f}")
    print(f"  avg  : MAE={m['average']['mae']:.4f}  RMSE={m['average']['rmse']:.4f}  R2={m['average']['r2']:.4f}")

    # Compare with previous SchNet
    prev_path = RESULTS_DIR / "phase4" / "schnet_metrics.json"
    if prev_path.exists():
        import json
        with open(prev_path) as f:
            prev = json.load(f)
        prev_mae = prev["metrics"]["average"]["mae"]
        prev_r2 = prev["metrics"]["average"]["r2"]
        print(f"\n  vs previous SchNet: MAE={prev_mae:.4f} R2={prev_r2:.4f}")
        print(f"  MAE improvement: {prev_mae - m['average']['mae']:.4f}")
        print(f"  R2  improvement: {m['average']['r2'] - prev_r2:.4f}")

    n_params = sum(p.numel() for p in model.parameters())

    torch.save(best_state, MODELS_DIR / "gnn_schnet_3d_tuned.pt")

    save_json({
        "model": "SchNet_3D_optuna",
        "params": best_params,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_trained": len(log_rows),
        "n_trials": args.n_trials,
        "metrics": m,
    }, OUT_DIR / "schnet_tuned_metrics.json")

    save_json({
        "phase": "4",
        "sub_stage": "4.4",
        "experiment": "phase4_schnet_optuna",
        "model": "SchNet_3D_optuna",
        "data_desc": "30k CHONSFCl",
        "elements": "C,Cl,F,H,N,O,S",
        "mw_range": "200-1000",
        "n_data": 30000,
        "split": "random_test",
        "metrics": m,
    }, RESULTS_DIR / "experiments" / "phase4_schnet_optuna.json")

    print(f"\n  Saved to {OUT_DIR}/")
    print(f"  Model saved to {MODELS_DIR / 'gnn_schnet_3d_tuned.pt'}")


if __name__ == "__main__":
    main()
