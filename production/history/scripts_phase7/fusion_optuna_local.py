"""
Phase 7: Optuna hyperparameter search for the hybrid fusion head.

Embeddings are pre-computed, so each trial trains in seconds. Searches the
fusion type, hidden width, dropout, lr, weight_decay, batch_size. Then retrains
the best config longer and reports test metrics — compare against the hand-set
baseline (gate, hidden=128: test Gap MAE 0.084).

Resumable via SQLite study storage.

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/fusion_optuna_local.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import RESULTS_DIR, MODELS_DIR
from molgap.fusion import FusionHead

PHASE7_DIR = RESULTS_DIR / "phase7"
SEED = 42

N_TRIALS = 60
SEARCH_EPOCHS = 80
SEARCH_PATIENCE = 12
FINAL_EPOCHS = 300
FINAL_PATIENCE = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    emb_3d = torch.load(PHASE7_DIR / "schnet_3d_embeddings.pt", weights_only=False)
    emb_2d = torch.load(PHASE7_DIR / "gps_2d_embeddings_aligned.pt", weights_only=False)
    graphs = torch.load(PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt", weights_only=False)
    labels = torch.stack([g.y.squeeze(0) for g in graphs])
    del graphs
    assert emb_2d.shape[0] == emb_3d.shape[0] == labels.shape[0]
    N = emb_3d.shape[0]
    idx = np.random.RandomState(SEED).permutation(N)
    n_tr, n_va = int(0.8 * N), int(0.1 * N)
    sp = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
    return emb_2d, emb_3d, labels, sp


def make_loader(emb_2d, emb_3d, labels, ii, bs, shuffle):
    ds = TensorDataset(emb_2d[ii], emb_3d[ii], labels[ii])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=0)


def train_eval(params, data, max_epochs, patience, return_test=False, trial=None):
    emb_2d, emb_3d, labels, sp = data
    model = FusionHead(params["fusion_type"], params["hidden"], params["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()

    tr = make_loader(emb_2d, emb_3d, labels, sp["train"], params["batch_size"], True)
    va = make_loader(emb_2d, emb_3d, labels, sp["val"], 2048, False)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for h2, h3, y in tr:
            h2, h3, y = h2.to(device), h3.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(h2, h3), y)
            loss.backward()
            opt.step()
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for h2, h3, y in va:
                h2, h3, y = h2.to(device), h3.to(device), y.to(device)
                vl += crit(model(h2, h3), y).item() * y.size(0); vc += y.size(0)
        vmae = vl / vc
        sched.step(vmae)
        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if trial is not None:
            trial.report(vmae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if wait >= patience:
            break

    if not return_test:
        return best_val

    model.load_state_dict(best_state)
    model.eval()
    te = make_loader(emb_2d, emb_3d, labels, sp["test"], 2048, False)
    P, T = [], []
    with torch.no_grad():
        for h2, h3, y in te:
            h2, h3 = h2.to(device), h3.to(device)
            P.append(model(h2, h3).cpu().numpy()); T.append(y.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    metrics = {"best_val_mae": float(best_val)}
    for i, t in enumerate(["HOMO", "LUMO", "Gap"]):
        metrics[t] = {"mae": float(mean_absolute_error(T[:, i], P[:, i])),
                      "r2": float(r2_score(T[:, i], P[:, i]))}
    return best_val, model, metrics


def main():
    torch.manual_seed(SEED)
    print(f"Device: {device}")
    print("Loading embeddings + labels...")
    data = load_data()
    print(f"  N = {data[0].shape[0]}")

    def objective(trial):
        params = {
            "fusion_type": trial.suggest_categorical("fusion_type", ["gate", "concat"]),
            "hidden": trial.suggest_categorical("hidden", [64, 128, 192, 256]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        }
        return train_eval(params, data, SEARCH_EPOCHS, SEARCH_PATIENCE, trial=trial)

    storage = f"sqlite:///{(PHASE7_DIR / 'fusion_optuna.db').as_posix()}"
    study = optuna.create_study(
        direction="minimize", study_name="fusion_head",
        storage=storage, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"\nSearch done in {(time.time()-t0)/60:.1f} min, {len(study.trials)} trials")
    print(f"Best val MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Retrain best longer
    print(f"\n=== Retrain best ({FINAL_EPOCHS} ep) ===")
    bp = study.best_params
    best_val, model, metrics = train_eval(bp, data, FINAL_EPOCHS, FINAL_PATIENCE,
                                          return_test=True)
    print(f"  Best val MAE: {best_val:.4f}")
    for t in ["HOMO", "LUMO", "Gap"]:
        print(f"  {t}: MAE={metrics[t]['mae']:.4f}  R2={metrics[t]['r2']:.4f}")

    # Baseline for reference
    print("\n  (hand-set baseline gate/h128: HOMO 0.071 LUMO 0.064 Gap 0.084)")

    torch.save(model.state_dict(), MODELS_DIR / "hybrid_fusion_optuna.pt")
    with open(PHASE7_DIR / "fusion_optuna_metrics.json", "w") as f:
        json.dump({"best_params": bp, "test_metrics": metrics,
                   "n_trials": len(study.trials)}, f, indent=2)
    print(f"\n  Saved model + metrics")


if __name__ == "__main__":
    main()
