"""
MoE / Topology-Specific Experts — controlled experiment vs the single FusionHead.

QUESTION (regression, unanswered by the literature):
  On 300k B3LYP HOMO/LUMO/Gap, does replacing the single fusion head with a
  learned-gating Mixture-of-Experts head improve accuracy over the Phase 7
  baseline — WITHOUT 1M data, WITHOUT touching the frozen encoders?

DESIGN (strict A/B, only the head structure changes):
  - Same frozen pre-computed embeddings (gps_2d_embeddings_aligned.pt, schnet_3d_embeddings.pt)
  - Same SEED=42 split as fusion_optuna_local.py (random AND scaffold variants)
  - Same optimizer/loss/epochs/patience
  - Baseline  = FusionHead (src/molgap/fusion.py, unchanged)
  - Treatment = MoEFusionHead (this file): shared gate-fused trunk -> learned soft
    routing over N expert MLP heads. Routing is LEARNED (not MW-binned).

Expert/gating design follows TopExpert (Kim et al., AAAI 2023; github.com/kimsu55/ToxExpert,
model.py classes `gate`/`expert`/`GNN_topexpert`): experts are light heads on a
SHARED trunk, a learnable gate produces per-molecule soft weights over experts.
Adapted from binary classification to 3-target regression.

Usage (on the 5060 box, inside .venv):
  .venv\\Scripts\\python.exe scripts/phase7/moe_experts_local.py --split random --experts 4
  .venv\\Scripts\\python.exe scripts/phase7/moe_experts_local.py --split scaffold --experts 4
  (also run --experts 1 as a sanity control: should ~= baseline)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import RESULTS_DIR, MODELS_DIR
from molgap.fusion import FusionHead

PHASE7_DIR = RESULTS_DIR / "phase7"
OUT_DIR = PHASE7_DIR / "moe_experiment"
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── MoE head (treatment) ──────────────────────────────────────────
class MoEFusionHead(nn.Module):
    """Gate-fuse 2D+3D embeddings into a shared trunk, then route over N expert
    MLP heads with learned per-molecule soft gating.

    Routing is data-driven (gate sees the fused embedding), NOT hand-partitioned.
    n_experts=1 collapses to a single head (control == baseline-ish).
    """

    def __init__(self, hidden=192, dropout=0.0, dim_2d=192, dim_3d=192,
                 n_targets=3, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.n_targets = n_targets
        self.proj_2d = nn.Linear(dim_2d, hidden)
        self.proj_3d = nn.Linear(dim_3d, hidden)
        self.gate_fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        # learned router over experts (soft gating, TopExpert-style)
        self.router = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_experts)
        )
        # N expert MLP heads on the shared trunk
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.SiLU(),
                nn.Linear(hidden // 2, n_targets),
            ) for _ in range(n_experts)
        ])

    def forward(self, h_2d, h_3d, return_gate=False):
        h_2d = self.proj_2d(h_2d)
        h_3d = self.proj_3d(h_3d)
        g = self.gate_fuse(torch.cat([h_2d, h_3d], dim=-1))
        h = g * h_2d + (1 - g) * h_3d              # shared trunk (N x hidden)
        w = torch.softmax(self.router(h), dim=-1)  # N x experts (soft weights)
        # stack expert outputs: N x experts x targets
        outs = torch.stack([e(h) for e in self.experts], dim=1)
        # weighted combination over experts
        y = torch.sum(w.unsqueeze(-1) * outs, dim=1)  # N x targets
        if return_gate:
            return y, w
        return y


# ── data ──────────────────────────────────────────
def scaffold_from_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def load_data(split="random"):
    emb_3d = torch.load(PHASE7_DIR / "schnet_3d_embeddings.pt", weights_only=False)
    emb_2d = torch.load(PHASE7_DIR / "gps_2d_embeddings_aligned.pt", weights_only=False)
    graphs = torch.load(PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt", weights_only=False)
    labels = torch.stack([g.y.squeeze(0) for g in graphs])
    N = emb_3d.shape[0]
    assert emb_2d.shape[0] == N == labels.shape[0]

    if split == "random":
        idx = np.random.RandomState(SEED).permutation(N)
        n_tr, n_va = int(0.8 * N), int(0.1 * N)
        sp = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
    elif split == "scaffold":
        # group by Bemis-Murcko scaffold; disjoint scaffolds across splits
        smis = [getattr(g, "smiles", None) for g in graphs]
        if any(s is None for s in smis):
            raise RuntimeError("graphs lack .smiles; cannot do scaffold split. "
                               "Re-run with --split random or add smiles to the graph cache.")
        from collections import defaultdict
        groups = defaultdict(list)
        for i, s in enumerate(smis):
            groups[scaffold_from_smiles(s)].append(i)
        scfs = sorted(groups.values(), key=len, reverse=True)
        rng = np.random.RandomState(SEED)
        rng.shuffle(scfs)
        tr, va, te = [], [], []
        n_tr, n_va = int(0.8 * N), int(0.1 * N)
        for grp in scfs:
            if len(tr) < n_tr:
                tr += grp
            elif len(va) < n_va:
                va += grp
            else:
                te += grp
        sp = {"train": np.array(tr), "val": np.array(va), "test": np.array(te)}
    else:
        raise ValueError(split)
    del graphs
    return emb_2d, emb_3d, labels, sp


def make_loader(emb_2d, emb_3d, labels, ii, bs, shuffle):
    ds = TensorDataset(emb_2d[ii], emb_3d[ii], labels[ii])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=0)


# ── train / eval (shared by baseline and MoE) ──────────────────────────────────────────
def train_eval(model, data, lr, wd, bs, max_epochs, patience):
    emb_2d, emb_3d, labels, sp = data
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()
    is_moe = isinstance(model, MoEFusionHead)

    tr = make_loader(emb_2d, emb_3d, labels, sp["train"], bs, True)
    va = make_loader(emb_2d, emb_3d, labels, sp["val"], 2048, False)
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for h2, h3, y in tr:
            h2, h3, y = h2.to(device), h3.to(device), y.to(device)
            opt.zero_grad()
            pred = model(h2, h3)
            loss = crit(pred, y)
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
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    te = make_loader(emb_2d, emb_3d, labels, sp["test"], 2048, False)
    P, T = [], []
    with torch.no_grad():
        for h2, h3, y in te:
            P.append(model(h2.to(device), h3.to(device)).cpu().numpy()); T.append(y.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    metrics = {"best_val_mae": float(best_val)}
    for i, t in enumerate(["HOMO", "LUMO", "Gap"]):
        metrics[t] = {"mae": float(mean_absolute_error(T[:, i], P[:, i])),
                      "r2": float(r2_score(T[:, i], P[:, i]))}
    metrics["n_params"] = int(sum(p.numel() for p in model.parameters()))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=5.4e-4)   # Phase 7 best
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=1024)       # Phase 7 best
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])  # repeat for variance
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | split={args.split} | experts={args.experts}")
    data = load_data(args.split)
    print(f"  N={data[0].shape[0]}  train/val/test="
          f"{len(data[3]['train'])}/{len(data[3]['val'])}/{len(data[3]['test'])}")

    results = {"split": args.split, "experts": args.experts, "runs": []}
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        base = FusionHead("gate", args.hidden, 0.0)
        m_base = train_eval(base, data, args.lr, args.wd, args.bs, args.epochs, args.patience)
        moe = MoEFusionHead(args.hidden, 0.0, n_experts=args.experts)
        m_moe = train_eval(moe, data, args.lr, args.wd, args.bs, args.epochs, args.patience)
        dt = time.time() - t0
        row = {"seed": seed, "baseline": m_base, "moe": m_moe, "time_s": dt}
        results["runs"].append(row)
        print(f"\n[seed {seed}] ({dt/60:.1f} min)")
        for name, m in [("baseline", m_base), ("moe", m_moe)]:
            print(f"  {name:8s} Gap MAE={m['Gap']['mae']:.4f} R2={m['Gap']['r2']:.4f} "
                  f"| HOMO {m['HOMO']['mae']:.4f} LUMO {m['LUMO']['mae']:.4f} "
                  f"| params={m['n_params']:,}")

    # aggregate Gap MAE across seeds
    bg = np.array([r["baseline"]["Gap"]["mae"] for r in results["runs"]])
    mg = np.array([r["moe"]["Gap"]["mae"] for r in results["runs"]])
    results["summary"] = {
        "baseline_gap_mae_mean": float(bg.mean()), "baseline_gap_mae_std": float(bg.std()),
        "moe_gap_mae_mean": float(mg.mean()), "moe_gap_mae_std": float(mg.std()),
        "delta_gap_mae_mean": float(mg.mean() - bg.mean()),
    }
    print("\n=== SUMMARY (Gap MAE, lower=better) ===")
    print(f"  baseline {bg.mean():.4f} ± {bg.std():.4f}")
    print(f"  MoE({args.experts}) {mg.mean():.4f} ± {mg.std():.4f}")
    print(f"  Δ = {mg.mean()-bg.mean():+.4f} eV  "
          f"({'MoE better' if mg.mean()<bg.mean() else 'baseline better/tie'})")

    out = OUT_DIR / f"moe_{args.split}_e{args.experts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
