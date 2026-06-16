"""A/B step 1: build the shared 10k subset (2D + 3D graphs + scaffold split).

Samples a fixed-seed 10k subset from the 300k training CSV, builds BOTH the 3D
ETKDG graph (with Gasteiger charges, for SchNet) and the 2D bond-topology graph
for each molecule in a single pass, and keeps only molecules where BOTH succeed —
so graphs_3d[i] and graphs_2d[i] are row-matched by construction (no separate
alignment step needed). A Murcko-scaffold split (80/10/10, scaffold-disjoint) is
saved once and reused by every arm so all encoders see the identical split.

Sharded + resumable (memory rule): builds in shards of SHARD_SIZE under
results/ab3d/_shards/, skips finished shards on rerun, then merges.

Outputs (results/ab3d/):
  graphs_3d.pt   list[Data(z,pos,charges,y)]        row-matched to graphs_2d
  graphs_2d.pt   list[Data(x,edge_index,edge_attr,y)]
  meta.json      {n, smiles:[...], scaffold:[...]}
  split.json     {train:[...], val:[...], test:[...]}  indices into the kept order

Usage:
  .venv\\Scripts\\python.exe scripts/ab3d/build_subset.py
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from molgap.constants import RAW_DIR, RESULTS_DIR, SEED, TARGET_COLS
from molgap.graphs import smiles_to_pyg, smiles_to_2d_pyg
from molgap.utils import ensure_dirs, murcko_scaffold_smiles

CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
OUT = RESULTS_DIR / "ab3d"
SHARD_DIR = OUT / "_shards"
N_SAMPLE = 10_000
SHARD_SIZE = 2_000
TEST_FRAC = 0.10
VAL_FRAC = 0.10
# Drop degenerate conformers: shortest real chemical bond is ~0.96 Å, so two
# distinct atoms closer than this are a broken ETKDG/MMFF geometry. They cause
# 0/0 = NaN in direction-normalizing encoders (ViSNet, TensorNet). Removing them
# cleans the data for ALL arms fairly. ~0.9% of the 10k.
MIN_ATOM_DIST = 0.5


def min_pair_dist(pos) -> float:
    if pos.shape[0] < 2:
        return 99.0
    dm = torch.cdist(pos, pos) + torch.eye(pos.shape[0]) * 99.0
    return float(dm.min())


def build_one(smi: str, target):
    """Build (3D, 2D) for one SMILES; return (g3d, g2d) or None if either fails
    or the 3D conformer is degenerate (atoms overlapping)."""
    g3d = smiles_to_pyg(smi, use_charges=True)
    if g3d is None:
        return None
    if min_pair_dist(g3d.pos) < MIN_ATOM_DIST:
        return None
    g2d = smiles_to_2d_pyg(smi)
    if g2d is None:
        return None
    y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    g3d.y = y
    g2d.y = y
    return g3d, g2d


def main():
    ensure_dirs(OUT, SHARD_DIR)

    df = pd.read_csv(CSV)
    df = df.dropna(subset=TARGET_COLS)
    df = df[df["gap"] > 0].reset_index(drop=True)
    df = df.sample(n=min(N_SAMPLE, len(df)),
                   random_state=np.random.RandomState(SEED)).reset_index(drop=True)
    smiles = df["smiles"].tolist()
    targets = df[TARGET_COLS].to_numpy(dtype=np.float32)
    print(f"Sampled {len(df)} molecules; building 2D+3D graphs in shards of {SHARD_SIZE}")

    n_shards = (len(df) + SHARD_SIZE - 1) // SHARD_SIZE
    for s in range(n_shards):
        shard_path = SHARD_DIR / f"shard_{s:03d}.pt"
        if shard_path.exists():
            print(f"  shard {s+1}/{n_shards} exists, skip")
            continue
        lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, len(df))
        g3d_list, g2d_list, smi_list = [], [], []
        for i in tqdm(range(lo, hi), desc=f"shard {s+1}/{n_shards}"):
            built = build_one(smiles[i], targets[i])
            if built is None:
                continue
            g3d, g2d = built
            g3d_list.append(g3d)
            g2d_list.append(g2d)
            smi_list.append(smiles[i])
        torch.save({"g3d": g3d_list, "g2d": g2d_list, "smiles": smi_list}, str(shard_path))
        print(f"  shard {s+1}: kept {len(smi_list)}/{hi-lo}")

    # Merge shards (preserves order), filtering degenerate conformers. The filter
    # also runs here (not just in build_one) so shards built before the filter was
    # added are cleaned on re-merge without rebuilding.
    print("Merging shards ...")
    g3d_all, g2d_all, smi_all = [], [], []
    dropped = 0
    for s in range(n_shards):
        d = torch.load(str(SHARD_DIR / f"shard_{s:03d}.pt"), weights_only=False)
        for g3, g2, smi in zip(d["g3d"], d["g2d"], d["smiles"]):
            if min_pair_dist(g3.pos) < MIN_ATOM_DIST:
                dropped += 1
                continue
            g3d_all.append(g3)
            g2d_all.append(g2)
            smi_all.append(smi)
    n = len(smi_all)
    print(f"Dropped {dropped} degenerate conformers (< {MIN_ATOM_DIST} A)")
    print(f"Total kept: {n}/{len(df)} ({100*n/len(df):.1f}%)")

    scaffolds = [murcko_scaffold_smiles(s) or "NONE" for s in smi_all]
    n_scaf = len(set(scaffolds))

    # Scaffold-disjoint split: test, then val out of the remainder.
    idx = np.arange(n)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    rest, test = next(gss1.split(idx, groups=scaffolds))
    val_frac_of_rest = VAL_FRAC / (1.0 - TEST_FRAC)
    rest_scaf = [scaffolds[i] for i in rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_rest, random_state=SEED)
    tr_rel, va_rel = next(gss2.split(rest, groups=rest_scaf))
    train = rest[tr_rel].tolist()
    val = rest[va_rel].tolist()
    test = test.tolist()
    print(f"Scaffolds: {n_scaf} unique. Split train/val/test = "
          f"{len(train)}/{len(val)}/{len(test)} (scaffold-disjoint)")

    labels = torch.stack([g.y.squeeze(0) for g in g3d_all])  # [n, 3]
    torch.save(g3d_all, str(OUT / "graphs_3d.pt"))
    torch.save(g2d_all, str(OUT / "graphs_2d.pt"))
    torch.save(labels, str(OUT / "labels.pt"))
    (OUT / "meta.json").write_text(json.dumps(
        {"n": n, "smiles": smi_all, "scaffold": scaffolds}), encoding="utf-8")
    (OUT / "split.json").write_text(json.dumps(
        {"train": train, "val": val, "test": test}), encoding="utf-8")
    print(f"[OK] Saved graphs + meta + split to {OUT}")


if __name__ == "__main__":
    main()
