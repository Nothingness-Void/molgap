"""
Build 2D bond-topology graphs for Phase 7 GPS training.
Pure RDKit topology — no 3D embedding needed, very fast.

Usage:
  .venv\Scripts\python.exe scripts/phase7/build_2d_graphs_local.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from molgap.constants import RAW_DIR, RESULTS_DIR, TARGET_COLS
from molgap.graphs import smiles_to_2d_pyg
from molgap.utils import ensure_dirs

PHASE7_DIR = RESULTS_DIR / "phase7"
DEFAULT_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
DEFAULT_OUT = PHASE7_DIR / "pyg_2d_graphs_bond_300k.pt"
SHARD_DIR = PHASE7_DIR / "graph_2d_shards"
SHARD_SIZE = 50000


def main():
    ensure_dirs(PHASE7_DIR, SHARD_DIR)

    df = pd.read_csv(DEFAULT_CSV)
    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=TARGET_COLS + ["smiles"])
    df = df[df["gap"] > 0].reset_index(drop=True)

    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    smiles_list = df[smiles_col].tolist()
    targets = df[TARGET_COLS].values.astype(np.float32)
    total = len(smiles_list)
    print(f"Total molecules: {total}")

    t0 = time.time()
    buf = []
    ok = 0
    shard_idx = 0

    for i, smi in enumerate(tqdm(smiles_list, desc="Building 2D graphs")):
        data = smiles_to_2d_pyg(smi)
        if data is not None:
            data.y = torch.tensor(targets[i], dtype=torch.float32).unsqueeze(0)
            buf.append(data)
            ok += 1

        if (i + 1) % SHARD_SIZE == 0 and buf:
            _save_shard(buf, shard_idx)
            tqdm.write(f"  Shard {shard_idx}: {len(buf)} graphs")
            buf.clear()
            shard_idx += 1

    if buf:
        _save_shard(buf, shard_idx)
        print(f"  Shard {shard_idx}: {len(buf)} graphs")

    elapsed = time.time() - t0
    print(f"\n{ok}/{total} graphs built in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    _merge_shards()


def _save_shard(graphs, idx):
    torch.save(graphs, SHARD_DIR / f"shard_{idx:04d}.pt")


def _merge_shards():
    import glob as g
    shard_files = sorted(g.glob(str(SHARD_DIR / "shard_*.pt")))
    all_graphs = []
    for f in shard_files:
        all_graphs.extend(torch.load(f, weights_only=False))
    print(f"Total 2D graphs: {len(all_graphs)}")
    torch.save(all_graphs, DEFAULT_OUT)
    size_mb = os.path.getsize(DEFAULT_OUT) / 1e6
    print(f"Saved: {DEFAULT_OUT} ({size_mb:.1f} MB)")
    for f in shard_files:
        os.remove(f)
    if SHARD_DIR.exists():
        SHARD_DIR.rmdir()
    print("Shards cleaned up.")


if __name__ == "__main__":
    main()
