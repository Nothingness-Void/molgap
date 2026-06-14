"""
Build ETKDG 3D graphs locally for Phase 7 300k dataset.
Uses multiprocessing. Saves shards to disk to avoid OOM.

Usage:
  .venv\Scripts\python.exe scripts/phase7/build_graphs_local.py
  .venv\Scripts\python.exe scripts/phase7/build_graphs_local.py --resume
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

from molgap.constants import RAW_DIR, RESULTS_DIR, TARGET_COLS
from molgap.graphs import _build_one_labeled
from molgap.utils import ensure_dirs

PHASE7_DIR = RESULTS_DIR / "phase7"
DEFAULT_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
DEFAULT_OUT = PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt"
SHARD_DIR = PHASE7_DIR / "graph_shards"
SHARD_SIZE = 10000
PROGRESS_FILE = PHASE7_DIR / "graph_build_progress.txt"


def main():
    parser = argparse.ArgumentParser(description="Build ETKDG 3D graphs locally")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import multiprocessing as mp
    if args.n_jobs is None:
        args.n_jobs = max(1, mp.cpu_count() - 1)

    ensure_dirs(PHASE7_DIR, SHARD_DIR)

    df = pd.read_csv(args.csv)
    for col in TARGET_COLS + ["mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=TARGET_COLS + ["smiles"])
    df = df[df["gap"] > 0].reset_index(drop=True)

    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    smiles_list = df[smiles_col].tolist()
    targets = df[TARGET_COLS].values.astype(np.float32)
    total = len(smiles_list)

    # Resume: find how many already processed
    start_idx = 0
    if args.resume and PROGRESS_FILE.exists():
        start_idx = int(PROGRESS_FILE.read_text().strip())
        print(f"Resumed: {start_idx}/{total} already processed")

    remaining = total - start_idx
    print(f"Molecules: {total} total, {remaining} remaining")
    print(f"Workers: {args.n_jobs}, shard size: {SHARD_SIZE}")

    if remaining <= 0:
        print("All done, merging shards...")
        _merge_shards(args.out)
        return

    from tqdm import tqdm
    t0 = time.time()
    buf = []
    processed = start_idx
    shard_idx = start_idx // SHARD_SIZE

    work = list(zip(smiles_list[start_idx:], targets[start_idx:].tolist()))

    try:
        with mp.Pool(args.n_jobs) as pool:
            for result in tqdm(
                pool.imap(_build_one_labeled, work, chunksize=500),
                total=len(work),
                desc=f"Building ({args.n_jobs}w)",
            ):
                processed += 1
                if result is not None:
                    buf.append(result)

                if processed % SHARD_SIZE == 0:
                    _save_shard(buf, shard_idx)
                    tqdm.write(f"  Shard {shard_idx}: {len(buf)} graphs saved, mem freed")
                    buf.clear()
                    shard_idx += 1
                    PROGRESS_FILE.write_text(str(processed))
    except KeyboardInterrupt:
        print(f"\nInterrupted at {processed}/{total}")

    # Save remaining buffer
    if buf:
        _save_shard(buf, shard_idx)
        print(f"  Shard {shard_idx}: {len(buf)} graphs saved")
    PROGRESS_FILE.write_text(str(processed))

    elapsed = time.time() - t0
    print(f"\nProcessed in {elapsed / 60:.1f} min")

    if processed >= total:
        _merge_shards(args.out)


def _save_shard(graphs, idx):
    path = SHARD_DIR / f"shard_{idx:04d}.pt"
    torch.save(graphs, path)


def _merge_shards(out_path):
    print("Merging shards...")
    shard_files = sorted(glob.glob(str(SHARD_DIR / "shard_*.pt")))
    all_graphs = []
    for f in shard_files:
        all_graphs.extend(torch.load(f, weights_only=False))
    print(f"Total graphs: {len(all_graphs)}")
    torch.save(all_graphs, out_path)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Saved: {out_path} ({size_gb:.2f} GB)")
    # Clean up shards
    for f in shard_files:
        os.remove(f)
    if SHARD_DIR.exists():
        SHARD_DIR.rmdir()
    progress = PHASE7_DIR / "graph_build_progress.txt"
    if progress.exists():
        progress.remove() if hasattr(progress, 'remove') else os.remove(progress)
    print("Shards cleaned up.")


if __name__ == "__main__":
    main()
