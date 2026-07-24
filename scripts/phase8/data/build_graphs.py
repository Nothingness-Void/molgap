"""
Build Phase 8 graph caches from the fixed-size replacement 300k dataset.

Outputs are intentionally separate from Phase 7 caches:
  results/phase8/pyg_2d_graphs_bond_replacement_300k.pt
  results/phase8/pyg_3d_graphs_etkdg_replacement_300k.pt

Each graph carries `source_idx`, the row index in the filtered CSV, so 2D/3D
embeddings can be aligned after ETKDG failures.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/data/build_graphs.py --which both
  .venv\\Scripts\\python.exe scripts/phase8/data/build_graphs.py --which 3d --resume
  .venv\\Scripts\\python.exe scripts/phase8/data/build_graphs.py --max-rows 1000
  .venv\\Scripts\\python.exe scripts/phase8/data/build_graphs.py --csv data/raw/phase7_chonsfcl_mw200_1000_300k.csv --tag old30k --max-rows 30000
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from molgap.constants import RAW_DIR, RESULTS_DIR, TARGET_COLS
from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"
DEFAULT_CSV = RAW_DIR / "phase8_replacement_300k.csv"
OUT_2D = PHASE8_DIR / "pyg_2d_graphs_bond_replacement_300k.pt"
OUT_3D = PHASE8_DIR / "pyg_3d_graphs_etkdg_replacement_300k.pt"
REPORT = PHASE8_DIR / "graph_build_report.json"
SHARD_SIZE_2D = 50000
SHARD_SIZE_3D = 10000


def _load_input(
    csv_path: Path,
    max_rows: int | None,
    start_row: int = 0,
) -> tuple[list[str], np.ndarray, list[int]]:
    df = pd.read_csv(csv_path)
    for col in TARGET_COLS + ["mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=TARGET_COLS + ["smiles"])
    df = df[df["gap"] > 0].reset_index(drop=True)
    if start_row < 0 or start_row >= len(df):
        raise ValueError(f"start_row must be in [0, {len(df) - 1}], got {start_row}")
    df = df.iloc[start_row:].reset_index(drop=True)
    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    smiles = df[smiles_col].astype(str).tolist()
    targets = df[TARGET_COLS].values.astype(np.float32)
    source_idx = list(range(start_row, start_row + len(df)))
    return smiles, targets, source_idx


def _attach_labels(data, target, source_idx: int):
    data.y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    data.source_idx = torch.tensor([source_idx], dtype=torch.long)
    return data


def _build_one_3d(args):
    """Build a graph in a worker without returning Torch tensors over mp IPC.

    PyTorch registers custom multiprocessing reducers for tensors. On large
    CPU jobs that moves every graph through shared memory and can exhaust the
    node's shared-memory backing store even when ordinary RAM is available.
    Return NumPy payloads instead; the parent reconstructs the PyG Data object.
    """
    source_idx, smi, target = args
    try:
        data = smiles_to_pyg(smi, use_charges=True, mmff_iters=200)
        if data is None:
            return None
        charges = data.charges.detach().cpu().numpy() if hasattr(data, "charges") else None
        return (
            source_idx,
            target,
            data.z.detach().cpu().numpy(),
            data.pos.detach().cpu().numpy(),
            charges,
        )
    except Exception:
        return None


def _restore_one_3d(payload):
    """Create a PyG Data object in the parent from a worker NumPy payload."""
    from torch_geometric.data import Data

    source_idx, target, z, pos, charges = payload
    data = Data(
        z=torch.from_numpy(z).to(dtype=torch.long),
        pos=torch.from_numpy(pos).to(dtype=torch.float32),
    )
    if charges is not None:
        data.charges = torch.from_numpy(charges).to(dtype=torch.float32)
    return _attach_labels(data, target, source_idx)


def _progress_path(kind: str, suffix: str, work_dir: Path = PHASE8_DIR) -> Path:
    return work_dir / f"graph_build_{kind}_{suffix}.progress"


def _shard_dir(kind: str, suffix: str, work_dir: Path = PHASE8_DIR) -> Path:
    return work_dir / f"graph_{kind}_shards_{suffix}"


def _save_shard(graphs, shard_dir: Path, idx: int):
    path = shard_dir / f"shard_{idx:04d}.pt"
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(graphs, temporary)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _merge_shards(shard_dir: Path, out_path: Path) -> int:
    print(f"Merging shards into {out_path} ...", flush=True)
    shard_files = sorted(glob.glob(str(shard_dir / "shard_*.pt")))
    all_graphs = []
    for f in shard_files:
        all_graphs.extend(torch.load(f, weights_only=False))
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    torch.save(all_graphs, temporary)
    os.replace(temporary, out_path)
    n = len(all_graphs)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Saved {n} graphs -> {out_path} ({size_gb:.2f} GB)", flush=True)
    for f in shard_files:
        os.remove(f)
    if shard_dir.exists():
        shard_dir.rmdir()
    return n


def build_2d(smiles: list[str], targets: np.ndarray, source_idx: list[int], out_path: Path,
             suffix: str, resume: bool, work_dir: Path = PHASE8_DIR) -> dict:
    shard_dir = _shard_dir("2d", suffix, work_dir)
    progress = _progress_path("2d", suffix, work_dir)
    ensure_dirs(PHASE8_DIR, work_dir, shard_dir, out_path.parent)

    start = int(progress.read_text().strip()) if resume and progress.exists() else 0
    if start:
        print(f"2D resume: {start}/{len(smiles)} processed", flush=True)

    buf, ok = [], 0
    processed = start
    shard_idx = start // SHARD_SIZE_2D
    t0 = time.time()
    for i in tqdm(range(start, len(smiles)), desc="2D graphs"):
        data = smiles_to_2d_pyg(smiles[i])
        processed += 1
        if data is not None:
            buf.append(_attach_labels(data, targets[i], source_idx[i]))
            ok += 1
        if processed % SHARD_SIZE_2D == 0 and buf:
            _save_shard(buf, shard_dir, shard_idx)
            tqdm.write(f"  2D shard {shard_idx}: {len(buf)} graphs")
            buf.clear()
            shard_idx += 1
            _atomic_write_text(progress, str(processed))

    if buf:
        _save_shard(buf, shard_dir, shard_idx)
        print(f"  2D shard {shard_idx}: {len(buf)} graphs", flush=True)
    _atomic_write_text(progress, str(processed))

    n_graphs = _merge_shards(shard_dir, out_path) if processed >= len(smiles) else ok
    if processed >= len(smiles) and progress.exists():
        progress.unlink()
    return {
        "kind": "2d",
        "processed": processed,
        "graphs": n_graphs,
        "failed": processed - n_graphs,
        "elapsed_s": time.time() - t0,
        "out": str(out_path),
    }


def build_3d(smiles: list[str], targets: np.ndarray, source_idx: list[int], out_path: Path,
             suffix: str, resume: bool, n_jobs: int,
             work_dir: Path = PHASE8_DIR) -> dict:
    shard_dir = _shard_dir("3d", suffix, work_dir)
    progress = _progress_path("3d", suffix, work_dir)
    ensure_dirs(PHASE8_DIR, work_dir, shard_dir, out_path.parent)

    start = int(progress.read_text().strip()) if resume and progress.exists() else 0
    if start:
        print(f"3D resume: {start}/{len(smiles)} processed", flush=True)

    work = [(source_idx[i], smiles[i], targets[i].tolist()) for i in range(start, len(smiles))]
    buf, ok = [], 0
    processed = start
    shard_idx = start // SHARD_SIZE_3D
    t0 = time.time()
    with mp.Pool(n_jobs) as pool:
        for result in tqdm(
            pool.imap(_build_one_3d, work, chunksize=500),
            total=len(work),
            desc=f"3D ETKDG ({n_jobs}w)",
        ):
            processed += 1
            if result is not None:
                buf.append(_restore_one_3d(result))
                ok += 1
            if processed % SHARD_SIZE_3D == 0 and buf:
                _save_shard(buf, shard_dir, shard_idx)
                tqdm.write(f"  3D shard {shard_idx}: {len(buf)} graphs")
                buf.clear()
                shard_idx += 1
                _atomic_write_text(progress, str(processed))

    if buf:
        _save_shard(buf, shard_dir, shard_idx)
        print(f"  3D shard {shard_idx}: {len(buf)} graphs", flush=True)
    _atomic_write_text(progress, str(processed))

    n_graphs = _merge_shards(shard_dir, out_path) if processed >= len(smiles) else ok
    if processed >= len(smiles) and progress.exists():
        progress.unlink()
    return {
        "kind": "3d",
        "processed": processed,
        "graphs": n_graphs,
        "failed": processed - n_graphs,
        "elapsed_s": time.time() - t0,
        "out": str(out_path),
        "n_jobs": n_jobs,
    }


def main():
    parser = argparse.ArgumentParser(description="Build Phase 8 replacement graph caches")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--which", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--out-2d", type=Path, default=OUT_2D)
    parser.add_argument("--out-3d", type=Path, default=OUT_3D)
    parser.add_argument("--tag", type=str, default=None,
                        help="output/run tag, e.g. old30k or replacement30k")
    parser.add_argument("--start-row", type=int, default=0,
                        help="first filtered CSV row to build; preserves original source_idx values")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PHASE8_DIR,
        help="durable shard/progress directory; defaults to results/phase8",
    )
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR)
    suffix = args.tag or ("replacement_300k" if args.max_rows is None else f"n{args.max_rows}")
    smiles, targets, source_idx = _load_input(args.csv, args.max_rows, args.start_row)
    print(
        f"Input rows: {len(smiles)} from {args.csv} "
        f"(start_row={args.start_row})",
        flush=True,
    )

    report = {
        "csv": str(args.csv),
        "n_input": len(smiles),
        "max_rows": args.max_rows,
        "start_row": args.start_row,
        "results": [],
    }
    if args.which in {"2d", "both"}:
        out = args.out_2d if args.out_2d != OUT_2D else PHASE8_DIR / f"pyg_2d_graphs_bond_{suffix}.pt"
        report["results"].append(
            build_2d(
                smiles, targets, source_idx, out, suffix, args.resume, args.work_dir
            )
        )
    if args.which in {"3d", "both"}:
        out = args.out_3d if args.out_3d != OUT_3D else PHASE8_DIR / f"pyg_3d_graphs_etkdg_{suffix}.pt"
        report["results"].append(
            build_3d(
                smiles,
                targets,
                source_idx,
                out,
                suffix,
                args.resume,
                args.n_jobs,
                args.work_dir,
            )
        )

    report_suffix = suffix if args.which == "both" else f"{args.which}_{suffix}"
    report_path = (
        REPORT
        if args.tag is None and args.max_rows is None and args.work_dir == PHASE8_DIR
        else args.work_dir / f"graph_build_report_{report_suffix}.json"
    )
    _atomic_write_text(report_path, json.dumps(report, indent=2))
    print(f"Report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
