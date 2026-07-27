"""Scan official PCQM4Mv2 train rows with the accepted dual-2D teacher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from molgap.multi2d import (
    DualGPSExpertPaths,
    add_mean_ensembles,
    load_dual_gps_experts,
    predict_dual_gps_experts,
)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(value: str) -> str | None:
    mol = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcqm-csv", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--historical-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--graph-batch-size", type=int, default=512)
    parser.add_argument("--hard-pool-size", type=int, default=200_000)
    parser.add_argument("--control-gps7", type=Path, required=True)
    parser.add_argument("--control-gps9", type=Path, required=True)
    parser.add_argument("--control-head", type=Path, required=True)
    parser.add_argument("--repair-gps7", type=Path, required=True)
    parser.add_argument("--repair-gps9", type=Path, required=True)
    parser.add_argument("--repair-head", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 3.38M-row residual scan")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    split = torch.load(args.split, map_location="cpu", weights_only=False)
    train_idx = torch.as_tensor(split["train"], dtype=torch.long)
    if not torch.equal(train_idx, torch.arange(len(train_idx), dtype=torch.long)):
        raise RuntimeError("Expected the official PCQM4Mv2 train split to be a contiguous prefix")
    n_train = len(train_idx)
    specs = [
        DualGPSExpertPaths("control_a", args.control_gps7, args.control_gps9, args.control_head),
        DualGPSExpertPaths("repair_v2", args.repair_gps7, args.repair_gps9, args.repair_head),
    ]
    device = torch.device("cuda")
    experts = load_dual_gps_experts(specs, device)
    expected_parts = (n_train + args.chunk_size - 1) // args.chunk_size
    atomic_json({
        "status": "runtime_ready", "official_split": "train", "n_train": n_train,
        "valid_test_excluded": True, "chunk_size": args.chunk_size,
        "expected_parts": expected_parts, "gpu": torch.cuda.get_device_name(0),
        "input_sha256": sha256_file(args.pcqm_csv),
    }, args.out_dir / "progress.json")

    completed = 0
    reader = pd.read_csv(args.pcqm_csv, compression="gzip", chunksize=args.chunk_size)
    for part_index, table in enumerate(reader):
        if int(table.idx.iloc[0]) >= n_train:
            break
        table = table.loc[table.idx < n_train, ["idx", "smiles", "homolumogap"]].copy()
        out_path = parts_dir / f"part-{part_index:04d}.parquet"
        report_path = parts_dir / f"part-{part_index:04d}.json"
        if out_path.is_file() and report_path.is_file():
            completed += 1
            print(f"reuse {out_path}", flush=True)
            continue
        source_rows = len(table)
        kept, raw = predict_dual_gps_experts(
            table.smiles, experts, device, graph_batch_size=args.graph_batch_size,
        )
        table = table.iloc[kept].reset_index(drop=True)
        predictions = add_mean_ensembles(raw, {"teacher": ["control_a", "repair_v2"]})
        target = table.homolumogap.to_numpy(np.float32)
        for name, value in predictions.items():
            table[f"{name}_gap"] = value[:, 2]
            table[f"{name}_abs_error"] = np.abs(value[:, 2] - target)
        table["expert_gap_disagreement"] = np.abs(
            predictions["control_a"][:, 2] - predictions["repair_v2"][:, 2]
        )
        temporary = out_path.with_name(f".{out_path.name}.tmp")
        table.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, out_path)
        atomic_json({
            "part": part_index, "source_start": int(table.idx.min()),
            "source_end": int(table.idx.max()), "rows": len(table),
            "invalid_rows": int(source_rows - len(table)), "sha256": sha256_file(out_path),
        }, report_path)
        completed += 1
        atomic_json({
            "status": "scanning", "completed_parts": completed,
            "expected_parts": expected_parts, "last_part": part_index,
        }, args.out_dir / "progress.json")
        print(f"part {part_index + 1}/{expected_parts}: {len(table):,} rows", flush=True)

    part_paths = sorted(parts_dir.glob("part-*.parquet"))
    if len(part_paths) != expected_parts:
        raise RuntimeError(f"Expected {expected_parts} parts, found {len(part_paths)}")
    per_part_keep = max(args.hard_pool_size * 2 // expected_parts, 2_000)
    hard_frames = []
    for path in part_paths:
        frame = pd.read_parquet(path)
        hard_frames.append(frame.nlargest(per_part_keep, "teacher_abs_error"))
    hard = pd.concat(hard_frames, ignore_index=True).nlargest(args.hard_pool_size * 2, "teacher_abs_error")

    historical = set()
    for chunk in pd.read_csv(args.historical_csv, usecols=["smiles"], chunksize=100_000):
        historical.update(value for value in map(canonical_smiles, chunk.smiles) if value)
    hard["canonical_smiles"] = [canonical_smiles(value) for value in hard.smiles]
    hard = hard.dropna(subset=["canonical_smiles"])
    hard = hard.drop_duplicates("canonical_smiles", keep="first")
    hard = hard.loc[~hard.canonical_smiles.isin(historical)]
    hard = hard.nlargest(args.hard_pool_size, "teacher_abs_error").reset_index(drop=True)
    hard_path = args.out_dir / "pcqm4mv2_train_hard_pool.parquet"
    temporary = hard_path.with_name(f".{hard_path.name}.tmp")
    hard.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, hard_path)
    manifest = {
        "status": "complete", "official_split": "train", "n_scanned": n_train,
        "valid_test_excluded": True, "parts": len(part_paths),
        "historical_source": str(args.historical_csv), "historical_unique": len(historical),
        "hard_pool_rows": len(hard), "hard_pool_sha256": sha256_file(hard_path),
        "teacher": "equal(control_a,repair_v2)",
    }
    atomic_json(manifest, args.out_dir / "manifest.json")
    atomic_json(manifest, args.out_dir / "progress.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
