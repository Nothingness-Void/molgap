"""Run frozen Base and Expert on the archive-r02 Oracle probe with resumable chunks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.constants import RESULTS_DIR
from molgap.inference import load_routed_dual_gps_hybrid, predict_smiles_batch_dual_gps_candidates
from molgap.utils import ensure_dirs


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
TARGETS = ("homo", "lumo", "gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=OUT_DIR / "oracle_probe_20k.parquet")
    parser.add_argument("--chunks-dir", type=Path, default=OUT_DIR / "oracle_probe_chunks")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "oracle_probe_predictions.parquet")
    parser.add_argument("--manifest-out", type=Path, default=OUT_DIR / "oracle_probe_prediction_manifest.json")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = pd.read_parquet(args.probe)
    ensure_dirs(args.chunks_dir, args.out.parent)
    if args.overwrite:
        for path in args.chunks_dir.glob("chunk-*.npz"):
            path.unlink()
        args.out.unlink(missing_ok=True)
        args.manifest_out.unlink(missing_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} rows={len(probe):,}", flush=True)
    models = load_routed_dual_gps_hybrid(device)
    started = time.perf_counter()
    for start in range(0, len(probe), args.chunk_size):
        stop = min(start + args.chunk_size, len(probe))
        path = args.chunks_dir / f"chunk-{start:06d}-{stop:06d}.npz"
        if path.exists():
            print(f"skip {start}:{stop}", flush=True)
            continue
        block = probe.iloc[start:stop]
        block_seed = args.seed + int(block.probe_idx.iloc[0])
        valid, base, expert, h2, h3, gps, schnet = predict_smiles_batch_dual_gps_candidates(
            block.canonical_smiles.tolist(), models=models, random_seed=block_seed
        )
        np.savez_compressed(
            path, probe_idx=block.iloc[valid].probe_idx.to_numpy(), base=base,
            expert=expert, h2=h2, h3=h3, gps=gps, schnet=schnet,
        )
        print(f"done {stop:,}/{len(probe):,} valid={len(valid):,}", flush=True)

    rows = []
    for path in sorted(args.chunks_dir.glob("chunk-*.npz")):
        with np.load(path) as chunk:
            block = pd.DataFrame({"probe_idx": chunk["probe_idx"]})
            for prefix in ("base", "expert", "gps", "schnet"):
                values = chunk[prefix]
                for i, target in enumerate(TARGETS):
                    block[f"{prefix}_{target}"] = values[:, i]
            rows.append(block)
    predictions = pd.concat(rows, ignore_index=True).sort_values("probe_idx")
    output = probe.merge(predictions, on="probe_idx", how="left", validate="one_to_one")
    output["prediction_success"] = output["base_gap"].notna()
    output.to_parquet(args.out, index=False)
    manifest = {
        "probe_n": len(probe),
        "valid_n": int(output.prediction_success.sum()),
        "failed_n": int((~output.prediction_success).sum()),
        "chunk_size": args.chunk_size,
        "random_seed_rule": "seed + probe_idx",
        "seed": args.seed,
        "geometry": "ETKDGv3 + MMFF; PubChemQC PM6 coordinates unused",
        "device": str(device),
        "elapsed_seconds_this_run": time.perf_counter() - started,
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
