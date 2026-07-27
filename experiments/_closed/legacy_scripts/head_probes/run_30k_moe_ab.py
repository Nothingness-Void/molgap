"""
Run the Phase 8 30k old-vs-replacement MoE A/B pipeline.

This orchestrates the thin Phase 8 CLIs without overwriting Phase 7 artifacts:
  1. build old30k and replacement30k 2D/3D graph caches;
  2. train GPS and SchNet encoders for each 30k dataset;
  3. train baseline FusionHead and MoEFusionHead for each dataset;
  4. write a compact comparison summary.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/head_probes/run_30k_moe_ab.py --rows 30000
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/head_probes/run_30k_moe_ab.py --rows 1000 --encoder-epochs 2 --fusion-epochs 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"
OLD_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
NEW_CSV = RAW_DIR / "phase8_replacement_300k.csv"


def run(cmd: list[str], skip_existing: Path | None = None):
    if skip_existing is not None and skip_existing.exists():
        print(f"[skip] {skip_existing}", flush=True)
        return
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def graph_path(kind: str, tag: str) -> Path:
    prefix = "pyg_2d_graphs_bond" if kind == "gps" else "pyg_3d_graphs_etkdg"
    return PHASE8_DIR / f"{prefix}_{tag}.pt"


def emb_path(kind: str, tag: str) -> Path:
    return PHASE8_DIR / f"{kind}_{tag}_embeddings.pt"


def metrics_path(kind: str, tag: str) -> Path:
    return PHASE8_DIR / f"{kind}_{tag}_metrics.json"


def model_path(kind: str, tag: str) -> Path:
    return PHASE8_DIR / f"{kind}_{tag}.pt"


def fusion_metrics_path(tag: str) -> Path:
    return PHASE8_DIR / f"moe_{tag}_metrics.json"


def run_dataset(name: str, csv: Path, rows: int, args):
    tag = f"{name}{rows // 1000}k" if rows >= 1000 else f"{name}{rows}"
    print(f"\n=== Dataset: {tag} ===", flush=True)

    g2d = graph_path("gps", tag)
    g3d = graph_path("schnet", tag)
    run([
        sys.executable, "scripts/phase8/data/build_graphs.py",
        "--csv", str(csv), "--tag", tag, "--max-rows", str(rows),
        "--which", "both", "--n-jobs", str(args.n_jobs), "--resume",
    ], skip_existing=g2d if g2d.exists() and g3d.exists() else None)

    run([
        sys.executable, "scripts/phase8/training/train_encoder.py",
        "--kind", "gps",
        "--graphs", str(g2d),
        "--model-out", str(model_path("gps", tag)),
        "--metrics-out", str(metrics_path("gps", tag)),
        "--embeddings-out", str(emb_path("gps", tag)),
        "--epochs", str(args.encoder_epochs),
        "--patience", str(args.encoder_patience),
    ], skip_existing=emb_path("gps", tag))

    run([
        sys.executable, "scripts/phase8/training/train_encoder.py",
        "--kind", "schnet",
        "--graphs", str(g3d),
        "--model-out", str(model_path("schnet", tag)),
        "--metrics-out", str(metrics_path("schnet", tag)),
        "--embeddings-out", str(emb_path("schnet", tag)),
        "--epochs", str(args.encoder_epochs),
        "--patience", str(args.encoder_patience),
    ], skip_existing=emb_path("schnet", tag))

    run([
        sys.executable, "scripts/phase8/training/train_fusion_head.py",
        "--emb-2d", str(emb_path("gps", tag)),
        "--emb-3d", str(emb_path("schnet", tag)),
        "--graphs-3d", str(g3d),
        "--experts", str(args.experts),
        "--epochs", str(args.fusion_epochs),
        "--patience", str(args.fusion_patience),
        "--out", str(fusion_metrics_path(tag)),
    ], skip_existing=fusion_metrics_path(tag))

    return tag


def summarize(tags: list[str], out: Path):
    rows = []
    for tag in tags:
        path = fusion_metrics_path(tag)
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "tag": tag,
            "n_aligned": m["n_aligned"],
            "baseline_avg_mae": m["baseline"]["average"]["mae"],
            "moe_avg_mae": m["moe"]["average"]["mae"],
            "delta_avg_mae": m["delta_average_mae"],
            "baseline_gap_mae": m["baseline"]["Gap"]["mae"],
            "moe_gap_mae": m["moe"]["Gap"]["mae"],
            "delta_gap_mae": m["delta_gap_mae"],
        })
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSummary -> {out}", flush=True)
    for r in rows:
        print(
            f"{r['tag']}: avg {r['baseline_avg_mae']:.4f}->{r['moe_avg_mae']:.4f} "
            f"(Δ {r['delta_avg_mae']:+.4f}); gap {r['baseline_gap_mae']:.4f}->{r['moe_gap_mae']:.4f} "
            f"(Δ {r['delta_gap_mae']:+.4f})",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Run old30k vs replacement30k MoE A/B")
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--encoder-epochs", type=int, default=80)
    parser.add_argument("--encoder-patience", type=int, default=15)
    parser.add_argument("--fusion-epochs", type=int, default=200)
    parser.add_argument("--fusion-patience", type=int, default=25)
    parser.add_argument("--experts", type=int, default=4)
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR)
    tags = [
        run_dataset("old", OLD_CSV, args.rows, args),
        run_dataset("replacement", NEW_CSV, args.rows, args),
    ]
    row_tag = f"{args.rows // 1000}k" if args.rows >= 1000 else str(args.rows)
    summarize(tags, PHASE8_DIR / f"moe_ab_{row_tag}_summary.json")


if __name__ == "__main__":
    main()
