"""Run the paired common/OOD/P8-hard gate for repaired-2M Fusion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from molgap.hierarchical_external_eval import (
    build_external_graph_cache,
    evaluate_paired_external,
    extract_external_schnet_embeddings,
    load_paired_prediction_tables,
    sha256,
)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-m-predictions", type=Path, required=True)
    parser.add_argument("--routed-v4-predictions", type=Path, required=True)
    parser.add_argument("--dense-gates", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--augmented-checkpoint", type=Path, required=True)
    parser.add_argument("--hierarchical-heads", type=Path, required=True)
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / "progress.json"
    atomic_json({"status": "loading"}, progress)
    reference, experts, routed_v4 = load_paired_prediction_tables(
        two_m_predictions=args.two_m_predictions,
        routed_v4_predictions=args.routed_v4_predictions,
    )
    atomic_json({"status": "building_graphs", "rows": len(reference)}, progress)
    graph_report = build_external_graph_cache(
        reference,
        args.graph_cache,
        workers=args.workers,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    atomic_json(
        {
            "status": "extracting_embeddings",
            "accepted_rows": graph_report["accepted_rows"],
            "device": device,
        },
        progress,
    )
    source_idx, primary, augmented = extract_external_schnet_embeddings(
        cache_dir=args.graph_cache,
        primary_checkpoint=args.primary_checkpoint,
        augmented_checkpoint=args.augmented_checkpoint,
        device=device,
        batch_size=args.batch_size,
    )
    atomic_json({"status": "evaluating", "rows": len(source_idx)}, progress)
    dense_paths = [
        args.dense_gates / f"dense_seed{seed}.pt"
        for seed in (42, 43, 44)
    ]
    metrics, predictions = evaluate_paired_external(
        reference=reference,
        experts=experts,
        routed_v4=routed_v4,
        source_idx=source_idx,
        primary_embeddings=primary,
        augmented_embeddings=augmented,
        dense_gate_paths=dense_paths,
        hierarchical_root=args.hierarchical_heads,
    )
    result = {
        "experiment": "repaired_2m_hierarchical_external_paired",
        "status": "complete",
        "comparison": "same common molecule identities and same accepted ETKDG rows",
        "source_rows": len(reference),
        "aligned_rows": len(source_idx),
        "graph_report": graph_report,
        "metrics": metrics,
        "inputs": {
            "primary_checkpoint_sha256": sha256(args.primary_checkpoint),
            "augmented_checkpoint_sha256": sha256(args.augmented_checkpoint),
        },
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    metrics_path = args.out_dir / "metrics.json"
    predictions_path = args.out_dir / "predictions.csv"
    atomic_json(result, metrics_path)
    temporary = predictions_path.with_name(f".{predictions_path.name}.tmp")
    predictions.to_csv(temporary, index=False)
    os.replace(temporary, predictions_path)
    atomic_json(
        {
            "status": "complete",
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
        },
        progress,
    )
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
