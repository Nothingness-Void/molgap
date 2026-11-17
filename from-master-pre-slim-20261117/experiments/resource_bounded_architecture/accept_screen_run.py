"""Accept one completed resource-bounded architecture screen run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from molgap.structural_encoding import sha256


def _atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=[
            "gps",
            "structural_gps",
            "normalized_structural_gps",
            "gated_structural_gps",
            "edge_state_structural_gps",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "model": args.run_dir / "model.pt",
        "last_checkpoint": args.run_dir / "training_state.pt",
        "metrics": args.run_dir / "metrics.json",
        "predictions": args.run_dir / "test_predictions.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    checkpoint = torch.load(paths["last_checkpoint"], map_location="cpu", weights_only=False)
    predictions = torch.load(paths["predictions"], map_location="cpu", weights_only=False)
    target_mode = metrics.get("target_mode", "all")
    if target_mode not in {"all", "gap"}:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    split_hash = sha256(args.split_csv)
    if metrics.get("kind") != args.kind or checkpoint.get("kind") != args.kind:
        raise ValueError("Run kind differs across artifacts")
    if metrics.get("seed") != args.seed or checkpoint.get("seed") != args.seed:
        raise ValueError("Run seed differs across artifacts")
    if checkpoint.get("target_mode", "all") != target_mode:
        raise ValueError("Checkpoint target mode differs")
    if predictions.get("target_mode", "all") != target_mode:
        raise ValueError("Prediction target mode differs")
    if metrics.get("split_contract", {}).get("sha256") != split_hash:
        raise ValueError("Metrics split hash differs")
    if checkpoint.get("split_contract", {}).get("sha256") != split_hash:
        raise ValueError("Checkpoint split hash differs")
    if predictions.get("split_contract", {}).get("sha256") != split_hash:
        raise ValueError("Prediction split hash differs")
    expected_rows = int(metrics["split_contract"]["rows"]["test"])
    source_idx = predictions.get("source_idx")
    targets = predictions.get("targets")
    predicted = predictions.get("predictions")
    if not all(isinstance(value, torch.Tensor) for value in (source_idx, targets, predicted)):
        raise TypeError("Prediction payload tensors are missing")
    expected_targets = 1 if target_mode == "gap" else 3
    if tuple(targets.shape) != (expected_rows, expected_targets) or predicted.shape != targets.shape:
        raise ValueError("Prediction payload shape differs from test split")
    if source_idx.numel() != expected_rows or source_idx.unique().numel() != expected_rows:
        raise ValueError("Prediction source_idx count or uniqueness differs")
    if not torch.isfinite(targets).all() or not torch.isfinite(predicted).all():
        raise ValueError("Prediction payload contains non-finite values")
    if int(checkpoint.get("next_epoch", 0)) <= 0 or int(metrics.get("best_epoch", -1)) < 0:
        raise ValueError("Training did not produce an accepted epoch")
    graph_hash = sha256(args.graph)
    if args.kind in {
        "structural_gps",
        "normalized_structural_gps",
        "gated_structural_gps",
        "edge_state_structural_gps",
    } and metrics.get("graph_contract", {}).get("sha256") != graph_hash:
        raise ValueError("Structural GPS graph contract differs")

    report = {
        "status": "accepted",
        "kind": args.kind,
        "seed": args.seed,
        "target_mode": target_mode,
        "graph_sha256": graph_hash,
        "split_sha256": split_hash,
        "test_rows": expected_rows,
        "best_epoch": int(metrics["best_epoch"]),
        "best_validation_average_mae_eV": float(metrics["best_val_mae"]),
        "test_average_mae_eV": float(metrics["test_metrics"]["average"]["mae"]),
        "test_gap_mae_eV": float(metrics["test_metrics"]["Gap"]["mae"]),
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
    }
    output = args.run_dir / "completion_manifest.json"
    _atomic_json(report, output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
