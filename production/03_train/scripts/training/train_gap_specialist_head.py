"""Train a frozen-encoder GPS head with B3LYP rehearsal and PCQM Gap labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.gap_specialization import train_gap_specialist_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--b3-embeddings", type=Path, required=True)
    parser.add_argument("--b3-graphs", type=Path, required=True)
    parser.add_argument("--pcqm-embeddings", type=Path, required=True)
    parser.add_argument("--pcqm-table", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--b3-train-rows", type=int, default=200_000)
    parser.add_argument("--b3-validation-rows", type=int, default=20_000)
    parser.add_argument("--pcqm-weight", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaffold-workers", type=int, default=8)
    args = parser.parse_args()
    result = train_gap_specialist_head(
        base_model_path=args.base_model,
        b3_embedding_manifest=args.b3_embeddings,
        b3_graph_path=args.b3_graphs,
        pcqm_embedding_manifest=args.pcqm_embeddings,
        pcqm_table_path=args.pcqm_table,
        run_dir=args.run_dir,
        model_out=args.model_out,
        b3_train_rows=args.b3_train_rows,
        b3_validation_rows=args.b3_validation_rows,
        pcqm_weight=args.pcqm_weight,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        scaffold_workers=args.scaffold_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
