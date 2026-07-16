"""Train one from-scratch dual-2D static candidate expert on the frozen 30k pilot split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from molgap.constants import RESULTS_DIR
from molgap.gps import GPSWrapper
from molgap.utils import ensure_dirs
from molgap.dual2d_static_candidate.local_gine import LocalGINEExpert
from molgap.dual2d_static_candidate.training import train_expert


OUT_DIR = RESULTS_DIR / "phase8" / "dual2d_static_candidate"


def make_model(kind: str):
    if kind == "local":
        return LocalGINEExpert()
    if kind == "global":
        return GPSWrapper(
            hidden_channels=192, num_layers=9, num_heads=4,
            dropout=0.05, pooling="mean_max",
        )
    raise ValueError(kind)


def sampling_weight(kind: str, source: str) -> float:
    weights = {
        "local": {
            "random": 1.0, "descriptor_diverse": 1.3,
            "narrow_conjugated": 1.2, "flexible": 1.0, "large_heteroatom": 1.5,
        },
        "global": {
            "random": 1.0, "descriptor_diverse": 1.3,
            "narrow_conjugated": 2.5, "flexible": 1.0, "large_heteroatom": 1.5,
        },
    }
    return weights[kind][source]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["local", "global"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    defaults = {
        "local": {"epochs": 40, "batch_size": 128, "lr": 2e-4},
        "global": {"epochs": 40, "batch_size": 256, "lr": 2e-4},
    }[args.kind]
    epochs = args.epochs or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    ensure_dirs(OUT_DIR / f"expert_{args.kind}")
    table = pd.read_parquet(OUT_DIR / "pilot_30k.parquet")
    if args.max_train_samples:
        train = table[table.split == "train"].sample(
            n=min(args.max_train_samples, (table.split == "train").sum()),
            random_state=args.seed,
        )
        validation = table[table.split == "validation"].sample(
            n=min(max(250, args.max_train_samples // 5), (table.split == "validation").sum()),
            random_state=args.seed,
        )
        table = pd.concat([train, validation], ignore_index=True)
    graph_path = OUT_DIR / "pilot_30k_graphs_2d.pt"
    graphs = torch.load(graph_path, weights_only=False)
    needed = set(table.source_idx.astype(int))
    graph_by_index = {
        int(graph.source_idx.item()): graph
        for graph in graphs
        if int(graph.source_idx.item()) in needed
    }
    del graphs
    available = table.source_idx.isin(graph_by_index)
    table = table.loc[available].copy()
    source_lookup = dict(zip(table.source_idx.astype(int), table.sampling_source))
    subsets = {
        split: [graph_by_index[int(index)] for index in part.source_idx]
        for split, part in table.groupby("split")
    }
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.kind).to(device)
    trained = train_expert(
        kind=args.kind,
        model=model,
        train_graphs=subsets["train"],
        validation_graphs=subsets["validation"],
        source_lookup=source_lookup,
        sample_weight=lambda source: sampling_weight(args.kind, source),
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=defaults["lr"],
        seed=args.seed,
    )
    expert_dir = OUT_DIR / f"expert_{args.kind}"
    checkpoint = expert_dir / f"seed{args.seed}.pt"
    torch.save(trained.pop("state_dict"), checkpoint)
    result = {
        "experiment": "dual-2D static candidate 30k from-scratch expert pilot",
        "kind": args.kind,
        "seed": args.seed,
        "random_initialization": True,
        "old_checkpoint_loaded": False,
        "n_available": len(table),
        "split_counts": table.split.value_counts().to_dict(),
        "epochs_cap": epochs,
        "batch_size": batch_size,
        "max_train_samples": args.max_train_samples,
        "learning_rate": defaults["lr"],
        "checkpoint": str(checkpoint),
        **trained,
    }
    (expert_dir / f"seed{args.seed}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
