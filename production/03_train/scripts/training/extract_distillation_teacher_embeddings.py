"""Extract one frozen GPS teacher into durable aligned embedding chunks."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from molgap.distillation import extract_gps_embedding_parts
from molgap.gps import GPSWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, choices=(7, 9), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; loading {args.graphs}", flush=True)
    graphs = torch.load(args.graphs, map_location="cpu", weights_only=False)
    if args.max_rows is not None:
        if args.max_rows <= 0 or args.max_rows > len(graphs):
            raise ValueError("--max-rows must fall within the graph cache")
        graphs = graphs[:args.max_rows]
        print(f"embedding prefix rows={len(graphs):,}", flush=True)
    model = GPSWrapper(
        hidden_channels=192,
        num_layers=args.num_layers,
        num_heads=4,
        dropout=0.05,
    )
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    manifest = extract_gps_embedding_parts(
        model.to(device),
        graphs,
        model_path=args.model,
        out_dir=args.out_dir,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    print(
        f"complete={manifest['complete']} rows={manifest['rows']:,} "
        f"parts={len(manifest['parts'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
