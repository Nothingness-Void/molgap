"""Extract aligned seed-42 validation/test predictions for three frozen GPS experts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from molgap.gps import GPSWrapper


EXPERTS = (
    ("gps7", 7, 192),
    ("gps9", 9, 192),
    ("gps11_160", 11, 160),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def split_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.random.RandomState(seed).permutation(n_rows)
    n_train, n_validation = int(0.8 * n_rows), int(0.1 * n_rows)
    return (
        permutation[n_train:n_train + n_validation],
        permutation[n_train + n_validation:],
    )


@torch.inference_mode()
def predict(
    model: GPSWrapper,
    graphs: list,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    subset = [graphs[int(index)] for index in indices]
    chunks = []
    model.eval()
    for batch_number, batch in enumerate(
        DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0),
        start=1,
    ):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            chunks.append(
                model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                ).float().cpu()
            )
        if batch_number % 100 == 0:
            print(
                f"  batches={batch_number} rows={sum(len(chunk) for chunk in chunks):,}",
                flush=True,
            )
    del subset
    return torch.cat(chunks).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--gps7", type=Path, required=True)
    parser.add_argument("--gps9", type=Path, required=True)
    parser.add_argument("--gps11-160", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/DCU device is required")
    checkpoints = {
        "gps7": args.gps7,
        "gps9": args.gps9,
        "gps11_160": args.gps11_160,
    }
    for path in (args.graphs, *checkpoints.values()):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    graphs = torch.load(args.graphs, weights_only=False)
    validation_indices, test_indices = split_indices(len(graphs), args.split_seed)
    if np.intersect1d(validation_indices, test_indices).size:
        raise RuntimeError("Validation and test indices overlap")
    source_idx = {
        "validation": np.asarray(
            [int(graphs[int(index)].source_idx.view(-1)[0]) for index in validation_indices],
            dtype=np.int64,
        ),
        "test": np.asarray(
            [int(graphs[int(index)].source_idx.view(-1)[0]) for index in test_indices],
            dtype=np.int64,
        ),
    }
    targets = {
        "validation": np.stack(
            [graphs[int(index)].y.view(-1).numpy() for index in validation_indices]
        ).astype(np.float32),
        "test": np.stack(
            [graphs[int(index)].y.view(-1).numpy() for index in test_indices]
        ).astype(np.float32),
    }
    device = torch.device("cuda")
    predictions = {
        "validation": np.empty((len(validation_indices), len(EXPERTS), 3), np.float32),
        "test": np.empty((len(test_indices), len(EXPERTS), 3), np.float32),
    }
    progress_path = args.manifest_out.with_name("progress.json")
    atomic_json(
        {
            "status": "graphs_loaded",
            "n_graphs": len(graphs),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
        },
        progress_path,
    )
    for expert_index, (name, layers, hidden) in enumerate(EXPERTS):
        model = GPSWrapper(
            hidden_channels=hidden,
            num_layers=layers,
            num_heads=4,
            dropout=0.05,
        ).to(device)
        model.load_state_dict(
            torch.load(checkpoints[name], map_location=device, weights_only=True),
            strict=True,
        )
        for split_name, indices in (
            ("validation", validation_indices),
            ("test", test_indices),
        ):
            print(f"{name}: {split_name} rows={len(indices):,}", flush=True)
            predictions[split_name][:, expert_index] = predict(
                model, graphs, indices, device, args.batch_size
            )
        del model
        torch.cuda.empty_cache()
        atomic_json(
            {
                "status": "expert_complete",
                "expert": name,
                "expert_index": expert_index,
                "completed_experts": expert_index + 1,
            },
            progress_path,
        )

    payload = {
        "format": "molgap-three-gps-holdout-v1",
        "split_seed": args.split_seed,
        "expert_names": [name for name, _, _ in EXPERTS],
        "target_names": ["homo", "lumo", "gap"],
        "validation": {
            "source_idx": torch.from_numpy(source_idx["validation"]),
            "targets": torch.from_numpy(targets["validation"]),
            "predictions": torch.from_numpy(predictions["validation"]),
        },
        "test": {
            "source_idx": torch.from_numpy(source_idx["test"]),
            "targets": torch.from_numpy(targets["test"]),
            "predictions": torch.from_numpy(predictions["test"]),
        },
    }
    atomic_torch_save(payload, args.out)
    result = {
        "format": payload["format"],
        "complete": True,
        "graph_path": str(args.graphs),
        "graph_sha256": sha256(args.graphs),
        "graph_rows": len(graphs),
        "split_seed": args.split_seed,
        "validation_rows": len(validation_indices),
        "test_rows": len(test_indices),
        "experts": [
            {
                "name": name,
                "layers": layers,
                "hidden_channels": hidden,
                "checkpoint": str(checkpoints[name]),
                "checkpoint_sha256": sha256(checkpoints[name]),
            }
            for name, layers, hidden in EXPERTS
        ],
        "payload": str(args.out),
        "payload_bytes": args.out.stat().st_size,
        "payload_sha256": sha256(args.out),
        "finite": bool(
            np.isfinite(predictions["validation"]).all()
            and np.isfinite(predictions["test"]).all()
            and np.isfinite(targets["validation"]).all()
            and np.isfinite(targets["test"]).all()
        ),
    }
    atomic_json(result, args.manifest_out)
    atomic_json({"status": "complete", **result}, progress_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

