"""Export resumable full-corpus GPS embeddings, predictions, and targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from molgap.gps import GPSWrapper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def accepted_part(
    path: Path,
    report_path: Path,
    *,
    start: int,
    end: int,
    hidden_channels: int,
    model_sha256: str,
) -> dict | None:
    if not path.is_file() or not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "start": start,
        "end": end,
        "rows": end - start,
        "hidden_channels": hidden_channels,
        "model_sha256": model_sha256,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return None
    if report.get("sha256") != sha256(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["embeddings"].shape != (end - start, hidden_channels)
        or payload["predictions"].shape != (end - start, 3)
        or payload["targets"].shape != (end - start, 3)
        or not torch.equal(
            payload["source_idx"],
            torch.arange(start, end, dtype=torch.long),
        )
        or not torch.isfinite(payload["embeddings"]).all()
        or not torch.isfinite(payload["predictions"]).all()
        or not torch.isfinite(payload["targets"]).all()
    ):
        return None
    return report


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden-channels", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--part-rows", type=int, default=50_000)
    parser.add_argument("--expected-rows", type=int, default=2_000_000)
    parser.add_argument("--graph-sha256", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/DCU device is required")
    if args.part_rows <= 0:
        raise ValueError("part-rows must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.json"
    model_sha256 = sha256(args.model)
    atomic_json({"status": "loading_graphs"}, progress_path)
    graphs = torch.load(args.graphs, map_location="cpu", weights_only=False)
    if len(graphs) != args.expected_rows:
        raise ValueError(
            f"Graph rows {len(graphs):,} differ from {args.expected_rows:,}"
        )
    device = torch.device("cuda")
    model = GPSWrapper(
        hidden_channels=args.hidden_channels,
        num_layers=args.layers,
        num_heads=4,
        dropout=0.05,
    ).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    reports = []
    for part, start in enumerate(range(0, len(graphs), args.part_rows)):
        end = min(start + args.part_rows, len(graphs))
        path = args.out_dir / f"part-{part:03d}.pt"
        report_path = args.out_dir / f"part-{part:03d}.json"
        report = accepted_part(
            path,
            report_path,
            start=start,
            end=end,
            hidden_channels=args.hidden_channels,
            model_sha256=model_sha256,
        )
        if report is not None:
            reports.append(report)
            continue
        embeddings, predictions, targets, source_indices = [], [], [], []
        for batch in DataLoader(
            graphs[start:end],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ):
            batch = batch.to(device)
            with torch.amp.autocast("cuda"):
                embedding = model.encode(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                prediction = model.head(embedding)
            embeddings.append(embedding.to(torch.float16).cpu())
            predictions.append(prediction.float().cpu())
            targets.append(batch.y.view(-1, 3).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())
        payload = {
            "format": "molgap-gps-embedding-part-v1",
            "name": args.name,
            "embeddings": torch.cat(embeddings),
            "predictions": torch.cat(predictions),
            "targets": torch.cat(targets),
            "source_idx": torch.cat(source_indices),
            "model_sha256": model_sha256,
            "graph_sha256": args.graph_sha256,
        }
        expected_source = torch.arange(start, end, dtype=torch.long)
        if (
            not torch.equal(payload["source_idx"], expected_source)
            or not torch.isfinite(payload["embeddings"]).all()
            or not torch.isfinite(payload["predictions"]).all()
            or not torch.isfinite(payload["targets"]).all()
        ):
            raise RuntimeError(f"Invalid embedding output for rows {start}:{end}")
        atomic_torch(payload, path)
        report = {
            "part": part,
            "start": start,
            "end": end,
            "rows": end - start,
            "hidden_channels": args.hidden_channels,
            "model_sha256": model_sha256,
            "graph_sha256": args.graph_sha256,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        atomic_json(report, report_path)
        reports.append(report)
        atomic_json(
            {
                "status": "running",
                "completed_parts": len(reports),
                "total_parts": (len(graphs) + args.part_rows - 1)
                // args.part_rows,
                "completed_rows": sum(item["rows"] for item in reports),
            },
            progress_path,
        )
        print(
            f"{args.name} part {part + 1}: rows={start:,}:{end:,}",
            flush=True,
        )
    completion = {
        "format": "molgap-gps-embedding-parts-v1",
        "complete": True,
        "name": args.name,
        "rows": len(graphs),
        "parts": len(reports),
        "hidden_channels": args.hidden_channels,
        "model": str(args.model),
        "model_sha256": model_sha256,
        "graph": str(args.graphs),
        "graph_sha256": args.graph_sha256,
        "artifacts": reports,
    }
    atomic_json(completion, args.out_dir / "completion.json")
    atomic_json({"status": "complete", **completion}, progress_path)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
