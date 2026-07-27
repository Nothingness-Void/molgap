"""Train a resumable bounded residual head over three frozen GPS embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.gps import GPSWrapper
from molgap.hierarchical_fusion import (
    HierarchicalFusionConfig,
    fit_hierarchical_fusion,
    predict_hierarchical_fusion,
)
from molgap.multi2d_router_fusion import metric_block


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_chunked(
    directory: Path,
    *,
    name: str,
    rows: int,
    hidden_channels: int,
) -> dict[str, torch.Tensor | str]:
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("format") != "molgap-gps-embedding-parts-v1"
        or not completion.get("complete")
        or completion.get("name") != name
        or int(completion.get("rows", -1)) != rows
        or int(completion.get("hidden_channels", -1)) != hidden_channels
    ):
        raise ValueError(f"Invalid completion contract for {directory}")
    embeddings, predictions, targets, source_indices = [], [], [], []
    expected_start = 0
    for report in completion["artifacts"]:
        path = Path(report["path"])
        if (
            int(report["start"]) != expected_start
            or int(report["end"]) - int(report["start"]) != int(report["rows"])
            or report["sha256"] != sha256(path)
        ):
            raise ValueError(f"Invalid part report for {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != "molgap-gps-embedding-part-v1"
            or payload.get("name") != name
            or payload.get("model_sha256") != completion["model_sha256"]
            or payload["embeddings"].shape
            != (int(report["rows"]), hidden_channels)
            or payload["predictions"].shape != (int(report["rows"]), 3)
            or payload["targets"].shape != (int(report["rows"]), 3)
            or not torch.equal(
                payload["source_idx"],
                torch.arange(
                    int(report["start"]),
                    int(report["end"]),
                    dtype=torch.long,
                ),
            )
        ):
            raise ValueError(f"Invalid embedding payload {path}")
        embeddings.append(payload["embeddings"])
        predictions.append(payload["predictions"])
        targets.append(payload["targets"])
        source_indices.append(payload["source_idx"])
        expected_start = int(report["end"])
    result = {
        "embeddings": torch.cat(embeddings),
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "source_idx": torch.cat(source_indices),
        "model_sha256": completion["model_sha256"],
        "completion_sha256": sha256(completion_path),
    }
    if (
        expected_start != rows
        or result["embeddings"].shape != (rows, hidden_channels)
        or not torch.equal(result["source_idx"], torch.arange(rows))
        or not torch.isfinite(result["embeddings"]).all()
        or not torch.isfinite(result["predictions"]).all()
        or not torch.isfinite(result["targets"]).all()
    ):
        raise ValueError(f"Combined payload failed for {name}")
    return result


@torch.inference_mode()
def gps11_predictions(
    embeddings: torch.Tensor,
    checkpoint: Path,
    device: torch.device,
    batch_size: int = 32768,
) -> torch.Tensor:
    model = GPSWrapper(
        hidden_channels=160,
        num_layers=11,
        num_heads=4,
        dropout=0.05,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    predictions = []
    for start in range(0, len(embeddings), batch_size):
        predictions.append(
            model.head(
                embeddings[start : start + batch_size].float().to(device)
            )
            .float()
            .cpu()
        )
    return torch.cat(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gps7-dir", type=Path, required=True)
    parser.add_argument("--gps9-dir", type=Path, required=True)
    parser.add_argument("--gps11-embeddings", type=Path, required=True)
    parser.add_argument("--gps11-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.json"
    atomic_json({"status": "loading_inputs"}, progress_path)
    gps7 = load_chunked(
        args.gps7_dir,
        name="gps7",
        rows=args.rows,
        hidden_channels=192,
    )
    gps9 = load_chunked(
        args.gps9_dir,
        name="gps9",
        rows=args.rows,
        hidden_channels=192,
    )
    if not (
        torch.equal(gps7["source_idx"], gps9["source_idx"])
        and torch.equal(gps7["targets"], gps9["targets"])
    ):
        raise ValueError("GPS7/GPS9 source or target alignment differs")
    gps11_payload = torch.load(
        args.gps11_embeddings,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    gps11_embedding = gps11_payload["embeddings"]
    gps11_source = gps11_payload["source_idx"]
    if (
        gps11_embedding.shape != (args.rows, 160)
        or not torch.equal(gps11_source, gps7["source_idx"])
        or not torch.isfinite(gps11_embedding).all()
    ):
        raise ValueError("GPS11 embedding alignment differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA/DCU device is required")
    prediction11 = gps11_predictions(
        gps11_embedding,
        args.gps11_model,
        device,
    )
    base = (gps7["predictions"] + gps9["predictions"]) / 2
    context = torch.cat(
        (
            gps7["predictions"],
            gps9["predictions"],
            prediction11,
            gps7["embeddings"].float(),
            gps9["embeddings"].float(),
            gps11_embedding.float(),
        ),
        dim=1,
    ).numpy()
    targets = gps7["targets"].numpy()
    base = base.numpy()
    permutation = np.random.RandomState(42).permutation(args.rows)
    n_train, n_validation = int(0.8 * args.rows), int(0.1 * args.rows)
    train = permutation[:n_train]
    validation = permutation[n_train : n_train + n_validation]
    test = permutation[n_train + n_validation :]
    contract = {
        "gps7_completion_sha256": gps7["completion_sha256"],
        "gps9_completion_sha256": gps9["completion_sha256"],
        "gps11_embeddings_sha256": sha256(args.gps11_embeddings),
        "gps11_model_sha256": sha256(args.gps11_model),
        "rows": args.rows,
        "split_seed": 42,
        "identity": "equal_gps7_gps9",
    }
    contract_id = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config = HierarchicalFusionConfig(
        hidden_channels=128,
        correction_scale_eV=0.10,
        batch_size=8192,
        epochs=60,
        patience=10,
        seed=args.seed,
    )
    checkpoint_path = args.out_dir / "last.pt"
    model, training = fit_hierarchical_fusion(
        base,
        context,
        targets,
        train,
        validation,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
        progress_path=progress_path,
        resume=checkpoint_path.is_file(),
        contract_id=contract_id,
    )
    prediction, correction = predict_hierarchical_fusion(
        model,
        base[test],
        context[test],
    )
    best_path = args.out_dir / "best.pt"
    atomic_torch(
        {
            "kind": "three_gps_embedding_bounded_residual",
            "seed": args.seed,
            "contract": contract,
            "contract_id": contract_id,
            "config": training["config"],
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
        },
        best_path,
    )
    metrics = {
        "experiment": "repaired_2m_three_gps_embedding_bounded_residual",
        "status": "complete",
        "seed": args.seed,
        "rows": args.rows,
        "split": {
            "seed": 42,
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "identity": "equal_gps7_gps9",
        "context_dim": context.shape[1],
        "correction_scale_eV": 0.10,
        "base_test": metric_block(targets[test], base[test]),
        "fusion_test": metric_block(targets[test], prediction),
        "correction": {
            "mean_abs_eV": float(np.abs(correction).mean()),
            "p95_abs_eV": float(np.quantile(np.abs(correction), 0.95)),
            "max_abs_eV": float(np.abs(correction).max()),
        },
        "training": training,
        "artifacts": {
            "best": str(best_path),
            "best_sha256": sha256(best_path),
            "last": str(checkpoint_path),
            "last_sha256": sha256(checkpoint_path),
        },
        "contract": contract,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(metrics, args.out_dir / "metrics.json")
    atomic_json(
        {
            "status": "complete",
            "seed": args.seed,
            "metrics": str(args.out_dir / "metrics.json"),
        },
        progress_path,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
