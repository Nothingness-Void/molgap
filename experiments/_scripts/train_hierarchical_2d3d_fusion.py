"""Train bounded dual-SchNet corrections over a frozen three-GPS 2D system."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from molgap.hierarchical_fusion import (
    HierarchicalFusionConfig,
    fit_hierarchical_fusion,
    hierarchical_context,
    predict_hierarchical_fusion,
)
from molgap.multi2d_router_fusion import metric_block
from molgap.router import paired_bootstrap_mean


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


def load_frozen_2d(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "molgap-three-gps-frozen-2d-test-v1":
        raise ValueError(f"Unsupported frozen 2D payload: {payload.get('format')}")
    result = {}
    for key in (
        "source_idx",
        "targets",
        "expert_predictions",
        "dense_prediction",
        "equal_gps7_gps9_prediction",
        "dense_weights",
        "hard_prediction",
        "hard_selection",
    ):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Frozen 2D payload misses tensor {key}")
        result[key] = value.numpy()
    rows = len(result["source_idx"])
    expected = {
        "targets": (rows, 3),
        "expert_predictions": (rows, 3, 3),
        "dense_prediction": (rows, 3),
        "equal_gps7_gps9_prediction": (rows, 3),
        "dense_weights": (rows, 3, 3),
        "hard_prediction": (rows, 3),
        "hard_selection": (rows, 3),
    }
    if len(np.unique(result["source_idx"])) != rows:
        raise ValueError("Frozen 2D source_idx values are duplicated")
    for key, shape in expected.items():
        if result[key].shape != shape:
            raise ValueError(f"{key} has shape {result[key].shape}, expected {shape}")
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    return result


def load_embedding_subset(
    directory: Path,
    source_idx: np.ndarray,
    targets: np.ndarray,
    *,
    expected_dim: int = 176,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    paths = sorted(directory.glob("embeddings_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No embedding parts under {directory}")
    maximum = int(source_idx.max())
    destination = np.full(maximum + 1, -1, dtype=np.int64)
    destination[source_idx] = np.arange(len(source_idx), dtype=np.int64)
    output = np.empty((len(source_idx), expected_dim), dtype=np.float32)
    present = np.zeros(len(source_idx), dtype=bool)
    reports = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"source_idx", "embeddings", "targets"}
        if missing := required.difference(payload):
            raise ValueError(f"{path} misses {sorted(missing)}")
        part_source = payload["source_idx"].numpy().astype(np.int64)
        part_embedding = payload["embeddings"].numpy().astype(np.float32)
        part_targets = payload["targets"].numpy().astype(np.float32)
        if (
            part_embedding.shape != (len(part_source), expected_dim)
            or part_targets.shape != (len(part_source), 3)
            or len(np.unique(part_source)) != len(part_source)
            or not np.isfinite(part_embedding).all()
            or not np.isfinite(part_targets).all()
        ):
            raise ValueError(f"Invalid embedding part {path}")
        eligible = part_source <= maximum
        part_positions = np.full(len(part_source), -1, dtype=np.int64)
        part_positions[eligible] = destination[part_source[eligible]]
        selected = part_positions >= 0
        positions = part_positions[selected]
        if np.any(present[positions]):
            raise ValueError(f"Duplicate requested source_idx in {path}")
        if not np.allclose(
            part_targets[selected],
            targets[positions],
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"Target alignment differs in {path}")
        output[positions] = part_embedding[selected]
        present[positions] = True
        reports.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(part_source),
                "selected_rows": int(selected.sum()),
            }
        )
    return output, present, reports


def scaffold_three_way_split(
    scaffolds: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(len(scaffolds))
    first = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train, temporary = next(first.split(rows, groups=scaffolds))
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    validation_local, test_local = next(
        second.split(temporary, groups=scaffolds[temporary])
    )
    validation = temporary[validation_local]
    test = temporary[test_local]
    split_groups = [
        set(scaffolds[index])
        for index in (train, validation, test)
    ]
    if (
        split_groups[0].intersection(split_groups[1])
        or split_groups[0].intersection(split_groups[2])
        or split_groups[1].intersection(split_groups[2])
    ):
        raise RuntimeError("Hierarchical Fusion scaffold split overlaps")
    return train, validation, test


def paired_delta(
    targets: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    result = {}
    names = ("homo", "lumo", "gap", "average")
    for index, name in enumerate(names):
        if name == "average":
            delta = np.abs(candidate - targets).mean(axis=1) - np.abs(
                base - targets
            ).mean(axis=1)
        else:
            delta = np.abs(candidate[:, index] - targets[:, index]) - np.abs(
                base[:, index] - targets[:, index]
            )
        result[name] = paired_bootstrap_mean(
            delta,
            n_bootstrap=10_000,
            seed=42 + index,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-2d", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-embeddings", type=Path, required=True)
    parser.add_argument("--augmented-embeddings", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "loading"}, args.out_dir / "progress.json")

    frozen = load_frozen_2d(args.frozen_2d)
    primary, primary_present, primary_reports = load_embedding_subset(
        args.primary_embeddings,
        frozen["source_idx"],
        frozen["targets"],
    )
    augmented, augmented_present, augmented_reports = load_embedding_subset(
        args.augmented_embeddings,
        frozen["source_idx"],
        frozen["targets"],
    )
    aligned = primary_present & augmented_present
    if aligned.sum() < 0.95 * len(aligned):
        raise RuntimeError(
            f"Only {aligned.sum():,}/{len(aligned):,} rows have both 3D views"
        )
    frozen = {key: value[aligned] for key, value in frozen.items()}
    primary = primary[aligned]
    augmented = augmented[aligned]
    manifest = pd.read_parquet(args.manifest).set_index("manifest_row", drop=False)
    rows = manifest.loc[frozen["source_idx"]].reset_index(drop=True)
    if not np.array_equal(
        rows.manifest_row.to_numpy(np.int64),
        frozen["source_idx"].astype(np.int64),
    ):
        raise ValueError("Manifest does not align to frozen 2D source_idx")
    scaffolds = rows.scaffold.fillna("").astype(str).to_numpy()
    train, validation, test = scaffold_three_way_split(scaffolds)
    context = hierarchical_context(
        frozen["expert_predictions"],
        frozen["dense_weights"],
        primary,
        augmented,
    )
    atomic_json(
        {
            "status": "training",
            "aligned_rows": int(aligned.sum()),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        args.out_dir / "progress.json",
    )

    result = {}
    base_keys = {
        "equal_gps7_gps9": "equal_gps7_gps9_prediction",
        "dense": "dense_prediction",
    }
    for base_name, base_key in base_keys.items():
        base = frozen[base_key].astype(np.float32)
        predictions, corrections, training = [], [], {}
        for seed in (42, 43, 44):
            config = HierarchicalFusionConfig(seed=seed)
            model, report = fit_hierarchical_fusion(
                base,
                context,
                frozen["targets"],
                train,
                validation,
                config=config,
                device=args.device,
            )
            prediction, correction = predict_hierarchical_fusion(
                model,
                base[test],
                context[test],
            )
            predictions.append(prediction)
            corrections.append(correction)
            training[str(seed)] = report
            atomic_torch(
                {
                    "kind": "hierarchical_2d_3d_bounded_residual",
                    "base": base_name,
                    "seed": seed,
                    "config": report["config"],
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                },
                args.out_dir / f"{base_name}_seed{seed}.pt",
            )
        prediction = np.mean(predictions, axis=0)
        correction = np.mean(corrections, axis=0)
        base_test = base[test]
        targets_test = frozen["targets"][test]
        result[base_name] = {
            "base_metrics": metric_block(targets_test, base_test),
            "fusion_metrics": metric_block(targets_test, prediction),
            "fusion_minus_base": paired_delta(
                targets_test,
                base_test,
                prediction,
            ),
            "correction": {
                "mean_abs_eV": float(np.abs(correction).mean()),
                "p95_abs_eV": float(np.quantile(np.abs(correction), 0.95)),
                "max_abs_eV": float(np.abs(correction).max()),
            },
            "training": training,
        }

    metrics = {
        "experiment": "repaired_2m_hierarchical_three_gps_dual_schnet_fusion",
        "status": "internal_gate_complete_not_production",
        "aligned_rows": int(aligned.sum()),
        "split": {
            "seed": 42,
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "scaffold_overlap": 0,
        },
        "correction_scale_eV": 0.10,
        "results": result,
        "inputs": {
            "frozen_2d": {
                "path": str(args.frozen_2d),
                "sha256": sha256(args.frozen_2d),
            },
            "manifest": {
                "path": str(args.manifest),
                "sha256": sha256(args.manifest),
            },
            "primary_embedding_parts": primary_reports,
            "augmented_embedding_parts": augmented_reports,
        },
        "external_evaluation_required": True,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(metrics, args.out_dir / "metrics.json")
    atomic_json(
        {"status": "complete", "metrics": str(args.out_dir / "metrics.json")},
        args.out_dir / "progress.json",
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
