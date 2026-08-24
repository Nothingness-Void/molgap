"""Build compact, identity-checked payloads for conservative 2D/3D fusion."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from .hierarchical_fusion import hierarchical_context


TRAINING_FORMAT = "molgap-conservative-fusion-training-v1"
EXTERNAL_FORMAT = "molgap-conservative-fusion-external-v1"


def sha256_file(path: Path) -> str:
    """Hash a payload without importing graph-model runtime dependencies."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _frozen_2d(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "molgap-three-gps-frozen-2d-test-v1":
        raise ValueError("Frozen 2D payload format differs")
    keys = (
        "source_idx",
        "targets",
        "expert_predictions",
        "dense_prediction",
        "equal_gps7_gps9_prediction",
        "dense_weights",
    )
    arrays = {}
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Frozen 2D payload misses {key}")
        arrays[key] = value.cpu().numpy()
    rows = len(arrays["source_idx"])
    expected = {
        "targets": (rows, 3),
        "expert_predictions": (rows, 3, 3),
        "dense_prediction": (rows, 3),
        "equal_gps7_gps9_prediction": (rows, 3),
        "dense_weights": (rows, 3, 3),
    }
    if len(np.unique(arrays["source_idx"])) != rows:
        raise ValueError("Frozen source_idx values are duplicated")
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Frozen tensor {key} differs from {shape}")
    return arrays


def _embedding_subset(
    directory: Path,
    source_idx: np.ndarray,
    targets: np.ndarray,
    *,
    expected_dim: int = 176,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    paths = sorted(directory.glob("embeddings_*.pt"))
    if len(paths) != 100:
        raise ValueError(f"Expected 100 embedding parts under {directory}")
    maximum = int(source_idx.max())
    destination = np.full(maximum + 1, -1, dtype=np.int64)
    destination[source_idx] = np.arange(len(source_idx), dtype=np.int64)
    output = np.empty((len(source_idx), expected_dim), dtype=np.float32)
    present = np.zeros(len(source_idx), dtype=bool)
    reports = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        part_source = payload["source_idx"].view(-1).numpy().astype(np.int64)
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
        positions = np.full(len(part_source), -1, dtype=np.int64)
        positions[eligible] = destination[part_source[eligible]]
        selected = positions >= 0
        selected_positions = positions[selected]
        if np.any(present[selected_positions]):
            raise ValueError(f"Duplicate requested source_idx in {path}")
        if not np.allclose(
            part_targets[selected], targets[selected_positions], atol=1e-6, rtol=0.0
        ):
            raise ValueError(f"Target alignment differs in {path}")
        output[selected_positions] = part_embedding[selected]
        present[selected_positions] = True
        reports.append(
            {
                "name": path.name,
                "rows": len(part_source),
                "selected_rows": int(selected.sum()),
                "sha256": sha256_file(path),
            }
        )
    return output, present, reports


def _scaffold_split(
    scaffold_group: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(len(scaffold_group))
    first = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train, temporary = next(first.split(rows, groups=scaffold_group))
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    validation_local, test_local = next(
        second.split(temporary, groups=scaffold_group[temporary])
    )
    validation = temporary[validation_local]
    test = temporary[test_local]
    groups = [set(scaffold_group[index]) for index in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("Scaffold split overlaps")
    return train, validation, test


def build_training_payload(
    *,
    frozen_2d_path: Path,
    scaffold_manifest_path: Path,
    primary_embeddings: Path,
    augmented_embeddings: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict:
    frozen = _frozen_2d(frozen_2d_path)
    primary, primary_present, primary_reports = _embedding_subset(
        primary_embeddings, frozen["source_idx"], frozen["targets"]
    )
    augmented, augmented_present, augmented_reports = _embedding_subset(
        augmented_embeddings, frozen["source_idx"], frozen["targets"]
    )
    aligned = primary_present & augmented_present
    if int(aligned.sum()) < int(0.95 * len(aligned)):
        raise RuntimeError("Less than 95% of frozen rows have both 3D embeddings")
    aligned_frozen = {key: value[aligned] for key, value in frozen.items()}
    with np.load(scaffold_manifest_path, allow_pickle=False) as manifest:
        manifest_row = manifest["manifest_row"].astype(np.int64)
        scaffold_group = manifest["scaffold_group"].astype(np.int64)
    if not np.array_equal(manifest_row, np.arange(2_000_000, dtype=np.int64)):
        raise ValueError("Scaffold manifest identity differs")
    aligned_groups = scaffold_group[aligned_frozen["source_idx"].astype(np.int64)]
    train, validation, test = _scaffold_split(aligned_groups)
    context = hierarchical_context(
        aligned_frozen["expert_predictions"],
        aligned_frozen["dense_weights"],
        primary[aligned],
        augmented[aligned],
    )
    payload = {
        "format": TRAINING_FORMAT,
        "source_idx": torch.from_numpy(aligned_frozen["source_idx"].astype(np.int64)),
        "targets": torch.from_numpy(aligned_frozen["targets"].astype(np.float32)),
        "equal_prediction": torch.from_numpy(
            aligned_frozen["equal_gps7_gps9_prediction"].astype(np.float32)
        ),
        "dense_prediction": torch.from_numpy(
            aligned_frozen["dense_prediction"].astype(np.float32)
        ),
        "context": torch.from_numpy(context).half(),
        "train_indices": torch.from_numpy(train.astype(np.int64)),
        "validation_indices": torch.from_numpy(validation.astype(np.int64)),
        "test_indices": torch.from_numpy(test.astype(np.int64)),
    }
    _atomic_torch(payload, output_path)
    report = {
        "format": "molgap-conservative-fusion-training-manifest-v1",
        "status": "accepted",
        "payload": {
            "name": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "rows": len(context),
        "context_dim": context.shape[1],
        "context_storage_dtype": "float16",
        "split": {
            "seed": 42,
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "scaffold_overlap": 0,
        },
        "inputs": {
            "frozen_2d_sha256": sha256_file(frozen_2d_path),
            "scaffold_manifest_sha256": sha256_file(scaffold_manifest_path),
            "primary_parts": primary_reports,
            "augmented_parts": augmented_reports,
        },
        "aligned_source_idx_sha256": hashlib.sha256(
            aligned_frozen["source_idx"].astype(np.int64).tobytes()
        ).hexdigest(),
    }
    _atomic_json(report, manifest_path)
    return report


def build_external_payload(
    *,
    three_gps_predictions: Path,
    accepted_external_predictions: Path,
    dense_gate_paths: list[Path],
    graph_cache: Path,
    primary_checkpoint: Path,
    augmented_checkpoint: Path,
    output_path: Path,
    manifest_path: Path,
    device: str,
    batch_size: int = 128,
) -> dict:
    """Build a compact common/OOD/P8-hard payload from accepted local assets."""
    import pandas as pd

    from .hierarchical_external_eval import (
        extract_external_schnet_embeddings,
        load_dense_ensemble,
    )

    table = pd.read_csv(three_gps_predictions)
    targets = table.loc[:, ["homo", "lumo", "gap"]].to_numpy(np.float32)
    experts = np.stack(
        [
            table.loc[
                :, [f"{expert}_{target}" for target in ("homo", "lumo", "gap")]
            ].to_numpy(np.float32)
            for expert in ("gps7", "gps9", "gps11_160")
        ],
        axis=1,
    )
    dense, weights = load_dense_ensemble(dense_gate_paths, experts)
    equal = experts[:, :2].mean(axis=1)
    source_idx, primary, augmented = extract_external_schnet_embeddings(
        cache_dir=graph_cache,
        primary_checkpoint=primary_checkpoint,
        augmented_checkpoint=augmented_checkpoint,
        device=device,
        batch_size=batch_size,
    )
    if source_idx.min() < 0 or source_idx.max() >= len(table):
        raise ValueError("External graph source_idx falls outside prediction table")
    context = hierarchical_context(
        experts[source_idx], weights[source_idx], primary, augmented
    )

    accepted = pd.read_csv(accepted_external_predictions)
    if accepted.duplicated(["eval_set", "cid"]).any():
        raise ValueError("Accepted external predictions contain duplicate identities")
    indexed = accepted.set_index(["eval_set", "cid"])
    aligned = table.iloc[source_idx].reset_index(drop=True)
    identity = pd.MultiIndex.from_frame(aligned.loc[:, ["eval_set", "cid"]])
    missing = identity.difference(indexed.index)
    if len(missing):
        raise ValueError(f"Accepted external predictions miss {len(missing)} rows")
    accepted_aligned = indexed.loc[identity].reset_index()
    routed = accepted_aligned.loc[
        :, [f"routed_v4_500k_{target}" for target in ("homo", "lumo", "gap")]
    ].to_numpy(np.float32)
    scope = np.where(
        aligned.eval_set.eq("ood1000").to_numpy(),
        1,
        np.where(aligned.eval_set.eq("p8_targeted_hard").to_numpy(), 2, 0),
    ).astype(np.int8)
    if np.any(scope == 0):
        raise ValueError("External payload contains an unknown evaluation scope")

    payload = {
        "format": EXTERNAL_FORMAT,
        "source_idx": torch.from_numpy(source_idx),
        "targets": torch.from_numpy(targets[source_idx]),
        "equal_prediction": torch.from_numpy(equal[source_idx]),
        "dense_prediction": torch.from_numpy(dense[source_idx]),
        "routed_v4_prediction": torch.from_numpy(routed),
        "context": torch.from_numpy(context).half(),
        "scope": torch.from_numpy(scope),
    }
    _atomic_torch(payload, output_path)
    report = {
        "format": "molgap-conservative-fusion-external-manifest-v1",
        "status": "accepted",
        "payload": {
            "name": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "rows": len(source_idx),
        "context_dim": context.shape[1],
        "scope_rows": {
            "all": len(scope),
            "ood1000": int((scope == 1).sum()),
            "p8_targeted_hard": int((scope == 2).sum()),
        },
        "inputs": {
            "three_gps_predictions_sha256": sha256_file(three_gps_predictions),
            "accepted_external_predictions_sha256": sha256_file(
                accepted_external_predictions
            ),
            "dense_gate_sha256": [sha256_file(path) for path in dense_gate_paths],
            "graph_completion_sha256": sha256_file(graph_cache / "completion.json"),
            "primary_checkpoint_sha256": sha256_file(primary_checkpoint),
            "augmented_checkpoint_sha256": sha256_file(augmented_checkpoint),
        },
    }
    _atomic_json(report, manifest_path)
    return report
