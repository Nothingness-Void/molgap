"""Repair PyG-batched validation identities without executing a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


CACHE_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
CANDIDATES = (
    "ogb_distance_angle_triangle_edge_state_graph_state9",
    "ogb_distance_angle_vector_state_triangle_edge_state_graph_state9",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_validation_identity(cache_root: Path):
    import torch

    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("aggregate_sha256") != CACHE_SHA:
        raise RuntimeError("geometry cache identity changed")
    row_ids = []
    targets = []
    shard_hashes = {}
    for shard in manifest.get("shards", []):
        if shard.get("role") != "validation":
            continue
        path = cache_root / shard["file"]
        actual_sha = sha256_file(path)
        if actual_sha != shard.get("sha256"):
            raise RuntimeError(f"validation shard hash changed: {path.name}")
        shard_hashes[path.name] = actual_sha
        graphs = load_torch(path)
        if len(graphs) != int(shard["graph_count"]):
            raise RuntimeError(f"validation shard count changed: {path.name}")
        for graph in graphs:
            row_ids.append(int(graph.row_index.view(-1)[0]))
            targets.append(float(graph.y.view(-1)[0]))
    if len(row_ids) != 10_000 or len(set(row_ids)) != 10_000:
        raise RuntimeError("cache validation identities are not 10,000 unique rows")
    return (
        torch.tensor(row_ids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32),
        shard_hashes,
    )


def repair(source_root: Path, cache_root: Path, output_root: Path) -> dict:
    import torch

    source_root = source_root.resolve()
    cache_root = cache_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root or output_root.exists():
        raise RuntimeError("repair output must be a new directory")
    completion = json.loads((source_root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise RuntimeError("source result is incomplete")
    if completion.get("geometry_cache_aggregate_sha256") != CACHE_SHA:
        raise RuntimeError("source result cache identity changed")

    correct_rows, correct_targets, shard_hashes = load_validation_identity(cache_root)
    shutil.copytree(source_root, output_root)
    repaired_runs = []
    artifact_changes = {}
    for candidate in CANDIDATES:
        result_root = output_root / "results" / candidate
        metrics_path = result_root / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        artifacts = metrics["artifacts"]
        payload_path = output_root / artifacts["validation_payload"]
        csv_path = output_root / artifacts["validation_csv"]
        payload = load_torch(payload_path)
        payload_targets = payload["target_eV"].reshape(-1).float()
        predictions = payload["prediction_eV"].reshape(-1).float()
        if not torch.equal(payload_targets, correct_targets):
            raise RuntimeError(f"cache target order changed for {candidate}")
        if predictions.numel() != correct_rows.numel():
            raise RuntimeError(f"prediction count changed for {candidate}")
        old_payload_sha = sha256_file(payload_path)
        old_csv_sha = sha256_file(csv_path)
        payload["row_index"] = correct_rows.clone()
        atomic_torch_save(payload_path, payload)
        temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
        with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index", "target_eV", "prediction_eV"])
            writer.writerows(zip(correct_rows.tolist(), correct_targets.tolist(), predictions.tolist()))
        os.replace(temporary_csv, csv_path)
        metrics["validation_row_index_sha256"] = tensor_sha256(correct_rows)
        artifacts["validation_payload_sha256"] = sha256_file(payload_path)
        artifacts["validation_csv_sha256"] = sha256_file(csv_path)
        atomic_json(metrics_path, metrics)
        repaired_runs.append(metrics)
        artifact_changes[candidate] = {
            "old_payload_sha256": old_payload_sha,
            "new_payload_sha256": artifacts["validation_payload_sha256"],
            "old_csv_sha256": old_csv_sha,
            "new_csv_sha256": artifacts["validation_csv_sha256"],
        }
    completion["runs"] = repaired_runs
    atomic_json(output_root / "completion.json", completion)
    result = {
        "format": "molgap-kunshan-validation-identity-repair-v1",
        "source_completion_sha256": sha256_file(source_root / "completion.json"),
        "cache_aggregate_sha256": CACHE_SHA,
        "validation_rows": int(correct_rows.numel()),
        "validation_unique_rows": len(set(correct_rows.tolist())),
        "validation_row_index_sha256": tensor_sha256(correct_rows),
        "validation_target_sha256": tensor_sha256(correct_targets),
        "validation_shard_sha256": shard_hashes,
        "artifact_changes": artifact_changes,
        "prediction_values_changed": False,
        "target_values_changed": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output_root / "identity_repair.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.source_root, args.cache_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
