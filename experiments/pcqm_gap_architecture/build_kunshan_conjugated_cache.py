"""Build an atomic conjugated-component cache from accepted geometry shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


PARENT_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
PARENT_FORMAT = "molgap-pcqm-gap100k-etkdg-geometry-cache-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build(input_root: Path, output_root: Path, source_commit: str) -> dict:
    import torch

    from molgap.pcqm_conjugated_cache import (
        FEATURE_NAMES,
        conjugated_counts,
        with_conjugated_components,
    )

    started = time.perf_counter()
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != PARENT_FORMAT or manifest.get("aggregate_sha256") != PARENT_SHA:
        raise RuntimeError("parent geometry cache identity changed")
    if manifest.get("train_graphs") != 100_000 or manifest.get("validation_graphs") != 10_000:
        raise RuntimeError("parent role counts changed")
    if manifest.get("official_validation_role_read") is not False or manifest.get("test_dev_role_read") is not False:
        raise RuntimeError("parent sealed-role flags changed")
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "manifest.json").exists():
        raise RuntimeError("accepted conjugated cache must not be overwritten")

    output_shards = []
    totals = {
        "train": {"graphs": 0, "components": 0, "component_atoms": 0, "graphs_with_components": 0, "max_components_per_graph": 0},
        "validation": {"graphs": 0, "components": 0, "component_atoms": 0, "graphs_with_components": 0, "max_components_per_graph": 0},
    }
    for ordinal, shard in enumerate(manifest["shards"]):
        parent_path = input_root / shard["file"]
        if sha256_file(parent_path) != shard["sha256"]:
            raise RuntimeError(f"parent shard hash changed: {shard['file']}")
        graphs = torch.load(parent_path, map_location="cpu", weights_only=False)
        if len(graphs) != int(shard["graph_count"]):
            raise RuntimeError(f"parent graph count changed: {shard['file']}")
        converted = [with_conjugated_components(graph) for graph in graphs]
        filename = f"{shard['role']}-{ordinal:04d}.pt"
        output_path = output_root / filename
        if output_path.exists():
            raise RuntimeError(f"untracked partial shard exists: {filename}")
        atomic_torch_save(output_path, converted)
        counts = conjugated_counts(converted)
        record = {
            "role": shard["role"],
            "file": filename,
            "source_geometry_file": shard["file"],
            "source_geometry_sha256": shard["sha256"],
            "graph_count": len(converted),
            "sha256": sha256_file(output_path),
            **{key: value for key, value in counts.items() if key != "graphs"},
        }
        output_shards.append(record)
        role_totals = totals[shard["role"]]
        role_totals["graphs"] += counts["graphs"]
        for key in ("components", "component_atoms", "graphs_with_components"):
            role_totals[key] += counts[key]
        role_totals["max_components_per_graph"] = max(
            role_totals["max_components_per_graph"],
            counts["max_components_per_graph"],
        )
        atomic_json(
            output_root / "progress.json",
            {
                "format": "molgap-pcqm-conjugated-cache-progress-v1",
                "complete": False,
                "source_commit": source_commit,
                "parent_geometry_cache_aggregate_sha256": PARENT_SHA,
                "shards": output_shards,
                "totals": totals,
                "elapsed_s": time.perf_counter() - started,
                "gpu_used": False,
                "model_inference_executed": False,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        print(
            f"conjugated cache {shard['role']} {ordinal}: "
            f"graphs={len(converted)} components={counts['components']}",
            flush=True,
        )

    aggregate = hashlib.sha256()
    for shard in output_shards:
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    result = {
        "format": "molgap-pcqm-gap100k-conjugated-component-cache-v1",
        "complete": True,
        "source_commit": source_commit,
        "parent_geometry_cache_aggregate_sha256": PARENT_SHA,
        "component_rule": "connected_components_of_ogb_conjugated_bonds",
        "local_component_order": "ascending_minimum_atom_index",
        "nonmember_component_id": -1,
        "feature_names": list(FEATURE_NAMES),
        "feature_channels": len(FEATURE_NAMES),
        "train_graphs": totals["train"]["graphs"],
        "validation_graphs": totals["validation"]["graphs"],
        "totals": totals,
        "shards": output_shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "elapsed_s": time.perf_counter() - started,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output_root / "manifest.json", result)
    atomic_json(output_root / "progress.json", {**result, "complete": True})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = build(args.input_root, args.output_root, args.source_commit)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
