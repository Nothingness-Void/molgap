"""Kaggle CPU derivation of the immutable sparse topology-wedge cache."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_sparse_triangle_wedge_cache")
EXPECTED_SOURCE_COMMIT = "35fadc9de63e22de7a1cfbe21e4f1af8888e075f"
EXPECTED_PARENT_SOURCE_COMMIT = "ba82461c53243d733474c8930ac1b86d82451c91"
EXPECTED_PARENT_AGGREGATE_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_parent_cache() -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == "molgap-pcqm-gap100k-cache-v2":
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one parent cache, found {candidates}")
    return candidates[0]


def verify_parent_cache(root: Path, manifest: dict) -> None:
    required = {
        "complete": True,
        "source_commit": EXPECTED_PARENT_SOURCE_COMMIT,
        "official_train_rows_read": 3_378_606,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "unresolved_graphs": 0,
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": 16,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Parent cache contract changed for {key}")
    for file_key, hash_key in (
        ("split_file", "split_file_sha256"),
        ("failures_file", "failures_file_sha256"),
        ("replacement_ledger_file", "replacement_ledger_sha256"),
    ):
        path = root / manifest[file_key]
        if sha256_file(path) != manifest[hash_key]:
            raise RuntimeError(f"Parent evidence hash changed: {path.name}")
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Parent shard hash changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Parent aggregate hash does not match its shards")
    if manifest["aggregate_sha256"] != EXPECTED_PARENT_AGGREGATE_SHA256:
        raise RuntimeError("Parent aggregate is not the frozen accepted cache")


def atomic_torch_save(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_wedge.py"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one sparse-wedge source tree, found {matches}")
    return matches[0].parents[1]


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(source_root()))
        import torch

        from molgap.pcqm_wedge import with_wedge_cache

        source_marker = next(
            Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt")
        )
        source_commit = source_marker.read_text(encoding="utf-8").strip()
        if source_commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError(f"Source commit changed: {source_commit}")
        parent_root, parent_manifest = find_parent_cache()
        verify_parent_cache(parent_root, parent_manifest)

        output_shards = []
        total_graphs = {"train": 0, "validation": 0}
        total_wedges = {"train": 0, "validation": 0}
        for ordinal, shard in enumerate(parent_manifest["shards"]):
            role = shard["role"]
            payload = torch.load(
                parent_root / shard["file"],
                map_location="cpu",
                weights_only=False,
            )
            if len(payload) != int(shard["graph_count"]):
                raise RuntimeError(f"Parent shard count changed: {shard['file']}")
            converted = [with_wedge_cache(graph) for graph in payload]
            wedge_total = sum(
                int(graph.wedge_edge_ids.shape[0]) for graph in converted
            )
            filename = f"{role}-{ordinal:04d}.pt"
            output_path = OUT / filename
            atomic_torch_save(output_path, converted)
            output_shards.append(
                {
                    "role": role,
                    "file": filename,
                    "graph_count": len(converted),
                    "wedge_count": wedge_total,
                    "sha256": sha256_file(output_path),
                }
            )
            total_graphs[role] += len(converted)
            total_wedges[role] += wedge_total
            print(
                f"converted {role} shard {ordinal}: graphs={len(converted)} "
                f"wedges={wedge_total}",
                flush=True,
            )

        aggregate = hashlib.sha256()
        for shard in output_shards:
            aggregate.update(
                f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                    "ascii"
                )
            )
        manifest = {
            "format": "molgap-pcqm-gap100k-sparse-wedge-cache-v1",
            "complete": True,
            "source_commit": source_commit,
            "parent_cache_source_commit": parent_manifest["source_commit"],
            "parent_cache_aggregate_sha256": parent_manifest["aggregate_sha256"],
            "parent_cache_manifest_sha256": sha256_file(
                parent_root / "manifest.json"
            ),
            "train_graphs": total_graphs["train"],
            "validation_graphs": total_graphs["validation"],
            "train_wedges": total_wedges["train"],
            "validation_wedges": total_wedges["validation"],
            "wedge_definition": "directed_nonbacktracking_i_to_j_to_k",
            "wedge_edge_id_shape": ["num_wedges", 2],
            "shards": output_shards,
            "aggregate_sha256": aggregate.hexdigest(),
            "gpu_used": False,
            "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "manifest.json", manifest)
        summary = {
            "format": "molgap-pcqm-gap100k-sparse-wedge-cache-run-v1",
            "complete": True,
            "source_commit": source_commit,
            "aggregate_sha256": manifest["aggregate_sha256"],
            "train_graphs": manifest["train_graphs"],
            "validation_graphs": manifest["validation_graphs"],
            "train_wedges": manifest["train_wedges"],
            "validation_wedges": manifest["validation_wedges"],
            "elapsed_s": time.perf_counter() - started,
            "gpu_used": False,
            "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "run_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
