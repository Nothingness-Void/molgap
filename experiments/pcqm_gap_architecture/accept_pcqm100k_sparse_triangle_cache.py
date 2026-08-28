"""No-runtime acceptance for the downloaded sparse-wedge PCQM cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PARENT_SOURCE_COMMIT = "ba82461c53243d733474c8930ac1b86d82451c91"
PARENT_AGGREGATE_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        manifest.get("format") == "molgap-pcqm-gap100k-sparse-wedge-cache-v1",
        "format",
    )
    require(manifest.get("complete") is True, "complete")
    require(manifest.get("source_commit") == expected_source_commit, "source commit")
    require(
        manifest.get("parent_cache_source_commit") == PARENT_SOURCE_COMMIT,
        "parent source commit",
    )
    require(
        manifest.get("parent_cache_aggregate_sha256") == PARENT_AGGREGATE_SHA256,
        "parent aggregate",
    )
    require(manifest.get("train_graphs") == 100_000, "train graph count")
    require(manifest.get("validation_graphs") == 10_000, "validation graph count")
    require(
        manifest.get("wedge_definition") == "directed_nonbacktracking_i_to_j_to_k",
        "wedge definition",
    )
    require(
        manifest.get("wedge_edge_id_shape") == ["num_wedges", 2],
        "wedge shape",
    )
    for key in (
        "gpu_used",
        "model_inference_executed",
        "official_validation_role_read",
        "test_dev_role_read",
    ):
        require(manifest.get(key) is False, key)

    aggregate = hashlib.sha256()
    graph_counts = {"train": 0, "validation": 0}
    wedge_counts = {"train": 0, "validation": 0}
    shards = manifest.get("shards", [])
    require(bool(shards), "shards")
    for shard in shards:
        role = shard.get("role")
        require(role in graph_counts, f"shard role {role}")
        path = root / shard.get("file", "__missing__")
        require(path.is_file(), f"missing shard {path.name}")
        if path.is_file():
            require(sha256_file(path) == shard.get("sha256"), f"shard hash {path.name}")
        graph_count = shard.get("graph_count")
        wedge_count = shard.get("wedge_count")
        require(isinstance(graph_count, int) and graph_count > 0, f"graph count {path.name}")
        require(isinstance(wedge_count, int) and wedge_count >= 0, f"wedge count {path.name}")
        if role in graph_counts and isinstance(graph_count, int):
            graph_counts[role] += graph_count
        if role in wedge_counts and isinstance(wedge_count, int):
            wedge_counts[role] += wedge_count
        aggregate.update(
            f"{role}\t{shard.get('file')}\t{shard.get('sha256')}\n".encode("ascii")
        )
    require(graph_counts == {"train": 100_000, "validation": 10_000}, "graph totals")
    require(
        aggregate.hexdigest() == manifest.get("aggregate_sha256"),
        "aggregate hash",
    )
    for role, count in wedge_counts.items():
        require(isinstance(count, int) and count >= 0, f"wedge total {role}")
        require(math.isfinite(float(count)), f"finite wedge total {role}")

    result = {
        "format": "molgap-pcqm-gap100k-sparse-wedge-cache-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "parent_cache_aggregate_sha256": PARENT_AGGREGATE_SHA256,
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "train_graphs": manifest.get("train_graphs"),
        "validation_graphs": manifest.get("validation_graphs"),
        "train_wedges": manifest.get("train_wedges"),
        "validation_wedges": manifest.get("validation_wedges"),
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
