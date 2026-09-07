"""No-model acceptance for the PCQM exact-shortest hop-path cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_GRAPH_SHA256 = "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
EXPECTED_WEDGE_SHA256 = "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
EXPECTED_GEOMETRY_SHA256 = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    required = {
        "format": "molgap-pcqm-gap100k-hop-path-cache-v1",
        "complete": True,
        "source_commit": expected_source_commit,
        "parent_graph_cache_aggregate_sha256": EXPECTED_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
        "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "hops": [2, 3],
        "feature_channels": 8,
        "exact_shortest_path": True,
        "simple_paths": True,
        "directed_relations": True,
        "failure_count": 0,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        require(manifest.get(key) == value, key)
    expected_definition = (
        "hop2,hop3,log1p_path_count,mean_single,mean_double,"
        "mean_triple,mean_aromatic,fully_conjugated_fraction"
    )
    require(manifest.get("feature_definition") == expected_definition, "features")
    role_graphs = {"train": 0, "validation": 0}
    role_hop2 = {"train": 0, "validation": 0}
    role_hop3 = {"train": 0, "validation": 0}
    role_multipath = {"train": 0, "validation": 0}
    aggregate = hashlib.sha256()
    maximum_path_count = 0
    require(len(manifest.get("shards", [])) == 22, "22 shards")
    for shard in manifest.get("shards", []):
        role = shard.get("role")
        require(role in role_graphs, f"role {role}")
        path = root / shard.get("file", "__missing__")
        require(path.is_file(), f"shard {path.name}")
        if path.is_file():
            require(sha256_file(path) == shard.get("sha256"), f"hash {path.name}")
        if role not in role_graphs:
            continue
        aggregate.update(
            f"{role}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
        role_graphs[role] += int(shard.get("graph_count", 0))
        role_hop2[role] += int(shard.get("hop2_relations", 0))
        role_hop3[role] += int(shard.get("hop3_relations", 0))
        role_multipath[role] += int(shard.get("multipath_relations", 0))
        maximum_path_count = max(
            maximum_path_count, int(shard.get("maximum_path_count", 0))
        )
        require(shard.get("finite_values") is True, f"finite {path.name}")
    require(role_graphs == {"train": 100_000, "validation": 10_000}, "role totals")
    require(min(role_hop2.values()) > 0, "nonempty hop2")
    require(min(role_hop3.values()) > 0, "nonempty hop3")
    require(min(role_multipath.values()) > 0, "nonempty multipath")
    require(maximum_path_count > 1, "path multiplicity")
    require(manifest.get("role_hop2_relations") == role_hop2, "hop2 totals")
    require(manifest.get("role_hop3_relations") == role_hop3, "hop3 totals")
    require(
        manifest.get("role_multipath_relations") == role_multipath,
        "multipath totals",
    )
    require(manifest.get("maximum_path_count") == maximum_path_count, "max paths")
    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate")
    result = {
        "format": "molgap-pcqm-gap100k-hop-path-cache-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": manifest.get("source_commit"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "role_graphs": role_graphs,
        "role_hop2_relations": role_hop2,
        "role_hop3_relations": role_hop3,
        "role_multipath_relations": role_multipath,
        "maximum_path_count": maximum_path_count,
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
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
