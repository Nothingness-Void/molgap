"""No-model acceptance for the PCQM non-covalent ContactState cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_SOURCE_COMMIT = "7f2f8ce476f654320f07e2c2e630f473d7d81c72"
EXPECTED_GRAPH_SHA256 = "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
EXPECTED_WEDGE_SHA256 = "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
EXPECTED_GEOMETRY_SHA256 = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
MAX_DIRECTED_EDGES = 10_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    required = {
        "format": "molgap-pcqm-gap100k-contactstate-cache-v1",
        "complete": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "parent_graph_cache_aggregate_sha256": EXPECTED_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
        "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
        "contact_method": "ETKDGv3+MMFF94s-distance-cutoff-exclude-covalent-hops",
        "contact_cutoff_angstrom": 5.0,
        "excluded_covalent_hops": 3,
        "directed_storage": True,
        "neighbor_cap": None,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "directed_edge_budget": MAX_DIRECTED_EDGES,
        "failure_count": 0,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        require(manifest.get(key) == value, key)
    failures_path = root / manifest.get("failures_file", "__missing__")
    require(failures_path.is_file(), "failure ledger")
    if failures_path.is_file():
        require(sha256_file(failures_path) == manifest.get("failures_file_sha256"), "failure hash")
        require(json.loads(failures_path.read_text()).get("failures") == [], "zero failures")

    aggregate = hashlib.sha256()
    role_graphs = {"train": 0, "validation": 0}
    role_edges = {"train": 0, "validation": 0}
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
        aggregate.update(f"{role}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii"))
        role_graphs[role] += int(shard.get("graph_count", 0))
        stats = shard.get("statistics", {})
        require(stats.get("graphs") == shard.get("graph_count"), f"stats graphs {path.name}")
        require(int(stats.get("directed_edges", -1)) == int(stats.get("undirected_pairs", -1)) * 2, f"paired edges {path.name}")
        require(int(stats.get("valid_geometry_graphs", -1)) + int(stats.get("invalid_geometry_graphs", -1)) == int(shard.get("graph_count", 0)), f"geometry count {path.name}")
        role_edges[role] += int(stats.get("directed_edges", 0))
    require(role_graphs == {"train": 100_000, "validation": 10_000}, "role graph totals")
    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate")
    require(0 < role_edges["train"], "train contacts")
    require(0 < role_edges["validation"], "validation contacts")
    require(sum(role_edges.values()) <= MAX_DIRECTED_EDGES, "edge budget")
    distance_min = manifest.get("distance_min_angstrom")
    distance_max = manifest.get("distance_max_angstrom")
    require(isinstance(distance_min, (int, float)) and math.isfinite(distance_min) and distance_min > 0, "distance min")
    require(isinstance(distance_max, (int, float)) and math.isfinite(distance_max) and distance_max <= 5.0, "distance max")
    for role in role_graphs:
        totals = manifest.get("totals", {}).get(role, {})
        require(totals.get("graphs") == role_graphs[role], f"{role} manifest graphs")
        require(totals.get("directed_edges") == role_edges[role], f"{role} manifest edges")
        require(totals.get("graphs_with_contacts", 0) > 0, f"{role} coverage")
        require(isinstance(totals.get("atom_type_pairs"), dict), f"{role} atom pairs")

    result = {
        "format": "molgap-pcqm-gap100k-contactstate-cache-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": manifest.get("source_commit"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "parent_geometry_cache_aggregate_sha256": manifest.get("parent_geometry_cache_aggregate_sha256"),
        "role_graphs": role_graphs,
        "role_directed_edges": role_edges,
        "totals": manifest.get("totals"),
        "distance_min_angstrom": distance_min,
        "distance_max_angstrom": distance_max,
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
