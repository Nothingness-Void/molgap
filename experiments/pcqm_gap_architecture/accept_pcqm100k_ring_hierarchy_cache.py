"""No-model acceptance for the downloaded PCQM ring-hierarchy cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_GRAPH_SHA256 = "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
EXPECTED_WEDGE_SHA256 = "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
EXPECTED_GEOMETRY_SHA256 = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
EXPECTED_SOURCE_COMMIT = "58f425258031062c3c3762f13b7d4c160dffba65"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"Ring cache acceptance failed: {label}")


def accept(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), "manifest exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "format": "molgap-pcqm-gap100k-ring-hierarchy-cache-v1",
        "complete": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "parent_graph_cache_aggregate_sha256": EXPECTED_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
        "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
        "official_train_rows_read": 3_378_606,
        "ring_method": "RDKit-GetSymmSSSR-canonical-atom-tuples",
        "ring_feature_channels": 12,
        "ring_edge_feature_channels": 4,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "failure_count": 0,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        require(manifest.get(key) == value, key)
    require(len(str(manifest.get("selected_smiles_sha256", ""))) == 64, "SMILES hash")
    require(
        manifest.get("ring_relations")
        == ["spiro", "fused", "direct_bond", "conjugated_direct_bond"],
        "ring relation contract",
    )
    failures_path = root / manifest["failures_file"]
    require(failures_path.is_file(), "failure ledger exists")
    require(
        sha256_file(failures_path) == manifest["failures_file_sha256"],
        "failure ledger hash",
    )
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    require(failures.get("failures") == [], "zero unresolved failures")

    aggregate = hashlib.sha256()
    role_graphs = {"train": 0, "validation": 0}
    totals = {
        "train": {"rings": 0, "memberships": 0, "directed_relations": 0, "acyclic_graphs": 0},
        "validation": {"rings": 0, "memberships": 0, "directed_relations": 0, "acyclic_graphs": 0},
    }
    require(len(manifest.get("shards", [])) == 22, "22 shards")
    for shard in manifest["shards"]:
        role = shard.get("role")
        require(role in role_graphs, f"role {role}")
        path = root / shard["file"]
        require(path.is_file(), f"shard exists {path.name}")
        require(sha256_file(path) == shard["sha256"], f"shard hash {path.name}")
        aggregate.update(
            f"{role}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
        count = int(shard["graph_count"])
        role_graphs[role] += count
        for key in totals[role]:
            value = int(shard[key[:-1] + "_count"] if key.endswith("s") else shard[key])
            require(value >= 0, f"nonnegative {key} {path.name}")
            totals[role][key] += value
    require(role_graphs == {"train": 100_000, "validation": 10_000}, "role counts")
    require(aggregate.hexdigest() == manifest["aggregate_sha256"], "aggregate hash")
    for role in totals:
        expected = manifest["totals"][role]
        require(expected["graphs"] == role_graphs[role], f"{role} graph total")
        for key, value in totals[role].items():
            require(expected[key] == value, f"{role} {key} total")
        require(expected["rings"] > 0, f"{role} has rings")
        require(expected["memberships"] >= expected["rings"] * 3, f"{role} memberships")

    return {
        "format": "molgap-pcqm-gap100k-ring-hierarchy-cache-acceptance-v1",
        "accepted": True,
        "source_commit": manifest["source_commit"],
        "aggregate_sha256": manifest["aggregate_sha256"],
        "selected_smiles_sha256": manifest["selected_smiles_sha256"],
        "train_graphs": role_graphs["train"],
        "validation_graphs": role_graphs["validation"],
        "totals": manifest["totals"],
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = accept(arguments.root)
    text = json.dumps(result, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
