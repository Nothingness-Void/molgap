"""No-model acceptance for the immutable PCQM Gap100K geometry cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_PARENT_GRAPH_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_WEDGE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
MINIMUM_VALID_FRACTION = 0.99


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
        manifest.get("format") == "molgap-pcqm-gap100k-etkdg-geometry-cache-v1",
        "format",
    )
    require(manifest.get("complete") is True, "complete")
    require(manifest.get("source_commit") == expected_source_commit, "source commit")
    require(
        manifest.get("parent_graph_cache_aggregate_sha256")
        == EXPECTED_PARENT_GRAPH_SHA256,
        "parent graph SHA",
    )
    require(
        manifest.get("parent_wedge_cache_aggregate_sha256")
        == EXPECTED_WEDGE_SHA256,
        "parent wedge SHA",
    )
    require(manifest.get("official_train_rows_read") == 3_378_606, "train prefix")
    require(manifest.get("train_graphs") == 100_000, "train count")
    require(manifest.get("validation_graphs") == 10_000, "validation count")
    require(manifest.get("geometry_method") == "ETKDGv3", "ETKDG method")
    require(manifest.get("optimization_method") == "MMFF94s", "MMFF method")
    require(manifest.get("single_conformer") is True, "single conformer")
    valid = manifest.get("valid_geometry_graphs")
    invalid = manifest.get("invalid_geometry_graphs")
    fraction = manifest.get("valid_geometry_fraction")
    require(isinstance(valid, int) and isinstance(invalid, int), "geometry counts")
    if isinstance(valid, int) and isinstance(invalid, int):
        require(valid + invalid == 110_000, "geometry count total")
    require(
        isinstance(fraction, (int, float))
        and math.isfinite(fraction)
        and fraction >= MINIMUM_VALID_FRACTION,
        "valid geometry fraction",
    )
    distance_min = manifest.get("distance_min")
    distance_max = manifest.get("distance_max")
    angle_min = manifest.get("angle_cos_min")
    angle_max = manifest.get("angle_cos_max")
    require(
        isinstance(distance_min, (int, float))
        and isinstance(distance_max, (int, float))
        and 0 < distance_min <= distance_max,
        "distance range",
    )
    require(
        isinstance(angle_min, (int, float))
        and isinstance(angle_max, (int, float))
        and -1.000001 <= angle_min <= angle_max <= 1.000001,
        "angle range",
    )
    failures_path = root / str(manifest.get("failures_file", "__missing__"))
    require(failures_path.is_file(), "failures file")
    if failures_path.is_file():
        require(
            sha256_file(failures_path) == manifest.get("failures_file_sha256"),
            "failures hash",
        )
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
        require(
            len(failures.get("failures", [])) == manifest.get("invalid_geometry_graphs"),
            "failure ledger count",
        )
    shards = manifest.get("shards", [])
    require(isinstance(shards, list) and len(shards) == 22, "shard count")
    aggregate = hashlib.sha256()
    role_counts = {"train": 0, "validation": 0}
    for shard in shards if isinstance(shards, list) else []:
        path = root / str(shard.get("file", "__missing__"))
        require(path.is_file(), f"missing shard {path.name}")
        if path.is_file():
            require(sha256_file(path) == shard.get("sha256"), f"hash {path.name}")
        role = shard.get("role")
        require(role in role_counts, f"role {role}")
        if role in role_counts:
            role_counts[role] += int(shard.get("graph_count", 0))
        aggregate.update(
            f"{role}\t{shard.get('file')}\t{shard.get('sha256')}\n".encode("ascii")
        )
    require(role_counts == {"train": 100_000, "validation": 10_000}, "roles")
    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate")
    require(manifest.get("gpu_used") is False, "GPU used")
    require(manifest.get("model_inference_executed") is False, "model inference")
    require(manifest.get("official_validation_role_read") is False, "official valid")
    require(manifest.get("test_dev_role_read") is False, "test-dev")
    result = {
        "format": "molgap-pcqm-gap100k-etkdg-geometry-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "valid_geometry_graphs": valid,
        "invalid_geometry_graphs": invalid,
        "valid_geometry_fraction": fraction,
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
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
