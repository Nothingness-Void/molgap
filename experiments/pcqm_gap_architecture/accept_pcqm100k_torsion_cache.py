"""No-model acceptance for the immutable PCQM Gap100K torsion cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_GEOMETRY_SOURCE_COMMIT = "e083bee19ee6a13cd9f72e91229752a9d5f56389"
EXPECTED_PARENT_GRAPH_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_PARENT_WEDGE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
EXPECTED_GEOMETRY_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
EXPECTED_INVALID_GEOMETRY_ROWS = 315


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_index_sha256(rows: list[int]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row}\n".encode("ascii"))
    return digest.hexdigest()


def accept(
    root: Path,
    expected_source_commit: str,
    expected_geometry_sha256: str,
) -> dict:
    import torch

    from molgap.pcqm_torsion import directed_nonbacktracking_torsions

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    required = {
        "format": "molgap-pcqm-gap100k-etkdg-torsion-cache-v1",
        "complete": True,
        "source_commit": expected_source_commit,
        "geometry_source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
        "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
        "parent_geometry_cache_aggregate_sha256": expected_geometry_sha256,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "invalid_geometry_rows": EXPECTED_INVALID_GEOMETRY_ROWS,
        "torsion_definition": "directed_nonbacktracking_i_j_k_l",
        "torsion_edge_id_shape": ["num_torsions", 3],
        "torsion_wedge_id_shape": ["num_torsions", 2],
        "torsion_feature_definition": "[sin(phi), cos(phi), sin(2phi), cos(2phi)]",
        "torsion_feature_dtype": "float32",
        "invalid_geometry_policy": "zero_torsion_features_and_zero_path_mask",
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        require(manifest.get(key) == value, key)

    failures_path = root / str(manifest.get("failures_file", "__missing__"))
    require(failures_path.is_file(), "failures file")
    if failures_path.is_file():
        require(
            sha256_file(failures_path) == manifest.get("failures_file_sha256"),
            "failures hash",
        )

    shards = manifest.get("shards", [])
    require(isinstance(shards, list) and bool(shards), "shards")
    aggregate = hashlib.sha256()
    graphs_by_role = {"train": 0, "validation": 0}
    paths_by_role = {"train": 0, "validation": 0}
    valid_paths_by_role = {"train": 0, "validation": 0}
    rows_by_role = {"train": [], "validation": []}
    invalid_rows: set[int] = set()
    for shard in shards:
        path = root / str(shard.get("file", "__missing__"))
        require(path.is_file(), f"missing shard {path.name}")
        if not path.is_file():
            continue
        actual_hash = sha256_file(path)
        require(actual_hash == shard.get("sha256"), f"shard hash {path.name}")
        aggregate.update(
            f"{shard.get('role')}\t{shard.get('file')}\t{shard.get('sha256')}\n".encode(
                "ascii"
            )
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(len(payload) == int(shard.get("graph_count", -1)), f"count {path.name}")
        role = shard.get("role")
        require(role in graphs_by_role, f"role {path.name}")
        if role not in graphs_by_role:
            continue
        shard_paths = 0
        shard_valid_paths = 0
        shard_invalid_rows = 0
        for graph in payload:
            row_index = int(graph.row_index.view(-1)[0])
            rows_by_role[role].append(row_index)
            require(
                tuple(graph.edge_distance.shape) == (graph.edge_index.shape[1], 1),
                f"distance alignment row {row_index}",
            )
            require(
                tuple(graph.wedge_angle_cos.shape)
                == (graph.wedge_edge_ids.shape[0], 1),
                f"angle alignment row {row_index}",
            )
            torsion_edges = graph.torsion_edge_ids
            torsion_wedges = graph.torsion_wedge_ids
            torsion_features = graph.torsion_fourier
            torsion_valid = graph.torsion_valid
            require(
                torsion_edges.ndim == 2 and torsion_edges.shape[1] == 3,
                f"torsion edge shape row {row_index}",
            )
            require(
                torsion_wedges.ndim == 2 and torsion_wedges.shape[1] == 2,
                f"torsion wedge shape row {row_index}",
            )
            torsion_count = int(torsion_edges.shape[0])
            require(
                torsion_wedges.shape[0] == torsion_count,
                f"torsion wedge alignment row {row_index}",
            )
            require(
                tuple(torsion_features.shape) == (torsion_count, 4),
                f"torsion feature alignment row {row_index}",
            )
            require(
                tuple(torsion_valid.shape) == (torsion_count, 1),
                f"torsion mask alignment row {row_index}",
            )
            if torsion_count:
                require(
                    int(torsion_edges.min()) >= 0
                    and int(torsion_edges.max()) < graph.edge_index.shape[1],
                    f"torsion edge range row {row_index}",
                )
                require(
                    int(torsion_wedges.min()) >= 0
                    and int(torsion_wedges.max()) < graph.wedge_edge_ids.shape[0],
                    f"torsion wedge range row {row_index}",
                )
                expected_edges, expected_wedges = directed_nonbacktracking_torsions(
                    graph.edge_index, graph.wedge_edge_ids
                )
                require(
                    torch.equal(torsion_edges, expected_edges),
                    f"torsion edge enumeration row {row_index}",
                )
                require(
                    torch.equal(torsion_wedges, expected_wedges),
                    f"torsion wedge enumeration row {row_index}",
                )
            require(
                bool(torch.isfinite(torsion_features).all()),
                f"torsion features finite row {row_index}",
            )
            require(
                bool(torch.isfinite(torsion_valid).all()),
                f"torsion mask finite row {row_index}",
            )
            require(
                bool(((torsion_valid >= 0) & (torsion_valid <= 1)).all()),
                f"torsion mask range row {row_index}",
            )
            valid_count = int(torsion_valid.sum().item())
            if not bool(graph.geometry_valid.view(-1)[0]):
                shard_invalid_rows += 1
                invalid_rows.add(row_index)
                require(
                    not bool(torsion_valid.any()),
                    f"invalid geometry torsion mask row {row_index}",
                )
                require(
                    not bool(torsion_features.any()),
                    f"invalid geometry torsion features row {row_index}",
                )
            if valid_count:
                selected = torsion_features[torsion_valid.view(-1) > 0]
                require(
                    bool(torch.allclose(
                        selected[:, 0] ** 2 + selected[:, 1] ** 2,
                        torch.ones(valid_count),
                        atol=2e-5,
                        rtol=2e-5,
                    )),
                    f"first periodic norm row {row_index}",
                )
                require(
                    bool(torch.allclose(
                        selected[:, 2] ** 2 + selected[:, 3] ** 2,
                        torch.ones(valid_count),
                        atol=2e-5,
                        rtol=2e-5,
                    )),
                    f"second periodic norm row {row_index}",
                )
            graphs_by_role[role] += 1
            paths_by_role[role] += torsion_count
            valid_paths_by_role[role] += valid_count
            shard_paths += torsion_count
            shard_valid_paths += valid_count
        require(
            shard_paths == int(shard.get("torsion_path_count", -1)),
            f"path count {path.name}",
        )
        require(
            shard_valid_paths == int(shard.get("valid_torsion_path_count", -1)),
            f"valid path count {path.name}",
        )
        require(
            shard_invalid_rows == int(shard.get("invalid_geometry_rows", -1)),
            f"invalid row count {path.name}",
        )

    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate hash")
    require(graphs_by_role == {"train": 100_000, "validation": 10_000}, "role graph counts")
    require(len(invalid_rows) == EXPECTED_INVALID_GEOMETRY_ROWS, "invalid geometry rows")
    require(
        sum(paths_by_role.values()) == sum(manifest.get("torsion_paths", {}).values()),
        "manifest path totals",
    )
    require(
        sum(valid_paths_by_role.values())
        == sum(manifest.get("valid_torsion_paths", {}).values()),
        "manifest valid path totals",
    )
    for role, expected in (("train", 100_000), ("validation", 10_000)):
        require(len(rows_by_role[role]) == expected, f"row count {role}")
        require(
            len(set(rows_by_role[role])) == expected,
            f"row uniqueness {role}",
        )

    result = {
        "format": "molgap-pcqm-gap100k-etkdg-torsion-cache-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "geometry_source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
        "parent_geometry_cache_aggregate_sha256": expected_geometry_sha256,
        "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "train_graphs": graphs_by_role["train"],
        "validation_graphs": graphs_by_role["validation"],
        "train_row_index_sha256": row_index_sha256(sorted(rows_by_role["train"])),
        "validation_row_index_sha256": row_index_sha256(
            sorted(rows_by_role["validation"])
        ),
        "invalid_geometry_rows": len(invalid_rows),
        "torsion_paths": paths_by_role,
        "valid_torsion_paths": valid_paths_by_role,
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
    parser.add_argument("--expected-geometry-sha256", default=EXPECTED_GEOMETRY_SHA256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit, args.expected_geometry_sha256)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
