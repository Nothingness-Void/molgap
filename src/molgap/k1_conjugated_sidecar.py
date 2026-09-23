"""CPU-only immutable sidecar derived from accepted fixed OGB graph features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pcqm_conjugated_cache import with_conjugated_components
from .pcqm_k1_variants_runner import (
    FIXED_GEOMETRY_SHA256, FIXED_MANIFEST_SHA256, find_fixed_cache,
    load_roles,
)
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file


FORMAT = "molgap-k1-fixed100k-conjugated-sidecar-v1"
SHARD_ROWS = 5_000


def build_sidecar(output: Path, *, source_commit: str) -> dict:
    if len(source_commit) != 40:
        raise ValueError("Frozen source commit required")
    import torch
    root, manifest = find_fixed_cache()
    roles = load_roles(root, manifest)
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    aggregate = hashlib.sha256()
    totals = {"train": 0, "development": 0}
    for role in ("train", "development"):
        graphs = roles[role]
        for start in range(0, len(graphs), SHARD_ROWS):
            rows = []
            end = min(start + SHARD_ROWS, len(graphs))
            for index in range(start, end):
                graph = with_conjugated_components(graphs[index])
                expected_source = index + (100_000 if role == "development" else 0)
                source_idx = int(graph.source_idx.view(-1)[0])
                if source_idx != expected_source:
                    raise RuntimeError("Fixed source-index order changed")
                ids = graph.conjugated_component_id.clone()
                count = int(graph.conjugated_component_count.view(-1)[0])
                if ids.shape != (graph.num_nodes,) or count < 0:
                    raise RuntimeError("Conjugated component graph alignment changed")
                rows.append({
                    "source_idx": source_idx,
                    "component_id": ids,
                    "component_count": count,
                    "node_count": int(graph.num_nodes),
                })
                totals[role] += count
            name = f"{role}_{start:06d}_{end:06d}.pt"
            path = output / name
            atomic_torch_save(path, rows)
            digest = sha256_file(path)
            shards.append({
                "role": role, "start": start, "end": end,
                "file": name, "sha256": digest, "rows": len(rows),
            })
            aggregate.update(f"{role}\t{name}\t{digest}\n".encode("ascii"))
            print(f"sidecar {role} {end}/{len(graphs)}", flush=True)
    result = {
        "format": FORMAT, "status": "complete",
        "source_commit": source_commit,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "train_rows": len(roles["train"]),
        "development_rows": len(roles["development"]),
        "derived_only_from_ogb_graph_features": True,
        "gap_labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "component_counts": totals,
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
    }
    atomic_json(output / "manifest.json", result)
    return result


def find_sidecar() -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("format") == FORMAT:
            candidates.append((path.parent, payload))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one conjugated sidecar: {candidates}")
    return candidates[0]


def accept_sidecar(root: Path, *, expected_source_commit: str) -> dict:
    """No-inference check of every row, shard digest and protected-role claim."""
    import torch
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != FORMAT
        or manifest.get("status") != "complete"
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
        or manifest.get("fixed_geometry_sha256") != FIXED_GEOMETRY_SHA256
        or manifest.get("train_rows") != 100_000
        or manifest.get("development_rows") != 50_000
        or manifest.get("derived_only_from_ogb_graph_features") is not True
        or manifest.get("gap_labels_read") is not False
        or any(manifest.get(key) is not False for key in (
            "official_validation_role_read", "test_dev_role_read",
            "test_challenge_role_read",
        ))
    ):
        raise RuntimeError("Conjugated sidecar manifest/roles invalid")
    seen = {"train": 0, "development": 0}
    aggregate = hashlib.sha256()
    for item in manifest["shards"]:
        role = item["role"]
        path = root / item["file"]
        if (
            role not in seen or item["start"] != seen[role]
            or item["end"] - item["start"] != item["rows"]
            or not 0 < item["rows"] <= SHARD_ROWS
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError("Conjugated sidecar shard identity invalid")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        if len(rows) != item["rows"]:
            raise RuntimeError("Conjugated sidecar shard size invalid")
        for local, row in enumerate(rows):
            expected = item["start"] + local + (100_000 if role == "development" else 0)
            ids = row["component_id"]
            count = row["component_count"]
            if (
                row["source_idx"] != expected
                or ids.shape != (row["node_count"],)
                or count < 0
                or bool((ids < -1).any())
                or bool((ids >= count).any())
            ):
                raise RuntimeError("Conjugated sidecar row invalid")
        seen[role] = item["end"]
        aggregate.update(
            f"{role}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    if seen != {"train": 100_000, "development": 50_000}:
        raise RuntimeError("Conjugated sidecar coverage incomplete")
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Conjugated sidecar aggregate invalid")
    return {
        "format": "molgap-k1-conjugated-sidecar-acceptance-v1",
        "accepted": True,
        "source_commit": expected_source_commit,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "aggregate_sha256": aggregate.hexdigest(),
        "rows_by_role": seen,
        "model_inference_executed": False,
        "gap_labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }


def attach_sidecar(roles, *, expected_source_commit: str):
    import torch
    root, manifest = find_sidecar()
    if (
        manifest.get("status") != "complete"
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
        or manifest.get("fixed_geometry_sha256") != FIXED_GEOMETRY_SHA256
        or manifest.get("gap_labels_read") is not False
        or any(manifest.get(key) is not False for key in (
            "official_validation_role_read", "test_dev_role_read",
            "test_challenge_role_read",
        ))
    ):
        raise RuntimeError("Conjugated sidecar identity/roles changed")
    all_rows = {"train": [], "development": []}
    aggregate = hashlib.sha256()
    for item in manifest["shards"]:
        path = root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError("Conjugated sidecar shard hash changed")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        role = item["role"]
        if len(rows) != item["rows"] or len(all_rows[role]) != item["start"]:
            raise RuntimeError("Conjugated sidecar ordering changed")
        all_rows[role].extend(rows)
        aggregate.update(
            f"{role}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Conjugated sidecar aggregate changed")

    class AttachedRole(torch.utils.data.Dataset):
        def __init__(self, graphs, rows, offset):
            if len(graphs) != len(rows):
                raise RuntimeError("Conjugated sidecar role size changed")
            self.graphs, self.rows, self.offset = graphs, rows, offset
            self.datasets = graphs.datasets

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            graph = self.graphs[index]
            row = self.rows[index]
            if (
                row["source_idx"] != index + self.offset
                or int(graph.source_idx.view(-1)[0]) != row["source_idx"]
                or int(graph.num_nodes) != row["node_count"]
                or row["component_id"].shape != (graph.num_nodes,)
            ):
                raise RuntimeError("Conjugated sidecar row alignment changed")
            graph.conjugated_component_id = row["component_id"]
            graph.conjugated_component_count = torch.tensor(
                [row["component_count"]], dtype=torch.long
            )
            return graph

    attached = {
        role: AttachedRole(roles[role], all_rows[role], offset)
        for role, offset in (("train", 0), ("development", 100_000))
    }
    return attached, manifest
