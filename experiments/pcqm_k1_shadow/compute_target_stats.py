"""Extract the frozen 100K target normalization without model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_CACHE_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    manifest_path = args.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("aggregate_sha256") != EXPECTED_CACHE_SHA256:
        raise RuntimeError("Frozen 100K geometry-cache identity changed")
    if manifest.get("official_validation_role_read") is not False:
        raise RuntimeError("Official validation role boundary changed")
    if manifest.get("test_dev_role_read") is not False:
        raise RuntimeError("Test-dev role boundary changed")

    targets = []
    rows = []
    shard_hashes = []
    for shard in manifest["shards"]:
        if shard["role"] != "train":
            continue
        path = args.cache_root / shard["file"]
        observed_sha = sha256_file(path)
        if observed_sha != shard["sha256"]:
            raise RuntimeError(f"Training shard changed: {path.name}")
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        if len(graphs) != int(shard["graph_count"]):
            raise RuntimeError(f"Training shard count changed: {path.name}")
        shard_hashes.append(observed_sha)
        for graph in graphs:
            targets.append(graph.y.view(()).float())
            rows.append(int(graph.row_index.view(-1)[0]))
    target = torch.stack(targets)
    if target.numel() != 100_000 or len(set(rows)) != 100_000:
        raise RuntimeError("Frozen target-normalization role changed")
    row_digest = hashlib.sha256(
        ",".join(str(row) for row in rows).encode("ascii")
    ).hexdigest()
    payload = {
        "format": "molgap-pcqm-k1-shadow-target-stats-v1",
        "source_graph_cache_aggregate_sha256": EXPECTED_CACHE_SHA256,
        "source_manifest_sha256": sha256_file(manifest_path),
        "train_rows": 100_000,
        "train_row_order_sha256": row_digest,
        "train_shard_sha256": shard_hashes,
        "target_mean_eV": float(target.mean()),
        "target_std_eV": float(target.std().clamp_min(1e-6)),
        "model_inference_executed": False,
        "shadow_labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
