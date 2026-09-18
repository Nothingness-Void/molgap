"""No-model acceptance for a downloaded K1-MoSE cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from molgap.pcqm_mose import MOSE_DIM, MOSE_UPSTREAM_COMMIT, pattern_fingerprint

    manifest_path = next(iter(args.root.rglob("mose_manifest.json")), None)
    if manifest_path is None:
        raise FileNotFoundError("mose_manifest.json not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "format": manifest.get("format") == "molgap-pcqm-mose-cache-v1",
        "complete": manifest.get("complete") is True,
        "feature_dim": manifest.get("feature_dim") == MOSE_DIM,
        "pattern": manifest.get("pattern_sha256") == pattern_fingerprint(),
        "upstream": manifest.get("upstream_commit") == MOSE_UPSTREAM_COMMIT,
        "rows": manifest.get("rows") == 150000,
        "sealed_roles": all(
            manifest.get(name) is False
            for name in (
                "official_validation_role_read",
                "test_dev_role_read",
                "test_challenge_role_read",
            )
        ),
    }
    aggregate = hashlib.sha256()
    rows = 0
    nodes = 0
    next_source = 0
    for item in manifest.get("parts", []):
        path = manifest_path.parent / item["file"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Part hash mismatch: {item['file']}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        counts = payload["counts"]
        source = payload["source_idx"]
        pointers = payload["node_ptr"]
        if source.tolist() != list(range(next_source, next_source + source.numel())):
            raise RuntimeError("Source indices are not contiguous")
        if counts.ndim != 2 or counts.shape[1] != MOSE_DIM or bool((counts < 0).any()):
            raise RuntimeError("Invalid MoSE counts")
        if int(pointers[-1]) != counts.shape[0] or pointers.numel() != source.numel() + 1:
            raise RuntimeError("Invalid MoSE node pointers")
        next_source += source.numel()
        rows += source.numel()
        nodes += counts.shape[0]
        aggregate.update(
            f"{item['file']}\t{item['sha256']}\t{item['source_start']}\t{item['source_stop']}\n".encode("ascii")
        )
    checks.update(
        {
            "part_rows": rows == manifest.get("rows"),
            "part_nodes": nodes == manifest.get("nodes"),
            "aggregate": aggregate.hexdigest() == manifest.get("aggregate_sha256"),
            "source_stop": next_source == 150000,
        }
    )
    accepted = all(checks.values())
    report = {
        "format": "molgap-pcqm-k1-mose-cache-acceptance-v1",
        "accepted": accepted,
        "model_inference_executed": False,
        "checks": checks,
        "manifest_sha256": sha256(manifest_path),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not accepted:
        raise RuntimeError(report)


if __name__ == "__main__":
    main()

