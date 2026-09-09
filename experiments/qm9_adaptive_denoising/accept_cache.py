"""Independent no-model acceptance for the QM9 ETKDG denoising cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from molgap.qm9_adaptive_denoising import sha256_file, verify_cache


def accept(root: Path, *, source_commit: str) -> dict:
    manifest = verify_cache(root)
    if manifest.get("source_commit") != source_commit:
        raise RuntimeError("Cache source commit changed")
    manifest_sha256 = sha256_file(root / "manifest.json")
    result = {
        "format": "molgap-qm9-adaptive-denoising-cache-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "manifest_sha256": manifest_sha256,
        "split_fingerprint": manifest["split_fingerprint"],
        "roles": manifest["roles"],
        "failure_count": manifest["failure_count"],
        "dft_coordinates_used": False,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.root, source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
