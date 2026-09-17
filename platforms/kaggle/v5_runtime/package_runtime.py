from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_bundle import build_v4_source_bundle


STATIC_PATHS = (
    "AGENTS.md",
    "ARCHITECTURE.md",
    "pyproject.toml",
    "platforms/V4_EXECUTION_CONTRACT.md",
    "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md",
    "docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--account", required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    source_paths = [
        path.relative_to(repo_root).as_posix()
        for path in sorted((repo_root / "src" / "molgap").rglob("*.py"))
    ]
    source_paths.extend(STATIC_PATHS)
    result = build_v4_source_bundle(
        repo_root=repo_root,
        relative_paths=source_paths,
        output_dir=output,
        source_commit=source_commit,
        archive_name="molgap_runtime.tar.gz",
    )
    manifest = {
        "format": "molgap-v5-desktop-runtime-v1",
        "status": "complete",
        "source_commit": source_commit,
        "archive": "molgap_runtime.tar.gz",
        "archive_sha256": result["archive_sha256"],
        "source_files_sha256": sha256_file(output / "SOURCE_FILES.json"),
        "file_count": result["file_count"],
        "payload_bytes": result["payload_bytes"],
        "operating_contract": "MOLGAP-COMMON-V5-FINAL",
        "desktop_contract": "MOLGAP-DESKTOP-V5-FINAL",
        "scientific_contract_included": False,
        "training_authorized": False,
    }
    atomic_json(output / "runtime_manifest.json", manifest)
    atomic_json(
        output / "dataset-metadata.json",
        {
            "id": f"{args.account}/molgap-v5-desktop-runtime",
            "title": "MolGap V5 Desktop Runtime",
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
