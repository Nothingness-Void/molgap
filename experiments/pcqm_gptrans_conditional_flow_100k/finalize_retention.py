"""Prune accepted Kaggle output without losing artifact provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


RECORD = Path(
    "platforms/_records/kaggle/training/"
    "pcqm_gptrans_conditional_flow_s42_v2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _prune_reason(relative: str) -> str | None:
    name = Path(relative).name
    if relative.startswith("_conditional_flow_source/"):
        return "expanded_source_copy_reproducible_from_bound_source_archive"
    if name.startswith("checkpoint_epoch_") and name.endswith(".pt"):
        return "superseded_rolling_recovery_checkpoint"
    if name == "last_checkpoint.pt":
        return "terminal_recovery_checkpoint_duplicate"
    return None


def _inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _by_path(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["path"]): row for row in rows}


def finalize_retention(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    record = (repo_root / RECORD).resolve()
    record.relative_to(repo_root)
    raw = (record / "raw").resolve()
    raw.relative_to(record)
    pre_manifest_path = record / "artifact_manifest.json"
    retained_manifest_path = record / "retained_artifact_manifest.json"
    receipt_path = record / "retention_receipt.json"
    quarantine = (record / ".retention_quarantine").resolve()
    quarantine.relative_to(record)

    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        retained = json.loads(retained_manifest_path.read_text(encoding="utf-8"))
        if _by_path(_inventory(raw)) != _by_path(retained["files"]):
            raise RuntimeError("retained artifact inventory changed after finalization")
        if _sha256(retained_manifest_path) != receipt["retained_manifest_sha256"]:
            raise RuntimeError("retained manifest hash differs from retention receipt")
        return {**receipt, "status": "ALREADY_FINALIZED"}

    if quarantine.exists():
        raise RuntimeError(f"unfinished retention quarantine exists: {quarantine}")

    pre_manifest = json.loads(pre_manifest_path.read_text(encoding="utf-8"))
    expected = pre_manifest["files"]
    actual = _inventory(raw)
    if _by_path(actual) != _by_path(expected):
        raise RuntimeError("raw artifacts do not match the pre-retention manifest")

    removed = []
    retained = []
    for row in expected:
        reason = _prune_reason(str(row["path"]))
        if reason is None:
            retained.append(row)
        else:
            removed.append({**row, "reason": reason})
    if not removed:
        raise RuntimeError("retention policy selected no artifacts")

    quarantine.mkdir(parents=True)
    try:
        for row in removed:
            source = raw / str(row["path"])
            destination = quarantine / str(row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

        for directory in sorted(
            (item for item in raw.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

        final_inventory = _inventory(raw)
        if _by_path(final_inventory) != _by_path(retained):
            raise RuntimeError("post-retention inventory differs from the retention plan")

        retained_manifest = {
            "format": "molgap-retained-artifact-manifest-v1",
            "source_manifest": pre_manifest_path.relative_to(repo_root).as_posix(),
            "source_manifest_sha256": _sha256(pre_manifest_path),
            "source_kernel": pre_manifest["kernel"],
            "source_version": pre_manifest["version"],
            "files": final_inventory,
        }
        _atomic_write(retained_manifest_path, retained_manifest)
        receipt = {
            "format": "molgap-artifact-retention-receipt-v1",
            "status": "complete",
            "policy": "accepted_negative_screen_compact_retention_v1",
            "pre_retention_manifest": pre_manifest_path.relative_to(repo_root).as_posix(),
            "pre_retention_manifest_sha256": _sha256(pre_manifest_path),
            "retained_manifest": retained_manifest_path.relative_to(repo_root).as_posix(),
            "retained_manifest_sha256": _sha256(retained_manifest_path),
            "before": {
                "files": len(expected),
                "bytes": sum(int(row["bytes"]) for row in expected),
            },
            "after": {
                "files": len(final_inventory),
                "bytes": sum(int(row["bytes"]) for row in final_inventory),
            },
            "removed": removed,
            "retained_requirements": {
                "best_models": 2,
                "aligned_development_predictions": 2,
                "training_traces": 2,
                "completion_manifests": 2,
                "runtime_preflights": 2,
                "logs_and_job_summary": True,
            },
            "recoverability": {
                "expanded_source": "bound Git/source-archive identities",
                "rolling_checkpoints": "not retained after accepted terminal negative decision",
                "terminal_models": "retained and hash bound",
            },
        }
        _atomic_write(receipt_path, receipt)
    except Exception:
        for source in sorted(item for item in quarantine.rglob("*") if item.is_file()):
            destination = raw / source.relative_to(quarantine)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        raise
    finally:
        if quarantine.exists():
            shutil.rmtree(quarantine)

    return {**receipt, "status": "FINALIZED"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_retention(args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
