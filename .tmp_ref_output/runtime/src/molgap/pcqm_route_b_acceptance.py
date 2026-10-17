"""Strict acceptance for completed PCQM Route B encoder outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .pcqm_route_b_training import atomic_json, sha256_file


EXPECTED_ROWS = {"train": 915_012, "dev": 81_961, "official": 4_981}
EXPECTED_DIMS = {
    "gps9": 192,
    "gps11_160": 160,
    "primary_schnet": 176,
    "augmented_schnet": 176,
}


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _accept_encoder_output(
    name: str,
    output_dir: Path,
) -> tuple[dict, dict[tuple[str, str], dict[str, str | int | None]]]:
    if name not in EXPECTED_DIMS:
        raise ValueError(f"unknown Route B encoder: {name}")
    completion_path = output_dir / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("name") != name:
        raise RuntimeError(f"{name} completion manifest is invalid")
    for relative, expected in completion["artifacts"].items():
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"{name} artifact size differs: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"{name} artifact hash differs: {relative}")

    manifest_path = output_dir / "embeddings" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("name") != name
        or manifest.get("embedding_dim") != EXPECTED_DIMS[name]
        or manifest.get("rows") != EXPECTED_ROWS
        or len(manifest.get("parts", [])) != 401
    ):
        raise RuntimeError(f"{name} embedding manifest differs")

    identities = {}
    seen = {role: set() for role in EXPECTED_ROWS}
    rows = {role: 0 for role in EXPECTED_ROWS}
    for item in manifest["parts"]:
        role = item["role"]
        path = output_dir / "embeddings" / item["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source_idx = payload["source_idx"].view(-1).long()
        embeddings = payload["embeddings"]
        if (
            payload.get("name") != name
            or payload.get("role") != role
            or len(source_idx) != item["rows"]
            or embeddings.shape != (item["rows"], EXPECTED_DIMS[name])
            or not torch.isfinite(embeddings).all()
        ):
            raise RuntimeError(f"{name} embedding payload differs: {item['path']}")
        indices = source_idx.tolist()
        if seen[role].intersection(indices):
            raise RuntimeError(f"{name}/{role} has duplicate source_idx")
        seen[role].update(indices)
        rows[role] += len(indices)

        target_hash = None
        if role in {"train", "dev"}:
            targets = payload.get("targets")
            if targets is None or len(targets.view(-1)) != len(source_idx):
                raise RuntimeError(f"{name}/{role} targets are missing")
            target_hash = _tensor_sha256(targets.view(-1).float())
        elif "targets" in payload:
            raise RuntimeError(f"{name}/official unexpectedly contains targets")

        key = (role, Path(item["source_shard"]).name)
        identities[key] = {
            "rows": len(source_idx),
            "source_idx_sha256": _tensor_sha256(source_idx),
            "target_sha256": target_hash,
        }
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"{name} accepted row counts differ: {rows}")
    report = {
        "status": "accepted",
        "best_dev_gap_mae_eV": json.loads(
            (output_dir / "metrics.json").read_text(encoding="utf-8")
        )["best_dev_gap_mae_eV"],
        "embedding_dim": EXPECTED_DIMS[name],
        "rows": rows,
        "parts": len(manifest["parts"]),
        "artifacts": len(completion["artifacts"]),
        "completion_manifest_sha256": sha256_file(completion_path),
    }
    return report, identities


def accept_pcqm_route_b_encoder(
    name: str,
    output_dir: Path,
    report_path: Path,
) -> dict:
    """Verify one encoder before a later cross-encoder alignment acceptance."""
    encoder_report, _ = _accept_encoder_output(name, output_dir)
    report = {
        "format": "molgap-pcqm-route-b-single-encoder-acceptance-v1",
        "status": "accepted",
        "encoder": {name: encoder_report},
        "cross_encoder_alignment": "not_checked",
        "official_targets_read": False,
        "official_test_used": False,
        "production_registry_changed": False,
    }
    atomic_json(report_path, report)
    return report


def accept_pcqm_route_b_encoders(
    output_dirs: dict[str, Path],
    report_path: Path,
) -> dict:
    """Verify artifacts, rows, identities, labels, and cross-model alignment."""
    if set(output_dirs) != set(EXPECTED_DIMS):
        raise ValueError("Route B acceptance requires exactly four encoders")
    reports = {}
    identities = {}
    for name, output_dir in output_dirs.items():
        reports[name], identities[name] = _accept_encoder_output(name, output_dir)

    reference = identities["gps9"]
    for name, encoder_identities in identities.items():
        if name == "gps9":
            continue
        for key, identity in encoder_identities.items():
            if reference.get(key) != identity:
                raise RuntimeError(f"{name} alignment differs for {key}")

    report = {
        "format": "molgap-pcqm-route-b-encoder-acceptance-v1",
        "status": "accepted",
        "encoders": reports,
        "cross_encoder_source_idx_aligned": True,
        "cross_encoder_train_dev_targets_aligned": True,
        "official_targets_read": False,
        "official_test_used": False,
        "production_registry_changed": False,
    }
    atomic_json(report_path, report)
    return report
