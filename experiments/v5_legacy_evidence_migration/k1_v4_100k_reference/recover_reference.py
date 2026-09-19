"""Recover a repository-bound V5 K1-v4 reference without model inference.

This utility verifies the retained V4 artifacts and fixed PCQM cache, then
emits only compact manifests.  It never constructs a model or reads a
protected PCQM role.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from molgap.comparison_readiness import target_transform_asset_digest


EXPERIMENT_REL = Path(
    "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference"
)
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
GEOMETRY_SHA256 = "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
PAYLOAD_SHA256 = "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91"
MODEL_SHA256 = "53f9118f34a95e02e3f0d798a56389af53ff55739753b4208c9f53c36d116d95"
CHECKPOINT_SHA256 = "156edf2fc3455433861fc4c31c75b3df68f779451012245040b7acc8f48c26f5"
TARGET_TRANSFORM_ID = "a5ebd05f82f481020f2059c77a7bc7b9dee84081fdb9806fdc18d93a6b8d89d7"
TARGET_IDENTITY = "106854975b2fc711b1d4548f6164ce847c845f31be6eac1ae436edb9684c1967"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def packed_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data, _ = torch.load(path, map_location="cpu", weights_only=False)
    return data.y.view(-1).float(), data.source_idx.view(-1).long()


def recover(repo_root: Path, cache_root: Path, record_root: Path) -> None:
    output = repo_root / EXPERIMENT_REL
    manifest_path = cache_root / "manifest.json"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("Fixed-cache manifest SHA changed")
    manifest = load_json(manifest_path)
    if manifest.get("geometry_aggregate_sha256") != GEOMETRY_SHA256:
        raise RuntimeError("Fixed-cache aggregate identity changed")
    if any(
        manifest.get(name) is not False
        for name in (
            "official_validation_role_read",
            "test_dev_role_read",
            "test_challenge_role_read",
        )
    ):
        raise RuntimeError("Protected role was marked as read")

    train_targets: list[torch.Tensor] = []
    train_indices: list[torch.Tensor] = []
    verified_shards: list[dict[str, Any]] = []
    for item in manifest["geometry_shards"]:
        shard = cache_root / item["file"]
        actual_sha = sha256_file(shard)
        if actual_sha != item["sha256"]:
            raise RuntimeError(f"Shard SHA changed: {item['file']}")
        target, source_idx = packed_tensors(shard)
        expected = torch.arange(
            item["source_idx_min"], item["source_idx_max"] + 1, dtype=torch.long
        )
        if not torch.equal(source_idx, expected):
            raise RuntimeError(f"Source order changed: {item['file']}")
        if item["role"] == "train":
            train_targets.append(target)
            train_indices.append(source_idx)
        verified_shards.append(
            {
                "role": item["role"],
                "file": item["file"],
                "rows": item["rows"],
                "source_idx_min": item["source_idx_min"],
                "source_idx_max": item["source_idx_max"],
                "sha256": actual_sha,
            }
        )

    train_y = torch.cat(train_targets)
    train_idx = torch.cat(train_indices)
    if train_y.numel() != 100_000 or not torch.equal(
        train_idx, torch.arange(100_000, dtype=torch.long)
    ):
        raise RuntimeError("Training membership changed")

    payload_path = record_root / "best_development_payload.pt"
    model_path = record_root / "best_model.pt"
    checkpoint_path = record_root / "last_checkpoint.pt"
    for path, expected in (
        (payload_path, PAYLOAD_SHA256),
        (model_path, MODEL_SHA256),
        (checkpoint_path, CHECKPOINT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"Historical artifact SHA changed: {path.name}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    dev_target = payload["target_eV"].view(-1).float()
    dev_prediction = payload["prediction_eV"].view(-1).float()
    dev_idx = payload["source_idx"].view(-1).long()
    if not torch.equal(dev_idx, torch.arange(100_000, 150_000, dtype=torch.long)):
        raise RuntimeError("Development row order changed")
    if not torch.isfinite(dev_target).all() or not torch.isfinite(dev_prediction).all():
        raise RuntimeError("Development payload is non-finite")

    row_manifest = {
        "format": "molgap-row-manifest-v1",
        "benchmark_identity": "pcqm4mv2-ogb-fixed-100k-gap-v4",
        "dataset_identity": f"pcqm4mv2-ogb-fixed-100k-v1@{MANIFEST_SHA256}",
        "roles": manifest["roles"],
        "verified_shards": verified_shards,
        "train_source_idx_sha256": tensor_sha256(train_idx),
        "development_source_idx_sha256": tensor_sha256(dev_idx),
        "ordering_semantics": "source_idx ascending within each fixed role",
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    row_manifest_path = output / "row_manifest.json"
    atomic_json(row_manifest_path, row_manifest)
    row_manifest_sha = sha256_file(row_manifest_path)

    target_manifest = {
        "format": "molgap-target-manifest-v1",
        "target_identity": TARGET_IDENTITY,
        "name": "pcqm4mv2-homo-lumo-gap",
        "column": "gap",
        "unit": "eV",
        "dtype": "torch.float32",
        "byte_order": "native-little-endian-on-recorded-runtime",
        "tensor_hash_semantics": "sha256-contiguous-c-order-bytes",
        "train_target_sha256": tensor_sha256(train_y),
        "development_target_sha256": tensor_sha256(dev_target),
        "train_rows": 100_000,
        "development_rows": 50_000,
    }
    target_manifest_path = output / "target_manifest.json"
    atomic_json(target_manifest_path, target_manifest)

    transform = {
        "format": "molgap-target-transform-asset-v1",
        "asset_id": TARGET_TRANSFORM_ID,
        "target_identity": TARGET_IDENTITY,
        "mean": float(train_y.mean()),
        "std": float(train_y.std(unbiased=True)),
        "ddof": 1,
        "variance_convention": "sample-standard-deviation-over-fixed-100k-train-role",
        "source_row_manifest_sha256": row_manifest_sha,
        "target_sha256": tensor_sha256(train_y),
    }
    transform["asset_sha256"] = target_transform_asset_digest(transform)
    atomic_json(output / "target_transform.json", transform)

    prediction_manifest = {
        "format": "molgap-aligned-prediction-manifest-v1",
        "artifact_locator": (
            "external://local-platform-record/kaggle/"
            "pcqm_k1_v4_reference_s42_v2/best_development_payload.pt"
        ),
        "artifact_sha256": PAYLOAD_SHA256,
        "prediction_sha256": tensor_sha256(dev_prediction),
        "source_idx_sha256": tensor_sha256(dev_idx),
        "target_sha256": tensor_sha256(dev_target),
        "ordering_semantics": "source_idx ascending",
        "evaluation_role_identity": (
            "pcqm4mv2-ogb-fixed-100k-v1:internal-development-100000-150000"
        ),
        "row_count": 50_000,
        "unique_source_idx": int(dev_idx.unique().numel()),
        "development_gap_mae_eV": float((dev_prediction - dev_target).abs().mean()),
        "tensor_hash_semantics": "sha256-contiguous-c-order-bytes",
    }
    atomic_json(output / "prediction_manifest.json", prediction_manifest)

    runtime = load_json(record_root / "runtime_certificate.json")
    atomic_json(output / "runtime_certificate.json", runtime)
    role_history = {
        "format": "molgap-role-history-index-v1",
        "trajectory_id": "TC-k1-v4-100k-reference-s42",
        "events": [
            {
                "ref": str(EXPERIMENT_REL / "roles/train.json").replace("\\", "/"),
                "sha256": sha256_file(output / "roles/train.json"),
            },
            {
                "ref": str(EXPERIMENT_REL / "roles/development.json").replace("\\", "/"),
                "sha256": sha256_file(output / "roles/development.json"),
            },
        ],
        "protected_roles_read": False,
    }
    atomic_json(output / "role_history.json", role_history)
    cost_index = {
        "format": "molgap-cost-record-index-v1",
        "trajectory_id": "TC-k1-v4-100k-reference-s42",
        "records": [
            {
                "ref": str(
                    EXPERIMENT_REL
                    / "costs/cost-TC-k1-v4-100k-reference-s42.json"
                ).replace("\\", "/"),
                "sha256": sha256_file(
                    output / "costs/cost-TC-k1-v4-100k-reference-s42.json"
                ),
            }
        ],
        "historical_measurement_status": "measurement_missing",
    }
    atomic_json(output / "cost_records.json", cost_index)

    identity = {
        "benchmark_identity": "pcqm4mv2-ogb-fixed-100k-gap-v4",
        "dataset_identity": f"pcqm4mv2-ogb-fixed-100k-v1@{MANIFEST_SHA256}",
        "data_role_identity": "6dadc354f329e38bbacdbe0a9e60b8011a9ce1cf9857174d731894dbefced024",
        "row_membership_identity": MANIFEST_SHA256,
        "row_order_identity": "e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34",
        "feature_identity": "3773a4dddb549c13b0103570020d7360f057f278f06a80cdee7513be55f356f1",
        "target_identity": TARGET_IDENTITY,
        "seed": 42,
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": 128,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "optimizer_identity": "1b37790838e8ad79c4212c382b030c4ef25766e9adb21986ca78813c38960152",
        "optimizer_mode": "adamw-single-parameter-group",
        "optimizer_fused": False,
        "schedule_identity": "97c8f6fa99991299b9855036aa8b24da20d5d5b87b0b3ea3750a7708d37a330f",
        "loss_identity": "bc9a919797dc84d2071edda7e5bd9e386db766b6ecd7e99a1297cf6e54a8ba1a",
        "target_transform_identity": TARGET_TRANSFORM_ID,
        "target_transform_asset_sha256": transform["asset_sha256"],
        "sample_presentations": 3_998_720,
        "optimizer_steps": 31_240,
        "checkpoint_selection_identity": "45ffbe5f7dd49f286fa4b2850898194574e3af74879db1f0fa39cf0f16bcef55",
        "evaluation_role_identity": prediction_manifest["evaluation_role_identity"],
        "selection_role_identity": prediction_manifest["evaluation_role_identity"],
        "architecture_config_identity": "f7c9d64141a0c178084f46e43256cdfbcc135eabd4c85626df1faa369940e01d",
        "runtime_certificate_scope": "single-device-fp32-no-tf32-bs128-optimizer-inclusive",
        "weight_semantics": "live",
        "ema_enabled": False,
        "ema_decay": None,
        "ema_update_frequency": None,
        "evaluation_weight_source": "live",
    }
    acceptance = {
        "format": "molgap-v5-reference-recovery-acceptance-v1",
        "accepted": True,
        "reference_id": "pcqm-k1-v4-100k-reference-s42",
        "training_executed": False,
        "model_inference_executed": False,
        "protected_role_read": False,
        "verified_external_artifacts": {
            "best_development_payload.pt": PAYLOAD_SHA256,
            "best_model.pt": MODEL_SHA256,
            "last_checkpoint.pt": CHECKPOINT_SHA256,
        },
        "compact_manifest_sha256": {
            "row_manifest.json": row_manifest_sha,
            "target_manifest.json": sha256_file(target_manifest_path),
            "prediction_manifest.json": sha256_file(output / "prediction_manifest.json"),
            "runtime_certificate.json": sha256_file(output / "runtime_certificate.json"),
            "role_history.json": sha256_file(output / "role_history.json"),
            "cost_records.json": sha256_file(output / "cost_records.json"),
            "trace_manifest.json": sha256_file(output / "trace_manifest.json"),
            "target_transform.json": sha256_file(output / "target_transform.json"),
        },
        "comparison_identity_sha256": canonical_sha256(identity),
    }
    atomic_json(output / "reference_acceptance.json", acceptance)

    def ref(name: str) -> str:
        return str(EXPERIMENT_REL / name).replace("\\", "/")

    bundle = {
        "format": "molgap-reference-bundle-v1",
        "reference_bundle_id": "reference-k1-v4-100k-s42-v5-recovered",
        "reference_id": "pcqm-k1-v4-100k-reference-s42",
        "contract_ref": "experiments/pcqm_k1_variants_100k/training_contract.json",
        "architecture_config_identity": identity["architecture_config_identity"],
        "source_commit_or_archive": "0006a689b74cf96e994892464830b79687ca60de",
        "checkpoint_identity": MODEL_SHA256,
        "runtime_certificate_ref": ref("runtime_certificate.json"),
        "prediction_manifest": {
            key: prediction_manifest[key]
            for key in (
                "prediction_sha256",
                "source_idx_sha256",
                "target_sha256",
                "ordering_semantics",
                "evaluation_role_identity",
                "row_count",
                "unique_source_idx",
            )
        },
        "row_manifest_ref": ref("row_manifest.json"),
        "target_manifest_ref": ref("target_manifest.json"),
        "trace_manifest_ref": ref("trace_manifest.json"),
        "role_history_ref": ref("role_history.json"),
        "target_transform_asset_ref": ref("target_transform.json"),
        "cost_records_ref": ref("cost_records.json"),
        "acceptance_ref": ref("reference_acceptance.json"),
        "decision_ref": "experiments/pcqm_k1_variants_100k/decision.md",
        "comparison_identity": identity,
        "stochasticity": {
            "row_bootstrap_uncertainty": {"status": "not_requested"},
            "training_stochasticity": {
                "status": "unavailable",
                "estimation_method": "unavailable",
                "source": "unavailable",
                "same_contract_repeat_ids": [],
                "n_repeats": 0,
                "stochasticity_floor_eV": None,
            },
        },
    }
    atomic_json(output / "reference_bundle.json", bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    args = parser.parse_args()
    recover(args.repo_root.resolve(), args.cache_root.resolve(), args.record_root.resolve())


if __name__ == "__main__":
    main()
