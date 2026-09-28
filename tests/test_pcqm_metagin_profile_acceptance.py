"""Synthetic receipt checks; no fixed-role data or model inference."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from molgap.constants import REPO_ROOT


MODULE = REPO_ROOT / "experiments/pcqm_metagin_2d_100k/accept_runtime_profile.py"
SPEC = importlib.util.spec_from_file_location("metagin_profile_acceptance", MODULE)
assert SPEC is not None and SPEC.loader is not None
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTANCE)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def output(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    record = tmp_path / "record"
    source.mkdir()
    record.mkdir()
    payload = b"source"
    digest = hashlib.sha256(payload).hexdigest()
    (source / "source_payload.bin").write_bytes(payload)
    (source / "SOURCE_ARCHIVE_SHA256.txt").write_text(digest)
    (source / "SOURCE_COMMIT.txt").write_text(ACCEPTANCE.SOURCE_RECEIPT["source_commit"])
    monkeypatch.setattr(ACCEPTANCE, "SOURCE_RECEIPT", {
        **ACCEPTANCE.SOURCE_RECEIPT, "source_archive_sha256": digest,
    })
    train = 0.2
    forward = 0.05
    projected = 40 * (781 * train + 391 * forward)
    profile = {
        "format": "molgap-metagin-2d-train-role-runtime-profile-v1",
        "complete": True, "training_screen_executed": False,
        "run_id": ACCEPTANCE.RUN_ID,
        "source_commit": ACCEPTANCE.SOURCE_RECEIPT["source_commit"],
        "sidecar_producer_commit": ACCEPTANCE.SIDECAR["source_commit"],
        "sidecar_aggregate_sha256": ACCEPTANCE.SIDECAR["aggregate_sha256"],
        "sidecar_accepted": True,
        "target_transform_asset_id": ACCEPTANCE.TRANSFORM["asset_id"],
        "target_transform_observation": {
            "train_target_sha256": ACCEPTANCE.TRANSFORM["target_sha256"],
            "train_source_indices_verified": True,
            "transform_source": "immutable_asset_exact_values",
        },
        "hardware": "Tesla T4", "allocated_device_count": 2,
        "precision": "fp32", "tf32_enabled": False,
        "physical_batch_per_device": 128,
        "warmup_step_seconds": [0.3] * 8,
        "measured_step_seconds": [train] * 72,
        "evaluation_forward_seconds": [forward] * 32,
        "steady_train_step_mean_seconds": train,
        "steady_train_step_median_seconds": train,
        "steady_train_step_max_seconds": train,
        "evaluation_step_mean_seconds": forward,
        "projected_epoch_seconds": projected / 40,
        "projected_40_epoch_seconds": projected,
        "projected_with_20_percent_reserve_seconds": 1.2 * projected,
        "profile_wall_seconds": 120,
        "peak_reserved_mib": 4000, "total_memory_mib": 16000,
        "train_role_read": True,
        "internal_development_graphs_loaded": True,
        "internal_development_labels_used": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    cost = {
        "format": "molgap-metagin-runtime-profile-cost-v1",
        "run_id": ACCEPTANCE.RUN_ID,
        "source_commit": ACCEPTANCE.SOURCE_RECEIPT["source_commit"],
        "wall_seconds": 130, "training_screen_executed": False,
    }
    _write(record / "runtime_profile.json", profile)
    _write(record / "native_cost.json", cost)
    return record, source, profile


def test_profile_receipt_recomputes_projection_and_native_device_cost(output):
    record, source, _ = output
    result = ACCEPTANCE.accept(record, source)
    assert result["accepted"] is True
    assert result["same_size_training_admissible_on_measured_hardware"] is True
    assert result["allocated_device_count"] == 2
    assert result["projected_allocated_device_hours_with_reserve"] > 0


@pytest.mark.parametrize("field,value", [
    ("test_dev_role_read", True),
    ("sidecar_aggregate_sha256", "0" * 64),
    ("training_screen_executed", True),
    ("physical_batch_per_device", 64),
    ("projected_40_epoch_seconds", 1),
])
def test_profile_acceptance_fails_closed_on_changed_evidence(output, field, value):
    record, source, profile = output
    profile[field] = value
    _write(record / "runtime_profile.json", profile)
    with pytest.raises(ValueError):
        ACCEPTANCE.accept(record, source)


def test_profile_rejects_failed_run_even_with_matching_json(output):
    record, source, _ = output
    (record / "failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="failure marker"):
        ACCEPTANCE.accept(record, source)
