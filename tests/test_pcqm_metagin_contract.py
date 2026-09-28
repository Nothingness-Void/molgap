"""Static MetaGIN V5 source/role/resource gates before any remote job."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.pcqm_metagin import model_parameters
from molgap.pcqm_metagin_screen import (
    ARCHITECTURE, BATCH_SIZE, EPOCHS, FIXED_MANIFEST_SHA256,
    MODEL_ID, ROW_ORDER_FINGERPRINT, SAMPLE_EXPOSURE,
)
from molgap.screen_policy import canonical_fingerprint


ROOT = REPO_ROOT / "experiments/pcqm_metagin_2d_100k"


def test_frozen_screen_matches_reference_scientific_fields():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    reference = json.loads((REPO_ROOT / contract["reference_bundle"]).read_text())
    identity = reference["comparison_identity"]
    assert contract["benchmark_id"] == identity["benchmark_identity"]
    assert contract["fixed_manifest_sha256"] == FIXED_MANIFEST_SHA256
    assert contract["row_order_sha256"] == identity["row_order_identity"] == ROW_ORDER_FINGERPRINT
    assert contract["physical_batch_per_device"] == identity["physical_batch_per_device"] == BATCH_SIZE
    assert contract["epochs"] == EPOCHS
    assert contract["optimizer_steps"] == identity["optimizer_steps"] == 31_240
    assert contract["sample_presentations"] == identity["sample_presentations"] == SAMPLE_EXPOSURE
    assert contract["precision"] == identity["precision"] == "fp32"
    assert contract["tf32_enabled"] is identity["tf32_enabled"] is False
    assert all(contract[f"{role}_role_read"] is False for role in (
        "official_validation", "test_dev", "test_challenge"
    ))
    assert contract["candidate"] == MODEL_ID
    assert model_parameters() == 5_268_481
    assert ARCHITECTURE["geometry"] is False
    assert canonical_fingerprint(ARCHITECTURE) != identity["architecture_config_identity"]


def test_cpu_gpu_kernels_have_only_frozen_inputs_and_separate_resources():
    cpu = json.loads((ROOT / "kaggle_cpu/kernel-metadata.json").read_text())
    gpu = json.loads((ROOT / "kaggle_gpu/kernel-metadata.json").read_text())
    assert cpu["enable_gpu"] is False and gpu["enable_gpu"] is True
    assert cpu["dataset_sources"] == [
        "kaseichou/molgap-metagin-2d-source",
        "kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
    ]
    assert gpu["dataset_sources"] == [*cpu["dataset_sources"],
        "kaseichou/molgap-metagin-2d-hop-cache-v1"]
    assert cpu["competition_sources"] == gpu["competition_sources"] == []
    assert cpu["model_sources"] == gpu["model_sources"] == []
