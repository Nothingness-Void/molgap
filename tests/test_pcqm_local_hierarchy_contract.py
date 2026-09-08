from __future__ import annotations

import ast
import json
from pathlib import Path

from molgap.pcqm_local_hierarchy import (
    BATCH_SIZE,
    FINETUNE_EPOCHS,
    FUNCTIONAL_GROUP_SMARTS,
    LEARNING_RATE,
    PRETRAIN_EPOCHS,
    SCRATCH_EPOCHS,
    VALIDATION_GRAPHS,
    WEIGHT_DECAY,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"


def test_training_contract_is_equal_exposure() -> None:
    assert SCRATCH_EPOCHS == PRETRAIN_EPOCHS + FINETUNE_EPOCHS == 40
    assert BATCH_SIZE == 48
    assert LEARNING_RATE == 1.6e-4
    assert WEIGHT_DECAY == 1e-6
    assert VALIDATION_GRAPHS == 10_000
    assert len(FUNCTIONAL_GROUP_SMARTS) == 12


def test_remote_scripts_parse_and_keep_roles_sealed() -> None:
    paths = [
        EXPERIMENT
        / "kaggle_pcqm_gap100k"
        / "local_hierarchy_label_cache"
        / "run_cache.py",
        EXPERIMENT
        / "kaggle_pcqm_gap100k"
        / "local_hierarchy_seed42"
        / "run_screen.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        assert "official validation" not in source.lower()
        assert "test-dev" not in source.lower()


def test_kaggle_resource_separation() -> None:
    cpu = json.loads(
        (
            EXPERIMENT
            / "kaggle_pcqm_gap100k"
            / "local_hierarchy_label_cache"
            / "kernel-metadata.json"
        ).read_text(encoding="utf-8")
    )
    gpu = json.loads(
        (
            EXPERIMENT
            / "kaggle_pcqm_gap100k"
            / "local_hierarchy_seed42"
            / "kernel-metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert cpu["enable_gpu"] == "false"
    assert gpu["enable_gpu"] == "true"
    assert gpu["machine_shape"] == "NvidiaTeslaT4"
    assert len(gpu["dataset_sources"]) == 3
