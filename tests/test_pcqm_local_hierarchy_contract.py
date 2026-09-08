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
    _capture_rng_state,
    _restore_rng_state,
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


def test_scnet_entrypoint_and_templates_are_portable() -> None:
    runner = ROOT / "platforms" / "scnet" / "run_pcqm_local_hierarchy.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    for name in (
        "pcqm_local_hierarchy_labels_kunshan.slurm",
        "pcqm_local_hierarchy_kunshan.slurm",
        "pcqm_local_hierarchy_xian.slurm",
    ):
        source = (ROOT / "platforms" / "scnet" / name).read_text(
            encoding="utf-8"
        )
        assert '"$HOME"/*' in source
        assert "scnaqkfcy3" not in source
        assert "acf9jvb3sm" not in source
        if "labels" not in name:
            assert "--gres=dcu:Hygon:1" in source
            assert "--time=12:00:00" in source


def test_rng_state_is_restorable() -> None:
    import random

    import numpy as np
    import torch

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    shuffle = torch.Generator().manual_seed(11)
    mask = torch.Generator().manual_seed(13)
    state = _capture_rng_state(
        shuffle_generator=shuffle, mask_generator=mask
    )
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(1),
        torch.rand(1, generator=shuffle),
        torch.rand(1, generator=mask),
    )
    random.random()
    np.random.random()
    torch.rand(3)
    _restore_rng_state(
        state, shuffle_generator=shuffle, mask_generator=mask
    )
    observed = (
        random.random(),
        float(np.random.random()),
        torch.rand(1),
        torch.rand(1, generator=shuffle),
        torch.rand(1, generator=mask),
    )
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    for left, right in zip(observed[2:], expected[2:]):
        assert torch.equal(left, right)


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
