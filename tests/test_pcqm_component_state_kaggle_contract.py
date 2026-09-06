"""Static ComponentState Kaggle confirmation contract; no model execution."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_gap_architecture"
KERNELS = EXPERIMENT / "kaggle_pcqm_gap100k"


def test_component_confirmation_sources_parse() -> None:
    paths = (
        ROOT / "src/molgap/pcqm_gap_architecture.py",
        ROOT / "src/molgap/pcqm_local_global_runner.py",
        EXPERIMENT / "accept_pcqm100k_component_cache.py",
        EXPERIMENT / "accept_pcqm100k_component_state_seed43.py",
        KERNELS / "component_state_cache/run_component_cache.py",
        KERNELS / "component_state_seed43/run_screen.py",
    )
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_component_confirmation_is_private_t4x2_seed43() -> None:
    root = KERNELS / "component_state_seed43"
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"].startswith("nothingnessvoid/")
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] == "true"
    assert metadata["is_private"] == "true"
    source = (root / "run_screen.py").read_text(encoding="utf-8")
    assert 'MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "conjugated_component_confirmation"' in source
    assert 'MOLGAP_LOCAL_GLOBAL_SEED"] = "43"' in source
    assert "__SOURCE_COMMIT__" in source
    assert "__CACHE_SHA256__" in source


def test_component_cache_is_cpu_only_and_role_sealed() -> None:
    root = KERNELS / "component_state_cache"
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "false"
    assert metadata["is_private"] == "true"
    source = (root / "run_component_cache.py").read_text(encoding="utf-8")
    assert "builder_module().build" in source
    acceptance = (EXPERIMENT / "accept_pcqm100k_component_cache.py").read_text(encoding="utf-8")
    assert "import torch" not in acceptance
    assert '"official_validation_role_read": False' in acceptance
    assert '"test_dev_role_read": False' in acceptance


def test_runner_freezes_component_pair_and_seed() -> None:
    source = (ROOT / "src/molgap/pcqm_local_global_runner.py").read_text(encoding="utf-8")
    for token in (
        '"conjugated_component_confirmation"',
        "ComponentState confirmation mode requires seed 43",
        "3_672_257",
        "3_694_033",
        'EXPECTED_GPU_TOKEN = "T4"',
        '"conjugated_component_state"',
    ):
        assert token in source
