from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/molgap/pcqm_local_statistics.py"
RUNNER = ROOT / "src/molgap/pcqm_local_global_runner.py"
ACCEPTANCE = ROOT / "experiments/pcqm_gap_architecture/accept_pcqm100k_local_statistics_seed42.py"
KERNEL_ROOT = ROOT / "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k"


def test_sources_parse_and_zero_start_paths_are_explicit():
    for path in (MODEL, RUNNER, ACCEPTANCE):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source = MODEL.read_text(encoding="utf-8")
    assert "scatter_reduce_" in source
    assert "mean, maximum, minimum, standard_deviation, degree" in source
    assert "nn.init.zeros_(self.value.weight)" in source
    assert "nn.init.zeros_(self.retention_value.weight)" in source


def test_runner_freezes_two_distinct_modes_and_parameter_counts():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"pna_statistics_graphstate"' in source
    assert '"edge_retention_graphstate"' in source
    assert "3_724_755" in source
    assert "3_743_281" in source
    assert "SEARCH_BUDGET_S = 14_400" in source
    assert "EXPECTED_GPU_COUNT = 2" in source


def test_kaggle_manifests_use_separate_accounts_and_t4x2():
    pna = json.loads((KERNEL_ROOT / "pna_statistics_graphstate_seed42/kernel-metadata.json").read_text(encoding="utf-8"))
    retention = json.loads((KERNEL_ROOT / "edge_retention_graphstate_seed42/kernel-metadata.json").read_text(encoding="utf-8"))
    assert pna["id"].startswith("nothingnessvoid/")
    assert retention["id"].startswith("kaseichou/")
    for manifest in (pna, retention):
        assert manifest["machine_shape"] == "NvidiaTeslaT4"
        assert manifest["enable_gpu"] == "true"
        assert manifest["is_private"] == "true"


def test_entry_points_freeze_seed_and_modes():
    pna = (KERNEL_ROOT / "pna_statistics_graphstate_seed42/run_screen.py").read_text(encoding="utf-8")
    retention = (KERNEL_ROOT / "edge_retention_graphstate_seed42/run_screen.py").read_text(encoding="utf-8")
    assert 'MOLGAP_LOCAL_GLOBAL_SEED"] = "42"' in pna
    assert 'MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "pna_statistics_graphstate"' in pna
    assert 'MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "edge_retention_graphstate"' in retention
