"""Static protocol checks; no model construction or training."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from molgap.constants import REPO_ROOT


ROOT = REPO_ROOT / "experiments/pcqm_k1_pair_token_node_return_100k"


def test_python_syntax():
    for path in (
        REPO_ROOT / "src/molgap/k1_pair_token.py",
        REPO_ROOT / "src/molgap/pcqm_k1_variants.py",
        ROOT / "run_candidate.py",
        ROOT / "package_source.py",
        ROOT / "freeze_release.py",
        ROOT / "accept.py",
        ROOT / "kaggle_candidate/run_candidate.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"))


def test_contract_is_exact_v5_screen():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    assert contract["physical_batch_per_device"] == 128
    assert contract["seed"] == 42
    assert contract["precision"] == "fp32"
    assert contract["total_optimizer_steps"] == 31_240
    assert contract["total_sample_presentations"] == 3_998_720
    assert contract["architecture"]["parameters"] == 3_687_905
    assert contract["architecture"]["added_vs_pair_token"] == 6_240
    assert contract["roles"]["official_validation_role_read"] is False
    assert contract["roles"]["test_dev_role_read"] is False


def test_single_new_mechanism():
    source = (REPO_ROOT / "src/molgap/k1_pair_token.py").read_text()
    assert "NODE_RETURN_MODE" in source
    assert "self.node_query" in source
    assert "self.return_norm" in source
    assert "nn.init.zeros_(self.return_projection.weight)" in source
    assert "edge_distance" not in source
    assert "shortest_path" not in source


def test_kaggle3_only_and_kaggle1_absent():
    metadata = json.loads((ROOT / "kaggle_candidate/kernel-metadata.json").read_text())
    assert metadata["id"].startswith("nvoid912/")
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert all("nothingnessvoid" not in item for item in metadata["dataset_sources"])
