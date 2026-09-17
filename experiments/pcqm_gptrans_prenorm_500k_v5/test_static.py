from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def test_v5_contract_is_exact_and_sealed():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["tail_batch_policy"] == "drop_last"
    assert contract["optimizer_steps_per_epoch"] == 3906
    assert contract["total_sample_presentations_per_arm"] == 29_998_080
    assert contract["arms"] == ["reference", "pair_prenorm"]
    assert contract["roles"]["official_validation_role_read"] is False
    assert contract["roles"]["test_dev_role_read"] is False


def test_entrypoint_and_runner_parse():
    ast.parse((ROOT / "run_arm.py").read_text())
    ast.parse((ROOT / "accept.py").read_text())
    ast.parse((REPO / "src/molgap/pcqm_gptrans_prenorm_500k.py").read_text())


def test_slurm_uses_one_dcu_per_arm_and_fixed_cache():
    text = (ROOT / "run_kunshan.slurm").read_text()
    assert "--gres=dcu:Hygon:1" in text
    assert "pcqm4mv2-ogb-fixed-500k-scnet-v1" in text
    assert "--time=16:00:00" in text
    assert "PRELIM" not in text
