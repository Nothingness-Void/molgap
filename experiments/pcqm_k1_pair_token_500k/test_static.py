from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_contract_is_exact_v4_scale_bridge():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["tail_batch_policy"] == "drop_last"
    assert contract["epochs"] == 60
    assert contract["optimizer_steps_per_epoch"] == 3906
    assert contract["total_sample_presentations"] == 29_998_080
    assert contract["architecture"]["parameters"] == 3_681_665
    assert contract["roles"]["official_validation_role_read"] is False
    assert contract["roles"]["test_dev_role_read"] is False


def test_slurm_uses_accepted_cache_and_one_dcu():
    text = (ROOT / "run_kunshan.slurm").read_text()
    assert "--gres=dcu:Hygon:1" in text
    assert "pcqm4mv2-ogb-fixed-500k-scnet-v1" in text
    assert "--time=24:00:00" in text


def test_runner_terminalizes_progress_before_completion_manifest():
    source = (
        ROOT.parents[1] / "src/molgap/pcqm_k1_pair_token_500k.py"
    ).read_text(encoding="utf-8")
    terminal = source.index('"status": "COMPLETE"')
    manifest = source.index('output / "completion_manifest.json"')
    assert terminal < manifest
