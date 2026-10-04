"""Metadata/synthetic clock checks; no real model or dataset execution."""
import ast
import json
import math
from pathlib import Path

from molgap.pcqm_gptrans_v4 import _ema_decay, _scientific_fields, _weight_decay
from molgap.gptrans_author_screen import validate_arm_allocation
from molgap.gptrans_scale_profile import clock_projection
from molgap.gptrans_author_acceptance import _matches_recomputed_lr_sum

ROOT = Path(__file__).resolve().parents[1]


def test_derived_lr_sum_accepts_only_roundoff_not_schedule_drift():
    expected = 23.842368
    observed = expected
    for _ in range(4):
        observed = math.nextafter(observed, math.inf)
    assert _matches_recomputed_lr_sum(observed, expected)
    assert _matches_recomputed_lr_sum(expected, expected)
    for bad in (expected + 1e-10, expected - 1e-10, expected * 5,
                float("nan"), float("inf"), True, "23.842368"):
        assert not _matches_recomputed_lr_sum(bad, expected)


def test_only_optimizer_fingerprint_changes():
    base = _scientific_fields("degree_scale_ema999")
    candidate = _scientific_fields("degree_decay001_ema999")
    assert {k for k in base if base[k] != candidate[k]} == {"optimizer_fingerprint"}
    assert _weight_decay("degree_decay001_ema999") == .01
    assert _weight_decay("degree_scale_ema999") == .05
    assert _ema_decay("degree_decay001_ema999") == .999


def test_one_candidate_requires_reason():
    assert validate_arm_allocation({"arms": {"degree_decay001_ema999": {}}, "single_arm_reason": "Reuse accepted reference"}) == ("degree_decay001_ema999",)


def test_scale_clock_is_counterfactual_not_fake_result():
    clocks = clock_projection()
    assert clocks["100000"]["optimizer_steps"] == 46860
    assert clocks["500000"]["optimizer_steps"] == 234360
    assert 4.99 < clocks["500000"]["lr_sum"] / clocks["100000"]["lr_sum"] < 5.01
    assert .30 < clocks["100000"]["decay_only_factor"] < .31
    assert .002 < clocks["500000"]["decay_only_factor"] < .003


def test_profile_boundaries_and_manifest_binding():
    contract = json.loads((ROOT / "experiments/pcqm_gptrans_scale_qualification/contract.json").read_text())
    assert contract["profiling_only"] and not contract["qualification_is_compute_release"]
    assert not contract["development_role_read"] and contract["disposable_optimizer_updates"] == 37
    assert contract["physical_batch"] == 128 and contract["maximum_wall_seconds"] == 2700
    source = (ROOT / "src/molgap/gptrans_scale_profile.py").read_text()
    ast.parse(source)
    assert 'if shard["role"] != "train":' in source
    assert "load_roles(" not in source
    assert "_optimizer_step(" in source and "_load_datasets(" in source
    assert "best_model" not in source


def test_new_sources_parse_without_execution():
    for name in ("src/molgap/gptrans_scale_profile.py", "platforms/kaggle/bootstrap_gptrans_profile.py",
        "experiments/pcqm_gptrans_scale_qualification/prepare.py", "experiments/pcqm_gptrans_scale_qualification/run.py",
        "src/molgap/gptrans_scale_profile_acceptance.py", "experiments/pcqm_gptrans_scale_qualification/accept.py",
        "experiments/pcqm_gptrans_decay_clock_100k/freeze.py"):
        ast.parse((ROOT / name).read_bytes())
