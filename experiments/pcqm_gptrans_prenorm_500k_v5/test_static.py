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


def test_gptrans_repeatability_uses_preaccepted_bounded_tolerance():
    source = (REPO / "src/molgap/pcqm_gptrans_prenorm_500k.py").read_text()
    assert "MAX_REPEAT_LOSS_DELTA = 1.0e-7" in source
    assert "MAX_REPEAT_PARAMETER_DELTA = 1.0e-7" in source
    assert "calibration_failure.json" in source


def test_slurm_uses_one_dcu_per_arm_and_fixed_cache():
    text = (ROOT / "run_kunshan.slurm").read_text()
    assert "--gres=dcu:Hygon:1" in text
    assert "pcqm4mv2-ogb-fixed-500k-scnet-v1" in text
    assert "--time=16:00:00" in text
    assert "PRELIM" not in text


def test_resume_slurm_preserves_source_and_requires_atomic_checkpoint():
    text = (ROOT / "run_kunshan_resume.slurm").read_text()
    assert "--gres=dcu:Hygon:1" in text
    assert "--time=16:00:00" in text
    assert 'test -f "${RESUME_ROOT}/last_checkpoint.pt"' in text
    assert 'test -f "${RESUME_ROOT}/best_model.pt"' in text
    assert '--source-commit "${SOURCE_COMMIT}"' in text
    assert '--resume "${RESUME_ROOT}"' in text
    assert "--reuse-runtime-certificate" in text
    assert "--execution-source-commit" in text


def test_certified_resume_reuses_evidence_without_loosening_tolerance():
    source = (REPO / "src/molgap/pcqm_gptrans_prenorm_500k.py").read_text()
    assert "reuse_runtime_certificate" in source
    assert "Checkpoint runtime certificate identity changed" in source
    assert "Certified resume runtime fingerprint changed" in source
    assert "MAX_REPEAT_PARAMETER_DELTA = 1.0e-7" in source
