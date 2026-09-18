import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_gptrans_pair_norm_500k_profile"


def _profile_module():
    path = EXPERIMENT / "run_profile.py"
    spec = importlib.util.spec_from_file_location("gptrans_500k_profile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_is_bounded_and_keeps_v5_batch_identity():
    module = _profile_module()
    assert module.BATCH_SIZE == 128
    assert module.WARMUP_STEPS == 10
    assert module.MEASURE_STEPS == 80
    assert module.WARMUP_STEPS + module.MEASURE_STEPS < module.STEPS_PER_EPOCH
    assert {case["workers"] for case in module.CASES} == {0, 2, 4}


def test_profile_uses_kaggle3_fixed_500k_inputs():
    metadata = json.loads((EXPERIMENT / "kernel-metadata.json").read_text())
    assert metadata["id"].startswith("nvoid912/")
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "nvoid912/molgap-gptrans-pair-norm-v5-source-v2",
        "nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1",
    ]

