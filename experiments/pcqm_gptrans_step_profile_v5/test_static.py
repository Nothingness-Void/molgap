import ast
import json
from pathlib import Path


ROOT = Path(__file__).parent
REPO = ROOT.parents[1]


def test_contract_is_bounded_train_only_v5_profile():
    contract = json.loads((ROOT / "profiling_contract.json").read_text())
    assert contract["profiling_only"] is True
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["role"] == "train-only"
    assert contract["development_role_read"] is False
    assert contract["official_validation_role_read"] is False
    assert contract["test_dev_role_read"] is False
    assert contract["scientific_contract_change_authorized"] is False


def test_profile_source_is_parseable_and_profiles_all_step_stages():
    source = (REPO / "src/molgap/pcqm_gptrans_step_profile.py").read_text()
    ast.parse(source)
    assert "WARMUP_STEPS = 2" in source
    assert "MEASURED_STEPS = 8" in source
    assert "REPEATS = 3" in source
    assert "FINITE_CHECK_INTERVAL = 50" in source
    assert "MAX_EQUIVALENCE_DELTA = 1.0e-7" in source
    for stage in (
        "loss_finite_sync",
        "gradient_finite_sync",
        "gradient_clip",
        "adamw",
        "ema",
    ):
        assert stage in source
    assert 'roles["validation"]' not in source


def test_variants_change_execution_only():
    source = (REPO / "src/molgap/pcqm_gptrans_step_profile.py").read_text()
    assert 'foreach=True' in source
    assert "torch._foreach_mul_" in source
    assert "torch._foreach_add_" in source
    assert "make_model(\"reference\")" in source
    assert "BATCH_SIZE" in source


def test_launcher_is_short_and_uses_fixed_cache():
    text = (ROOT / "run_kunshan.slurm").read_text()
    assert "#SBATCH --time=01:00:00" in text
    assert "#SBATCH --gres=dcu:Hygon:1" in text
    assert "pcqm4mv2-ogb-fixed-500k-scnet-v1" in text
    assert "--cache-root" in text


def test_acceptance_never_authorizes_science():
    source = (ROOT / "accept.py").read_text()
    ast.parse(source)
    assert '"scientific_status": "NOT_APPLICABLE"' in source
    assert '"scientific_result_accepted": False' in source
