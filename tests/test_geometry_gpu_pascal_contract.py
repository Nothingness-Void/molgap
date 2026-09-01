from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "experiments"
    / "pcqm_gap_architecture"
    / "kaggle_pcqm_gap100k"
    / "geometry_bottom_fusion_seed42"
    / "run_geometry_screen.py"
)


def test_geometry_pascal_compatibility_bootstrap_is_pinned() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'capability != (6, 0)' in source
    assert '"sm_60" in set(torch.cuda.get_arch_list())' in source
    assert '"torch==2.7.1"' in source
    assert '"nvidia-cusparselt-cu12==0.6.3"' in source
    assert '"https://download.pytorch.org/whl/cu126"' in source
    assert '"torch==2.5.1"' not in source
    assert '"https://download.pytorch.org/whl/cu124"' not in source


def test_geometry_scientific_contract_is_unchanged_in_runner() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "SEED = 42" in source
    assert "BATCH_SIZE = 48" in source
    assert "LEARNING_RATE = 1.6e-4" in source
    assert "WEIGHT_DECAY = 1.0e-6" in source
    assert "MAX_EPOCHS = 40" in source
    assert "PATIENCE = 8" in source
    assert "PARAMETER_BUDGET = 5_200_000" in source
    assert '"target": "gap"' in source
    assert '"precision": "fp32"' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
