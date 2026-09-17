from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/molgap/pcqm_k1_pair_token_batch_profile.py"
SLURM = Path(__file__).with_name("run_kunshan.slurm")
PROTOCOL = Path(__file__).with_name("protocol.md")
ACCEPTANCE = Path(__file__).with_name("accept_profile.py")


def test_profile_is_train_only_and_non_scientific():
    source = MODULE.read_text(encoding="utf-8")
    assert 'record["role"] != "train"' in source
    assert '"scientific_result_produced": False' in source
    assert '"training_checkpoint_produced": False' in source
    assert '"development_role_read": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "atomic_torch_save" not in source


def test_profile_sweep_and_memory_guard_are_frozen():
    source = MODULE.read_text(encoding="utf-8")
    assert "BATCH_SIZES = (64, 128, 256, 512, 1024, 2048)" in source
    assert "MIN_MEMORY_RESERVE_FRACTION = 0.15" in source
    assert "configure_fp32_determinism(SEED)" in source
    assert "torch.cuda.synchronize()" in source


def test_launcher_uses_accepted_cache_and_short_budget():
    source = SLURM.read_text(encoding="utf-8")
    assert "pcqm4mv2-ogb-fixed-500k-scnet-v1" in source
    assert "#SBATCH --time=01:30:00" in source
    assert "--allocation-seconds" in source
    assert "validation" not in source.lower()


def test_protocol_preserves_bs128_scientific_contract():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "BS128 remains authoritative" in text
    assert "No validation, checkpoint, model bundle, or scientific MAE" in text


def test_acceptance_is_mechanical_and_no_inference():
    source = ACCEPTANCE.read_text(encoding="utf-8")
    assert '"model_inference_executed": False' in source
    assert '"scientific_result_accepted": False' in source
    assert '"v5_profiling_coverage": "PARTIAL"' in source
    assert '"scientific_contract_change_authorized": False' in source
    assert "make_model" not in source
    assert "torch.load" not in source
