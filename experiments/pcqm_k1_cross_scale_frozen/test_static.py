"""Contract checks that require neither torch nor a remote accelerator."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/molgap/pcqm_k1_cross_scale_diagnostic.py"
ACCEPTANCE = ROOT / "experiments/pcqm_k1_cross_scale_frozen/accept_and_analyze.py"


def test_no_optimizer_or_training_code():
    source = MODULE.read_text(encoding="utf-8")
    assert "torch.no_grad()" in source
    assert "model.load_state_dict(state, strict=True)" in source
    assert "optimizer =" not in source
    assert ".backward(" not in source
    assert "model.train(" not in source


def test_reproduction_gate_precedes_500k_role():
    source = MODULE.read_text(encoding="utf-8")
    assert '(("original_100k", cache_100k), ("unseen_500k", cache_500k))' in source
    assert 'raise RuntimeError(f"{arm} original checkpoint failed prediction reproduction' in source
    assert "500_000, 550_000" in source
    assert 'shards = [item for item in manifest["geometry_shards"] if item["role"] == "development"]' in source


def test_atomic_chunk_and_no_protected_roles():
    source = MODULE.read_text(encoding="utf-8")
    assert "atomic_torch_save(chunk_path, row)" in source
    assert "sha256_file(chunk_path)" in source
    assert '"official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"' in source


def test_acceptance_is_read_only_and_binds_all_cells():
    source = ACCEPTANCE.read_text(encoding="utf-8")
    assert '"original_100k"' in source and '"unseen_500k"' in source
    assert 'if len(terminal.get("chunk_sha256", {})) != 40:' in source
    assert 'torch.equal(payload["target"].view(-1).float(), unseen[arm]["target_eV"])' in source
    assert '"model_inference_executed_in_acceptance": False' in source
    assert "model.load_state_dict" not in source
