import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_adapter_reuses_worker_and_frozen_runtime():
    source = (ROOT / "platforms/kaggle/training/k1_paired_500k.py").read_text()
    ast.parse(source)
    assert '"molgap.colab_k1_screen", "run"' in source
    assert 'CUDA_VISIBLE_DEVICES="0"' in source
    assert '"torch==2.4.1"' in source
    assert '"numpy==1.26.4"' in source
    assert 'deadline - time.time() - 30' in source
    assert source.count("__PAYLOAD_SHA256__") == 1
    assert "kaggle.json" not in source and "KAGGLE_KEY" not in source


def test_control_preparation_preserves_parent_initialization():
    source = (ROOT / "experiments/pcqm_k1_single_ema_500k/t4_native/prepare.py").read_text()
    ast.parse(source)
    assert "build_initial_state" not in source
    assert 'sha256_file(initial) != parent_inputs["initial_file_sha256"]' in source
    assert 'allocation_wall_ceiling_seconds=32400' in source
    assert '"allocation_wall_limit_seconds": 32400' in source
