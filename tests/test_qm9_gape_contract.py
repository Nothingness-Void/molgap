import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE = ROOT / "src" / "molgap" / "qm9_gape.py"
PROTOCOL = ROOT / "experiments" / "qm9_gape_pretraining" / "protocol.md"
GPU_RUNNER = (
    ROOT / "experiments" / "qm9_gape_pretraining" / "gpu_seed42" / "run_screen.py"
)
CACHE_RUNNER = (
    ROOT / "experiments" / "qm9_gape_pretraining" / "cache_prep" / "run_cache.py"
)


def source():
    return MODULE.read_text(encoding="utf-8")


def test_gape_module_parses_without_model_execution():
    ast.parse(source())
    ast.parse(GPU_RUNNER.read_text(encoding="utf-8"))
    ast.parse(CACHE_RUNNER.read_text(encoding="utf-8"))


def test_screen_uses_exact_batch128_and_one_seed():
    text = source()
    assert "BATCH_SIZE = 128" in text
    assert "SEED = 42" in text
    assert "GAP_EPOCHS = 40" in text
    assert "GAPE_PRETRAIN_EPOCHS = 10" in text


def test_three_arm_and_equal_compute_control_are_explicit():
    text = source()
    assert '"baseline"' in text
    assert '"shuffled_control"' in text
    assert '"matched_gape"' in text
    assert "initial_generator_sha256" in text
    assert "initial_augmented_model_sha256" in text


def test_no_test_or_pcqm_role_is_loaded():
    text = source()
    assert 'roles["test"]' not in text
    assert "official_pcqm_roles_read" in text
    assert "test_role_read" in text


def test_protocol_freezes_same_task_platform_and_batch():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "one Kaggle2 T4x2 task" in text
    assert "physical\nbatch exactly 128" in text
    assert "QM9 test" in text


def test_t4x2_runner_isolates_one_worker_per_visible_gpu():
    text = GPU_RUNNER.read_text(encoding="utf-8")
    assert '"0" if role == "baseline" else "1"' in text
    assert 'torch").cuda.device_count() != 2' in text
    assert "process.wait()" in text
