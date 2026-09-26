"""No model construction: platform isolation and frozen-contract checks."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "experiments/pcqm_k1_relation_resolution_100k"


def test_exact_account_and_data_sources():
    for slot in ("dual", "rrwp"):
        record = json.loads((REL / f"kaggle_{slot}/kernel-metadata.json").read_text())
        assert record["id"].startswith("kaseichou/")
        assert record["dataset_sources"] == [
            "kaseichou/molgap-k1-relation-resolution-source",
            "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"]
        assert record["is_private"] is True
        assert not record["kernel_sources"] and not record["competition_sources"]
    assert json.loads((REL / "kaggle_dual/kernel-metadata.json").read_text())["machine_shape"] == "NvidiaTeslaT4"


def test_training_contract_exposure_and_roles():
    c = json.loads((REL / "training_contract.json").read_text())
    assert c["precision"] == "fp32" and c["tf32_enabled"] is False
    assert c["physical_batch_per_device"] == 128 and c["seed"] == 42
    assert c["total_optimizer_steps_per_arm"] == 40 * (100000 // 128)
    assert c["total_sample_presentations_per_arm"] == c["total_optimizer_steps_per_arm"] * 128
    assert c["train_role"] == [0, 100000]
    assert c["selection_role"] == [100000, 150000]
    assert c["no_train_audit_role"] == [500000, 550000]
    assert c["audit_requires_accepted_training"] is True
    assert len(c["arms"]) == 3


def test_runtime_isolation_trace_and_no_inline_audit():
    code = (ROOT / "src/molgap/k1_relation_study_runtime.py").read_text()
    ast.parse(code)
    for required in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "physical_run_id=", "trajectory_id=",
                     "native_cost.json", "failure.json", "completion_manifest.json", "ThreadPoolExecutor"):
        assert required in code
    assert "k1_portability_audit" not in code
    assert "resume_from=" not in code
    assert "deadline - time.monotonic()" in code
    assert "source_payload.bin" not in code  # only minimal kernel bootstrap discovers a mount


def test_bootstrap_hash_before_extract_and_safe_path_filter():
    for slot in ("dual", "rrwp"):
        code = (REL / f"kaggle_{slot}/run.py").read_text()
        ast.parse(code)
        assert code.index("hashlib.sha256") < code.index("extractall")
        assert 'filter="data"' in code and 'exist_ok=False' in code
        assert '"src.zip"' not in code


def test_new_modes_use_existing_training_loop_and_atomic_trace():
    code = (ROOT / "src/molgap/pcqm_k1_variants_runner.py").read_text()
    assert "+ LINEAR_MODES + RESOLUTION_MODES" in code
    assert "canonical = recorder(output, trajectory_id, physical_run_id)" in code
    assert "check_resolution(model, batch)" in code
    assert "list(model.relation_token.parameters())" in code
    assert "recovery_epoch_" in code
