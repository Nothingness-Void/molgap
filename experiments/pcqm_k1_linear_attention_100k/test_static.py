"""No-model syntax, telemetry and release-wiring tests."""
import ast
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.k1_linear_attention import MODE, EXPECTED_PARAMETERS
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS
from molgap.research_memory.trace import load_canonical_trace
from molgap.k1_screen_trace import recorder, record_epoch

ROOT = REPO_ROOT / "experiments/pcqm_k1_linear_attention_100k"


def test_registered_single_2d_mechanism():
    config = ARCHITECTURE_CONFIGS[MODE]
    assert config["geometry"] is False and config["teacher"] is False
    assert config["exchange_layers"] == [3, 6, 9]
    assert EXPECTED_PARAMETERS == 3_582_209


def test_syntax_without_importing_or_instantiating_models():
    paths = list(ROOT.rglob("*.py")) + [REPO_ROOT / "src/molgap" / name for name in (
        "k1_linear_attention.py", "k1_screen_trace.py", "k1_portability_audit.py",
        "pcqm_k1_variants.py", "pcqm_k1_variants_runner.py")]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_new_geometry_or_dense_atom_matrix_in_kernel():
    source = (REPO_ROOT / "src/molgap/k1_linear_attention.py").read_text()
    tree = ast.parse(source)
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "kernel_read")
    text = ast.get_source_segment(source, kernel)
    assert "k.transpose(1, 2).bmm(v)" in text
    assert "q.bmm(memory)" in text and "clamp_min(1e-6)" in text
    assert not any(isinstance(n, ast.Name) and n.id == "pos" for n in ast.walk(kernel))
    assert "etkdg" not in source.lower()


def test_fixed_contract_and_single_device_request():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    metadata = json.loads((ROOT / "kaggle_gpu/kernel-metadata.json").read_text())
    assert contract["arms"] == [MODE]
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32" and contract["tf32_enabled"] is False
    assert contract["total_optimizer_steps_per_arm"] == 31240
    assert contract["total_sample_presentations_per_arm"] == 3998720
    assert metadata["id"] == "nothingnessvoid/molgap-k1-linear-attention-s42"
    assert metadata["is_private"] is True and metadata["machine_shape"] == "NvidiaTeslaP100"
    assert len(metadata["dataset_sources"]) == 3


def test_remote_preflight_and_recovery_registered():
    text = (REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py").read_text()
    assert text.count("+ PORTABILITY_MODES + LINEAR_MODES") == 2
    assert "check_linear(model, batch)" in text
    assert "observed_steps += 1" in text and "observed_samples += int(target.numel())" in text
    assert '"canonical_trace.json", "observed_role_history.json"' in text


def test_release_uses_real_bundle_and_native_rml_plan():
    text = (ROOT / "freeze_release.py").read_text()
    assert "repo_root=REPO_ROOT, reference_bundle_path=REFERENCE" in text
    assert "print(plan(REPO_ROOT" in text and "policy_version" in text
    assert '"evidence_ids": [], "evidence_refs": []' in text


def test_remote_audit_binds_exact_candidate_and_no_optimizer():
    entry = (ROOT / "kaggle_gpu/run.py").read_text()
    assert "modes=MODES" in entry and '"--audit"' in entry
    assert entry.index("statuses = {}") < entry.index('env=env, timeout=5400')
    assert "source_payload.bin" in entry and "src.zip" not in entry
    assert "torch==2.4.1+cu121" in entry


def test_cublas_environment_is_in_parent_before_any_torch_import():
    for entry in (ROOT / "kaggle_gpu/run.py", ROOT / "kaggle_audit/run.py"):
        tree = ast.parse(entry.read_text())
        environment = next(node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "CUBLAS_WORKSPACE_CONFIG" for target in node.targets))
        assert environment.value.value == ":4096:8"
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)
                   and any(alias.name == "torch" for alias in node.names)]
        assert imports and all(environment.lineno < node.lineno for node in imports)


def test_recovery_is_frozen_no_train_single_device():
    entry = (ROOT / "kaggle_audit/run.py").read_text()
    spec = json.loads((ROOT / "audit_recovery.json").read_text())
    metadata = json.loads((ROOT / "kaggle_audit/kernel-metadata.json").read_text())
    assert spec["training_authorized"] is False
    assert spec["source_commit"] in entry and spec["source_archive_sha256"] in entry
    assert spec["mode"] == MODE and spec["physical_batch"] == 128
    assert spec["precision"] == "fp32" and spec["tf32_enabled"] is False
    assert spec["audit_seconds_limit"] == 5400 and "timeout=5400" in entry
    assert "train_arm" not in entry and "optimizer.step" not in entry
    assert "candidate_payload.bin" in entry and 'filter="data"' in entry
    assert metadata["id"] == spec["recovery_kernel"] and metadata["is_private"] is True
    assert metadata["competition_sources"] == [] and len(metadata["dataset_sources"]) == 4


def test_split_acceptance_preserves_failed_outputs_and_checkpoint_binding():
    source = (ROOT / "accept.py").read_text()
    assert "def accept_training(" in source and '"--training-only"' in source
    assert '"--audit-root"' in source
    assert 'terminal["checkpoint_sha256"].get(mode) != sha256_file(role_path / "best_model.pt")' in source


def test_observed_trace_records_real_checkpoint_hash_and_terminal(tmp_path):
    checkpoint = tmp_path / "last_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-metadata-not-a-model")
    stream = recorder(tmp_path, "TC-synthetic-static", "synthetic:v1")
    record_epoch(stream, tmp_path, {"epoch": 0, "learning_rate": 0.0004,
        "train_normalized_mae": 0.5, "development_gap_mae_eV": 0.2},
        observed_steps=7, observed_samples=896, elapsed=2.5)
    checkpoint.write_bytes(b"second-synthetic-checkpoint-identity")
    record_epoch(stream, tmp_path, {"epoch": 39, "learning_rate": 0.000001,
        "train_normalized_mae": 0.1, "development_gap_mae_eV": 0.14},
        observed_steps=20, observed_samples=2560, elapsed=3.5)
    trace = load_canonical_trace(tmp_path / "canonical_trace.json")
    assert trace["observations"][0]["optimizer_step"] == 7
    assert trace["observations"][-1]["event"] == "terminal"
    assert trace["observations"][-1]["cumulative_device_time_seconds"] == 6.0
    assert trace["observations"][0]["checkpoint_identity"] != trace["observations"][1]["checkpoint_identity"]
