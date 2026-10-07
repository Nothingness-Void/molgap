import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "experiments/pcqm_k1_colab_execution_profile"
PAYLOAD = ROOT / "platforms/_records/colab/staging/k1-profile-a100-20261007/payload"


def test_gradient_relation_handles_aligned_conflicting_and_zero_cpu_gradients():
    torch = pytest.importorskip("torch")
    from molgap.k1_execution_profile import gradient_relation

    aligned = gradient_relation([torch.tensor([1.0, 0.0])], [torch.tensor([2.0, 0.0])])
    assert aligned["norm_ratio"] == pytest.approx(2.0)
    assert aligned["cosine"] == pytest.approx(1.0)

    conflicting = gradient_relation([torch.tensor([1.0, 0.0])], [torch.tensor([-2.0, 0.0])])
    assert conflicting["norm_ratio"] == pytest.approx(2.0)
    assert conflicting["cosine"] == pytest.approx(-1.0)

    zero_regularizer = gradient_relation([torch.tensor([1.0, 0.0])], [torch.zeros(2)])
    assert zero_regularizer["norm_ratio"] == pytest.approx(0.0)
    assert zero_regularizer["cosine"] is None

    zero_supervised = gradient_relation([torch.zeros(2)], [torch.tensor([1.0, 0.0])])
    assert zero_supervised["norm_ratio"] is None
    assert zero_supervised["cosine"] is None


def test_notebook_code_cells_compile_without_running_them():
    notebook = json.loads((PROFILE / "MolGap_K1_A100_Profile.ipynb").read_text(encoding="utf-8"))
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_prospective_record_is_planned_before_training_shard_decode():
    source = (PROFILE / "prepare.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    plan_calls = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "plan"]
    decode_calls = [node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_PackedGraphDatasetFactory"]
    assert len(plan_calls) == 1
    assert len(decode_calls) == 1
    assert plan_calls[0].lineno < decode_calls[0].lineno

    trajectory = json.loads((PROFILE / "rml/trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["record_mode"] == "prospective"
    assert trajectory["decision"]["outcome"] == "ACTIVE"


def test_payload_binds_source_checkpoint_roles_and_bounded_fp32_profile():
    inputs = json.loads((PROFILE / "inputs.json").read_text(encoding="utf-8"))
    role_plan = json.loads((PROFILE / "role_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROFILE / "payload_manifest.json").read_text(encoding="utf-8"))

    assert len(inputs["sample_source_idx"]) == 4096
    assert len(set(inputs["sample_source_idx"])) == 4096
    assert all(0 <= row < 500000 for row in inputs["sample_source_idx"])
    assert manifest["sample_source_idx"] == inputs["sample_source_idx"]
    assert all(shard["role"] == "train" for shard in manifest["parent_training_shards"])
    assert sum(shard["rows"] for shard in manifest["parent_training_shards"]) == 500000
    assert role_plan["train"] == {
        "range": [0, 500000], "sample_rows": 4096,
        "local_decoded_rows": 500000, "selection_used": False,
    }
    assert all(role_plan[name] == "untouched" for name in (
        "development", "official_validation", "test_dev", "test_challenge"))

    checkpoint = manifest["checkpoint"]
    assert checkpoint["sha256"] == inputs["selected"]["sha256"]
    assert inputs["frozen_source_files"]
    for name, digest in inputs["frozen_source_files"].items():
        assert manifest["files"][name] == digest
    for name in ("selected.pt", "src/molgap/k1_execution_profile.py"):
        actual = hashlib.sha256((PAYLOAD / name).read_bytes()).hexdigest()
        assert actual == manifest["files"][name]
    assert manifest["files"]["selected.pt"] == checkpoint["sha256"]

    worker_source = (PAYLOAD / "src/molgap/k1_execution_profile.py").read_text(encoding="utf-8")
    assert "sha256_file(root / name) != digest" in worker_source
    assert "trajectory[\"record_mode\"] != \"prospective\"" in worker_source
    assert "expected_sha256=state_meta[\"sha256\"]" in worker_source
    assert "expected_source_sha256=state_meta[\"source_sha256\"]" in worker_source
    assert "expected_epoch=48" in worker_source and 'checkpoint_kind="selected"' in worker_source
    assert "not torch.cuda.is_available() or \"A100\" not in torch.cuda.get_device_name(0)" in worker_source
    assert "deadline = started + 1200" in worker_source
    assert "WARMUP, MEASURE, BATCH_SIZE = 5, 24, 128" in worker_source
    assert '"development_rows": []' in worker_source
    assert '"protected_roles": "untouched"' in worker_source

    reproducibility = (ROOT / "src/molgap/training_reproducibility.py").read_text(encoding="utf-8")
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in reproducibility
    assert "torch.backends.cudnn.allow_tf32 = False" in reproducibility
    assert 'torch.set_float32_matmul_precision("highest")' in reproducibility
