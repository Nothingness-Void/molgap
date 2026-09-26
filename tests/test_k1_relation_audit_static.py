"""Static NO_TRAIN boundaries; remote model execution is intentionally absent."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/pcqm_k1_relation_resolution_100k"


def test_audit_reuses_reference_and_opens_500k_after_reproduction():
    text = (ROOT / "src/molgap/k1_relation_audit.py").read_text()
    tree = ast.parse(text)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Name) and n.func.id in {"train_arm", "AdamW"} for n in calls)
    assert '"reference_model_inference_executed": False' in text
    assert text.index('raise ValueError(f"Original-role reproduction failed') < text.index('unseen = _graphs(cache_500k')
    assert 'tuple(spec["modes"]) != MODES' in text
    assert 'old_summary["source_acceptance_sha256"]' in text


def test_gpu_budget_is_allocated_device_time_and_environment_precedes_cuda():
    text = (EXP / "kaggle_audit/run.py").read_text()
    ast.parse(text)
    assert 'spec["max_allocated_device_seconds"] / len(devices)' in text
    assert text.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]') < text.index('module.run(')
    assert 'filter="data"' in text
    assert 'timeout=remaining' in text
    assert '"optimizer_steps": 0' in text


def test_dataset_owner_roles_and_exact_slug():
    meta = json.loads((EXP / "kaggle_audit/kernel-metadata.json").read_text())
    assert meta["id"] == "kaseichou/molgap-k1-relation-audit-s42"
    assert all(s.startswith("kaseichou/") for s in meta["dataset_sources"])
    assert meta["is_private"] is True and not meta["kernel_sources"] and not meta["competition_sources"]
    assert len(meta["dataset_sources"]) == 4
