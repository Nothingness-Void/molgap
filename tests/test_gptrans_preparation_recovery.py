"""CPU retry layout/identity checks; no data, model or platform execution."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/pcqm_gptrans_author_alignment"


def test_recovery_preserves_input_science_and_has_separate_identity():
    original = json.loads((BASE / "preparation_contract.json").read_text())
    recovery = json.loads((BASE / "verification_recovery/contract.json").read_text())
    assert all(recovery[k] == v for k, v in original.items())
    assert recovery["recovery_source"]["version"] == 1
    assert len(recovery["recovery_sidecar_manifest_sha256"]) == 64
    metadata = json.loads((BASE / "verification_recovery/kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["id"] != recovery["recovery_source"]["kernel"]
    assert metadata["kernel_sources"] == [recovery["recovery_source"]["kernel"]]


def test_entry_reuses_verified_chunks_without_building_in_recovery_branch():
    entry = (BASE / "kaggle_prepare/run.py").read_text()
    assert entry.count("__PIN_SOURCE_ARCHIVE_SHA256__") == 1
    tree = ast.parse(entry)
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == "recovery")
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
             for statement in branch.body for n in ast.walk(statement)
             if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert {"mounted", "validate_sidecar", "copytree"} <= calls
    assert "build_sidecar" not in calls
    assert 'dataset_root=manifest.parent, rederive=True' in entry
    ast.parse((BASE / "stage_preparation.py").read_text())
