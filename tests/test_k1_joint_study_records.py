"""Prospective adapter routing checks; never freeze a real plan in tests."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_objective_release_uses_real_bundle_and_shared_plan_apis():
    source = (ROOT / "src/molgap/k1_joint_study_records.py").read_text()
    tree = ast.parse(source)
    functions = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    freeze = functions["freeze"]
    assert 'experiment_purpose="training_objective_comparison"' in freeze
    assert 'declared_intervention_fields=["loss_identity"]' in freeze
    assert "reference_bundle_path=REPO_ROOT / REFERENCE" in freeze
    assert "plan_many(REPO_ROOT" in freeze and 'audit_plan = item("audit")' in freeze
    assert '"automatic_training_successor_authorized": False' in freeze
    assert '"outcome": "ACTIVE"' in freeze
    assert '"result": {"evidence_ids": [], "evidence_refs": []}' in freeze
    package = functions["package"]
    assert "validate_server_scientific_prelaunch(" in package
    assert "build_v4_source_bundle(" in package
    assert "reference_bundle_path=REPO_ROOT / REFERENCE" in package
    assert 'archive_name="source_payload.bin"' in package
    assert "file_digest(REPO_ROOT / TRANSFORM) != TRANSFORM_FILE_SHA" in package
    assert "if output.exists():" in package


def test_thin_entries_have_no_training_or_platform_submitters():
    root = ROOT / "experiments/pcqm_k1_joint_atom_reconstruction_100k"
    for name in ("freeze_release.py", "package_source.py", "accept.py", "bind_submission.py"):
        source = (root / name).read_text()
        ast.parse(source)
        assert "KaggleApi" not in source and "train_arm(" not in source
        assert "torch" not in source
