"""Static boundaries for the one dual-arm objective workload; no model runs."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_joint_launch_matches_fixed_assets_and_two_isolated_arms():
    from molgap.k1_joint_study_runtime import ATTEMPT, RECIPES, RUN_ID, TRAJECTORIES
    root = ROOT / "experiments/pcqm_k1_joint_atom_reconstruction_100k"
    metadata = json.loads((root / "kaggle/kernel-metadata.json").read_text())
    contract = json.loads((root / "training_contract.json").read_text())
    assert metadata["id"] + f":v{ATTEMPT}" == RUN_ID
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert set(contract["arms"]) == set(RECIPES) == set(TRAJECTORIES)
    assert len(set(TRAJECTORIES.values())) == 2
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32" and contract["tf32_enabled"] is False
    assert set(metadata["dataset_sources"]) == {
        "kaseichou/molgap-k1-joint-atom-source",
        contract["fixed_100k_dataset"], contract["fixed_500k_dataset"]}


def test_audit_has_no_optimizer_and_reproduction_precedes_portability():
    path = ROOT / "src/molgap/k1_joint_study_runtime.py"
    source = path.read_text()
    tree = ast.parse(source)
    audit = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "audit")
    text = ast.get_source_segment(source, audit)
    assert "Adam" not in text and ".backward(" not in text and ".step(" not in text
    assert text.index("difference > 0.001") < text.index('MANIFESTS["unseen_500k"]')
    assert '"optimizer_steps_in_audit_stage": 0' in text
    assert '"selection_used_in_audit_stage": False' in text
    assert '"neural_atom_k1_v4"' in text


def test_existing_relation_worker_keeps_backward_compatible_defaults():
    import inspect
    from molgap.k1_relation_study_runtime import worker
    parameters = inspect.signature(worker).parameters
    assert parameters["output_root"].default is None
    assert parameters["child_module"].default == "molgap.k1_relation_study_runtime"
    assert parameters["run_id"].default is None


def test_bootstrap_binds_objective_source_and_transform_before_runtime(tmp_path):
    from molgap.k1_joint_objective import objective_fingerprint
    from molgap.k1_joint_study_runtime import RECIPES, RUN_ID, TRAJECTORIES, verify_source
    source = tmp_path / "extracted"
    (source / "src").mkdir(parents=True)
    (source / "src/probe.py").write_bytes(b"# source\n")
    archive = tmp_path / "source_payload.bin"
    archive.write_bytes(b"test archive")
    (tmp_path / "SOURCE_COMMIT.txt").write_text("a" * 40)
    (tmp_path / "SOURCE_ARCHIVE_SHA256.txt").write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    (tmp_path / "SOURCE_FILES.json").write_text(json.dumps({"files": [{
        "path": "src/probe.py", "sha256": hashlib.sha256(b"# source\n").hexdigest()}]}))
    (tmp_path / "target_transform.json").write_bytes(b"test transform")
    release = {"source_commit": "a" * 40, "run_id": RUN_ID,
        "target_transform_file_sha256": hashlib.sha256(b"test transform").hexdigest(),
        "arms": {r: {"prelaunch_ready": True, "trajectory_id": TRAJECTORIES[r],
                     "loss_identity": objective_fingerprint(r)} for r in RECIPES}}
    (tmp_path / "JOINT_RELEASE.json").write_text(json.dumps(release))
    assert verify_source(source, archive)["source_commit"] == "a" * 40
    release["arms"][RECIPES[0]]["loss_identity"] = objective_fingerprint(RECIPES[1])
    (tmp_path / "JOINT_RELEASE.json").write_text(json.dumps(release))
    with pytest.raises(RuntimeError, match="Objective/trajectory"):
        verify_source(source, archive)
