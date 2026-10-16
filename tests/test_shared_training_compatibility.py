"""Pin the reviewed mechanism snapshot; branch adoption is recorded separately."""
import ast
import json
from pathlib import Path

from molgap.v4_runtime import normalized_source_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_shared_mechanism_snapshot_matches_reviewed_inventory():
    record = json.loads((ROOT / "docs/operations/SHARED_TRAINING_CORE_FILES.json").read_text())
    assert record["format"] == "molgap-source-files-v1"
    paths = [item["path"] for item in record["files"]]
    assert paths == sorted(set(paths))
    assert {"src/molgap/family_training_" + name + ".py" for name in
            ("core", "data", "recipe", "kernel", "loaders", "session", "runner", "outputs")} <= set(paths)
    for item in record["files"]:
        assert item["path"].startswith("src/molgap/")
        assert normalized_source_sha256(ROOT / item["path"]) == item["sha256"], item["path"]


def test_accelerated_kernel_has_no_legacy_trainer_or_controller_imports():
    tree = ast.parse((ROOT / "src/molgap/family_training_kernel.py").read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {"k1_screen_training", "pcqm_gptrans_v4", "server_control"} & imports
    runner = ast.parse((ROOT / "src/molgap/family_training_runner.py").read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module and
                   node.module.endswith("_infrastructure_comparison") for node in ast.walk(runner))


def test_branch_policy_is_explicit_and_reference_has_no_extra_roles_or_segments():
    from molgap.family_training_policy import mode_policy
    for family in ("neural_atom_k1", "gptrans_t"):
        policy = mode_policy(family, "reference")
        assert policy.segment_passes is None
        assert policy.extra_development_weights == ()
        assert not policy.process_timing and not policy.input_sidecar_required
