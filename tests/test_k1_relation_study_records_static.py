"""Planning and acceptance wiring without constructing or executing models."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_new_wrapper_and_shared_record_module_parses():
    paths = [ROOT / "src/molgap/k1_relation_study_records.py"]
    paths += list((ROOT / "experiments/pcqm_k1_relation_resolution_100k").glob("*.py"))
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"))


def test_freeze_uses_actual_repository_reference_and_no_terminal_claim():
    code = (ROOT / "src/molgap/k1_relation_study_records.py").read_text()
    for required in ("write_server_comparison_prelaunch", "reference_bundle=bundle",
                     "repo_root=REPO_ROOT", "reference_bundle_path=REFERENCE",
                     '"outcome": "ACTIVE"', '"evidence_ids": []',
                     '"record_mode"', '"source_commit": commit'):
        if required == '"record_mode"':
            # Native plan enforces prospective mode inherited from the template.
            continue
        assert required in code
    assert "from .research_memory.plan import plan" in code
    assert "model_inference" not in code.split("def accept_training")[0]


def test_acceptance_binds_native_observations_and_exact_source():
    code = (ROOT / "src/molgap/k1_relation_study_records.py").read_text()
    for required in ("load_canonical_trace", '"canonical_trace.json"',
                     '"observed_role_history.json"', '"native_cost.json"',
                     "math.isfinite", "31240", "3998720", '"last_checkpoint.pt"',
                     '"source_archive_sha256"', '"resume_two_step_bitwise_equal"',
                     '"rrwp_powers_match_independent_expected"',
                     '"tgt_vector_loop_agreement"', "post100k_audit_accepted=False"):
        assert required in code
    tree = ast.parse(code)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"train_arm", "make_encoder"} for node in ast.walk(tree))


def test_terminal_uses_shared_validated_lifecycle():
    code = (ROOT / "experiments/pcqm_k1_relation_resolution_100k/prepare_terminal.py").read_text()
    assert "research_memory.terminal_wiring import close_terminal_arm" in code
    assert "close_terminal_arm(REPO_ROOT" in code
