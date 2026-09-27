"""Metadata/synthetic-array checks; no model or remote execution."""
import ast
import json
from pathlib import Path
import shutil

import pytest

from molgap.k1_relation_audit_records import measured_cost, paired_summary


def test_allocated_cost_includes_unused_device():
    result = measured_cost({"allocated_devices": ["Tesla T4", "Tesla T4"],
        "wall_seconds": 100, "allocated_device_seconds": 200,
        "spec": {"max_allocated_device_seconds": 5400}})
    assert result["device_hours"]["value"] == 200 / 3600
    assert result["queue_hours"]["status"] == "measurement_missing"


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), 99, 6000])
def test_inconsistent_or_unqualified_cost_rejected(seconds):
    with pytest.raises(ValueError):
        measured_cost({"allocated_devices": ["Tesla T4", "Tesla T4"],
            "wall_seconds": 100, "allocated_device_seconds": seconds,
            "spec": {"max_allocated_device_seconds": 5400}})


def test_paired_direction_and_nullable_margins():
    result = paired_summary([.2, .2, .2], [.1, .1, .1])
    assert result["candidate_minus_reference_eV"] == pytest.approx(.1)
    assert result["paired_row_bootstrap_95_eV"][0] > 0
    assert result["candidate_win_fraction"] == 0
    assert result["winning_margin_mean_eV"] is None


@pytest.mark.parametrize("candidate,reference", [([], []), ([.1, .2], [.1]), ([float("nan")], [.1])])
def test_invalid_pair_rejected(candidate, reference):
    with pytest.raises(ValueError):
        paired_summary(candidate, reference)


def test_adapter_has_no_training_or_model_execution():
    module = Path(__file__).resolve().parents[1] / "src/molgap/k1_relation_audit_records.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert not calls.intersection({"_model", "_infer", "make_encoder", "backward", "step", "kernels_push"})


def test_shared_terminal_translation_preserves_historical_bytes(tmp_path):
    """Only retained metadata is copied; historical predictions are never read."""
    from molgap.k1_relation_audit_records import prepare_no_train_terminal
    root = Path(__file__).resolve().parents[1]
    prefix = "experiments/pcqm_k1_relation_resolution_100k/audit"
    original = root / prefix / "results/terminal.json"
    terminal = json.loads(original.read_text())
    for ref in terminal["artifact_hashes"]:
        target = tmp_path / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / ref, target)
    frozen = json.loads((tmp_path / prefix / "rml_plan/trajectory.json").read_text())
    evidence = terminal["evidence"]
    result = prepare_no_train_terminal(prefix=prefix, frozen=frozen, run_id=terminal["run_id"],
        evidence_id=evidence["evidence_id"], outcome=evidence["outcome"], scope=evidence["scope"],
        finalized_at=terminal["finalized_at"], acceptance_name="acceptance_summary",
        artifact_refs=[a["locator"] for a in evidence["artifacts"]],
        authority=evidence["authority"]["pointers"], repo_root=tmp_path)
    for name in ("terminal.json", "terminal_evidence.json", "terminal_acceptance.json",
                 "role_history.json", "cost_records.json"):
        assert (tmp_path / prefix / "results" / name).read_bytes() == (root / prefix / "results" / name).read_bytes()
    assert result["terminal"] == f"{prefix}/results/terminal.json"
    frozen["actions"][0]["run_ids"] = ["different-run"]
    with pytest.raises(ValueError, match="prospective"):
        prepare_no_train_terminal(prefix="unused", frozen=frozen, run_id=terminal["run_id"],
            evidence_id=evidence["evidence_id"], outcome=evidence["outcome"], scope=evidence["scope"],
            finalized_at=terminal["finalized_at"], acceptance_name="acceptance_summary",
            artifact_refs=[], authority=[], repo_root=tmp_path)
    assert not (tmp_path / "unused").exists()
