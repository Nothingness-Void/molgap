"""Metadata-only receipt binding; no model, hardware or platform calls."""

import json

import pytest

from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.research_memory.trace import file_digest, json_bytes
from test_terminal_trace_closure import setup_mock_repo, create_candidate_arm


def diagnostic(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(tmp_path, "diag", "TC-diag", "123", "ev-diag",
                               include_trace_artifact=False)
    traj = json.loads(arm["traj_path"].read_bytes())
    traj["actions"][0].update(run_ids=[], type="planned_NO_TRAIN_diagnostic")
    arm["traj_path"].write_bytes(json_bytes(traj))
    terminal = json.loads(arm["terminal_path"].read_bytes())
    terminal["decision"]["outcome"] = "NO_TRAIN"
    terminal["evidence"]["outcome"].update(comparison_status="context_only", execution_status="complete")
    acceptance_path = tmp_path / terminal["acceptance_ref"]
    acceptance = json.loads(acceptance_path.read_bytes())
    acceptance.update(outcome=terminal["evidence"]["outcome"], trajectory_decision=terminal["decision"])
    acceptance_path.write_bytes(json_bytes(acceptance))
    terminal["artifact_hashes"][terminal["acceptance_ref"]] = file_digest(acceptance_path)
    for artifact in terminal["evidence"]["artifacts"]:
        if artifact["locator"] == terminal["acceptance_ref"]:
            artifact["sha256"] = file_digest(acceptance_path)
    documents = {
        "submission": {"format": "molgap-representation-submission-v1", "owner": "server",
                       "purpose": "NO_TRAIN", "job_id": "123", "source_commit": "1" * 40,
                       "source_archive_sha256": "a" * 64},
        "scheduler": {"job_id": "123", "state": "COMPLETED", "exit_code": "0:0"},
        "started": {"job_id": "123", "source_commit": "1" * 40,
                    "source_archive_sha256": "a" * 64, "training_executed": False,
                    "model_inference_executed": True, "official_validation_role_read": False,
                    "test_dev_role_read": False, "test_challenge_role_read": False},
    }
    binding = {"format": "molgap-no-train-postlaunch-binding-v1", "trajectory_id": "TC-diag",
               "action_id": "act-1", "run_id": "123",
               "original_trajectory_sha256": file_digest(arm["traj_path"])}
    for name, obj in documents.items():
        pointer = f"experiments/diag/{name}.json"
        (tmp_path / pointer).write_bytes(json_bytes(obj))
        digest = file_digest(tmp_path / pointer)
        terminal["artifact_hashes"][pointer] = digest
        binding[name + "_ref"] = pointer
        binding[name + "_sha256"] = digest
    pointer = "experiments/diag/binding.json"
    (tmp_path / pointer).write_bytes(json_bytes(binding))
    terminal["postlaunch_run_binding_ref"] = pointer
    terminal["artifact_hashes"][pointer] = file_digest(tmp_path / pointer)
    arm["terminal_path"].write_bytes(json_bytes(terminal))
    return arm, terminal, binding


def test_diagnostic_binding_preserves_plan_and_is_idempotent(tmp_path):
    arm, _, _ = diagnostic(tmp_path)
    original = arm["traj_path"].read_bytes()
    result = finalize(tmp_path, arm["traj_path"], arm["terminal_path"])
    assert result["status"] == "FINALIZED"
    directory = arm["exp_dir"] / "rml_finalized"
    assert arm["traj_path"].read_bytes() == original
    assert (directory / "prospective_snapshot.json").read_bytes() == original
    assert json.loads((directory / "trajectory.json").read_bytes())["actions"][0]["run_ids"] == ["123"]
    assert "postlaunch_run_binding.json" in verified_receipt(directory)["published_hashes"]
    assert finalize(tmp_path, arm["traj_path"], arm["terminal_path"])["status"] == "ALREADY_FINALIZED"


@pytest.mark.parametrize("attack", ["wrong_plan", "wrong_job", "wrong_source", "missing_file",
                                  "changed_receipt", "existing_run", "train", "strict", "replay",
                                  "trace", "protected", "no_binding"])
def test_diagnostic_binding_attacks_fail_closed(tmp_path, attack):
    arm, terminal, binding = diagnostic(tmp_path)
    if attack == "wrong_plan":
        binding["original_trajectory_sha256"] = "0" * 64
    elif attack in {"wrong_job", "wrong_source", "protected"}:
        path = tmp_path / binding["started_ref"]
        obj = json.loads(path.read_bytes())
        field, value = {"wrong_job": ("job_id", "999"), "wrong_source": ("source_commit", "2" * 40),
                        "protected": ("test_dev_role_read", True)}[attack]
        obj[field] = value
        path.write_bytes(json_bytes(obj))
        binding["started_sha256"] = file_digest(path)
        terminal["artifact_hashes"][binding["started_ref"]] = file_digest(path)
    elif attack == "missing_file":
        (tmp_path / binding["started_ref"]).unlink()
    elif attack == "changed_receipt":
        (tmp_path / binding["started_ref"]).write_bytes(b"{}")
    elif attack == "existing_run":
        obj = json.loads(arm["traj_path"].read_bytes())
        obj["actions"][0]["run_ids"] = ["other"]
        arm["traj_path"].write_bytes(json_bytes(obj))
        binding["original_trajectory_sha256"] = file_digest(arm["traj_path"])
    elif attack == "train":
        terminal["decision"]["outcome"] = "CLOSED"
    elif attack == "strict":
        terminal["evidence"]["outcome"]["comparison_status"] = "strict_causal"
    elif attack == "replay":
        terminal["action_replay"] = {}
    elif attack == "trace":
        terminal["trace_manifest"] = {}
    else:
        terminal.pop("postlaunch_run_binding_ref")
    path = tmp_path / "experiments/diag/binding.json"
    path.write_bytes(json_bytes(binding))
    terminal["artifact_hashes"]["experiments/diag/binding.json"] = file_digest(path)
    arm["terminal_path"].write_bytes(json_bytes(terminal))
    with pytest.raises((ValueError, FileNotFoundError)):
        finalize(tmp_path, arm["traj_path"], arm["terminal_path"])
    assert not (arm["exp_dir"] / "rml_finalized").exists()
