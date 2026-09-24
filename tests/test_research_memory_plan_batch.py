from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

from molgap.research_memory.discovery import DiscoveredRecords
from molgap.research_memory.paired import PAIR_SCHEMA


plan_module = import_module("molgap.research_memory.plan")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def planning_root(tmp_path, monkeypatch):
    root = tmp_path
    seed = root / "experiments/seed"
    _write_json(seed / "trajectory.json", {
        "trajectory_id": "T-existing",
        "hypothesis": {"hypothesis_id": "H-existing"},
    })
    _write_json(seed / "v5_evidence.json", {"evidence_id": "E-existing"})
    _write_json(seed / "costs/C-existing.json", {"cost_event_id": "C-existing"})
    for name in ("contract-a.json", "contract-b.json", "roles.json", "budget.json"):
        (root / name).write_text(name, encoding="utf-8")
    for name in ("decision-a.md", "decision-b.md"):
        (root / name).write_text(name, encoding="utf-8")

    discovery_calls = []

    def discover(path):
        assert Path(path) == root
        discovery_calls.append(path)
        return DiscoveredRecords(
            evidence=tuple(sorted(root.glob("experiments/**/v5_evidence.json"))),
            trajectories=tuple(sorted(root.glob("experiments/**/trajectory.json"))),
            costs=tuple(sorted(root.glob("experiments/**/costs/*.json"))),
            roles=(),
            traces=(),
            ready=(),
        )

    monkeypatch.setattr(plan_module, "discover_records", discover)
    monkeypatch.setattr(plan_module.subprocess, "check_output", lambda *args, **kwargs: "b" * 40 + "\n")
    monkeypatch.setattr(
        "molgap.research_memory.policy.load_policy_registry",
        lambda _: [{"policy_id": "fixture-policy", "version": "v1"}],
    )
    return root, discovery_calls


def _item(label: str, output: str | Path | None = None) -> dict:
    trajectory_id = f"T-{label}"
    hypothesis_id = f"H-{label}"
    cost_id = f"C-{label}"
    action_id = f"A-{label}"
    return {
        "output": output if output is not None else f"experiments/arm-{label}",
        "spec": {
            "trajectory": {
                "trajectory_id": trajectory_id,
                "track": "B",
                "owner": "desktop",
                "family_id": f"fixture-{label}",
                "question": "Is the fixture action decision-relevant?",
                "hypothesis": {
                    "hypothesis_id": hypothesis_id,
                    "supporting_evidence_ids": ["E-existing"],
                    "alternative_explanations": ["Fixture alternative"],
                    "related_closed_family_ids": ["fixture-closed"],
                    "observed_deficiency": "Fixture deficiency",
                    "changed_mechanism": "Fixture mechanism",
                    "cheapest_falsifier": "Fixture diagnostic",
                    "expected_native_cost_ref": cost_id,
                    "decision_changed_if_positive": "Continue fixture",
                    "decision_changed_if_negative": "Stop fixture",
                },
                "state_at_start": {
                    "source_commit": "placeholder",
                    "source_config_identity": f"fixture-config-{label}",
                    "contract_refs": [f"contract-{label}.json"],
                    "reference_ids": ["E-existing"],
                    "parent_trajectory_ids": [],
                    "prior_evidence_ids": ["E-existing"],
                    "role_snapshot_refs": ["roles.json"],
                    "budget_snapshot_ref": "budget.json",
                },
                "actions": [{
                    "action_id": action_id,
                    "type": "DIAGNOSTIC",
                    "source_commit": "b" * 40,
                    "run_ids": [f"run-{label}"],
                    "attempt_ids": [f"attempt-{label}"],
                    "evidence_refs": [],
                    "cost_event_ids": [cost_id],
                }],
                "decision": {
                    "outcome": "ACTIVE",
                    "decision_ref": f"decision-{label}.md",
                    "next_allowed_actions": [action_id],
                    "reopen_conditions": [],
                },
            },
            "decision_state": {
                "available_actions": [action_id],
                "chosen_action": action_id,
                "policy_id": "fixture-policy",
                "policy_version": "v1",
                "state_timestamp": "2026-01-01T00:00:00Z",
            },
            "costs": [{
                "schema": "molgap-cost-event-v1",
                "cost_event_id": cost_id,
                "action_id": action_id,
                "run_id": f"run-{label}",
                "attempt_id": f"attempt-{label}",
                "platform": "fixture",
                "hardware": "fixture-cpu",
                "evidence_ref": f"contract-{label}.json",
                "category": "preflight",
                "measurement": {
                    key: {"value": None, "status": "measurement_missing"}
                    for key in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")
                },
            }],
        },
    }


@pytest.mark.parametrize("plans", [
    [],
    [{}],
    [{"output": "experiments/arm-a"}],
    [{"spec": {}, "output": "experiments/arm-a", "extra": 1}],
])
def test_batch_rejects_empty_or_nonexact_items(planning_root, plans):
    root, _ = planning_root
    with pytest.raises(ValueError):
        plan_module.plan_many(root, plans)
    assert not (root / "experiments/arm-a").exists()


@pytest.mark.parametrize("identity", ["trajectory", "hypothesis", "cost"])
def test_batch_rejects_cross_arm_duplicate_ids_before_writing(planning_root, identity):
    root, _ = planning_root
    first, second = _item("a"), _item("b")
    if identity == "trajectory":
        second["spec"]["trajectory"]["trajectory_id"] = "T-a"
        match = "duplicate trajectory_id"
    elif identity == "hypothesis":
        second["spec"]["trajectory"]["hypothesis"]["hypothesis_id"] = "H-a"
        match = "duplicate hypothesis_id"
    else:
        second["spec"]["costs"][0]["cost_event_id"] = "C-a"
        match = "duplicate cost_event_id"
    with pytest.raises(ValueError, match=match):
        plan_module.plan_many(root, [first, second])
    assert not (root / "experiments/arm-a").exists()
    assert not (root / "experiments/arm-b").exists()


@pytest.mark.parametrize("identity", ["trajectory", "hypothesis", "cost"])
def test_batch_rejects_existing_ids_before_writing(planning_root, identity):
    root, _ = planning_root
    item = _item("a")
    if identity == "trajectory":
        item["spec"]["trajectory"]["trajectory_id"] = "T-existing"
    elif identity == "hypothesis":
        item["spec"]["trajectory"]["hypothesis"]["hypothesis_id"] = "H-existing"
    else:
        item["spec"]["costs"][0]["cost_event_id"] = "C-existing"
    with pytest.raises(ValueError, match="duplicate"):
        plan_module.plan_many(root, [item])
    assert not (root / "experiments/arm-a").exists()


def test_batch_shares_frozen_snapshot_and_preserves_input_order(planning_root):
    root, discovery_calls = planning_root
    result = plan_module.plan_many(root, [_item("b"), _item("a")])
    assert len(discovery_calls) == 1
    assert result["status"] == "PLANNED"
    assert result["batch"]["size"] == 2
    assert result["results"] == [
        {"trajectory_id": "T-b", "status": "PLANNED", "path": "experiments/arm-b"},
        {"trajectory_id": "T-a", "status": "PLANNED", "path": "experiments/arm-a"},
    ]
    records = [
        json.loads((root / item["path"] / "trajectory.json").read_text(encoding="utf-8"))
        for item in result["results"]
    ]
    first_state, second_state = (record["decision_state"] for record in records)
    for key in ("known_trajectory_ids", "known_evidence_ids", "source_commit",
                "source_hashes", "policy_sha256"):
        assert first_state[key] == second_state[key]
    assert first_state["known_trajectory_ids"] == ["T-existing"]
    assert first_state["known_evidence_ids"] == ["E-existing"]
    assert first_state["source_commit"] == "b" * 40
    assert records[0]["state_at_start"]["source_commit"] == records[1]["state_at_start"]["source_commit"] == "b" * 40
    assert {"contract-a.json", "contract-b.json", "decision-a.md", "decision-b.md"} <= set(first_state["source_hashes"])
    assert result["batch"]["source_commit"] == first_state["source_commit"]
    assert result["batch"]["policy_sha256"] == first_state["policy_sha256"]
    assert records[0]["trajectory_id"] != records[1]["trajectory_id"]
    assert records[0]["hypothesis"]["hypothesis_id"] != records[1]["hypothesis"]["hypothesis_id"]


def _paired_items():
    reference, candidate = _item("a"), _item("b")
    common = {
        "schema": PAIR_SCHEMA,
        "spec_identity": "a" * 64,
        "logical_run_id": "one-job",
        "reference_arm_id": "a",
        "reference_trajectory_id": "T-a",
        "reference_trajectory_ref": "experiments/arm-a/trajectory.json",
    }
    for item, role, arm_id in ((reference, "reference", "a"), (candidate, "candidate", "b")):
        item["spec"]["trajectory"]["state_at_start"]["same_run_replay"] = {
            **common, "comparison_role": role, "arm_id": arm_id,
        }
    return reference, candidate


def test_same_run_pair_is_frozen_in_one_batch(planning_root):
    root, _ = planning_root
    reference, candidate = _paired_items()
    with pytest.raises(ValueError, match="one frozen multi-arm batch"):
        plan_module.plan(root, reference["spec"], reference["output"])
    result = plan_module.plan_many(root, [reference, candidate])
    assert result["status"] == "PLANNED"
    records = [json.loads((root / item["path"] / "trajectory.json").read_text()) for item in result["results"]]
    assert records[0]["state_at_start"]["same_run_replay"]["comparison_role"] == "reference"
    assert records[1]["state_at_start"]["same_run_replay"]["reference_trajectory_id"] == "T-a"
    assert records[0]["decision_state"]["known_evidence_ids"] == records[1]["decision_state"]["known_evidence_ids"] == ["E-existing"]


def test_same_run_pair_rejects_unbound_peer_before_publication(planning_root):
    root, _ = planning_root
    reference, candidate = _paired_items()
    candidate["spec"]["trajectory"]["state_at_start"]["same_run_replay"]["reference_trajectory_id"] = "T-other"
    with pytest.raises(ValueError, match="does not bind the batch reference"):
        plan_module.plan_many(root, [reference, candidate])
    assert not (root / "experiments/arm-a").exists()


def test_same_run_pair_rejects_different_source_commits(planning_root):
    root, _ = planning_root
    reference, candidate = _paired_items()
    candidate["spec"]["trajectory"]["state_at_start"]["source_commit"] = "2" * 40
    with pytest.raises(ValueError, match="one source commit"):
        plan_module.plan_many(root, [reference, candidate])
    assert not (root / "experiments/arm-a").exists()


@pytest.mark.parametrize("second_output", [
    "experiments/arm-a",
    "experiments/arm-a/.",
    "experiments/existing",
    "other/arm-b",
    "experiments/../escape",
])
def test_batch_rejects_unsafe_or_duplicate_outputs(planning_root, second_output):
    root, _ = planning_root
    (root / "experiments/existing").mkdir()
    with pytest.raises(ValueError):
        plan_module.plan_many(root, [_item("a"), _item("b", second_output)])
    assert not (root / "experiments/arm-a").exists()


def test_batch_rejects_absolute_output(planning_root):
    root, _ = planning_root
    with pytest.raises(ValueError, match="repository-relative"):
        plan_module.plan_many(root, [_item("a", root / "experiments/arm-a")])
    assert not (root / "experiments/arm-a").exists()


def test_single_plan_still_discovers_each_time(planning_root):
    root, discovery_calls = planning_root
    first = plan_module.plan(root, _item("a")["spec"], "experiments/arm-a")
    second = plan_module.plan(root, _item("b")["spec"], "experiments/arm-b")
    assert first == {"trajectory_id": "T-a", "status": "PLANNED", "path": "experiments/arm-a"}
    assert second == {"trajectory_id": "T-b", "status": "PLANNED", "path": "experiments/arm-b"}
    assert len(discovery_calls) == 2
    first_state = json.loads((root / first["path"] / "decision_state.json").read_text(encoding="utf-8"))
    second_state = json.loads((root / second["path"] / "decision_state.json").read_text(encoding="utf-8"))
    assert first_state["known_trajectory_ids"] == ["T-existing"]
    assert second_state["known_trajectory_ids"] == ["T-a", "T-existing"]
    assert str(Path("experiments/arm-a/trajectory.json")) in second_state["source_hashes"]


def test_second_arm_failure_reports_partial_publication_without_rollback(planning_root):
    root, discovery_calls = planning_root
    second = _item("b")
    second["spec"]["costs"][0]["measurement"]["device_hours"] = {
        "value": 1.0, "status": "measured",
    }
    with pytest.raises(plan_module.PlanBatchError, match="item 2/2") as caught:
        plan_module.plan_many(root, [_item("a"), second])
    error = caught.value
    assert error.failed_index == 1
    assert error.completed_results == (
        {"trajectory_id": "T-a", "status": "PLANNED", "path": "experiments/arm-a"},
    )
    assert isinstance(error.__cause__, ValueError)
    assert "prospective cost cannot be measured" in str(error.__cause__)
    assert (root / "experiments/arm-a/trajectory.json").is_file()
    assert (root / "experiments/arm-a/costs/C-a.json").is_file()
    assert not (root / "experiments/arm-b").exists()
    assert len(discovery_calls) == 1
