"""Synthetic modular lifecycle fixtures, adapted from server 938ddc11.

No real records, model assets, protected roles or platform API are consumed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

from molgap import experiment_launch as launch
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint
from test_research_memory_plan_batch import _item
from test_terminal_trace_closure import create_candidate_arm, setup_mock_repo


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _synthetic_policy(repo: Path) -> dict:
    source = Path(__file__).resolve().parents[1] / (
        "research_memory/policies/pcqm-k1-complete-module-audit.1.json"
    )
    policy = json.loads(source.read_bytes())
    target = repo / "research_memory/policies/synthetic.v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json_bytes(policy))
    return policy


def _prepare_reference_inputs(repo: Path) -> None:
    """Add the local pointers used by prospective planning."""
    setup_mock_repo(repo)
    ref = repo / "experiments/ref_exp"
    (ref / "role_plan.json").write_bytes(json_bytes({"schema": "role-plan-v1", "synthetic": True}))
    (ref / "budget.json").write_bytes(json_bytes({"schema": "budget-v1", "synthetic": True}))
    _synthetic_policy(repo)


def _plan_input(repo: Path, arm: dict, binding: dict, index: int, policy: dict) -> dict:
    """Build a complete synthetic RML plan input from the real plan schema."""
    label = f"registered-{index}"
    source = _item(label)
    trajectory = source["spec"]["trajectory"]
    action_id = "act-1"
    run_id = f"synthetic-workflow-run:{binding['arm_id']}:downstream"
    cost_id = f"cost-{binding['arm_id']}"
    trajectory.update(
        trajectory_id=binding["trajectory_id"],
        question=f"Synthetic registered workflow question {binding['arm_id']}",
    )
    trajectory["hypothesis"].update(
        hypothesis_id=f"H-{binding['arm_id']}",
        supporting_evidence_ids=["ev-ref-1"],
        expected_native_cost_ref=cost_id,
    )
    trajectory["state_at_start"].update(
        source_commit=_git(repo, "rev-parse", "HEAD"),
        source_config_identity=canonical_fingerprint(arm),
        contract_refs=["experiments/ref_exp/contract.json"],
        reference_ids=["ev-ref-1"],
        prior_evidence_ids=["ev-ref-1"],
        role_snapshot_refs=["experiments/ref_exp/role_plan.json"],
        budget_snapshot_ref="experiments/ref_exp/budget.json",
    )
    trajectory["actions"][0].update(
        action_id=action_id,
        source_commit=_git(repo, "rev-parse", "HEAD"),
        run_ids=[run_id],
        attempt_ids=[binding["arm_id"] + "-v1"],
        evidence_refs=["experiments/ref_exp/v5_evidence.json"],
        cost_event_ids=[cost_id],
    )
    trajectory["decision"].update(
        decision_ref="experiments/ref_exp/decision.md",
        next_allowed_actions=[action_id],
    )
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}

    cost = copy.deepcopy(source["spec"]["costs"][0])
    cost.update(
        cost_event_id=cost_id,
        trajectory_id=binding["trajectory_id"],
        action_id=action_id,
        run_id=run_id,
        attempt_id=binding["arm_id"] + "-v1",
        platform="kaggle",
        hardware="synthetic-cpu",
        evidence_ref="experiments/ref_exp/v5_evidence.json",
    )
    decision_state = {
        "available_actions": [action_id],
        "chosen_action": action_id,
        "policy_id": policy["policy_id"],
        "policy_version": policy["version"],
        "state_timestamp": "2026-10-02T00:00:00Z",
    }
    return {"trajectory": trajectory, "decision_state": decision_state, "costs": [cost]}


def _launch_receipt(spec, package_dir: Path, package_identity: str, repo: Path) -> Path:
    reference = "synthetic-account/synthetic-pair-run"
    binding = launch.build_launch_receipt(
        spec, package_dir, expected_package_identity=package_identity
    )["binding"]
    response = {
        "format": launch.RESPONSE_FORMAT,
        "version": launch.VERSION,
        "mode": "observed",
        "outcome": "accepted",
        "conflict_kind": None,
        "binding": binding,
        "canonical_platform_reference": {"value": reference, "missing_reason": None},
        "platform_version": {"value": "1", "missing_reason": None},
        "physical_runs": {"value": [{
            "run_identity": "synthetic-physical-run",
            "canonical_reference": reference,
            "platform_version": {"value": "1", "missing_reason": None},
            "arm_ids": [arm["arm_id"] for arm in spec.to_dict()["arms"]],
        }], "missing_reason": None},
        "timestamp": {"value": None, "missing_reason": "not_observed"},
        "monitor_paths": {"value": None, "missing_reason": "not_observed"},
    }
    raw = launch.canonical_json(launch.reconcile_platform_response(
        spec, package_dir, launch.canonical_json(response),
        expected_package_identity=package_identity,
    ))
    receipt_dir = repo / "receipts"
    receipt_dir.mkdir()
    return launch.write_launch_receipt(
        raw, receipt_dir, spec, package_dir,
        expected_package_identity=package_identity,
    )


def _terminal_locations(repo: Path, spec: ExperimentSpec, output_roots: dict[str, Path],
                        source_commit: str) -> dict:
    locations = {}
    for arm in spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        binding = next(row for row in spec.to_dict()["prospective"]["arms"]
                       if row["arm_id"] == arm_id)
        run_id = f"synthetic-workflow-run:{arm_id}:downstream"
        candidate = create_candidate_arm(
            repo, f"authority-{arm_id}", binding["trajectory_id"], run_id,
            f"ev-authority-{arm_id}",
        )
        # The planned trajectory is the one authoritative prospective record.
        candidate["traj_path"].unlink()
        terminal = json.loads(candidate["terminal_path"].read_bytes())
        trace = output_roots[arm_id] / "canonical_trace.json"
        relative_trace = trace.resolve().relative_to(repo.resolve()).as_posix()
        artifact = next(row for row in terminal["evidence"]["artifacts"]
                        if row["name"] == "training_trace")
        terminal["artifact_hashes"].pop(artifact["locator"])
        artifact.update(locator=relative_trace, sha256=file_digest(trace))
        terminal["artifact_hashes"][relative_trace] = artifact["sha256"]
        candidate["terminal_path"].write_bytes(json_bytes(terminal))
        locations[arm_id] = {
            "trajectory_id": binding["trajectory_id"],
            "run_id": run_id,
            "trajectory": binding["output"] + "/trajectory.json",
            "terminal": candidate["terminal_path"].resolve().relative_to(repo.resolve()).as_posix(),
            "trace": relative_trace,
        }
    return locations
