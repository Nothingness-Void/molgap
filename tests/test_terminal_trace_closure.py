"""Tests for terminal-side trace closure and orchestration contract.

Verifies that:
1. Retained trace declared in evidence is automatically resolved and canonicalized,
   preventing silent finalization with trace_status=unavailable when --trace is omitted.
2. Missing or corrupt retained trace files fail closed.
3. Multi-arm runs (e.g. conditional-flow) execute independent 1:1 finalizations.
4. Cross-arm trace ingestion is rejected with fail-closed identity mismatch.
5. Genuine no-trace experiments legally finalize with trace_status=unavailable.
6. CLI terminal-pipeline enforces the fail-closed trace guard.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from molgap.constants import REPO_ROOT
from molgap.evidence_pointers import load_json_object
from molgap.research_memory.finalize import verified_receipt
from molgap.research_memory.terminal_wiring import (
    build_default_trace_manifest,
    close_terminal_arm,
    close_terminal_multi_arm,
    inspect_trace_retention_evidence,
    is_trace_artifact,
    resolve_trace_for_terminal_arm,
)
from molgap.research_memory.trace import (
    canonicalize_trace,
    file_digest,
    json_bytes,
    load_canonical_trace,
)
from molgap.research_memory.validate import validate_repository_records

REPO_ROOT_PATH = Path(REPO_ROOT).resolve()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def setup_mock_repo(temp_dir: Path) -> dict[str, Path]:
    """Create minimal mock repository with valid schemas and reference experiment."""
    schemas_dst = temp_dir / "research_memory/schemas"
    schemas_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT_PATH / "research_memory/schemas", schemas_dst)

    # Reference experiment for prospective hypothesis link
    ref_dir = temp_dir / "experiments/ref_exp"
    ref_dir.mkdir(parents=True, exist_ok=True)

    contract_text = json.dumps({"schema": "contract-v1", "name": "contract-ref"}) + "\n"
    (ref_dir / "contract.json").write_text(contract_text, encoding="utf-8")
    decision_text = "# Decision\nAccepted reference.\n"
    (ref_dir / "decision.md").write_text(decision_text, encoding="utf-8")

    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": "ev-ref-1",
        "track": "B",
        "scope": "100k",
        "legacy_contract": "molgap-test-contract",
        "outcome": {
            "execution_status": "complete",
            "artifact_status": "durable_local_verified",
            "comparison_status": "context_only",
            "scientific_status": "POSITIVE_UNDER_CONTRACT",
            "transfer_status": "reusable_reference",
            "budget_decision": "closed",
            "full_handoff_status": "not_applicable",
        },
        "artifacts": [
            {
                "locator": "experiments/ref_exp/decision.md",
                "name": "decision",
                "availability": "durable_local_verified",
                "sha256": sha256_bytes(decision_text.encode()),
            }
        ],
        "authority": {"pointers": ["experiments/ref_exp/decision.md"]},
        "role_use": {
            "official_validation": "untouched",
            "test_dev": "untouched",
            "test_challenge": "untouched",
        },
        "migration": {
            "migrated_at": "2026-09-21",
            "verification_scope": "Test reference",
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
        },
    }
    (ref_dir / "v5_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )

    ref_traj = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": "TB-ref-exp",
        "record_mode": "retrospective_partial",
        "track": "B",
        "owner": "historical",
        "family_id": "F-ref",
        "question": "Reference trajectory",
        "hypothesis": {
            "hypothesis_id": "H-ref",
            "supporting_evidence_ids": [],
            "alternative_explanations": [],
            "related_closed_family_ids": [],
            "historical_unknowns": ["None"],
        },
        "state_at_start": {
            "source_commit": "0" * 40,
            "source_config_identity": "cfg-ref",
            "contract_refs": ["experiments/ref_exp/contract.json"],
            "reference_ids": [],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": [],
            "role_snapshot_refs": [],
        },
        "actions": [
            {
                "action_id": "act-ref",
                "type": "TRAIN",
                "source_commit": "0" * 40,
                "run_ids": ["run-ref-1"],
                "attempt_ids": ["att-ref-1"],
                "evidence_refs": ["experiments/ref_exp/v5_evidence.json"],
                "cost_event_ids": [],
            }
        ],
        "result": {
            "evidence_ids": ["ev-ref-1"],
            "evidence_refs": ["experiments/ref_exp/v5_evidence.json"],
        },
        "decision": {
            "outcome": "POSITIVE_UNDER_CONTRACT",
            "final": True,
            "decision_ref": "experiments/ref_exp/decision.md",
            "next_allowed_actions": [],
            "reopen_conditions": [],
        },
    }
    (ref_dir / "trajectory.json").write_text(
        json.dumps(ref_traj, indent=2) + "\n", encoding="utf-8"
    )

    cost_event = {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": "cost-test-1",
        "trajectory_id": "TB-ref-exp",
        "action_id": "act-ref",
        "run_id": "run-ref-1",
        "attempt_id": "att-ref-1",
        "platform": "kaggle",
        "hardware": "Tesla_T4",
        "category": "training",
        "evidence_ref": "experiments/ref_exp/v5_evidence.json",
        "measurement": {
            "device_hours": {"value": 1.0, "status": "measured"},
            "cpu_hours": {"value": None, "status": "not_applicable"},
            "wall_hours": {"value": 1.1, "status": "measured"},
            "queue_hours": {"value": None, "status": "not_applicable"},
        },
    }
    costs_dir = ref_dir / "costs"
    costs_dir.mkdir(parents=True, exist_ok=True)
    (costs_dir / "cost-test-1.json").write_text(
        json.dumps(cost_event, indent=2) + "\n", encoding="utf-8"
    )

    return {"temp_dir": temp_dir, "ref_dir": ref_dir}


def create_candidate_arm(
    temp_dir: Path,
    exp_name: str,
    traj_id: str,
    run_id: str,
    ev_id: str,
    *,
    include_trace_artifact: bool = True,
    trace_artifact_name: str = "training_trace",
    trace_relative_path: str = "raw_trace.json",
    create_trace_file: bool = True,
    manifest_in_terminal: bool = False,
) -> dict[str, Any]:
    """Create a prospective candidate experiment arm in mock repo."""
    exp_dir = temp_dir / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    contract_path = exp_dir / "contract.json"
    contract_bytes = (
        json.dumps(
            {
                "schema": "contract-v1",
                "name": f"contract-{exp_name}",
                "target": "Gap",
                "metric": "MAE",
                "unit": "eV",
                "direction": "minimize",
            }
        )
        + "\n"
    ).encode()
    contract_path.write_bytes(contract_bytes)

    role_plan_path = exp_dir / "role_plan.json"
    role_plan_bytes = (json.dumps({"schema": "role-plan-v1", "plan": "testing"}) + "\n").encode()
    role_plan_path.write_bytes(role_plan_bytes)

    budget_path = exp_dir / "budget.json"
    budget_bytes = (json.dumps({"schema": "budget-v1", "budget": "testing"}) + "\n").encode()
    budget_path.write_bytes(budget_bytes)

    decision_path = exp_dir / "decision.md"
    decision_bytes = f"# Terminal Decision for {traj_id}\nNEGATIVE_UNDER_CONTRACT.\n".encode()
    decision_path.write_bytes(decision_bytes)

    evidence_outcome = {
        "execution_status": "complete",
        "artifact_status": "durable_local_verified",
        "comparison_status": "context_only",
        "scientific_status": "NEGATIVE_UNDER_CONTRACT",
        "transfer_status": "stop",
        "budget_decision": "closed",
        "full_handoff_status": "not_applicable",
    }

    acceptance_path = exp_dir / "acceptance.json"
    acceptance = {
        "format": "acceptance-v1",
        "evidence_id": ev_id,
        "run_id": run_id,
        "outcome": evidence_outcome,
        "trajectory_decision": {
            "outcome": "NEGATIVE_UNDER_CONTRACT",
            "final": True,
            "decision_ref": f"experiments/{exp_name}/decision.md",
            "next_allowed_actions": ["CLOSE"],
            "reopen_conditions": [],
        },
        "role_use": {},
    }
    acceptance_bytes = (json.dumps(acceptance, indent=2) + "\n").encode()
    acceptance_path.write_bytes(acceptance_bytes)

    traj = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": traj_id,
        "record_mode": "prospective",
        "track": "B",
        "owner": "server",
        "family_id": f"F-{exp_name}",
        "question": f"Question for {traj_id}",
        "hypothesis": {
            "hypothesis_id": f"H-{traj_id}",
            "supporting_evidence_ids": ["ev-ref-1"],
            "alternative_explanations": ["noise"],
            "related_closed_family_ids": ["F-ref"],
            "observed_deficiency": "Deficiency observed",
            "changed_mechanism": "Changed mechanism",
            "cheapest_falsifier": "Falsifier",
            "expected_native_cost_ref": "cost-test-1",
            "decision_changed_if_positive": "Promote",
            "decision_changed_if_negative": "Close",
        },
        "state_at_start": {
            "source_commit": "1" * 40,
            "source_config_identity": f"cfg-{exp_name}",
            "contract_refs": [f"experiments/{exp_name}/contract.json"],
            "reference_ids": ["ev-ref-1"],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": ["ev-ref-1"],
            "role_snapshot_refs": [f"experiments/{exp_name}/role_plan.json"],
            "budget_snapshot_ref": f"experiments/{exp_name}/budget.json",
        },
        "actions": [
            {
                "action_id": "act-1",
                "type": "TRAIN",
                "source_commit": "1" * 40,
                "run_ids": [run_id],
                "attempt_ids": ["att-1"],
                "evidence_refs": [f"experiments/{exp_name}/acceptance.json"],
                "cost_event_ids": ["cost-test-1"],
            }
        ],
        "result": {
            "evidence_ids": [],
            "evidence_refs": [],
        },
        "decision": {
            "outcome": "ACTIVE",
            "final": False,
            "decision_ref": f"experiments/{exp_name}/decision.md",
            "next_allowed_actions": ["CLOSE"],
            "reopen_conditions": [],
        },
    }
    traj_path = exp_dir / "trajectory.json"
    traj_path.write_bytes(json_bytes(traj))

    artifacts = [
        {
            "locator": f"experiments/{exp_name}/decision.md",
            "name": "decision",
            "availability": "durable_local_verified",
            "sha256": sha256_bytes(decision_bytes),
        },
        {
            "locator": f"experiments/{exp_name}/acceptance.json",
            "name": "acceptance",
            "availability": "durable_local_verified",
            "sha256": sha256_bytes(acceptance_bytes),
        },
    ]

    artifact_hashes = {
        f"experiments/{exp_name}/contract.json": sha256_bytes(contract_bytes),
        f"experiments/{exp_name}/role_plan.json": sha256_bytes(role_plan_bytes),
        f"experiments/{exp_name}/budget.json": sha256_bytes(budget_bytes),
        f"experiments/{exp_name}/decision.md": sha256_bytes(decision_bytes),
        f"experiments/{exp_name}/acceptance.json": sha256_bytes(acceptance_bytes),
    }

    trace_file_path = exp_dir / trace_relative_path
    if include_trace_artifact:
        raw_rows = [
            {
                "epoch": 0,
                "step": 100,
                "lr": 0.001,
                "train_mae_eV": 0.45,
                "development_mae_eV": 0.40,
                "elapsed_seconds": 15.0,
            },
            {
                "epoch": 1,
                "step": 200,
                "lr": 0.0005,
                "train_mae_eV": 0.35,
                "development_mae_eV": 0.32,
                "elapsed_seconds": 30.0,
            },
        ]
        raw_bytes = json_bytes({"rows": raw_rows})
        trace_digest = sha256_bytes(raw_bytes)

        if create_trace_file:
            trace_file_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file_path.write_bytes(raw_bytes)

        trace_locator = f"experiments/{exp_name}/{trace_relative_path}"
        artifacts.append(
            {
                "locator": trace_locator,
                "name": trace_artifact_name,
                "availability": "durable_local_verified",
                "sha256": trace_digest,
            }
        )
        artifact_hashes[trace_locator] = trace_digest

    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": ev_id,
        "track": "B",
        "scope": "100k",
        "legacy_contract": "molgap-test-contract",
        "outcome": evidence_outcome,
        "artifacts": artifacts,
        "authority": {
            "pointers": [
                f"experiments/{exp_name}/decision.md",
                f"experiments/{exp_name}/acceptance.json",
            ]
        },
        "role_use": {
            "official_validation": "untouched",
            "test_dev": "untouched",
            "test_challenge": "untouched",
        },
        "migration": {
            "migrated_at": "2026-09-21",
            "verification_scope": "Candidate testing",
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
        },
    }

    terminal = {
        "format": "molgap-rml-terminal-package-v1",
        "trajectory_id": traj_id,
        "action_id": "act-1",
        "run_id": run_id,
        "finalized_at": "2026-09-21T15:00:00+09:00",
        "evidence": evidence,
        "acceptance_ref": f"experiments/{exp_name}/acceptance.json",
        "artifact_hashes": artifact_hashes,
        "decision": {
            "outcome": "NEGATIVE_UNDER_CONTRACT",
            "final": True,
            "decision_ref": f"experiments/{exp_name}/decision.md",
            "next_allowed_actions": ["CLOSE"],
            "reopen_conditions": [],
        },
        "role_use": {},
        "roles": [],
        "costs": [],
    }

    if manifest_in_terminal and include_trace_artifact:
        trace_record = canonicalize_trace(
            {
                "trajectory_id": traj_id,
                "run_id": run_id,
                "metric_semantics": {
                    "live_train_metric": {
                        "metric": "MAE",
                        "unit": "eV",
                        "target": "Gap",
                        "role_identity": "train",
                        "weights": "live",
                        "direction": "minimize",
                    },
                    "live_dev_metric": None,
                    "ema_dev_metric": {
                        "metric": "MAE",
                        "unit": "eV",
                        "target": "Gap",
                        "role_identity": "dev",
                        "weights": "ema",
                        "direction": "minimize",
                    },
                },
                "observations": [
                    {
                        "epoch_or_pass": 0,
                        "optimizer_step": 100,
                        "sample_presentations": None,
                        "learning_rate": 0.001,
                        "live_train_metric": 0.45,
                        "ema_dev_metric": 0.40,
                        "wall_time_seconds": 15.0,
                    }
                ],
            }
        )
        terminal["trace_manifest"] = build_default_trace_manifest(
            temp_dir, traj, terminal, trace_record
        )

    terminal_path = exp_dir / "terminal.json"
    terminal_path.write_bytes(json_bytes(terminal))

    return {
        "exp_dir": exp_dir,
        "traj_path": traj_path,
        "terminal_path": terminal_path,
        "trace_file_path": trace_file_path,
        "traj_id": traj_id,
        "run_id": run_id,
        "ev_id": ev_id,
    }


# =========================================================================
# Test A: Retained raw trace auto-resolves and finalizes available
# =========================================================================
def test_declared_trace_auto_resolves_and_finalizes_available(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_auto_trace",
        "TB-auto-trace",
        "run-auto-1",
        "ev-auto-1",
        include_trace_artifact=True,
    )

    # Caller omits trace entirely
    result = close_terminal_arm(
        repo_root=tmp_path,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=None,
    )

    assert result["pipeline_status"] == "COMPLETE"
    assert result["finalization_status"] == "FINALIZED"

    # Verify published receipt
    finalized_dir = arm["exp_dir"] / "rml_finalized"
    assert finalized_dir.is_dir()
    receipt = verified_receipt(finalized_dir)
    assert receipt["trace_status"] == "available"
    assert (finalized_dir / "trace.json").is_file()
    assert (finalized_dir / "trace_manifest.json").is_file()

    # Verify canonical trace content
    trace_obj = load_canonical_trace(finalized_dir / "trace.json")
    assert trace_obj["trajectory_id"] == "TB-auto-trace"
    assert trace_obj["run_id"] == "run-auto-1"
    assert len(trace_obj["observations"]) == 2
    assert trace_obj["observations"][0]["optimizer_step"] == 100


# =========================================================================
# Test B: Retention evidence declares trace, but file missing -> FAIL CLOSED
# =========================================================================
def test_declared_trace_missing_file_fails_closed(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_missing_trace",
        "TB-missing-trace",
        "run-missing-1",
        "ev-missing-1",
        include_trace_artifact=True,
        create_trace_file=False,  # declared in evidence, but file does not exist on disk
    )

    with pytest.raises(FileNotFoundError, match="FAIL CLOSED: declared retained trace file missing"):
        close_terminal_arm(
            repo_root=tmp_path,
            trajectory=arm["traj_path"],
            terminal=arm["terminal_path"],
            trace=None,
        )

    # Verify nothing was finalized
    assert not (arm["exp_dir"] / "rml_finalized").exists()


# =========================================================================
# Test C: Retained trace file digest mismatch -> FAIL CLOSED
# =========================================================================
def test_declared_trace_digest_mismatch_fails_closed(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_corrupt_trace",
        "TB-corrupt-trace",
        "run-corrupt-1",
        "ev-corrupt-1",
        include_trace_artifact=True,
    )

    # Corrupt trace file on disk
    arm["trace_file_path"].write_bytes(b'{"rows": [{"corrupt": true}]}')

    with pytest.raises(ValueError, match="FAIL CLOSED: retained trace file digest mismatch"):
        close_terminal_arm(
            repo_root=tmp_path,
            trajectory=arm["traj_path"],
            terminal=arm["terminal_path"],
            trace=None,
        )

    assert not (arm["exp_dir"] / "rml_finalized").exists()


# =========================================================================
# Test D: Genuine no-trace experiment legally unavailable
# =========================================================================
def test_genuine_no_trace_experiment_legally_unavailable(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_no_trace",
        "TB-no-trace",
        "run-notrace-1",
        "ev-notrace-1",
        include_trace_artifact=False,
    )

    result = close_terminal_arm(
        repo_root=tmp_path,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=None,
    )

    assert result["pipeline_status"] == "COMPLETE"
    assert result["finalization_status"] == "FINALIZED"

    finalized_dir = arm["exp_dir"] / "rml_finalized"
    assert finalized_dir.is_dir()
    receipt = verified_receipt(finalized_dir)
    assert receipt["trace_status"] == "unavailable"
    assert not (finalized_dir / "trace.json").exists()
    assert not (finalized_dir / "trace_manifest.json").exists()


# =========================================================================
# Test E: Multi-arm independent closures (conditional-flow pattern)
# =========================================================================
def test_multi_arm_independent_finalizations(tmp_path):
    setup_mock_repo(tmp_path)

    # Create two arms: readback and recurrence
    arm1 = create_candidate_arm(
        tmp_path,
        "exp_multi_readback",
        "TB-cf-s42-readback",
        "run-cf-readback",
        "ev-cf-readback",
        include_trace_artifact=True,
        trace_artifact_name="conditional_pair_readback_trace",
        trace_relative_path="raw_trace_readback.json",
    )
    arm2 = create_candidate_arm(
        tmp_path,
        "exp_multi_recurrence",
        "TB-cf-s42-recurrence",
        "run-cf-recurrence",
        "ev-cf-recurrence",
        include_trace_artifact=True,
        trace_artifact_name="conditional_pair_recurrence_trace",
        trace_relative_path="raw_trace_recurrence.json",
    )

    multi_arms = [
        {
            "trajectory": arm1["traj_path"],
            "terminal": arm1["terminal_path"],
            "arm_identifier": "readback",
        },
        {
            "trajectory": arm2["traj_path"],
            "terminal": arm2["terminal_path"],
            "arm_identifier": "recurrence",
        },
    ]

    results = close_terminal_multi_arm(tmp_path, multi_arms)
    assert len(results) == 2
    assert results[0]["pipeline_status"] == "COMPLETE"
    assert results[1]["pipeline_status"] == "COMPLETE"

    # Verify Arm 1
    dir1 = arm1["exp_dir"] / "rml_finalized"
    assert dir1.is_dir()
    rec1 = verified_receipt(dir1)
    assert rec1["trace_status"] == "available"
    trace1 = load_canonical_trace(dir1 / "trace.json")
    assert trace1["trajectory_id"] == "TB-cf-s42-readback"
    assert trace1["run_id"] == "run-cf-readback"

    # Verify Arm 2
    dir2 = arm2["exp_dir"] / "rml_finalized"
    assert dir2.is_dir()
    rec2 = verified_receipt(dir2)
    assert rec2["trace_status"] == "available"
    trace2 = load_canonical_trace(dir2 / "trace.json")
    assert trace2["trajectory_id"] == "TB-cf-s42-recurrence"
    assert trace2["run_id"] == "run-cf-recurrence"


# =========================================================================
# Test F: Cross-arm trace mismatch fails closed
# =========================================================================
def test_cross_arm_trace_mismatch_fails_closed(tmp_path):
    setup_mock_repo(tmp_path)

    arm1 = create_candidate_arm(
        tmp_path,
        "exp_arm1",
        "TB-arm1",
        "run-arm1",
        "ev-arm1",
        include_trace_artifact=True,
    )
    arm2 = create_candidate_arm(
        tmp_path,
        "exp_arm2",
        "TB-arm2",
        "run-arm2",
        "ev-arm2",
        include_trace_artifact=True,
    )

    # First auto-resolve Arm 2's trace so we have a canonical trace
    _, arm2_trace = resolve_trace_for_terminal_arm(
        tmp_path, arm2["traj_path"], arm2["terminal_path"]
    )
    assert arm2_trace.is_file()

    # Attempt to feed Arm 2's trace into Arm 1
    with pytest.raises(ValueError, match="FAIL CLOSED:.*trajectory_id mismatch"):
        close_terminal_arm(
            repo_root=tmp_path,
            trajectory=arm1["traj_path"],
            terminal=arm1["terminal_path"],
            trace=arm2_trace,
        )


# =========================================================================
# Test G: CLI terminal-pipeline integration tests
# =========================================================================
def test_cli_terminal_pipeline_auto_resolves_trace(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_cli_auto",
        "TB-cli-auto",
        "run-cli-auto",
        "ev-cli-auto",
        include_trace_artifact=True,
    )

    # Invoke CLI terminal-pipeline without --trace
    cmd = [
        sys.executable,
        "-m",
        "molgap.research_memory",
        "--repo-root",
        str(tmp_path),
        "terminal-pipeline",
        "--trajectory",
        f"experiments/exp_cli_auto/trajectory.json",
        "--terminal",
        f"experiments/exp_cli_auto/terminal.json",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT_PATH, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}\n{res.stdout}"
    out = json.loads(res.stdout)
    assert out["pipeline_status"] == "COMPLETE"
    assert out["finalization_status"] == "FINALIZED"

    # Verify receipt on disk
    finalized_dir = arm["exp_dir"] / "rml_finalized"
    rec = verified_receipt(finalized_dir)
    assert rec["trace_status"] == "available"


def test_cli_terminal_pipeline_missing_trace_exits_nonzero(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_cli_missing",
        "TB-cli-missing",
        "run-cli-missing",
        "ev-cli-missing",
        include_trace_artifact=True,
        create_trace_file=False,
    )

    cmd = [
        sys.executable,
        "-m",
        "molgap.research_memory",
        "--repo-root",
        str(tmp_path),
        "terminal-pipeline",
        "--trajectory",
        f"experiments/exp_cli_missing/trajectory.json",
        "--terminal",
        f"experiments/exp_cli_missing/terminal.json",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT_PATH, capture_output=True, text=True)
    assert res.returncode != 0
    assert "FAIL CLOSED: declared retained trace file missing" in res.stderr
    assert not (arm["exp_dir"] / "rml_finalized").exists()


def test_cli_terminal_pipeline_genuine_no_trace(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_cli_notrace",
        "TB-cli-notrace",
        "run-cli-notrace",
        "ev-cli-notrace",
        include_trace_artifact=False,
    )

    cmd = [
        sys.executable,
        "-m",
        "molgap.research_memory",
        "--repo-root",
        str(tmp_path),
        "terminal-pipeline",
        "--trajectory",
        f"experiments/exp_cli_notrace/trajectory.json",
        "--terminal",
        f"experiments/exp_cli_notrace/terminal.json",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT_PATH, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}\n{res.stdout}"
    out = json.loads(res.stdout)
    assert out["pipeline_status"] == "COMPLETE"

    finalized_dir = arm["exp_dir"] / "rml_finalized"
    rec = verified_receipt(finalized_dir)
    assert rec["trace_status"] == "unavailable"


# =========================================================================
# Test H: Non-training traces excluded & semantic fields prioritized
# =========================================================================
def test_non_training_traces_are_excluded():
    for non_train in (
        "analysis_trace",
        "debug_trace",
        "inference_trace",
        "profile_trace",
        "eval_trace",
    ):
        assert not is_trace_artifact(
            {"name": non_train, "locator": f"experiments/exp/{non_train}.json"}
        )
        assert not is_trace_artifact(
            {"name": "trace", "locator": f"experiments/exp/{non_train}.json"}
        )
        assert not is_trace_artifact(
            {"name": "trace", "locator": "experiments/exp/trace.json", "purpose": non_train}
        )

    # Semantic fields prioritize training
    assert is_trace_artifact(
        {
            "name": "custom_log",
            "locator": "experiments/exp/custom.dat",
            "artifact_type": "training_trace",
        }
    )
    assert is_trace_artifact(
        {
            "name": "run_log",
            "locator": "experiments/exp/run.json",
            "purpose": "training_trace",
        }
    )


# =========================================================================
# Test I: Default trace manifest derives strictly from contract without guessing
# =========================================================================
def test_build_default_trace_manifest_derives_from_contract_without_guessing(tmp_path):
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_manifest_test",
        "TB-manifest-test",
        "run-m-1",
        "ev-m-1",
        include_trace_artifact=True,
    )
    contract_file = tmp_path / "experiments/exp_manifest_test/contract.json"
    contract_file.write_text(
        json.dumps(
            {
                "format": "custom-contract-v1",
                "benchmark_id": "custom_pcqm_benchmark",
                "model_id": "custom_gptrans_model",
                "data_role_fingerprint": "fp_dataset_custom_123",
                "row_order_fingerprint": "fp_split_custom_456",
                "precision": "bf16",
                "optimizer": "adamw_custom",
            }
        ),
        encoding="utf-8",
    )

    traj_obj = json.loads(arm["traj_path"].read_text(encoding="utf-8"))
    term_obj = json.loads(arm["terminal_path"].read_text(encoding="utf-8"))
    trace_rec = {
        "trajectory_id": "TB-manifest-test",
        "run_id": "run-m-1",
        "observations": [
            {"optimizer_step": 10, "live_train_metric": 0.5, "ema_dev_metric": 0.4}
        ],
    }

    manifest = build_default_trace_manifest(tmp_path, traj_obj, term_obj, trace_rec)
    comp = manifest["comparability_identity"]
    assert comp["dataset_identity"] == "fp_dataset_custom_123"
    assert comp["row_split_identity"] == "fp_split_custom_456"
    assert comp["architecture_identity"] == "custom_gptrans_model"
    assert comp["precision_identity"] == "bf16"
    assert comp["optimizer_identity"] == "adamw_custom"
    assert manifest["backtest_eligibility"]["eligible"] is False


# =========================================================================
# Test J: Hardening Point 1 - Explicit canonical trace bypass prevention
# =========================================================================
def test_explicit_canonical_without_provenance_binding_fails_closed(tmp_path):
    """Declared raw trace in evidence + explicit canonical without provenance -> FAIL CLOSED."""
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_prov_fail",
        "TB-prov-fail",
        "run-pf-1",
        "ev-pf-1",
        include_trace_artifact=True,
    )
    unbound_canonical = arm["exp_dir"] / "unbound_canonical.json"
    trace_rec = canonicalize_trace(
        {
            "trajectory_id": "TB-prov-fail",
            "run_id": "run-pf-1",
            "metric_semantics": {
                "live_train_metric": {
                    "metric": "MAE",
                    "unit": "eV",
                    "target": "Gap",
                    "role_identity": "internal_train",
                    "weights": "live",
                    "direction": "minimize",
                },
                "live_dev_metric": None,
                "ema_dev_metric": None,
            },
            "observations": [
                {"sequence": 0, "event": "observation", "optimizer_step": 1, "live_train_metric": 0.5}
            ],
            "provenance": [
                {
                    "source": "some/other/path/raw.json",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                }
            ],
        }
    )
    unbound_canonical.write_bytes(json_bytes(trace_rec))

    with pytest.raises(
        ValueError,
        match="FAIL CLOSED: explicit canonical trace provenance does not bind to declared trace artifact",
    ):
        close_terminal_arm(
            repo_root=tmp_path,
            trajectory=arm["traj_path"],
            terminal=arm["terminal_path"],
            trace=unbound_canonical,
        )


def test_explicit_canonical_with_valid_provenance_binding_succeeds(tmp_path):
    """Declared raw trace in evidence + explicit canonical with matching provenance -> PASS."""
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_prov_pass",
        "TB-prov-pass",
        "run-pp-1",
        "ev-pp-1",
        include_trace_artifact=True,
    )
    raw_rel = arm["trace_file_path"].relative_to(tmp_path).as_posix()
    raw_sha = file_digest(arm["trace_file_path"])

    bound_canonical = arm["exp_dir"] / "bound_canonical.json"
    trace_rec = canonicalize_trace(
        {
            "trajectory_id": "TB-prov-pass",
            "run_id": "run-pp-1",
            "metric_semantics": {
                "live_train_metric": {
                    "metric": "MAE",
                    "unit": "eV",
                    "target": "Gap",
                    "role_identity": "internal_train",
                    "weights": "live",
                    "direction": "minimize",
                },
                "live_dev_metric": None,
                "ema_dev_metric": None,
            },
            "observations": [
                {"sequence": 0, "event": "observation", "optimizer_step": 1, "live_train_metric": 0.5}
            ],
            "provenance": [
                {
                    "source": raw_rel,
                    "sha256": raw_sha,
                }
            ],
        }
    )
    bound_canonical.write_bytes(json_bytes(trace_rec))

    result = close_terminal_arm(
        repo_root=tmp_path,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=bound_canonical,
    )
    assert result["pipeline_status"] == "COMPLETE"
    receipt = verified_receipt(arm["exp_dir"] / "rml_finalized")
    assert receipt["trace_status"] == "available"


def test_explicit_canonical_declared_itself_matching_digest_succeeds(tmp_path):
    """Declared artifact in evidence is canonical trace + matching digest -> PASS."""
    setup_mock_repo(tmp_path)
    exp_dir = tmp_path / "experiments/exp_canon_self"
    exp_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = exp_dir / "canonical_trace.json"
    trace_rec = canonicalize_trace(
        {
            "trajectory_id": "TB-canon-self",
            "run_id": "run-cs-1",
            "metric_semantics": {
                "live_train_metric": {
                    "metric": "MAE",
                    "unit": "eV",
                    "target": "Gap",
                    "role_identity": "internal_train",
                    "weights": "live",
                    "direction": "minimize",
                },
                "live_dev_metric": None,
                "ema_dev_metric": None,
            },
            "observations": [
                {"sequence": 0, "event": "observation", "optimizer_step": 1, "live_train_metric": 0.5}
            ],
        }
    )
    canonical_bytes = json_bytes(trace_rec)
    canonical_file.write_bytes(canonical_bytes)
    sha = sha256_bytes(canonical_bytes)

    arm = create_candidate_arm(
        tmp_path,
        "exp_canon_self",
        "TB-canon-self",
        "run-cs-1",
        "ev-cs-1",
        include_trace_artifact=True,
        trace_artifact_name="training_trace",
        trace_relative_path="canonical_trace.json",
        create_trace_file=False,
    )
    term_data = load_json_object(arm["terminal_path"])
    for a in term_data["evidence"]["artifacts"]:
        if a["name"] == "training_trace":
            a["sha256"] = sha
    term_data["artifact_hashes"]["experiments/exp_canon_self/canonical_trace.json"] = sha
    arm["terminal_path"].write_bytes(json_bytes(term_data))

    result = close_terminal_arm(
        repo_root=tmp_path,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=canonical_file,
    )
    assert result["pipeline_status"] == "COMPLETE"
    receipt = verified_receipt(exp_dir / "rml_finalized")
    assert receipt["trace_status"] == "available"


def test_explicit_canonical_declared_itself_mismatched_digest_fails_closed(tmp_path):
    """Declared artifact in evidence is canonical trace + mismatched digest -> FAIL CLOSED."""
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_canon_mismatch",
        "TB-canon-mismatch",
        "run-cm-1",
        "ev-cm-1",
        include_trace_artifact=True,
        trace_artifact_name="training_trace",
        trace_relative_path="canonical_trace.json",
        create_trace_file=False,
    )
    canonical_file = arm["exp_dir"] / "canonical_trace.json"
    trace_rec = canonicalize_trace(
        {
            "trajectory_id": "TB-canon-mismatch",
            "run_id": "run-cm-1",
            "metric_semantics": {
                "live_train_metric": {
                    "metric": "MAE",
                    "unit": "eV",
                    "target": "Gap",
                    "role_identity": "internal_train",
                    "weights": "live",
                    "direction": "minimize",
                },
                "live_dev_metric": None,
                "ema_dev_metric": None,
            },
            "observations": [
                {"sequence": 0, "event": "observation", "optimizer_step": 1, "live_train_metric": 0.5}
            ],
        }
    )
    canonical_file.write_bytes(json_bytes(trace_rec))

    with pytest.raises(
        ValueError,
        match="FAIL CLOSED: explicit canonical trace digest mismatch with retention evidence",
    ):
        close_terminal_arm(
            repo_root=tmp_path,
            trajectory=arm["traj_path"],
            terminal=arm["terminal_path"],
            trace=canonical_file,
        )


# =========================================================================
# Test K: Hardening Point 2 - Scientific metric semantics & default manifest identity
# =========================================================================
def test_raw_trace_without_metric_semantics_fails_closed(tmp_path):
    """Raw trace without authoritative metric semantics anywhere -> FAIL CLOSED."""
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_no_semantics",
        "TB-no-semantics",
        "run-ns-1",
        "ev-ns-1",
        include_trace_artifact=True,
    )
    contract_path = arm["exp_dir"] / "contract.json"
    contract_path.write_bytes(b'{"schema": "contract-v1", "name": "c-no-sem"}\n')

    with pytest.raises(
        ValueError,
        match="FAIL CLOSED: cannot determine authoritative metric semantics for raw trace",
    ):
        close_terminal_arm(
            repo_root=tmp_path,
            trajectory=arm["traj_path"],
            terminal=arm["terminal_path"],
            trace=None,
        )


def test_raw_trace_with_contract_metric_semantics_succeeds(tmp_path):
    """Contract specifies metric semantics -> PASS and correctly propagated."""
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_contract_sem",
        "TB-contract-sem",
        "run-cs-1",
        "ev-cs-1",
        include_trace_artifact=True,
    )
    contract_path = arm["exp_dir"] / "contract.json"
    c_bytes = (
        json.dumps(
            {
                "schema": "contract-v1",
                "metric_semantics": {
                    "live_train_metric": {
                        "metric": "MSE",
                        "unit": "eV^2",
                        "target": "HOMO",
                        "role_identity": "train_role_x",
                        "weights": "live",
                        "direction": "minimize",
                    },
                    "live_dev_metric": None,
                    "ema_dev_metric": {
                        "metric": "MSE",
                        "unit": "eV^2",
                        "target": "HOMO",
                        "role_identity": "dev_role_x",
                        "weights": "ema",
                        "direction": "minimize",
                    },
                },
            }
        ).encode()
        + b"\n"
    )
    contract_path.write_bytes(c_bytes)
    term_data = load_json_object(arm["terminal_path"])
    term_data["artifact_hashes"]["experiments/exp_contract_sem/contract.json"] = sha256_bytes(c_bytes)
    arm["terminal_path"].write_bytes(json_bytes(term_data))

    result = close_terminal_arm(
        repo_root=tmp_path,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=None,
    )
    assert result["pipeline_status"] == "COMPLETE"
    finalized_trace = load_canonical_trace(arm["exp_dir"] / "rml_finalized/trace.json")
    assert (
        finalized_trace["metric_semantics"]["live_train_metric"]["metric"] == "MSE"
    )
    assert (
        finalized_trace["metric_semantics"]["live_train_metric"]["target"] == "HOMO"
    )


def test_raw_trace_with_recovery_spec_succeeds(tmp_path):
    """Raw trace with explicit recovery_spec metric semantics -> PASS."""
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_rec_spec",
        "TB-rec-spec",
        "run-rs-1",
        "ev-rs-1",
        include_trace_artifact=True,
    )
    contract_path = arm["exp_dir"] / "contract.json"
    c_bytes = b'{"schema": "contract-v1", "name": "c-no-sem"}\n'
    contract_path.write_bytes(c_bytes)
    term_data = load_json_object(arm["terminal_path"])
    term_data["artifact_hashes"]["experiments/exp_rec_spec/contract.json"] = sha256_bytes(c_bytes)
    arm["terminal_path"].write_bytes(json_bytes(term_data))

    rec_spec = {
        "metric_semantics": {
            "live_train_metric": {
                "metric": "MAE",
                "unit": "eV",
                "target": "LUMO",
                "role_identity": "train_lumo",
                "weights": "live",
                "direction": "minimize",
            },
            "live_dev_metric": None,
            "ema_dev_metric": {
                "metric": "MAE",
                "unit": "eV",
                "target": "LUMO",
                "role_identity": "dev_lumo",
                "weights": "ema",
                "direction": "minimize",
            },
        }
    }

    result = close_terminal_arm(
        repo_root=tmp_path,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=None,
        recovery_spec=rec_spec,
    )
    assert result["pipeline_status"] == "COMPLETE"
    finalized_trace = load_canonical_trace(arm["exp_dir"] / "rml_finalized/trace.json")
    assert (
        finalized_trace["metric_semantics"]["live_train_metric"]["target"] == "LUMO"
    )


def test_build_default_trace_manifest_unspecified_when_missing_contract(tmp_path):
    """When contract lacks identities, manifest fields are 'unspecified_in_contract' without guessing."""
    setup_mock_repo(tmp_path)
    arm = create_candidate_arm(
        tmp_path,
        "exp_manifest_unspecified",
        "TB-manifest-unspecified",
        "run-mu-1",
        "ev-mu-1",
        include_trace_artifact=True,
    )
    contract_file = tmp_path / "experiments/exp_manifest_unspecified/contract.json"
    contract_file.write_text("{}", encoding="utf-8")

    traj_obj = json.loads(arm["traj_path"].read_text(encoding="utf-8"))
    term_obj = json.loads(arm["terminal_path"].read_text(encoding="utf-8"))
    trace_rec = {
        "trajectory_id": "TB-manifest-unspecified",
        "run_id": "run-mu-1",
        "observations": [
            {"optimizer_step": 10, "live_train_metric": 0.5}
        ],
    }

    manifest = build_default_trace_manifest(tmp_path, traj_obj, term_obj, trace_rec)
    comp = manifest["comparability_identity"]
    assert comp["dataset_identity"] == "unspecified_in_contract"
    assert comp["row_split_identity"] == "unspecified_in_contract"
    assert comp["architecture_identity"] == "unspecified_in_contract"
    assert comp["precision_identity"] == "unspecified_in_contract"
    assert comp["optimizer_identity"] == "unspecified_in_contract"


# =========================================================================
# Test L: Hardening Point 3 - Multi-arm binding without string-guessing
# =========================================================================
def test_single_trace_auto_resolves_without_arm_identifier():
    """Single declared trace artifact resolves automatically when arm_identifier is None."""
    ev = {
        "artifacts": [
            {"name": "training_trace", "locator": "path/to/trace.json", "sha256": "abc"}
        ]
    }
    decl, art = inspect_trace_retention_evidence(ev, arm_identifier=None)
    assert decl is True
    assert art["locator"] == "path/to/trace.json"


def test_multi_trace_without_arm_identifier_fails_closed():
    """Multi-trace evidence without arm_identifier -> FAIL CLOSED (no string guessing)."""
    ev = {
        "artifacts": [
            {"name": "readback_trace", "locator": "path/arm1/trace.json", "sha256": "abc"},
            {"name": "recurrence_trace", "locator": "path/arm2/trace.json", "sha256": "def"},
        ]
    }
    with pytest.raises(
        ValueError,
        match="FAIL CLOSED: ambiguous multi-arm evidence \\(2 traces declared\\); explicit arm_identifier is required",
    ):
        inspect_trace_retention_evidence(ev, arm_identifier=None)


def test_multi_trace_with_explicit_arm_identifier_succeeds():
    """Multi-trace evidence with explicit arm_identifier selects the matching arm trace."""
    ev = {
        "artifacts": [
            {"name": "readback_trace", "locator": "path/readback/trace.json", "sha256": "abc"},
            {"name": "recurrence_trace", "locator": "path/recurrence/trace.json", "sha256": "def"},
        ]
    }
    decl, art = inspect_trace_retention_evidence(ev, arm_identifier="readback")
    assert decl is True
    assert art["name"] == "readback_trace"

    decl2, art2 = inspect_trace_retention_evidence(ev, arm_identifier="recurrence")
    assert decl2 is True
    assert art2["name"] == "recurrence_trace"


def test_multi_trace_with_unknown_arm_identifier_fails_closed():
    """Multi-trace evidence with unknown arm_identifier -> FAIL CLOSED."""
    ev = {
        "artifacts": [
            {"name": "readback_trace", "locator": "path/readback/trace.json", "sha256": "abc"},
            {"name": "recurrence_trace", "locator": "path/recurrence/trace.json", "sha256": "def"},
        ]
    }
    with pytest.raises(
        ValueError,
        match="FAIL CLOSED: retention evidence declares 2 traces, but none matched arm identifier",
    ):
        inspect_trace_retention_evidence(ev, arm_identifier="nonexistent")


def test_multi_trace_with_ambiguous_arm_identifier_fails_closed():
    """Multi-trace evidence with ambiguous arm_identifier matching multiple -> FAIL CLOSED."""
    ev = {
        "artifacts": [
            {"name": "arm_variant_1_trace", "locator": "path/arm1/trace.json", "sha256": "abc"},
            {"name": "arm_variant_2_trace", "locator": "path/arm2/trace.json", "sha256": "def"},
        ]
    }
    with pytest.raises(
        ValueError,
        match="FAIL CLOSED: ambiguous trace retention evidence for arm",
    ):
        inspect_trace_retention_evidence(ev, arm_identifier="arm_variant")
