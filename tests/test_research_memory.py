from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from molgap.research_memory.backtest import build_screening_backtest
from molgap.research_memory.compiler import compile_research_memory
from molgap.research_memory.cost import build_cost_ledger
from molgap.research_memory.derived import EXPECTED_FORMATS, validate_derived_outputs
from molgap.research_memory.discovery import DiscoveredRecords
from molgap.research_memory.portability import (
    committed_head_differences,
    missing_locally_claimed_artifacts,
    uncommitted_closure_paths,
)
from molgap.research_memory.ready import build_ready_package
from molgap.research_memory.schemas import validate_id, validate_trajectory
from molgap.research_memory.summary import render_summary_markdown
from molgap.research_memory.validate import validate_repository_records


REPO_ROOT = Path(__file__).resolve().parents[1]


def _cost_event(
    event_id: str,
    device_status: str,
    device_value: float | None,
    *,
    cpu_status: str = "measurement_missing",
    cpu_value: float | None = None,
) -> dict:
    return {
        "cost_event_id": event_id,
        "trajectory_id": "TB-cost",
        "category": "training",
        "hardware": "T4",
        "measurement": {
            "device_hours": {"value": device_value, "status": device_status},
            "cpu_hours": {"value": cpu_value, "status": cpu_status},
            "wall_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "measurement_missing"},
        },
    }


def _ready_fixture() -> tuple[dict, dict, dict]:
    candidate_id = "ev-candidate"
    reference_id = "ev-reference"
    trajectory = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": "TB-ready-test",
        "record_mode": "prospective",
        "track": "B",
        "owner": "server",
        "family_id": "family-ready",
        "question": "Should the qualified candidate be transferred?",
        "hypothesis": {
            "hypothesis_id": "hyp-ready",
            "supporting_evidence_ids": [candidate_id],
            "alternative_explanations": ["run noise"],
            "related_closed_family_ids": ["closed-family"],
            "observed_deficiency": "The reference remains above the target MAE.",
            "changed_mechanism": "One frozen mechanism.",
            "cheapest_falsifier": "The frozen paired comparison.",
            "expected_native_cost_ref": "cost-ready",
            "decision_changed_if_positive": "Transfer.",
            "decision_changed_if_negative": "Close.",
        },
        "state_at_start": {
            "source_commit": "a" * 40,
            "source_config_identity": "source-config-v1",
            "contract_refs": ["contract.json"],
            "reference_ids": [reference_id],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": [reference_id],
            "role_snapshot_refs": ["roles.json"],
            "budget_snapshot_ref": "budget.json",
        },
        "actions": [
            {
                "action_id": "action-ready",
                "type": "TRAIN",
                "source_commit": "a" * 40,
                "run_ids": ["run-1"],
                "attempt_ids": ["attempt-1"],
                "evidence_refs": ["comparison.json"],
                "cost_event_ids": ["cost-ready"],
            }
        ],
        "result": {
            "evidence_ids": [candidate_id],
            "evidence_refs": ["comparison.json"],
        },
        "decision": {
            "outcome": "READY_FOR_DESKTOP",
            "final": True,
            "decision_ref": "decision.md",
            "next_allowed_actions": ["TRANSFER"],
            "reopen_conditions": [],
        },
        "readiness": {
            "candidate_evidence_id": candidate_id,
            "qualification_100k_evidence_ids": [candidate_id],
            "qualification_500k_evidence_ids": [candidate_id],
            "funnel_requires_500k": True,
            "compatible_reference_id": reference_id,
            "candidate_contract_identity": "contract-v5",
            "reference_contract_identity": "contract-v5",
            "source_config_identity": "source-config-v1",
            "paired_comparison_ref": "comparison.json",
            "role_history_refs": ["roles.json"],
            "native_cost_event_ids": ["cost-ready"],
            "known_limitations": ["single seed"],
            "full_scale_question": "Does this exact candidate retain its gain at full scale?",
        },
    }
    evidence = {
        candidate_id: {
            "evidence_id": candidate_id,
            "legacy_contract": "contract-v5",
            "outcome": {
                "transfer_status": "qualified",
                "comparison_status": "complete",
            },
            "artifacts": [
                {
                    "name": "checkpoint",
                    "locator": "external://checkpoint",
                    "sha256": "b" * 64,
                    "availability": "durable_remote_verified",
                }
            ],
        },
        reference_id: {
            "evidence_id": reference_id,
            "legacy_contract": "contract-v5",
            "outcome": {},
            "artifacts": [],
        },
    }
    costs = {
        "cost-ready": {
            "cost_event_id": "cost-ready",
            "measurement": {
                "device_hours": {"value": 2.0, "status": "measured"},
                "cpu_hours": {"value": None, "status": "not_applicable"},
            },
        }
    }
    return trajectory, evidence, costs


def test_dangling_evidence_is_rejected_with_source_and_field(tmp_path, monkeypatch):
    pointer = tmp_path / "pointer.txt"
    pointer.write_text("fixture", encoding="utf-8")
    evidence_path = tmp_path / "v5_evidence.json"
    evidence_path.write_text(json.dumps({"evidence_id": "ev-known"}), encoding="utf-8")
    trajectory_path = tmp_path / "trajectory.json"
    trajectory = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": "TB-dangling",
        "record_mode": "retrospective_partial",
        "track": "B",
        "owner": "historical",
        "family_id": "family",
        "question": "Historical record.",
        "hypothesis": {
            "hypothesis_id": "hyp-dangling",
            "supporting_evidence_ids": [],
            "alternative_explanations": [],
            "related_closed_family_ids": [],
        },
        "state_at_start": {
            "source_commit": "unknown_retrospective",
            "contract_refs": [],
            "reference_ids": [],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": [],
            "role_snapshot_refs": [],
        },
        "actions": [],
        "result": {
            "evidence_ids": ["ev-missing"],
            "evidence_refs": [],
        },
        "decision": {
            "outcome": "CLOSED",
            "decision_ref": pointer.name,
            "next_allowed_actions": [],
            "reopen_conditions": [],
        },
    }
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    discovered = DiscoveredRecords(
        evidence=(evidence_path,),
        trajectories=(trajectory_path,),
        costs=(),
        roles=(),
        traces=(),
        ready=(),
    )
    monkeypatch.setattr(
        "molgap.research_memory.validate.discover_records", lambda _: discovered
    )
    monkeypatch.setattr(
        "molgap.research_memory.validate.validate_v5_evidence_envelope",
        lambda record, repo_root: record,
    )
    with pytest.raises(ValueError, match=r"trajectory\.json:result\.evidence_ids.*ev-missing"):
        validate_repository_records(tmp_path)


def test_portability_check_detects_uncommitted_pointer_closure(tmp_path):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")

    source = tmp_path / "experiments" / "trace_manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    sidecar = source.parent / "trace_migration.json"
    sidecar.write_text(
        json.dumps({"source_trace_ref": "source_trace.json"}), encoding="utf-8"
    )
    (tmp_path / "source_trace.json").write_text("{}", encoding="utf-8")
    evidence = tmp_path / "v5_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "authority": {"pointers": ["contract.json"]},
                "artifacts": [
                    {
                        "locator": "remote_only.pt",
                        "availability": "durable_remote_verified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    git("add", "experiments/trace_manifest.json", "contract.json")
    git("commit", "-qm", "fixture")

    records = {
        "traces": [(source, {"contract_ref": "contract.json"})],
        "evidence": [
            (
                evidence,
                {
                    **json.loads(evidence.read_text(encoding="utf-8")),
                    "artifacts": [
                        {
                            "locator": "remote_only.pt",
                            "availability": "durable_remote_verified",
                        },
                        {
                            "locator": "locally_retained.pt",
                            "availability": "durable_local_verified",
                        },
                    ],
                },
            )
        ],
    }
    assert missing_locally_claimed_artifacts(tmp_path, records) == [
        f"{evidence}: locally_retained.pt"
    ]
    assert uncommitted_closure_paths(tmp_path, records) == [
        "experiments/trace_migration.json",
        "source_trace.json",
        "v5_evidence.json",
    ]

    git(
        "add",
        "experiments/trace_migration.json",
        "source_trace.json",
        "v5_evidence.json",
    )
    git("commit", "-qm", "close evidence paths")
    assert uncommitted_closure_paths(tmp_path, records) == []
    (tmp_path / "locally_retained.pt").write_bytes(b"artifact")
    assert missing_locally_claimed_artifacts(tmp_path, records) == []


def test_portability_check_rejects_dirty_runtime_and_derived_outputs(tmp_path):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")

    paths = [
        "pyproject.toml",
        "src/molgap/__init__.py",
        "src/molgap/constants.py",
        "src/molgap/comparison_readiness.py",
        "src/molgap/evidence_pointers.py",
        "src/molgap/v5_common.py",
        "src/molgap/research_memory/__init__.py",
        "src/molgap/research_memory/cli.py",
        "research_memory/schemas/trajectory.json",
        "research_memory/policies/registry.json",
        "experiments/trajectory.json",
        "experiments/contract.json",
    ]
    from molgap.research_memory.compiler import DERIVED_FILENAMES

    paths.extend(f"research_memory/derived/{name}" for name in DERIVED_FILENAMES)
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "baseline")

    source = tmp_path / "experiments/trajectory.json"
    records = {
        "trajectories": [(source, {"contract_ref": "experiments/contract.json"})]
    }
    assert committed_head_differences(tmp_path, records) == {
        "missing_from_head": [],
        "changed_from_head": [],
    }

    (tmp_path / "src/molgap/research_memory/cli.py").write_text(
        "# local runtime change\n", encoding="utf-8"
    )
    (tmp_path / "research_memory/derived/research_summary.json").write_text(
        '{"stale": true}\n', encoding="utf-8"
    )
    untracked_module = tmp_path / "src/molgap/research_memory/new_module.py"
    untracked_module.write_text("# untracked runtime module\n", encoding="utf-8")

    assert committed_head_differences(tmp_path, records) == {
        "missing_from_head": ["src/molgap/research_memory/new_module.py"],
        "changed_from_head": [
            "research_memory/derived/research_summary.json",
            "src/molgap/research_memory/cli.py",
        ],
    }


def test_ready_rejects_retrospective():
    trajectory, evidence, costs = _ready_fixture()
    trajectory["record_mode"] = "retrospective_partial"
    with pytest.raises(ValueError, match="retrospective"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_missing_reference():
    trajectory, evidence, costs = _ready_fixture()
    evidence.pop("ev-reference")
    with pytest.raises(ValueError, match="dangling evidence"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_empty_role_history():
    trajectory, evidence, costs = _ready_fixture()
    trajectory["readiness"]["role_history_refs"] = []
    with pytest.raises(ValueError, match="role history"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_missing_artifact_sha():
    trajectory, evidence, costs = _ready_fixture()
    evidence["ev-candidate"]["artifacts"][0]["sha256"] = None
    with pytest.raises(ValueError, match="lacks SHA256"):
        build_ready_package(trajectory, evidence, costs)


def test_ready_rejects_unknown_native_cost():
    trajectory, evidence, costs = _ready_fixture()
    costs["cost-ready"]["measurement"]["device_hours"] = {
        "value": None,
        "status": "measurement_missing",
    }
    with pytest.raises(ValueError, match="measured native"):
        build_ready_package(trajectory, evidence, costs)


def test_cost_ledger_separates_measured_estimated_missing_and_not_applicable():
    ledger = build_cost_ledger(
        [
            _cost_event(
                "cost-m",
                "measured",
                2.0,
                cpu_status="not_applicable",
            ),
            _cost_event(
                "cost-e",
                "estimated",
                3.0,
                cpu_status="not_applicable",
            ),
        ]
    )
    device = ledger["totals_by_hardware"]["T4"]["device_hours"]
    assert device["measured"] == {"known_total": 2.0, "known_records": 1}
    assert device["estimated"] == {"known_total": 3.0, "known_records": 1}
    assert "total" not in device
    assert ledger["totals_by_hardware"]["T4"]["cpu_hours"]["not_applicable_records"] == 2
    assert ledger["totals_by_hardware"]["T4"]["wall_hours"]["unknown_records"] == 2
    assert ledger["cross_hardware_total"] is None


def test_edge_state_gpu_cost_preserves_unknown_cpu_accounting():
    record = json.loads(
        (
            REPO_ROOT
            / "experiments/pcqm_edge_state_full/costs/"
            "cost-TB-edgestate-rich-full-training.json"
        ).read_text(encoding="utf-8")
    )
    assert record["hardware"] == "IMS_GPU_UNSPECIFIED"
    assert record["measurement"]["cpu_hours"] == {
        "value": None,
        "status": "measurement_missing",
    }


def test_real_cost_completeness_separates_absent_and_incomplete_events():
    payloads = compile_research_memory(REPO_ROOT)
    report = json.loads(payloads["completeness_report.json"])
    trajectories = json.loads(payloads["trajectory_index.json"])["trajectories"]
    assert {record["trajectory_id"] for record in trajectories} >= {
        "TB-edgestate-rich-full-v1",
        "TB-geometry-transfer-500k-context",
        "TB-gine-1m-gap-specialist-v7",
        "TB-gptrans-noisy-nodes-100k-s42",
        "TB-gptrans-noisy-nodes-500k-s42",
        "TB-gptrans-noisy-pair-norm-100k-s42",
        "TB-gptrans-noisy-pair-norm-500k-s42",
        "TB-gptrans-t-100k-v4-reference",
        "TB-gptrans-t-500k-bridge",
        "TB-k1-gptrans-full-fusion-r3",
        "TB-k1-residual-reconciliation",
        "TB-matched-500k-v4-three-arm",
        "TB-pcqm-scale-transfer-attribution",
        "TB-pcqm-scale-transfer-diagnostic",
        "TB-recurrent-graph-state-100k-seed42",
        "TB-route-b-four-encoder-specialist",
        "TC-gptrans-flow-ablation-100k-s42",
        "TH-gptrans-conditional-flow-100k-s42-conditional_pair_readback",
        "TH-gptrans-conditional-flow-100k-s42-conditional_pair_recurrence",
    }
    costs = report["cost_completeness"]
    assert costs["trajectories_total"] == len(trajectories)
    assert costs["with_cost_event"] + costs["without_cost_event"] == len(trajectories)
    assert (costs["with_complete_native_measurement"]
            + costs["with_incomplete_native_measurement"] == costs["with_cost_event"])
    assert costs["with_complete_native_measurement"] >= 4
    issues = {item["code"]: item["count"] for item in report["issues"]}
    assert issues["missing_cost_event"] == costs["without_cost_event"]
    assert issues["incomplete_native_cost_measurement"] == costs["with_incomplete_native_measurement"]
    assert "missing_native_cost" not in issues


def test_local_desktop_records_have_v5_evidence_and_explicit_roles():
    payloads = compile_research_memory(REPO_ROOT)
    report = json.loads(payloads["completeness_report.json"])
    summary = json.loads(payloads["research_summary.json"])
    trajectories = {
        record["trajectory_id"]: record
        for record in json.loads(payloads["trajectory_index.json"])["trajectories"]
    }

    assert summary["evidence"]["validated"] >= 22
    assert summary["roles"]["explicit_events"] >= 30
    no_evidence_trajectories = next(
        (item["items"] for item in report["issues"] if item["code"] == "trajectory_without_v5_evidence"),
        [],
    )
    assert set(no_evidence_trajectories) == {
        "TB-gptrans-100k-transfer-control-local-20260925",
        "TB-gptrans-centered-logits-100k-s42",
        "TB-gptrans-pair-memory-replay-memory-value-100k-s42",
        "TB-gptrans-pair-memory-replay-memory-message-100k-s42",
        "TH-gptrans-conditional-flow-100k-s42-conditional_pair_readback",
        "TH-gptrans-conditional-flow-100k-s42-conditional_pair_recurrence",
    }
    for arm in ("reference", "joint"):
        trajectory_id = f"TB-gptrans-100k-local-{arm}-control-20260925"
        assert trajectories[trajectory_id]["completeness_status"] == "complete"
        assert trajectory_id not in no_evidence_trajectories
    assert trajectories["TB-gptrans-centered-logits-100k-s42"]["outcome"] == "INCONCLUSIVE"
    assert trajectories["TB-gptrans-centered-logits-100k-s42"]["result_evidence_ids"] == []
    assert trajectories["TB-gptrans-noisy-pair-norm-500k-s42"]["completeness_status"] == "complete"
    assert trajectories["TB-gptrans-noisy-pair-norm-500k-s42"]["outcome"] == "NEGATIVE_UNDER_CONTRACT"
    assert trajectories["TB-gptrans-pair-norm-500k-s42"]["outcome"] == "NEGATIVE_UNDER_CONTRACT"
    pair_norm_replay = [
        entry for entry in json.loads(payloads["replay_pool.json"])["entries"]
        if entry["trajectory_id"] == "TB-gptrans-pair-norm-500k-s42"
    ]
    assert len(pair_norm_replay) == 1
    assert pair_norm_replay[0]["capability"] == "complete"
    assert pair_norm_replay[0]["exclusion_reasons"] == []
    assert trajectories["TB-matched-500k-v4-three-arm"]["result_evidence_ids"] == [
        "pcqm-matched-500k-v4-three-arm"
    ]
    assert trajectories["TB-k1-residual-reconciliation"]["result_evidence_ids"] == [
        "pcqm-k1-residual-reconciliation"
    ]
    assert trajectories["TB-pcqm-scale-transfer-attribution"]["result_evidence_ids"] == [
        "pcqm-scale-transfer-attribution-no-train"
    ]
    assert trajectories["TB-k1-full-convergence-20260919"]["outcome"] == "CLOSED"
    assert trajectories["TB-gptrans-full-convergence-20260919"]["outcome"] == "INCONCLUSIVE"
    assert trajectories["TC-k1-pair-value-decoupled-100k-s42"]["outcome"] == "INCONCLUSIVE"
    assert trajectories["TB-gptrans-feature-denoise-full-atom-100k-s42"]["outcome"] == "INCONCLUSIVE"
    assert trajectories["TB-gptrans-feature-denoise-full-atom-bond-100k-s42"]["outcome"] == "INCONCLUSIVE"
    assert trajectories["TB-distance-angle-triangle-500k-20260911"]["outcome"] == "POSITIVE_UNDER_CONTRACT"
    assert trajectories["TB-xian-determinism-audit-20260915"]["outcome"] == "NEGATIVE_UNDER_CONTRACT"


def test_empty_numeric_basis_serializes_as_null():
    ledger = build_cost_ledger(
        [_cost_event("cost-missing", "measurement_missing", None)]
    )
    measured = ledger["totals_by_hardware"]["T4"]["device_hours"]["measured"]
    assert measured == {"known_total": None, "known_records": 0}


def test_measured_zero_remains_real_zero():
    ledger = build_cost_ledger([_cost_event("cost-zero", "measured", 0.0)])
    measured = ledger["totals_by_hardware"]["T4"]["device_hours"]["measured"]
    assert measured == {"known_total": 0.0, "known_records": 1}


def test_not_applicable_only_is_not_zero_or_unknown():
    ledger = build_cost_ledger(
        [
            _cost_event(
                "cost-na",
                "not_applicable",
                None,
                cpu_status="not_applicable",
            )
        ]
    )
    device = ledger["totals_by_hardware"]["T4"]["device_hours"]
    assert device["measured"]["known_total"] is None
    assert device["estimated"]["known_total"] is None
    assert device["unknown_records"] == 0
    assert device["not_applicable_records"] == 1


def test_estimated_only_does_not_create_measured_zero():
    ledger = build_cost_ledger([_cost_event("cost-est", "estimated", 1.25)])
    device = ledger["totals_by_hardware"]["T4"]["device_hours"]
    assert device["measured"] == {"known_total": None, "known_records": 0}
    assert device["estimated"] == {"known_total": 1.25, "known_records": 1}


def _summary_with_bucket(bucket: dict) -> dict:
    return {
        "evidence": {"validated": 0, "without_trajectory": 0},
        "trajectories": {"outcomes": {}},
        "transfer": {"ready_valid": 0, "ready_blocked": 0, "missing_references": 0},
        "roles": {"explicit_events": 0, "coarse_historical": 0, "repeatedly_selected": 0},
        "cost": {
            "totals_by_hardware": {"T4": {"device_hours": bucket}},
            "unknown_events": 1,
            "completeness": {
                "trajectories_total": 1,
                "with_cost_event": 1,
                "without_cost_event": 0,
                "with_complete_native_measurement": 0,
                "with_incomplete_native_measurement": 1,
            },
        },
        "backtest": {"status": "insufficient_evidence", "eligible": 0, "excluded": 0},
        "memory_gaps": [],
    }


def test_markdown_renders_unknown_without_false_zero():
    bucket = build_cost_ledger(
        [_cost_event("cost-missing", "measurement_missing", None)]
    )["totals_by_hardware"]["T4"]["device_hours"]
    markdown = render_summary_markdown(_summary_with_bucket(bucket))
    assert "measured=unknown (0 measured records)" in markdown
    assert "0.000000 (0 records)" not in markdown


def test_markdown_renders_real_measured_zero():
    bucket = build_cost_ledger(
        [_cost_event("cost-zero", "measured", 0.0)]
    )["totals_by_hardware"]["T4"]["device_hours"]
    markdown = render_summary_markdown(_summary_with_bucket(bucket))
    assert "measured=0.000000 (1 measured record)" in markdown


def test_all_derived_json_schemas_match_generated_top_level_contracts():
    payloads = compile_research_memory(REPO_ROOT)
    for output_name in EXPECTED_FORMATS:
        output = json.loads(payloads[output_name])
        schema_name = output_name.replace("_", "-").replace(
            ".json", "-v1.schema.json"
        )
        schema = json.loads(
            (REPO_ROOT / "research_memory/schemas" / schema_name).read_text(
                encoding="utf-8"
            )
        )
        assert set(schema["required"]) == set(output), output_name
        assert set(schema["properties"]) == set(output), output_name
        assert schema["additionalProperties"] is False

    ledger_schema = json.loads(
        (
            REPO_ROOT / "research_memory/schemas/cost-ledger-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert "estimated_event_count" in ledger_schema["required"]
    assert set(ledger_schema["$defs"]["basis"]["properties"]["known_total"]["type"]) == {
        "number",
        "null",
    }


def test_prospective_missing_hypothesis_field_is_rejected():
    trajectory, _, _ = _ready_fixture()
    del trajectory["hypothesis"]["cheapest_falsifier"]
    with pytest.raises(ValueError, match="cheapest_falsifier"):
        validate_trajectory(trajectory)


@pytest.mark.parametrize("value", ["bad id", "../x", "a/b", "a\\b", "x\ny", " C:"])
def test_malformed_ids_are_rejected(value):
    with pytest.raises(ValueError):
        validate_id(value, "fixture.id")


def _trace(run_id, role, optimizer):
    return {
        "trajectory_id": f"TB-{run_id}",
        "run_id": run_id,
        "reference_id": "ev-reference",
        "comparison_role": role,
        "x_axis": "optimizer_steps",
        "backtest_eligibility": {"eligible": True, "exclusion_reasons": []},
        "exposure": {"optimizer_steps": 100, "sample_presentations": 12800},
        "comparability_identity": {
            "scientific_contract": "v5",
            "dataset_identity": "100k",
            "row_split_identity": "rows-v1",
            "architecture_identity": "arch",
            "optimizer_identity": optimizer,
            "lr_schedule_identity": "cosine",
            "target_transform_identity": "direct-gap",
            "precision_identity": "fp32",
            "ema_semantics": "none",
            "evaluation_role_identity": "development",
            "selection_role_identity": "development",
            "x_axis_semantics": "optimizer_steps",
            "terminal_endpoint_identity": "step-100",
            "matched_architecture_required": False,
        },
    }


def test_incompatible_traces_remain_insufficient():
    result = build_screening_backtest(
        [_trace("candidate", "candidate", "adamw"), _trace("reference", "reference", "sgd")]
    )
    assert result["status"] == "insufficient_evidence"
    assert result["recommended_early_stop"] is None
    assert result["false_stop_count"] is None
    assert result["native_cost_saved"] is None
    assert result["policy_activated"] is False


def test_invalid_derived_output_is_rejected():
    payloads = compile_research_memory(REPO_ROOT)
    outputs = {
        name: json.loads(payload)
        for name, payload in payloads.items()
        if name.endswith(".json")
    }
    outputs["cost_ledger.json"]["cross_hardware_total"] = 0
    with pytest.raises(ValueError, match="must not aggregate across hardware"):
        validate_derived_outputs(outputs)
