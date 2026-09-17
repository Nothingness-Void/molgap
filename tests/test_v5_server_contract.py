from __future__ import annotations

from pathlib import Path

import pytest

from molgap.cost_ledger import CostEntry, NativeCostLedger
from molgap.evidence_index import load_evidence_index, write_evidence_index
from molgap.research_funnel import HypothesisCard, validate_trajectory
from molgap.runtime_profiling import PROFILE_STAGES, RuntimeProfile
from molgap.screen_policy import REFERENCE_MATCH_FIELDS, canonical_fingerprint
from molgap.screen_backtest import ScaleObservation, backtest_low_cost_screening
from molgap.server_acceptance import (
    READY_FOR_DESKTOP,
    assess_strict_comparison,
    validate_prediction_bundle,
    validate_paired_predictions,
)
from molgap.server_control import (
    BoundRun,
    EventClaimedError,
    LocalServerControlStore,
    StaleGenerationError,
)


def make_run(generation: int = 1, run_id: str = "run-1") -> BoundRun:
    return BoundRun(
        campaign_id="campaign-1",
        chain_id="chain-1",
        run_id=run_id,
        attempt_id=f"attempt-{generation}",
        a_thread_id="thread-a",
        b_thread_id="thread-b",
        monitor_generation=generation,
        remote_platform="synthetic-platform",
        remote_job_identity={"job": run_id},
        release_identity="release-sha",
        reference_identity="reference-sha",
        budget_reserved_native={"T4_device_hours": 1.0},
    )


def bind_store(tmp_path: Path) -> LocalServerControlStore:
    store = LocalServerControlStore(tmp_path / "server-control.json")
    store.bind_run(make_run())
    return store


def test_healthy_server_job_is_silent(tmp_path: Path) -> None:
    store = bind_store(tmp_path)
    result = store.observe(
        run_id="run-1",
        attempt_id="attempt-1",
        monitor_generation=1,
        remote_status="RUNNING",
    )
    assert result.action == "SILENT"
    assert store.pending_events() == []
    assert store.snapshot()["last_observation"]["status"] == "RUNNING"


def test_duplicate_terminal_delivery_creates_one_event_and_one_decision(tmp_path: Path) -> None:
    store = bind_store(tmp_path)
    first = store.observe(
        run_id="run-1",
        attempt_id="attempt-1",
        monitor_generation=1,
        remote_status="COMPLETE",
        remote_state_version="terminal-1",
    )
    second = store.observe(
        run_id="run-1",
        attempt_id="attempt-1",
        monitor_generation=1,
        remote_status="COMPLETE",
        remote_state_version="terminal-1",
    )
    assert first.event_id == second.event_id
    assert len(store.snapshot()["events"]) == 1
    store.deliver_to_a(first.event_id)
    store.deliver_to_a(first.event_id)
    assert store.snapshot()["events"][first.event_id]["delivery_attempts"] == 1
    store.claim_event(first.event_id, "thread-a")
    committed = store.commit_decision(
        first.event_id,
        a_thread_id="thread-a",
        decision_ref="decision-1",
        outcome={"scientific_status": "PENDING"},
    )
    repeated = store.commit_decision(
        first.event_id,
        a_thread_id="thread-a",
        decision_ref="decision-1",
        outcome={"scientific_status": "different-but-ignored"},
    )
    assert committed["decision_ref"] == "decision-1"
    assert repeated["decision_ref"] == "decision-1"
    store.finalize_event(first.event_id, "PAUSED")
    assert store.pending_events() == []


def test_a_busy_retains_event_and_a_crash_is_recoverable(tmp_path: Path) -> None:
    store = bind_store(tmp_path)
    event = store.observe(
        run_id="run-1",
        attempt_id="attempt-1",
        monitor_generation=1,
        remote_status="FAILED",
    )
    assert store.pending_events()[0]["event_status"] == "EVENT_DURABLE"
    store.claim_event(event.event_id, "thread-a")
    with pytest.raises(EventClaimedError):
        store.claim_event(event.event_id, "other-a")
    store.recover_claim(event.event_id, "thread-a")
    assert store.pending_events()[0]["event_status"] == "EVENT_DURABLE"
    reclaimed = store.claim_event(event.event_id, "thread-a")
    assert reclaimed["event_id"] == event.event_id


def test_submit_timeout_is_durable_and_never_retryable(tmp_path: Path) -> None:
    store = bind_store(tmp_path)
    result = store.record_submit_unknown(
        run_id="run-1",
        attempt_id="attempt-1",
        monitor_generation=1,
        detail={"request_id": "request-1"},
    )
    event = store.snapshot()["events"][result.event_id]
    assert event["event_type"] == "SUBMIT_UNKNOWN"
    assert event["payload"]["retry_allowed"] is False


def test_stale_monitor_generation_cannot_mutate_new_chain(tmp_path: Path) -> None:
    store = bind_store(tmp_path)
    store.bind_run(make_run(generation=2, run_id="run-2"))
    with pytest.raises(StaleGenerationError):
        store.observe(
            run_id="run-1",
            attempt_id="attempt-1",
            monitor_generation=1,
            remote_status="COMPLETE",
        )


def test_desktop_owned_job_is_ignored(tmp_path: Path) -> None:
    store = bind_store(tmp_path)
    result = store.observe(
        run_id="run-1",
        attempt_id="attempt-1",
        monitor_generation=1,
        remote_status="COMPLETE",
        owner_scope="desktop",
    )
    assert result.action == "IGNORED"
    assert store.snapshot()["events"] == {}


def make_contract(model_id: str, platform: str, run_id: str, *, frozen: bool) -> tuple[dict, dict]:
    certificate = {
        "format": "molgap-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": platform,
        "accelerator": "T4",
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": 128,
        "tail_batch_policy": "drop_last",
        "software_fingerprint": "1" * 64,
        "determinism_fingerprint": "2" * 64,
        "calibration_fixture_sha256": "3" * 64,
        "calibration_output_sha256": "4" * 64,
        "runtime_fingerprint": "5" * 64,
        "calibration_checks_passed": True,
    }
    contract = {
        "run_id": run_id,
        "model_id": model_id,
        "architecture_fingerprint": ("a" if frozen else "b") * 64,
        "source_archive_sha256": ("c" if frozen else "d") * 64,
        "result_artifact_sha256": ("e" if frozen else "f") * 64,
        "platform_id": platform,
        "accelerator": "T4",
        "runtime_certificate_id": canonical_fingerprint(certificate),
        "benchmark_id": "pcqm-fixed-100k-v1",
        "data_role_fingerprint": "1" * 64,
        "row_order_fingerprint": "2" * 64,
        "feature_fingerprint": "features-v1",
        "target_fingerprint": "gap-eV",
        "seed": 42,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-v1",
        "schedule_fingerprint": "cosine-v1",
        "loss_fingerprint": "l1-v1",
        "target_transform_fingerprint": "target-v1",
        "selection_fingerprint": "dev-v1",
        "role_access_fingerprint": "roles-v1",
        "sample_exposure": 4_000_000,
        "tail_batch_policy": "drop_last",
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "scale_rows": 500_000,
        "execution_status": "COMPLETE",
    }
    if frozen:
        contract.update(
            {
                "frozen_reference": True,
                "stochasticity_floor_eV": 0.003,
                "minimum_material_gain_eV": 0.003,
            }
        )
    assert all(field in contract for field in REFERENCE_MATCH_FIELDS)
    return contract, certificate


def make_bundle() -> dict:
    return {
        "prediction": [0.1, 0.2, 0.3],
        "target": [0.1, 0.1, 0.4],
        "source_idx": [10, 11, 12],
    }


def test_prediction_validation_rejects_nan_inf_shape_and_alignment() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_prediction_bundle(
            {"prediction": [float("nan")], "target": [0.0], "source_idx": [1]},
            "candidate",
        )
    with pytest.raises(ValueError, match="wrong shape"):
        validate_prediction_bundle(
            {"prediction": [[0.0]], "target": [0.0], "source_idx": [1]},
            "candidate",
        )
    with pytest.raises(ValueError, match="aligned"):
        validate_paired_predictions(
            make_bundle(),
            {"prediction": [0.1, 0.2, 0.3], "target": [0.1, 0.1, 0.4], "source_idx": [10, 12, 13]},
        )


def test_missing_reference_or_bootstrap_is_pending_not_retraining() -> None:
    pending = assess_strict_comparison(
        reference_contract=None,
        candidate_contract={"scale_rows": 500_000},
    )
    assert pending["comparison_status"] == "PENDING"
    assert pending["reason"] == "missing_reference"

    reference, _ = make_contract("baseline", "kaggle", "ref", frozen=True)
    candidate, _ = make_contract("candidate", "scnet", "cand", frozen=False)
    pending = assess_strict_comparison(
        reference_contract=reference,
        candidate_contract=candidate,
        runtime_certificates={},
        reference_bundle=make_bundle(),
        candidate_bundle=make_bundle(),
    )
    assert pending["comparison_status"] == "PENDING"
    assert "paired_bootstrap" in pending["missing_evidence"]


def test_old_v3_reference_is_not_a_strict_v5_comparator() -> None:
    result = assess_strict_comparison(
        reference_contract={"policy": "molgap-screen-comparability-v3"},
        candidate_contract={"scale_rows": 500_000},
    )
    assert result["comparison_status"] == "REJECTED"
    assert result["reason"] == "old_reference_policy_incompatible"


def test_strict_positive_500k_only_emits_ready_for_desktop(tmp_path: Path) -> None:
    reference, reference_certificate = make_contract("baseline", "kaggle", "ref", frozen=True)
    candidate, candidate_certificate = make_contract("candidate", "scnet", "cand", frozen=False)
    result = assess_strict_comparison(
        reference_contract=reference,
        candidate_contract=candidate,
        runtime_certificates={
            reference["runtime_certificate_id"]: reference_certificate,
            candidate["runtime_certificate_id"]: candidate_certificate,
        },
        reference_bundle=make_bundle(),
        candidate_bundle=make_bundle(),
        paired_analysis={"method": "paired-mae"},
        bootstrap={"method": "row-bootstrap", "upper_eV": -0.001},
        artifact_hashes={"reference": "1" * 64, "candidate": "2" * 64},
        resume_cursor={"step": 100},
        role_history={"development": "reused"},
        actual_native_cost={"T4_device_hours": 2.0},
        scientific_status="QUALIFIED",
        budget_decision="APPROVED",
    )
    assert result["comparison_status"] == "ACCEPTED"
    assert result["full_handoff_status"] == READY_FOR_DESKTOP
    assert result["transfer_status"] == READY_FOR_DESKTOP
    assert "accepted" not in result

    package = {
        "format": "molgap-ready-for-desktop-v5",
        "source_config_identity": {"source": "source-sha"},
        "scale_decisions": {"100K": "qualified", "500K": "qualified"},
        "reference_candidate_identity": {"reference": "ref", "candidate": "cand"},
        "paired_analysis": {"method": "paired-mae"},
        "role_history": {"development": "reused"},
        "recovery_state": {"checkpoint": "checkpoint-sha"},
        "native_cost": {"T4_device_hours": 2.0},
        "limitations": ["desktop full decision remains independent"],
        "full_scale_question": "Does the qualified mechanism justify full scale?",
    }
    from molgap.server_acceptance import write_ready_for_desktop_package

    write_ready_for_desktop_package(tmp_path / "ready.json", package)
    assert (tmp_path / "ready.json").is_file()


def test_cost_ledger_keeps_native_units_separate(tmp_path: Path) -> None:
    ledger = NativeCostLedger(tmp_path / "cost.json")
    entry = CostEntry("run", "training", "T4_device_hours", 2.0, "scnet", entry_id="one")
    assert ledger.record(entry) == ledger.record(entry)
    ledger.record(CostEntry("run", "acceptance", "queue_hours", 3.0, "scnet", entry_id="two"))
    totals = ledger.totals_by_native_unit()
    assert totals["T4_device_hours"] == 2.0
    assert totals["queue_hours"] == 3.0


def test_runtime_profile_and_evidence_index_are_durable(tmp_path: Path) -> None:
    profile = RuntimeProfile()
    with profile.measure("forward_loss"):
        pass
    profile.add("allocation", 0.25)
    profile.write(tmp_path / "profile.json")
    assert set(profile.to_dict()["stages"]) == set(PROFILE_STAGES)
    assert (tmp_path / "profile.json").is_file()

    entries = [
        {
            "evidence_id": "negative-1",
            "family_id": "closed-family",
            "status": "negative",
            "source": "experiments/closed/decision.md",
            "decision_ref": "decision-1",
            "contract_fingerprint": "contract-1",
        }
    ]
    write_evidence_index(tmp_path / "evidence.json", entries)
    assert load_evidence_index(tmp_path / "evidence.json")[0]["status"] == "NEGATIVE"


def test_hypothesis_card_and_trajectory_require_evidence_first() -> None:
    card = HypothesisCard(
        hypothesis_id="h1",
        family_id="pair-token",
        current_baseline_deficiency="weak sparse-molecule transfer",
        supporting_evidence=["decision.md#stage2"],
        alternative_explanation="development selection noise",
        changed_mechanism="normalized learned pair bottleneck",
        earliest_cheap_falsifier="frozen checkpoint information-flow audit",
        closed_related_routes=["uniform-pair", "diagonal-pair"],
        expected_native_cost={"T4_device_hours": 2},
        decision_changed="release one 100K screen or stop",
        literature_refs=["paper:pair-relations"],
    )
    assert card.validate()["literature_refs"] == ["paper:pair-relations"]
    record = {
        "state": {"baseline": "k1"},
        "evidence": {"refs": ["decision.md"]},
        "action": {"kind": "NO_TRAIN"},
        "result": {"outcome": "NO_TRAIN"},
        "decision": {"status": "closed"},
    }
    assert validate_trajectory(record)["action"]["kind"] == "NO_TRAIN"


def test_scale_backtest_does_not_release_ladder_from_heterogeneous_history() -> None:
    result = backtest_low_cost_screening(
        [
            ScaleObservation("k1", "100K", "500K", 0.009, 0.010, False, False),
            ScaleObservation("graphstate", "100K", "full", 0.002, -0.019, False, False),
            ScaleObservation("gptrans", "100K", "500K", 0.001, 0.004, False, False),
        ]
    )
    assert result["calibration_status"] == "PENDING"
    assert result["early_stop_rule_released"] is False


def test_scale_backtest_requires_three_same_contract_trace_pairs() -> None:
    observations = [
        ScaleObservation(f"family-{index}", "B/16", "B", 0.001, 0.002, True, True)
        for index in range(3)
    ]
    result = backtest_low_cost_screening(observations)
    assert result["calibration_status"] == "CALIBRATED"
    assert result["early_stop_rule_released"] is True
