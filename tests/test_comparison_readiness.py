from __future__ import annotations

import copy

import pytest

from molgap.comparison_readiness import (
    REQUIRED_REFERENCE_ARTIFACTS,
    STRICT_IDENTITY_FIELDS,
    assess_comparison_readiness,
    validate_comparison_readiness,
    validate_reference_bundle,
    validate_stochasticity,
)
from molgap.research_memory.roles import build_role_reuse_index
from molgap.server_acceptance import (
    validate_server_scientific_prelaunch,
    write_server_comparison_prelaunch,
)


def _identity(architecture: str) -> dict:
    values = {field: f"same-{field}" for field in STRICT_IDENTITY_FIELDS}
    values.update(
        {
            "architecture_config_identity": architecture,
            "seed": 42,
            "tf32_enabled": False,
            "deterministic_algorithms": True,
            "physical_batch_per_device": 128,
            "gradient_accumulation_steps": 1,
            "optimizer_fused": False,
            "sample_presentations": 4_000_000,
            "optimizer_steps": 31_250,
            "ema_enabled": False,
            "ema_decay": None,
            "ema_update_frequency": None,
            "weight_semantics": "live",
            "evaluation_weight_source": "live",
        }
    )
    return values


def _side(architecture: str) -> dict:
    return {
        "comparison_identity": _identity(architecture),
        "artifacts": {name: "complete" for name in REQUIRED_REFERENCE_ARTIFACTS},
        "prediction_status": "complete",
        "row_alignment_status": "aligned",
        "runtime_certificate_status": "accepted",
        "role_status": "complete",
        "trace_status": "complete",
        "stochasticity_status": "unavailable",
        "terminal_complete": True,
    }


def _assess(candidate: dict | None = None, reference: dict | None = None, **kwargs) -> dict:
    return assess_comparison_readiness(
        candidate_id="candidate",
        candidate=candidate or _side("candidate-arch"),
        reference_id="reference",
        reference=reference or _side("reference-arch"),
        declared_intervention_fields=["architecture_config_identity"],
        **kwargs,
    )


def test_all_fields_and_complete_artifacts_are_strict_causal():
    result = _assess()
    assert result["comparison_class"] == "STRICT_CAUSAL"
    assert result["strict_ready"] is True
    validate_comparison_readiness(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_identity", "different-optimizer"),
        ("checkpoint_selection_identity", "different-selection"),
        ("sample_presentations", 5_000_000),
    ],
)
def test_scientific_recipe_mismatch_is_not_strict(field, value):
    candidate = _side("candidate-arch")
    candidate["comparison_identity"][field] = value
    result = _assess(candidate=candidate)
    assert result["strict_ready"] is False
    assert result["comparison_class"] == "PAIRED_ENDPOINT"
    assert field in result["mismatched_fields"]


def test_ema_and_live_weights_are_not_strictly_equal():
    candidate = _side("candidate-arch")
    candidate["comparison_identity"].update(
        {
            "weight_semantics": "EMA",
            "ema_enabled": True,
            "ema_decay": 0.999,
            "ema_update_frequency": 1,
            "evaluation_weight_source": "EMA",
        }
    )
    result = _assess(candidate=candidate)
    assert result["comparison_class"] == "PAIRED_ENDPOINT"
    assert "weight_semantics" in result["mismatched_fields"]


def test_missing_predictions_leave_strict_pending():
    reference = _side("reference-arch")
    reference["prediction_status"] = "unavailable"
    reference["artifacts"]["prediction_manifest"] = "missing"
    result = _assess(reference=reference)
    assert result["strict_ready"] is False
    assert result["strict_status"] == "PENDING"
    assert "MISSING_ALIGNED_PREDICTIONS" in result["blocker_codes"]


def test_row_alignment_mismatch_rejects_paired_claim():
    candidate = _side("candidate-arch")
    candidate["row_alignment_status"] = "mismatch"
    result = _assess(candidate=candidate)
    assert result["comparison_class"] == "CONTEXT_ONLY"
    assert "ROW_ALIGNMENT_NOT_CONFIRMED" in result["blocker_codes"]


def test_terminal_missing_but_prefix_matched_is_matched_prefix():
    candidate = _side("candidate-arch")
    candidate["terminal_complete"] = False
    candidate["artifacts"]["checkpoint"] = "missing"
    result = _assess(candidate=candidate, prefix_aligned=True)
    assert result["comparison_class"] == "MATCHED_PREFIX"
    assert "STOP_FOR_COST" in result["decision_scope"]


def test_same_rows_different_training_recipe_is_paired_endpoint():
    candidate = _side("candidate-arch")
    candidate["comparison_identity"]["schedule_identity"] = "other-schedule"
    result = _assess(candidate=candidate)
    assert result["comparison_class"] == "PAIRED_ENDPOINT"
    assert "causal attribution to architecture or mechanism" in result["forbidden_claims"]


def test_scalar_only_history_is_context_only():
    candidate = {"comparison_identity": {}, "terminal_complete": True}
    reference = {"comparison_identity": {}, "terminal_complete": True}
    result = _assess(
        candidate=candidate,
        reference=reference,
        scalar_context_available=True,
    )
    assert result["comparison_class"] == "CONTEXT_ONLY"
    assert result["decision_scope"] == ["historical context"]


def test_missing_field_is_not_implicitly_matched():
    candidate = _side("candidate-arch")
    del candidate["comparison_identity"]["optimizer_mode"]
    result = _assess(candidate=candidate)
    assert "candidate.optimizer_mode" in result["missing_fields"]
    assert "optimizer_mode" not in result["matched_fields"]
    assert result["strict_ready"] is False


def test_manual_strict_record_cannot_hide_unaccounted_identity():
    result = _assess()
    result["matched_fields"].pop("optimizer_mode")
    with pytest.raises(ValueError, match="every strict identity field"):
        validate_comparison_readiness(result)


def test_consumed_role_is_not_reported_as_untouched():
    role_event = {
        "role_event_id": "role-1",
        "trajectory_id": "trajectory-1",
        "dataset_identity": "dataset",
        "row_manifest_hash": "rows",
        "role_name": "official_validation",
        "access_kind": "metric_computed",
        "selection_used": True,
    }
    evidence = [
        {
            "evidence_id": "ev-1",
            "role_use": {
                "official_validation": "untouched",
                "test_dev": "untouched",
                "test_challenge": "untouched",
            },
        }
    ]
    index = build_role_reuse_index([role_event], evidence)
    assert "official_validation" not in index["protected_roles_reported_untouched"]
    assert "official_validation" in index["full_training_membership_conflicts"]


def test_row_bootstrap_cannot_be_training_stochasticity():
    with pytest.raises(ValueError, match="row bootstrap"):
        validate_stochasticity(
            {
                "row_bootstrap_uncertainty": {"status": "measured"},
                "training_stochasticity": {
                    "status": "measured",
                    "estimation_method": "paired row bootstrap",
                    "source": "row_bootstrap_uncertainty",
                    "same_contract_repeat_ids": ["a", "b"],
                    "n_repeats": 2,
                    "stochasticity_floor_eV": 0.001,
                },
            }
        )


def test_scalar_metric_cannot_validate_as_reference_bundle():
    with pytest.raises(ValueError, match="reference-bundle format"):
        validate_reference_bundle({"mae_eV": 0.1})


def test_server_prelaunch_requires_strict_and_future_role_trace_plans(tmp_path):
    readiness = _assess()
    result = validate_server_scientific_prelaunch(
        comparison_readiness=readiness,
        experiment_purpose="architecture_comparison",
        planned_role_event_kinds=[
            "training_membership",
            "prediction_input",
            "labels_read",
            "metric_computed",
            "selection_used",
            "external_submission",
        ],
        trace_field_declarations={
            "optimizer_step": True,
            "sample_presentations": True,
            "epoch_or_pass": True,
            "learning_rate": True,
            "live_train_metric": True,
            "live_dev_metric": True,
            "ema_dev_metric": False,
            "checkpoint_identity": True,
        },
    )
    assert result["gate"] == "PASS"
    digest = write_server_comparison_prelaunch(
        tmp_path / "comparison_readiness_prelaunch.json",
        comparison_readiness=readiness,
        experiment_purpose="architecture_comparison",
        planned_role_event_kinds=[
            "training_membership",
            "prediction_input",
            "labels_read",
            "metric_computed",
            "selection_used",
            "external_submission",
        ],
        trace_field_declarations={
            "optimizer_step": True,
            "sample_presentations": True,
            "epoch_or_pass": True,
            "learning_rate": True,
            "live_train_metric": True,
            "live_dev_metric": True,
            "ema_dev_metric": False,
            "checkpoint_identity": True,
        },
    )
    assert len(digest) == 64
    assert (tmp_path / "comparison_readiness_prelaunch.json").is_file()

    non_strict = copy.deepcopy(readiness)
    non_strict["strict_ready"] = False
    non_strict["strict_status"] = "PENDING"
    non_strict["comparison_class"] = "CONTEXT_ONLY"
    non_strict["blocker_codes"] = ["MISSING_ALIGNED_PREDICTIONS"]
    with pytest.raises(ValueError, match="blocked"):
        validate_server_scientific_prelaunch(
            comparison_readiness=non_strict,
            experiment_purpose="architecture_comparison",
            planned_role_event_kinds=[],
            trace_field_declarations={},
        )
