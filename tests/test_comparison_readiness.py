from __future__ import annotations

import copy
import json

import pytest

from molgap.comparison_readiness import (
    CAUSAL_REQUIRED_ROLE_KINDS,
    REQUIRED_CANDIDATE_OBSERVED_BINDINGS,
    REQUIRED_OBSERVED_BINDINGS,
    REQUIRED_REFERENCE_ARTIFACTS,
    STRICT_IDENTITY_FIELDS,
    assess_comparison_prelaunch,
    assess_comparison_readiness,
    reference_bundle_digest,
    target_transform_asset_digest,
    validate_comparison_readiness,
    validate_reference_bundle,
    validate_server_comparison_prelaunch,
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
            "target_transform_identity": "test-target-transform",
            "target_transform_asset_sha256": _target_transform_asset()["asset_sha256"],
        }
    )
    return values


def _target_transform_asset() -> dict:
    asset = {
        "format": "molgap-target-transform-asset-v1",
        "asset_id": "test-target-transform",
        "target_identity": "same-target_identity",
        "mean": 0.0,
        "std": 1.0,
        "ddof": 0,
        "variance_convention": "population",
        "source_row_manifest_sha256": "4" * 64,
        "target_sha256": "3" * 64,
    }
    asset["asset_sha256"] = target_transform_asset_digest(asset)
    return asset


def _side(architecture: str) -> dict:
    role_plan = {
        kind: (
            "applicable" if kind in CAUSAL_REQUIRED_ROLE_KINDS else "not_applicable"
        )
        for kind in (
            "training_membership",
            "prediction_input",
            "labels_read",
            "metric_computed",
            "selection_used",
            "external_submission",
        )
    }
    trace_fields = {
        "optimizer_step": True,
        "sample_presentations": True,
        "epoch_or_pass": True,
        "learning_rate": True,
        "live_train_metric": True,
        "live_dev_metric": True,
        "ema_dev_metric": False,
        "checkpoint_identity": True,
    }
    bindings = {
        name: {"ref": f"evidence/{architecture}/{name}.json", "sha256": "a" * 64}
        for name in REQUIRED_CANDIDATE_OBSERVED_BINDINGS
    }
    return {
        "comparison_identity": _identity(architecture),
        "artifacts": {
            **{name: "complete" for name in REQUIRED_REFERENCE_ARTIFACTS},
            "paired_analysis": "complete",
        },
        "prediction_status": "complete",
        "row_alignment_status": "aligned",
        "runtime_certificate_status": "accepted",
        "role_status": "complete",
        "role_applicability_plan": role_plan,
        "observed_role_event_kinds": sorted(CAUSAL_REQUIRED_ROLE_KINDS),
        "trace_status": "complete",
        "trace_field_availability": trace_fields,
        "stochasticity_status": "unavailable",
        "terminal_complete": True,
        "artifact_bindings": bindings,
    }


def _reference_side() -> dict:
    side = _side("reference-arch")
    bundle = _reference_bundle()
    side["reference_bundle_id"] = bundle["reference_bundle_id"]
    side["reference_bundle_sha256"] = reference_bundle_digest(bundle)
    side["artifact_bindings"] = {
        name: {"ref": f"evidence/reference-arch/{name}.json", "sha256": "b" * 64}
        for name in REQUIRED_OBSERVED_BINDINGS
    }
    return side


def _reference_bundle() -> dict:
    return {
        "format": "molgap-reference-bundle-v1",
        "reference_bundle_id": "reference-bundle",
        "reference_id": "reference",
        "contract_ref": "experiments/reference/contract.json",
        "architecture_config_identity": "reference-arch",
        "source_commit_or_archive": "1" * 40,
        "checkpoint_identity": "checkpoint-reference",
        "runtime_certificate_ref": "experiments/reference/runtime.json",
        "prediction_manifest": {
            "prediction_sha256": "1" * 64,
            "source_idx_sha256": "2" * 64,
            "target_sha256": "3" * 64,
            "ordering_semantics": "source_idx ascending",
            "evaluation_role_identity": "same-evaluation_role_identity",
            "row_count": 10,
            "unique_source_idx": 10,
        },
        "row_manifest_ref": "experiments/reference/rows.json",
        "target_manifest_ref": "experiments/reference/targets.json",
        "trace_manifest_ref": "experiments/reference/trace.json",
        "role_history_ref": "experiments/reference/roles.json",
        "target_transform_asset_ref": "experiments/reference/target_transform.json",
        "cost_records_ref": "experiments/reference/cost.json",
        "acceptance_ref": "experiments/reference/acceptance.json",
        "decision_ref": "experiments/reference/decision.md",
        "comparison_identity": _identity("reference-arch"),
        "stochasticity": {
            "row_bootstrap_uncertainty": {"status": "not_requested"},
            "training_stochasticity": {
                "status": "unavailable",
                "estimation_method": "unavailable",
                "source": "unavailable",
                "same_contract_repeat_ids": [],
                "n_repeats": 0,
                "stochasticity_floor_eV": None,
            },
        },
    }


def _write_reference_bundle_tree(repo_root, bundle: dict):
    evidence_root = repo_root / "experiments" / "reference"
    evidence_root.mkdir(parents=True)
    for pointer in (
        bundle["contract_ref"],
        bundle["runtime_certificate_ref"],
        bundle["row_manifest_ref"],
        bundle["target_manifest_ref"],
        bundle["trace_manifest_ref"],
        bundle["role_history_ref"],
        bundle["cost_records_ref"],
        bundle["acceptance_ref"],
    ):
        (repo_root / pointer).write_text("{}\n", encoding="utf-8")
    (repo_root / bundle["decision_ref"]).write_text("decision\n", encoding="utf-8")
    (repo_root / bundle["target_transform_asset_ref"]).write_text(
        json.dumps(_target_transform_asset(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle_path = evidence_root / "reference_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle_path


def _prelaunch(reference_bundle: dict | None = None) -> dict:
    reference_bundle = reference_bundle or _reference_bundle()
    return assess_comparison_prelaunch(
        candidate_id="candidate",
        candidate_plan={
            "comparison_identity": _identity("candidate-arch"),
            "source_config_status": "frozen",
            "source_commit_or_archive": "2" * 40,
        },
        reference_id="reference",
        reference_bundle=reference_bundle,
        experiment_purpose="architecture_comparison",
        intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
        role_applicability_plan=_side("candidate-arch")["role_applicability_plan"],
        trace_plan=_side("candidate-arch")["trace_field_availability"],
        runtime_qualification_plan={
            "status": "declared",
            "runtime_certificate_required": True,
            "qualification_scope": "candidate runtime tuple",
        },
    )


def _assess(candidate: dict | None = None, reference: dict | None = None, **kwargs) -> dict:
    declared_intervention_fields = kwargs.pop(
        "declared_intervention_fields", ["architecture_config_identity"]
    )
    return assess_comparison_readiness(
        candidate_id="candidate",
        candidate=candidate or _side("candidate-arch"),
        reference_id="reference",
        reference=reference or _reference_side(),
        declared_intervention_fields=declared_intervention_fields,
        **kwargs,
    )


def test_all_fields_and_complete_artifacts_are_strict_causal():
    result = _assess()
    assert result["comparison_class"] == "STRICT_CAUSAL"
    assert result["strict_ready"] is True
    validate_comparison_readiness(
        result, evidence_verifier=lambda _pointer, _digest: None
    )


@pytest.mark.parametrize(
    "experiment_purpose",
    (
        "diagnostic",
        "transfer_study",
        "delivery_experiment",
        "contextual_experiment",
        "NO_TRAIN",
    ),
)
def test_noncausal_purpose_with_complete_endpoint_evidence_is_not_strict(
    experiment_purpose,
):
    candidate = _side("reference-arch")
    reference = _reference_side()
    for side in (candidate, reference):
        side["role_applicability_plan"] = {
            kind: "not_applicable" for kind in side["role_applicability_plan"]
        }
        side["observed_role_event_kinds"] = []
        side["trace_field_availability"] = {
            field: False for field in side["trace_field_availability"]
        }
    result = _assess(
        candidate=candidate,
        reference=reference,
        declared_intervention_fields=[],
        experiment_purpose=experiment_purpose,
        intervention_group_id="noncausal-endpoint",
    )
    assert result["strict_ready"] is False
    assert result["comparison_class"] == "PAIRED_ENDPOINT"
    assert result["blocker_codes"] == []
    validate_comparison_readiness(result)


def test_validator_rejects_forged_noncausal_strict_record():
    candidate = _side("reference-arch")
    reference = _reference_side()
    for side in (candidate, reference):
        side["role_applicability_plan"] = {
            kind: "not_applicable" for kind in side["role_applicability_plan"]
        }
        side["observed_role_event_kinds"] = []
        side["trace_field_availability"] = {
            field: False for field in side["trace_field_availability"]
        }
    forged = _assess(
        candidate=candidate,
        reference=reference,
        declared_intervention_fields=[],
        experiment_purpose="diagnostic",
        intervention_group_id="noncausal-endpoint",
    )
    forged["comparison_class"] = "STRICT_CAUSAL"
    forged["strict_ready"] = True
    forged["strict_status"] = "READY"
    with pytest.raises(ValueError, match="causal comparison purpose"):
        validate_comparison_readiness(
            forged, evidence_verifier=lambda _pointer, _digest: None
        )


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
    candidate["trace_field_availability"]["ema_dev_metric"] = True
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
        validate_comparison_readiness(
            result, evidence_verifier=lambda _pointer, _digest: None
        )


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


def test_server_prelaunch_requires_planned_identity_and_real_reference_bundle(tmp_path):
    bundle = _reference_bundle()
    bundle_path = _write_reference_bundle_tree(tmp_path, bundle)
    prelaunch = _prelaunch(bundle)
    result = validate_server_scientific_prelaunch(
        comparison_prelaunch=prelaunch,
        experiment_purpose="architecture_comparison",
        reference_bundle=bundle,
        repo_root=tmp_path,
        reference_bundle_path=bundle_path,
    )
    assert result["gate"] == "PASS"
    digest = write_server_comparison_prelaunch(
        tmp_path / "comparison_readiness_prelaunch.json",
        comparison_prelaunch=prelaunch,
        experiment_purpose="architecture_comparison",
        reference_bundle=bundle,
        repo_root=tmp_path,
        reference_bundle_path=bundle_path,
    )
    assert len(digest) == 64
    assert (tmp_path / "comparison_readiness_prelaunch.json").is_file()

    forged = copy.deepcopy(prelaunch)
    forged["reference_bundle_id"] = "fake-reference-bundle-id"
    with pytest.raises(ValueError, match="does not match"):
        validate_server_scientific_prelaunch(
            comparison_prelaunch=forged,
            experiment_purpose="architecture_comparison",
            reference_bundle=bundle,
            repo_root=tmp_path,
            reference_bundle_path=bundle_path,
        )

    fabricated = _reference_bundle()
    for field in (
        "contract_ref",
        "runtime_certificate_ref",
        "row_manifest_ref",
        "target_manifest_ref",
        "trace_manifest_ref",
        "role_history_ref",
        "target_transform_asset_ref",
        "cost_records_ref",
        "acceptance_ref",
        "decision_ref",
    ):
        fabricated[field] = fabricated[field].replace(
            "experiments/reference/", "experiments/nonexistent/"
        )
    fabricated_path = tmp_path / "experiments" / "fabricated" / "reference_bundle.json"
    fabricated_path.parent.mkdir(parents=True)
    fabricated_path.write_text(
        json.dumps(fabricated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="required repository pointer is missing"):
        validate_server_scientific_prelaunch(
            comparison_prelaunch=_prelaunch(fabricated),
            experiment_purpose="architecture_comparison",
            reference_bundle=fabricated,
            repo_root=tmp_path,
            reference_bundle_path=fabricated_path,
        )
    fabricated_prelaunch_path = tmp_path / "fabricated" / "comparison_readiness_prelaunch.json"
    with pytest.raises(ValueError, match="required repository pointer is missing"):
        write_server_comparison_prelaunch(
            fabricated_prelaunch_path,
            comparison_prelaunch=_prelaunch(fabricated),
            experiment_purpose="architecture_comparison",
            reference_bundle=fabricated,
            repo_root=tmp_path,
            reference_bundle_path=fabricated_path,
        )
    assert not fabricated_prelaunch_path.exists()


def test_noncausal_prelaunch_does_not_claim_missing_comparison_identity():
    prelaunch = assess_comparison_prelaunch(
        candidate_id="reference-recovery",
        candidate_plan={
            "source_config_status": "frozen",
            "source_commit_or_archive": "2" * 40,
        },
        reference_id=None,
        reference_bundle=None,
        experiment_purpose="delivery_experiment",
        intervention_group_id="reference-recovery",
        declared_intervention_fields=[],
        role_applicability_plan=_side("candidate-arch")["role_applicability_plan"],
        trace_plan=_side("candidate-arch")["trace_field_availability"],
        runtime_qualification_plan={
            "status": "declared",
            "runtime_certificate_required": True,
            "qualification_scope": "candidate runtime tuple",
        },
    )
    assert prelaunch["planned_status"] == "PRELAUNCH_NONCAUSAL_PLANNED"
    assert prelaunch["prelaunch_ready"] is True
    assert prelaunch["matched_fields"] == {}
    assert prelaunch["missing_fields"] == []
    assert validate_server_comparison_prelaunch(
        prelaunch, experiment_purpose="delivery_experiment", reference_bundle=None
    )["gate"] == "PASS"
    with pytest.raises(ValueError, match="requires a reference bundle"):
        validate_server_scientific_prelaunch(
            comparison_prelaunch=prelaunch,
            experiment_purpose="delivery_experiment",
            reference_bundle=None,
            repo_root=".",
            reference_bundle_path="missing-reference-bundle.json",
        )


def test_causal_prelaunch_still_blocks_missing_candidate_identity():
    prelaunch = assess_comparison_prelaunch(
        candidate_id="candidate",
        candidate_plan={
            "source_config_status": "frozen",
            "source_commit_or_archive": "2" * 40,
        },
        reference_id="reference",
        reference_bundle=_reference_bundle(),
        experiment_purpose="architecture_comparison",
        intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
        role_applicability_plan=_side("candidate-arch")["role_applicability_plan"],
        trace_plan=_side("candidate-arch")["trace_field_availability"],
        runtime_qualification_plan={
            "status": "declared",
            "runtime_certificate_required": True,
            "qualification_scope": "candidate runtime tuple",
        },
    )
    assert prelaunch["planned_status"] == "PRELAUNCH_BLOCKED"
    assert "PLANNED_IDENTITY_INCOMPLETE" in prelaunch["blocker_codes"]
    with pytest.raises(ValueError, match="blocked"):
        validate_server_comparison_prelaunch(
            prelaunch,
            experiment_purpose="architecture_comparison",
            reference_bundle=_reference_bundle(),
        )


def test_causal_prelaunch_rejects_all_not_applicable_roles():
    bundle = _reference_bundle()
    with pytest.raises(ValueError, match="applicable role kinds"):
        assess_comparison_prelaunch(
            candidate_id="candidate",
            candidate_plan={
                "comparison_identity": _identity("candidate-arch"),
                "source_config_status": "frozen",
                "source_commit_or_archive": "2" * 40,
            },
            reference_id="reference",
            reference_bundle=bundle,
            experiment_purpose="architecture_comparison",
            intervention_group_id="architecture",
            declared_intervention_fields=["architecture_config_identity"],
            role_applicability_plan={
                kind: "not_applicable"
                for kind in _side("candidate-arch")["role_applicability_plan"]
            },
            trace_plan=_side("candidate-arch")["trace_field_availability"],
            runtime_qualification_plan={
                "status": "declared",
                "runtime_certificate_required": True,
                "qualification_scope": "candidate runtime tuple",
            },
        )


def test_causal_prelaunch_rejects_all_false_trace_plan():
    bundle = _reference_bundle()
    with pytest.raises(ValueError, match="requires trace fields"):
        assess_comparison_prelaunch(
            candidate_id="candidate",
            candidate_plan={
                "comparison_identity": _identity("candidate-arch"),
                "source_config_status": "frozen",
                "source_commit_or_archive": "2" * 40,
            },
            reference_id="reference",
            reference_bundle=bundle,
            experiment_purpose="architecture_comparison",
            intervention_group_id="architecture",
            declared_intervention_fields=["architecture_config_identity"],
            role_applicability_plan=_side("candidate-arch")[
                "role_applicability_plan"
            ],
            trace_plan={
                field: False
                for field in _side("candidate-arch")[
                    "trace_field_availability"
                ]
            },
            runtime_qualification_plan={
                "status": "declared",
                "runtime_certificate_required": True,
                "qualification_scope": "candidate runtime tuple",
            },
        )


def test_strict_postrun_requires_external_artifact_verification():
    result = _assess()
    with pytest.raises(ValueError, match="artifact evidence verifier"):
        validate_comparison_readiness(result)
