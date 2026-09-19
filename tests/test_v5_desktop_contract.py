from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import molgap.v5_desktop as v5
from molgap.comparison_readiness import (
    CAUSAL_REQUIRED_ROLE_KINDS,
    REQUIRED_CANDIDATE_OBSERVED_BINDINGS,
    REQUIRED_OBSERVED_BINDINGS,
    REQUIRED_REFERENCE_ARTIFACTS,
    STRICT_IDENTITY_FIELDS,
    assess_comparison_readiness,
    reference_bundle_digest,
    target_transform_asset_digest,
)
from molgap.screen_policy import REFERENCE_MATCH_FIELDS, REFERENCE_PROVENANCE_FIELDS


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_TARGET_IDENTITY = "synthetic-target"


def _binding() -> dict[str, object]:
    return {
        "owner": "desktop",
        "remote_job_id": "job-500k-1",
        "source_config_identity": "source-config-sha",
        "accepted_input_identity": "graph-cache-sha",
        "remote_output_locator": "s3://durable/output",
        "remote_checkpoint_locator": "s3://durable/checkpoint",
        "resume_contract": "checkpoint-v1",
        "provenance": "experiments/example/submission.json",
        "source_config_frozen": True,
        "input_cache_accepted": True,
        "remote_artifacts_durable": True,
        "resume_supported": True,
    }


def _write_synthetic_file(
    root: Path, relative_path: str, content: bytes
) -> dict[str, str]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "ref": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_synthetic_json(
    root: Path, relative_path: str, payload: dict
) -> dict[str, str]:
    return _write_synthetic_file(
        root,
        relative_path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


def _synthetic_identity(
    architecture: str, transform_sha256: str
) -> dict[str, object]:
    identity: dict[str, object] = {
        field: f"synthetic-{field}" for field in STRICT_IDENTITY_FIELDS
    }
    identity.update(
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
            "target_identity": SYNTHETIC_TARGET_IDENTITY,
            "target_transform_identity": "synthetic-target-transform",
            "target_transform_asset_sha256": transform_sha256,
        }
    )
    return identity


def _synthetic_full_admission_evidence(tmp_path: Path) -> dict[str, object]:
    """Build a complete local V5 comparison package without production evidence."""

    root = tmp_path / "synthetic-v5-repository"
    prefix = "experiments/synthetic_full_admission"

    transform = {
        "format": "molgap-target-transform-asset-v1",
        "asset_id": "synthetic-target-transform",
        "target_identity": SYNTHETIC_TARGET_IDENTITY,
        "mean": 0.0,
        "std": 1.0,
        "ddof": 0,
        "variance_convention": "population",
        "source_row_manifest_sha256": "4" * 64,
        "target_sha256": "3" * 64,
    }
    transform["asset_sha256"] = target_transform_asset_digest(transform)
    transform_binding = _write_synthetic_json(
        root, f"{prefix}/target_transform.json", transform
    )

    def artifact_binding(name: str, *, side: str) -> dict[str, str]:
        if side == "reference" and name == "target_transform_asset":
            return transform_binding
        return _write_synthetic_json(
            root,
            f"{prefix}/{side}_{name}.json",
            {"artifact": name, "side": side},
        )

    candidate_bindings = {
        name: artifact_binding(name, side="candidate")
        for name in sorted(REQUIRED_CANDIDATE_OBSERVED_BINDINGS)
    }
    reference_bindings = {
        name: artifact_binding(name, side="reference")
        for name in sorted(REQUIRED_OBSERVED_BINDINGS)
    }
    contract_binding = _write_synthetic_json(
        root, f"{prefix}/contract.json", {"contract": "synthetic-v5"}
    )
    prediction_manifest = {
        "prediction_sha256": _write_synthetic_file(
            root, f"{prefix}/reference_predictions.bin", b"predictions"
        )["sha256"],
        "source_idx_sha256": _write_synthetic_file(
            root, f"{prefix}/reference_source_idx.bin", b"source-index"
        )["sha256"],
        "target_sha256": _write_synthetic_file(
            root, f"{prefix}/reference_targets.bin", b"targets"
        )["sha256"],
        "ordering_semantics": "source_idx ascending",
        "evaluation_role_identity": "synthetic-development",
        "row_count": 10,
        "unique_source_idx": 10,
    }
    reference_bindings["prediction_manifest"] = _write_synthetic_json(
        root, f"{prefix}/reference_prediction_manifest.json", prediction_manifest
    )
    reference_bindings["target_manifest"] = _write_synthetic_json(
        root,
        f"{prefix}/reference_target_manifest.json",
        {
            "target_identity": SYNTHETIC_TARGET_IDENTITY,
            "target_sha256": prediction_manifest["target_sha256"],
            "row_count": prediction_manifest["row_count"],
        },
    )

    bundle = {
        "format": "molgap-reference-bundle-v1",
        "reference_bundle_id": "synthetic-reference-bundle",
        "reference_id": "synthetic-reference",
        "contract_ref": contract_binding["ref"],
        "architecture_config_identity": "synthetic-reference-architecture",
        "source_commit_or_archive": "1" * 40,
        "checkpoint_identity": "synthetic-reference-checkpoint",
        "runtime_certificate_ref": reference_bindings["runtime_certificate"]["ref"],
        "prediction_manifest": prediction_manifest,
        "row_manifest_ref": reference_bindings["row_manifest"]["ref"],
        "target_manifest_ref": reference_bindings["target_manifest"]["ref"],
        "trace_manifest_ref": reference_bindings["trace_manifest"]["ref"],
        "role_history_ref": reference_bindings["role_history"]["ref"],
        "target_transform_asset_ref": transform_binding["ref"],
        "cost_records_ref": reference_bindings["cost_records"]["ref"],
        "acceptance_ref": reference_bindings["acceptance"]["ref"],
        "decision_ref": reference_bindings["decision"]["ref"],
        "comparison_identity": _synthetic_identity(
            "synthetic-reference-architecture", transform["asset_sha256"]
        ),
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
    bundle_binding = _write_synthetic_json(
        root, f"{prefix}/reference_bundle.json", bundle
    )

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

    def side(architecture: str, bindings: dict[str, dict[str, str]]) -> dict[str, object]:
        return {
            "comparison_identity": _synthetic_identity(
                architecture, transform["asset_sha256"]
            ),
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

    reference_side = side("synthetic-reference-architecture", reference_bindings)
    reference_side["reference_bundle_id"] = bundle["reference_bundle_id"]
    reference_side["reference_bundle_sha256"] = reference_bundle_digest(bundle)
    readiness = assess_comparison_readiness(
        candidate_id="synthetic-candidate",
        candidate=side("synthetic-candidate-architecture", candidate_bindings),
        reference_id=bundle["reference_id"],
        reference=reference_side,
        experiment_purpose="architecture_comparison",
        intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
    )
    readiness_binding = _write_synthetic_json(
        root, f"{prefix}/comparison_readiness.json", readiness
    )
    return {
        "repo_root": root,
        "comparison_readiness_ref": readiness_binding["ref"],
        "reference_bundle_ref": bundle_binding["ref"],
    }


def _complete_d8_candidate() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {"qualified": True},
        {
            "complete": True,
            "qualified": True,
            "identity_frozen": True,
            "target_hardware_cost_recorded": True,
            "role_use_history_recorded": True,
            "recovery_schedule_recorded": True,
            "duplicate_full_evidence_checked": True,
        },
    )


def test_desktop_shutdown_requires_durable_artifacts_without_server_handoff():
    result = v5.validate_desktop_shutdown_binding(_binding())

    assert result["shutdown_ready"]
    assert result["owner"] == "desktop"
    assert result["server_handoff_required"] is False
    assert result["artifact_status"] == "durable"
    assert result["full_handoff_status"] == "not_applicable"


def test_shutdown_rejects_server_monitor_dependency():
    binding = _binding()
    binding["server_monitor_required"] = True

    with pytest.raises(ValueError, match="server handoff"):
        v5.validate_desktop_shutdown_binding(binding)


def test_stale_desktop_conversation_uses_authoritative_remote_state():
    result = v5.reconcile_desktop_remote_state(
        _binding(),
        {
            "authoritative": True,
            "remote_job_id": "job-500k-1",
            "state": "complete",
        },
    )

    assert result["next_action"] == "retrieve_and_accept_artifacts"
    assert result["automatic_resubmission"] is False
    assert result["stale_local_state_ignored"] is True


def test_non_authoritative_remote_state_cannot_trigger_resubmission():
    result = v5.reconcile_desktop_remote_state(
        _binding(),
        {"state": "failed", "remote_job_id": "job-500k-1"},
    )

    assert result["next_action"] == "query_authoritative_status"
    assert result["automatic_resubmission"] is False


def test_authoritative_status_without_exact_job_identity_stays_unknown():
    result = v5.reconcile_desktop_remote_state(
        _binding(),
        {"authoritative": True, "state": "complete"},
    )

    assert result["execution_status"] == "unknown"
    assert result["next_action"] == "query_authoritative_status"
    assert result["automatic_resubmission"] is False


def test_server_owned_active_500k_is_not_duplicated_by_desktop():
    result = v5.authorize_desktop_500k_action(
        experiment_owner="server",
        desktop_online=True,
        remote_state="running",
    )

    assert result["allowed"] is False
    assert result["reason"] == "server_owned_active_experiment"
    assert result["server_fallback"] is False


def test_server_owned_terminal_500k_is_not_taken_over_by_desktop():
    result = v5.authorize_desktop_500k_action(
        experiment_owner="server",
        desktop_online=True,
        remote_state="complete",
    )

    assert result["allowed"] is False
    assert result["reason"] == "server_owned_experiment"
    assert result["next_action"] == "leave_with_server_owner"
    assert result["ownership_transfer"] is False


def test_desktop_owned_500k_remains_owned_while_desktop_is_offline():
    result = v5.authorize_desktop_500k_action(
        experiment_owner="desktop",
        desktop_online=False,
        remote_state="running",
    )

    assert result["allowed"] is False
    assert result["reason"] == "desktop_offline_silent_time"
    assert result["owner"] == "desktop"
    assert result["ownership_transfer"] is False


def test_missing_reference_is_pending_and_never_retrains_baseline():
    result = v5.classify_reference_comparison(reference=None, candidate={})

    assert result["comparison_status"] == "pending"
    assert result["baseline_rerun_allowed"] is False
    assert result["reason"] == "reference_payload_missing"


def test_incompatible_v3_result_is_not_a_strict_v4_comparator():
    result = v5.classify_reference_comparison(
        reference={"policy": "molgap-screen-comparability-v3"},
        candidate={"policy": "molgap-screen-comparability-v3"},
    )

    assert result["comparison_status"] == "incompatible"
    assert result["strict_v4_comparator"] is False
    assert result["baseline_rerun_allowed"] is False


def test_complete_reference_delegates_to_existing_v4_policy(monkeypatch):
    fields = (*REFERENCE_MATCH_FIELDS, *REFERENCE_PROVENANCE_FIELDS)
    reference = {field: f"reference-{field}" for field in fields}
    candidate = {field: f"reference-{field}" for field in fields}
    reference.update(
        {
            "frozen_reference": True,
            "physical_batch_per_device": 128,
            "device_count": 1,
            "gradient_accumulation_steps": 1,
            "tail_batch_policy": "drop_last",
            "stochasticity_floor_eV": 0.001,
            "minimum_material_gain_eV": 0.003,
        }
    )
    candidate.update(
        {
            "physical_batch_per_device": 128,
            "device_count": 1,
            "gradient_accumulation_steps": 1,
            "tail_batch_policy": "drop_last",
        }
    )
    monkeypatch.setattr(
        v5,
        "validate_reference_screen_contract",
        lambda **_: {"policy": v5.REFERENCE_SCREEN_POLICY},
    )

    result = v5.classify_reference_comparison(
        reference=reference,
        candidate=candidate,
    )

    assert result["comparison_status"] == "ready"
    assert result["strict_v4_comparator"] is True
    assert result["baseline_rerun_allowed"] is False


def test_consumed_role_is_not_called_untouched():
    result = v5.decide_role_use(
        {"official_validation": True, "test_dev": False},
        "official_validation",
    )

    assert result["role_use_status"] == "consumed"
    assert result["can_read"] is False


def test_untouched_protected_role_still_requires_explicit_authorization():
    result = v5.decide_role_use({}, "test_dev")

    assert result["role_use_status"] == "untouched"
    assert result["can_read"] is False
    assert result["authorization_status"] == "required"


def test_explicitly_authorized_untouched_protected_role_can_be_read():
    result = v5.decide_role_use(
        {},
        "test_dev",
        explicit_authorization=True,
    )

    assert result["can_read"] is True
    assert result["authorization_status"] == "authorized"


def test_positive_100k_with_incomplete_500k_does_not_admit_full(tmp_path):
    evidence = _synthetic_full_admission_evidence(tmp_path)
    result = v5.admit_full_scale(
        candidate_100k={"qualified": True},
        candidate_500k={"complete": False, "qualified": False},
        reference={"complete": True},
        paired_comparison={"complete": False},
        **evidence,
    )

    assert result["allowed"] is False
    assert result["automatic_full"] is False
    assert result["reason"] == "500k_incomplete"


def test_bare_completion_booleans_do_not_admit_full():
    result = v5.admit_full_scale(
        candidate_100k={"qualified": True},
        candidate_500k={
            "complete": True,
            "qualified": True,
            "identity_frozen": True,
            "target_hardware_cost_recorded": True,
            "role_use_history_recorded": True,
            "recovery_schedule_recorded": True,
            "duplicate_full_evidence_checked": True,
        },
        reference={"complete": True},
        paired_comparison={
            "complete": True,
            "artifacts_aligned": True,
            "strict_comparison_passed": True,
            "statistical_limitations_recorded": True,
        },
        explicit_desktop_authorization=True,
    )

    assert result["allowed"] is False
    assert result["reason"] == "comparison_evidence_not_bound"


def test_complete_d8_evidence_requires_authorization_and_admits_when_given(tmp_path):
    evidence = _synthetic_full_admission_evidence(tmp_path)
    candidate_100k, candidate_500k = _complete_d8_candidate()
    without_authorization = v5.admit_full_scale(
        candidate_100k=candidate_100k,
        candidate_500k=candidate_500k,
        reference={},
        paired_comparison={},
        explicit_desktop_authorization=False,
        **evidence,
    )
    result = v5.admit_full_scale(
        candidate_100k=candidate_100k,
        candidate_500k=candidate_500k,
        reference={},
        paired_comparison={},
        explicit_desktop_authorization=True,
        **evidence,
    )

    assert without_authorization["allowed"] is False
    assert without_authorization["reason"] == "explicit_desktop_authorization_required"
    assert result["allowed"] is True
    assert result["full_handoff_status"] == "authorized"
    assert result["automatic_full"] is False


@pytest.mark.parametrize(
    ("comparison_goal", "comparison_class", "allowed"),
    (
        ("causal_architecture", "STRICT_CAUSAL", True),
        ("causal_architecture", "PAIRED_ENDPOINT", False),
        ("delivered_model", "STRICT_CAUSAL", True),
        ("delivered_model", "PAIRED_ENDPOINT", True),
        ("delivered_model", "MATCHED_PREFIX", False),
        ("convergence", "MATCHED_PREFIX", False),
    ),
)
def test_full_admission_comparison_goal_mapping(
    comparison_goal, comparison_class, allowed
):
    assert (
        v5._comparison_goal_allows_full_admission(
            comparison_goal, comparison_class
        )
        is allowed
    )


def test_track_b_positive_does_not_change_track_a_registry_automatically():
    result = v5.track_b_delivery_guard(positive=True)

    assert result["production_registry_change_allowed"] is False
    assert result["automatic"] is False


def test_shared_helper_failure_has_no_local_bypass():
    result = v5.shared_helper_failure_decision("shared_acceptance", "boom")

    assert result["local_bypass_allowed"] is False
    assert result["next_action"] == "preserve_failure_and_report"


def test_full_runner_checkpoint_retains_resume_state(tmp_path):
    import torch

    from molgap.pcqm_k1_full_runner import _save_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    certificate_id = "a" * 64
    source_commit = "b" * 40
    source_archive_sha256 = "c" * 64
    checkpoint_path = tmp_path / "last_checkpoint.pt"

    _save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=128,
        target_stats={"mean_eV": 0.0, "sample_std_eV": 1.0},
        trace=[{"global_step": 128}],
        runtime_fingerprint="d" * 64,
        runtime_certificate_ids=[certificate_id],
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha256,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert {
        "model",
        "optimizer",
        "scheduler",
        "rng_state",
        "global_step",
        "pass_index",
        "next_batch_in_pass",
    }.issubset(checkpoint)
    assert checkpoint["global_step"] == 128
    assert checkpoint["runtime_certificate_ids"] == [certificate_id]
    assert checkpoint["source_commit"] == source_commit
    assert checkpoint["source_archive_sha256"] == source_archive_sha256


def test_desktop_agents_patch_has_no_server_style_monitor():
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    section = text.split("## Desktop offline behavior", maxsplit=1)[1]

    assert "no default heartbeat monitor" in section
    assert "server agent automatically monitors" in section
    assert "30-minute" not in section


@pytest.mark.parametrize(
    "relative_path",
    (
        "experiments/pcqm_gptrans_t_100k_v4/v5_evidence.json",
        "experiments/pcqm_k1_gptrans_full_fusion/v5_evidence.json",
        "experiments/pcqm_geometry_transfer_500k/v5_evidence.json",
        "experiments/pcqm_edge_state_full/v5_evidence.json",
        "experiments/pcqm_gine_expert/v5_evidence.json",
        "experiments/pcqm_route_b/v5_evidence.json",
        "experiments/pcqm_gap_architecture/v5_evidence.json",
        "experiments/pcqm_gptrans_t_500k/v5_evidence.json",
    ),
)
def test_migrated_v5_evidence_envelopes_validate(relative_path):
    evidence = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))

    result = v5.validate_v5_evidence_envelope(evidence, repo_root=REPO_ROOT)

    assert result["valid"] is True
    assert result["evidence_id"] == evidence["evidence_id"]


def test_geometry_migration_does_not_upgrade_incompatible_evidence():
    path = REPO_ROOT / "experiments/pcqm_geometry_transfer_500k/v5_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["outcome"]["comparison_status"] == "incompatible_with_strict_v4"
    assert evidence["outcome"]["transfer_status"] == "blocked"
    assert evidence["migration"]["scientific_reinterpretation"] is False


def test_external_edgestate_migration_preserves_submission_boundary():
    path = REPO_ROOT / "experiments/pcqm_edge_state_full/v5_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["outcome"]["scientific_status"] == "external_submission_pending_review"
    assert evidence["role_use"]["test_dev"] == "consumed"
    assert evidence["migration"]["scientific_reinterpretation"] is False


def test_gine_migration_preserves_specialist_only_boundary():
    path = REPO_ROOT / "experiments/pcqm_gine_expert/v5_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["outcome"]["scientific_status"] == "accepted_specialist_not_leaderboard"
    assert evidence["outcome"]["transfer_status"] == "task_routed_only"
    assert evidence["migration"]["scientific_reinterpretation"] is False


def test_matched_500k_migration_waits_for_second_durable_k1_copy():
    envelope = REPO_ROOT / "experiments/pcqm_500k_v4_evidence/v5_evidence.json"
    reference_index = (REPO_ROOT / "models/REFERENCE_INDEX.md").read_text(
        encoding="utf-8"
    )

    assert not envelope.exists()
    assert "second durable" in reference_index


def test_v5_evidence_rejects_overloaded_accepted_flag():
    evidence = {
        "format": v5.V5_EVIDENCE_FORMAT,
        "contract": v5.V5_CONTRACT_ID,
        "accepted": True,
    }

    with pytest.raises(ValueError, match="separate outcome dimensions"):
        v5.validate_v5_evidence_envelope(evidence)
