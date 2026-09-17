from __future__ import annotations

from pathlib import Path

import pytest

import molgap.v5_desktop as v5
from molgap.screen_policy import REFERENCE_MATCH_FIELDS, REFERENCE_PROVENANCE_FIELDS


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_server_owned_active_500k_is_not_duplicated_by_desktop():
    result = v5.authorize_desktop_500k_action(
        experiment_owner="server",
        desktop_online=True,
        remote_state="running",
    )

    assert result["allowed"] is False
    assert result["reason"] == "server_owned_active_experiment"
    assert result["server_fallback"] is False


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


def test_positive_100k_with_incomplete_500k_does_not_admit_full():
    result = v5.admit_full_scale(
        candidate_100k={"qualified": True},
        candidate_500k={"complete": False, "qualified": False},
        reference={"complete": True},
        paired_comparison={"complete": False},
    )

    assert result["allowed"] is False
    assert result["automatic_full"] is False
    assert result["reason"] == "500k_incomplete"


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
