import pytest

from molgap.screen_policy import (
    PHYSICAL_BATCH_PER_DEVICE,
    REQUIRED_PAIRED_FIELDS,
    SCREEN_POLICY,
    canonical_fingerprint,
    evaluate_reference_gain,
    validate_paired_screen_contract,
    validate_reference_screen_contract,
    validate_screen_arm,
)


def make_arm(name="baseline", **changes):
    arm = {
        "arm": name,
        "task_id": "qm9-gape-s42-v1",
        "platform_id": "kaggle2",
        "accelerator": "NvidiaTeslaT4",
        "data_role_fingerprint": "split-sha",
        "seed": 42,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5",
        "schedule_fingerprint": "cosine40-eta1e-6",
        "sample_exposure": "30000x40",
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }
    arm.update(changes)
    return arm


def test_exact_batch128_is_the_only_screen_batch():
    assert validate_screen_arm(physical_batch_per_device=128) == {
        "policy": SCREEN_POLICY,
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "effective_batch_per_optimizer_step": 128,
    }
    assert PHYSICAL_BATCH_PER_DEVICE == 128


@pytest.mark.parametrize("batch", [48, 96, 127, 129, 256])
def test_non_128_physical_batch_is_rejected(batch):
    with pytest.raises(ValueError, match="exactly 128"):
        validate_screen_arm(physical_batch_per_device=batch)


@pytest.mark.parametrize("field", REQUIRED_PAIRED_FIELDS)
def test_paired_screen_rejects_every_shared_contract_mismatch(field):
    candidate = make_arm("candidate")
    candidate[field] = f"different-{field}"
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_paired_screen_contract([make_arm(), candidate])


def test_paired_screen_accepts_only_matched_arms():
    result = validate_paired_screen_contract(
        [make_arm(), make_arm("random-control"), make_arm("candidate")]
    )
    assert result["arm_count"] == 3
    assert result["shared"]["task_id"] == "qm9-gape-s42-v1"
    assert result["resource"]["physical_batch_per_device"] == 128


def make_certificate(platform_id: str, accelerator: str) -> dict:
    return {
        "format": "molgap-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": platform_id,
        "accelerator": accelerator,
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


def make_reference_arm(
    *, platform_id: str, accelerator: str, model_id: str, run_id: str
) -> tuple[dict, dict]:
    certificate = make_certificate(platform_id, accelerator)
    arm = {
        "run_id": run_id,
        "model_id": model_id,
        "architecture_fingerprint": ("a" if model_id == "baseline" else "b") * 64,
        "source_archive_sha256": ("c" if model_id == "baseline" else "d") * 64,
        "result_artifact_sha256": ("7" if model_id == "baseline" else "8") * 64,
        "platform_id": platform_id,
        "accelerator": accelerator,
        "runtime_certificate_id": canonical_fingerprint(certificate),
        "benchmark_id": "pcqm-fixed-100k-v1",
        "data_role_fingerprint": "e" * 64,
        "row_order_fingerprint": "f" * 64,
        "feature_fingerprint": "ogb9-bond3-rwse16",
        "target_fingerprint": "gap-eV",
        "seed": 42,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine-step-v1",
        "loss_fingerprint": "normalized-gap-l1",
        "target_transform_fingerprint": "train-mean-sample-std",
        "selection_fingerprint": "internal-dev-min-mae",
        "role_access_fingerprint": "train-plus-internal-dev-no-sealed-v1",
        "sample_exposure": 4_000_000,
        "tail_batch_policy": "drop_last",
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }
    return arm, certificate


def test_reference_policy_allows_cross_platform_without_retraining_baseline():
    reference, reference_certificate = make_reference_arm(
        platform_id="kaggle1", accelerator="Tesla T4", model_id="baseline", run_id="ref-v1"
    )
    reference["frozen_reference"] = True
    reference["result_artifact_sha256"] = "9" * 64
    reference["stochasticity_floor_eV"] = 0.003
    reference["minimum_material_gain_eV"] = 0.003
    candidate, candidate_certificate = make_reference_arm(
        platform_id="scnet-kunshan", accelerator="DCU K100", model_id="candidate", run_id="candidate-v3"
    )
    certificates = {
        reference["runtime_certificate_id"]: reference_certificate,
        candidate["runtime_certificate_id"]: candidate_certificate,
    }
    result = validate_reference_screen_contract(
        reference=reference,
        candidate=candidate,
        runtime_certificates=certificates,
    )
    assert result["comparison_mode"] == "immutable-reference-no-baseline-rerun"
    assert result["cross_platform"] is True
    assert result["reference_run_id"] == "ref-v1"
    assert result["candidate_run_id"] == "candidate-v3"


def test_reference_policy_rejects_a_scientific_contract_change():
    reference, reference_certificate = make_reference_arm(
        platform_id="kaggle1", accelerator="Tesla T4", model_id="baseline", run_id="ref-v1"
    )
    reference["frozen_reference"] = True
    reference["result_artifact_sha256"] = "9" * 64
    reference["stochasticity_floor_eV"] = 0.003
    reference["minimum_material_gain_eV"] = 0.003
    candidate, candidate_certificate = make_reference_arm(
        platform_id="kaggle2", accelerator="Tesla T4", model_id="candidate", run_id="candidate-v1"
    )
    candidate["sample_exposure"] += 128
    with pytest.raises(ValueError, match="Reference screen contract mismatch"):
        validate_reference_screen_contract(
            reference=reference,
            candidate=candidate,
            runtime_certificates={
                reference["runtime_certificate_id"]: reference_certificate,
                candidate["runtime_certificate_id"]: candidate_certificate,
            },
        )


def test_reference_gain_never_treats_row_bootstrap_as_training_uncertainty():
    too_small = evaluate_reference_gain(
        reference_mae_eV=0.140,
        candidate_mae_eV=0.138,
        stochasticity_floor_eV=0.003,
        minimum_material_gain_eV=0.001,
        paired_row_bootstrap_upper_eV=-0.0001,
    )
    assert too_small["passed"] is False
    assert too_small["row_bootstrap_is_sufficient_alone"] is False
    material = evaluate_reference_gain(
        reference_mae_eV=0.140,
        candidate_mae_eV=0.136,
        stochasticity_floor_eV=0.003,
        minimum_material_gain_eV=0.001,
    )
    assert material["passed"] is True
