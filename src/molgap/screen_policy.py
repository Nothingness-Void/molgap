"""Executable guards for historical paired and new reference-based screens.

Version 3 is retained so completed paired experiments remain reproducible.
Version 4 compares a candidate with one immutable baseline record.  Hardware is
provenance, not a scientific matching field; a platform/runtime combination is
qualified once and then reused instead of retraining the baseline per job.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


SCREEN_POLICY = "molgap-screen-comparability-v3"
REFERENCE_SCREEN_POLICY = "molgap-reference-comparability-v4"
RUNTIME_CERTIFICATE_FORMAT = "molgap-runtime-certificate-v1"
PHYSICAL_BATCH_PER_DEVICE = 128
REQUIRED_PAIRED_FIELDS = (
    "task_id",
    "platform_id",
    "accelerator",
    "data_role_fingerprint",
    "seed",
    "precision",
    "optimizer_fingerprint",
    "schedule_fingerprint",
    "sample_exposure",
)

# These fields define the scientific experiment.  They must match the frozen
# reference even when the two runs used different physical platforms.
REFERENCE_MATCH_FIELDS = (
    "benchmark_id",
    "data_role_fingerprint",
    "row_order_fingerprint",
    "feature_fingerprint",
    "target_fingerprint",
    "seed",
    "precision",
    "optimizer_fingerprint",
    "schedule_fingerprint",
    "loss_fingerprint",
    "target_transform_fingerprint",
    "selection_fingerprint",
    "role_access_fingerprint",
    "sample_exposure",
    "tail_batch_policy",
)

# These fields are required for provenance but are intentionally allowed to
# differ between the candidate and reference.
REFERENCE_PROVENANCE_FIELDS = (
    "run_id",
    "model_id",
    "architecture_fingerprint",
    "source_archive_sha256",
    "result_artifact_sha256",
    "platform_id",
    "accelerator",
    "runtime_certificate_id",
)


def canonical_fingerprint(value: Mapping) -> str:
    """Return a stable SHA-256 for a JSON-compatible mapping."""
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_screen_arm(
    *,
    physical_batch_per_device: int,
    device_count: int = 1,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """Validate the immutable resource contract of one independent arm."""
    values = {
        "physical_batch_per_device": physical_batch_per_device,
        "device_count": device_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    if any(not isinstance(value, int) or value < 1 for value in values.values()):
        raise ValueError(f"Screen resource values must be positive integers: {values}")
    if physical_batch_per_device != PHYSICAL_BATCH_PER_DEVICE:
        raise ValueError(
            "New screens require physical batch exactly "
            f"{PHYSICAL_BATCH_PER_DEVICE} per model and device"
        )
    if device_count != 1:
        raise ValueError("Each independently optimized screen arm uses one visible device")
    if gradient_accumulation_steps != 1:
        raise ValueError("Screen arms do not use gradient accumulation")
    return {
        "policy": SCREEN_POLICY,
        **values,
        "effective_batch_per_optimizer_step": PHYSICAL_BATCH_PER_DEVICE,
    }


def validate_paired_screen_contract(arms: Sequence[Mapping]) -> dict:
    """Reject comparisons confounded by task, platform, or training exposure."""
    if len(arms) < 2:
        raise ValueError("A paired screen requires at least two arms")
    normalized = []
    for index, arm in enumerate(arms):
        missing = [field for field in REQUIRED_PAIRED_FIELDS if field not in arm]
        if missing:
            raise ValueError(f"Arm {index} is missing paired fields: {missing}")
        resource = validate_screen_arm(
            physical_batch_per_device=arm.get("physical_batch_per_device"),
            device_count=arm.get("device_count", 1),
            gradient_accumulation_steps=arm.get("gradient_accumulation_steps", 1),
        )
        normalized.append({**dict(arm), **resource})
    reference = normalized[0]
    mismatches = {
        field: [arm[field] for arm in normalized]
        for field in REQUIRED_PAIRED_FIELDS
        if any(arm[field] != reference[field] for arm in normalized[1:])
    }
    if mismatches:
        raise ValueError(f"Paired screen contract mismatch: {mismatches}")
    return {
        "policy": SCREEN_POLICY,
        "arm_count": len(normalized),
        "shared": {field: reference[field] for field in REQUIRED_PAIRED_FIELDS},
        "resource": {
            key: reference[key]
            for key in (
                "physical_batch_per_device",
                "device_count",
                "gradient_accumulation_steps",
                "effective_batch_per_optimizer_step",
            )
        },
    }


def validate_runtime_certificate(certificate: Mapping, contract: Mapping) -> dict:
    """Validate one reusable platform/runtime qualification certificate."""
    required = {
        "format": RUNTIME_CERTIFICATE_FORMAT,
        "status": "accepted",
        "platform_id": contract.get("platform_id"),
        "accelerator": contract.get("accelerator"),
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH_PER_DEVICE,
        "tail_batch_policy": "drop_last",
    }
    mismatches = {
        key: {"expected": expected, "actual": certificate.get(key)}
        for key, expected in required.items()
        if certificate.get(key) != expected
    }
    for key in (
        "software_fingerprint",
        "determinism_fingerprint",
        "calibration_fixture_sha256",
        "calibration_output_sha256",
        "runtime_fingerprint",
    ):
        value = certificate.get(key)
        if not isinstance(value, str) or len(value) != 64:
            mismatches[key] = {"expected": "sha256", "actual": value}
    if certificate.get("calibration_checks_passed") is not True:
        mismatches["calibration_checks_passed"] = {
            "expected": True,
            "actual": certificate.get("calibration_checks_passed"),
        }
    if mismatches:
        raise ValueError(f"Runtime certificate mismatch: {mismatches}")
    certificate_id = canonical_fingerprint(dict(certificate))
    if contract.get("runtime_certificate_id") != certificate_id:
        raise ValueError("Runtime certificate identity does not match the run contract")
    return {"certificate_id": certificate_id, "platform_id": certificate["platform_id"]}


def validate_reference_screen_contract(
    *,
    reference: Mapping,
    candidate: Mapping,
    runtime_certificates: Mapping[str, Mapping],
) -> dict:
    """Validate a cross-platform comparison against one frozen baseline.

    This intentionally does not require a new baseline arm in the candidate
    task.  Each platform/runtime certificate is reusable until any recorded
    execution-semantic field changes.
    """
    normalized = []
    runtime = []
    for label, arm in (("reference", reference), ("candidate", candidate)):
        missing = [
            field
            for field in (*REFERENCE_MATCH_FIELDS, *REFERENCE_PROVENANCE_FIELDS)
            if field not in arm
        ]
        if missing:
            raise ValueError(f"{label} is missing reference fields: {missing}")
        resource = validate_screen_arm(
            physical_batch_per_device=arm.get("physical_batch_per_device"),
            device_count=arm.get("device_count", 1),
            gradient_accumulation_steps=arm.get("gradient_accumulation_steps", 1),
        )
        if arm.get("tail_batch_policy") != "drop_last":
            raise ValueError(f"{label} must use drop_last for exact physical batch 128")
        certificate_id = arm["runtime_certificate_id"]
        if certificate_id not in runtime_certificates:
            raise ValueError(f"Missing runtime certificate for {label}: {certificate_id}")
        runtime.append(
            validate_runtime_certificate(runtime_certificates[certificate_id], arm)
        )
        normalized.append({**dict(arm), **resource})

    baseline, experiment = normalized
    mismatches = {
        field: {"reference": baseline[field], "candidate": experiment[field]}
        for field in REFERENCE_MATCH_FIELDS
        if experiment[field] != baseline[field]
    }
    if mismatches:
        raise ValueError(f"Reference screen contract mismatch: {mismatches}")
    if reference.get("frozen_reference") is not True:
        raise ValueError("Reference record is not frozen")
    for label, arm in (("reference", reference), ("candidate", candidate)):
        if not isinstance(arm.get("result_artifact_sha256"), str) or len(
            arm["result_artifact_sha256"]
        ) != 64:
            raise ValueError(f"{label} result artifact identity is missing")
    for key in ("stochasticity_floor_eV", "minimum_material_gain_eV"):
        value = reference.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Reference {key} must be a finite non-negative number")

    shared = {field: baseline[field] for field in REFERENCE_MATCH_FIELDS}
    return {
        "policy": REFERENCE_SCREEN_POLICY,
        "comparison_mode": "immutable-reference-no-baseline-rerun",
        "scientific_contract_sha256": canonical_fingerprint(shared),
        "shared": shared,
        "reference_run_id": baseline["run_id"],
        "candidate_run_id": experiment["run_id"],
        "cross_platform": baseline["platform_id"] != experiment["platform_id"],
        "runtime_certificates": runtime,
        "resource": {
            "physical_batch_per_device": PHYSICAL_BATCH_PER_DEVICE,
            "device_count_per_model": 1,
            "gradient_accumulation_steps": 1,
            "tail_batch_policy": "drop_last",
        },
    }


def evaluate_reference_gain(
    *,
    reference_mae_eV: float,
    candidate_mae_eV: float,
    stochasticity_floor_eV: float,
    minimum_material_gain_eV: float,
    paired_row_bootstrap_upper_eV: float | None = None,
) -> dict:
    """Apply a gain floor that includes training-run variation.

    Row bootstrap uncertainty is optional supporting evidence.  It can tighten
    the gate, but it never replaces the training-stochasticity floor.
    """
    values = {
        "reference_mae_eV": reference_mae_eV,
        "candidate_mae_eV": candidate_mae_eV,
        "stochasticity_floor_eV": stochasticity_floor_eV,
        "minimum_material_gain_eV": minimum_material_gain_eV,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError(f"Reference gain inputs must be finite: {values}")
    if stochasticity_floor_eV < 0 or minimum_material_gain_eV < 0:
        raise ValueError("Gain floors must be non-negative")
    gain = float(reference_mae_eV) - float(candidate_mae_eV)
    required = max(float(stochasticity_floor_eV), float(minimum_material_gain_eV))
    bootstrap_passed = (
        paired_row_bootstrap_upper_eV is None
        or (
            math.isfinite(float(paired_row_bootstrap_upper_eV))
            and paired_row_bootstrap_upper_eV < 0
        )
    )
    return {
        "gain_eV": gain,
        "required_gain_eV": required,
        "training_stochasticity_accounted": True,
        "row_bootstrap_is_sufficient_alone": False,
        "paired_row_bootstrap_upper_eV": paired_row_bootstrap_upper_eV,
        "passed": gain >= required and bootstrap_passed,
    }
