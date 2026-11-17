"""Validation and identity helpers for small V4 experiment submissions."""
from __future__ import annotations

import importlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from .screen_policy import (
    REFERENCE_MATCH_FIELDS,
    canonical_fingerprint,
    validate_runtime_certificate,
    validate_screen_arm,
)


RUN_SPEC_FORMAT = "molgap-v4-run-spec-v1"


def runtime_calibration_fingerprint(spec: Mapping) -> str:
    """Bind calibration to code, architecture, scientific contract and batch."""
    return canonical_fingerprint(
        {
            "source_bundle_sha256": spec["source"]["bundle_sha256"],
            "model_family": spec["model_family"],
            "model_id": spec["model_id"],
            "architecture_sha256": spec["architecture_sha256"],
            "model_config_sha256": spec["model_config_sha256"],
            "scientific_contract": dict(spec["scientific_contract"]),
            "physical_batch_per_device": spec["physical_batch_per_device"],
            "device_count": spec.get("device_count", 1),
            "gradient_accumulation_steps": spec.get("gradient_accumulation_steps", 1),
        }
    )


def validate_run_spec(spec: Mapping, runtime_certificate: Mapping) -> dict:
    """Validate a model/config payload before it is copied to a remote account."""
    if spec.get("format") != RUN_SPEC_FORMAT:
        raise ValueError("Unsupported V4 run-spec format")
    for field in (
        "run_id",
        "model_family",
        "model_id",
        "architecture_sha256",
        "platform_id",
        "accelerator",
        "runner_entrypoint",
    ):
        if not isinstance(spec.get(field), str) or not spec[field].strip():
            raise ValueError(f"V4 run spec requires non-empty {field}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", spec["run_id"]):
        raise ValueError("run_id must be a single safe path component")
    if ":" not in spec["runner_entrypoint"]:
        raise ValueError("runner_entrypoint must use module:function form")
    module_name, function_name = spec["runner_entrypoint"].split(":", 1)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module_name) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", function_name
    ):
        raise ValueError("runner_entrypoint contains an invalid Python identifier")

    model_config = spec.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("model_config must be a JSON object")
    if len(spec["architecture_sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in spec["architecture_sha256"].lower()
    ):
        raise ValueError("architecture_sha256 must be a SHA-256 hex digest")
    model_config_sha = canonical_fingerprint(model_config)
    if spec.get("model_config_sha256") != model_config_sha:
        raise ValueError("Model configuration fingerprint does not match its payload")

    source = spec.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("source must be a JSON object")
    for field in ("bundle_sha256", "source_commit"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"V4 source identity requires {field}")
    if len(source["bundle_sha256"]) != 64:
        raise ValueError("source.bundle_sha256 must be a SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in source["bundle_sha256"].lower()):
        raise ValueError("source.bundle_sha256 must be hexadecimal")
    if not isinstance(spec.get("runner_parameters", {}), Mapping):
        raise ValueError("runner_parameters must be a JSON object")

    contract = spec.get("scientific_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("scientific_contract must be a JSON object")
    missing = [field for field in REFERENCE_MATCH_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"V4 scientific contract is missing fields: {missing}")
    for field in (
        "benchmark_id",
        "data_role_fingerprint",
        "row_order_fingerprint",
        "feature_fingerprint",
        "target_fingerprint",
        "optimizer_fingerprint",
        "schedule_fingerprint",
        "loss_fingerprint",
        "target_transform_fingerprint",
        "selection_fingerprint",
        "role_access_fingerprint",
        "tail_batch_policy",
    ):
        if not isinstance(contract[field], str) or not contract[field].strip():
            raise ValueError(f"V4 contract field {field} must be a non-empty string")
    if contract["precision"] != "fp32":
        raise ValueError("V4 reference screens require FP32")
    if contract["tail_batch_policy"] != "drop_last":
        raise ValueError("V4 reference screens require drop_last")
    if not isinstance(contract["sample_exposure"], (str, int)):
        raise ValueError("sample_exposure must be a stable string or positive integer")
    if isinstance(contract["sample_exposure"], int) and contract["sample_exposure"] <= 0:
        raise ValueError("sample_exposure must be positive")
    if not isinstance(contract["seed"], int) or contract["seed"] < 0:
        raise ValueError("V4 seed must be a non-negative integer")

    resource = validate_screen_arm(
        physical_batch_per_device=spec.get("physical_batch_per_device"),
        device_count=spec.get("device_count", 1),
        gradient_accumulation_steps=spec.get("gradient_accumulation_steps", 1),
    )
    if contract["seed"] < 0:
        raise ValueError("V4 seed must be non-negative")
    if contract["precision"] != "fp32":
        raise ValueError("V4 scientific precision must remain FP32")

    certificate_id = spec.get("runtime_certificate_id")
    calibration_fingerprint = runtime_calibration_fingerprint(spec)
    if spec.get("runtime_calibration_fingerprint") not in (None, calibration_fingerprint):
        raise ValueError("Runtime calibration fingerprint does not match the run identity")
    if runtime_certificate is None:
        if certificate_id is not None:
            raise ValueError("A runtime certificate is required for the supplied certificate id")
    else:
        if not isinstance(certificate_id, str) or len(certificate_id) != 64:
            raise ValueError("runtime_certificate_id must be a SHA-256 fingerprint")
        certificate_contract = {
            **dict(contract),
            "platform_id": spec["platform_id"],
            "accelerator": spec["accelerator"],
            "runtime_certificate_id": certificate_id,
            "runtime_calibration_fingerprint": calibration_fingerprint,
        }
        validate_runtime_certificate(runtime_certificate, certificate_contract)

    normalized = {
        **{
            key: value
            for key, value in spec.items()
            if key not in {"resource", "scientific_contract_sha256", "submission_payload_sha256"}
        },
        "model_config_sha256": model_config_sha,
        "scientific_contract": dict(contract),
        "resource": resource,
        "runtime_certificate_id": certificate_id,
        "runtime_calibration_fingerprint": calibration_fingerprint,
    }
    normalized["scientific_contract_sha256"] = canonical_fingerprint(dict(contract))
    normalized["submission_payload_sha256"] = canonical_fingerprint(normalized)
    return normalized


def write_run_spec(
    path: Path, spec: Mapping, runtime_certificate: Mapping | None = None
) -> dict:
    normalized = validate_run_spec(spec, runtime_certificate)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(normalized, handle, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(path)
    return normalized


def bind_runtime_certificate(
    spec: Mapping,
    runtime_certificate: Mapping,
    *,
    runner_parameter_updates: Mapping | None = None,
) -> dict:
    """Create the training-stage manifest from an accepted preflight result."""
    candidate = {
        key: value
        for key, value in spec.items()
        if key not in {"resource", "scientific_contract_sha256", "submission_payload_sha256"}
    }
    candidate["runtime_certificate_id"] = canonical_fingerprint(runtime_certificate)
    if runner_parameter_updates:
        parameters = dict(candidate.get("runner_parameters", {}))
        parameters.update(dict(runner_parameter_updates))
        candidate["runner_parameters"] = parameters
    return validate_run_spec(candidate, runtime_certificate)


def read_run_spec(path: Path, runtime_certificate: Mapping | None = None) -> dict:
    return validate_run_spec(json.loads(path.read_text(encoding="utf-8")), runtime_certificate)


def execute_run_spec(
    spec: Mapping,
    *,
    mode: str,
    runtime_certificate: Mapping | None = None,
):
    """Dispatch a validated spec to its small family-specific adapter."""
    if mode not in {"preflight", "train"}:
        raise ValueError("V4 execution mode must be preflight or train")
    if mode == "preflight" and runtime_certificate is not None:
        raise ValueError("Preflight creates the platform certificate; it does not consume one")
    if mode == "train" and runtime_certificate is None:
        raise ValueError("Training requires an accepted platform runtime certificate")
    normalized = validate_run_spec(spec, runtime_certificate)
    module_name, function_name = normalized["runner_entrypoint"].split(":", 1)
    adapter = getattr(importlib.import_module(module_name), function_name)
    return adapter(
        mode=mode,
        run_spec=normalized,
        runtime_certificate=runtime_certificate,
    )
