"""No-inference mechanical acceptance for a completed frozen K1 full run."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from molgap.pcqm_k1_full_runner import (
    ARCHITECTURE_FILE_SHA256,
    ARCHITECTURE_SOURCE_COMMIT,
    EXPECTED_INITIAL_MODEL_SHA256,
    EXPECTED_PARAMETERS,
    MAX_OPTIMIZER_STEPS,
    MIN_LEARNING_RATE,
    PHYSICAL_BATCH,
    SAMPLE_PRESENTATIONS,
    TRAINING_CONTRACT,
    TRAINING_CONTRACT_SHA256,
)
from molgap.training_reproducibility import (
    CHECKPOINT_FORMAT,
    MODEL_BUNDLE_FORMAT,
    assert_finite_state_dict,
    canonical_fingerprint,
    sha256_file,
)


def _require_finite_numbers(value, *, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_numbers(item, label=f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite numeric evidence: {label}")


def accept(root: Path) -> dict:
    import torch

    from molgap.qm9_neural_atom import make_encoder

    root = root.resolve()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads((root / "training_contract.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "runtime_manifest.json").read_text(encoding="utf-8"))
    certificate = json.loads(
        (root / "runtime_certificate.json").read_text(encoding="utf-8")
    )
    for payload_name, payload in (("summary", summary), ("completion", completion)):
        for key, expected in {
            "format": "molgap-pcqm-k1-full-training-result-v1",
            "status": "complete",
            "complete": True,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "global_step": MAX_OPTIMIZER_STEPS,
            "sample_presentations": SAMPLE_PRESENTATIONS,
            "resume_required": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        }.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"{payload_name} contract changed: {key}")
    if contract != TRAINING_CONTRACT or canonical_fingerprint(contract) != TRAINING_CONTRACT_SHA256:
        raise RuntimeError("Frozen training contract changed")
    if runtime.get("runtime_fingerprint") != canonical_fingerprint(
        {key: value for key, value in runtime.items() if key != "runtime_fingerprint"}
    ):
        raise RuntimeError("Runtime manifest fingerprint changed")
    certificate_id = canonical_fingerprint(certificate)
    if certificate_id != summary.get("runtime_certificate_id"):
        raise RuntimeError("Runtime certificate identity changed")
    for key, expected in {
        "status": "accepted",
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "tail_batch_policy": "drop_last",
        "calibration_checks_passed": True,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
    }.items():
        if certificate.get(key) != expected:
            raise RuntimeError(f"Runtime certificate changed: {key}")

    checkpoint_path = root / "last_checkpoint.pt"
    bundle_path = root / "model_bundle.pt"
    if sha256_file(checkpoint_path) != summary.get("checkpoint_sha256"):
        raise RuntimeError("Final checkpoint SHA changed")
    if sha256_file(bundle_path) != summary.get("model_bundle_sha256"):
        raise RuntimeError("Model bundle SHA changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    for key, expected in {
        "format": CHECKPOINT_FORMAT,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "global_step": MAX_OPTIMIZER_STEPS,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"Final checkpoint changed: {key}")
    if checkpoint.get("scheduler", {}).get("last_epoch") != MAX_OPTIMIZER_STEPS:
        raise RuntimeError("Scheduler step count changed")
    optimizer_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
    if len(optimizer_groups) != 1 or not math.isclose(
        float(optimizer_groups[0]["lr"]), MIN_LEARNING_RATE, abs_tol=1e-12
    ):
        raise RuntimeError("Final optimizer learning rate changed")
    assert_finite_state_dict(checkpoint["model"], label="final checkpoint model")
    for state in checkpoint.get("optimizer", {}).get("state", {}).values():
        assert_finite_state_dict(state, label="final optimizer state")

    if bundle.get("format") != MODEL_BUNDLE_FORMAT:
        raise RuntimeError("Model bundle format changed")
    for key, expected in {
        "model_id": "neural_atom_k1",
        "training_contract": TRAINING_CONTRACT,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "target_stats": summary["target_stats"],
        "source_commit": summary["source_commit"],
        "source_archive_sha256": summary["source_archive_sha256"],
        "runtime_fingerprint": summary["runtime_fingerprint"],
        "runtime_certificate_id": certificate_id,
        "completed_optimizer_steps": MAX_OPTIMIZER_STEPS,
    }.items():
        if bundle.get(key) != expected:
            raise RuntimeError(f"Model bundle changed: {key}")
    if not math.isclose(
        float(bundle.get("final_learning_rate")), MIN_LEARNING_RATE, abs_tol=1e-12
    ):
        raise RuntimeError("Bundle final learning rate changed")
    assert_finite_state_dict(bundle["state_dict"], label="model bundle")
    model = make_encoder("neural_atom_k1")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Frozen K1 parameter count changed")
    model.load_state_dict(bundle["state_dict"], strict=True)
    expected_architecture = {
        "factory": "molgap.qm9_neural_atom.make_encoder",
        "factory_argument": "neural_atom_k1",
        "architecture_source_commit": ARCHITECTURE_SOURCE_COMMIT,
        "architecture_file_sha256": ARCHITECTURE_FILE_SHA256,
        "parameter_count": EXPECTED_PARAMETERS,
        "seed42_initial_model_sha256": EXPECTED_INITIAL_MODEL_SHA256,
    }
    if bundle.get("architecture") != expected_architecture:
        raise RuntimeError("Frozen K1 architecture identity changed")

    trace = checkpoint.get("trace", [])
    if not trace or trace[-1].get("global_step") != MAX_OPTIMIZER_STEPS:
        raise RuntimeError("Training trace is incomplete")
    steps = [int(row["global_step"]) for row in trace]
    if steps != sorted(set(steps)):
        raise RuntimeError("Training trace steps are not strictly increasing")
    for row in trace:
        if row.get("sample_presentations") != int(row["global_step"]) * PHYSICAL_BATCH:
            raise RuntimeError("Trace sample exposure changed")
    _require_finite_numbers(trace, label="trace")
    _require_finite_numbers(summary, label="summary")

    for relative, expected in completion.get("artifact_sha256", {}).items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"Artifact path escaped result root: {relative}") from error
        if sha256_file(path) != expected:
            raise RuntimeError(f"Artifact changed: {relative}")
    return {
        "format": "molgap-pcqm-k1-full-acceptance-v1",
        "accepted": True,
        "model_inference_executed": False,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "runtime_certificate_id": certificate_id,
        "source_commit": summary["source_commit"],
        "source_archive_sha256": summary["source_archive_sha256"],
        "global_step": MAX_OPTIMIZER_STEPS,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "model_bundle_sha256": summary["model_bundle_sha256"],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = accept(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
