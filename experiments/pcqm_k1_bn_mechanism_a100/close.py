"""Accept one retained K1 BN mechanism attempt and finalize its RML record."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.finalize import finalize
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = ROOT / "experiments/pcqm_k1_bn_mechanism_a100"
REL = HERE.relative_to(ROOT).as_posix()
RUN = "k1-bn-mechanism-a100-20261008"
TID = "TB-k1-bn-mechanism-a100-20261008"
EID = "pcqm-k1-bn-mechanism-a100-20261008"
ATTEMPT = "attempt-001"
CASES = {
    "dropout_off_pass1": (False, 1),
    "dropout_off_pass2": (False, 2),
    "dropout_on_pass1": (True, 1),
    "dropout_on_pass2": (True, 2),
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _row_hash(rows) -> str:
    return hashlib.sha256(np.asarray(list(rows), dtype="<i8").tobytes()).hexdigest()


def _finite_number(value, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"Nonfinite {label}")
    return result


def _manifest_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    _require(not posix.is_absolute() and ".." not in posix.parts,
             f"Unsafe manifest path: {relative}")
    path = (root / Path(*posix.parts)).resolve()
    _require(path.is_relative_to(root.resolve()), f"Manifest path escaped root: {relative}")
    return path


def _load_prediction(path: Path, expected_rows: int, expected_ids: torch.Tensor,
                     expected_targets: torch.Tensor | None = None):
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, dict), f"Prediction artifact is not a mapping: {path.name}")
    required = {"source_idx", "target_eV", "prediction_eV"}
    _require(required <= value.keys(), f"Prediction fields missing: {path.name}")
    source_idx = value["source_idx"].long().reshape(-1)
    target = value["target_eV"].reshape(-1)
    prediction = value["prediction_eV"].reshape(-1)
    _require(len(source_idx) == expected_rows and len(target) == expected_rows and
             len(prediction) == expected_rows, f"Prediction row count differs: {path.name}")
    _require(torch.equal(source_idx, expected_ids), f"Prediction source IDs differ: {path.name}")
    _require(torch.isfinite(target).all() and torch.isfinite(prediction).all(),
             f"Nonfinite prediction/target: {path.name}")
    if expected_targets is not None:
        _require(torch.equal(target, expected_targets), f"Prediction targets differ: {path.name}")
    return value, target, prediction


def _cost_event(event_id: str, category: str, hardware: str, *,
                wall_seconds=None, device_seconds=None, cpu_seconds=None,
                device_status: str | None = None, platform: str = "colab"):
    def measure(seconds, status):
        return {"value": seconds / 3600 if seconds is not None else None,
                "status": status}

    return {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": event_id,
        "trajectory_id": TID,
        "action_id": "A001",
        "run_id": RUN,
        "attempt_id": ATTEMPT,
        "platform": platform,
        "hardware": hardware,
        "category": category,
        "evidence_ref": f"{REL}/acceptance.json",
        "measurement": {
            "wall_hours": measure(wall_seconds,
                "measured" if wall_seconds is not None else "measurement_missing"),
            "device_hours": measure(device_seconds, device_status or (
                "measured" if device_seconds is not None else "measurement_missing")),
            "cpu_hours": measure(cpu_seconds,
                "measured" if cpu_seconds is not None else "measurement_missing"),
            "queue_hours": measure(None, "measurement_missing"),
        },
    }


def main():
    # The owning decision and attribution are written from the accepted result;
    # close.py only binds their bytes and performs mechanical acceptance.
    terminal_decision_path = HERE / "terminal_decision.md"
    attribution_path = HERE / "attribution.md"
    _require(terminal_decision_path.is_file() and attribution_path.is_file(),
             "Parent-written terminal_decision.md and attribution.md are required")

    result_dir = HERE / "results" / ATTEMPT
    _require(result_dir.is_dir(), f"Copied Drive results are missing: {result_dir}")
    inputs = _read(HERE / "inputs.json")
    local_manifest_path = HERE / "payload_manifest.json"
    remote_manifest_path = result_dir / "payload_manifest.json"
    staged_payload = (ROOT / "platforms/_records/colab/staging" / RUN / "payload").resolve()
    local_manifest_bytes = local_manifest_path.read_bytes()
    remote_manifest_bytes = remote_manifest_path.read_bytes()
    staged_manifest_bytes = (staged_payload / "payload_manifest.json").read_bytes()
    _require(local_manifest_bytes == remote_manifest_bytes == staged_manifest_bytes,
             "Local, staged and returned payload manifests are not byte-identical")
    payload_sha256 = hashlib.sha256(local_manifest_bytes).hexdigest()
    manifest = json.loads(local_manifest_bytes)
    _require(manifest.get("format") == "molgap-k1-bn-mechanism-payload-v1",
             "Unexpected frozen payload manifest format")

    # Verify every staged payload file, then bind executable source and RML
    # prospective records to the copies retained in this experiment.
    payload_files = manifest.get("files")
    _require(isinstance(payload_files, dict) and payload_files,
             "Payload file hashes are missing")
    for relative, digest in payload_files.items():
        staged_file = _manifest_path(staged_payload, relative)
        _require(staged_file.is_file() and sha256_file(staged_file) == digest,
                 f"Staged payload bytes differ: {relative}")
        if relative.startswith("src/molgap/"):
            retained_source = HERE / "frozen_source" / relative
            _require(retained_source.is_file() and sha256_file(retained_source) == digest,
                     f"Executed source differs from the frozen payload: {relative}")
            if relative not in inputs["source_files"]:
                _require(sha256_file(ROOT / relative) == digest,
                         f"Shared diagnostic helper changed: {relative}")
        elif relative.startswith("prospective/"):
            canonical = _manifest_path(HERE / "rml", relative.removeprefix("prospective/"))
            _require(canonical.is_file() and sha256_file(canonical) == digest,
                     f"Prospective RML binding differs: {relative}")
        elif relative == "protocol.md":
            _require(sha256_file(HERE / "protocol.md") == digest,
                     "Frozen protocol bytes differ")

    source_files = inputs.get("source_files")
    _require(isinstance(source_files, dict) and source_files,
             "Input source hashes are missing")
    for relative, digest in source_files.items():
        _require(payload_files.get(relative) == digest and
                 sha256_file(HERE / "frozen_source" / relative) == digest,
                 f"Input source binding differs: {relative}")
    trajectory = _read(HERE / "rml/trajectory.json")
    _require(trajectory.get("record_mode") == "prospective" and
             trajectory.get("trajectory_id") == TID and
             trajectory.get("decision", {}).get("outcome") == "ACTIVE",
             "The frozen active prospective trajectory differs")
    source_commit = inputs.get("source_commit")
    _require(source_commit and trajectory.get("state_at_start", {}).get("source_commit") == source_commit and
             trajectory.get("actions", [{}])[0].get("source_commit") == source_commit,
             "Source commit is not bound to the prospective trajectory")

    expected_checkpoint = inputs.get("checkpoint", {}).get("sha256")
    expected_predictions = inputs.get("predictions", {}).get("sha256")
    _require(expected_checkpoint and manifest.get("checkpoint", {}).get("sha256") == expected_checkpoint and
             payload_files.get("selected.pt") == expected_checkpoint,
             "Selected checkpoint SHA256 differs from inputs")
    _require(expected_predictions and payload_files.get("original_prediction.pt") == expected_predictions,
             "Accepted prediction SHA256 differs from inputs")
    if inputs.get("checkpoint", {}).get("source_sha256") is not None:
        _require(manifest["checkpoint"].get("source_sha256") == inputs["checkpoint"]["source_sha256"],
                 "Checkpoint source SHA256 differs from inputs")

    completion = _read(result_dir / "completion.json")
    _require(completion.get("complete") is True and
             completion.get("payload_sha256") == payload_sha256,
             "Completion status/payload identity differs")
    completion_files = completion.get("files")
    _require(isinstance(completion_files, dict) and completion_files,
             "Completion file hashes are missing")
    required_result_files = {
        "payload_manifest.json", "protocol.md", "setup_observation.json", "runtime.json",
        "result.json", "original.pt",
    }
    for name in CASES:
        required_result_files.update({f"{name}.json", f"{name}.pt", f"{name}_bn_buffers.pt"})
    _require(required_result_files <= completion_files.keys(),
             "Completion manifest omits required diagnostic outputs")
    for relative, digest in completion_files.items():
        artifact = _manifest_path(result_dir, relative)
        _require(artifact.is_file() and sha256_file(artifact) == digest,
                 f"Completion artifact hash differs: {relative}")
    # liveworker.log is deliberately excluded by the worker from its completion
    # map. If an outer Drive copy lists it, the generic loop above verifies it.
    _require(completion_files.get("payload_manifest.json") == payload_sha256,
             "Completion does not bind the exact payload manifest")

    runtime = _read(result_dir / "runtime.json")
    _require(runtime.get("payload_sha256") == payload_sha256,
             "Runtime payload SHA256 differs")
    _require(str(runtime.get("torch", "")).split("+")[0] == "2.4.1" and
             runtime.get("cuda") == "12.1" and str(runtime.get("python", "")).startswith("3.11") and
             runtime.get("torch_geometric") == "2.6.1" and "A100" in runtime.get("gpu", ""),
             "Observed Torch/Python/PyG/CUDA/A100 runtime differs from the frozen contract")
    determinism = runtime.get("determinism", {})
    _require(determinism.get("precision") == "fp32" and
             determinism.get("tf32_enabled") is False and
             determinism.get("cudnn_benchmark") is False and
             determinism.get("cudnn_deterministic") is True and
             determinism.get("deterministic_algorithms") is True,
             "Observed precision/determinism settings differ")
    _require(runtime.get("scientific_training") is False and
             runtime.get("optimizer_created") is False,
             "Runtime reports training or optimizer construction")

    process = _read(result_dir / "worker_process_observation.json")
    process_wall = _finite_number(process.get("worker_wall_seconds_including_imports"),
                                  "worker process wall seconds")
    _require(process.get("returncode") == 0 and process.get("wall_ceiling_seconds") == 1200 and
             0 < process_wall <= 1200,
             "Worker process did not return zero within its 1200-second bound")

    result = _read(result_dir / "result.json")
    _require(result.get("runtime") == runtime and
             result.get("train_labels_read") is True and
             result.get("train_labels_used_for_objective") is False and
             result.get("optimizer_created") is False and
             result.get("gradients_computed") is False,
             "Result scope/runtime differs from the frozen diagnostic")
    _require(_finite_number(result.get("wall_seconds"), "worker wall seconds") <= 1200,
             "Worker-reported wall time exceeds its frozen bound")

    train_ids = inputs.get("train_source_idx", [])
    dev_ids = inputs.get("development_source_idx", [])
    _require(len(train_ids) == 16384 and len(set(train_ids)) == 16384 and
             all(0 <= int(row) < 500000 for row in train_ids) and
             manifest.get("train_source_idx") == train_ids and
             manifest.get("sample_source_idx") == train_ids,
             "Frozen training feature rows differ")
    expected_dev_ids = list(range(500000, 550000))
    _require(dev_ids == expected_dev_ids and manifest.get("development_source_idx") == expected_dev_ids,
             "Expected the exact ordered 50K development cohort")
    expected_ids = torch.tensor(expected_dev_ids, dtype=torch.long)

    accepted = torch.load(staged_payload / "original_prediction.pt",
                          map_location="cpu", weights_only=True)
    _require(isinstance(accepted, dict) and {"source_idx", "target", "prediction"} <= accepted.keys(),
             "Accepted prediction payload fields are missing")
    accepted_ids = accepted["source_idx"].long().reshape(-1)
    accepted_targets = accepted["target"].reshape(-1)
    accepted_predictions = accepted["prediction"].reshape(-1)
    _require(len(accepted_ids) == 50000 and torch.equal(accepted_ids, expected_ids) and
             len(accepted_targets) == 50000 and len(accepted_predictions) == 50000 and
             torch.isfinite(accepted_targets).all() and torch.isfinite(accepted_predictions).all(),
             "Accepted prediction cohort is not finite/aligned on the exact 50K rows")

    original, targets, original_prediction = _load_prediction(
        result_dir / "original.pt", 50000, expected_ids)
    _require(torch.equal(targets, accepted_targets),
             "Reconstructed and accepted prediction targets differ")
    reconstruction_error = float((original_prediction.double() - accepted_predictions.double()).abs().max())
    _require(math.isfinite(reconstruction_error) and reconstruction_error <= 1e-4 and
             _finite_number(result.get("reconstruction_max_abs_eV"), "reported reconstruction error") <= 1e-4,
             "Selected-model reconstruction exceeds the frozen 1e-4 eV tolerance")

    predictions = {"original": original_prediction.detach().cpu().numpy().astype(np.float64, copy=False)}
    target_array = targets.detach().cpu().numpy().astype(np.float64, copy=False)
    errors = {"original": np.abs(predictions["original"] - target_array)}
    details_by_case = {}
    case_rows = result.get("cases", [])
    _require(isinstance(case_rows, list) and len(case_rows) == len(CASES),
             "Result does not contain exactly four frozen cases")
    seen_cases = set()
    for row in case_rows:
        name = row.get("case")
        _require(name in CASES and name not in seen_cases, f"Unexpected/duplicate case: {name}")
        seen_cases.add(name)
        dropout, passes = CASES[name]
        detail = _read(result_dir / f"{name}.json")
        _require(detail == row, f"Per-case summary differs from result.json: {name}")
        calibration = row.get("calibration", {})
        expected_batches = 128 * passes
        _require(row.get("dropout_enabled") is dropout and row.get("passes_per_batch") == passes and
                 calibration.get("rows") == 16384 and calibration.get("batches") == expected_batches and
                 calibration.get("passes_per_batch") == passes and
                 calibration.get("dropout_disabled") is (not dropout) and
                 calibration.get("labels_used_for_calibration") is False and
                 calibration.get("parameters_unchanged") is True and
                 calibration.get("non_bn_buffers_unchanged") is True and
                 calibration.get("buffers_restored") is True and
                 calibration.get("buffer_sha256_before") == result.get("original_buffer_sha256") and
                 row.get("parameters_unchanged") is True and
                 row.get("original_buffers_restored") is True and
                 row.get("original_first128_prediction_restored_exactly") is True,
                 f"Frozen-state/update checks failed: {name}")
        counters = calibration.get("num_batches_tracked", {})
        _require(isinstance(counters, dict) and counters and
                 all(int(value) == expected_batches for value in counters.values()),
                 f"BatchNorm counter updates differ: {name}")

        buffers = torch.load(result_dir / f"{name}_bn_buffers.pt",
                             map_location="cpu", weights_only=True)
        _require(isinstance(buffers, dict) and buffers and all(
            key.endswith(("running_mean", "running_var", "num_batches_tracked"))
            for key in buffers), f"Unexpected BatchNorm snapshot: {name}")
        _require(all(torch.isfinite(value).all() for key, value in buffers.items()
                     if key.endswith(("running_mean", "running_var"))),
                 f"Nonfinite BatchNorm snapshot: {name}")
        for key, value in buffers.items():
            if key.endswith("num_batches_tracked"):
                _require(int(value) == expected_batches,
                         f"Saved BatchNorm update count differs: {name}/{key}")

        _, case_targets, case_prediction = _load_prediction(
            result_dir / f"{name}.pt", 50000, expected_ids, targets)
        _require(torch.equal(case_targets, targets), f"Case targets are not exact: {name}")
        predictions[name] = case_prediction.detach().cpu().numpy().astype(np.float64, copy=False)
        errors[name] = np.abs(predictions[name] - target_array)
        details_by_case[name] = row
    _require(seen_cases == set(CASES), "One or more frozen cases are missing")

    case_metrics = {
        "original": {"mae_eV": float(np.mean(errors["original"], dtype=np.float64)),
                     "rows": 50000},
    }
    for name in CASES:
        case_metrics[name] = {
            "mae_eV": float(np.mean(errors[name], dtype=np.float64)),
            "original_gain_eV": float(np.mean(errors["original"] - errors[name], dtype=np.float64)),
            "rows": 50000,
            "dropout_enabled": CASES[name][0],
            "passes_per_batch": CASES[name][1],
            "bn_updates_per_module": 128 * CASES[name][1],
            "original_gain_paired_bootstrap": paired_bootstrap_mean(
                errors["original"] - errors[name], n_bootstrap=1000, seed=20261008),
            "original_gain_probability_semantics": "P(mean original error minus case error < 0); smaller means stronger gain evidence",
        }

    contrast_specs = {
        "dropout_pass1_on_error_minus_off": ("dropout_on_pass1", "dropout_off_pass1"),
        "dropout_pass2_on_error_minus_off": ("dropout_on_pass2", "dropout_off_pass2"),
        "pass_count_off_pass2_error_minus_pass1": ("dropout_off_pass2", "dropout_off_pass1"),
        "pass_count_on_pass2_error_minus_pass1": ("dropout_on_pass2", "dropout_on_pass1"),
    }
    contrasts = {}
    for label, (left, right) in contrast_specs.items():
        delta = errors[left] - errors[right]
        bootstrap = paired_bootstrap_mean(delta, n_bootstrap=1000, seed=20261008)
        contrasts[label] = {
            "delta_definition": f"{left} absolute error minus {right} absolute error, eV",
            "mean_error_difference_eV": float(np.mean(delta, dtype=np.float64)),
            "paired_bootstrap": bootstrap,
            "probability_better_semantics": "P(mean error difference < 0) under paired row bootstrap",
            "multiplicity_correction": "none; four predeclared exploratory contrasts",
        }

    source_idx_sha256 = _row_hash(expected_dev_ids)
    analysis = {
        "format": "molgap-k1-bn-mechanism-analysis-v1",
        "trajectory_id": TID,
        "run_id": RUN,
        "attempt_id": ATTEMPT,
        "payload_manifest_sha256": payload_sha256,
        "source_commit": source_commit,
        "cohort": {"dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
                   "development_rows": 50000,
                   "source_idx_sha256": source_idx_sha256,
                   "target_sha256_float64_le": hashlib.sha256(
                       target_array.astype("<f8", copy=False).tobytes()).hexdigest()},
        "metric_precision": "float64 absolute errors and means",
        "cases": case_metrics,
        "predeclared_contrasts": contrasts,
        "diagnostic_scope": "Frozen-parameter BN buffer interventions; no optimizer, gradients, training objective, selection, promotion or replay claim.",
        "reconstruction_max_abs_eV": reconstruction_error,
        "bootstrap": {"helper": "molgap.router.paired_bootstrap_mean",
                      "draws": 1000, "seed": 20261008, "interval": "95% row-bootstrap CI",
                      "interpretation_limit": "Paired row uncertainty is not training-seed variance."},
    }
    atomic_json(HERE / "analysis.json", analysis)

    outcome = {
        "execution_status": "complete_frozen_bn_mechanism_diagnostic",
        "artifact_status": "local_hash_verified",
        "comparison_status": "paired_consumed_internal_development_diagnostic",
        "scientific_status": "NO_TRAIN",
        "transfer_status": "not_evaluated",
        "budget_decision": "bounded_a100_diagnostic_complete",
        "full_handoff_status": "not_applicable",
    }
    decision = {
        "outcome": "NO_TRAIN",
        "decision_ref": f"{REL}/terminal_decision.md",
        "next_allowed_actions": [],
        "reopen_conditions": ["A separate prospective qualification is independently authorized"],
    }
    role_use = {
        "train_decoded": "consumed",
        "train_features": "consumed",
        "internal_development": "consumed",
        "official_validation": "untouched",
        "test_dev": "untouched",
        "test_challenge": "untouched",
        "common": "untouched",
        "ood": "untouched",
    }
    roles = []

    def add_role(role_name: str, access_kind: str, row_hash: str, *, selection_used: bool):
        roles.append({
            "schema": "molgap-role-event-v1",
            "role_event_id": f"role-k1-bn-mechanism-{role_name}-{access_kind}",
            "trajectory_id": TID,
            "action_id": "A001",
            "run_id": RUN,
            "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
            "row_manifest_hash": row_hash,
            "role_name": role_name,
            "access_kind": access_kind,
            "selection_used": selection_used,
            "evidence_ref": f"{REL}/acceptance.json",
        })

    train_full_hash = _row_hash(range(500000))
    train_sample_hash = _row_hash(train_ids)
    dev_hash = source_idx_sha256
    add_role("train_decoded", "labels_read", train_full_hash, selection_used=False)
    add_role("train_features", "prediction_input", train_sample_hash, selection_used=False)
    add_role("train_features", "labels_read", train_sample_hash, selection_used=False)
    for access in ("labels_read", "prediction_input", "metric_computed", "selection_used"):
        add_role("internal_development", access, dev_hash, selection_used=True)

    setup = _read(result_dir / "setup_observation.json")
    setup_wall = _finite_number(setup.get("setup_wall_seconds"), "setup wall seconds")
    _require(setup_wall >= 0, "Invalid setup wall time")
    worker_process_cpu = _finite_number(result.get("process_cpu_seconds"),
                                        "worker process CPU seconds")
    costs = [
        _cost_event("cost-k1-bn-mechanism-worker-attempt001", "inference", runtime["gpu"],
                    wall_seconds=process_wall, device_seconds=process_wall,
                    cpu_seconds=worker_process_cpu),
        _cost_event("cost-k1-bn-mechanism-setup-attempt001", "preflight",
                    f"Colab A100 allocation unmeasured ({runtime['gpu']})",
                    wall_seconds=setup_wall, device_seconds=None,
                    device_status="measurement_missing"),
        _cost_event("cost-k1-bn-mechanism-idle-attempt001", "other", runtime["gpu"]),
        _cost_event("cost-k1-bn-mechanism-local-staging", "cache_build",
                    "CPU; no accelerator", platform="local-windows",
                    device_seconds=None, device_status="not_applicable"),
    ]
    checks = {
        "payload_manifest_exact_bytes": True,
        "payload_runtime_sha256": True,
        "source_prospective_binding": True,
        "checkpoint_sha256": True,
        "completion_artifact_hashes": True,
        "worker_process_returncode_and_wall": True,
        "native_a100_torch241_fp32_no_tf32": True,
        "all_four_cases_and_update_counts": True,
        "frozen_parameters_and_non_bn_buffers": True,
        "original_buffers_restored_and_first128_exact": True,
        "aligned_finite_50k_predictions_and_targets": True,
        "protected_roles_untouched": True,
    }
    acceptance = {
        "format": "molgap-k1-bn-mechanism-a100-acceptance-v1",
        "evidence_id": EID,
        "run_id": RUN,
        "outcome": outcome,
        "trajectory_decision": decision,
        "role_use": role_use,
        "roles": roles,
        "costs": costs,
        "checks": checks,
        "analysis_ref": f"{REL}/analysis.json",
        "runtime_release": {
            "status": _read(HERE / "download_receipt.json")["runtime_release"],
            "evidence_ref": f"{REL}/download_receipt.json",
            "independent_browser_verification": False,
        },
        "comparison_class": "CONTEXT_ONLY",
        "metrics": analysis["cases"],
        "predeclared_contrasts_ref": f"{REL}/analysis.json#predeclared_contrasts",
        "role_scope": "Staging decoded backing labels for all500K training and50K development rows. The diagnostic read 16,384 fixed training graphs as feature input; training labels were not an objective. It computed metrics on the historically selection-used 50K development cohort. Official validation, test, common and OOD roles were untouched.",
        "cost_scope": "Measured worker process wall is treated as allocated A100 wall time, not GPU busy time. The process CPU value is the worker's own process time; child CPU is unknown. Setup wall is measured separately while setup allocation is unknown. Idle allocation and local staging time are unknown.",
        "execution_scope": "Four frozen BN-buffer interventions and aligned 50K inference only; no optimizer, gradients, training objective, training trace, replay-readiness claim, model promotion or full handoff.",
        "limitations": ["Development rows were historically selection-used.",
                       "Paired row bootstrap does not estimate training-seed variance.",
                       "This diagnostic does not establish training-time causality or transfer."],
    }
    atomic_json(HERE / "acceptance.json", acceptance)

    policy_path = ROOT / "research_memory/policies/pcqm-k1-bn-mechanism-a100.1.json"
    _require(policy_path.is_file(), "Frozen RML policy snapshot is missing")
    evidence_paths = [path for path in HERE.rglob("*") if path.is_file()
                      and "rml" not in path.relative_to(HERE).parts
                      and "rml_finalized" not in path.parts
                      and "__pycache__" not in path.parts
                      and path.name != "terminal.json"]
    evidence_paths.extend(HERE / "frozen_source" / relative for relative in payload_files
                          if relative.startswith("src/molgap/"))
    evidence_paths.append(policy_path)
    unique_paths = {path.resolve() for path in evidence_paths}
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path)
              for path in sorted(unique_paths, key=lambda item: item.as_posix())}
    retained = sorted(name for name in hashes if name.startswith(f"{REL}/"))
    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EID,
        "track": "B",
        "scope": "desktop_k1_a100_bn_mechanism_diagnostic",
        "legacy_contract": "pcqm-k1-bn-mechanism-a100-v1",
        "outcome": outcome,
        "authority": {"pointers": [f"{REL}/{name}" for name in (
            "protocol.md", "terminal_decision.md", "attribution.md",
            "acceptance.json", "analysis.json")]},
        "role_use": role_use,
        "migration": {
            "migrated_at": datetime.now(timezone.utc).date().isoformat(),
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
            "verification_scope": "Mechanical acceptance of one retained prospective A100 frozen-buffer diagnostic; close.py executes no model.",
        },
        "observed_execution": {
            "training_executed": False,
            "inference_executed": True,
            "training_replay_ready": False,
            "execution_ref": f"{REL}/results/{ATTEMPT}/result.json",
        },
        "artifacts": [{
            "name": name.rsplit("/", 1)[-1],
            "locator": name,
            "sha256": hashes[name],
            "availability": "locally_retained_hash_verified",
        } for name in retained],
    }
    terminal = {
        "format": "molgap-rml-terminal-package-v1",
        "trajectory_id": TID,
        "run_id": RUN,
        "action_id": "A001",
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_ref": f"{REL}/acceptance.json",
        "artifact_hashes": hashes,
        "evidence": evidence,
        "decision": decision,
        "costs": costs,
        "roles": roles,
    }
    atomic_json(HERE / "terminal.json", terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json")))


if __name__ == "__main__":
    main()
