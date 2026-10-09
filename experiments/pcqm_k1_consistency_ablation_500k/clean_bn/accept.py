"""Accept saved paired diagnostics and retain observed-only RML trace recovery."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from molgap.k1_bn_diagnostic import ARMS, SETTINGS, _json, validate_inputs
from molgap.research_memory.recovery import recover_trace
from molgap.research_memory.schemas import validate_trace_manifest
from molgap.research_memory.terminal_wiring import build_default_trace_manifest
from molgap.training_reproducibility import atomic_json, sha256_file, canonical_fingerprint
from molgap.v4_runtime import state_dict_sha256


ARTIFACTS = {"original.pt", "calibrated.pt", "restored.pt", "calibrated_buffers.pt"}


def _bounded(value, ceiling, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(value) or not 0 <= value <= ceiling:
        raise ValueError(f"Invalid {label}")


def validate_report(report, inputs):
    validate_inputs(inputs)
    if report["inputs"] != inputs or inputs["settings"] != SETTINGS:
        raise ValueError("Report/frozen input provenance differs")
    if report["status"] != "complete" or set(report["arms"]) != set(ARMS):
        raise ValueError("Both bounded diagnostic arms must complete")
    _bounded(report["wall_seconds"], SETTINGS["pair_wall_seconds"], "pair wall time")
    indices = np.random.default_rng(SETTINGS["sample_seed"]).choice(
        SETTINGS["train_rows"], SETTINGS["calibration_rows"], replace=False)
    expected_hashes = {
        "calibration_source_idx_sha256": state_dict_sha256({"source_idx": torch.as_tensor(indices, dtype=torch.long)}),
        "development_source_idx_sha256": state_dict_sha256({"source_idx": torch.arange(500000, 550000)}),
    }
    for arm in ARMS:
        observation = report["arms"][arm]
        if observation["arm"] != arm or observation["status"] != "complete" or \
                observation["phase"] != "complete" or observation["worker_exitcode"] != 0:
            raise ValueError("Incomplete/mismatched diagnostic arm")
        _bounded(observation["parent_observed_wall_seconds"], 600, "arm wall time")
        _bounded(observation["wall_seconds"], 600, "worker wall time")
        _bounded(observation["reconstruction_max_abs_eV"], 1e-4, "prediction reconstruction")
        if any(observation[key] != digest for key, digest in expected_hashes.items()):
            raise ValueError("Prescribed calibration/development row hash differs")
        runtime = observation["runtime"]
        if runtime["runtime_fingerprint"] != canonical_fingerprint(
                {key: value for key, value in runtime.items() if key != "runtime_fingerprint"}):
            raise ValueError("Runtime metadata fingerprint differs")
        required = {"device": "cpu", "cuda_device_count_visible": 0, "precision": "fp32",
                    "tf32_enabled": False, "seed": SETTINGS["sample_seed"],
                    "deterministic_algorithms": True, "float32_matmul_precision": "highest"}
        if runtime["accelerator"] is not None or any(
                runtime["determinism"].get(key) != value for key, value in required.items()) or \
                observation["device"] != "cpu" or observation["cpu_threads"] != 4 or \
                observation["autocast_enabled"] is not False:
            raise ValueError("CPU FP32/no-TF32 runtime assertions differ")
        if any(not runtime.get(key) for key in ("python", "python_executable", "platform", "torch",
                                               "installed_distributions", "installed_distributions_sha256")):
            raise ValueError("Missing actual software/runtime metadata")
        calibration = observation["calibration"]
        if calibration["rows"] != 16384 or calibration["batches"] != 128 or \
                calibration["bn_modules"] <= 0 or any(calibration[key] is not True for key in (
                "parameters_unchanged", "non_bn_buffers_unchanged", "dropout_disabled", "buffers_restored")) or \
                calibration["labels_used_for_calibration"] is not False or \
                observation["restored_prediction_exact"] is not True:
            raise ValueError("BN intervention/restoration assertions differ")
        if observation["parameter_sha256_before"] != observation["parameter_sha256_calibrated"] or \
                observation["parameter_sha256_before"] != observation["parameter_sha256_restored"] or \
                observation["buffer_sha256_before"] != observation["buffer_sha256_restored"] or \
                observation["buffer_sha256_before"] != calibration["buffer_sha256_before"]:
            raise ValueError("Original parameters/buffers differ")


def load_artifacts(directory, observation):
    if set(observation["artifacts"]) != ARTIFACTS:
        raise ValueError("Expected exactly four diagnostic artifacts")
    values = {}
    for name, digest in observation["artifacts"].items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or sha256_file(path) != digest:
            raise ValueError("Diagnostic artifact path/hash changed")
        values[name] = torch.load(path, map_location="cpu", weights_only=True)
    buffers = values["calibrated_buffers.pt"]
    if not buffers or any(not torch.is_tensor(value) or not torch.isfinite(value).all()
                          for value in buffers.values()) or state_dict_sha256(buffers) != \
            observation["calibration"]["buffer_sha256_calibrated"]:
        raise ValueError("Saved calibrated buffer state hash differs")
    predictions = {name: values[f"{name}.pt"] for name in ("original", "calibrated", "restored")}
    for record in predictions.values():
        if set(record) != {"source_idx", "target_eV", "prediction_eV"} or any(
                not torch.is_tensor(value) or value.ndim != 1 or len(value) != 50000 or
                not torch.isfinite(value).all() for value in record.values()) or \
                record["source_idx"].dtype != torch.long or not torch.equal(
                record["source_idx"], torch.arange(500000, 550000)):
            raise ValueError("Prediction schema/full development role differs")
    if not torch.equal(predictions["original"]["target_eV"], predictions["calibrated"]["target_eV"]) or any(
            not torch.equal(value, predictions["restored"][name]) for name, value in predictions["original"].items()):
        raise ValueError("Saved targets/full prediction restoration differs")
    return predictions


def validate_exposure(trace):
    rows = trace["observations"]
    if len(rows) != 60 or any(row["epoch_or_pass"] != index or
            row["optimizer_step"] != 3906 * (index + 1) or
            row["sample_presentations"] != 499968 * (index + 1)
            for index, row in enumerate(rows)):
        raise ValueError("Full observed epoch/step/presentation exposure differs")


def add_gate_fields(record, *, primary):
    gain = record["paired_gain_eV"]
    low, high = record["row_bootstrap_95pct_eV"]
    if primary:
        record["original_1mev_point_gate_passed"] = abs(gain) >= 0.001
        record["row_interval_positive"] = low > 0
        record["primary_consistency_gate_passed"] = gain >= 0.001 and low > 0
        record["primary_removal_gate_passed"] = gain <= -0.001 and high < 0
        record["original_material_nomination_gate_passed"] = (
            record["primary_consistency_gate_passed"] or record["primary_removal_gate_passed"])
    else:
        record["secondary_consistency_gate_passed"] = gain >= 0.001 and low > 0
        record["secondary_removal_direction_gate_passed"] = gain <= -0.001 and high < 0
        record["secondary_absolute_delta_eV"] = abs(gain)
        record["primary_promotion_gate_applicable"] = False


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    question = here.parent
    results = args.results.resolve()
    report = _json(results / "pair_report.json")
    validate_report(report, _json(args.inputs))
    helper = root / "experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py"
    spec = importlib.util.spec_from_file_location("retained_pair_analysis", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paired = module.paired_metrics
    predictions, normalization, audit = {}, {}, {}
    artifact_hashes = {}
    for arm in ARMS:
        observation = report["arms"][arm]
        predictions[arm] = load_artifacts(results / arm, observation)
        for filename, digest in observation["artifacts"].items():
            path = results / arm / filename
            if sha256_file(path) != digest:
                raise ValueError("Diagnostic artifact changed")
            artifact_hashes[path.relative_to(root).as_posix()] = digest
        normalization[arm] = paired(predictions[arm]["original"], predictions[arm]["calibrated"], 500000, 550000)
        normalization[arm].pop("material_3mev_point_gain")
        bound = report["inputs"]["arms"][arm]
        trajectory_path = Path(bound["trajectory"]["path"])
        if sha256_file(trajectory_path) != bound["trajectory"]["sha256"]:
            raise ValueError("Original trajectory changed")
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        source = question / "submission_kaggle3_v1/terminal_inspection_20261009/evidence/stages" / arm / "trace.json"
        recovered = recover_trace([source], {
            "trajectory_id": trajectory["trajectory_id"],
            "run_id": trajectory["actions"][0]["run_ids"][0],
            "rows_key": "epochs", "field_mapping": {
                "epoch_or_pass": "epoch", "optimizer_step": "global_step",
                "sample_presentations": "sample_presentations", "learning_rate": "lr",
                "live_train_metric": "train_mae_eV", "live_dev_metric": "live_development_mae_eV",
                "ema_dev_metric": "ema_development_mae_eV", "wall_time_seconds": "seconds",
            },
            "metric_semantics": {
                "live_train_metric": {
                    "metric": "Mean optimization objective times training std; not clean inference MAE",
                    "unit": "eV-equivalent", "target": "gap", "role_identity": "train-prefix500K",
                    "weights": "live", "direction": "minimize"},
                "live_dev_metric": {
                    "metric": "MAE", "unit": "eV", "target": "gap",
                    "role_identity": "consumed-internal-development50K", "weights": "live", "direction": "minimize"},
                "ema_dev_metric": None,
            },
            "device_time_semantics": "Not recorded per observation; null, do not infer from wall time",
        }, results / "rml_audit" / arm / "canonical_trace.json", repo_root=root)
        validate_exposure(recovered)
        manifest = build_default_trace_manifest(root, trajectory,
            {"run_id": recovered["run_id"]}, recovered)
        atomic_json(results / "rml_audit" / arm / "trace_manifest.draft.json", manifest)
        try:
            validate_trace_manifest(manifest)
        except ValueError as exc:
            audit[arm] = {
                "frozen_reference_ids": trajectory["state_at_start"]["reference_ids"],
                "manifest_schema_error": str(exc),
                "trace_retained": True, "terminal_finalized": False,
                "strict_comparison_qualified": False, "replay_ready": False,
            }
        else:
            raise ValueError("Unexpected manifest qualification; review before finalization")
    first, second = ARMS
    for field in ("calibration_source_idx_sha256", "development_source_idx_sha256"):
        if report["arms"][first][field] != report["arms"][second][field]:
            raise ValueError("Paired calibration/development member identity differs")
    comparison = paired(predictions[first]["calibrated"], predictions[second]["calibrated"], 500000, 550000)
    comparison.pop("material_3mev_point_gain")
    original = paired(predictions[first]["original"], predictions[second]["original"], 500000, 550000)
    original.pop("material_3mev_point_gain")
    add_gate_fields(original, primary=True)
    add_gate_fields(comparison, primary=False)
    failed_attempts = []
    for name in ("results_20261009", "results_attempt2_20261009"):
        path = here / name / "pair_report.json"
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["status"] != "failed":
            raise ValueError("Expected retained failed bootstrap attempt")
        failed_attempts.append({"report_ref": path.relative_to(root).as_posix(),
            "report_sha256": sha256_file(path), "wall_seconds": previous["wall_seconds"],
            "worker_process_cpu_seconds": sum(value["process_cpu_seconds"] for value in previous["arms"].values()),
            "errors": [value["error"] for value in previous["arms"].values()]})
    analysis = {
        "format": "molgap-k1-clean-bn-acceptance-v1", "mechanical_status": "passed",
        "original_cpu_pair": original, "both_calibrated_pair": comparison,
        "per_arm_normalization": normalization, "rml_finalization_audit": audit,
        "role_observation": {"per_arm_train_decoded_labels": 500000,
            "per_arm_calibration_features": 16384, "labels_used_for_bn": False,
            "per_arm_consumed_development_rows": 50000, "development_prediction_passes": 3,
            "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"},
        "cost_observation": {"pair_wall_seconds": report["wall_seconds"],
            "worker_process_cpu_seconds": sum(value["process_cpu_seconds"] for value in report["arms"].values()),
            "parent_process_cpu_seconds": None, "device_hours_status": "not_applicable",
            "queue_hours_status": "not_applicable", "unit": "CPU process seconds; separate from wall/T4 device hours",
            "failed_attempts": failed_attempts,
            "all_diagnostic_attempt_wall_seconds": report["wall_seconds"] + sum(item["wall_seconds"] for item in failed_attempts),
            "all_diagnostic_worker_process_cpu_seconds": sum(value["process_cpu_seconds"] for value in report["arms"].values()) +
                sum(item["worker_process_cpu_seconds"] for item in failed_attempts),
            "scope": "Bounded diagnostic invocations only; planning, tests and acceptance-analysis time not measured"},
        "artifact_hashes": artifact_hashes,
        "input_provenance": {"inputs_ref": str(args.inputs.resolve()), "inputs_sha256": sha256_file(args.inputs),
                             "pair_report_sha256": sha256_file(results / "pair_report.json")},
        "analysis_source": {str(Path(__file__).resolve()): sha256_file(Path(__file__)), str(helper): sha256_file(helper)},
        "limits": ["One training seed", "Consumed selection role", "CPU runtime differs from native T4",
            "Secondary diagnostic cannot replace primary raw gate", "Missing frozen reference blocks canonical traced finalization"],
    }
    atomic_json(results / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
