"""CPU-only inspection of the frozen Kaggle3 pair; no training or RML writes."""
from pathlib import Path
import json

import torch

from molgap.constants import REPO_ROOT
from molgap.experiment_family_workflow import (
    TargetIdentityBinding, inspect_terminal_output, prepare_terminal_outputs,
)
from molgap.experiment_launch import (
    build_launch_receipt, canonical_json, reconcile_platform_response,
    write_launch_receipt,
)
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_screen_training import validate_runtime_preflight
from molgap.research_memory.trace import load_canonical_trace
from molgap.router import paired_bootstrap_mean
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_json, sha256_file


ROOT = Path(REPO_ROOT)
QUESTION = ROOT / "experiments/pcqm_k1_t4_cost_quality"
PACKAGE = ROOT / "platforms/_records/kaggle/staging/k1-t4-cost-quality-100k-s42-v1-prepared/package"
RAW = ROOT / "platforms/_records/kaggle/training/k1_t4_cost_quality_100k_s42_v1/terminal_20261010"
OUT = QUESTION / "terminal_acceptance"


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def accept():
    spec = ExperimentSpec.from_json((QUESTION / "experiment_spec.json").read_text())
    declaration = spec.to_dict()
    submission = read(QUESTION / "submission_response.json")
    observation = read(QUESTION / "scheduler_acceptance_20261010.json")
    source = read(QUESTION / "source_acceptance_20261010.json")
    metadata = source["metadata"]["metadata"]
    kernel = submission["kernel"]
    require(observation["kernel"] == kernel and observation["state"]["status"] == "COMPLETE",
            "Exact scheduler completion is required")
    require(metadata["ref"] == kernel and metadata["id"] == submission["kernel_id"]
            and metadata["currentVersionNumber"] == submission["version_number"] == 1,
            "Remote physical kernel/version changed")
    require(source["raw_source_matches_frozen"] is True, "Remote entry source differs")
    identity = submission["release_binding"]["package_identity"]
    binding = build_launch_receipt(spec, PACKAGE, expected_package_identity=identity)["binding"]
    require(binding["source_archive_sha256"] == submission["release_binding"]["source_archive_sha256"],
            "Submitted package differs")
    fact = lambda value: {"value": value, "missing_reason": None}
    response = {
        "format": "molgap-platform-response", "version": 1, "mode": "observed",
        "outcome": "existing_found", "conflict_kind": None, "binding": binding,
        "canonical_platform_reference": fact(kernel), "platform_version": fact("1"),
        "physical_runs": fact([{
            "run_identity": str(metadata["id"]) + ":v1", "canonical_reference": kernel,
            "platform_version": fact("1"), "arm_ids": [a["arm_id"] for a in declaration["arms"]],
        }]),
        "timestamp": fact(observation["observed_at"]),
        "monitor_paths": {"value": None, "missing_reason": "not_reported"},
    }
    receipt = reconcile_platform_response(spec, PACKAGE, canonical_json(response),
                                          expected_package_identity=identity)
    receipt_dir = OUT / "launch"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = write_launch_receipt(canonical_json(receipt), receipt_dir, spec, PACKAGE,
                                       expected_package_identity=identity)
    plan_path = QUESTION / "family_acceptance_plan.json"
    plan = read(plan_path)
    supplied = {a["arm_id"]: {
        "output_dir": (RAW / "experiment" / a["arm_id"]).relative_to(ROOT).as_posix(),
        "expected": a["expected"],
    } for a in plan["arms"]}
    outputs = prepare_terminal_outputs(spec, ROOT, supplied, receipt_path=receipt_path,
        package_dir=PACKAGE, expected_package_identity=identity)
    target_binding = TargetIdentityBinding.from_acceptance_plan(spec, ROOT,
        plan_path.relative_to(ROOT).as_posix(), plan_sha256=sha256_file(plan_path))
    mechanical, predictions, traces, qualified, anomalies = {}, {}, {}, {}, []
    for arm_id, item in outputs.items():
        item["target_identity"] = target_binding
        report = inspect_terminal_output(item)
        mechanical[arm_id] = report
        require(report["status"] == "MECHANICALLY_VERIFIED", str(report["blockers"]))
        directory = Path(item["output_dir"])
        provenance = read(directory / "runtime_provenance.json")
        certificate = validate_runtime_preflight(directory, provenance)
        manifest = read(directory / "output_manifest.json")
        recipe = read(directory / "training_contract.json")
        require(manifest["runtime"]["runtime_certificate_id"] == canonical_fingerprint(certificate),
                "Output does not bind the retained runtime certificate")
        require(provenance["context"] == manifest["context"]
                and provenance["row_order_fingerprint"] == recipe["row_order_fingerprint"]
                and provenance["initialization_sha256"] == recipe["initialization_sha256"],
                "Observed preflight/training identity differs")
        qualified[arm_id] = certificate
        predictions[arm_id] = torch.load(directory / "development_predictions.pt",
                                        map_location="cpu", weights_only=True)
        traces[arm_id] = load_canonical_trace(directory / "canonical_trace.json")
        prospective = read(QUESTION / "kaggle3_v1" / arm_id / "trajectory.json")
        action = prospective["actions"][0]
        if traces[arm_id]["run_id"] not in action["run_ids"]:
            anomalies.append({"arm_id": arm_id, "field": "run_id",
                "planned": action["run_ids"], "observed": traces[arm_id]["run_id"]})
        if traces[arm_id]["trajectory_id"] != prospective["trajectory_id"]:
            anomalies.append({"arm_id": arm_id, "field": "trajectory_id",
                "planned": prospective["trajectory_id"], "observed": traces[arm_id]["trajectory_id"]})
        if action["source_commit"] != item["context"].source_commit:
            anomalies.append({"arm_id": arm_id, "field": "source_commit",
                "planned": action["source_commit"], "observed": item["context"].source_commit})
        if arm_id + "-v1" not in action["attempt_ids"]:
            anomalies.append({"arm_id": arm_id, "field": "descriptor_attempt_id",
                "planned": action["attempt_ids"], "shared_descriptor_value": arm_id + "-v1"})
    reference, candidate = predictions["mean2"], predictions["single"]
    for key in ("source_idx", "target_eV"):
        require(torch.equal(reference[key], candidate[key]), "Pair alignment differs: " + key)
    errors = {arm: (data["prediction_eV"].double() - data["target_eV"].double()).abs().numpy()
              for arm, data in predictions.items()}
    bootstrap = paired_bootstrap_mean(errors["single"] - errors["mean2"], n_bootstrap=1000, seed=42)
    quality_pass = bootstrap["ci95"][1] <= 0.001
    mean_preflight = read(RAW / "experiment/mean2/architecture_preflight.json")
    single_preflight = read(RAW / "experiment/single/architecture_preflight.json")
    require(mean_preflight["accepted"] and single_preflight["accepted"], "Qualification failed")
    same_device_saving = 1 - mean_preflight["timings"]["reference"]["median_step_seconds"] / mean_preflight["timings"]["mean2"]["median_step_seconds"]
    cross_device_saving = 1 - single_preflight["timings"]["reference"]["median_step_seconds"] / mean_preflight["timings"]["mean2"]["median_step_seconds"]
    trace_summary = {}
    for arm_id, trace in traces.items():
        rows = [r for r in trace["observations"] if r["event"] == "observation"]
        best = min(rows, key=lambda r: r["live_dev_metric"])
        trace_summary[arm_id] = {"selected_epoch": mechanical[arm_id]["observed"]["selected_epoch"],
            "best_dev_eV": best["live_dev_metric"], "final_dev_eV": rows[-1]["live_dev_metric"],
            "final_online_train_eV": rows[-1]["live_train_metric"],
            "train_metric_semantics": trace["metric_semantics"]["live_train_metric"],
            "cumulative_device_time_available": trace["device_time_semantics"] is not None}
    pair_state = read(RAW / "experiment/pair_state.json")
    require(pair_state["status"] == "complete" and pair_state["spec_identity"] == spec.identity
            and pair_state["hardware"] == ["Tesla T4", "Tesla T4"], "Pair orchestration differs")
    allocations = {a: read(RAW / "experiment" / a / "allocation_cost.json") for a in outputs}
    # Kaggle SDK writes log text with the Windows account's CP936 encoding.
    log = json.loads((RAW / (kernel.split("/")[1] + ".log")).read_bytes().decode("cp936"))
    cost_review = {
        "matched_step_saving_pass": same_device_saving >= 0.25,
        "training_invocation_device_hours_lower_bound": {a: d["costs"][1]["value"] / 3600
                                                        for a, d in allocations.items()},
        "idle_inclusive_pair_window_T4_device_hours": 2 * pair_state["pair_process_wall_seconds"] / 3600,
        "pair_window_scope": pair_state["scope"],
        "bootstrap_seconds_reconstructed_from_frozen_clock":
            14400 - 60 - pair_state["maximum_wall_seconds"],
        "last_logger_event_seconds": max(row["time"] for row in log),
        "retained_log_encoding": "cp936 (SDK Windows text export; not raw API bytes)",
        "whole_allocation_T4_device_hours": {"status": "measurement_missing", "value": None},
        "allocation_budget_pass": None,
        "reason": "Logger and bootstrap clocks do not observe final platform allocation release; CPU/queue also unknown.",
    }
    result = {
        "format": "molgap-k1-cost-quality-retained-acceptance-v1",
        "physical_kernel": kernel, "kernel_id": metadata["id"], "version": 1,
        "spec_identity": spec.identity, "source_commit": binding["source_commit"],
        "source_archive_sha256": binding["source_archive_sha256"],
        "runtime_qualification": qualified, "native_cost_review": cost_review,
        "mechanical": mechanical, "development_rows": len(errors["mean2"]),
        "mae_eV": {arm: float(error.mean()) for arm, error in errors.items()},
        "single_minus_mean2_eV": bootstrap,
        "quality_noninferiority_pass": quality_pass,
        "material_precision_pass": -bootstrap["delta"] >= 0.003 and bootstrap["ci95"][0] > 0,
        "same_device_preflight_step_saving_fraction": same_device_saving,
        "cross_device_preflight_step_saving_fraction": cross_device_saving,
        "scientific_disposition": "NEGATIVE_UNDER_CONTRACT" if not quality_pass else "PENDING_COST_REVIEW",
        "rml_closure": {"status": "BLOCKED", "anomalies": anomalies,
            "strict_comparison_readiness": "NOT_ESTABLISHED", "finalization_executed": False,
            "reason": "Frozen prospective identities do not bind producer trace/package or translated attempt; originals preserved."},
        "trace_summary": trace_summary,
        "limitations": ["One seed; row bootstrap is not training stochasticity.",
            "50K development has been repeatedly consumed for selection.",
            "Preflight timing is a fixed-fixture measurement, not isolated formal-training step time.",
            "Full entry allocation including bootstrap/idle/cleanup is not measured by per-arm cost files.",
            "No500K/full release or production adoption."],
        "training_executed_locally": False, "inference_executed_locally": False,
        "bindings": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in RAW.rglob("*") if p.is_file()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "result.json", result)
    return result


if __name__ == "__main__":
    result = accept()
    print(json.dumps({k: result[k] for k in ("mae_eV", "single_minus_mean2_eV",
        "quality_noninferiority_pass", "same_device_preflight_step_saving_fraction",
        "scientific_disposition", "rml_closure")}, indent=2))
