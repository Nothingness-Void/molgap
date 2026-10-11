"""Saved-output acceptance for the explicitly bounded execution assay."""
import json
import math
from pathlib import Path
import tarfile

from .experiment_package import verify_experiment_source_package
from .experiment_spec import ExperimentSpec
from .gptrans_author_acceptance import _require
from .k1_terminal_analysis import paired_saved_errors
from .research_memory.trace import load_canonical_trace
from .screen_policy import canonical_fingerprint, validate_runtime_certificate
from .training_reproducibility import sha256_file

BASE = "experiments/pcqm_gptrans_execution40/gpu"
ARMS = ("execution_original", "execution_optimized")
PROTECTED = ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")


def verified_files(folder, manifest):
    files = manifest["files"]
    required = {"training/partial_manifest.json", "training/trace.json", "training/canonical_trace.json",
        "training/last_checkpoint.pt", "training/best_model.pt", "training/development_predictions.pt",
        "preflight/preflight.json", "preflight/runtime_certificate.json", "arm_binding.json",
        *(f"training/checkpoint_epoch_{epoch:02d}.pt" for epoch in (9,19,29,39)),
        *(f"phase_timing/epoch_{epoch:02d}.json" for epoch in range(40))}
    _require(required <= set(files), "Incomplete execution/retention artifacts")
    for name, digest in files.items():
        path = (folder / name).resolve()
        _require(path.is_relative_to(folder.resolve()) and path.is_file()
                 and sha256_file(path) == digest, "Output confinement/hash: " + name)


def accept(root, records, package):
    """Load predictions only; no model/checkpoint inference or training."""
    import torch
    root, records, package = Path(root), Path(records), Path(package)
    load = lambda p: json.loads(p.read_text())
    base = root / BASE
    config, receipt = load(base / "screen_config.json"), load(base / "submission_v1.json")
    spec = ExperimentSpec.from_json((base / "spec.json").read_text())
    frozen = verify_experiment_source_package(package)
    _require(receipt["status"] == "submitted" and receipt["reconciliation_required"] is False, "Physical receipt")
    from .kaggle_accelerator_push import _observed_identity
    actual = _observed_identity(receipt["platform_response"], "kaseichou")
    _require(not actual["identity_conflicts"] and all(receipt[k] == actual[k]
        for k in ("kernel", "kernel_id", "version_number")), "Returned run/version")
    for key in ("spec_identity", "package_identity", "source_commit"):
        _require(frozen[key] == receipt["release_binding"][key], "Receipt/package: " + key)
    _require(frozen["spec_identity"] == spec.identity, "Spec identity")
    _require(receipt["requested_kernel"] == config["requested_kernel"]
        and frozen["archive_sha256"] == receipt["release_binding"]["source_archive_sha256"], "Requested source binding")
    with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
        _require(config == json.loads(archive.extractfile(BASE+"/screen_config.json").read()), "Frozen configuration")
    screen = records / config["output_subdirectory"]
    startup, summary, cost = (load(screen / name) for name in ("startup.json", "job_summary.json", "native_cost.json"))
    _require(startup["spec_identity"] == spec.identity and startup["source_archive_sha256"] == frozen["archive_sha256"], "Startup/source")
    _require({r["variant"] for r in summary["outcomes"]} == set(ARMS)
        and all(r["complete"] is True for r in summary["outcomes"]), "Both forty-pass workers")
    _require(cost["allocated_gpu_count"] == cost["used_gpu_count"] == 2
        and len(cost["allocated_gpu_inventory"]) == 2 and all("T4" in v for v in cost["allocated_gpu_inventory"]), "T4x2 allocation")
    _require(cost["source_archive_sha256"] == frozen["archive_sha256"]
        and cost["wall_seconds"] == summary["elapsed_seconds"] and 0 < cost["wall_seconds"] <= 25200
        and abs(cost["allocated_device_hours"] - cost["wall_seconds"] * 2 / 3600) < 1e-10, "Native allocation/cost")
    results, payloads = {}, {}
    for arm in ARMS:
        folder = screen / arm
        completion = load(folder / "execution_completion.json")
        _require(completion["complete"] is True and completion["epochs"] == 40
            and completion["native_sixty_pass_complete"] is False and completion["arm"] == arm
            and completion["optimizer_steps"] == 31240 and completion["sample_presentations"] == 3998720
            and all(completion[k] is False for k in PROTECTED), "Forty-pass scope/roles")
        verified_files(folder, completion)
        binding = load(folder / "arm_binding.json")
        _require(binding == completion["binding"] and binding["spec_identity"] == spec.identity
            and binding["source_archive_sha256"] == frozen["archive_sha256"]
            and binding["trajectory_id"] == config["arms"][arm]["trajectory_id"], "Arm/source/prospective identity")
        partial = load(folder / "training/partial_manifest.json")
        _require(partial == completion["native_partial_manifest"] and partial["complete"] is False
            and partial["completed_epochs"] == partial["next_epoch"] == 40
            and partial["canonical_trace_sha256"] == sha256_file(folder / "training/canonical_trace.json")
            and partial["checkpoint_sha256"] == sha256_file(folder / "training/last_checkpoint.pt"), "Native prefix/checkpoint")
        preflight = load(folder / "preflight/preflight.json")
        _require(preflight["accepted"] is True and preflight["parameters"] == completion["parameters"] == 5871201
            and preflight["initial_state_artifact_sha256"] == completion["initial_state_artifact_sha256"] == config["arms"][arm]["initial_file_sha256"]
            and preflight["source_archive_sha256"] == frozen["archive_sha256"], "Calibration/initialization")
        fields = completion["scientific_fields"]
        validate_runtime_certificate(preflight["runtime_certificate"], {**fields,
            "platform_id":config["platform_id"], "accelerator":preflight["runtime_certificate"]["accelerator"],
            "runtime_certificate_id":preflight["runtime_certificate_id"]})
        _require(canonical_fingerprint(preflight["runtime_certificate"]) == preflight["runtime_certificate_id"]
            and partial["runtime_certificate_id"] == preflight["runtime_certificate_id"], "Certificate digest")
        rows = load(folder / "training/trace.json")["rows"]
        trace = load_canonical_trace(folder / "training/canonical_trace.json")
        _require(len(rows) == len(trace["observations"]) == 40, "All forty observations")
        from .pcqm_gptrans_v4 import FrozenEpochScheduler
        for epoch, (r,t) in enumerate(zip(rows, trace["observations"])):
            _require(r["epoch"] == epoch and r["cumulative_optimizer_steps"] == t["optimizer_step"] == (epoch+1)*781
                and r["cumulative_sample_presentations"] == t["sample_presentations"] == (epoch+1)*99968
                and r["learning_rate"] == FrozenEpochScheduler.learning_rate(epoch), "Native exposure/LR")
            for native, canonical in (("train_mae_eV","live_train_metric"), ("live_development_mae_eV","live_dev_metric"),
                    ("development_mae_eV","ema_dev_metric")):
                _require(math.isfinite(r[native]) and r[native] == t[canonical], "Canonical metrics")
        payload = torch.load(folder / "training/development_predictions.pt", map_location="cpu", weights_only=True)
        _require(payload["source_idx"].view(-1).numel() == 50000
            and torch.equal(payload["source_idx"].view(-1), torch.arange(100000,150000))
            and all(payload[k] is False for k in PROTECTED), "Prediction rows/roles")
        selected = min(rows, key=lambda r:r["development_mae_eV"])
        mae = float((payload["prediction_eV"].view(-1).double()-payload["target_eV"].view(-1).double()).abs().mean())
        _require(abs(mae-selected["development_mae_eV"]) < 1e-6, "Selected saved metric")
        wall = completion["stage_wall_seconds_including_loading_and_publication"]
        _require(math.isfinite(wall) and wall > 0, "Complete stage timing")
        results[arm] = {"parameters":completion["parameters"], "stage_wall_seconds":wall,
            "best_epoch":selected["epoch"], "selected_ema_mae_eV":mae,
            "final_live_mae_eV":rows[-1]["live_development_mae_eV"], "final_ema_mae_eV":rows[-1]["development_mae_eV"],
            "epoch_seconds":[r["elapsed_seconds"] for r in rows], "phase_timing":completion["phase_timing"],
            "runtime_certificate_id":preflight["runtime_certificate_id"]}
        payloads[arm] = payload
    paired = paired_saved_errors(payloads[ARMS[0]],payloads[ARMS[1]])
    speed = results[ARMS[0]]["stage_wall_seconds"] / results[ARMS[1]]["stage_wall_seconds"]
    interval_pass = paired["paired_row_bootstrap"]["ci95"][1] <= .001
    accuracy_pass = paired["candidate_minus_reference_eV"] <= .0005 and interval_pass
    return {"accepted":True,"format":"molgap-execution40-acceptance-v1", "model_inference_executed":False,
        "training_executed":False,"comparison_class":"PAIRED_ENDPOINT", "experiment_purpose":"delivery_experiment",
        "arms":results,"paired_accuracy":paired,"complete_stage_speedup":speed,"native_cost":cost,
        "speed_gate_passed":speed >= 1.1,
        "selected_accuracy_gate_passed":accuracy_pass, "paired_interval_gate_passed":interval_pass,
        "delivery_gate_passed":speed >= 1.1 and accuracy_pass,
        "protected_role_access":False, "scientific_caveat":"Execution/accuracy qualification, not architecture or scale superiority"}
