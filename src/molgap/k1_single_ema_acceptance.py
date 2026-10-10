"""CPU-only inspection of trusted, retained K1 single/EMA worker artifacts.

No model construction, forward pass, graph loading, platform access or RML
promotion occurs here. The remote manifest remains distinct from local custody.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .edge_state_training_core import EpochPermutationBatchSampler, _finite_tensors
from .pcqm_500k_v4_evidence import schedule
from .router import paired_bootstrap_mean
from .research_memory.trace import RMLTraceRecorder, canonicalize_trace, load_canonical_trace
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, assert_finite_state_dict, sha256_file
from .v4_runtime import state_dict_sha256, torch_load_compat, validate_standard_source_bundle

ARMS = ("reference", "ema999")
TRAIN_ROWS, DEV_ROWS, BS, HORIZON = 500_000, 50_000, 128, 60


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_trace(state, retained, *, train_rows=TRAIN_ROWS, batch_size=BS):
    """Check every optimizer coordinate and complete-epoch metric, without replay."""
    epoch, offset, steps = state["epoch"], state["offset"], state["steps"]
    batches = train_rows // batch_size
    _require(type(epoch) is int and 0 <= epoch <= HORIZON, "Invalid epoch")
    _require(type(offset) is int and 0 <= offset <= batches, "Invalid offset")
    _require(epoch < HORIZON or offset == 0, "Offset after horizon")
    _require(steps == epoch * batches + offset and state["samples"] == steps * batch_size,
             "Checkpoint exposure differs from cursor")
    _require(retained == {"epochs": state["trace"], "steps": state["step_trace"]},
             "External trace differs from checkpoint")
    _require(len(retained["epochs"]) == epoch and len(retained["steps"]) == steps,
             "Trace length differs from cursor")
    sampler = EpochPermutationBatchSampler.from_state(state["sampler"],
        dataset_size=train_rows, batch_size=batch_size, seed=42, epoch=epoch)
    _require(sampler.start_batch == offset, "Sampler offset differs")
    previous = -1.0
    for index, row in enumerate(retained["steps"]):
        number = index + 1
        _require((row["step"], row["epoch"], row["next_batch"], row["samples"]) ==
                 (number, index // batches, index % batches + 1, number * batch_size),
                 "Step trace coordinate differs")
        _require(all(math.isfinite(row[k]) for k in ("loss", "seconds", "allocation_seconds"))
                 and row["seconds"] >= 0 and row["allocation_seconds"] >= previous,
                 "Nonfinite/nonmonotonic step timing")
        previous = row["allocation_seconds"]
    previous = -1.0
    for index, row in enumerate(retained["epochs"]):
        _require((row["epoch"], row["steps"], row["samples"]) ==
                 (index, (index + 1) * batches, (index + 1) * batches * batch_size),
                 "Epoch exposure differs")
        # Linux and Windows libm can differ at the last bit of cosine.
        _require(math.isclose(row["lr"], schedule(index), rel_tol=1e-14, abs_tol=1e-18),
                 "Cosine60 LR differs")
        order = EpochPermutationBatchSampler(train_rows, batch_size, seed=42, epoch=index)
        _require(row["order_sha256"] == order.order_sha256, "Epoch sampler hash differs")
        section = retained["steps"][index * batches:(index + 1) * batches]
        _require(math.isclose(row["train_normalized_l1"],
                 math.fsum(s["loss"] for s in section) / batches, abs_tol=1e-12),
                 "Epoch loss differs from step trace")
        _require(all(math.isfinite(row[k]) for k in ("raw_mae_eV", "calibrated_mae_eV",
                 "round_seconds", "allocation_seconds")) and row["round_seconds"] >= 0
                 and row["allocation_seconds"] >= previous
                 and row["allocation_seconds"] >= section[-1]["allocation_seconds"],
                 "Invalid epoch metrics/timing")
        previous = row["allocation_seconds"]
    partial = retained["steps"][epoch * batches:]
    _require(math.isclose(state["train_loss_sum"], math.fsum(s["loss"] for s in partial),
                         abs_tol=1e-9), "Partial loss accumulator differs")
    return {"completed_epochs": epoch, "partial_batches": offset, "steps": steps,
            "samples": state["samples"],
            "step_seconds": math.fsum(row["seconds"] for row in retained["steps"]),
            "complete_round_seconds": math.fsum(row["round_seconds"] for row in retained["epochs"])}


def inspect_predictions(payload, *, start=TRAIN_ROWS, rows=DEV_ROWS):
    _require(set(payload) == {"source_idx", "target_eV", "prediction_eV", "mae_eV"},
             "Prediction fields differ")
    idx, target, prediction = (payload[k] for k in ("source_idx", "target_eV", "prediction_eV"))
    _require(idx.dtype == torch.int64 and idx.shape == target.shape == prediction.shape == (rows,),
             "Prediction dtype/shape differs")
    _require(torch.equal(idx, torch.arange(start, start + rows)), "Development source order differs")
    _require(target.dtype == prediction.dtype == torch.float32 and
             bool(torch.isfinite(target).all() and torch.isfinite(prediction).all()),
             "Prediction is not finite FP32")
    mae = float((prediction - target).abs().mean())
    _require(math.isclose(mae, payload["mae_eV"], rel_tol=5e-7, abs_tol=1e-8),
             "Saved MAE differs from predictions")
    return {"mae_eV": payload["mae_eV"], "cpu_recomputed_mae_eV": mae,
            "source_idx_sha256": state_dict_sha256({"source_idx": idx}),
            "target_sha256": state_dict_sha256({"target_eV": target}),
            "tensor_sha256": state_dict_sha256({
        "source_idx": idx, "target_eV": target, "prediction_eV": prediction})}


def compare_predictions(reference, candidate, *, n_bootstrap=10_000):
    _require(torch.equal(reference["source_idx"], candidate["source_idx"]) and
             torch.equal(reference["target_eV"], candidate["target_eV"]),
             "Paired membership/targets differ")
    target = reference["target_eV"].double().numpy()
    ref_error = np.abs(reference["prediction_eV"].double().numpy() - target)
    cand_error = np.abs(candidate["prediction_eV"].double().numpy() - target)
    result = paired_bootstrap_mean(cand_error - ref_error, n_bootstrap=n_bootstrap, seed=42)
    return {"reference_mae_eV": float(ref_error.mean()), "candidate_mae_eV": float(cand_error.mean()),
            "reference_minus_candidate_eV": float((ref_error - cand_error).mean()),
            "candidate_minus_reference_bootstrap": result,
            "uncertainty_scope": "paired development rows; not training-seed variance",
            "selection_role": "repeatedly consumed internal development; not independent holdout"}


def _inspect_resume(state, name):
    for field in ("model", "optimizer", "rng", "ema"):
        _finite_tensors(state[field], f"{name}.{field}")
    for field in ("model", "ema"):
        if state[field] is not None:
            assert_finite_state_dict(state[field], label=field)
            _require(all(not v.is_floating_point() or v.dtype == torch.float32
                         for v in state[field].values()), "State is not FP32")
    _require((state["ema"] is None) == (name == "reference"), "EMA arm differs")
    rng = state["rng"]
    _require(set(rng) == {"python", "numpy", "torch", "cuda"} and len(rng["cuda"]) == 1
             and rng["torch"].dtype == torch.uint8 and rng["torch"].numel() > 0
             and rng["cuda"][0].dtype == torch.uint8 and rng["cuda"][0].numel() > 0,
             "Missing RNG resume contents")
    optimizer = state["optimizer"]
    _require(len(optimizer["param_groups"]) == 1, "Optimizer groups differ")
    group = optimizer["param_groups"][0]
    active_epoch = state["epoch"] if state["offset"] else state["epoch"] - 1
    _require(math.isclose(group["lr"], schedule(active_epoch), rel_tol=1e-14, abs_tol=1e-18)
             and group["weight_decay"] == 1e-5
             and group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
             and group["foreach"] is False and group["fused"] is False
             and group["amsgrad"] is False, "Optimizer recipe differs")
    _require(set(optimizer["state"]) == set(group["params"]), "Optimizer states missing")
    for value in optimizer["state"].values():
        _require(set(value) == {"step", "exp_avg", "exp_avg_sq"}
                 and int(value["step"]) == state["steps"]
                 and value["exp_avg"].shape == value["exp_avg_sq"].shape
                 and value["exp_avg"].dtype == value["exp_avg_sq"].dtype == torch.float32,
                 "Optimizer resume state differs")
    return {"model_sha256": state_dict_sha256(state["model"]),
            "ema_sha256": None if state["ema"] is None else state_dict_sha256(state["ema"]),
            "resume_contents": "inspected only; no local runtime replay performed"}


def inspect_saved_run(attempt, owner, *, n_bootstrap=10_000):
    attempt, owner = Path(attempt), Path(owner)
    worker = attempt / "worker"
    inputs, package = _read(owner / "inputs.json"), _read(owner / "submission/package_binding.json")
    submission = _read(owner / "submission/kernel_push.json")
    observation = _read(owner / "submission/output_listing_20261011.json")
    _require(observation["kernel"] == submission["kernel"]
             and observation["status"]["status"] == "COMPLETE",
             "Retained scheduler observation differs from submitted run")
    manifest = _read(worker / "artifacts.json")
    _require(manifest["manifest_complete"] is True, "Incomplete remote artifact manifest")
    hashes = {}

    def verified(relative):
        path = (worker / relative).resolve()
        _require(worker.resolve() in path.parents, "Unsafe artifact pointer")
        _require(path.is_file(), f"Missing retained artifact: {relative}")
        digest = sha256_file(path)
        _require(manifest["artifacts"].get(relative) == digest, f"Artifact SHA differs: {relative}")
        hashes[relative] = digest
        return path

    metadata = {name: _read(verified(name + ".json")) for name in
                ("terminal", "last", "trace", "runtime", "preflight", "binding", "runconfig", "sample_manifest")}
    terminal, last, runtime = (metadata[k] for k in ("terminal", "last", "runtime"))
    _require(terminal["status"] == last["status"] == manifest["status"] == "STOP_FOR_COST"
             and terminal["complete"] is False and last["complete"] is False,
             "Not a STOP_FOR_COST partial endpoint")
    config, binding = metadata["runconfig"], metadata["binding"]
    payload = _read(attempt / "frozen_payload/k1_t4_payload.json")
    _require(sha256_file(attempt / "frozen_payload/k1_t4_payload.json") == package["payload_sha256"],
             "Frozen payload differs")
    _require({k: v for k, v in config.items() if k != "allocation_started_unix"} == payload["runconfig"],
             "Frozen config differs")
    _require(config == binding["config"] and hashes["runconfig.json"] == binding["config_sha256"],
             "Config binding differs")
    _require(config["source_commit"] == inputs["source_commit"] == package["source"]["source_commit"]
             and config["accelerator"] == inputs["accelerator"] == "T4"
             and config["allocation_wall_limit_seconds"] == inputs["allocation_wall_ceiling_seconds"] == 32400,
             "Source/allocation contract differs")
    _require(inputs["seed"] == 42 and inputs["ema_decay"] == 0.999 and
             inputs["full_schedule_epochs"] == HORIZON and inputs["batch_size"] == BS and
             inputs["arms"] == list(ARMS) and inputs["bn_calibration_members"] ==
             {"start": 0, "stop": 16384}, "Frozen scientific inputs differ")
    _require(hashes["sample_manifest.json"] == config["cpu_accepted_dataset_manifest_sha256"]
             == inputs["manifest_sha256"], "Fixed cache manifest differs")
    _require(metadata["sample_manifest"]["roles"] == {
        "train": {"source_idx_start": 0, "source_idx_stop": TRAIN_ROWS, "rows": TRAIN_ROWS},
        "development": {"source_idx_start": TRAIN_ROWS, "source_idx_stop": TRAIN_ROWS + DEV_ROWS,
                        "rows": DEV_ROWS}}, "Cache role membership differs")
    initial_path, source_path = verified("initial.pt"), verified("source.archive")
    _require(hashes["initial.pt"] == inputs["initial_file_sha256"] == binding["initial_file_sha256"],
             "Initial file binding differs")
    initial = torch_load_compat(initial_path, map_location="cpu", weights_only=False)
    _require(initial["format"] == config["initial_format"] and
             state_dict_sha256(initial["model_state"]) == initial["state_sha256"] ==
             config["initial_state_sha256"] == inputs["initial_tensor_sha256"], "Initial tensors differ")
    assert_finite_state_dict(initial["model_state"], label="initial")
    source_sidecars = {}
    with tempfile.TemporaryDirectory(prefix="molgap-source-inspection-") as temporary:
        staging = Path(temporary)
        shutil.copyfile(source_path, staging / "source.archive")
        for name in ("SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
            sidecar = attempt / "frozen_payload" / name
            if not sidecar.is_file():
                sidecar = Path(package["source"]["archive"]).parent / name
            _require(sha256_file(sidecar) == payload["files"][name], "Source sidecar differs")
            source_sidecars[name] = {"path": str(sidecar.resolve()), "sha256": sha256_file(sidecar)}
            shutil.copyfile(sidecar, staging / name)
        validate_standard_source_bundle(staging / "source.archive",
            package["source"]["archive_sha256"], inputs["source_commit"])
    _require(runtime["runtime_fingerprint"] == canonical_fingerprint({
        k: v for k, v in runtime.items() if k != "runtime_fingerprint"}), "Runtime fingerprint differs")
    _require(runtime["installed_distributions_sha256"] == hashlib.sha256(
        "\n".join(runtime["installed_distributions"]).encode()).hexdigest(), "Package fingerprint differs")
    _require(runtime["determinism"]["precision"] == "fp32" and
             runtime["determinism"]["tf32_enabled"] is False and
             runtime["determinism"]["seed"] == 42 and
             runtime["determinism"]["deterministic_algorithms"] is True and
             runtime["accelerator"]["name"] == "Tesla T4" and
             runtime["accelerator"]["device_count_visible"] == 1, "Runtime qualification differs")
    preflight = metadata["preflight"]
    for name in ARMS:
        row = preflight[name]
        _require(row["exact_resume"] is True and row["formal_samples"] == 0 and
                 row["initial_state_sha256"] == inputs["initial_tensor_sha256"] and
                 len(row["fixture_sha256"]) == 2 and
                 all(math.isfinite(v) for v in row["repeat"]["losses"]), "Preflight differs")
    _require(preflight["reference"]["fixture_sha256"] == preflight["ema999"]["fixture_sha256"],
             "Preflight fixture mismatch")
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        _require(terminal[role] is False, "Protected role read reported")
    checkpoint = torch_load_compat(verified("last.pt"), map_location="cpu", weights_only=False)
    _require(hashes["last.pt"] == last["sha256"] == terminal["sha256"] and
             hashes["trace.json"] == last["trace_sha256"] == terminal["trace_sha256"],
             "Terminal digest differs")
    _require(checkpoint["format"] == "molgap-colab-k1-paired500k-v1" and
             checkpoint["status"] == "STOP_FOR_COST" and set(checkpoint["arms"]) == set(ARMS)
             and checkpoint["runtime"] == runtime["runtime_fingerprint"], "Checkpoint identity differs")
    full_binding = checkpoint["binding"]
    _require({k: v for k, v in full_binding.items() if k != "target_statistics"} == binding,
             "Checkpoint source/config binding differs")
    exposure = {}
    for name in ARMS:
        state = checkpoint["arms"][name]
        _require({k: state[k] for k in terminal["arms"][name]} == terminal["arms"][name]
                 == last["arms"][name], "Terminal arm cursor differs")
        exposure[name] = {**inspect_trace(state, metadata["trace"][name]), **_inspect_resume(state, name)}
    ref, ema = (checkpoint["arms"][name] for name in ARMS)
    _require((ref["epoch"] == ema["epoch"] and ema["offset"] == 0) or
             (ref["epoch"] == ema["epoch"] + 1 and ref["offset"] == 0), "Alternating pair differs")
    matched = min(state["epoch"] for state in checkpoint["arms"].values())
    _require(matched == terminal["matched_completed_epochs"] == last["matched_completed_epochs"]
             and 0 < matched < HORIZON, "Matched partial prefix differs")
    comparisons, endpoints = {}, {}
    for selection in ("best_prefix", "matched_last"):
        paired = {}
        endpoints[selection] = {}
        for name in ARMS:
            rows = metadata["trace"][name]["epochs"][:matched]
            row = min(rows, key=lambda r: r["calibrated_mae_eV"]) if selection == "best_prefix" else rows[-1]
            relative = f"{name}/epoch_{row['epoch']:02d}.pt"
            if selection == "best_prefix":
                expected = {"epoch": row["epoch"], "calibrated_mae_eV": row["calibrated_mae_eV"],
                            "snapshot": relative}
                _require(expected == terminal["matched_prefix_selection"][name] ==
                         last["matched_prefix_selection"][name], "Matched selection differs")
            saved = torch_load_compat(verified(relative), map_location="cpu", weights_only=False)
            selected = saved["selected"]
            _require(saved["epoch"] == selected["epoch"] == row["epoch"] and
                     saved["binding"] == selected["binding"] == full_binding and
                     selected["selection"] == ("live-calibrated" if name == "reference" else "ema-calibrated"),
                     "Endpoint binding differs")
            calibration = saved["calibration"]
            _require(calibration == selected["calibration"] and calibration["rows"] == 16384
                     and calibration["batches"] == 128 and calibration["bn_modules"] == 18
                     and calibration["passes_per_batch"] == 1
                     and calibration["labels_used_for_calibration"] is False
                     and all(calibration[k] is True for k in ("parameters_unchanged",
                         "non_bn_buffers_unchanged", "dropout_disabled", "buffers_restored"))
                     and set(calibration["num_batches_tracked"].values()) == {128},
                     "Features-only BN calibration differs")
            clean, raw = inspect_predictions(saved["calibrated"]), inspect_predictions(saved["raw"])
            _require(clean["mae_eV"] == row["calibrated_mae_eV"] and raw["mae_eV"] == row["raw_mae_eV"],
                     "Trace endpoint metrics differ")
            _require(torch.equal(saved["raw"]["target_eV"], saved["calibrated"]["target_eV"]),
                     "Raw/clean targets differ")
            selected_summary = inspect_predictions(selected["payload"])
            _require(selected_summary == clean and selected["mean"] == full_binding["target_statistics"]["mean"]
                     and selected["std"] == full_binding["target_statistics"]["std"], "Selected payload differs")
            assert_finite_state_dict(selected["model"], label="selected")
            buffers = {k: v for k, v in selected["model"].items() if
                       k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
            _require(state_dict_sha256(buffers) == calibration["buffer_sha256_calibrated"],
                     "Selected BN buffer digest differs")
            endpoints[selection][name] = {"epoch_zero_based": row["epoch"], "clean": clean,
                "raw_secondary": raw, "selected_model_sha256": state_dict_sha256(selected["model"])}
            paired[name] = saved
        comparisons[selection] = {"clean_primary": compare_predictions(
            paired["reference"]["calibrated"], paired["ema999"]["calibrated"], n_bootstrap=n_bootstrap),
            "raw_secondary": compare_predictions(paired["reference"]["raw"],
                paired["ema999"]["raw"], n_bootstrap=n_bootstrap)}
    allocation = _read(attempt / "allocation.json")
    wall = allocation["observed_allocation_wall_seconds"]
    _require(allocation["physical_gpu_names"] == ["Tesla T4", "Tesla T4"] and
             allocation["cuda_visible_devices"] == "0" and math.isfinite(wall) and
             terminal["elapsed_allocation_seconds"] <= wall <= 32400, "Allocation differs")
    listing = _read(owner / "submission/output_listing_20261011.json")
    receipt = _read(owner / "submission/kernel_push.json")
    _require(listing["kernel"] == package["requested_kernel"] and listing["status"]["status"] == "COMPLETE",
             "Scheduler receipt differs")
    _require(receipt["kernel"] == package["requested_kernel"] and
             receipt["version_number"] == 1 and receipt["kernel_id"] == 137983743,
             "Exact Kaggle job receipt differs")
    return {"outcome": "STOP_FOR_COST", "science": "INCONCLUSIVE", "adoption": False,
        "mechanical_acceptance": "PASS_RETAINED_PARTIAL",
        "scheduler_status": observation["status"]["status"],
        "worker_status": "STOP_FOR_COST", "source_commit": inputs["source_commit"],
        "kernel": receipt["kernel"], "version": receipt["version_number"],
        "kernel_id": receipt["kernel_id"], "runtime_fingerprint": runtime["runtime_fingerprint"],
        "run_id": inputs["run_id"], "matched_completed_epochs": matched,
        "matched_steps_per_arm": matched * (TRAIN_ROWS // BS),
        "matched_samples_per_arm": matched * (TRAIN_ROWS // BS) * BS,
        "strict_60epoch_replay": False, "exposure": exposure, "endpoints": endpoints,
        "comparisons": comparisons, "verified_local_artifact_sha256": hashes,
        "source_sidecar_custody": source_sidecars,
        "remote_manifest_not_locally_retrieved": sorted(set(manifest["artifacts"]) - set(hashes)),
        "native_cost": {"allocation_wall_seconds": wall, "allocated_T4_count": 2,
            "used_T4_count": 1, "allocated_T4_hours": 2 * wall / 3600,
            "visible_device_allocation_hours": wall / 3600,
            "provider_billing_units": allocation["provider_billing_units"]},
        "limitations": ["No complete60 endpoint or promotion gate test",
            "Source inspection establishes calibration membership; retained report has no membership digest",
            "Fixture hashes and exact-resume certificate inspected; no fixture graph/model replay",
            "Target equality checked across retained predictions; no source shard target reload",
            "Optimizer/RNG contents inspected; local CPU is not the qualified T4 runtime",
            "RML finalization pending accepted V5 terminal envelope and trace manifests/reference bindings"]}


def export_canonical_traces(attempt, owner, output_directory, acceptance):
    """Translate observed epoch rows through the RML owner, not a new schema."""
    attempt, owner, output_directory = Path(attempt), Path(owner), Path(output_directory)
    raw = _read(attempt / "worker/trace.json")
    _require(sha256_file(attempt / "worker/trace.json") ==
             acceptance["verified_local_artifact_sha256"]["trace.json"], "Trace changed after acceptance")
    trajectory = _read(owner / "rml/trajectory.json")
    exported = {}
    for arm in ARMS:
        def semantics(metric, unit, role, weights):
            return {"metric": metric, "unit": unit, "target": "gap", "role_identity": role,
                    "weights": weights, "direction": "minimize"}
        definitions = {"live_train_metric": semantics("normalized stochastic online L1", "normalized",
            "train source_idx0:500000", "live"), "live_dev_metric": None, "ema_dev_metric": None}
        field = "live_dev_metric" if arm == "reference" else "ema_dev_metric"
        definitions[field] = semantics("clean BN-calibrated MAE", "eV",
            "consumed internal development source_idx500000:550000", "live" if arm == "reference" else "ema")
        path = output_directory / f"{arm}_trace.json"
        observations = []
        for row in raw[arm]["epochs"]:
            observations.append(dict(optimizer_step=row["steps"], sample_presentations=row["samples"],
                epoch_or_pass=row["epoch"] + 1, learning_rate=row["lr"],
                live_train_metric=row["train_normalized_l1"],
                wall_time_seconds=row["round_seconds"], cumulative_wall_time_seconds=row["allocation_seconds"],
                **{field: row["calibrated_mae_eV"]}))
        exposure = acceptance["exposure"][arm]
        partial = exposure["partial_batches"] > 0
        observations.append(dict(event="terminal", optimizer_step=exposure["steps"] if partial else None,
            sample_presentations=exposure["samples"] if partial else None,
            epoch_or_pass=exposure["completed_epochs"] + exposure["partial_batches"] / (TRAIN_ROWS // BS),
            checkpoint_identity=acceptance["verified_local_artifact_sha256"]["last.pt"],
            cumulative_wall_time_seconds=_read(attempt / "worker/terminal.json")["elapsed_allocation_seconds"]))
        expected = canonicalize_trace({"trajectory_id": trajectory["trajectory_id"],
            "run_id": acceptance["run_id"] + ":" + arm, "metric_semantics": definitions,
            "device_time_semantics": None, "observations": observations})
        if path.exists():
            _require(load_canonical_trace(path) == expected, "Existing canonical trace differs from accepted source")
        else:
            recorder = RMLTraceRecorder(path, trajectory_id=trajectory["trajectory_id"],
                run_id=acceptance["run_id"] + ":" + arm, metric_semantics=definitions)
            for observation in expected["observations"]:
                recorder.append_observation(**{k: v for k, v in observation.items() if k != "sequence"})
        exported[arm] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    result = inspect_saved_run(args.attempt, args.owner, n_bootstrap=args.n_bootstrap)
    result["canonical_epoch_traces"] = export_canonical_traces(
        args.attempt, args.owner, args.output.parent, result)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("outcome", "science", "mechanical_acceptance",
                                                  "matched_completed_epochs", "native_cost")}))


if __name__ == "__main__":
    main()
