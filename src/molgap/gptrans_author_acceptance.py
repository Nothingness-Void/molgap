"""Verify retained author-input preparation and dual-arm training artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .gptrans_author_inputs import (
    DEGREE_SCALE, INITIAL_MODEL_SHA256, INITIAL_STATE_SHA256,
    INITIAL_SCALED_FORMAT, state_digest, validate_sidecar,
)
from .gptrans_author_variants import DEGREE_INITIAL_SHA256
from .training_reproducibility import sha256_file

NO_READ = ("training_executed", "model_inference_executed", "labels_read",
           "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")


def accept_training_outputs(repo_root: Path, records: Path, package: Path, *,
                            experiment_ref="experiments/pcqm_gptrans_author_alignment/gpu", selected_arms=None) -> dict:
    """Verify native G1/G2 tensors and telemetry without loading model weights.

    Reuse saved-error analysis, runtime and canonical-trace validators. The
    receipt, actual source package and prospective bindings qualify identities;
    an arbitrary completed directory cannot be adopted as this experiment.
    """
    import hashlib
    import tarfile
    import torch
    from .experiment_package import verify_experiment_source_package
    from .experiment_spec import ExperimentSpec
    from .k1_terminal_analysis import paired_saved_errors
    from .research_memory.trace import load_canonical_trace
    from .screen_policy import validate_runtime_certificate, canonical_fingerprint
    from .pcqm_gptrans_v4 import BATCHES_PER_EPOCH, EPOCHS, EXPECTED_PARAMETERS, PHYSICAL_BATCH, RUN_FORMAT

    root, records = Path(repo_root).resolve(), Path(records).resolve()
    base = root / experiment_ref
    load = lambda p: json.loads(p.read_bytes())
    require = lambda condition, message: _require(condition, message)
    spec = ExperimentSpec.from_json((base / "spec.json").read_text())
    receipt, config = load(base / "submission_v1.json"), load(base / "screen_config.json")
    frozen = verify_experiment_source_package(package)
    require(receipt["status"] == "submitted" and receipt["reconciliation_required"] is False,
            "Unqualified physical submission")
    legacy = experiment_ref == "experiments/pcqm_gptrans_author_alignment/gpu"
    if legacy:
        require((receipt["kernel_id"], receipt["version_number"]) == (136543794, 1), "Physical run identity")
    else:
        from .kaggle_accelerator_push import _observed_identity
        actual = _observed_identity(receipt["platform_response"], config["requested_kernel"].split("/")[0])
        require(receipt["requested_kernel"] == config.get("requested_kernel", "nvoid912/molgap-gptrans-g1-path-ema-dual-s42")
            and receipt["kernel_id"] > 0 and receipt["version_number"] == 1
            and not actual["identity_conflicts"]
            and all(receipt[k] == actual[k] for k in ("kernel", "kernel_id", "version_number")), "Physical run identity")
    for key in ("spec_identity", "package_identity", "source_commit"):
        require(frozen[key] == receipt["release_binding"][key], "Package receipt binding: " + key)
    source_sha = receipt["release_binding"]["source_archive_sha256"]
    require(frozen["archive_sha256"] == source_sha and frozen["spec_identity"] == spec.identity,
            "Executable source binding")
    with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
        variant_sha = hashlib.sha256(archive.extractfile("src/molgap/gptrans_author_variants.py").read()).hexdigest()
        archived_config = json.loads(archive.extractfile(experiment_ref + "/screen_config.json").read())
    require(config == archived_config, "Question configuration changed since release")
    screen = records / config.get("output_subdirectory", "gptrans_author_screen" if legacy else "gptrans_input_ema_screen")
    startup, cost, summary = load(screen / "startup.json"), load(screen / "native_cost.json"), load(screen / "job_summary.json")
    require(startup["spec_identity"] == spec.identity and startup["source_archive_sha256"] == source_sha,
            "Startup identity")
    selected = _selected_completed_arms(config["arms"], summary["outcomes"], selected_arms)
    from .gptrans_author_screen import validate_arm_allocation
    modes = validate_arm_allocation(config)
    require(startup["arms"] == list(modes), "Declared startup arms")
    if len(modes) == 1:
        require(startup.get("single_arm_reason") == config["single_arm_reason"], "Single-arm justification binding")
    require(cost["allocated_gpu_count"] == 2 and cost["used_gpu_count"] == len(modes)
            and len(cost["allocated_gpu_inventory"]) == 2 and all("T4" in v for v in cost["allocated_gpu_inventory"]), "T4x2 cost")
    require(cost["source_archive_sha256"] == source_sha and math.isfinite(cost["wall_seconds"])
            and 0 < cost["wall_seconds"] <= config["maximum_wall_seconds"]
            and cost["wall_seconds"] == summary["elapsed_seconds"], "Native wall/source accounting")
    require(abs(cost["allocated_device_hours"] - cost["wall_seconds"] * 2 / 3600) < 1e-12, "Native allocation total")
    bundle = load(root / config["reference_bundle_ref"])
    require(sha256_file(root / config["reference_bundle_ref"]) == config["reference_bundle_sha256"], "Reference bundle changed")
    ref_meta = (load(root / "experiments/pcqm_gptrans_v5_audit_reference/results/terminal/prediction_manifest.json")
                if legacy else bundle["prediction_manifest"])
    ref_path = root / ref_meta["artifact_locator"]
    require(sha256_file(ref_path) == ref_meta["artifact_sha256"], "Reference prediction artifact")
    reference = torch.load(ref_path, map_location="cpu", weights_only=True)
    ref_manifest = load(root / bundle["trace_manifest_ref"])
    ref_trace = load_canonical_trace(root / ref_manifest["trace_artifact_ref"])
    arms = {}
    for mode, arm in config["arms"].items():
        if mode not in selected:
            continue
        folder = screen / mode
        expected_parameters = arm.get("expected_parameters", EXPECTED_PARAMETERS)
        if "expected_parameters" in arm:
            if mode == "degree_pair_transition_ema999":
                from .gptrans_pair_transition import PARAMETER_CAP, architecture_identity
            else:
                from .gptrans_capacity import PARAMETER_CAP, architecture_identity
            require(0 < expected_parameters <= PARAMETER_CAP and architecture_identity(mode) == arm["comparison_identity"]["architecture_config_identity"], "Capacity allowance/implementation")
        training = folder / "training"
        manifest, observed, preflight = (load(training / "completion_manifest.json"),
            load(training / "frozen_reference.json"), load(folder / "preflight/preflight.json"))
        plan = load(base / mode / "rml_plan/trajectory.json")
        require(plan["trajectory_id"] == arm["trajectory_id"] and plan["actions"][0]["run_ids"] == [observed["run_id"]], "Prospective run binding")
        require(load(folder / "arm_binding.json") == {"spec_identity": spec.identity, "arm_id": mode,
            "trajectory_id": arm["trajectory_id"], "source_archive_sha256": source_sha}, "Worker arm binding")
        require(manifest["format"] == RUN_FORMAT and manifest["complete"] is True
            and manifest["v5_audit"] is True and manifest["epochs"] == EPOCHS
            and manifest["parameters"] == expected_parameters and manifest["variant"] == mode, "Completion/architecture")
        require(manifest["optimizer_steps"] == BATCHES_PER_EPOCH * EPOCHS
            and manifest["sample_presentations"] == BATCHES_PER_EPOCH * EPOCHS * PHYSICAL_BATCH, "Terminal exposure")
        for record in (manifest, preflight):
            require(record["source_archive_sha256"] == source_sha and record["source_commit"] == frozen["source_commit"], "Arm source")
            require(record["variant_source_sha256"] == variant_sha and record["manifest_sha256"] == config["dataset_manifest_sha256"], "Variant/data identity")
            require(record["initial_state_artifact_sha256"] == arm["initial_file_sha256"], "Initial file binding")
        require(preflight["accepted"] is True and preflight["parameters"] == expected_parameters, "Preflight")
        if "expected_parameters" in arm:
            with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
                module = "gptrans_pair_transition.py" if mode == "degree_pair_transition_ema999" else "gptrans_capacity.py"
                capacity_sha = hashlib.sha256(archive.extractfile("src/molgap/" + module).read()).hexdigest()
            require(preflight["capacity_implementation_sha256"] == capacity_sha and
                preflight["capacity_architecture_identity"] == arm["comparison_identity"]["architecture_config_identity"], "Capacity source binding")
        if mode == "degree_pair_transition_ema999":
            expected_config = __import__("molgap.gptrans_pair_transition", fromlist=["configuration"]).configuration(mode)
            require(observed.get("capacity_configuration") == expected_config, "Pair-transition configuration")
        validate_runtime_certificate(preflight["runtime_certificate"], observed)
        require(observed["runtime_certificate_id"] == manifest["runtime_certificate_id"] == preflight["runtime_certificate_id"], "Runtime binding")
        if mode in {"path_bond_mean", "degree_path_bond_mean", "degree_path_bond_mean_ema999"}:
            sidecar = manifest["path_sidecar_identity"]
            require(sidecar["valid"] is True and sidecar["content_hashes_checked"] is True
                and sidecar["sidecar_manifest_sha256"] == config["path_manifest_sha256"]
                and sidecar["dataset_manifest_sha256"] == config["dataset_manifest_sha256"], "Path sidecar binding")
        else:
            require(manifest["path_sidecar_identity"] is None, "Unexpected path consumption")
        for name, key in (("best_model.pt", "best_model_sha256"), ("development_predictions.pt", "development_predictions_sha256"),
                          ("last_checkpoint.pt", "checkpoint_sha256"), ("canonical_trace.json", "canonical_trace_sha256"),
                          ("frozen_reference.json", "frozen_reference_sha256")):
            require(sha256_file(training / name) == manifest[key], "Artifact hash: " + name)
        require(set(manifest["checkpoint_chunks"]) == {f"checkpoint_epoch_{e:02d}.pt" for e in (9,19,29,39,49,59)}, "Checkpoint chunk inventory")
        for name, digest in manifest["checkpoint_chunks"].items():
            require(sha256_file(training / name) == digest, "Checkpoint chunk: " + name)
        trace = load_canonical_trace(training / "canonical_trace.json")
        rows = load(training / "trace.json")["rows"]
        require((trace["trajectory_id"], trace["run_id"]) == (arm["trajectory_id"], observed["run_id"]), "Trace identity")
        require(len(trace["observations"]) == len(rows) == EPOCHS, "Trace coverage")
        if mode == "degree_pair_transition_ema999":
            for row in rows:
                diagnostic = row.get("pair_transition_diagnostics", {})
                require(diagnostic.get("before_blocks") == [3, 6, 9, 12], "Transition diagnostic insertion identity")
                values = diagnostic.get("input_return_output_rms", [])
                require(len(values) == 4 and all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in values), "Finite transition diagnostics")
        require(trace["metric_semantics"] == ref_trace["metric_semantics"], "Trace metric semantics")
        for i, (row, native, ref_row) in enumerate(zip(rows, trace["observations"], ref_trace["observations"])):
            require(row["epoch"] == i and native["epoch_or_pass"] == i + 1, "Epoch order")
            for native_name, canonical in (("train_mae_eV", "live_train_metric"), ("development_mae_eV", "ema_dev_metric"),
                                     ("live_development_mae_eV", "live_dev_metric"), ("learning_rate", "learning_rate"),
                                     ("cumulative_optimizer_steps", "optimizer_step"), ("cumulative_sample_presentations", "sample_presentations"),
                                     ("cumulative_wall_time_seconds", "cumulative_wall_time_seconds")):
                require(math.isfinite(row[native_name]) and row[native_name] == native[canonical], "Native metric/counter: " + canonical)
            require(native["optimizer_step"] == (i+1)*BATCHES_PER_EPOCH
                and native["sample_presentations"] == (i+1)*BATCHES_PER_EPOCH*PHYSICAL_BATCH
                and native["learning_rate"] == ref_row["learning_rate"], "Matched exposure/schedule")
            require(native["event"] == "checkpoint" and native["checkpoint_identity"], "Observed checkpoint")
        require(trace["observations"][-1]["checkpoint_identity"] == "sha256:" + manifest["checkpoint_sha256"], "Terminal checkpoint binding")
        payload = torch.load(training / "development_predictions.pt", map_location="cpu", weights_only=True)
        for record in (manifest, payload, startup):
            require(all(record.get(k) is False for k in NO_READ[3:]), "Protected role consumption")
        require(torch.equal(payload["source_idx"].view(-1), torch.arange(100000,150000)), "Development row order")
        analysis = paired_saved_errors(reference, payload)
        require(abs(analysis["candidate_mae_eV"] - manifest["best_development_mae_eV"]) < 1e-7, "Recomputed MAE")
        best = min(rows, key=lambda r: r["development_mae_eV"])
        require(best["epoch"] == manifest["best_epoch"] and best["development_mae_eV"] == manifest["best_development_mae_eV"], "Frozen EMA selection")
        prediction_manifest = {**bundle["prediction_manifest"], "artifact_locator": (training / "development_predictions.pt").relative_to(root).as_posix(),
            "artifact_sha256": manifest["development_predictions_sha256"]}
        for field, tensor in (("prediction_sha256", "prediction_eV"), ("source_idx_sha256", "source_idx"), ("target_sha256", "target_eV")):
            prediction_manifest[field] = hashlib.sha256(payload[tensor].contiguous().numpy().tobytes()).hexdigest()
        require(all(prediction_manifest[k] == bundle["prediction_manifest"][k] for k in ("source_idx_sha256", "target_sha256")), "Reference row/target hashes")
        identity = ({**bundle["comparison_identity"], "architecture_config_identity": observed["architecture_fingerprint"]}
                    if legacy else arm["comparison_identity"])
        require(identity["architecture_config_identity"] == observed["architecture_fingerprint"], "Architecture identity")
        require(manifest.get("ema_decay", .9999) == identity["ema_decay"], "EMA identity")
        if mode == "degree_pair_depth_scale_ema999":
            with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
                source_sha = hashlib.sha256(archive.extractfile("src/molgap/gptrans_pair_scale.py").read()).hexdigest()
            require(source_sha == preflight["pair_scale_implementation_sha256"], "Pair scaling source")
            require(all(row["pair_scale_diagnostics"]["scale"] == 12 ** -0.5
                and row["pair_scale_diagnostics"]["source"] == "first_scheduled_training_batch_before_update"
                and len(row["pair_scale_diagnostics"]["layer_input_raw_update_output_rms"]) == 12
                and all(len(layer) == 3 and all(math.isfinite(v) and v >= 0 for v in layer)
                        for layer in row["pair_scale_diagnostics"]["layer_input_raw_update_output_rms"])
                for row in rows), "Finite observed pair scaling diagnostics")
        if mode in {"degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999"}:
            with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
                readout_sha = hashlib.sha256(archive.extractfile("src/molgap/gptrans_readout.py").read()).hexdigest()
            require(preflight["readout_implementation_sha256"] == readout_sha, "Readout implementation binding")
        if mode in {"degree_group_decay_ema999", "degree_path_endpoints_ema999"}:
            with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
                endpoint_sha = hashlib.sha256(archive.extractfile("src/molgap/gptrans_endpoint_paths.py").read()).hexdigest()
            require(preflight["endpoint_implementation_sha256"] == endpoint_sha, "Endpoint/optimizer implementation binding")
        if mode == "degree_group_decay_ema999":
            require(all("optimizer_diagnostics" in row and math.isfinite(row["optimizer_diagnostics"]["mean_preclip_gradient_norm"])
                        and 0 <= row["optimizer_diagnostics"]["clip_frequency"] <= 1 for row in rows), "Grouped optimizer diagnostic coverage")
            from .gptrans_endpoint_paths import parameter_groups_fingerprint
            require(all(row["optimizer_diagnostics"]["parameter_groups_sha256"] == parameter_groups_fingerprint(preflight["optimizer_parameter_groups"])
                        and all(math.isfinite(v) for k in ("group_weight_norms", "group_adam_first_moment_norms") for v in row["optimizer_diagnostics"][k])
                        for row in rows), "Grouped optimizer identity and finite norms")
        if mode == "degree_decay001_ema999":
            from .pcqm_gptrans_v4 import FrozenEpochScheduler
            require(all(row["optimizer_diagnostics"]["weight_decay"] == .01
                and _matches_recomputed_lr_sum(row["optimizer_diagnostics"]["cumulative_lr_sum"],
                    BATCHES_PER_EPOCH * math.fsum(FrozenEpochScheduler.learning_rate(i) for i in range(row["epoch"] + 1)))
                and 0 <= row["optimizer_diagnostics"]["clip_frequency"] <= 1
                and all(math.isfinite(v) for key in ("group_weight_norms", "group_adam_first_moment_norms") for v in row["optimizer_diagnostics"][key])
                for row in rows), "Coefficient-only optimizer diagnostics")
        declared = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == mode)
        require(plan["state_at_start"]["source_config_identity"] == canonical_fingerprint(declared), "Frozen full-arm declaration")
        prelaunch = load(base / mode / "comparison_readiness_prelaunch.json")
        for field, mismatch in prelaunch["mismatched_fields"].items():
            require(mismatch["candidate"] == identity[field], "Prelaunch intervention binding: " + field)
        fields = {"data_role_fingerprint": "data_role_identity", "row_order_fingerprint": "row_order_identity", "feature_fingerprint": "feature_identity",
            "optimizer_fingerprint": "optimizer_identity", "schedule_fingerprint": "schedule_identity", "target_transform_fingerprint": "target_transform_identity",
            "selection_fingerprint": "checkpoint_selection_identity", "physical_batch_per_device": "physical_batch_per_device",
            "sample_exposure": "sample_presentations", "precision": "precision", "seed": "seed"}
        require(all(observed[k] == identity[v] for k,v in fields.items()), "Observed scientific identity")
        gain = -analysis["candidate_minus_reference_eV"]
        arms[mode] = {"accepted": True, "trajectory_id": arm["trajectory_id"], "run_id": observed["run_id"],
            "parameters": manifest["parameters"], "best_epoch": manifest["best_epoch"], "comparison_identity": identity,
            "prediction_manifest": prediction_manifest, "paired_analysis": analysis,
            "material_gain_eV": gain, "material_gate_eV": config.get("material_gate_eV", .003),
            "material_gate_passed": (gain > config.get("material_gate_eV", .003) if legacy
                else _followup_material_gate(analysis, config["material_gate_eV"])),
            "runtime_certificate": preflight["runtime_certificate"], "completion_manifest_sha256": sha256_file(training / "completion_manifest.json"),
            "epoch_mean_seconds": sum(r["elapsed_seconds"] for r in rows)/EPOCHS,
            "final_live_dev_eV": rows[-1]["live_development_mae_eV"],
            "final_ten_ema_change_eV": rows[-1]["development_mae_eV"]-rows[-11]["development_mae_eV"],
            "matched_trajectory": [{"epoch": i, "candidate_ema_dev_eV": r["ema_dev_metric"], "reference_ema_dev_eV": b["ema_dev_metric"],
                "gain_eV": b["ema_dev_metric"]-r["ema_dev_metric"], "candidate_live_dev_eV": r["live_dev_metric"], "reference_live_dev_eV": b["live_dev_metric"]}
                for i,(r,b) in enumerate(zip(trace["observations"], ref_trace["observations"]))]}
    return {"format": "molgap-gptrans-author-dual-acceptance-v1", "accepted": True, "arms": arms,
        "acceptance_scope": "selected_completed_arms" if selected_arms is not None else "all_arms",
        "all_arms_complete": all(r["complete"] is True for r in summary["outcomes"]),
        "unaccepted_outcomes": [r for r in summary["outcomes"] if r["variant"] not in selected],
        "physical_run_id": receipt["kernel"]+":v1", "source_commit": frozen["source_commit"],
        "source_archive_sha256": source_sha, "spec_identity": spec.identity, "native_cost": cost,
        "model_inference_executed": False, "local_training_executed": False,
        "scientific_caveat": "single seed and reused development role; row bootstrap is not training stochasticity; no scale claim"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _matches_recomputed_lr_sum(observed: float, expected: float) -> bool:
    # Recomputed float sums can differ across Python runtimes. This derived
    # diagnostic gets an eight-ULP bound; recorded schedule/trace identity,
    # counters and the actual weight-decay coefficient remain exact checks.
    return (type(observed) in (int, float) and math.isfinite(observed)
        and math.isfinite(expected) and expected > 0
        and abs(observed - expected) <= 8 * math.ulp(expected))


def _selected_completed_arms(declared, outcomes, selected=None):
    """An ERROR parent may retain a complete child, never an incomplete one."""
    declared = set(declared)
    _require(len(outcomes) == len(declared) and {r["variant"] for r in outcomes} == declared, "Worker outcome identity")
    chosen = declared if selected is None else set(selected)
    _require(bool(chosen) and chosen <= declared, "Selected arm identity")
    _require(all(r["complete"] is True for r in outcomes if r["variant"] in chosen), "Selected arm completion")
    return chosen


def _followup_material_gate(analysis: dict, threshold: float) -> bool:
    """Apply both prospectively frozen criteria, not a point estimate alone."""
    interval = analysis["paired_row_bootstrap"]["ci95"]
    if len(interval) != 2 or not all(math.isfinite(v) for v in interval) or interval[0] > interval[1]:
        raise ValueError("Malformed paired confidence interval")
    return -analysis["candidate_minus_reference_eV"] > threshold and interval[1] < 0


def accept_prepared_inputs(output: Path, staged_package: Path) -> dict:
    """Check retrieved source, complete paths, exact tensor intervention and cost."""
    import torch

    output, staged_package = Path(output), Path(staged_package)
    release = json.loads((staged_package / "release.json").read_text())
    if release["status"] != "LOCAL_RELEASE_INPUTS_VERIFIED" or release["errors"]:
        raise ValueError("CPU source package was not qualified")
    startup = json.loads((output / "startup.json").read_text())
    result = json.loads((output / "preparation_result.json").read_text())
    cost = json.loads((output / "native_cost.json").read_text())
    if (startup["source_archive_sha256"] != release["source_archive_sha256"]
            or startup["source_commit"] != release["source_commit"]
            or startup["manifest_sha256"] != "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
            or startup["initial_state_file_sha256"] != INITIAL_STATE_SHA256):
        raise ValueError("CPU startup does not bind the released input/source identities")
    for record in (startup, result):
        if any(record.get(field) is not False for field in NO_READ):
            raise ValueError("CPU preparation used a forbidden model/label/evaluation role")
    if (cost.get("status") != "COMPLETE" or cost.get("allocated_gpu_count") != 0
            or (output / "failure.json").exists()):
        raise ValueError("CPU preparation was not completely successful")
    for key in ("wall_seconds", "process_cpu_seconds"):
        value = cost.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing/nonfinite CPU cost")
    if cost["wall_seconds"] > 3 * 3600:
        raise ValueError("CPU wall budget exceeded")
    inventory = json.loads((staged_package / "source/SOURCE_FILES.json").read_text())
    source = (output / "verified_source").resolve()
    for item in inventory["files"]:
        member = (source / item["path"]).resolve()
        if not member.is_relative_to(source) or sha256_file(member) != item["sha256"]:
            raise ValueError("Retrieved CPU implementation differs from the release")

    paths = validate_sidecar(output / "paths")
    expected = result["path_acceptance"]
    if expected.get("source_rederived") is not True or paths["runtime_compact_cache"]["within_cap"] is not True:
        raise ValueError("Independent CPU path content/cap verification is absent")
    for key, value in paths.items():
        if key != "source_rederived" and expected.get(key) != value:
            raise ValueError(f"Saved path acceptance changed: {key}")

    base_path = staged_package / "inputs/initial_state.pt"
    if sha256_file(base_path) != INITIAL_STATE_SHA256:
        raise ValueError("Retained base initialization changed")
    base = torch.load(base_path, map_location="cpu", weights_only=True)["model_state"]
    prepared_path = output / "degree_initial_state.pt"
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=True)
    state = prepared["model_state"]
    if (prepared["format"] != INITIAL_SCALED_FORMAT or set(state) != set(base)
            or state_digest(base) != INITIAL_MODEL_SHA256
            or state_digest(state) != DEGREE_INITIAL_SHA256
            or prepared["state_sha256"] != DEGREE_INITIAL_SHA256):
        raise ValueError("Prepared degree initialization has wrong tensor identity")
    for name, tensor in base.items():
        wanted = tensor * DEGREE_SCALE if name in ("in_degree_encoder.weight", "out_degree_encoder.weight") else tensor
        actual = state[name]
        if (actual.dtype != wanted.dtype or actual.shape != wanted.shape
                or not bool(torch.isfinite(actual).all()) or not torch.equal(actual, wanted)):
            raise ValueError(f"Unexpected initial tensor modification: {name}")
    if sha256_file(prepared_path) != result["degree_initialization"]["output_sha256"]:
        raise ValueError("Prepared initialization file changed")
    return {
        "format": "molgap-gptrans-author-inputs-acceptance-v1", "accepted": True,
        "source_commit": release["source_commit"], "source_archive_sha256": release["source_archive_sha256"],
        "package_identity": release["package_identity"], "path_acceptance": paths,
        "cpu_source_rederivation_verified": True,
        "degree_initial_state_sha256": DEGREE_INITIAL_SHA256,
        "degree_initial_file_sha256": sha256_file(prepared_path),
        "native_cost": cost, "gpu_compute_released": False,
        **{field: False for field in NO_READ},
    }
