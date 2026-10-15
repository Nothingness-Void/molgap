"""Bounded same-checkpoint K1 state attribution; no optimizer or training."""
from pathlib import Path
import json
import os
import time


def _bound(item):
    from .training_reproducibility import sha256_file
    path = Path(item["path"])
    if not path.is_absolute() or sha256_file(path) != item["sha256"]:
        raise ValueError("Bound diagnostic input differs: " + str(path))
    return path


def run(experiment, output):
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[key] = "-1"
    started, cpu_started = time.perf_counter(), time.process_time()
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    repo = next(parent for parent in experiment.parents if (parent / "AGENTS.md").is_file())
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    inputs, trajectory = read(experiment / "inputs.json"), read(experiment / "rml/trajectory.json")
    from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file
    if (trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE" or
            trajectory["state_at_start"]["source_config_identity"] != sha256_file(experiment / "inputs.json")):
        raise ValueError("Matching active prospective required before inference")
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(repo / name) != digest:
            raise ValueError("Executed diagnostic source differs: " + name)
    from .v4_runtime import normalized_source_sha256, state_dict_sha256
    pins = {item["path"]: item["sha256"] for item in read(_bound(inputs["source_inventory"]))["files"]}
    for name in inputs["model_source_files"]:
        if normalized_source_sha256(repo / name) != pins[name]:
            raise ValueError("Producer model source differs: " + name)
    _bound(inputs["source_archive"])
    from .k1_bn_diagnostic import _cpu_runtime
    runtime = _cpu_runtime()
    import numpy as np
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from .family_training_data import DatasetSelection, load_training_data
    from .family_training_state import FamilyTrainingState
    from .gptrans_execution_hotpath import BoundForeachEMA
    from .k1_frozen_inference import read_session_k1_states, make_encoder, predict_clean
    from .k1_bn_calibration import recalibrated_batch_norm
    from .k1_clean_fit_diagnostic import validate_prediction
    from .router import paired_bootstrap_mean

    deadline = started + inputs["ceiling_seconds"]
    output.mkdir(exist_ok=False)
    report = {"format": "molgap-k1-session-state-diagnostic-v1", "status": "running", "cases": {},
              "runtime": {"device": "cpu", "cpu_threads": torch.get_num_threads(), "software": runtime}}
    def progress(phase):
        report.update(phase=phase, wall_seconds=time.perf_counter() - started,
                      process_cpu_seconds=time.process_time() - cpu_started)
        atomic_json(output / "progress.json", report)
    def check_time():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Same-checkpoint diagnostic exhausted wall budget")
    progress("loading_hash_bound_data")
    data = load_training_data(Path(inputs["cache_root"]), DatasetSelection(**inputs["dataset"]))
    if (data.train_rows, data.development_rows, data.train_source_start, data.development_source_start) != (500000, 50000, 0, 500000):
        raise ValueError("Diagnostic role boundaries differ")
    check_time()
    ids = {"train": torch.tensor(inputs["train_source_idx"], dtype=torch.int64),
           "development": torch.tensor(inputs["development_source_idx"], dtype=torch.int64)}
    calibration_ids = {key: np.asarray(values) for key, values in inputs["calibration_source_idx"].items()}
    if any(set(ids["train"].tolist()) & set(values.tolist()) for values in calibration_ids.values()):
        raise ValueError("Descriptive train sample overlaps calibration")
    loaders = {"train": DataLoader(Subset(data.train, ids["train"].tolist()), batch_size=128, shuffle=False),
               "development": DataLoader(data.development, batch_size=128, shuffle=False)}
    calibration_loaders = {key: DataLoader(Subset(data.train, values.tolist()), batch_size=128, shuffle=False)
                           for key, values in calibration_ids.items()}
    targets, errors = {}, {}
    retained = torch.load(_bound(inputs["retained_predictions"]), map_location="cpu", weights_only=True)
    retained = {key: retained[key] for key in ("source_idx", "target_eV", "prediction_eV")}
    validate_prediction(retained, ids["development"])
    targets["development"] = retained["target_eV"]
    def predict(model, recipe, case):
        before = state_dict_sha256(model.state_dict())
        report["cases"][case] = {"state_sha256": before}
        for role, loader in loaders.items():
            value, timing = predict_clean(model, loader, mean=recipe.target_mean_eV, std=recipe.target_std_eV,
                                         device="cpu", deadline=deadline)
            validate_prediction(value, ids[role], target=targets.get(role))
            targets.setdefault(role, value["target_eV"])
            error = (value["prediction_eV"].double() - value["target_eV"].double()).abs().numpy()
            report["cases"][case][role] = {"mae_eV": float(error.mean()), **timing}
            errors[(case, role)] = error
            atomic_torch_save(output / f"{case}_{role}.pt", value)
            if case == "selected_clean" and role == "development":
                maximum = float((value["prediction_eV"] - retained["prediction_eV"]).abs().max())
                report["native_prediction_max_abs_diff_eV"] = maximum
                if maximum > inputs["prediction_tolerance_eV"]:
                    raise ValueError("CPU replay differs from retained native Clean predictions")
            check_time()
        if state_dict_sha256(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen inference changed state or gradients")
        progress(case)
    for endpoint in ("selected", "final"):
        item = inputs["checkpoints"][endpoint]
        state, recipe = read_session_k1_states(_bound(item["artifact"]), expected_sha256=item["artifact"]["sha256"],
            expected_context=inputs["context"], recipe_path=_bound(inputs["recipe"]),
            expected_step=item["step"], checkpoint_kind=endpoint)
        if abs(data.target_mean_eV - recipe.target_mean_eV) > 1e-7 or abs(data.target_std_eV - recipe.target_std_eV) > 1e-7:
            raise ValueError("Fixed-data target transform differs")
        factory = lambda: make_encoder("neural_atom_k1").eval().requires_grad_(False)
        model = factory()
        if sum(p.numel() for p in model.parameters()) != 3658817:
            raise ValueError("K1 factory parameter identity differs")
        owner = FamilyTrainingState(model, BoundForeachEMA(model, recipe.ema_decay), clean_factory=factory)
        owner.load_state_dict(state, calibration_policy=recipe.clean_bn)
        if owner.clean_status != "ready":
            raise ValueError("Retained Clean state is not ready")
        # Reconstruct the native selected output before interpreting new views.
        for name, view in (("clean", "clean-ema"), ("live", "live"), ("raw", "raw-ema")):
            owner.evaluate(view, lambda current: predict(current, recipe, endpoint + "_" + name),
                           calibration_policy=recipe.clean_bn if name == "clean" else None)
        if endpoint == "selected":
            for weight, view, calibration in inputs["additional_calibrations"]:
                def calibrated(current):
                    with recalibrated_batch_norm(current, calibration_loaders[calibration],
                            source_idx=calibration_ids[calibration], source_bounds=(0, 500000),
                            device="cpu", deadline=deadline) as receipt:
                        case = f"selected_{weight}_clean_{calibration}"
                        predict(current, recipe, case)
                        report["cases"][case]["calibration"] = receipt
                owner.evaluate(view, calibrated)
        del owner, model, state
    pairs = [("selected_live", "selected_raw"), ("selected_raw", "selected_clean"),
             ("selected_live", "selected_clean"), ("selected_clean", "final_clean"),
             ("selected_live", "final_live"), ("selected_raw", "final_raw"),
             ("selected_clean", "selected_ema_clean_uniform"),
             ("selected_live", "selected_live_clean_prefix"),
             ("selected_live_clean_prefix", "selected_live_clean_uniform"),
             ("selected_live_clean_uniform", "selected_ema_clean_uniform")]
    contrasts = {f"{role}:{first}->{second}": paired_bootstrap_mean(
        errors[(second, role)] - errors[(first, role)], n_bootstrap=1000, seed=42)
        for role in loaders for first, second in pairs}
    report.update(status="complete", phase="complete", contrasts=contrasts,
        sign="positive paired error change means second is worse; row CI is not seed variance",
        inputs_sha256=sha256_file(experiment / "inputs.json"), prospective_sha256=sha256_file(experiment / "rml/trajectory.json"),
        training_executed=False, gradients_computed=False, optimizer_created=False,
        checks={"hash_bound_source_inputs": True, "exact_rows_targets": True, "finite_predictions": True,
                "native_clean_reconstruction": True, "matching_endpoint_states": True,
                "state_restoration": True, "protected_roles_untouched": True})
    progress("complete")
    check_time()
    atomic_json(output / "result.json", report)
    atomic_json(output / "completion.json", {"complete": True, "inputs_sha256": report["inputs_sha256"],
        "files": {p.name: sha256_file(p) for p in output.iterdir() if p.suffix in (".pt", ".json")}})
    return report
