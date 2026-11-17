"""Fixed-endpoint clean fit, using existing frozen K1 owners only."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import time


@contextmanager
def retained_buffers(model, buffers, *, expected_sha256):
    import torch
    from .v4_runtime import state_dict_sha256
    current = dict(model.named_buffers())
    original = {name: value.detach().clone() for name, value in current.items()}
    parameters = state_dict_sha256(dict(model.named_parameters()))
    bn_keys = {f"{name}.{key}" if name else key for name, module in model.named_modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for key in ("running_mean", "running_var", "num_batches_tracked")}
    if set(buffers) != set(current) or state_dict_sha256(buffers) != expected_sha256 or any(
            buffers[key].shape != value.shape or buffers[key].dtype != value.dtype or
            not torch.isfinite(buffers[key]).all() for key, value in current.items()) or any(
            not torch.equal(buffers[key], original[key]) for key in current if key not in bn_keys):
        raise ValueError("Retained calibrated buffers are not the matching BN-only state")
    try:
        with torch.no_grad():
            for name, value in current.items():
                value.copy_(buffers[name])
        yield
        if parameters != state_dict_sha256(dict(model.named_parameters())):
            raise ValueError("Retained-buffer inference changed parameters")
    finally:
        with torch.no_grad():
            for name, value in current.items():
                value.copy_(original[name])


def validate_prediction(record, indices, *, target=None):
    import torch
    if set(record) != {"source_idx", "target_eV", "prediction_eV"} or any(
            value.ndim != 1 or len(value) != len(indices) or not torch.isfinite(value).all()
            for value in record.values()) or not torch.equal(record["source_idx"], indices) or \
            (target is not None and not torch.equal(record["target_eV"], target)):
        raise ValueError("Prediction identity/shape/targets differ")


def run(experiment, output):
    # No numerical libraries/model owners before masking and frozen source switch.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started, cpu_started = time.perf_counter(), time.process_time()
    experiment, output = Path(experiment), Path(output)
    inputs = json.loads((experiment / "inputs.json").read_text(encoding="utf-8"))
    trajectory = json.loads((experiment / "rml/trajectory.json").read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required before inference")
    from . import k1_bn_diagnostic as owner
    owner._frozen_imports(inputs)
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from .training_reproducibility import sha256_file, atomic_json, atomic_torch_save
    from .v4_runtime import state_dict_sha256
    from .k1_frozen_inference import load_native500k_k1, predict_clean
    from .k1_bn_calibration import recalibrated_batch_norm
    if trajectory["state_at_start"]["source_config_identity"] != sha256_file(experiment / "inputs.json"):
        raise ValueError("Prospective input identity differs")
    owner._bound(inputs["source_archive"])
    owner._bound(inputs["source_inventory"])
    prior = owner._json(owner._bound(inputs["prior_pair_report"]))
    if prior["status"] != "complete" or any(inputs["arms"][arm]["prior_report"] != prior["arms"][arm] for arm in owner.ARMS):
        raise ValueError("Reused selected evidence provenance differs")
    deadline = started + inputs["ceiling_seconds"]
    runtime = owner._cpu_runtime()
    if runtime["torch"] != inputs["prior_runtime_torch"] or runtime["installed_distributions_sha256"] != inputs["prior_software_sha256"]:
        raise ValueError("Reused development predictions require identical CPU software")
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(experiment.parents[1] / name) != digest:
            raise ValueError("Executed diagnostic source differs")
    inventory = owner._json(inputs["source_inventory"]["path"])
    for entry in inventory["files"]:
        if entry["path"].startswith("src/molgap/"):
            from .v4_runtime import normalized_source_sha256
            if normalized_source_sha256(Path(inputs["frozen_source_root"]) / entry["path"]) != entry["sha256"]:
                raise ValueError("Frozen dependency differs")
    output.mkdir(exist_ok=False)
    report = {"status": "running", "started_at_utc": owner._utc(), "cases": {}, "runtime": {
        "device": "cpu", "cpu_threads": torch.get_num_threads(), "software": runtime}}
    def progress(phase):
        report.update(phase=phase, updated_at_utc=owner._utc(), wall_seconds=time.perf_counter() - started,
                      process_cpu_seconds=time.process_time() - cpu_started)
        atomic_json(output / "progress.json", report)
    def bound(item):
        return owner._bound(item)
    indices = np.random.default_rng(20261008).choice(500000, 16384, replace=False)
    train, dev = owner._load_graphs(inputs, indices, deadline)
    owner._verify_frozen_modules(inputs)
    loaders = {"train": DataLoader(train, batch_size=128, shuffle=False, num_workers=0),
               "development": DataLoader(dev, batch_size=128, shuffle=False, num_workers=0)}
    ids = {"train": torch.as_tensor(indices), "development": torch.arange(500000, 550000)}
    targets = {}
    def predict(model, metadata, case, role):
        value, timing = predict_clean(model, loaders[role], mean=metadata["mean"], std=metadata["std"],
                                      device="cpu", deadline=deadline)
        owner._deadline(deadline)
        validate_prediction(value, ids[role], target=targets.get(role))
        targets.setdefault(role, value["target_eV"])
        atomic_torch_save(output / f"{case}_{role}.pt", value)
        report["cases"].setdefault(case, {})[role] = timing
        progress(f"{case}_{role}")
    progress("graphs_bound")
    for arm in owner.ARMS:
        selected_metadata = None
        for epoch, kind in ((48, "selected"), (59, "final")):
            item = inputs["arms"][arm][kind]
            model, metadata = load_native500k_k1(bound(item), expected_sha256=item["sha256"],
                expected_source_sha256=inputs["source_archive"]["sha256"], expected_epoch=epoch,
                checkpoint_kind=kind, expected_arm=arm)
            owner._verify_frozen_modules(inputs)
            if selected_metadata is not None and any(metadata[key] != selected_metadata[key] for key in ("mean", "std", "contract")):
                raise ValueError("Endpoint normalization/contract differs")
            selected_metadata = metadata
            before = state_dict_sha256(model.state_dict())
            case = f"{arm}_epoch{epoch + 1}"
            if kind == "selected":
                previous = inputs["arms"][arm]["prior_report"]
                if state_dict_sha256(dict(model.named_parameters())) != previous["parameter_sha256_before"] or \
                        state_dict_sha256(dict(model.named_buffers())) != previous["buffer_sha256_before"]:
                    raise ValueError("Selected model differs from reused BN state")
                for state in ("original", "calibrated"):
                    value = torch.load(bound(inputs["arms"][arm][state]), map_location="cpu", weights_only=True)
                    validate_prediction(value, ids["development"], target=targets.get("development"))
                    targets.setdefault("development", value["target_eV"])
                    atomic_torch_save(output / f"{case}_{state}_development.pt", value)
                predict(model, metadata, case + "_original", "train")
                buffers = torch.load(bound(inputs["arms"][arm]["buffers"]), map_location="cpu", weights_only=True)
                with retained_buffers(model, buffers, expected_sha256=previous["calibration"]["buffer_sha256_calibrated"]):
                    predict(model, metadata, case + "_calibrated", "train")
            else:
                for role in loaders:
                    predict(model, metadata, case + "_original", role)
                with recalibrated_batch_norm(model, loaders["train"], source_idx=indices,
                        source_bounds=(0, 500000), device="cpu", deadline=deadline) as calibration:
                    atomic_torch_save(output / f"{case}_calibrated_buffers.pt", {
                        name: value.detach().clone() for name, value in model.named_buffers()})
                    for role in loaders:
                        predict(model, metadata, case + "_calibrated", role)
                report["cases"][case + "_calibrated"]["calibration"] = calibration
            if state_dict_sha256(model.state_dict()) != before:
                raise ValueError("Endpoint state not restored")
            del model
    owner._deadline(deadline)
    report.update(format="molgap-k1-clean-fit-diagnostic-v1", status="complete", phase="complete",
        inputs_sha256=sha256_file(experiment / "inputs.json"), prospective_sha256=sha256_file(experiment / "rml/trajectory.json"),
        training_executed=False, gradients_computed=False, optimizer_created=False,
        checks={"source_and_artifact_hashes": True, "exact_rows_targets": True, "finite_predictions": True,
                "matching_endpoint_buffers": True, "state_restoration": True, "protected_roles_untouched": True},
        calibration_source_idx_sha256=state_dict_sha256({"source_idx": ids["train"]}),
        development_source_idx_sha256=state_dict_sha256({"source_idx": ids["development"]}), ended_at_utc=owner._utc())
    progress("complete")
    atomic_json(output / "result.json", report)
    atomic_json(output / "completion.json", {"complete": True, "inputs_sha256": report["inputs_sha256"],
        "files": {p.name: sha256_file(p) for p in output.iterdir() if p.suffix in (".pt", ".json")}})
    return report


def analyze(experiment):
    import torch
    from .router import paired_bootstrap_mean
    from .training_reproducibility import atomic_json, sha256_file
    experiment = Path(experiment)
    output = experiment / "results"
    started, cpu_started = time.perf_counter(), time.process_time()
    completion = json.loads((output / "completion.json").read_text())
    for name, digest in completion["files"].items():
        if sha256_file(output / name) != digest:
            raise ValueError("Saved numerical output changed")
    errors, metrics = {}, {}
    for arm in ("k1_pretrained_mean2", "k1_pretrained_consistency"):
        for epoch in (49, 60):
            for state in ("original", "calibrated"):
                case = f"{arm}_epoch{epoch}_{state}"
                metrics[case] = {}
                for role in ("train", "development"):
                    value = torch.load(output / f"{case}_{role}.pt", map_location="cpu", weights_only=True)
                    error = (value["prediction_eV"].double() - value["target_eV"].double()).abs().numpy()
                    errors[(case, role)] = error
                    metrics[case][role + "_mae_eV"] = float(error.mean())
                metrics[case]["dev_minus_train_gap_eV"] = metrics[case]["development_mae_eV"] - metrics[case]["train_mae_eV"]
    contrasts = {}
    for role in ("train", "development"):
        pairs = [(f"epoch_change_{arm}_{state}", f"{arm}_epoch49_{state}", f"{arm}_epoch60_{state}")
                 for arm in ("k1_pretrained_mean2", "k1_pretrained_consistency") for state in ("original", "calibrated")]
        pairs += [(f"arm_change_epoch{epoch}_{state}", f"k1_pretrained_mean2_epoch{epoch}_{state}", f"k1_pretrained_consistency_epoch{epoch}_{state}")
                  for epoch in (49, 60) for state in ("original", "calibrated")]
        pairs += [(f"bn_change_{arm}_epoch{epoch}", f"{arm}_epoch{epoch}_original", f"{arm}_epoch{epoch}_calibrated")
                  for arm in ("k1_pretrained_mean2", "k1_pretrained_consistency") for epoch in (49, 60)]
        for label, first, second in pairs:
            contrasts[f"{role}_{label}"] = {"first": first, "second": second,
                "error_change_second_minus_first_eV": paired_bootstrap_mean(errors[(second, role)] - errors[(first, role)], n_bootstrap=1000, seed=42)}
    result = {"metrics": metrics, "contrasts": contrasts, "sign": "positive error change means second is worse",
              "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
              "limits": ["Train sample is calibration in-sample", "Development selection consumed", "No causal underfit proof", "Row CI is not seed variation"]}
    atomic_json(experiment / "analysis.json", result)
    return result
