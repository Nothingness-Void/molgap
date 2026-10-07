"""Four bounded, frozen K1 BN mechanism interventions; no optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

CASES = (("dropout_off_pass1", False, 1), ("dropout_off_pass2", False, 2),
         ("dropout_on_pass1", True, 1), ("dropout_on_pass2", True, 2))


def run(root: Path, output: Path):
    import torch
    import torch_geometric
    from torch_geometric.loader import DataLoader
    from .k1_bn_calibration import recalibrated_batch_norm
    from .k1_frozen_inference import load_native500k_k1, predict_clean
    from .k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from .training_reproducibility import atomic_json, atomic_torch_save, configure_fp32_determinism, sha256_file
    from .v4_runtime import state_dict_sha256

    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + 1200
    root, output = root.resolve(), output.resolve()
    manifest = json.loads((root / "payload_manifest.json").read_text(encoding="utf-8"))
    required = {"selected.pt", "train_probe.pt", "development_probe.pt", "original_prediction.pt",
                "prospective/trajectory.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Required frozen payload file binding missing")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError(f"Payload bytes/path differ: {name}")
    trajectory = json.loads((root / "prospective/trajectory.json").read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    if (output / "result.json").exists() or (output / "completion.json").exists():
        raise FileExistsError("Completed diagnostic must not be overwritten")
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    if torch.__version__.split("+")[0] != "2.4.1":
        raise ValueError("Frozen Torch 2.4.1 required")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("Actual A100 required")
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "python": platform.python_version(), "torch_geometric": torch_geometric.__version__,
               "gpu": torch.cuda.get_device_name(0), "determinism": settings,
               "payload_sha256": sha256_file(root / "payload_manifest.json"),
               "started_at_unix": time.time(), "scientific_training": False,
               "optimizer_created": False, "wall_ceiling_seconds": 1200}
    atomic_json(output / "runtime.json", runtime)

    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError("BN mechanism wall budget exhausted")

    def graphs(name, key, count, bounds):
        guard()
        values = torch.load(root / name, map_location="cpu", weights_only=False)
        observed = [int(g.source_idx) for g in values]
        expected = manifest[key]
        lower, upper = bounds
        if len(values) != count or observed != expected or len(set(observed)) != count or \
                any(not lower <= i < upper for i in observed):
            raise ValueError(f"Frozen role/order differs: {name}")
        if any(any(field in g for field in FORBIDDEN_MODEL_FIELDS) for g in values):
            raise ValueError("Geometry reached pure2D payload")
        if any(not torch.isfinite(g.y).all() for g in values):
            raise ValueError("Nonfinite retained labels")
        return values

    train = graphs("train_probe.pt", "train_source_idx", 16384, (0, 500000))
    development = graphs("development_probe.pt", "development_source_idx", 50000, (500000, 550000))
    if manifest["development_source_idx"] != list(range(500000, 550000)):
        raise ValueError("Expected entire accepted development50K in original order")
    accepted = torch.load(root / "original_prediction.pt", map_location="cpu", weights_only=True)
    original = {"source_idx": accepted["source_idx"].long().view(-1),
                "target_eV": accepted["target"].view(-1),
                "prediction_eV": accepted["prediction"].view(-1)}
    if any(len(v) != 50000 or not torch.isfinite(v).all() for v in original.values()) or \
            original["source_idx"].tolist() != manifest["development_source_idx"]:
        raise ValueError("Accepted prediction identity differs")
    checkpoint = manifest["checkpoint"]
    model, meta = load_native500k_k1(root / "selected.pt",
        expected_sha256=checkpoint["sha256"], expected_source_sha256=checkpoint["source_sha256"],
        expected_epoch=48, checkpoint_kind="selected")
    model.cuda()
    if any(p.dtype != torch.float32 or p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Expected frozen FP32 parameters without gradients")
    parameters_before = state_dict_sha256(dict(model.named_parameters()))
    buffers_before = state_dict_sha256(dict(model.named_buffers()))

    def loader(values):
        return DataLoader(values, batch_size=128, shuffle=False, drop_last=False,
                          num_workers=0, pin_memory=True)

    def predict(values):
        return predict_clean(model, loader(values), mean=meta["mean"], std=meta["std"],
                             device="cuda:0", deadline=deadline)

    def save(name, value):
        guard()
        atomic_torch_save(output / name, value)

    def aligned(prediction):
        if not torch.equal(prediction["source_idx"], original["source_idx"]) or \
                not torch.equal(prediction["target_eV"], original["target_eV"]):
            raise ValueError("Development IDs or targets differ from accepted predictions")

    reconstruction, reconstruction_timing = predict(development)
    aligned(reconstruction)
    error = float((reconstruction["prediction_eV"] - original["prediction_eV"]).abs().max())
    if error > 1e-4:
        raise ValueError(f"Original50K reconstruction exceeds 1e-4 eV: {error}")
    save("original.pt", reconstruction)
    print(f"original50K verified: reconstruction max abs {error:.9g} eV", flush=True)
    reference128 = reconstruction["prediction_eV"][:128].clone()
    baseline_error = (reconstruction["prediction_eV"] - reconstruction["target_eV"]).abs()
    cases = []
    for name, dropout, passes in CASES:
        guard()
        configure_fp32_determinism(42)
        torch.cuda.synchronize()
        case_started = time.perf_counter()
        with recalibrated_batch_norm(model, loader(train), source_idx=manifest["train_source_idx"],
                source_bounds=(0, 500000), device="cuda:0", deadline=deadline,
                dropout_enabled=dropout, passes_per_batch=passes) as calibration:
            bn_buffers = {key: value.detach().cpu().clone() for key, value in model.named_buffers()
                          if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
            if calibration["batches"] != 128 * passes:
                raise ValueError("Expected 128/256 BN counter updates")
            print(f"{name}: calibration complete, {calibration['batches']} BN updates; predicting development50K", flush=True)
            prediction, timing = predict(development)
            aligned(prediction)
            save(f"{name}.pt", prediction)
            save(f"{name}_bn_buffers.pt", bn_buffers)
        torch.cuda.synchronize()
        parameters_ok = state_dict_sha256(dict(model.named_parameters())) == parameters_before
        buffers_ok = state_dict_sha256(dict(model.named_buffers())) == buffers_before
        restored, restoration_timing = predict(development[:128])
        prediction_ok = torch.equal(restored["prediction_eV"], reference128)
        if not parameters_ok or not buffers_ok or not prediction_ok or \
                any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Original frozen state/prediction was not restored exactly")
        improvement = baseline_error - (prediction["prediction_eV"] - original["target_eV"]).abs()
        row = {"case": name, "dropout_enabled": dropout, "passes_per_batch": passes,
               "seed": 42, "calibration": calibration, "inference": timing,
               "restoration_inference": restoration_timing,
               "case_wall_seconds": time.perf_counter() - case_started,
               "parameters_unchanged": parameters_ok, "original_buffers_restored": buffers_ok,
               "original_first128_prediction_restored_exactly": prediction_ok,
               "mae_eV": float((prediction["prediction_eV"] - original["target_eV"]).abs().mean()),
               "paired_mean_improvement_eV": float(improvement.mean()),
               "paired_uncertainty": "pending_local_bootstrap"}
        cases.append(row)
        atomic_json(output / f"{name}.json", row)
        print(json.dumps(row), flush=True)
    report = {"runtime": runtime, "cases": cases, "original_mae_eV": float(baseline_error.mean()),
              "reconstruction_max_abs_eV": error, "reconstruction_timing": reconstruction_timing,
              "parameter_sha256": parameters_before, "original_buffer_sha256": buffers_before,
              "train_labels_read": True, "train_labels_used_for_objective": False,
              "optimizer_created": False, "gradients_computed": False,
              "wall_seconds": time.perf_counter() - started,
              "process_cpu_seconds": time.process_time() - cpu_started}
    guard()
    atomic_json(output / "result.json", report)
    excluded = {"completion.json", "liveworker.log", "worker_process_observation.json"}
    hashes = {path.relative_to(output).as_posix(): sha256_file(path)
              for path in sorted(output.rglob("*")) if path.is_file() and path.name not in excluded}
    guard()
    atomic_json(output / "completion.json", {"complete": True, "files": hashes,
        "payload_sha256": runtime["payload_sha256"], "wall_seconds": time.perf_counter() - started})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
