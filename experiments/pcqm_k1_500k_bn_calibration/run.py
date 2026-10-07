"""Execute the prospectively bound BN-buffer intervention on retained K1."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started, cpu_started = time.perf_counter(), time.process_time()
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    prospective = HERE / "rml/trajectory.json"
    trajectory = json.loads(prospective.read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Expected active prospective before state adaptation")
    result_dir = HERE / "results"
    if result_dir.exists():
        raise FileExistsError("Never overwrite an executed diagnostic")
    frozen = Path(inputs["frozen_source_root"])
    sys.path.insert(0, str(frozen))
    import molgap
    molgap.__path__.append(str(ROOT / "src/molgap"))
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.k1_screen_training import _PackedGraphDatasetFactory
    from molgap.k1_frozen_inference import load_native500k_k1, predict_clean
    from molgap.k1_bn_calibration import recalibrated_batch_norm
    from molgap.pcqm_k1_scale_runner import find_cache
    from molgap.router import paired_bootstrap_mean
    from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file, configure_fp32_determinism
    from molgap.v4_runtime import state_dict_sha256

    for name, digest in inputs["frozen_source_files"].items():
        if sha256_file(frozen.parent / name) != digest:
            raise ValueError("Accepted model source differs")
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Diagnostic source differs from prospective")
    if Path(sys.modules["molgap.qm9_neural_atom"].__file__).resolve() != (frozen / "molgap/qm9_neural_atom.py").resolve():
        raise ValueError("Wrong family source imported")
    settings = configure_fp32_determinism(inputs["sample_seed"])
    torch.set_num_threads(inputs["cpu_threads"])
    deadline = started + inputs["worker_wall_ceiling_seconds"]
    os.environ["MOLGAP_PCQM_500K_V4_ROOT"] = inputs["cache_root"]
    cache, manifest = find_cache(inputs["manifest_sha256"])
    calibration_indices = np.random.default_rng(inputs["sample_seed"]).choice(
        500000, inputs["calibration_rows"], replace=False).astype(np.int64)
    ordered = np.sort(calibration_indices)
    calibration_graphs = {}
    development = None
    offset = 0
    for item in manifest["geometry_shards"]:
        if time.perf_counter() >= deadline:
            raise TimeoutError("Graph load exhausted diagnostic ceiling")
        path = cache / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Accepted graph shard differs")
        count = item["rows"]
        data = _PackedGraphDatasetFactory.load(path)
        if len(data) != count or not torch.equal(data._data.source_idx.view(-1).long(), torch.arange(offset, offset + count)):
            raise ValueError("Accepted graph source identity differs")
        if offset < 500000:
            wanted = ordered[(ordered >= offset) & (ordered < offset + count)]
            calibration_graphs.update({int(index): data[int(index - offset)].clone() for index in wanted})
        else:
            if development is not None or offset != 500000 or count != 50000:
                raise ValueError("Development shard differs")
            development = data
        offset += count
        if offset <= 500000:
            del data
    if offset != 550000 or development is None or len(calibration_graphs) != len(calibration_indices):
        raise ValueError("Bound calibration/development membership differs")
    calibration = [calibration_graphs[int(index)] for index in calibration_indices]
    bound = inputs["checkpoints"]["best_model.pt"]
    model, metadata = load_native500k_k1(Path(bound["path"]), expected_sha256=bound["sha256"],
        expected_source_sha256="0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a",
        expected_epoch=48, checkpoint_kind="selected")
    state_before = state_dict_sha256(model.state_dict())
    params_before = state_dict_sha256(dict(model.named_parameters()))
    loader = DataLoader(development, batch_size=inputs["batch_size"], shuffle=False, num_workers=0)
    original, baseline_timing = predict_clean(model, loader, mean=metadata["mean"], std=metadata["std"], device="cpu", deadline=deadline)
    if not torch.equal(original["source_idx"], torch.arange(500000, 550000)):
        raise ValueError("Development rows differ")
    saved_bound = inputs["checkpoints"]["best_predictions.pt"]
    if sha256_file(Path(saved_bound["path"])) != saved_bound["sha256"]:
        raise ValueError("Accepted prediction bytes differ")
    saved = torch.load(saved_bound["path"], map_location="cpu", weights_only=True)
    if not torch.equal(original["source_idx"], saved["source_idx"].view(-1).long()) or not torch.equal(original["target_eV"], saved["target"].view(-1)):
        raise ValueError("Saved row/target alignment differs")
    reconstruction = float((original["prediction_eV"] - saved["prediction"].view(-1)).abs().max())
    if reconstruction > inputs["prediction_tolerance_eV"]:
        raise ValueError("Accepted predictions do not reconstruct")
    result_dir.mkdir()
    atomic_torch_save(result_dir / "original.pt", original)
    print(json.dumps({"phase": "baseline_verified", "rows": 50000, "wall_seconds": time.perf_counter() - started}), flush=True)
    calibration_loader = DataLoader(calibration, batch_size=inputs["batch_size"], shuffle=False, num_workers=0)
    with recalibrated_batch_norm(model, calibration_loader, source_idx=calibration_indices,
            source_bounds=(0, 500000), device="cpu", deadline=deadline) as calibration_report:
        print(json.dumps({"phase": "bn_calibrated", "rows": calibration_report["rows"], "wall_seconds": time.perf_counter() - started}), flush=True)
        calibrated, calibrated_timing = predict_clean(model, loader, mean=metadata["mean"], std=metadata["std"], device="cpu", deadline=deadline)
        atomic_torch_save(result_dir / "calibrated.pt", calibrated)
        atomic_torch_save(result_dir / "calibrated_buffers.pt", {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()})
        calibrated_state = state_dict_sha256(model.state_dict())
    control_loader = DataLoader([development[index] for index in range(inputs["batch_size"])],
        batch_size=inputs["batch_size"], shuffle=False, num_workers=0)
    restored, restored_timing = predict_clean(model, control_loader, mean=metadata["mean"], std=metadata["std"], device="cpu", deadline=deadline)
    restoration_exact = torch.equal(restored["prediction_eV"], original["prediction_eV"][:inputs["batch_size"]])
    if state_dict_sha256(model.state_dict()) != state_before or not restoration_exact:
        raise ValueError("Original model state/inference not restored")
    if not torch.equal(calibrated["source_idx"], original["source_idx"]) or not torch.equal(calibrated["target_eV"], original["target_eV"]):
        raise ValueError("Intervention rows/targets differ")
    target = original["target_eV"].numpy().astype(np.float64)
    before = original["prediction_eV"].numpy().astype(np.float64)
    after = calibrated["prediction_eV"].numpy().astype(np.float64)
    error_before, error_after = np.abs(before - target), np.abs(after - target)
    paired = paired_bootstrap_mean(error_before - error_after, n_bootstrap=1000, seed=inputs["sample_seed"])
    if time.perf_counter() >= deadline:
        raise TimeoutError("Analysis exhausted diagnostic ceiling")
    metrics = {"original_mae_eV": float(error_before.mean()), "calibrated_mae_eV": float(error_after.mean()),
        "gain_eV": paired, "nomination_passed": paired["delta"] >= .001 and paired["ci95"][0] > 0,
        "mean_abs_prediction_shift_eV": float(np.abs(after - before).mean()),
        "mean_signed_error_before_eV": float((before - target).mean()),
        "mean_signed_error_after_eV": float((after - target).mean()),
        "error_quantiles": {name: {str(q): float(np.quantile(error, q)) for q in [.5, .9, .95, .99]}
            for name, error in [("original", error_before), ("calibrated", error_after)]}}
    role_hash = lambda rows: hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()
    report = {"status": "complete", "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": sha256_file(HERE / "inputs.json"), "prospective_sha256": sha256_file(prospective),
        "sample_source_idx_sha256": {"calibration": role_hash(calibration_indices), "development": role_hash(np.arange(500000, 550000))},
        "decoded_role_manifest_hashes": {"train_prefix": role_hash(np.arange(500000)), "internal_development": role_hash(np.arange(500000, 550000))},
        "max_selected_prediction_reconstruction_eV": reconstruction,
        "metrics": metrics, "calibration": calibration_report, "metadata": metadata,
        "state_sha256": {"original": state_before, "calibrated": calibrated_state, "parameters": params_before},
        "runtime": {"torch": torch.__version__, "settings": settings, "cpu_threads": inputs["cpu_threads"]},
        "cost": {"worker_wall_seconds": time.perf_counter() - started, "worker_process_cpu_seconds": time.process_time() - cpu_started,
            "inference_timings": {"original": baseline_timing, "calibrated": calibrated_timing, "restored_control": restored_timing},
            "accelerator": "not_applicable"},
        "checks": {"strict_state_loading": True, "source_and_cache_hashes": True, "selected_prediction_reconstruction": True,
            "exact_rows_targets": True, "finite_predictions": True, "training_only_calibration": True,
            "parameters_unchanged": params_before == state_dict_sha256(dict(model.named_parameters())),
            "non_bn_buffers_unchanged": calibration_report["non_bn_buffers_unchanged"],
            "buffers_restored": calibration_report["buffers_restored"], "restored_inference_exact": restoration_exact,
            "no_optimization": True, "protected_roles_untouched": True},
        "limits": ["Consumed development used for prior checkpoint selection; row CI is not seed variance or independent generalization",
            "One fixed clean-feature calibration sample/mode; not a causal test of double-forward BN update harm",
            "NO_TRAIN is parameter-free BN state adaptation, not training replay readiness or model adoption"]}
    if not all(report["checks"].values()):
        raise ValueError("Diagnostic failed a required check")
    atomic_json(result_dir / "analysis.json", report)
    print(json.dumps({"status": "complete", "metrics": metrics, "cost": report["cost"]}), flush=True)


if __name__ == "__main__":
    main()
