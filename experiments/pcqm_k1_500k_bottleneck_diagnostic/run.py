"""Execute the bounded plan through frozen family and shared inference owners."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    prospective = HERE / "rml/trajectory.json"
    trajectory = json.loads(prospective.read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Expected active prospective before inference")
    if (HERE / "results").exists():
        raise FileExistsError("Never overwrite an executed diagnostic")
    frozen = Path(inputs["frozen_source_root"])
    sys.path.insert(0, str(frozen))
    import molgap
    molgap.__path__.append(str(ROOT / "src/molgap"))
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.k1_screen_training import _PackedGraphDatasetFactory
    from molgap.k1_frozen_inference import load_native500k_k1, predict_clean, scale_slot_return
    from molgap.pcqm_k1_scale_runner import find_cache
    from molgap.router import paired_bootstrap_mean
    from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file, configure_fp32_determinism
    from molgap.v4_runtime import state_dict_sha256
    import os

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
    rng = np.random.default_rng(inputs["sample_seed"])
    sample = np.sort(np.concatenate([
        rng.choice(100000, inputs["train_prefix_rows"], replace=False),
        rng.choice(400000, inputs["train_extension_rows"], replace=False) + 100000,
        rng.choice(50000, inputs["development_rows"], replace=False) + 500000])).astype(np.int64)
    graphs = []
    offset = 0
    for item in manifest["geometry_shards"]:
        if time.perf_counter() >= deadline:
            raise TimeoutError("Graph load exhausted diagnostic ceiling")
        path = cache / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Accepted graph shard differs")
        count = item["rows"]
        wanted = sample[(sample >= offset) & (sample < offset + count)]
        data = _PackedGraphDatasetFactory.load(path)
        if len(data) != count or not torch.equal(data._data.source_idx.view(-1).long(), torch.arange(offset, offset + count)):
            raise ValueError("Accepted graph source identity differs")
        graphs.extend(data[int(index - offset)].clone() for index in wanted)
        del data
        offset += count
    if offset != 550000 or len(graphs) != len(sample):
        raise ValueError("Bounded sample membership differs")
    result_dir = HERE / "results"
    result_dir.mkdir()
    graph_load_wall = time.perf_counter() - started
    checkpoints = inputs["checkpoints"]
    arrays, timings, metadata = {}, {}, {}
    for name, filename, epoch, kind, source in [
        ("selected", "best_model.pt", 48, "selected", "0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a"),
        ("final", "last_checkpoint.pt", 59, "final", inputs["archive"]["sha256"]),
    ]:
        bound = checkpoints[filename]
        model, info = load_native500k_k1(Path(bound["path"]), expected_sha256=bound["sha256"],
            expected_source_sha256=source, expected_epoch=epoch, checkpoint_kind=kind)
        before = state_dict_sha256(model.state_dict())
        loader = DataLoader(graphs, batch_size=inputs["batch_size"], shuffle=False, num_workers=0)
        output, timing = predict_clean(model, loader, mean=info["mean"], std=info["std"], device="cpu", deadline=deadline)
        if not np.array_equal(output["source_idx"].numpy(), sample):
            raise ValueError("Inference sample order differs")
        arrays[name], timings[name], metadata[name] = output, timing, info
        atomic_torch_save(result_dir / f"{name}.pt", output)
        if name == "selected":
            saved_bound = checkpoints["best_predictions.pt"]
            if sha256_file(Path(saved_bound["path"])) != saved_bound["sha256"]:
                raise ValueError("Accepted prediction bytes differ")
            saved = torch.load(saved_bound["path"], map_location="cpu", weights_only=True)
            if not torch.equal(saved["source_idx"].view(-1).long(), torch.arange(500000, 550000)):
                raise ValueError("Accepted development prediction rows differ")
            dev = sample >= 500000
            offsets = sample[dev] - 500000
            if not torch.equal(output["target_eV"][dev], saved["target"].view(-1)[offsets]):
                raise ValueError("Development targets differ")
            max_reconstruction = float((output["prediction_eV"][dev] - saved["prediction"].view(-1)[offsets]).abs().max())
            if max_reconstruction > inputs["prediction_tolerance_eV"]:
                raise ValueError("Saved selected predictions do not reconstruct")
            dev_graphs = [graph for graph in graphs if int(graph.source_idx) >= 500000]
            loader = DataLoader(dev_graphs, batch_size=inputs["batch_size"], shuffle=False, num_workers=0)
            for intervention, layers in inputs["interventions"].items():
                with scale_slot_return(model, layers=tuple(layers), scale=0.0):
                    rows, timing = predict_clean(model, loader, mean=info["mean"], std=info["std"], device="cpu", deadline=deadline)
                arrays[intervention], timings[intervention] = rows, timing
                atomic_torch_save(result_dir / f"{intervention}.pt", rows)
            control_graphs = dev_graphs[:inputs["batch_size"]]
            control_loader = DataLoader(control_graphs, batch_size=inputs["batch_size"], shuffle=False, num_workers=0)
            with scale_slot_return(model, layers=(3, 6, 9), scale=1.0):
                control, timing = predict_clean(model, control_loader, mean=info["mean"], std=info["std"], device="cpu", deadline=deadline)
            if not torch.equal(control["prediction_eV"], output["prediction_eV"][dev][:len(control_graphs)]):
                raise ValueError("Identity control or hook restoration differs")
            timings["identity_control"] = timing
        if before != state_dict_sha256(model.state_dict()):
            raise ValueError("Frozen parameter/buffer state changed")
        del model

    if metadata["selected"]["mean"] != metadata["final"]["mean"] or metadata["selected"]["std"] != metadata["final"]["std"]:
        raise ValueError("Selected/final target transforms differ")
    if not torch.equal(arrays["selected"]["target_eV"], arrays["final"]["target_eV"]):
        raise ValueError("Selected/final targets differ")
    target = arrays["selected"]["target_eV"].numpy().astype(np.float64)
    best = arrays["selected"]["prediction_eV"].numpy().astype(np.float64)
    last = arrays["final"]["prediction_eV"].numpy().astype(np.float64)
    groups = {"train_100k_prefix": sample < 100000, "train_extension": (sample >= 100000) & (sample < 500000),
              "train_combined": sample < 500000, "development": sample >= 500000}
    summaries = {}
    for name, mask in groups.items():
        original_error = np.abs(best[mask] - target[mask])
        final_error = np.abs(last[mask] - target[mask])
        summaries[name] = {"rows": int(mask.sum()), "selected_mae_eV": float(original_error.mean()),
            "final_mae_eV": float(final_error.mean()), "final_minus_selected": paired_bootstrap_mean(
                final_error - original_error, n_bootstrap=5000, seed=inputs["sample_seed"])}
    effects = {}
    for name in inputs["interventions"]:
        rows = arrays[name]
        if not torch.equal(rows["target_eV"], arrays["selected"]["target_eV"][groups["development"]]) or not torch.equal(rows["source_idx"], arrays["selected"]["source_idx"][groups["development"]]):
            raise ValueError("Intervention rows/targets differ")
        prediction = rows["prediction_eV"].numpy().astype(np.float64)
        base = best[groups["development"]]
        labels = target[groups["development"]]
        effects[name] = {"mae_eV": float(np.abs(prediction - labels).mean()),
            "ablation_minus_selected": paired_bootstrap_mean(np.abs(prediction - labels) - np.abs(base - labels),
                n_bootstrap=5000, seed=inputs["sample_seed"]),
            "mean_abs_prediction_shift_eV": float(np.abs(prediction - base).mean())}
    report = {"status": "complete", "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": sha256_file(HERE / "inputs.json"), "prospective_sha256": sha256_file(prospective),
        "sample_source_idx_sha256": {name: __import__("hashlib").sha256(sample[mask].astype("<i8").tobytes()).hexdigest() for name, mask in groups.items()},
        "max_selected_prediction_reconstruction_eV": max_reconstruction,
        "groups": summaries, "slot_interventions": effects, "metadata": metadata,
        "runtime": {"torch": torch.__version__, "settings": settings, "cpu_threads": inputs["cpu_threads"]},
        "cost": {"worker_wall_seconds": time.perf_counter() - started,
            "worker_process_cpu_seconds": time.process_time() - cpu_started,
            "loading_and_startup_wall_seconds": graph_load_wall, "inference_timings": timings,
            "accelerator": "not_applicable"},
        "checks": {"strict_state_loading": True, "source_and_cache_hashes": True,
            "selected_prediction_reconstruction": True, "exact_rows_targets": True,
            "frozen_parameters_buffers_unchanged": True, "identity_control": True,
            "no_training": True, "protected_roles_untouched": True},
        "limits": ["Posthoc selected development; row intervals do not estimate seed variance",
            "Deletion tests this frozen predictor, not retraining capacity or expanded-pretraining benefit",
            "Positional train cohort differences are confounded and cannot attribute pretraining efficacy"]}
    atomic_json(result_dir / "analysis.json", report)
    print(json.dumps({"status": "complete", "groups": summaries, "slot_interventions": effects, "cost": report["cost"]}))


if __name__ == "__main__":
    main()
