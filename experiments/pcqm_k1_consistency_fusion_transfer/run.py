"""Frozen K1 equal-blend qualification on a common internal-development role."""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from molgap.constants import REPO_ROOT
from molgap.k1_frozen_inference import load_selected_k1, predict_clean
from molgap.k1_screen_training import _PackedGraphDatasetFactory
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json, atomic_torch_save, configure_fp32_determinism, sha256_file

HERE = Path(__file__).resolve().parent


def main():
    inputs = json.loads((HERE / "inputs.json").read_text())
    trajectory = json.loads((HERE / "rml/trajectory.json").read_text())
    for relative, digest in trajectory["decision_state"]["source_hashes"].items():
        if sha256_file(REPO_ROOT / relative) != digest:
            raise ValueError(f"Prospective source changed: {relative}")
    for binding in inputs["bindings"].values():
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError(f"Input changed: {binding['path']}")
    if (HERE / "results/measurement.json").exists():
        raise FileExistsError("A completed measurement cannot be overwritten")
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + 600
    runtime = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized local accelerator unavailable")
    device = torch.cuda.get_device_name(0)
    if device != inputs["hardware"]:
        raise ValueError("Local accelerator differs from plan")
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    calibration = _PackedGraphDatasetFactory.load(Path(inputs["bindings"]["calibration_shard"]["path"]))
    development = _PackedGraphDatasetFactory.load(Path(inputs["bindings"]["development_shard"]["path"]))
    for dataset, first in ((calibration, 100000), (development, 500000)):
        if not torch.equal(dataset._data.source_idx.view(-1).long(), torch.arange(first, first + 50000)):
            raise ValueError("Frozen development row order differs")
    offsets = np.linspace(0, 49999, 2048, dtype=int).tolist()
    loaders = [DataLoader(Subset(calibration, offsets), batch_size=128, shuffle=False, num_workers=0),
               DataLoader(development, batch_size=128, shuffle=False, num_workers=0)]
    arms, timings, equivalence = {}, {}, {}
    allocation_start = time.perf_counter()
    for arm in ("mean2", "consistency2"):
        model, context = load_selected_k1(Path(inputs["bindings"][arm + "_model"]["path"]), expected_epoch=37, expected_parameters=3658817)
        saved = torch.load(inputs["bindings"][arm + "_predictions"]["path"], map_location="cpu", weights_only=True)
        if saved["context"] != context:
            raise ValueError("Selected model context differs from accepted predictions")
        old, qualification_timing = predict_clean(model, loaders[0], mean=inputs["mean"], std=inputs["std"], device="cuda", deadline=deadline)
        if not torch.equal(old["target_eV"], saved["target_eV"][offsets]):
            raise ValueError("Reconstruction labels differ")
        if not torch.equal(old["source_idx"], saved["source_idx"][offsets]):
            raise ValueError("Reconstruction source indices differ")
        delta = (old["prediction_eV"] - saved["prediction_eV"][offsets]).abs()
        equivalence[arm] = {"rows": 2048, "max_abs_delta_eV": delta.max().item(),
                            "mean_abs_delta_eV": delta.mean().item(), "tolerance_eV": 1e-4}
        if delta.max().item() > 1e-4:
            raise ValueError("Accepted prediction reconstruction failed")
        rows, timing = predict_clean(model, loaders[1], mean=inputs["mean"], std=inputs["std"], device="cuda", deadline=deadline)
        arms[arm] = rows
        timings[arm] = {"qualification": qualification_timing, "development": timing}
        del model
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocation_seconds = time.perf_counter() - allocation_start
    if not torch.equal(arms["mean2"]["target_eV"], arms["consistency2"]["target_eV"]):
        raise ValueError("Arm targets differ")
    blend = {**arms["mean2"], "prediction_eV": .5 * arms["mean2"]["prediction_eV"] + .5 * arms["consistency2"]["prediction_eV"]}
    arms["fixed_equal_blend"] = blend
    errors = {name: (rows["prediction_eV"].double() - rows["target_eV"].double()).abs().numpy() for name, rows in arms.items()}
    metrics = {name: float(error.mean()) for name, error in errors.items()}
    comparisons = {}
    for reference in ("mean2", "consistency2"):
        result = paired_bootstrap_mean(errors[reference] - errors["fixed_equal_blend"], n_bootstrap=1000, seed=42)
        comparisons[reference] = {"gain_meV": result["delta"] * 1000, "ci95_meV": [x * 1000 for x in result["ci95"]]}
    artifacts = {}
    for name, rows in arms.items():
        path = output / (name + ".pt")
        atomic_torch_save(path, rows)
        artifacts[path.relative_to(REPO_ROOT).as_posix()] = sha256_file(path)
    torch.cuda.synchronize()
    report = {"status": "complete_no_training", "rows": 50000, "source_idx_range": [500000, 550000],
              "inputs_sha256": sha256_file(HERE / "inputs.json"), "prospective_sha256": sha256_file(HERE / "rml/trajectory.json"),
              "hardware": device, "runtime": {**runtime, "torch": torch.__version__, "cuda": torch.version.cuda},
              "equivalence": equivalence, "timings": timings, "mae_eV": metrics, "comparisons": comparisons,
              "cohort_source_idx_sha256": {"calibration": hashlib.sha256(old["source_idx"].numpy().astype("<i8").tobytes()).hexdigest(),
                                          "development": hashlib.sha256(blend["source_idx"].numpy().astype("<i8").tobytes()).hexdigest()},
              "row_manifest_encoding": "ascending source_idx signed int64 little-endian raw bytes",
              "nomination_passed": all(x["gain_meV"] >= 1 and x["ci95_meV"][0] > 0 for x in comparisons.values()),
              "artifact_hashes": artifacts, "allocation_seconds": allocation_seconds,
              "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "cost_scope": "one assigned RTX 5060; two reconstruction probes and two common-cohort passes; loader/transfer waits included; hashing/startup and report/bootstrap excluded from allocation",
              "limitations": ["One training seed; row bootstrap does not measure training stochasticity.", "Internal-development rows previously used for other experiments; not a sealed test.", "RTX inference reconstruction is tolerance-qualified, not a native T4 training comparison.", "Historical training costs and replay exclusions remain unchanged."]}
    atomic_json(output / "measurement.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
