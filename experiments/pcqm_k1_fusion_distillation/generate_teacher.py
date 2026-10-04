"""Execute only the prospective train-role teacher cache prerequisite."""
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from molgap.constants import REPO_ROOT as ROOT
from molgap.k1_frozen_inference import load_selected_k1, predict_clean
from molgap.k1_screen_training import _PackedGraphDatasetFactory
from molgap.training_reproducibility import atomic_json, atomic_torch_save, configure_fp32_determinism, sha256_file

HERE = Path(__file__).resolve().parent


def main():
    bindings = json.loads((HERE / "teacher_inputs.json").read_text())
    prospective = HERE / "cache_prospective/trajectory.json"
    planned = json.loads(prospective.read_text())
    if planned["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Teacher prerequisite is not active")
    for path, digest in planned["decision_state"]["source_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Teacher generation source changed: " + path)
    for binding in bindings["files"].values():
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Teacher generation input changed")
    destination = HERE / "teacher_cache"
    if (destination / "manifest.json").exists():
        raise FileExistsError("Teacher cache is immutable")
    destination.mkdir(exist_ok=True)
    runtime = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + 600
    graphs = ConcatDataset([_PackedGraphDatasetFactory.load(Path(bindings["files"][name]["path"]))
                            for name in ("shard0", "shard1")])
    loader = DataLoader(graphs, batch_size=128, shuffle=False, num_workers=0)
    allocation_start = time.perf_counter()
    rows, timings = {}, {}
    for arm in ("mean2", "consistency2"):
        model, context = load_selected_k1(Path(bindings["files"][arm]["path"]), expected_epoch=37, expected_parameters=3658817)
        output, timing = predict_clean(model, loader, mean=bindings["mean"], std=bindings["std"], device="cuda", deadline=deadline)
        if not torch.equal(output["source_idx"], torch.arange(100000)):
            raise ValueError("Teacher cache must contain exactly original training rows")
        rows[arm] = output
        timings[arm] = timing
        del model
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocation_seconds = time.perf_counter() - allocation_start
    if not torch.equal(rows["mean2"]["target_eV"], rows["consistency2"]["target_eV"]):
        raise ValueError("Teacher constituent targets differ")
    prediction = .5 * rows["mean2"]["prediction_eV"] + .5 * rows["consistency2"]["prediction_eV"]
    cache_path = destination / "teacher_predictions.pt"
    atomic_torch_save(cache_path, {"source_idx": rows["mean2"]["source_idx"], "prediction_eV": prediction})
    manifest = {"format": "molgap-k1-teacher-cache-v1", "teacher_identity": bindings["teacher_identity"],
                "role": "internal_training", "source_idx_range": [0, 100000],
                "dataset_manifest_sha256": bindings["dataset_manifest_sha256"],
                "path": "teacher_predictions.pt", "sha256": sha256_file(cache_path)}
    atomic_json(destination / "manifest.json", manifest)
    from molgap.k1_teacher_cache import load_teacher_cache
    load_teacher_cache(destination, {"weight": 0.1, "teacher_identity": bindings["teacher_identity"],
                                     "cache_manifest_sha256": sha256_file(destination / "manifest.json")}, mode="distill_weak")
    report = {"status": "complete_no_training", "rows": 100000, "source_idx_range": [0, 100000],
        "inputs_sha256": sha256_file(HERE / "teacher_inputs.json"), "prospective_sha256": sha256_file(prospective),
        "runtime": runtime, "hardware": torch.cuda.get_device_name(0), "timings": timings,
        "allocation_seconds": allocation_seconds, "wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "teacher_training_mae_eV": (prediction.double() - rows["mean2"]["target_eV"].double()).abs().mean().item(),
        "manifest_sha256": sha256_file(destination / "manifest.json"), "payload_sha256": manifest["sha256"],
        "source_idx_sha256": hashlib.sha256(rows["mean2"]["source_idx"].numpy().astype("<i8").tobytes()).hexdigest(),
        "cost_scope": "assigned local RTX inference window includes both model loads and two train-cohort passes; excludes graph loading/startup/hash, cache export and metric validation",
        "protected_roles_untouched": True, "geometry_constructed": False, "parameter_updates": False}
    atomic_json(HERE / "teacher_generation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
