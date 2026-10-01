"""Thin CPU loader qualification against the prepared frozen source."""
from pathlib import Path
import json
import time


def main():
    from molgap.k1_screen_training import (
        find_fixed_cache, load_roles, _target_stats, _train_loader, FORBIDDEN_MODEL_FIELDS,
    )
    from molgap.training_reproducibility import atomic_json
    import torch
    from torch_geometric.loader import DataLoader
    root = Path(__file__).resolve().parents[2]
    output = root / "experiments/pcqm_k1_node_width256/cpu_loader_qualification.json"
    if output.exists():
        raise FileExistsError("Retained CPU qualification exists; reconcile before replay")
    wall, cpu = time.perf_counter(), time.process_time()
    cache, manifest = find_fixed_cache(Path("D:/文档/molgap/data/pcqm_fixed_100k_v1"))
    roles = load_roles(cache, manifest)
    if len(roles["train"]) != 100_000 or len(roles["development"]) != 50_000:
        raise RuntimeError("Frozen role sizes changed")
    mean, std = _target_stats(roles["train"])
    loader = _train_loader(roles["train"], 0)
    # Windows spawn cannot pickle the retained Linux loader's local dataset
    # class. Inspect identical dataset/sampler batches in the CPU process;
    # the formal Linux workers=2 recipe stays frozen and qualifies remotely.
    cpu_loader = DataLoader(loader.dataset, batch_sampler=loader.batch_sampler, num_workers=0)
    batch = next(iter(cpu_loader))
    if batch.num_graphs != 128 or batch.x.shape[1] != 9 or batch.edge_attr.shape[1] != 3:
        raise RuntimeError("Frozen batch/features changed")
    if any(field in batch for field in FORBIDDEN_MODEL_FIELDS):
        raise RuntimeError("Geometry fields survived the pure2D loader")
    if not torch.isfinite(batch.y).all() or not torch.isfinite(batch.random_walk_pe).all():
        raise RuntimeError("Nonfinite retained inputs")
    report = {"status": "CPU_REAL_SHARD_LOADER_QUALIFIED", "source_owner": __import__("molgap.k1_screen_training", fromlist=["__file__"]).__file__,
              "train_rows": len(roles["train"]), "development_rows": len(roles["development"]),
              "batch_size": batch.num_graphs, "node_features": 9, "edge_features": 3,
              "target_mean": mean, "target_std": std, "geometry_model_input": False,
              "gpu_qualification": "pending", "model_execution": False,
              "cpu_loader_workers": 0, "formal_linux_loader_workers": 2,
              "windows_multiprocess_loader": "unsupported_local_dataset_class_pickling",
              "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu}
    atomic_json(output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
