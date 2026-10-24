"""Predeclared fixed-TRAIN and structural summaries; no model execution."""
from collections import deque
import argparse
import json
from pathlib import Path
import time


def run(retained, output, data_root):
    from molgap.family_training_runner import read_recipe
    from molgap.family_training_data import load_training_data
    from molgap.experiment_attribution import _normalize_trace_rows
    from molgap.training_reproducibility import atomic_json, sha256_file
    import numpy as np
    import torch
    began, cpu_began = time.perf_counter(), time.process_time()
    torch.set_num_threads(4)
    roots = {arm: Path(retained) / arm for arm in ("hop_mixed", "hop_ordered")}
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    values, cohorts, curves = {}, {}, {}
    for arm, root in roots.items():
        contract, recipe = read_recipe(root / "training_contract.json")
        values[arm] = torch.load(root / "development_predictions.pt", map_location="cpu", weights_only=True)
        curves[arm] = _normalize_trace_rows(read(root / "canonical_trace.json"), recipe.target_std_eV)
        manifest = read(root / "train_cohort/manifest.json")
        if set(manifest["files"]) != {f"pass_{p:02d}.json" for p in (10, 20, 30, 40)}:
            raise ValueError("Incomplete fixed TRAIN receipt inventory")
        cohorts[arm] = []
        for p in (10, 20, 30, 40):
            name = f"pass_{p:02d}.json"
            path = root / "train_cohort" / name
            if sha256_file(path) != manifest["files"][name]["sha256"]:
                raise ValueError("TRAIN receipt hash differs")
            record = read(path)
            expected = {"pass_index": p, "optimizer_step": p * 781, "rows": 4096,
                "weights": "live", "model_mode": "eval", "metric": "MAE", "target": "Gap",
                "role": "TRAIN", "unit": "eV", "selection_used": False, "formal_training_exposure": 0}
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError("Fixed TRAIN receipt semantics differ")
            dev = next(row["live_dev_metric"] for row in curves[arm] if row["pass_index"] == p)
            cohorts[arm].append({"pass": p, "optimizer_step": p * 781,
                "train_fixed_eval_mae_eV": record["mae_eV"], "development_live_mae_eV": dev,
                "development_minus_fixed_train_eV": dev - record["mae_eV"],
                "source_idx_sha256": record["source_idx_sha256"], "target_eV_sha256": record["target_eV_sha256"]})
    a, b = (values[arm] for arm in roots)
    if (not torch.equal(a["source_idx"], b["source_idx"]) or
            not torch.equal(a["target_eV"], b["target_eV"]) or
            not torch.equal(a["source_idx"], torch.arange(100000, 150000))):
        raise ValueError("Finite development rows are not aligned")
    for x, y in zip(*cohorts.values()):
        if any(x[key] != y[key] for key in ("source_idx_sha256", "target_eV_sha256")):
            raise ValueError("Fixed TRAIN cohorts differ")
    data = load_training_data(Path(data_root), recipe.dataset)
    bins = {"diameter<=4": [], "diameter>4": [], "disconnected": [],
            "cycle_rank0": [], "cycle_rank1": [], "cycle_rank2": [], "cycle_rank>=3": [],
            "conjugated_fraction0": [], "conjugated_fraction(0,0.5]": [], "conjugated_fraction>0.5": []}
    for index, graph in enumerate(data.development):
        n = graph.x.shape[0]
        adjacency = [set() for _ in range(n)]
        edges = {}
        for k, (u, v) in enumerate(graph.edge_index.t().tolist()):
            if u == v:
                raise ValueError("Unexpected chemical self bond")
            adjacency[u].add(v); adjacency[v].add(u)
            key = tuple(sorted((u, v)))
            conjugated = int(graph.edge_attr[k, 2])
            if conjugated not in (0, 1) or (key in edges and edges[key] != conjugated):
                raise ValueError("Conjugation encoding or directed bond copies differ")
            edges[key] = conjugated
        unseen, components, diameter_above4 = set(range(n)), 0, False
        while unseen:
            components += 1
            found, queue = {next(iter(unseen))}, deque([next(iter(unseen))])
            while queue:
                for v in adjacency[queue.popleft()] - found:
                    found.add(v); queue.append(v)
            unseen -= found
        if components > 1:
            bins["disconnected"].append(index)
        else:
            for start in range(n):
                seen, frontier = {start}, {start}
                for _ in range(4):
                    frontier = set().union(*(adjacency[u] for u in frontier)) - seen if frontier else set()
                    seen |= frontier
                if len(seen) < n:
                    diameter_above4 = True
                    break
            bins["diameter>4" if diameter_above4 else "diameter<=4"].append(index)
        rank = len(edges) - n + components
        bins["cycle_rank>=3" if rank >= 3 else f"cycle_rank{rank}"].append(index)
        fraction = sum(edges.values()) / len(edges) if edges else 0.0
        bins["conjugated_fraction0" if fraction == 0 else
             "conjugated_fraction(0,0.5]" if fraction <= .5 else "conjugated_fraction>0.5"].append(index)
    errors = [np.abs(value["prediction_eV"].double().numpy() - value["target_eV"].double().numpy()) for value in (a, b)]
    slices = {}
    for name, members in bins.items():
        left, right = (error[members] for error in errors)
        slices[name] = {"rows": len(members), "mixed_mae_eV": float(left.mean()) if members else None,
            "ordered_mae_eV": float(right.mean()) if members else None,
            "ordered_minus_mixed_eV": float((right-left).mean()) if members else None}
    result = {"format": "molgap-k1-shell-retained-summary-v1", "training_executed": False,
        "inference_executed": False, "fixed_train": cohorts, "structural_slices": slices,
        "roles": {"train_labels_decoded": [0, 100000], "development": [100000, 150000],
                  "protected_roles_consumed": False}, "dataset_manifest_sha256": recipe.dataset.manifest_sha256,
        "disconnected_diameter_semantics": "Excluded from finite connected diameter bins; recorded separately.",
        "limitations": ["Descriptive selected-development slices, no slice-based selection or multiple-testing claim.",
                         "The fixed4096 TRAIN cohort is not the entire TRAIN distribution."],
        "wall_seconds": time.perf_counter()-began, "process_cpu_seconds": time.process_time()-cpu_began}
    atomic_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default="D:/molgap-data")
    args = parser.parse_args()
    report = run(args.retained_root, args.output, args.data_root)
    print(json.dumps({"output": args.output, "wall_seconds": report["wall_seconds"]}))
