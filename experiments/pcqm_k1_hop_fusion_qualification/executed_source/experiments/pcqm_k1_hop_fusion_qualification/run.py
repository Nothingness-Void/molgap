"""Thin fixed-blend arithmetic over prospectively pinned retained tensors."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from molgap.experiment_attribution import _compare_paired_errors
from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file


def main():
    start, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(4)
    here = Path(__file__).resolve().parent
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    inputs = read(here / "inputs.json")
    trajectory = here / "rml/trajectory.json"
    plan = read(trajectory)
    if plan["record_mode"] != "prospective" or plan["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Original prospective plan required")
    output = here / "results"
    if output.exists():
        raise ValueError("Preserve an existing result instead of overwriting it")
    arrays, truth, ids = {}, None, None
    for name, entry in inputs["artifacts"].items():
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Pinned payload changed: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        def field(keys):
            value = payload
            for key in keys:
                value = value[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Expected tensor: {name}/{keys}")
            return value.detach().cpu().reshape(-1)
        row, target, pred = [field(entry["fields"][k]) for k in ("source_idx", "target", "prediction")]
        if row.dtype != torch.int64 or not torch.equal(row, torch.arange(*inputs["development_bounds"])):
            raise ValueError(f"Exact source membership/order differs: {name}")
        if any(len(t) != len(row) or not t.is_floating_point() or not torch.isfinite(t).all() for t in (target, pred)):
            raise ValueError(f"Invalid finite Gap vector: {name}")
        if hashlib.sha256(row.numpy().astype("<i8").tobytes()).hexdigest() != inputs["source_idx_sha256"]:
            raise ValueError("Canonical row hash differs")
        if target.dtype != torch.float32 or hashlib.sha256(target.numpy().astype("<f4").tobytes()).hexdigest() != inputs["target_sha256"]:
            raise ValueError("Canonical float32 Gap target hash differs")
        if truth is not None and not torch.equal(target, truth):
            raise ValueError(f"Paired target bytes differ: {name}")
        ids, truth = row, target
        arrays[name] = pred.double().numpy()
    for name, weights in inputs["fixed_formulas"].items():
        if not np.isclose(sum(weights.values()), 1., rtol=0, atol=1e-15):
            raise ValueError("Fixed convex weights must sum to one")
        arrays[name] = sum(weight * arrays[key] for key, weight in weights.items())
    y = truth.double().numpy()
    errors = {name: np.abs(pred - y) for name, pred in arrays.items()}
    comparisons = {name: _compare_paired_errors(errors[reference], errors[candidate],
        threshold_eV=.003, n_bootstrap=inputs["n_bootstrap"], seed=inputs["seed"])
        for name, (reference, candidate) in inputs["comparisons"].items()}
    output.mkdir()
    atomic_torch_save(output / "fixed_predictions.pt", dict(source_idx=ids, target_eV=truth,
        predictions_eV={name: torch.from_numpy(arrays[name].copy()) for name in inputs["fixed_formulas"]}))
    result = dict(format="molgap-k1-saved-analysis-v1", kind="fixed_fusion_qualification", status="complete",
        rows=len(y), mae_eV={name: float(error.mean()) for name, error in errors.items()},
        comparisons=comparisons, fixed_formulas=inputs["fixed_formulas"],
        residual_correlations={name: float(np.corrcoef(arrays[a] - y, arrays[b] - y)[0, 1])
            for name, (a, b) in inputs["correlations"].items()},
        source_hashes={name: entry["sha256"] for name, entry in inputs["artifacts"].items()},
        inputs_sha256=sha256_file(here / "inputs.json"), prospective_sha256=sha256_file(trajectory),
        source_idx_sha256=inputs["source_idx_sha256"], target_sha256=inputs["target_sha256"],
        runtime=dict(device="cpu", cpu_threads=torch.get_num_threads(), torch=torch.__version__),
        training_executed=False, inference_executed=False, optimizer_created=False, gradients_computed=False,
        checks=dict(input_hashes_verified=True, prospective_before_execution=True,
            no_model_execution=True, alignment_verified=True, fixed_weights_preserved=True,
            consumed_development_limits_preserved=True),
        wall_seconds=time.perf_counter() - start, process_cpu_seconds=time.process_time() - cpu,
        limitations=inputs["comparison_limits"])
    if result["wall_seconds"] > inputs["ceiling_seconds"]:
        raise TimeoutError("Saved arithmetic ceiling exceeded")
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", dict(complete=True, inputs_sha256=result["inputs_sha256"],
        files={p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=result["status"], rows=result["rows"], mae_eV=result["mae_eV"],
        wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    main()
