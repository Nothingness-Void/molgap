"""Fixed saved-prediction arithmetic using the existing alignment/paired owners."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from molgap.experiment_attribution import _compare_paired_errors, _pearson, _strata_table
from molgap.research_memory.saved_analysis import load_aligned
from molgap.training_reproducibility import atomic_json, sha256_file


def main():
    started, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(4)
    here = Path(__file__).resolve().parent
    inputs = json.loads((here / "inputs.json").read_text(encoding="utf-8"))
    prospective = here / "rml/trajectory.json"
    trajectory = json.loads(prospective.read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("An active prospective plan is required before arithmetic")
    output = here / "results"
    if output.exists():
        raise ValueError("Preserve existing results; do not overwrite the diagnostic")
    paths = {}
    for name, artifact in inputs["artifacts"].items():
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Accepted prediction bytes changed: {name}")
        paths[name] = path
    names, signed, ids = load_aligned(paths)
    base = dict(zip(names, signed))
    target = torch.load(paths["gptrans"], map_location="cpu", weights_only=True)["target_eV"].reshape(-1)
    if not np.array_equal(ids, np.arange(*inputs["development_bounds"], dtype=np.int64)):
        raise ValueError("Canonical development membership differs")
    if not np.isfinite(target.numpy()).all():
        raise ValueError("Non-finite target")
    for name, weights in inputs["fixed_formulas"].items():
        base[name] = sum(weight * base[member] for member, weight in weights.items())
    absolute = {name: np.abs(error) for name, error in base.items()}
    comparisons = {name: _compare_paired_errors(absolute[reference], absolute[candidate],
        threshold_eV=inputs["material_threshold_eV"], n_bootstrap=inputs["n_bootstrap"], seed=inputs["seed"])
        for name, (reference, candidate) in inputs["comparisons"].items()}
    primary = comparisons["equal_vs_gptrans"]
    result = dict(format="molgap-k1-saved-analysis-v1", kind="fixed_fusion_qualification",
        status="complete", rows=len(ids), mae_eV={name: float(error.mean()) for name, error in absolute.items()},
        comparisons=comparisons, fixed_formulas=inputs["fixed_formulas"],
        primary_point_gate=bool(primary["mean_delta_eV"] <= -inputs["material_threshold_eV"]
            and primary["upper_95_eV"] < 0),
        residual_correlation=_pearson(base["k1"], base["gptrans"]),
        absolute_error_correlation=_pearson(absolute["k1"], absolute["gptrans"]),
        primary_row_win_fraction=float(np.mean(absolute["equal"] < absolute["gptrans"])),
        strata_vs_gptrans=_strata_table(target.double().numpy(), absolute["gptrans"], absolute["equal"]),
        strata_vs_two_gptrans=_strata_table(target.double().numpy(), absolute["two_gptrans"], absolute["equal"]),
        source_hashes={name: artifact["sha256"] for name, artifact in inputs["artifacts"].items()},
        inputs_sha256=sha256_file(here / "inputs.json"), prospective_sha256=sha256_file(prospective),
        analysis_source_sha256=sha256_file(Path(__file__)),
        runtime=dict(device="cpu", cpu_threads=torch.get_num_threads(), torch=torch.__version__),
        training_executed=False, inference_executed=False,
        wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu,
        limitations=inputs["comparison_limits"])
    if result["wall_seconds"] > inputs["ceiling_seconds"]:
        raise TimeoutError("CPU arithmetic ceiling exceeded")
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", dict(complete=True, files={"result.json": sha256_file(output / "result.json")}))
    print(json.dumps({key: result[key] for key in ("status", "mae_eV", "primary_point_gate", "wall_seconds")}))


if __name__ == "__main__":
    main()
