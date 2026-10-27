"""Fixed post-hoc blends and unattainable oracle on retained predictions."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
INPUTS = {
    "k1": (Path("D:/文档/molgap/experiments/pcqm_k1_500k_bn_calibration/results/calibrated.pt"), "3d2a1074bc7627f611a5704558ab0a0767921f11242c6d22d7bf1ebcde40943a"),
    "gptrans_1e3": (Path("D:/molgap-artifacts/reconciliation/gptrans-bondlocal-pairnorm-lr-500k/v1/manifest/experiment/pairnorm_lr1e3/development_predictions.pt"), "6909b9a00d322499853e6dc36a982571696061b2b2f1af74c670428ab4ed55b2"),
    "gptrans_5e4": (Path("D:/molgap-artifacts/reconciliation/gptrans-bondlocal-pairnorm-lr-500k/v1/manifest/experiment/pairnorm_lr5e4/development_predictions.pt"), "bbb4d17c985bfd59634e98901170cefc377f69b245d3b64383eca364d8872873"),
}

start, cpu = time.perf_counter(), time.process_time()
if not (HERE / "rml/trajectory.json").exists():
    raise RuntimeError("Prospective trajectory must exist first")
data = {}
for name, (path, digest) in INPUTS.items():
    if sha256_file(path) != digest:
        raise ValueError(f"Input hash mismatch: {name}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    data[name] = {"idx": obj["source_idx"].view(-1),
                  "target": obj.get("target_eV", obj.get("target")).double().view(-1).numpy(),
                  "prediction": obj.get("prediction_eV", obj.get("prediction")).double().view(-1).numpy()}
truth = data["k1"]["target"]
for name, item in data.items():
    if not torch.equal(item["idx"], torch.arange(500000, 550000)) or not np.array_equal(item["target"], truth):
        raise ValueError(f"Row or target alignment mismatch: {name}")
    if not np.isfinite(item["target"]).all() or not np.isfinite(item["prediction"]).all():
        raise ValueError(f"Non-finite values: {name}")
k, g, h = (data[n]["prediction"] for n in ("k1", "gptrans_1e3", "gptrans_5e4"))
predictions = {"k1": k, "gptrans_1e3": g, "gptrans_5e4": h,
    "k1_gptrans_1e3_equal": (k + g) / 2, "k1_gptrans_5e4_equal": (k + h) / 2,
    "gptrans_pair_equal": (g + h) / 2, "three_equal": (k + g + h) / 3}
errors = {name: np.abs(value - truth) for name, value in predictions.items()}
errors["oracle_k1_gptrans_1e3"] = np.minimum(errors["k1"], errors["gptrans_1e3"])
errors["oracle_three"] = np.minimum.reduce([errors["k1"], errors["gptrans_1e3"], errors["gptrans_5e4"]])
comparisons = {}
for candidate, reference in [("k1_gptrans_1e3_equal", "gptrans_1e3"), ("k1_gptrans_1e3_equal", "gptrans_pair_equal"), ("three_equal", "gptrans_pair_equal")]:
    comparisons[f"{candidate}_minus_{reference}"] = paired_bootstrap_mean(errors[candidate] - errors[reference], n_bootstrap=1000, seed=20261027)
result = {"schema":"molgap-fixed-prediction-fusion-analysis-v1", "role":"selection-used internal development", "rows":len(truth),
    "inputs":{name:{"path":str(path), "sha256":digest} for name, (path, digest) in INPUTS.items()},
    "mae_eV":{name:float(error.mean()) for name,error in errors.items()}, "paired_row_bootstrap":comparisons,
    "k1_vs_gptrans1e3_residual_correlation":float(np.corrcoef(k-truth,g-truth)[0,1]),
    "k1_wins_fraction":float((errors["k1"] < errors["gptrans_1e3"]).mean()),
    "oracle_warning":"Uses ground truth per row; not a deployable router or independent score",
    "training_executed":False, "inference_executed":False, "wall_seconds":time.perf_counter()-start, "cpu_seconds":time.process_time()-cpu}
atomic_json(HERE / "analysis.json", result)
print(json.dumps(result["mae_eV"], indent=2))
print(json.dumps(comparisons, indent=2))
