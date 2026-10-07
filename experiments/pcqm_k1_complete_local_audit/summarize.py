"""Posthoc saved-prediction descriptors; no model import, execution or fitting."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from molgap.pcqm_k1_gptrans_fusion import mae
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json, sha256_file


if __name__ == "__main__":
    start, cpu = time.perf_counter(), time.process_time()
    here = Path(__file__).resolve().parent
    inputs = json.loads((here / "inputs.json").read_text())
    def load(key):
        item = inputs["artifacts"][key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Retained prediction bytes differ")
        value = torch.load(item["path"], map_location="cpu", weights_only=True)
        return {"source_idx":value["source_idx"].view(-1),
                "target":value.get("target_eV",value.get("target")).double().view(-1).numpy(),
                "prediction":value.get("prediction_eV",value.get("prediction")).double().view(-1).numpy()}
    k, g = load("selected_clean_predictions"), load("gptrans_predictions")
    if not torch.equal(k["source_idx"],g["source_idx"]) or not np.array_equal(k["target"],g["target"]) or not torch.equal(k["source_idx"],torch.arange(500000,550000)):
        raise ValueError("Exact retained50K membership/targets differ")
    truth, kp, gp = k["target"], k["prediction"], g["prediction"]
    combined = .5 * kp + .5 * gp
    ek, eg, ef = np.abs(kp-truth), np.abs(gp-truth), np.abs(combined-truth)
    stats = paired_bootstrap_mean(eg-ef,n_bootstrap=1000,seed=20261008)
    stats["probability_nonnegative_gain"] = 1-stats.pop("probability_better")
    report = dict(format="molgap-k1-retained-prediction-description-v1", rows=50000,
        interpretation="Posthoc fixed50:50 blend on historically selection-used rows; no weight fit, new training or independent qualification",
        source_hashes={key:inputs["artifacts"][key]["sha256"] for key in ("selected_clean_predictions","gptrans_predictions")},
        k1_clean_mae_eV=mae(kp,truth), gptrans_ema_mae_eV=mae(gp,truth), fixed_equal_blend_mae_eV=mae(combined,truth),
        paired_blend_gain_over_gptrans=stats, residual_pearson=float(np.corrcoef(kp-truth,gp-truth)[0,1]),
        k1_row_win_fraction=float((ek<eg).mean()), opposite_residual_sign_fraction=float(((kp-truth)*(gp-truth)<0).mean()),
        absolute_error_quantiles={n:np.quantile(e,[.5,.9,.95,.99]).tolist() for n,e in (("k1",ek),("gptrans",eg),("equal_blend",ef))},
        quantile_probabilities=[.5,.9,.95,.99], training_executed=False,inference_executed=False,
        optimizer_created=False, gradients_computed=False, wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu)
    atomic_json(here / "retained_prediction_summary.json",report)
    print(json.dumps({k:report[k] for k in ("k1_clean_mae_eV","gptrans_ema_mae_eV","fixed_equal_blend_mae_eV","residual_pearson")}))
