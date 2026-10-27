"""Cross-fit the already-studied prediction-only gates on new saved experts."""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if not (HERE / "rml/trajectory.json").exists():
    raise RuntimeError("Prospective trajectory required")
if (HERE / "analysis.json").exists():
    raise RuntimeError("Existing result requires reconciliation, not overwrite")
wall, cpu = time.perf_counter(), time.process_time()
prior = json.loads((ROOT / "experiments/pcqm_k1_gptrans_pairnorm_fusion_500k/analysis.json").read_text())
helper_path = ROOT / "experiments/pcqm_500k_v4_evidence/analyze_local_ablation.py"
spec = importlib.util.spec_from_file_location("matched_saved_predictions", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
data = {}
for name, binding in prior["inputs"].items():
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"Input digest mismatch: {name}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    data[name] = {"idx":obj["source_idx"].view(-1).numpy(),
        "y":obj.get("target_eV",obj.get("target")).double().view(-1).numpy(),
        "p":obj.get("prediction_eV",obj.get("prediction")).double().view(-1).numpy()}
anchor = data["k1"]
for name, item in data.items():
    if not np.array_equal(item["idx"], np.arange(500000,550000)) or not np.array_equal(item["y"], anchor["y"]):
        raise ValueError(f"Source rows or targets differ: {name}")
    if not np.isfinite(item["p"]).all() or not np.isfinite(item["y"]).all():
        raise ValueError(f"Nonfinite input: {name}")
y, k, g, h = anchor["y"], data["k1"]["p"], data["gptrans_1e3"]["p"], data["gptrans_5e4"]["p"]
folds = anchor["idx"] % 5
results, retained = {}, {"source_idx":anchor["idx"], "target":y}
with threadpool_limits(limits=4):
    for name, second, fixed_alpha in (("pair",g,.5), ("three",(g+h)/2,1/3)):
        d = k - second
        x = np.column_stack([k,second,d,np.abs(d),(k+second)/2])
        winner = (np.abs(k-y) < np.abs(second-y)).astype(int)
        alpha = np.zeros_like(y)
        np.divide(y-second,d,out=alpha,where=d != 0)
        alpha = np.clip(alpha,0,1)
        predictions = {key:np.empty_like(y) for key in ("global", "hard", "soft")}
        probabilities = np.empty_like(y)
        fold_report = []
        for fold in range(5):
            train, test = folds != fold, folds == fold
            weight = helper._weighted_median_weight(k[train],second[train],y[train])
            predictions["global"][test] = weight*k[test] + (1-weight)*second[test]
            hard = make_pipeline(StandardScaler(), LogisticRegression(C=1,max_iter=500))
            hard.fit(x[train],winner[train])
            probabilities[test] = hard.predict_proba(x[test])[:,1]
            predictions["hard"][test] = np.where(probabilities[test] >= .5,k[test],second[test])
            soft = HistGradientBoostingRegressor(loss="absolute_error",max_iter=100,max_leaf_nodes=7,
                max_depth=3,min_samples_leaf=200,l2_regularization=1,learning_rate=.1,random_state=42)
            soft.fit(x[train],alpha[train],sample_weight=np.abs(d[train]))
            predictions["soft"][test] = second[test] + np.clip(soft.predict(x[test]),0,1)*d[test]
            fold_report.append({"fold":fold,"rows":int(test.sum()),"global_weight_k1":weight,
                "mae_eV":{key:float(np.abs(v[test]-y[test]).mean()) for key,v in predictions.items()}})
        predictions["fixed"] = fixed_alpha*k + (1-fixed_alpha)*second
        errors = {key:np.abs(value-y) for key,value in predictions.items()}
        gates = {}
        for key in ("hard","soft"):
            diff = errors[key]-errors["global"]
            stats = paired_bootstrap_mean(diff,n_bootstrap=2000,seed=20261027)
            improved = sum(row["mae_eV"][key] < row["mae_eV"]["global"] for row in fold_report)
            gates[key] = {"candidate_minus_global":stats,"improving_folds":improved,
                "nomination_pass":bool(stats["delta"] <= -.001 and stats["ci95"][1] < 0 and improved >= 4)}
        results[name] = {"mae_eV":{key:float(error.mean()) for key,error in errors.items()},
            "oracle_select_between_k1_and_second_mae_eV":float(np.minimum(np.abs(k-y),np.abs(second-y)).mean()),
            "model_passes":2 if name == "pair" else 3, "hard_winner_auc":float(roc_auc_score(winner,probabilities)),
            "gates":gates,"folds":fold_report}
        for key,value in predictions.items():
            retained[f"{name}_{key}"] = value
out = Path("D:/molgap-artifacts/diagnostics/pcqm-pairnorm-prediction-router-500k-20261027")
out.mkdir(parents=True,exist_ok=True)
oof = out / "gate_oof_predictions.npz"
if oof.exists():
    raise FileExistsError("Retained OOF output already exists")
np.savez_compressed(oof,**retained)
report = {"format":"molgap-pairnorm-prediction-router-diagnostic-v1", "rows":len(y),
    "interpretation":"Five-fold gate OOF on development-selected backbone outputs; not independent model validation",
    "inputs":prior["inputs"], "groups":results, "retained_oof":{"path":str(oof),"sha256":sha256_file(oof)},
    "training_executed":False, "backbone_inference_executed":False, "gate_fit":True,
    "official_validation_read":False, "test_read":False, "cpu_threads":4,
    "wall_seconds":time.perf_counter()-wall,"cpu_seconds":time.process_time()-cpu,
    "source_sha256":sha256_file(Path(__file__)),"helper_sha256":sha256_file(helper_path)}
atomic_json(HERE / "analysis.json",report)
print(json.dumps(results,indent=2))
