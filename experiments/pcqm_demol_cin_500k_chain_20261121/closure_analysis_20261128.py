"""One-off closure analysis for the DeMol-CIN 500K chain (2026-11-28).

Reads only saved per-row predictions and already-scored JSON. No checkpoint is loaded and no new tail
score is computed: the tail prediction files were produced on 2026-11-25 (J1-J5) and 2026-11-11 (Kaggle cohort).
Paired row bootstrap: 2000 draws, numpy default_rng(42), mean of |err_a| - |err_b| (repo convention).
Run: .venv\\Scripts\\python.exe -I closure_analysis_20261128.py
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ART = Path("D:/molgap-artifacts")
OUT = Path(__file__).resolve().parent / "evidence" / "closure_analysis_20261128.json"
T = ART / "tail_eval_demol_ims_20261125"
F0 = ART / "ims_weekend_20261120" / "f0" / "preds"
TAILS = ART / "tail_holdout_predictions"
DRAWS, SEED = 2000, 42


def load(path):
    d = torch.load(path, weights_only=False, map_location="cpu")
    return d["pred_ev"].double().numpy(), d["truth_ev"].double().numpy()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mae_mev(pred, truth):
    return float(np.abs(pred - truth).mean() * 1000)


def boot(err_a, err_b):
    gain = (err_a - err_b) * 1000
    rng = np.random.default_rng(SEED)
    draws = np.array([gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(DRAWS)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta_meV": float(gain.mean()), "ci95_meV": [float(lo), float(hi)], "draws": DRAWS, "seed": SEED}


def ens_mae(a, b, truth):
    return mae_mev((a + b) / 2, truth)


out = {"inputs_sha256": {}}


def get(name, path):
    out["inputs_sha256"][name] = sha(path)
    return load(path)


# ---- dev rows (100K view, rows 100000..149999) ----
j1, t_dev = get("j1_dev", T / "preds_j1" / "j1_development.pt")
j2, t2 = get("j2_dev", T / "preds_j2" / "j2_development.pt")
d42, t3 = get("f0_demol_dev", F0 / "demol_cin_development.pt")
g42, t4 = get("f0_pun_dev", F0 / "pair_update_norm_development.pt")
for t in (t2, t3, t4):
    assert np.array_equal(t, t_dev), "dev truth differs"

# Rule (a)
res_a = boot(np.abs(j1 - t_dev), np.abs(j2 - t_dev))
res_a.update(mae_j1_meV=mae_mev(j1, t_dev), mae_j2_meV=mae_mev(j2, t_dev),
             rule="J1 - J2 dev paired row bootstrap; replicated needs delta <= -3.0 and CI upper < 0")
out["rule_a"] = res_a

# Rule (b) P100: G42=Kaggle pair_update_norm (F0), G43=J2, D42=Kaggle DeMol (F0), D43=J1
models = {"G42": g42, "G43": j2, "D42": d42, "D43": j1}
mae_single = {k: mae_mev(v, t_dev) for k, v in models.items()}
same = {"G42+G43": ens_mae(g42, j2, t_dev), "D42+D43": ens_mae(d42, j1, t_dev)}
cross = {f"{g}+{d}": ens_mae(models[g], models[d], t_dev) for g in ("G42", "G43") for d in ("D42", "D43")}
out["rule_b_p100"] = {
    "single_mae_meV": mae_single, "same_pairs_meV": same, "cross_pairs_meV": cross,
    "same_mean_meV": float(np.mean(list(same.values()))), "cross_mean_meV": float(np.mean(list(cross.values()))),
    "P_meV": float(np.mean(list(same.values())) - np.mean(list(cross.values()))),
    "thresholds": "P >= 3.0 worth a 500K test; 1.0 <= P < 3.0 inconclusive; P < 1.0 close the hetero line",
    "note": "Mixed platforms inside a pair (Kaggle s42 + IMS s43), as the frozen rule allows. Dev rows are the rows used for checkpoint selection, so P is optimistic.",
}

# ---- tail rows ----
tail = {}
for name in ("j3", "j4", "j5"):
    tail[name], tt = get(f"{name}_tail", T / f"preds_{name}" / f"{name}_tail_holdout_v1.pt")
    if name == "j3":
        t_tail = tt
    else:
        assert np.array_equal(tt, t_tail)
out["tail_mae_meV"] = {k: mae_mev(v, t_tail) for k, v in tail.items()}
out["tail_j3_minus_j5"] = boot(np.abs(tail["j3"] - t_tail), np.abs(tail["j5"] - t_tail))
out["tail_j4_minus_j5"] = boot(np.abs(tail["j4"] - t_tail), np.abs(tail["j5"] - t_tail))
out["tail_j3_minus_j4_descriptive"] = boot(np.abs(tail["j3"] - t_tail), np.abs(tail["j4"] - t_tail))

# P500
cohort = {}
for s in (43, 44, 45, 46):
    cohort[f"s{s}"], tt = get(f"kaggle_gptt_init_s{s}_tail", TAILS / f"gptrans_500k_init_s{s}_tail_holdout_v1.pt")
    assert np.array_equal(tt, t_tail)
gsame = {f"{a}+{b}": ens_mae(cohort[a], cohort[b], t_tail) for a, b in itertools.combinations(cohort, 2)}
dsame = ens_mae(tail["j3"], tail["j4"], t_tail)
het = {f"{d}+{g}": ens_mae(tail[d], cohort[g], t_tail) for d in ("j3", "j4") for g in cohort}
assert len(gsame) == 6 and len(het) == 8
gsame_mean = float(np.mean(list(gsame.values())))
same_val = min(gsame_mean, dsame)
out["rule_b_p500"] = {
    "gptt_members": "Kaggle cohort init_s43..s46 tail predictions (tail_holdout_predictions/gptrans_500k_init_s4x)",
    "gptt_single_mae_meV": {k: mae_mev(v, t_tail) for k, v in cohort.items()},
    "gptt_same_pairs_meV": gsame, "gptt_same_mean_meV": gsame_mean,
    "demol_same_pair_j3_j4_meV": dsame, "Same_meV": same_val,
    "Same_is": "gptt_same_mean" if gsame_mean <= dsame else "demol_same_pair",
    "hetero_pairs_meV": het, "Hetero_meV": float(np.mean(list(het.values()))),
    "P500_meV": float(same_val - np.mean(list(het.values()))),
    "thresholds": "as rule (b): P >= 3.0 worth a full-scale test; 1.0 <= P < 3.0 inconclusive; P < 1.0 close",
    "note": "J5 (IMS replay of s44) is not a cohort member in this rule. Tail rows, one read per checkpoint, no selection.",
}

# ---- migration rule ----
j3, j4 = out["tail_mae_meV"]["j3"], out["tail_mae_meV"]["j4"]
G = 107.940
out["transfer"] = {"G_meV": G, "mean_j3_j4_meV": (j3 + j4) / 2, "delta_meV": (j3 + j4) / 2 - G,
                   "band": "(-3.0, -1.5] -> INCONCLUSIVE" if -3.0 < (j3 + j4) / 2 - G <= -1.5 else "outside"}
out["j5_shift_meV"] = out["tail_mae_meV"]["j5"] - 107.273

# ---- cost ----
rs = json.loads((ART / "ims_retrieved_20261125" / "demol" / "serial_summary.json").read_text())
wall = {k: sum(a["wall_s"] for a in v["attempts"]) for k, v in rs["items"].items()}
out["cost_wall_s"] = wall
out["cost_ratio"] = {"j3_over_j5": wall["j3-demol-500k-s42"] / wall["j5-gptt-500k-s44"],
                     "j4_over_j5": wall["j4-demol-500k-s43"] / wall["j5-gptt-500k-s44"],
                     "mean_demol_over_j5": (wall["j3-demol-500k-s42"] + wall["j4-demol-500k-s43"]) / 2 / wall["j5-gptt-500k-s44"],
                     "note": "A100 wall per item summed over attempts; J5 had two attempts (pause then resume)."}

def log_elapsed(item):
    """Sum of train_log elapsed_s; elapsed restarts at 0 after a resume, so add each segment's last value."""
    rows = [json.loads(x) for x in (ART / "ims_retrieved_20261125" / "demol" / item / "train_log.jsonl").read_text().splitlines() if x.strip()]
    el = [x["elapsed_s"] for x in rows]
    cut = [i for i in range(1, len(el)) if el[i] < el[i - 1]]
    return float(sum(el[i - 1] for i in cut) + el[-1])


le = {k: log_elapsed(k) for k in wall}
out["cost_train_log_elapsed_s"] = le
out["cost_ratio_train_log"] = {
    "j3_over_j5": le["j3-demol-500k-s42"] / le["j5-gptt-500k-s44"],
    "j4_over_j5": le["j4-demol-500k-s43"] / le["j5-gptt-500k-s44"],
    "mean_demol_over_j5": (le["j3-demol-500k-s42"] + le["j4-demol-500k-s43"]) / 2 / le["j5-gptt-500k-s44"],
    "note": "Frozen rule source (train_log.jsonl). J5 elapsed restarts at epoch 33 after the pause.",
}

# ---- give-back (descriptive): selected step dev vs final-epoch dev from train_log ----
gb = {}
for item in rs["items"]:
    d = ART / "ims_retrieved_20261125" / "demol" / item
    sel = json.loads((d / "selected_eval.json").read_text())
    rows = [json.loads(x) for x in (d / "train_log.jsonl").read_text().splitlines() if x.strip()]
    last = rows[-1]
    gb[item] = {"selected_step": sel["step"], "selected_dev_meV": sel["dev_mae_eV"] * 1000,
                "final_epoch": last["epoch"], "final_step": last["step"], "final_dev_meV": last["dev_mae_eV"] * 1000,
                "giveback_meV": (last["dev_mae_eV"] - sel["dev_mae_eV"]) * 1000}
out["giveback_descriptive"] = gb

# pairs[] block for closure facts (same field names as rml analyze summary pairs)
def _pair(cand, ref, src):
    return {"candidate": cand, "reference": ref, "role": src["role"], "delta": src["res"]["delta_meV"],
            "ci95": src["res"]["ci95_meV"], "n_bootstrap": DRAWS, "seed": SEED}


out["pairs"] = [
    _pair("j1-demol-100k-s43", "j2-gptt-100k-s43", {"role": "dev_100k", "res": out["rule_a"]}),
    _pair("j3-demol-500k-s42", "j5-gptt-500k-s44", {"role": "tail_holdout_v1", "res": out["tail_j3_minus_j5"]}),
    _pair("j4-demol-500k-s43", "j5-gptt-500k-s44", {"role": "tail_holdout_v1", "res": out["tail_j4_minus_j5"]}),
]
out["script"] = "closure_analysis_20261128.py (one-off, old-style; user 11-28 authorisation)"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("inputs_sha256",)}, indent=1))
