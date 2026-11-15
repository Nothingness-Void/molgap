"""Arm A verdict on tail_holdout_v1 for protocol.md (same-recipe B1024 all-init ensemble; equal weights, nothing fitted).

Written and committed before the s44/s45 tail numbers were computed. Mapping of the frozen rules:
  band: each B1024 single-model tail MAE must lie in 87.89 +/- 1.5 meV; a member outside is INCONCLUSIVE (kept in the ensemble);
      if s44 or s45 is outside, the A outcome is INCONCLUSIVE and the criterion below is reported descriptively.
  noise floor: sample std (ddof 1) and range of the four single MAEs; N = max(1.08, range). M = their mean.
  primary: four = mean of S42..S45 predictions; diff = four - pair(S42, S43) per row (the pair is the post-hoc 85.349 baseline,
      reproduced from the saved predictions). PASS iff mean diff <= -1.0 meV and the one-sided 99% upper bound
      (99th percentile of the paired bootstrap mean diff) < 0.
  Bootstrap: 2000 row resamples, numpy default_rng(42), index sets drawn exactly as in fusion_verdict.py.
  Descriptive only (no verdict): K = 1..4 equal-weight curve (mean, min, max over all subsets), pairwise residual correlation,
      and heterogeneous combinations with the K1 members when given (K1S42=, K1S43=).
Usage: python verdict_A.py --out OUT.json S42=pred.pt S43=pred.pt S44=pred.pt S45=pred.pt [K1S42=pred.pt K1S43=pred.pt]
"""
import argparse, hashlib, itertools, json, sys
from pathlib import Path

import numpy as np
import torch

BOOT, SEED, GPT = 2000, 42, ("S42", "S43", "S44", "S45")
BAND_CENTER, BAND_HALF, BASELINE, MIN_NOISE, THRESHOLD, UPPER_PCT = 87.89, 1.5, 85.349, 1.08, -1.0, 99.0


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("members", nargs="+", metavar="NAME=PRED.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    paths = dict(m.split("=", 1) for m in args.members)
    assert set(GPT) <= set(paths) <= set(GPT) | {"K1S42", "K1S43"}, f"members must be {GPT} plus optional K1S42/K1S43"
    preds, truth, inputs = {}, None, {}
    for name, path in paths.items():
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        data = torch.load(path, map_location="cpu", weights_only=True)
        y = data["truth_ev"].double().numpy()
        truth = y if truth is None else truth
        assert np.array_equal(truth, y), f"{name}: truth differs; members must score the same rows"
        preds[name] = data["pred_ev"].double().numpy()
    rows = len(truth)
    ens = lambda names: sum(preds[n] for n in names) / len(names)
    mae = lambda names: float(np.abs(ens(names) - truth).mean() * 1000)
    single = {n: mae([n]) for n in GPT}
    band = {n: {"mae_mev": single[n], "inside": abs(single[n] - BAND_CENTER) <= BAND_HALF} for n in GPT}
    values = np.array([single[n] for n in GPT])
    noise = {"std_ddof1_mev": float(values.std(ddof=1)), "range_mev": float(values.max() - values.min()), "mean_M_mev": float(values.mean())}
    noise["N_mev"] = max(MIN_NOISE, noise["range_mev"])

    pair_mae, four_mae = mae(["S42", "S43"]), mae(GPT)
    assert abs(pair_mae - BASELINE) < 5e-4, f"S42+S43 pair {pair_mae:.4f} does not reproduce the recorded {BASELINE}"
    d = np.abs(ens(GPT) - truth) - np.abs(ens(["S42", "S43"]) - truth)
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, rows, rows) for _ in range(BOOT)]
    boots = np.array([d[i].mean() for i in idx]) * 1000
    diff = float(d.mean() * 1000)
    upper = float(np.percentile(boots, UPPER_PCT))
    criterion_pass = bool(diff <= THRESHOLD and upper < 0)
    in_band = band["S44"]["inside"] and band["S45"]["inside"]
    outcome = ("PASS" if criterion_pass else "FAIL") if in_band else "INCONCLUSIVE"

    curve = {}
    for k in range(1, 5):
        subsets = {"+".join(c): mae(c) for c in itertools.combinations(GPT, k)}
        v = np.array(list(subsets.values()))
        curve[k] = {"mean_mev": float(v.mean()), "min_mev": float(v.min()), "max_mev": float(v.max()), "subsets": subsets}
    resid = {n: preds[n] - truth for n in preds}
    corr = {}
    for a, b in itertools.combinations(sorted(preds), 2):
        corr[f"{a}-{b}"] = {"pearson": float(np.corrcoef(resid[a], resid[b])[0, 1]),
                            "spearman": float(np.corrcoef(ranks(resid[a]), ranks(resid[b]))[0, 1])}
    hetero = {}
    k1 = [n for n in ("K1S42", "K1S43") if n in preds]
    if k1:
        hetero["gpt4_plus_k1_each_equal"] = mae(list(GPT) + k1)
        g4, kp = ens(GPT), ens(k1)
        hetero["half_gpt4_half_k1"] = float(np.abs(0.5 * g4 + 0.5 * kp - truth).mean() * 1000)
        hetero["resid_corr_gpt4_vs_k1"] = float(np.corrcoef(g4 - truth, kp - truth)[0, 1])

    out = {"format": "molgap-gptrans-levers-verdict-a-v1", "role": "tail_holdout_v1", "rows": rows, "inputs": inputs,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "bootstrap": {"n": BOOT, "seed": SEED, "paired": True, "upper_percentile": UPPER_PCT},
           "single_mae_mev": single, "band": {"center_mev": BAND_CENTER, "half_width_mev": BAND_HALF, "members": band},
           "noise_floor": noise, "pair_s42_s43_mev": pair_mae, "four_member_mev": four_mae,
           "primary": {"minus": ["four", "pair_s42_s43"], "mev": diff, "threshold_mev": THRESHOLD, "upper_99_one_sided_mev": upper,
                       "upper_95_one_sided_mev": float(np.percentile(boots, 95)), "criterion_pass": criterion_pass,
                       "s44_s45_in_band": in_band, "outcome": outcome},
           "descriptive": {"ensemble_curve_equal_weight": curve, "residual_correlation": corr, "hetero_with_k1": hetero}}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("single_mae_mev", "noise_floor", "pair_s42_s43_mev", "four_member_mev", "primary")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
