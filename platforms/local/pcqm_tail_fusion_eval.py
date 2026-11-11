"""Route 1 verdict: equal-weight heterogeneous fusion vs homogeneous ensemble on tail_holdout_v1 (NO_TRAIN).

Members are per-row prediction files written by pcqm_tail_holdout_eval.py --save-predictions (.pt with pred_ev/truth_ev,
all on the same tail rows). Weights are fixed 0.5/0.5; nothing is fitted. Criteria (frozen, do not edit):
  1. hetero tail MAE - strongest single member <= -3 meV
  2. hetero - homo <= -3 meV
  3. paired row-bootstrap 95% upper bound of both differences < 0
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

THRESHOLD_MEV, BOOT, SEED = -3.0, 2000, 42


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--member", action="append", required=True, metavar="NAME=PRED.pt")
    parser.add_argument("--hetero", required=True, help="two member names, comma separated")
    parser.add_argument("--homo", required=True, help="two member names, comma separated")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    errors, truth = {}, None
    for text in args.member:
        name, path = text.split("=", 1)
        data = torch.load(path, map_location="cpu", weights_only=True)
        pred, y = data["pred_ev"].double().numpy(), data["truth_ev"].double().numpy()
        if truth is None:
            truth = y
        assert np.array_equal(truth, y), f"{name}: truth differs, members must score the same rows"
        errors[name] = pred
    hetero, homo = args.hetero.split(","), args.homo.split(",")
    for pair in (hetero, homo):
        assert len(pair) == 2 and all(n in errors for n in pair), "each group needs two known members"
    preds = errors
    rows = len(truth)
    single = {n: np.abs(p - truth) for n, p in preds.items()}
    fused = lambda pair: np.abs(0.5 * preds[pair[0]] + 0.5 * preds[pair[1]] - truth)
    err_het, err_homo = fused(hetero), fused(homo)
    best = min(single, key=lambda n: single[n].mean())
    diffs = {"hetero_minus_best_single": err_het - single[best], "hetero_minus_homo": err_het - err_homo}
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, rows, rows) for _ in range(BOOT)]
    report = {}
    for key, d in diffs.items():
        boots = np.array([d[i].mean() for i in idx]) * 1000
        report[key] = {"mev": float(d.mean() * 1000), "ci95_mev": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    crit = {"1_hetero_vs_best_single_le_-3meV": report["hetero_minus_best_single"]["mev"] <= THRESHOLD_MEV,
            "2_hetero_vs_homo_le_-3meV": report["hetero_minus_homo"]["mev"] <= THRESHOLD_MEV,
            "3_both_ci_upper_below_0": all(r["ci95_mev"][1] < 0 for r in report.values())}
    out = {"format": "molgap-tail-fusion-eval-v1", "rows": rows, "weights": [0.5, 0.5], "hetero": hetero, "homo": homo,
           "bootstrap": {"n": BOOT, "seed": SEED, "paired": True}, "single_mae_ev": {n: float(e.mean()) for n, e in single.items()},
           "strongest_single": best, "hetero_mae_ev": float(err_het.mean()), "homo_mae_ev": float(err_homo.mean()),
           "differences": report, "criteria": crit, "verdict": "PASS" if all(crit.values()) else "FAIL"}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
