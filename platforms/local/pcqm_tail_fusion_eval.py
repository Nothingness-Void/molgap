"""Route 1 verdict: equal-weight heterogeneous fusion vs homogeneous ensemble on tail_holdout_v1 (NO_TRAIN).

Members are per-row prediction files written by pcqm_tail_holdout_eval.py --save-predictions (.pt with pred_ev/truth_ev,
all on the same tail rows). Weights are fixed 0.5/0.5; nothing is fitted. Criteria (frozen, do not edit):
  1. hetero tail MAE - strongest single member <= -3 meV
  2. hetero - homo <= -3 meV
  3. paired row-bootstrap 95% upper bound of both differences < 0
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

THRESHOLD_MEV, BOOT, SEED = -3.0, 2000, 42


def residual_report(args, preds, truth, inputs):
    """DESCRIPTIVE only: residual r=pred-truth correlations per pair; the optimal-weight fusion is fitted on these same rows, never a selection basis."""
    from scipy.stats import spearmanr
    pairs = [p.split(",") for p in (args.pairs or []) + [g for g in (args.hetero, args.homo) if g]]
    assert pairs and all(len(p) == 2 and all(n in preds for n in p) for p in pairs), "--pairs needs known member names, two per pair"
    edges = np.quantile(truth, np.linspace(0, 1, 11))
    decile = np.clip(np.searchsorted(edges, truth, side="right") - 1, 0, 9)
    res = {n: p - truth for n, p in preds.items()}
    out_pairs = []
    for a, b in pairs:
        ra, rb = res[a], res[b]
        cov = np.cov(ra, rb)
        w = float((cov[1, 1] - cov[0, 1]) / (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
        mae = lambda x: float(np.abs(x).mean())
        buckets = [{"decile": d, "truth_range_ev": [float(truth[m].min()), float(truth[m].max())], "rows": int(m.sum()), "mae_ev": {a: mae(ra[m]), b: mae(rb[m])},
                    "fused_5050_mae_ev": mae(0.5 * (ra[m] + rb[m])), "residual_pearson": float(np.corrcoef(ra[m], rb[m])[0, 1])} for d in range(10) for m in [decile == d]]
        out_pairs.append({"pair": [a, b], "residual_pearson": float(np.corrcoef(ra, rb)[0, 1]), "residual_spearman": float(spearmanr(ra, rb)[0]),
                          "mae_ev": {a: mae(ra), b: mae(rb)}, "residual_sd_ev": {a: float(ra.std()), b: float(rb.std())}, "fused_5050_mae_ev": mae(0.5 * (ra + rb)),
                          "descriptive_optimal_weight_a": w, "descriptive_optimal_fused_mae_ev": mae(w * ra + (1 - w) * rb), "deciles_by_truth": buckets})
    out = {"format": "molgap-tail-residual-report-v1", "descriptive": True, "note": "optimal weights are fitted on the evaluated rows; descriptive, not a selection basis",
           "rows": len(truth), "inputs": inputs, "pairs": out_pairs}
    Path(args.residual_report).write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--member", action="append", default=[], metavar="NAME=PRED.pt")
    parser.add_argument("--hetero", help="two member names, comma separated")
    parser.add_argument("--homo", help="two member names, comma separated")
    parser.add_argument("--spec", help='frozen JSON {"members": {name: path}, "hetero": [a,b], "homo": [c,d]}')
    parser.add_argument("--out")
    parser.add_argument("--residual-report", metavar="OUT.json", help="descriptive residual-correlation diagnostic; needs --pairs or --hetero/--homo")
    parser.add_argument("--pairs", nargs="+", metavar="A,B", help="member pairs for --residual-report")
    args = parser.parse_args(argv)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    members, spec_sha = {}, None
    if args.spec:
        spec, spec_sha = json.loads(Path(args.spec).read_text(encoding="utf-8")), sha(args.spec)
        members.update(spec["members"])
        args.hetero = args.hetero or ",".join(spec["hetero"])
        args.homo = args.homo or ",".join(spec["homo"])
    members.update(t.split("=", 1) for t in args.member)
    if args.residual_report:
        args.hetero = args.homo = args.hetero or args.homo or ""
    else:
        assert args.out, "--out is required"
    assert members and (args.residual_report or (args.hetero and args.homo)), "need members, --hetero and --homo (directly or via --spec)"
    errors, truth, inputs = {}, None, {}
    for name, path in members.items():
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": sha(path)}
        data = torch.load(path, map_location="cpu", weights_only=True)
        pred, y = data["pred_ev"].double().numpy(), data["truth_ev"].double().numpy()
        if truth is None:
            truth = y
        assert np.array_equal(truth, y), f"{name}: truth differs, members must score the same rows"
        errors[name] = pred
    if args.residual_report:
        return residual_report(args, errors, truth, inputs)
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
    out = {"format": "molgap-tail-fusion-eval-v1", "rows": rows, "inputs": inputs, "spec_sha256": spec_sha, "weights": [0.5, 0.5], "hetero": hetero, "homo": homo,
           "bootstrap": {"n": BOOT, "seed": SEED, "paired": True}, "single_mae_ev": {n: float(e.mean()) for n, e in single.items()},
           "strongest_single": best, "hetero_mae_ev": float(err_het.mean()), "homo_mae_ev": float(err_homo.mean()),
           "differences": report, "criteria": crit, "verdict": "PASS" if all(crit.values()) else "FAIL"}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
