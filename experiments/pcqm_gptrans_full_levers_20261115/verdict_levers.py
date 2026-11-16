"""Arms B, C, D, E verdict on tail_holdout_v1 for protocol.md (equal weights, nothing fitted).

Written and committed before any C, D, B256 tail number was computed. Mapping of the frozen rules:
  M, sigma: mean and sample std (ddof 1) of the four B1024 tail MAEs (S42..S45); bar = max(1.5, 3 sigma) meV.
  C, D: PASS iff tail(arm) <= M - bar and the paired per-row |err| difference against B1024 S42 has a one-sided 99% upper bound < 0.
  B: PASS iff mean(B S42, B S43) <= M - bar and, for each init (B s42 vs S42, B s43 vs S43), the paired upper bound < 0.
  E: members are every B1024 init plus every supplied run of C, D and B, equal weight per run. Reference = the lowest tail MAE among
     the all-B1024 ensemble and the B-arm ensemble (if both B inits are given). PASS iff (E - reference) <= -1.0 meV and upper bound < 0.
  Bootstrap: 2000 row resamples, numpy default_rng(42), the same index sets for every comparison (as verdict_A.py). Five comparisons
  (A, B, C, D, E), Bonferroni: one-sided 99% upper bound (99th percentile of the bootstrap difference).
An arm whose predictions are not given is reported as pending. E is only computed when told with --final (all arms in).
Usage: python verdict_levers.py --out OUT.json S42=p S43=p S44=p S45=p [C=p] [D=p] [B42=p B43=p] [--final]
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

BOOT, SEED, B1024 = 2000, 42, ("S42", "S43", "S44", "S45")
MIN_BAR, UPPER_PCT, E_THRESHOLD = 1.5, 99.0, -1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("members", nargs="+", metavar="NAME=PRED.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", action="store_true", help="all arms are in: also compute E")
    args = ap.parse_args(argv)
    paths = dict(m.split("=", 1) for m in args.members)
    allowed = set(B1024) | {"C", "D", "B42", "B43"}
    assert set(B1024) <= set(paths) <= allowed, f"members must be {B1024} plus any of C, D, B42, B43"
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
    err = lambda names: np.abs(ens(names) - truth)

    values = np.array([mae([n]) for n in B1024])
    M, sigma = float(values.mean()), float(values.std(ddof=1))
    bar = max(MIN_BAR, 3 * sigma)
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, rows, rows) for _ in range(BOOT)]

    def paired(a, b):
        d = err(a) - err(b)
        boots = np.array([d[i].mean() for i in idx]) * 1000
        return {"mev": float(d.mean() * 1000), "upper_99_one_sided_mev": float(np.percentile(boots, UPPER_PCT))}

    def vs_mean(arm_mae, pair):
        return {"tail_mev": arm_mae, "threshold_mev": M - bar, "meets_bar": bool(arm_mae <= M - bar), "paired": pair}

    out = {"format": "molgap-gptrans-levers-verdict-levers-v1", "role": "tail_holdout_v1", "rows": rows, "inputs": inputs,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "bootstrap": {"n": BOOT, "seed": SEED, "paired": True, "upper_percentile": UPPER_PCT, "comparisons": 5},
           "single_mae_mev": {n: mae([n]) for n in paths}, "M_mev": M, "sigma_mev": sigma, "bar_mev": bar}
    for arm in ("C", "D"):
        if arm in preds:
            r = vs_mean(mae([arm]), paired([arm], ["S42"]))
            r["outcome"] = "PASS" if r["meets_bar"] and r["paired"]["upper_99_one_sided_mev"] < 0 else "FAIL"
            out[arm] = r
        else:
            out[arm] = {"outcome": "pending"}
    if {"B42", "B43"} <= set(preds):
        pairs = {"s42": paired(["B42"], ["S42"]), "s43": paired(["B43"], ["S43"])}
        r = vs_mean(mae(["B42", "B43"]), pairs)
        r["outcome"] = "PASS" if r["meets_bar"] and all(p["upper_99_one_sided_mev"] < 0 for p in pairs.values()) else "FAIL"
        out["B"] = r
    else:
        out["B"] = {"outcome": "pending"}
    if args.final:
        members = list(preds)
        same = {"all_B1024": list(B1024)}
        if {"B42", "B43"} <= set(preds):
            same["B_arm"] = ["B42", "B43"]
        best = min(same, key=lambda k: mae(same[k]))
        e = paired(members, same[best])
        e.update({"E_mev": mae(members), "members": members, "reference": best, "reference_mev": mae(same[best]),
                  "threshold_mev": E_THRESHOLD})
        e["outcome"] = "PASS" if e["mev"] <= E_THRESHOLD and e["upper_99_one_sided_mev"] < 0 else "FAIL"
        out["E"] = e
    else:
        out["E"] = {"outcome": "pending"}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "bootstrap")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
