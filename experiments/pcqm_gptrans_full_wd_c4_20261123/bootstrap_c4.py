"""Paired row bootstrap of (C4 - reference s43) tail MAE. 2000 draws, numpy default_rng(42), 95% interval.
Reference per-row predictions: the s43 file saved by the same scorer (tail_eval_full_minus_tail_20261115).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

A = Path("D:/molgap-artifacts")
REF = A / "tail_eval_full_minus_tail_20261115/preds/gptrans_minus_tail_s43_tail_holdout_v1.pt"
NEW = A / "tail_eval_c4_wd022_20261123/preds/gptrans_c4_wd022_s43_tail_holdout_v1.pt"
OUT = Path(__file__).with_name("results.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"STOP: {OUT} exists")
    r, n = torch.load(REF), torch.load(NEW)
    assert torch.equal(r["truth_ev"], n["truth_ev"]) and len(r["truth_ev"]) == 128606
    er = (r["pred_ev"].double() - r["truth_ev"].double()).abs().numpy() * 1000
    en = (n["pred_ev"].double() - n["truth_ev"].double()).abs().numpy() * 1000
    d = en - er
    rng = np.random.default_rng(42)
    m = len(d)
    draws = np.empty(2000)
    for i in range(2000):
        draws[i] = d[rng.integers(0, m, m)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    res = {
        "n_rows": m, "ref_mae_meV": float(er.mean()), "c4_mae_meV": float(en.mean()),
        "delta_meV": float(d.mean()), "boot_draws": 2000, "seed": 42,
        "ci95_meV": [float(lo), float(hi)], "boot_sd_meV": float(draws.std(ddof=1)),
        "frac_rows_c4_better": float((d < 0).mean()),
        "inputs": {"ref_pred_sha256": sha(REF), "c4_pred_sha256": sha(NEW)},
    }
    gate = ("real" if res["delta_meV"] <= -1.0 and hi < 0 else
            "inconclusive" if -1.0 < res["delta_meV"] <= -0.5 else
            "no_effect" if res["delta_meV"] > -0.5 else "inconclusive_ci_not_below_zero")
    res["gate"] = gate
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=2))


main()
