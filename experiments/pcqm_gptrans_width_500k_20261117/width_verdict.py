"""Verdict on tail_holdout_v1 for protocol.md of the GPT-T 500K capacity question (equal weights, nothing fitted).

Written and committed before any run of this question was scored. Mapping of the frozen rules (names: R, L20, W320, F3; seeds 42, 43):
  gain(arm)   = mean over the two seeds of [tail MAE(R, seed) - tail MAE(arm, seed)], in meV; positive means the arm is better.
  paired(arm) = per row, mean over the two seeds of [|err arm| - |err R|] with the same seed; its bootstrap mean is the paired difference.
  Bootstrap: 2000 row resamples, numpy default_rng(42), the same index sets for every comparison; upper bound = the 99th percentile of the
  resampled paired difference (one-sided 99%). Rows are resampled, training runs are not: the bound does not cover training noise.
  NOMINATED (W320, F3 only) iff gain >= 1.5 meV AND the paired difference of each seed is below 0 AND the bootstrap upper bound is below 0.
  L20 (calibration, never nominated): gain <= 0.5 -> NO_SIGNAL; gain >= 0.9 -> PROXY_VALID; between -> UNCERTAIN; missing run -> UNAVAILABLE.
  An arm with any of R42, R43, arm42, arm43 missing is INCONCLUSIVE and its single-seed numbers are descriptive only.
  500K only nominates. Adopting anything needs the full-scale 1.5 meV gate again, outside this script.
Usage: python width_verdict.py --out OUT.json R42=p R43=p [L2042=p L2043=p] [W32042=p W32043=p] [F342=p F343=p]
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

BOOT, SEED, UPPER_PCT = 2000, 42, 99.0
NOMINATE_MEV, CAL_NONE_MEV, CAL_PROXY_MEV = 1.5, 0.5, 0.9
SEEDS, ARMS, NOMINEES = (42, 43), ("L20", "W320", "F3"), ("W320", "F3")
INIT_SD_500K_B128_MEV = 0.47  # earlier 500K GPT-T B128 init scatter; context only
EARLIER_500K_B1024_TAIL_MEV = {42: 111.672, 43: 110.958}  # earlier B1024 500K tail MAEs of the same two inits; descriptive regression check only
READING = {  # (arm outcome, calibration) -> consequence for the route; written before any score
    ("NOMINATED", None): "nominated for one full-scale test; adoption needs the full-scale 1.5 meV gate again",
    ("NOT_NOMINATED", "PROXY_VALID"): "no gain at 500K and 500K sees capacity: the route is negative at 500K",
    ("NOT_NOMINATED", "UNCERTAIN"): "no gain at 500K but the calibration is unclear: the route is neither closed nor reopened; no full-scale run follows from 500K alone",
    ("NOT_NOMINATED", "NO_SIGNAL"): "500K cannot see capacity effects: this zero result must not close the route",
    ("NOT_NOMINATED", "UNAVAILABLE"): "no calibration: this zero result must not close the route",
    ("INCONCLUSIVE", None): "a run is missing: rerun it under this protocol (a skipped or failed task may be resubmitted) and read it by the same rules",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("members", nargs="+", metavar="NAME=PRED.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    allowed = {f"{a}{s}" for a in ("R",) + ARMS for s in SEEDS}
    paths = dict(m.split("=", 1) for m in args.members)
    assert set(paths) <= allowed, f"members must be among {sorted(allowed)}"
    preds, truth, inputs = {}, None, {}
    for name, path in paths.items():
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        data = torch.load(path, map_location="cpu", weights_only=True)
        y = data["truth_ev"].double().numpy()
        truth = y if truth is None else truth
        assert np.array_equal(truth, y), f"{name}: truth differs; members must score the same rows"
        preds[name] = data["pred_ev"].double().numpy()
    rows = len(truth)
    err = {n: np.abs(p - truth) for n, p in preds.items()}
    mae = {n: float(e.mean() * 1000) for n, e in err.items()}
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, rows, rows) for _ in range(BOOT)]

    def compare(arm):
        names = [f"{x}{s}" for s in SEEDS for x in ("R", arm)]
        if not set(names) <= set(preds):
            return None
        per_seed = {s: mae[f"R{s}"] - mae[f"{arm}{s}"] for s in SEEDS}
        diff = sum(err[f"{arm}{s}"] - err[f"R{s}"] for s in SEEDS) / len(SEEDS)
        boots = np.array([diff[i].mean() for i in idx]) * 1000
        return {"gain_mev": float(np.mean(list(per_seed.values()))), "gain_by_seed_mev": per_seed,
                "paired_mev": float(diff.mean() * 1000), "upper_99_one_sided_mev": float(np.percentile(boots, UPPER_PCT))}

    cmp = {arm: compare(arm) for arm in ARMS}
    out = {"format": "molgap-gptrans-width-500k-verdict-v1", "role": "tail_holdout_v1", "rows": rows, "inputs": inputs,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "bootstrap": {"n": BOOT, "seed": SEED, "paired": True, "upper_percentile": UPPER_PCT, "rows_only_not_training_noise": True},
           "mae_mev": mae}
    if {"R42", "R43"} <= set(preds):
        out["R_seed_difference_mev"] = abs(mae["R42"] - mae["R43"])
        out["R_vs_earlier_500K_B1024_mev"] = {str(s): mae[f"R{s}"] - EARLIER_500K_B1024_TAIL_MEV[s] for s in SEEDS}
        out["context_init_sd_500K_B128_mev"] = INIT_SD_500K_B128_MEV
    cal = cmp["L20"]
    if cal is None:
        out["calibration"] = {"outcome": "UNAVAILABLE", "reason": "L20 or R missing for a seed"}
    else:
        g = cal["gain_mev"]
        out["calibration"] = {**cal, "outcome": "NO_SIGNAL" if g <= CAL_NONE_MEV else "PROXY_VALID" if g >= CAL_PROXY_MEV else "UNCERTAIN"}
    for arm in NOMINEES:
        c = cmp[arm]
        if c is None:
            r = {"outcome": "INCONCLUSIVE", "missing": sorted({f"{x}{s}" for s in SEEDS for x in ("R", arm)} - set(preds))}
            key = ("INCONCLUSIVE", None)
        else:
            ok = c["gain_mev"] >= NOMINATE_MEV and all(v > 0 for v in c["gain_by_seed_mev"].values()) and c["upper_99_one_sided_mev"] < 0
            r = {**c, "outcome": "NOMINATED" if ok else "NOT_NOMINATED"}
            key = ("NOMINATED", None) if ok else ("NOT_NOMINATED", out["calibration"]["outcome"])
        r["reading"] = READING[key]
        out[arm] = r
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "bootstrap")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
