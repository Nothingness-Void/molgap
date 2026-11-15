"""Fusion verdict on tail_holdout_v1 for protocol.md + protocol_supplement_second_init.md (NO_TRAIN, equal weights, nothing fitted).

Written before any fusion number was read. Mapping of the frozen rules:
  original (protocol.md): hetero = 0.5*K1 s42 + 0.5*GPT-T s42; diff = hetero - best of {K1 s42, GPT-T s42};
      PASS iff diff <= -3 meV and the 95% paired-bootstrap upper bound < 0. "95% upper bound" is read as the upper end of the
      two-sided 95% percentile interval (97.5th percentile), as pcqm_tail_fusion_eval.py computed it for earlier fusion verdicts.
  supplement (three comparisons, Bonferroni): upper bound = one-sided 98.3% (100*(1-0.05/3) = 98.33th percentile), as written.
      (a) hetero pair - better homogeneous pair (K1 s42+s43 or GPT-T s42+s43, the lower tail MAE): <= -3.0 and upper < 0,
          else "ensemble gain only";
      (b) 4-member (0.25 each) - strongest single of the four: <= -3.0 and upper < 0, else 4-member is not a candidate;
      (c) 4-member - hetero pair: <= -1.0 and upper < 0, else the final candidate stays at 2 members.
  Bootstrap: 2000 row resamples, numpy default_rng(42), the same index sets for every difference (paired), drawn exactly as in
  pcqm_tail_fusion_eval.py. The two-sided 98.3% upper end (99.17th percentile) is reported too, descriptive only.
Usage: python fusion_verdict.py --out OUT.json K1S42=pred.pt K1S43=pred.pt GPTS42=pred.pt GPTS43=pred.pt
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

BOOT, SEED, NAMES = 2000, 42, ("K1S42", "K1S43", "GPTS42", "GPTS43")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("members", nargs=4, metavar="NAME=PRED.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    paths = dict(m.split("=", 1) for m in args.members)
    assert set(paths) == set(NAMES), f"members must be exactly {NAMES}"
    preds, truth, inputs = {}, None, {}
    for name in NAMES:
        inputs[name] = {"path": str(Path(paths[name]).resolve()), "sha256": hashlib.sha256(Path(paths[name]).read_bytes()).hexdigest()}
        data = torch.load(paths[name], map_location="cpu", weights_only=True)
        y = data["truth_ev"].double().numpy()
        truth = y if truth is None else truth
        assert np.array_equal(truth, y), f"{name}: truth differs; members must score the same rows"
        preds[name] = data["pred_ev"].double().numpy()
    rows = len(truth)
    err = lambda *names: np.abs(sum(preds[n] for n in names) / len(names) - truth)
    e = {n: err(n) for n in NAMES}
    e["hetero"], e["homo_k1"], e["homo_gpt"], e["four"] = err("K1S42", "GPTS42"), err("K1S42", "K1S43"), err("GPTS42", "GPTS43"), err(*NAMES)
    mae = {k: float(v.mean() * 1000) for k, v in e.items()}
    best2 = min(("K1S42", "GPTS42"), key=lambda n: mae[n])
    best4 = min(NAMES, key=lambda n: mae[n])
    homo = min(("homo_k1", "homo_gpt"), key=lambda n: mae[n])
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, rows, rows) for _ in range(BOOT)]

    def compare(a, b, threshold, upper_pct):
        d = e[a] - e[b]
        boots = np.array([d[i].mean() for i in idx]) * 1000
        upper = float(np.percentile(boots, upper_pct))
        mev = float(d.mean() * 1000)
        return {"minus": [a, b], "mev": mev, "threshold_mev": threshold, "upper_percentile": upper_pct, "upper_mev": upper,
                "upper_95_one_sided_mev": float(np.percentile(boots, 95)), "upper_983_two_sided_mev": float(np.percentile(boots, 100 - 100 * 0.05 / 6)),
                "pass": bool(mev <= threshold and upper < 0)}

    bonf = 100 * (1 - 0.05 / 3)
    original = compare("hetero", best2, -3.0, 97.5)
    sup = {"a_hetero_minus_better_homo": compare("hetero", homo, -3.0, bonf),
           "b_four_minus_strongest_single": compare("four", best4, -3.0, bonf),
           "c_four_minus_hetero": compare("four", "hetero", -1.0, bonf)}
    out = {"format": "molgap-full-minus-tail-fusion-verdict-v1", "role": "tail_holdout_v1", "rows": rows, "inputs": inputs,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "bootstrap": {"n": BOOT, "seed": SEED, "paired": True},
           "mae_mev": mae, "strongest_single_of_pair": best2, "strongest_single_of_four": best4, "better_homogeneous_pair": homo,
           "noise_floor_mev": {"K1": abs(mae["K1S42"] - mae["K1S43"]), "GPT-T": abs(mae["GPTS42"] - mae["GPTS43"])},
           "original": {**original, "verdict": "PASS" if original["pass"] else "FAIL"},
           "supplement": {**sup, "outcomes": {
               "a": "PASS" if sup["a_hetero_minus_better_homo"]["pass"] else "ensemble gain only",
               "b": "PASS" if sup["b_four_minus_strongest_single"]["pass"] else "4-member not a candidate",
               "c": "PASS" if sup["c_four_minus_hetero"]["pass"] else "final candidate stays at 2 members"}}}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
