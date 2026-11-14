"""Local analysis of direction-screen arms (CPU only, numpy). See DESIGN_opus.md section C.

Usage: analyze_local.py --runs-dir DIR (with runs/<arm>/ or <arm>/) --features features_v1.npz
                        [--historical-preds DIR_or_json] --out OUT
Delta = arm - baseline (negative = better). Baseline = framework mean of k1_l1_s42 / k1_l1_s43
(number = mean of the two per-seed MAEs; per-row paired baseline error = mean of the two |errors|).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

DEV0, DEV1 = 100_000, 150_000
NBOOT, SEED = 2000, 42
MEV = 1000.0


def load_pred(path):
    """Load source_idx,target_eV,prediction_eV from .npz or torch .pt (dict of tensors)."""
    p = Path(path)
    if p.suffix == ".pt":
        import torch
        d = torch.load(p, map_location="cpu", weights_only=False)
        return {k: np.asarray(d[k]) for k in ("source_idx", "target_eV", "prediction_eV")}
    z = np.load(p)
    return {k: z[k] for k in z.files}


def align(d, feat_tgt, feat_idx):
    """Return prediction sorted to source_idx 100000..149999; raise on any inconsistency."""
    si = np.asarray(d["source_idx"]).astype(np.int64)
    if si.shape != (DEV1 - DEV0,) or not np.array_equal(np.sort(si), np.arange(DEV0, DEV1)):
        raise ValueError("source_idx is not exactly 100000..149999")
    o = np.argsort(si)
    if not np.array_equal(feat_idx[si[o]], si[o]):
        raise ValueError("source_idx not aligned with features")
    t = np.asarray(d["target_eV"], np.float32)[o]
    if not np.allclose(t, feat_tgt[si[o]], atol=1e-5):
        raise ValueError("target_eV disagrees with features target_eV")
    p = np.asarray(d["prediction_eV"], np.float64)[o]
    if not np.isfinite(p).all():
        raise ValueError("non-finite prediction")
    return p


def rjson(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def build_buckets(y, tags):
    rad, rings, arom, heavy, het, chg = (tags[:, i] for i in range(6))
    B = {"all": np.ones(len(y), bool)}
    edges = [(-np.inf, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, np.inf)]
    for lo, hi in edges:
        if lo == -np.inf:
            nm = f"gap<{hi:g}"
        elif hi == np.inf:
            nm = f"gap>={lo:g}"
        else:
            nm = f"gap[{lo:g},{hi:g})"
        B[nm] = (y >= lo) & (y < hi)
    B["tail_gap[2,4)"] = (y >= 2) & (y < 4)
    B["tail_gap>=8"] = y >= 8
    B["radical=0"], B["radical=1"] = rad == 0, rad == 1
    for k in range(3):
        B[f"rings={k}"] = rings == k
        B[f"arom_rings={k}"] = arom == k
    B["rings>=3"], B["arom_rings>=3"] = rings >= 3, arom >= 3
    for nm, lo, hi in [("heavy<=12", 0, 12), ("heavy13-15", 13, 15), ("heavy16-18", 16, 18), ("heavy>=19", 19, 999)]:
        B[nm] = (heavy >= lo) & (heavy <= hi)
    for nm, lo, hi in [("hetero<=2", 0, 2), ("hetero3-4", 3, 4), ("hetero5-6", 5, 6), ("hetero>=7", 7, 999)]:
        B[nm] = (het >= lo) & (het <= hi)
    B["charged=0"], B["charged=1"] = chg == 0, chg > 0
    return B


class Boot:
    """Row-paired bootstrap shared by all arms/buckets (same resamples, seed 42)."""

    def __init__(self, n, masks):
        rng = np.random.default_rng(SEED)
        self.C = np.empty((NBOOT, n), np.uint8)
        for i in range(NBOOT):
            self.C[i] = np.minimum(np.bincount(rng.integers(0, n, n), minlength=n), 255)
        self.M = np.stack(list(masks.values()), 1).astype(np.float32)
        self.den = self._mm(self.M)

    def _mm(self, X):
        out = np.empty((NBOOT, X.shape[1]), np.float64)
        for s in range(0, NBOOT, 250):
            out[s:s + 250] = self.C[s:s + 250].astype(np.float32) @ X
        return out

    def delta_ci(self, d):
        """95% CI of mean(d) within each bucket (d = per-row arm_err - base_err)."""
        num = self._mm(self.M * d[:, None].astype(np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            b = num / self.den
        return np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)


def fit_pow(Ns, maes):
    if len(Ns) < 2:
        return None
    s, c = np.polyfit(np.log(np.asarray(Ns, float)), np.log(np.asarray(maes, float)), 1)
    return {"a": float(np.exp(c)), "b": float(-s)}


def nnls(X, y):
    from scipy.optimize import nnls as _n
    return _n(X, y)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--historical-preds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    F = np.load(a.features)
    f_idx, f_tgt, f_tags = F["source_idx"], F["target_eV"], F["tags"]
    y = f_tgt[DEV0:DEV1].astype(np.float64)
    B = build_buckets(y, f_tags[DEV0:DEV1])
    B = {n: m for n, m in B.items() if m.any()}  # drop empty buckets
    names = list(B)
    root = Path(a.runs_dir)
    root = root / "runs" if (root / "runs").is_dir() else root

    arms, failed = {}, {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            pr = align(load_pred(d / "dev_pred_final.npz"), f_tgt, f_idx)
        except Exception as e:
            failed[d.name] = f"{type(e).__name__}: {e}"
            continue
        arm = {"pred": pr, "curve": rjson(d / "curve.json"), "done": rjson(d / "done.json"), "probe": None}
        tp = d / "train_probe_final.npz"
        if tp.exists():
            try:
                z = np.load(tp)
                arm["probe"] = float(np.mean(np.abs(z["prediction_eV"].astype(np.float64) - z["target_eV"])))
            except Exception:
                pass
        arms[d.name] = arm

    def mae(p, m=None):
        e = np.abs(p - y)
        return float(e.mean() if m is None else e[m].mean())

    if not {"k1_l1_s42", "k1_l1_s43"} <= set(arms):
        raise SystemExit(f"baseline arms missing/failed: {failed}")
    p42, p43 = arms["k1_l1_s42"]["pred"], arms["k1_l1_s43"]["pred"]
    e42, e43 = np.abs(p42 - y), np.abs(p43 - y)
    ebase = (e42 + e43) / 2
    base_mae = float(ebase.mean())
    noise = max(1.63, abs(float(e42.mean() - e43.mean())) * MEV)
    base_bias = {n: float(((p42 - y)[m].mean() + (p43 - y)[m].mean()) / 2) for n, m in B.items()}
    base_b_mae = {n: float(ebase[m].mean()) for n, m in B.items()}
    boot = Boot(len(y), B)

    res = {}
    for k, ar in arms.items():
        p = ar["pred"]
        err = np.abs(p - y)
        lo, hi = boot.delta_ci(err - ebase)
        rows = {}
        for j, n in enumerate(names):
            m = B[n]
            if m.sum() == 0:
                continue
            bm = float(err[m].mean())
            d = bm - base_b_mae[n]
            rows[n] = {"n": int(m.sum()), "mae_meV": bm * MEV, "bias_meV": float((p - y)[m].mean() * MEV),
                       "base_mae_meV": base_b_mae[n] * MEV, "delta_meV": d * MEV,
                       "delta_pct": d / base_b_mae[n] * 100, "ci95_meV": [float(lo[j] * MEV), float(hi[j] * MEV)]}
        cur = ar["curve"]
        fin = best = None
        if isinstance(cur, list):
            v = [c["dev_mae_eV"] for c in cur if c.get("dev_mae_eV") is not None]
            if v:
                fin, best = v[-1] * MEV, min(v) * MEV
        res[k] = {"mae_meV": mae(p) * MEV, "buckets": rows, "curve_final_dev_meV": fin, "curve_best_dev_meV": best,
                  "done_status": (ar["done"] or {}).get("status"),
                  "train_probe_mae_meV": None if ar["probe"] is None else ar["probe"] * MEV}

    # ---------------- scaling ----------------
    def fam_points(fam):
        pts = {}
        for k in arms:
            mm = re.fullmatch(rf"{fam}_N(\d+)", k)
            if mm:
                pts[int(mm.group(1))] = k
        if fam == "k1":
            pts[100000] = "BASE"
        elif fam == "gine" and "gine_h192" in arms and 100000 not in pts:
            pts[100000] = "gine_h192"
        return dict(sorted(pts.items()))

    subs = {"all": None, "radical=1": B["radical=1"], "arom_rings=0": B["arom_rings=0"]}
    scaling = {}
    for fam in ("k1", "gine", "fpmlp"):
        pts = fam_points(fam)
        if len(pts) < 2:
            continue
        fs = {}
        for sn, m in subs.items():
            Ns, ms = list(pts), []
            for k in pts.values():
                if k == "BASE":
                    ms.append(float(np.mean(ebase if m is None else ebase[m])) * MEV)
                else:
                    ms.append(mae(arms[k]["pred"], m) * MEV)
            ratios = {f"{Ns[i]}/{Ns[i-1]}": ms[i] / ms[i - 1] for i in range(1, len(Ns))
                      if abs(Ns[i] / Ns[i - 1] - 2) < 1e-9}
            fs[sn] = {"N": Ns, "mae_meV": ms, "doubling_ratio": ratios, "powerlaw": fit_pow(Ns, ms)}
        scaling[fam] = fs
    data_lim = None
    if "k1_n50000" in arms:
        r = base_mae / mae(arms["k1_n50000"]["pred"])
        probes = [arms[k]["probe"] for k in ("k1_l1_s42", "k1_l1_s43") if arms[k]["probe"] is not None]
        gap = None if not probes else (base_mae - float(np.mean(probes))) / base_mae
        data_lim = {"ratio_100K_50K": r, "train_dev_gap_frac_of_dev": gap,
                    "signal": bool(r <= 0.93 and gap is not None and gap > 0.30),
                    "note": "train probe missing" if gap is None else ""}

    # ---------------- ensembles ----------------
    ens = {}
    seeds = [k for k in ("k1_l1_s42", "k1_l1_s43", "k1_l1_s44") if k in arms]
    ens["homogeneous_k1"] = {str(i): mae(np.mean([arms[k]["pred"] for k in seeds[:i]], 0)) * MEV
                             for i in range(1, len(seeds) + 1)}
    hist = {}
    if a.historical_preds:
        hp = Path(a.historical_preds)
        files = {}
        if hp.is_file() and hp.suffix == ".json":
            files = {n: Path(v) for n, v in json.loads(hp.read_text()).items()}
        elif hp.is_file():
            files = {hp.stem: hp}
        elif hp.is_dir():
            files = {f.stem: f for f in sorted(list(hp.glob("*.pt")) + list(hp.glob("*.npz")))}
        for n, f in files.items():
            try:
                hist[n] = align(load_pred(f), f_tgt, f_idx)
            except Exception as e:
                failed[f"hist:{n}"] = str(e)
    if hist:
        allp = [arms[k]["pred"] for k in seeds] + list(hist.values())
        ens["with_historical"] = {
            "note": "historical inits come from a separate pipeline",
            "hist_models": list(hist),
            "curve": {str(i): mae(np.mean(allp[:i], 0)) * MEV for i in range(1, len(allp) + 1)},
            "hist_only_each": {n: mae(p) * MEV for n, p in hist.items()}}
    fams = {"k1": np.mean([arms[k]["pred"] for k in seeds], 0)}
    for fam in ("gine", "fpmlp"):
        k = next((x for x in (f"{fam}_n100000", "gine_h192" if fam == "gine" else f"{fam}_base") if x in arms), None)
        if k:
            fams[fam] = arms[k]["pred"]
    if len(fams) >= 2:
        perm = np.random.default_rng(SEED).permutation(len(y))
        halves = [perm[: len(y) // 2], perm[len(y) // 2:]]
        X = np.stack(list(fams.values()), 1)
        pr = np.zeros(len(y))
        ws = []
        for i in range(2):
            tr, te = halves[i], halves[1 - i]
            w = nnls(X[tr], y[tr])
            ws.append(w.tolist())
            pr[te] = X[te] @ w
        ens["heterogeneous_crossfit"] = {"members": list(fams), "weights_per_fold": ws, "mae_meV": mae(pr) * MEV,
                                         "k1_alone_meV": mae(fams["k1"]) * MEV}

    # ---------------- signals ----------------
    sig = {}
    sub_names = [n for n in names if not n.startswith(("gap", "tail")) and n != "all"]
    for k, r in res.items():
        b = r["buckets"]
        o = b["all"]
        t = []
        if (abs(o["delta_pct"]) >= 5 and abs(o["delta_meV"]) >= 3 * noise
                and not (o["ci95_meV"][0] <= 0 <= o["ci95_meV"][1])):
            t.append("整体信号(" + ("更好" if o["delta_meV"] < 0 else "更差") + ")")
        for n in sub_names:
            x = b.get(n)
            if x and x["n"] >= 2000 and x["delta_pct"] <= -10 and x["ci95_meV"][1] < 0 and o["delta_pct"] <= 2:
                t.append(f"子群信号({n})")
        for n in ("tail_gap[2,4)", "tail_gap>=8"):
            x = b.get(n)
            if x and abs(base_bias[n]) > 1e-9 and abs(x["bias_meV"]) <= 0.7 * abs(base_bias[n]) * MEV:
                t.append(f"尾部信号({n})")
        if re.search(r"w384|d18|ep80", k) and o["delta_pct"] <= -5:
            t.append("容量/优化受限")
        if k in ("k1_l1_s42", "k1_l1_s43"):
            t = []
        sig[k] = t or ["描述"]
    if data_lim and data_lim["signal"]:
        sig["k1_n50000"] = [x for x in sig["k1_n50000"] if x != "描述"] + ["数据受限"]

    summary = {"n_dev": len(y), "baseline_mae_meV": base_mae * MEV,
               "seed_mae_meV": {"s42": float(e42.mean() * MEV), "s43": float(e43.mean() * MEV)},
               "noise_floor_meV": noise, "failed_arms": failed, "arms": res, "signals": sig, "scaling": scaling,
               "data_limited": data_lim, "ensembles": ens}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# 方向筛查分析摘要", "",
         f"开发集 n={len(y)};基线(k1_l1 s42/s43 两 seed MAE 均值)= {base_mae*MEV:.2f} meV;"
         f"噪声底 = {noise:.2f} meV;Δ = 臂 - 基线(负=更好);CI 为行配对 bootstrap 95%(2000 次,seed 42)。", "",
         "## 各臂总体", "",
         "| 臂 | MAE meV | Δ meV | Δ% | CI95 meV | 自由基=1 MAE | 无芳环 MAE | 信号 |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        o = r["buckets"]["all"]
        rd, na = r["buckets"].get("radical=1"), r["buckets"].get("arom_rings=0")
        L.append(f"| {k} | {r['mae_meV']:.2f} | {o['delta_meV']:+.2f} | {o['delta_pct']:+.2f} | "
                 f"[{o['ci95_meV'][0]:+.2f}, {o['ci95_meV'][1]:+.2f}] | {rd['mae_meV']:.1f} | {na['mae_meV']:.1f} | "
                 f"{'、'.join(sig[k])} |")
    for k, why in failed.items():
        L.append(f"| {k} | 失败 | | | | | | {why} |")
    L += ["", "## 分桶(n / MAE meV(偏差, Δ [CI]))", "",
          "| 桶 | n | 基线 MAE | " + " | ".join(res) + " |", "|---|---|---|" + "---|" * len(res)]
    first = next(iter(res))
    for n in names:
        if n not in res[first]["buckets"]:
            continue
        cells = []
        for k in res:
            x = res[k]["buckets"][n]
            cells.append(f"{x['mae_meV']:.1f} (偏{x['bias_meV']:+.0f}, Δ{x['delta_meV']:+.1f} "
                         f"[{x['ci95_meV'][0]:+.1f},{x['ci95_meV'][1]:+.1f}])")
        L.append(f"| {n} | {res[first]['buckets'][n]['n']} | {base_b_mae[n]*MEV:.1f} | " + " | ".join(cells) + " |")
    if scaling:
        L += ["", "## 缩放(MAE meV 随 N;描述性幂律 a·N^-b)", ""]
        for fam, fs in scaling.items():
            for sn, s in fs.items():
                rt = ", ".join(f"{k}: {v:.3f}" for k, v in s["doubling_ratio"].items())
                L.append(f"- {fam}/{sn}: N={s['N']} MAE={[round(v, 1) for v in s['mae_meV']]} 翻倍比={{{rt}}} "
                         f"斜率 b={s['powerlaw']['b']:.3f}")
    if data_lim:
        L += ["", f"数据受限判据:100K/50K = {data_lim['ratio_100K_50K']:.3f},训练探针差距占比 = "
                  f"{data_lim['train_dev_gap_frac_of_dev']},信号 = {data_lim['signal']} {data_lim['note']}"]
    L += ["", "## 集成", "",
          "- 同构 K1(前 k 个 seed 均值预测)MAE meV:" + str({k: round(v, 2) for k, v in ens["homogeneous_k1"].items()})]
    h = ens.get("with_historical")
    if h:
        L.append(f"- 并入历史 init(另一套流水线,单独标注){h['hist_models']}:"
                 f"{ {k: round(v, 2) for k, v in h['curve'].items()} };历史单模型:"
                 f"{ {k: round(v, 2) for k, v in h['hist_only_each'].items()} }")
    h = ens.get("heterogeneous_crossfit")
    if h:
        L.append(f"- 异构交叉拟合 {h['members']}:MAE {h['mae_meV']:.2f} meV(K1 均值单独 {h['k1_alone_meV']:.2f});"
                 f"权重 {h['weights_per_fold']}")
    L += ["", "判据未满足的一律为“描述”。子群/尾部判据对多个桶逐一检验,未做多重比较校正,属线索而非结论。"]
    (out / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
