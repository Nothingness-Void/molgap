"""C3 full-scale tail: MC-dropout (K=32) vs deterministic for the four B1024 minus-tail GPT-T models.

Reuses probe_c3 (prepare_batches, run_det, run_mc) and the tail scorer's recipe/loader. Two stages:
  predict: label-free. Streams one npz per (model, chunk) to OUT, resumable. Never reads labels.
  score:   reads labels once, writes results.json. Refuses to run if results.json exists (tail is scored once).
Single process, no DataLoader workers (num_workers stays 0).
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MINUS_TAIL = Path("D:/molgap-worktrees/gptrans-full-minus-tail-20261110")  # holds the scorer that rebuilt these recipes
RETRIEVED = Path("D:/molgap-artifacts/retrieved")
DATA_ROOT = Path("D:/molgap-data/pcqm_full_train_v1")
SAVED_PREDS = {"s42": "tail_eval_full_minus_tail_20261115", "s43": "tail_eval_full_minus_tail_20261115",
               "s44": "tail_eval_full_levers_20261116", "s45": "tail_eval_full_levers_20261116"}
MODELS = {"s42": ("pcqm-gptrans-b1024-minus-tail-20261114-s42", "gptrans_init.pt"),
          "s43": ("pcqm-gptrans-b1024-minus-tail-s43-20261115", "gptrans_init_s43.pt"),
          "s44": ("pcqm-gptrans-b1024-minus-tail-s44-20261115", "gptrans_init_s44.pt"),
          "s45": ("pcqm-gptrans-b1024-minus-tail-s45-20261115", "gptrans_init_s45.pt")}
TAIL = (3_250_000, 3_378_606)
SHARD_ROWS, CHUNK = 50_000, 8192
SEED_BASE = 20261123
BOOT_SEED, BOOT_N = 20261123, 2000
DET_TAIL_ENSEMBLE_MEV = 83.992  # recorded in the master leaderboard / decision_A.md


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def graphs_for(start, stop):
    from molgap.pcqm_topology import load_packed_topology_shard
    out = []
    for s in range(start // SHARD_ROWS, (stop - 1) // SHARD_ROWS + 1):
        out += list(load_packed_topology_shard(DATA_ROOT / "store" / "topology" / "train" / f"train_shard_{s:04d}.pt"))
    first = int(out[0].source_idx)
    out = out[start - first:stop - first]
    assert [int(g.source_idx) for g in out] == list(range(start, stop)), "stored shards do not cover the rows in order"
    return out


def load_models(device):
    spec = importlib.util.spec_from_file_location("tail_eval_runner", MINUS_TAIL / "platforms/local/pcqm_tail_holdout_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MINUS_TAIL / "platforms/local"))
    spec.loader.exec_module(mod)
    models, recipe0, meta = {}, None, {}
    for name, (folder, init) in MODELS.items():
        d = RETRIEVED / folder
        recipe = mod.runner_recipe(d / "run_config.json")
        load, _, _ = mod.gptrans_scorer(recipe, d / init, "model")
        models[name] = load(d / "ema_model.pt", device)
        meta[name] = {"ema_model_sha256": sha256(d / "ema_model.pt"), "run_config_sha256": sha256(d / "run_config.json"),
                      "mean_eV": recipe.target_mean_eV, "std_eV": recipe.target_std_eV}
        recipe0 = recipe0 or recipe
    meta["_scorer_sha256"] = sha256(MINUS_TAIL / "platforms/local/pcqm_tail_holdout_eval.py")
    return models, recipe0, meta


def predict(out, rows, k):
    import probe_c3 as c3
    from molgap.family_training_kernel import mode_policy
    start, stop = rows
    assert not (out / "results.json").exists(), "STOP: results.json exists; this endpoint was already scored"
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    device = "cuda"
    c3.MC_K = k
    models, recipe, meta = load_models(device)
    (out / "models.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    policy = mode_policy(recipe.family, recipe.mode)
    graphs = graphs_for(start, stop)
    ids_sha = hashlib.sha256(json.dumps(list(range(start, stop))).encode()).hexdigest()
    (out / "rows.json").write_text(json.dumps({"rows": [start, stop], "k": k, "chunk": CHUNK, "source_idx_sha256": ids_sha}), encoding="utf-8")
    t_all = perf_counter()
    for ci, a in enumerate(range(0, stop - start, CHUNK)):
        b = min(a + CHUNK, stop - start)
        todo = [m for m in MODELS if not (out / f"{m}_chunk{ci:03d}.npz").exists()]
        if not todo:
            continue
        batches = c3.prepare_batches(graphs[a:b], policy)
        for mi, m in enumerate(MODELS):
            if m not in todo:
                continue
            t0 = perf_counter()
            mean, std = meta[m]["mean_eV"], meta[m]["std_eV"]
            det = c3.run_det(models[m], batches, mean, std, device)
            c3.MC_SEED = SEED_BASE + mi * 10_000_000 + ci * 1000
            (mc8, mc16, mc32, sd), n_drop = c3.run_mc(models[m], batches, mean, std, device)
            tmp = out / f"{m}_chunk{ci:03d}.tmp.npz"
            with tmp.open("wb") as f:
                np.savez(f, det=det, mc8=mc8, mc16=mc16, mc32=mc32, std=sd, n_drop=n_drop)
            tmp.replace(out / f"{m}_chunk{ci:03d}.npz")
            print(f"{m} chunk {ci} rows {start+a}:{start+b} {perf_counter()-t0:.0f}s dropout_modules={n_drop} "
                  f"total {perf_counter()-t_all:.0f}s", flush=True)
        del batches
    print("predict done", flush=True)


def gather(out, rows, m):
    n = rows[1] - rows[0]
    parts = [np.load(out / f"{m}_chunk{ci:03d}.npz") for ci in range(-(-n // CHUNK))]
    return {key: np.concatenate([p[key] for p in parts]) for key in ("det", "mc8", "mc16", "mc32", "std")}


def score(out, rows):
    from molgap.router import paired_bootstrap_mean
    res_path = out / "results.json"
    assert not res_path.exists(), "STOP: results.json exists; this endpoint was already scored"
    start, stop = rows
    meta = json.loads((out / "models.json").read_text())
    y = torch.cat([g.y for g in graphs_for(start, stop)]).double().numpy().ravel()
    pred = {m: gather(out, rows, m) for m in MODELS}
    names = list(MODELS)
    ae = lambda p: np.abs(p - y) * 1000.0
    err = {}
    for m in names:
        for key in ("det", "mc8", "mc16", "mc32"):
            err[(m, key)] = ae(pred[m][key])
    for key in ("det", "mc8", "mc16", "mc32"):
        err[("ens", key)] = ae(np.mean([pred[m][key] for m in names], axis=0))
    boot = lambda d: paired_bootstrap_mean(d, n_bootstrap=BOOT_N, seed=BOOT_SEED)
    result = {"rows": [start, stop], "n": int(len(y)), "mae_meV": {}, "mc32_minus_det_meV": {}, "attribution": {}}
    for who in names + ["ens"]:
        result["mae_meV"][who] = {k: float(err[(who, k)].mean()) for k in ("det", "mc8", "mc16", "mc32")}
        result["mc32_minus_det_meV"][who] = boot(err[(who, "mc32")] - err[(who, "det")])
        result["mae_meV"][who]["mc8_minus_det"] = None
    for who in names + ["ens"]:
        result["mae_meV"][who].pop("mc8_minus_det")
    result["primary"] = {"ens_mc32_mae_meV": float(err[("ens", "mc32")].mean()),
                         "recorded_det_ens_meV": DET_TAIL_ENSEMBLE_MEV,
                         "ens_mc32_minus_recorded_det_meV": float(err[("ens", "mc32")].mean()) - DET_TAIL_ENSEMBLE_MEV,
                         "ens_mc32_minus_same_path_det_meV": result["mc32_minus_det_meV"]["ens"]}
    d = result["primary"]["ens_mc32_minus_same_path_det_meV"]["delta"]
    # attribution
    from scipy.stats import spearmanr
    att = result["attribution"]
    for m in names:
        sd = pred[m]["std"] * 1000.0
        att[m] = {"mean_mc_std_meV": float(sd.mean()),
                  "pearson_std_abs_err_det": float(np.corrcoef(sd, err[(m, "det")])[0, 1]),
                  "spearman_std_abs_err_det": float(spearmanr(sd, err[(m, "det")])[0]),
                  "mean_abs_mc_minus_det_meV": float(np.abs(pred[m]["mc32"] - pred[m]["det"]).mean() * 1000.0)}
        q = np.quantile(sd, [0.25, 0.5, 0.75])
        bins = np.digitize(sd, q)
        att[m]["gain_by_std_quartile_meV"] = [float((err[(m, "mc32")] - err[(m, "det")])[bins == b].mean()) for b in range(4)]
    det_stack = np.stack([pred[m]["det"] for m in names]); mc_stack = np.stack([pred[m]["mc32"] for m in names])
    res_det = (det_stack - y) * 1000.0; res_mc = (mc_stack - y) * 1000.0
    offd = lambda r: float(np.mean([np.corrcoef(r[i], r[j])[0, 1] for i in range(4) for j in range(i + 1, 4)]))
    att["ensemble"] = {"mean_pairwise_resid_corr_det": offd(res_det), "mean_pairwise_resid_corr_mc32": offd(res_mc),
                       "member_mean_mae_det": float(np.mean([result["mae_meV"][m]["det"] for m in names])),
                       "member_mean_mae_mc32": float(np.mean([result["mae_meV"][m]["mc32"] for m in names])),
                       "mean_ens_std_across_members_det_meV": float(det_stack.std(axis=0).mean() * 1000.0),
                       "mean_ens_std_across_members_mc32_meV": float(mc_stack.std(axis=0).mean() * 1000.0),
                       "mc_noise_in_ens_mean_meV": float(np.sqrt(np.mean([(pred[m]["std"] ** 2).mean() for m in names]) / (4 * 32)) * 1000.0)}
    # same-path vs earlier recorded predictions (batch 1024 scorer)
    off = start if start < TAIL[0] else start - TAIL[0]
    suffix = "development" if start < TAIL[0] else "tail_holdout_v1"
    stem = {"s42": "gptrans_minus_tail_s42", "s43": "gptrans_minus_tail_s43", "s44": "gptrans_minus_tail_s44", "s45": "gptrans_minus_tail_s45"}
    parity = {}
    for m in names:
        saved = torch.load(Path("D:/molgap-artifacts") / SAVED_PREDS[m] / "preds" / f"{stem[m]}_{suffix}.pt", map_location="cpu", weights_only=True)
        sp = saved["pred_ev"].double().numpy().ravel()[off:off + len(y)]
        parity[m] = {"max_abs_diff_eV": float(np.abs(sp - pred[m]["det"]).max()), "saved_mae_meV": float(ae(sp).mean()), "same_path_mae_meV": float(err[(m, "det")].mean())}
    sp_ens = np.mean([torch.load(Path("D:/molgap-artifacts") / SAVED_PREDS[m] / "preds" / f"{stem[m]}_{suffix}.pt", map_location="cpu", weights_only=True)["pred_ev"].double().numpy().ravel()[off:off + len(y)] for m in names], axis=0)
    parity["ens"] = {"saved_mae_meV": float(ae(sp_ens).mean()), "same_path_mae_meV": float(err[("ens", "det")].mean())}
    result["det_parity_vs_saved"] = parity
    result["meta"] = meta
    result["script_sha256"] = sha256(__file__)
    res_path.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"mae": result["mae_meV"], "primary": result["primary"], "parity": parity}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "score"])
    ap.add_argument("--rows", default=f"{TAIL[0]}:{TAIL[1]}")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = tuple(int(x) for x in a.rows.split(":"))
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/pcqm_c3_checkpoint_space"),
                    str(ROOT / "experiments/pcqm_pairnorm_residual_2d_p0"), str(ROOT / "experiments/pcqm_h1_heldout_budget")]
    out = Path(a.out)
    if a.stage == "predict":
        predict(out, rows, a.k)
    else:
        score(out, rows)
