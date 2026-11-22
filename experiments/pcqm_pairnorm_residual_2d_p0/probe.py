"""Local P0 stages; retain immutable row outputs, then open score labels for analysis."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from importlib.metadata import version
import io
from itertools import islice
import json
from pathlib import Path
import sys
from time import perf_counter, process_time
from zipfile import ZipFile

import numpy as np

from chemistry import COUNTS, NAMES, bucket, feature_row, keys
from molgap.pcqm_fixed_datasets import OFFICIAL_ARCHIVE_SHA256
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-pairnorm-residual-2d-p0")
ARCHIVE = Path("D:/molgap-data/ogb_pcqm4mv2_raw/pcqm4m-v2.zip")
DATA = Path("D:/molgap-data/pcqm_full_train_v1/store/topology/train")
RUN = Path("D:/molgap-artifacts/retrieved/pcqm-gptrans-pairnorm-init-noise-500k-s43/v1/version1")
INITS = Path("D:/molgap-artifacts/initializations/gptrans-pairnorm-init-noise-500k")
PARENT = ROOT / "experiments/pcqm_gptrans_pairnorm_init_noise_500k"
ARMS = ["s43", "s44"]
MANIFESTS = ["9f5131f307df7ee373e066c36afa96f17c84767c8712ead844b8073ce77457ac",
             "2a439ae42a10179897ebf51d11a8a6db03d45029647c15781ec84af86e7a0f0a"]
SHARDS = {8: "1c00ca8c6c0eb1baf63b06a51bf239652373d38112d57256b1e70913119bf79c",
          9: "39aeb3e52f340442c33272139bb5d900ee34698bd4a5a11d64ceb176441b45e2",
          10: "3879dc27106034c27c893bac7d182d54fbbe93906efb83108e9dac541e0894ae"}
SEED = 20261122


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verified(path, digest):
    check(sha256_file(path) == digest, f"Identity mismatch: {path}")
    return path


def save(name, **arrays):
    path = OUT / (name + ".npz")
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def read(name):
    receipt = json.loads((OUT / (name + ".json")).read_text())
    return np.load(verified(OUT / (name + ".npz"), receipt["sha256"]), allow_pickle=False)


def features():
    from multiprocessing import get_context

    check(len(NAMES) == 217, "RDKit descriptor registry changed")
    start = 400000
    with ZipFile(verified(ARCHIVE, OFFICIAL_ARCHIVE_SHA256)) as z, z.open("pcqm4m-v2/raw/data.csv.gz") as member:
        with gzip.open(member, "rb") as f, io.TextIOWrapper(f, encoding="utf-8") as text:
            check(next(text).strip() == "idx,smiles,homolumogap", "CSV schema")
            records = list(csv.reader(islice(text, start, 550000)))
    ids = np.array([int(i) for i, _, _ in records])
    check(np.array_equal(ids, np.arange(start, 550000)), "Raw ID range")
    smiles = [s for _, s, _ in records]
    # Cast to the official graph-cache label dtype before comparing with retained tensors.
    target = np.array([y for _, _, y in records], dtype=np.float32)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        groups = list(pool.map(keys, smiles, chunksize=256))
        scaffolds = np.array([k for k, _ in groups])
        canonical = np.array([k for _, k in groups])
        even = np.array([bucket(k, 2) == 0 for k in scaffolds])
        a, b = np.flatnonzero(even & (ids < 500000)), np.flatnonzero(even & (ids >= 500000))
        n = min(len(a), len(b))
        rng = np.random.default_rng(SEED)
        take = lambda x: x if len(x) == n else np.sort(rng.choice(x, n, replace=False))
        i_train, i_fit = take(a), take(b)
        i_score = np.flatnonzero(~even & (ids >= 500000))
        positions = np.r_[i_train, np.arange(100000, 150000)]
        check(not (set(scaffolds[i_score]) & set(scaffolds[np.r_[i_train, i_fit]])), "Scaffold leak")
        print(f"Feature rows {len(positions)}, matched fit {n}, score {len(i_score)}", flush=True)
        x = np.empty((len(positions), len(NAMES) + 6 + 1024), np.float32)
        g = np.empty((len(positions), 6), np.int16)
        for i, (row, tags) in enumerate(pool.map(feature_row, (smiles[p] for p in positions), chunksize=128)):
            x[i], g[i] = row, tags
            if (i + 1) % 10000 == 0:
                print(f"Featurized {i + 1}/{len(positions)}", flush=True)
    save("features", idx=ids[positions], y=target[positions], x=x, groups=g,
         scaffold=scaffolds[positions], canonical=canonical[positions], devfit=ids[i_fit], score=ids[i_score])
    return {"feature_names": NAMES + COUNTS + ["MorganR2count:" + str(i) for i in range(1024)],
            "source_archive_sha256": OFFICIAL_ARCHIVE_SHA256, "eligible_train": len(a), "eligible_dev": len(b),
            "matched_fit": n, "score": len(i_score), "invalid_feature_rows": int((g[:, 0] == -1).sum()),
            "nonfinite_cells": int((~np.isfinite(x)).sum())}


def basefiles(arm):
    folder = RUN / ("init_" + arm)
    manifest = json.loads(verified(folder / "output_manifest.json", MANIFESTS[ARMS.index(arm)]).read_text())
    check(manifest["context"]["arm_id"] == "init_" + arm, "Arm mismatch")
    return {k: verified(folder / v["path"], v["sha256"])
            for k, v in manifest["artifacts"].items() if k in {"contract", "selected_model", "predictions"}}


def retained(files, f):
    import torch
    from molgap.research_memory.saved_analysis import load_aligned

    paths = {arm: bindings["predictions"] for arm, bindings in files.items()}
    names, errors, ids = load_aligned(paths)
    check(names == ARMS and np.array_equal(ids, np.arange(500000, 550000)), "Prediction IDs")
    dev_y = f["y"][f["idx"] >= 500000]
    result = {}
    for i, arm in enumerate(names):
        payload = torch.load(paths[arm], weights_only=True, map_location="cpu")
        order = payload["source_idx"].view(-1).argsort()
        actual = payload["target_eV"].flatten()[order].numpy()
        check(np.array_equal(actual, dev_y), "SMILES/raw label vs saved target")
        result[arm] = payload["prediction_eV"].flatten()[order].double().numpy()
        check(np.allclose(result[arm] - dev_y, errors[i], rtol=0, atol=1e-12), "Saved error alignment")
    return result


def replay():
    import torch
    from molgap.pcqm_topology import load_packed_topology_shard

    sys.path.insert(0, str(ROOT / "platforms/local"))
    from pcqm_tail_holdout_eval import gptrans_scorer
    from pcqm_official_valid_eval import setup_determinism

    setup_determinism()
    torch.set_num_threads(2)
    f = read("features")
    ids, truth = f["idx"], f["y"]
    it = ids[ids < 500000]
    files = {arm: basefiles(arm) for arm in ARMS}
    dev = retained(files, f)
    shards = {s: load_packed_topology_shard(verified(DATA / f"train_shard_{s:04d}.pt", h)) for s, h in SHARDS.items()}
    result, deltas = {}, {}
    for arm in ARMS:
        init = json.loads((PARENT / ("init_" + arm) / "initialization_provenance.json").read_text())
        load, predict, recipe = gptrans_scorer(files[arm]["contract"], verified(INITS / (arm + ".pt"), init["state_file_sha256"]), "model")
        model = load(files[arm]["selected_model"], "cuda")
        def run(rowids):
            blocks = []
            for offset in range(0, len(rowids), 512):
                batch = rowids[offset:offset + 512]
                graphs = [shards[int(i // 50000)][int(i % 50000)] for i in batch]
                check(np.array_equal([int(g.source_idx) for g in graphs], batch), "Shard order")
                check(np.array_equal([float(g.y) for g in graphs], truth[np.searchsorted(ids, batch)]), "Shard label")
                blocks.append(predict(model, graphs, recipe.target_mean_eV, recipe.target_std_eV, "cuda", batch_size=64).numpy().reshape(-1))
            return np.concatenate(blocks)
        diff = np.abs(run(np.arange(500000, 500128)) - dev[arm][:128]).max()
        check(diff <= 1e-4, f"{arm} prediction replay: {diff}")
        result[arm] = np.r_[run(it), dev[arm]]
        check(np.isfinite(result[arm]).all(), "Nonfinite replay")
        deltas[arm] = float(diff)
    save("replay", idx=ids, **result)
    return {"replay_max_abs_eV": deltas, "train_predictions_per_init": len(it),
            "local_device": torch.cuda.get_device_name(), "torch": torch.__version__, "manifest_sha256": MANIFESTS}


def make_gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=500, max_leaf_nodes=63,
                                         learning_rate=.1, early_stopping=False, random_state=0)


def calibrate(name, p, y, q):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import QuantileRegressor

    if name == "identity":
        return q
    if name == "median":
        return q + np.median(y-p)
    if name == "affine":
        return QuantileRegressor(quantile=.5, alpha=0, solver="highs").fit(p[:, None], y).predict(q[:, None])
    if name == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(q)
    return q + make_gbm().fit(p[:, None], y-p).predict(q[:, None])


def fit_one(arm, source):
    f, r = read("features"), read("replay")
    ids = f["idx"]
    check(np.array_equal(ids, r["idx"]), "Replay order")
    idx = np.flatnonzero(ids < 500000) if source == "train" else np.searchsorted(ids, f["devfit"])
    score = np.searchsorted(ids, f["score"])
    val = np.array([bucket("control:" + k, 5) == 0 for k in f["scaffold"][idx]])
    p, y = r[arm], f["y"][idx].astype(np.float64)
    fitp, q = p[idx], p[score]
    losses = {k: float(np.abs(calibrate(k, fitp[~val], y[~val], fitp[val]) - y[val]).mean())
              for k in ("identity", "median", "affine", "isotonic", "gbm")}
    chosen = min(losses, key=losses.get)
    control = calibrate(chosen, fitp, y, q)
    x = np.column_stack((f["x"], p.astype(np.float32)))
    stack = q + make_gbm().fit(x[idx], y-fitp).predict(x[score])
    check(np.isfinite(stack).all() and np.isfinite(control).all(), "Nonfinite corrector")
    save(arm + "-" + source, idx=ids[score], fit_idx=ids[idx], raw=q, control=control, stack=stack)
    return {"control": chosen, "fit_only_validation_mae_eV": losses, "fit_rows": len(idx), "score_rows": len(score)}


def analyze():
    from molgap.router import paired_bootstrap_mean

    f, r = read("features"), read("replay")
    ids = f["idx"]
    y = f["y"][np.searchsorted(ids, f["score"])].astype(np.float64)
    results = {}
    for arm in ARMS:
        for source in ("dev", "train"):
            key = arm + "-" + source
            v = read(key)
            check(np.array_equal(v["idx"], f["score"]), "Scoring row mismatch")
            err = {k: np.abs(v[k]-y)*1000 for k in ("raw", "control", "stack")}
            results[key] = {"mae_meV": {k: float(e.mean()) for k, e in err.items()},
                            "stack_minus_control_meV": paired_bootstrap_mean(err["stack"]-err["control"], n_bootstrap=2000, seed=SEED)}
    delta = [results[a + "-dev"]["stack_minus_control_meV"]["delta"] for a in ARMS]
    verdict = "PASS" if max(delta) <= -2 else "KILL" if max(delta) > -1 else "INCONCLUSIVE"
    mask = ids >= 500000
    e = np.stack([r[a][mask]-f["y"][mask] for a in ARMS])*1000
    ae = np.abs(e)
    comiss = (e[0]*e[1] > 0) & (ae.min(axis=0) >= 150)
    rad, no_pi, rings, chno, halogen, sp = f["groups"][mask].T
    heavy = f["x"][mask, NAMES.index("HeavyAtomCount")]
    can = f["canonical"][mask]
    counts = Counter(can)
    sels = {"radical": rad == 1, "no_pi": no_pi == 1, "CHNO_only": chno == 1, "halogen": halogen == 1,
            "S_or_P": sp == 1, "heavy_le10": heavy <= 10, "heavy_11_15": (heavy > 10) & (heavy <= 15),
            "heavy_ge16": heavy >= 16, "duplicate": np.array([bool(s) and counts[s] > 1 for s in can]),
            "invalid": rad < 0, **{f"rings_{i}": rings == i for i in range(3)}, "rings_ge3": rings >= 3}
    strata = {}
    for k, s in sels.items():
        if not s.any():
            strata[k] = {"rows": 0, "flag": False}
            continue
        mae = ae[:, s].mean(axis=1)
        ratio, share = mae/ae.mean(axis=1), ae[:, s].sum(axis=1)/ae.sum(axis=1)
        rate = float(comiss[s].mean())
        enrich = rate/comiss.mean() if comiss.any() else 0.
        strata[k] = dict(rows=int(s.sum()), mae_meV=mae.tolist(), ratio=ratio.tolist(), share=share.tolist(),
                         comiss_rate=rate, enrichment=float(enrich), flag=bool(min(ratio)>=1.5 and min(share)>=.1 and enrich>=1.5))
    return {"primary": verdict, "results": results, "n4_global_mae_meV": ae.mean(axis=1).tolist(),
            "n4_global_comiss_rate": float(comiss.mean()), "n4": strata,
            "stage_receipts": {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["features", "replay", *[a+"-"+s for a in ARMS for s in ("dev", "train")], "analyze"])
    stage = parser.parse_args().stage
    check((HERE/"rml/trajectory.json").is_file(), "Prospective publication required")
    OUT.mkdir(exist_ok=True)
    dest = HERE/"analysis.json" if stage == "analyze" else OUT/(stage+".json")
    check(not dest.exists(), f"Already completed: {stage}")
    wall, cpu = perf_counter(), process_time()
    if stage == "analyze":
        result = analyze()
    elif stage == "replay":
        result = replay()
    elif stage == "features":
        result = features()
    else:
        result = fit_one(*stage.split("-"))
    if stage != "analyze":
        result["sha256"] = sha256_file(OUT/(stage+".npz"))
    result.update(wall_seconds=perf_counter()-wall, parent_process_cpu_seconds=process_time()-cpu,
                  packages={n: version(n) for n in ("numpy", "rdkit", "scikit-learn", "torch")},
                  prospective_sha256=sha256_file(HERE/"rml/trajectory.json"), analyzer_sha256=sha256_file(__file__),
                  chemistry_sha256=sha256_file(HERE/"chemistry.py"))
    atomic_json(dest, result)
    print(stage, "complete", dest, flush=True)


if __name__ == "__main__":
    main()
