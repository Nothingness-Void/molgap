"""Fit P0's corrector on rows spread over the index range; score held-back blocks once."""
import argparse
import csv
import gzip
import io
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter, process_time
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / "pcqm_pairnorm_residual_2d_p0"), str(HERE.parent / "pcqm_h1_heldout_budget")]
from chemistry import NAMES, bucket, feature_row, keys
import probe as p0
from prepare import pinned, read, write
from molgap.pcqm_fixed_datasets import OFFICIAL_ARCHIVE_SHA256
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import sha256_file

OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-residual-spread-fit")
PREV = Path("D:/molgap-artifacts/diagnostics/pcqm-residual-block-transfer")
FULL = Path("D:/molgap-data/pcqm_full_train_v1")
INTERIOR, BEYOND, FAR = np.arange(1850000, 1900000), np.arange(3150000, 3200000), np.arange(3200000, 3250000)
FRESH = {"interior": INTERIOR, "beyond": BEYOND}
SIZES = {"30k": 30155, "120k": 120000}
SEED, DRAW_SEED = 20261122, 20261123
CONTROLS = ("identity", "median", "affine", "isotonic", "gbm")


def draw():
    """The 120,000 fit rows in draw order; the first 30,155 are the small fit."""
    eligible = np.r_[np.arange(600000, 1850000), np.arange(1900000, 3150000)]
    return np.random.default_rng(DRAW_SEED).choice(eligible, SIZES["120k"], replace=False)


def pick(rows, field):
    """One CSV field for increasing row IDs. Other lines are not parsed; nothing past the last row is read."""
    rows = np.asarray(rows)
    assert np.all(np.diff(rows) > 0) and rows[0] >= 600000 and rows[-1] < 3250000
    wanted = iter(rows.tolist())
    target, values = next(wanted), []
    with ZipFile(pinned(p0.ARCHIVE, OFFICIAL_ARCHIVE_SHA256)) as z, z.open("pcqm4m-v2/raw/data.csv.gz") as f:
        with gzip.open(f, "rb") as gz, io.TextIOWrapper(gz, encoding="utf8") as lines:
            assert next(lines).strip() == "idx,smiles,homolumogap"
            for i, line in enumerate(lines):
                if i != target:
                    continue
                record = next(csv.reader([line]))
                assert int(record[0]) == i and len(record) == 3
                values.append(record[field])
                target = next(wanted, None)
                if target is None:
                    break
    assert len(values) == len(rows)
    return values


def save(name, **values):
    with (OUT / (name + ".npz")).open("xb") as f:
        np.savez(f, **values)


def load(folder, name):
    return np.load(pinned(folder / (name + ".npz"), read(folder / (name + ".json"))["sha256"]), allow_pickle=False)


def shard(num):
    from molgap.pcqm_topology import load_packed_topology_shard
    assert num == 10 or 12 <= num <= 64     # shard 65 starts the reserved tail
    m = read(pinned(FULL / "manifest.json", "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"))
    rec, = [r for r in m["assets"]["topology"] if r["source_idx_min"] == num*50000]
    assert rec["role"] == "train" and rec["source_idx_max"] == (num+1)*50000-1
    ds = load_packed_topology_shard(pinned(FULL / rec["file"], rec["sha256"]))
    assert np.array_equal(ds._data.source_idx.ravel().numpy(), np.arange(num*50000, (num+1)*50000))
    return ds


def featurize():
    assert len(NAMES) == 217
    rows = np.sort(np.r_[draw(), INTERIOR, BEYOND])
    assert len(np.unique(rows)) == len(rows) == 220000
    smiles = pick(rows, 1)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        scaffold, canonical = zip(*pool.map(keys, smiles, chunksize=256))
        x = np.empty((len(smiles), len(NAMES)+6+1024), np.float32)
        g = np.empty((len(smiles), 6), np.int16)
        for i, (v, tags) in enumerate(pool.map(feature_row, smiles, chunksize=128)):
            x[i], g[i] = v, tags
    save("features", idx=rows, x=x, g=g, scaffold=np.array(scaffold), canonical=np.array(canonical))
    return {"source_archive": OFFICIAL_ARCHIVE_SHA256, "rows": len(smiles), "invalid_rows": int((g[:, 0] < 0).sum())}


def predict_all():
    import torch
    sys.path.insert(0, str(ROOT / "platforms/local"))
    from pcqm_official_valid_eval import setup_determinism
    from pcqm_tail_holdout_eval import gptrans_scorer
    setup_determinism()
    torch.set_num_threads(2)
    fit = np.sort(draw())
    targets = {"fit": fit, **FRESH}
    models, hashes, checks = {}, {}, {}
    dev_store = shard(10)

    def run(arm, store, ids):
        model, predict, recipe = models[arm]
        values = []
        for j in range(0, len(ids), 512):
            batch = ids[j:j+512]
            graphs = [store[int(i % 50000)] for i in batch]
            assert np.array_equal([int(g.source_idx) for g in graphs], batch)
            values.append(predict(model, graphs, recipe.target_mean_eV, recipe.target_std_eV, "cuda", batch_size=64).numpy().ravel())
        return np.concatenate(values)

    for arm in p0.ARMS:
        files = p0.basefiles(arm)
        provenance = read(p0.PARENT / ("init_" + arm) / "initialization_provenance.json")
        init = pinned(p0.INITS / (arm + ".pt"), provenance["state_file_sha256"])
        load_model, predict, recipe = gptrans_scorer(files["contract"], init, "model")
        models[arm] = (load_model(files["selected_model"], "cuda"), predict, recipe)
        hashes[arm] = dict(init=sha256_file(init), **{k: sha256_file(p) for k, p in files.items()})
        dev = torch.load(files["predictions"], weights_only=True, map_location="cpu")
        ix = dev["source_idx"].ravel().argsort()[:128]
        assert np.array_equal(dev["source_idx"].ravel()[ix].numpy(), np.arange(500000, 500128))
        checks[arm] = float(np.max(np.abs(run(arm, dev_store, np.arange(500000, 500128))-dev["prediction_eV"].ravel()[ix].numpy())))
        assert checks[arm] <= 1e-4, checks[arm]
    del dev_store
    preds = {f"{name}_{arm}": np.full(len(ids), np.nan) for name, ids in targets.items() for arm in p0.ARMS}
    fit_y = np.full(len(fit), np.nan, np.float32)
    shards = sorted({int(n) for ids in targets.values() for n in np.unique(ids // 50000)})
    for num in shards:
        store = shard(num)
        for name, ids in targets.items():
            sel = np.flatnonzero(ids // 50000 == num)
            if not len(sel):
                continue
            for arm in p0.ARMS:
                preds[f"{name}_{arm}"][sel] = run(arm, store, ids[sel])
            if name == "fit":      # fit labels only; score-block labels stay unread
                fit_y[sel] = store._data.y.ravel().numpy()[ids[sel] % 50000]
        del store
        print("shard", num, flush=True)
    assert all(np.isfinite(v).all() for v in preds.values()) and np.isfinite(fit_y).all()
    assert np.array_equal(fit_y, np.array(pick(fit, 2), dtype=np.float32))
    save("predict", fit_idx=fit, fit_y=fit_y, **preds)
    return {"bindings": hashes, "replay_max_abs_eV": checks, "device": torch.cuda.get_device_name(), "shards": shards}


def correct():
    order = draw()
    fit = np.sort(order)
    with load(OUT, "features") as f, load(OUT, "predict") as q, load(PREV, "features") as ff, load(PREV, "predict") as fq:
        ids = f["idx"]
        pos = np.searchsorted(ids, fit)
        assert np.array_equal(ids[pos], fit) and np.array_equal(q["fit_idx"], fit)
        assert np.array_equal(ff["idx"], FAR) and np.array_equal(fq["idx"], FAR)
        x_all, scaffolds = f["x"], f["scaffold"][pos]
        fit_x, y = x_all[pos], q["fit_y"].astype(np.float64)
        backbone = {arm: q["fit_" + arm] for arm in p0.ARMS}
        blocks = {name: (x_all[np.searchsorted(ids, rows)], {a: q[f"{name}_{a}"] for a in p0.ARMS}) for name, rows in FRESH.items()}
        blocks["far"] = (ff["x"], {a: fq[a] for a in p0.ARMS})
        del x_all
    result, info = {}, {}
    for arm in p0.ARMS:
        p = backbone[arm]
        x = np.column_stack((fit_x, p.astype(np.float32)))
        for tag, n in SIZES.items():
            rows = np.sort(np.searchsorted(fit, order[:n]))
            a, b = p[rows], y[rows]
            val = np.array([bucket("control:" + k, 5) == 0 for k in scaffolds[rows]])
            losses = {k: float(np.abs(p0.calibrate(k, a[~val], b[~val], a[val])-b[val]).mean()) for k in CONTROLS}
            chosen = min(losses, key=losses.get)
            reg = p0.make_gbm().fit(x[rows], b-a)
            info[arm + "-" + tag] = {"control": chosen, "inner_rows": int(val.sum()), "fit_rows": len(rows),
                                     "fit_validation_mae_eV": losses}
            for name, (features, preds) in blocks.items():
                v = preds[arm]
                result[f"{name}-{arm}-{tag}-scalar"] = p0.calibrate(chosen, a, b, v)
                result[f"{name}-{arm}-{tag}-chem"] = v + reg.predict(np.column_stack((features, v.astype(np.float32))))
            print(arm, tag, chosen, flush=True)
    assert all(np.isfinite(v).all() and len(v) == 50000 for v in result.values())
    save("correct", fit_idx=fit, small_fit_idx=np.sort(order[:SIZES["30k"]]), interior_idx=INTERIOR, beyond_idx=BEYOND,
         far_idx=FAR, **result)
    return {"fits": info}


def score():
    with load(OUT, "correct") as c, load(OUT, "features") as f, load(OUT, "predict") as q, \
            load(PREV, "features") as ff, load(PREV, "predict") as fq, load(PREV, "correct") as single:
        ids, scaffolds, canon, groups = f["idx"], f["scaffold"], f["canonical"], f["g"][:, 0]
        assert np.array_equal(c["far_idx"], FAR) and np.array_equal(ff["idx"], FAR) and np.array_equal(fq["idx"], FAR)
        assert np.array_equal(single["far_idx"], FAR)
        fits = {"30k": np.searchsorted(ids, c["small_fit_idx"]), "120k": np.searchsorted(ids, c["fit_idx"])}
        sources = {}
        for name, rows in FRESH.items():
            at = np.searchsorted(ids, rows)
            assert np.array_equal(ids[at], rows) and np.array_equal(c[name + "_idx"], rows)
            sources[name] = (rows, scaffolds[at], canon[at], groups[at], {a: q[f"{name}_{a}"] for a in p0.ARMS})
        sources["far"] = (FAR, ff["scaffold"], ff["canonical"], ff["g"][:, 0], {a: fq[a] for a in p0.ARMS})
        blocks = {}
        for name, (rows, sc, can, g0, preds) in sources.items():
            y = np.array(pick(rows, 2), dtype=np.float32)
            assert np.array_equal(y, shard(int(rows[0] // 50000))._data.y.ravel().numpy()) and np.isfinite(y).all()
            y = y.astype(np.float64)
            block = {"rows": len(y), "invalid_rows": int((g0 < 0).sum())}
            for arm in p0.ARMS:
                raw = np.abs(preds[arm]-y)*1000
                for tag, pos in fits.items():
                    scalar, chem = (np.abs(c[f"{name}-{arm}-{tag}-{k}"]-y)*1000 for k in ("scalar", "chem"))
                    new = ~np.isin(sc, np.unique(scaffolds[pos]))
                    entry = {
                        "mae_meV": {"raw": float(raw.mean()), "scalar": float(scalar.mean()), "chemistry": float(chem.mean())},
                        "rescue": paired_bootstrap_mean(chem-raw, n_bootstrap=2000, seed=SEED),
                        "features": paired_bootstrap_mean(chem-scalar, n_bootstrap=2000, seed=SEED),
                        "unseen_fit_scaffolds": {"rows": int(new.sum()), "mae_meV": {"raw": float(raw[new].mean()),
                            "scalar": float(scalar[new].mean()), "chemistry": float(chem[new].mean())}},
                        "rows_with_fit_canonical": int((np.isin(can, np.unique(canon[pos])) & (can != "")).sum())}
                    if name == "far" and tag == "30k":
                        # Same rows, same fit size: spread fit minus the single-block fit.
                        old = np.abs(single[f"far-{arm}-30k-chem"]-y)*1000
                        entry["minus_single_block_fit"] = paired_bootstrap_mean(chem-old, n_bootstrap=2000, seed=SEED)
                    block[arm + "-" + tag] = entry
            blocks[name] = block
    primary = [blocks["beyond"][arm + "-120k"] for arm in p0.ARMS]
    passed = all(v["rescue"]["delta"] <= -2 and v["rescue"]["ci95"][1] < 0 and v["features"]["ci95"][1] < 0 for v in primary)
    failed = any(v["rescue"]["ci95"][0] > -2 or v["features"]["ci95"][0] >= 0 for v in primary)
    return {"verdict": "PASS" if passed else "FAIL" if failed else "INCONCLUSIVE", "blocks": blocks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["features", "predict", "correct", "score"])
    stage = parser.parse_args().stage
    assert (HERE / "rml/trajectory.json").is_file()
    OUT.mkdir(exist_ok=True)
    dest = HERE / "results.json" if stage == "score" else OUT / (stage + ".json")
    assert not dest.exists() and not (stage != "score" and (OUT/(stage+".npz")).exists())
    start, cpu = perf_counter(), process_time()
    res = {"features": featurize, "predict": predict_all, "correct": correct, "score": score}[stage]()
    if stage != "score":
        res.update(sha256=sha256_file(OUT/(stage+".npz")), score_labels_used=False)
    else:
        res["receipts"] = {k: read(OUT/(k+".json")) for k in ("features", "predict", "correct")}
    res.update(wall_seconds=perf_counter()-start, parent_cpu_seconds=process_time()-cpu,
               versions={k: version(k) for k in ("scikit-learn", "torch", "rdkit", "numpy")},
               prospective_sha256=sha256_file(HERE/"rml/trajectory.json"), source_sha256=sha256_file(Path(__file__)))
    write(dest, res)
    print(stage, dest, flush=True)
