"""Move only the score rows: P0's fit and corrector, scored on near and far blocks."""
import argparse
import csv
import gzip
import io
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from itertools import islice
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

OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-residual-block-transfer")
H1_OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-h1-heldout-budget")
FULL = Path("D:/molgap-data/pcqm_full_train_v1")
NEAR, FAR = np.arange(550000, 600000), np.arange(3200000, 3250000)
SEED = 20261122
CONTROLS = ("identity", "median", "affine", "isotonic", "gbm")


def column(rows, field):
    """One CSV field for a row range; never reads past the last requested row."""
    assert rows[-1] < 3250000
    with ZipFile(pinned(p0.ARCHIVE, OFFICIAL_ARCHIVE_SHA256)) as z, z.open("pcqm4m-v2/raw/data.csv.gz") as f:
        with gzip.open(f, "rb") as gz, io.TextIOWrapper(gz, encoding="utf8") as lines:
            assert next(lines).strip() == "idx,smiles,homolumogap"
            recs = list(csv.reader(islice(lines, int(rows[0]), int(rows[-1]) + 1)))
    assert np.array_equal([int(r[0]) for r in recs], rows)
    return [r[field] for r in recs]


def save(name, **values):
    with (OUT / (name + ".npz")).open("xb") as f:
        np.savez(f, **values)


def load(folder, name):
    return np.load(pinned(folder / (name + ".npz"), read(folder / (name + ".json"))["sha256"]), allow_pickle=False)


def shard(num):
    from molgap.pcqm_topology import load_packed_topology_shard
    assert num in (10, 11, 64)
    m = read(pinned(FULL / "manifest.json", "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"))
    rec, = [r for r in m["assets"]["topology"] if r["source_idx_min"] == num*50000]
    assert rec["role"] == "train" and rec["source_idx_max"] == (num+1)*50000-1
    ds = load_packed_topology_shard(pinned(FULL / rec["file"], rec["sha256"]))
    assert np.array_equal(ds._data.source_idx.ravel().numpy(), np.arange(num*50000, (num+1)*50000))
    return ds


def featurize():
    assert len(NAMES) == 217
    smiles = column(FAR, 1)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        scaffold, canonical = zip(*pool.map(keys, smiles, chunksize=256))
        x = np.empty((len(smiles), len(NAMES)+6+1024), np.float32)
        g = np.empty((len(smiles), 6), np.int16)
        for i, (v, tags) in enumerate(pool.map(feature_row, smiles, chunksize=128)):
            x[i], g[i] = v, tags
    save("features", idx=FAR, x=x, g=g, scaffold=np.array(scaffold), canonical=np.array(canonical))
    return {"source_archive": OFFICIAL_ARCHIVE_SHA256, "rows": len(smiles), "invalid_rows": int((g[:, 0] < 0).sum())}


def predict_far():
    import torch
    sys.path.insert(0, str(ROOT / "platforms/local"))
    from pcqm_official_valid_eval import setup_determinism
    from pcqm_tail_holdout_eval import gptrans_scorer
    setup_determinism()
    torch.set_num_threads(2)
    stores = {n: shard(n) for n in (10, 64)}
    result, checks, hashes = {}, {}, {}
    for arm in p0.ARMS:
        files = p0.basefiles(arm)
        provenance = read(p0.PARENT / ("init_" + arm) / "initialization_provenance.json")
        init = pinned(p0.INITS / (arm + ".pt"), provenance["state_file_sha256"])
        load_model, predict, recipe = gptrans_scorer(files["contract"], init, "model")
        model = load_model(files["selected_model"], "cuda")
        def run(ids):
            values = []
            for j in range(0, len(ids), 512):
                batch = ids[j:j+512]
                graphs = [stores[int(i//50000)][int(i%50000)] for i in batch]
                assert np.array_equal([int(g.source_idx) for g in graphs], batch)
                values.append(predict(model, graphs, recipe.target_mean_eV, recipe.target_std_eV, "cuda", batch_size=64).numpy().ravel())
            return np.concatenate(values)
        dev = torch.load(files["predictions"], weights_only=True, map_location="cpu")
        ix = dev["source_idx"].ravel().argsort()[:128]
        assert np.array_equal(dev["source_idx"].ravel()[ix].numpy(), np.arange(500000, 500128))
        checks[arm] = float(np.max(np.abs(run(np.arange(500000, 500128))-dev["prediction_eV"].ravel()[ix].numpy())))
        assert checks[arm] <= 1e-4, checks[arm]
        result[arm] = run(FAR)
        hashes[arm] = dict(init=sha256_file(init), **{k: sha256_file(p) for k, p in files.items()})
        del model
    assert all(np.isfinite(v).all() for v in result.values())
    save("predict", idx=FAR, **result)
    return {"bindings": hashes, "replay_max_abs_eV": checks, "device": torch.cuda.get_device_name()}


def correct():
    with p0.read("features") as f, p0.read("replay") as r:
        ids, base, target, scaffolds = f["idx"], f["x"], f["y"], f["scaffold"]
        assert np.array_equal(ids, r["idx"])
        fit, old = np.searchsorted(ids, f["devfit"]), np.searchsorted(ids, f["score"])
        old_ids = f["score"]
        backbone = {arm: r[arm] for arm in p0.ARMS}
    assert len(fit) == 30155 and ids[fit].min() >= 500000 and ids[fit].max() < 550000
    small = np.sort(np.random.default_rng(SEED).choice(len(fit), 6000, replace=False))
    with load(H1_OUT, "features") as nf, load(H1_OUT, "predict") as nq, load(OUT, "features") as ff, load(OUT, "predict") as fq:
        assert np.array_equal(nf["idx"][6000:], NEAR) and np.array_equal(nq["idx"], NEAR)
        assert np.array_equal(ff["idx"], FAR) and np.array_equal(fq["idx"], FAR)
        blocks = {"near": (nf["x"][6000:], {a: nq[a] for a in p0.ARMS}), "far": (ff["x"], {a: fq[a] for a in p0.ARMS})}
    y = target[fit].astype(np.float64)
    result, info = {}, {}
    for arm in p0.ARMS:
        p = backbone[arm]
        x = np.column_stack((base, p.astype(np.float32)))
        for tag, rows in (("30k", np.arange(len(fit))), ("6k", small)):
            pos = fit[rows]
            a, b = p[pos], y[rows]
            val = np.array([bucket("control:" + k, 5) == 0 for k in scaffolds[pos]])
            losses = {k: float(np.abs(p0.calibrate(k, a[~val], b[~val], a[val])-b[val]).mean()) for k in CONTROLS}
            chosen = min(losses, key=losses.get)
            reg = p0.make_gbm().fit(x[pos], b-a)
            info[arm + "-" + tag] = {"control": chosen, "inner_rows": int(val.sum()), "fit_rows": len(pos),
                                     "fit_validation_mae_eV": losses}
            if tag == "30k":
                # The refit must be P0's corrector, not merely P0's recipe.
                with p0.read(arm + "-dev") as saved:
                    assert np.array_equal(saved["idx"], old_ids) and np.array_equal(saved["fit_idx"], ids[pos])
                    gap = float(np.max(np.abs(p[old]+reg.predict(x[old])-saved["stack"])))
                    gap_control = float(np.max(np.abs(p0.calibrate(chosen, a, b, p[old])-saved["control"])))
                assert chosen == read(p0.OUT / (arm + "-dev.json"))["control"] and max(gap, gap_control) <= 1e-4, (chosen, gap, gap_control)
                info[arm + "-30k"]["p0_refit_max_abs_eV"] = {"chemistry": gap, "control": gap_control}
            for name, (features, preds) in blocks.items():
                q = preds[arm]
                result[f"{name}-{arm}-{tag}-scalar"] = p0.calibrate(chosen, a, b, q)
                result[f"{name}-{arm}-{tag}-chem"] = q + reg.predict(np.column_stack((features, q.astype(np.float32))))
    assert all(np.isfinite(v).all() and len(v) == 50000 for v in result.values())
    save("correct", near_idx=NEAR, far_idx=FAR, fit_idx=ids[fit], small_fit_idx=ids[fit[small]], **result)
    return {"fits": info}


def score():
    with p0.read("features") as f:
        ids, scaffolds, canon = f["idx"], f["scaffold"], f["canonical"]
    with load(OUT, "correct") as c, load(H1_OUT, "features") as nf, load(H1_OUT, "predict") as nq, \
            load(OUT, "features") as ff, load(OUT, "predict") as fq:
        assert np.array_equal(c["near_idx"], NEAR) and np.array_equal(c["far_idx"], FAR)
        fits = {"30k": np.searchsorted(ids, c["fit_idx"]), "6k": np.searchsorted(ids, c["small_fit_idx"])}
        blocks = {}
        for name, rows, num, feats, preds, cut in (("near", NEAR, 11, nf, nq, 6000), ("far", FAR, 64, ff, fq, 0)):
            assert np.array_equal(feats["idx"][cut:], rows) and np.array_equal(preds["idx"], rows)
            y = np.array(column(rows, 2), dtype=np.float32)
            assert np.array_equal(y, shard(num)._data.y.ravel().numpy()) and np.isfinite(y).all()
            y = y.astype(np.float64)
            sc, can = feats["scaffold"][cut:], feats["canonical"][cut:]
            block = {"rows": len(y), "invalid_rows": int((feats["g"][cut:, 0] < 0).sum())}
            for arm in p0.ARMS:
                raw = np.abs(preds[arm]-y)*1000
                for tag, pos in fits.items():
                    scalar, chem = (np.abs(c[f"{name}-{arm}-{tag}-{k}"]-y)*1000 for k in ("scalar", "chem"))
                    new = ~np.isin(sc, np.unique(scaffolds[pos]))
                    block[arm + "-" + tag] = {
                        "mae_meV": {"raw": float(raw.mean()), "scalar": float(scalar.mean()), "chemistry": float(chem.mean())},
                        "rescue": paired_bootstrap_mean(chem-raw, n_bootstrap=2000, seed=SEED),
                        "features": paired_bootstrap_mean(chem-scalar, n_bootstrap=2000, seed=SEED),
                        "unseen_fit_scaffolds": {"rows": int(new.sum()), "mae_meV": {"raw": float(raw[new].mean()),
                            "scalar": float(scalar[new].mean()), "chemistry": float(chem[new].mean())}},
                        "rows_with_fit_canonical": int((np.isin(can, np.unique(canon[pos])) & (can != "")).sum())}
            blocks[name] = block
    primary = [blocks["far"][arm + "-30k"] for arm in p0.ARMS]
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
    res = {"features": featurize, "predict": predict_far, "correct": correct, "score": score}[stage]()
    if stage != "score":
        res.update(sha256=sha256_file(OUT/(stage+".npz")), score_labels_used=False)
    else:
        res["receipts"] = {k: read(OUT/(k+".json")) for k in ("features", "predict", "correct")}
    res.update(wall_seconds=perf_counter()-start, parent_cpu_seconds=process_time()-cpu,
               versions={k: version(k) for k in ("scikit-learn", "torch", "rdkit", "numpy")},
               prospective_sha256=sha256_file(HERE/"rml/trajectory.json"), source_sha256=sha256_file(Path(__file__)))
    write(dest, res)
    print(stage, dest, flush=True)
