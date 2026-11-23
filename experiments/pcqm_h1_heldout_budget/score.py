"""One prospective evaluation of H1 and the N4 replication on ordinary OGB train rows."""
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

from prepare import ANCHOR, ARM, FULL, HERE, INIT, OUT, P0_DIR, ROOT, pinned, read, sha256_file, write

sys.path.insert(0, str(P0_DIR))
from chemistry import NAMES, bucket, feature_row, keys
from probe import ARMS, INITS, basefiles, calibrate, make_gbm
from molgap.pcqm_fixed_datasets import OFFICIAL_ARCHIVE_SHA256
from molgap.router import paired_bootstrap_mean

ARCHIVE = Path("D:/molgap-data/ogb_pcqm4mv2_raw/pcqm4m-v2.zip")
FIT = np.arange(94000, 100000)
SCORE = np.arange(550000, 600000)
SEED = 20261122


def raw_records():
    with ZipFile(pinned(ARCHIVE, OFFICIAL_ARCHIVE_SHA256)) as z, z.open("pcqm4m-v2/raw/data.csv.gz") as f:
        with gzip.open(f, "rb") as gz, io.TextIOWrapper(gz, encoding="utf8") as lines:
            assert next(lines).strip() == "idx,smiles,homolumogap"
            cursor = 0
            for rows in (FIT, SCORE):
                recs = list(csv.reader(islice(lines, int(rows[0])-cursor, int(rows[-1])+1-cursor)))
                assert np.array_equal([int(r[0]) for r in recs], rows)
                yield recs
                cursor = int(rows[-1])+1


def save(name, **values):
    with (OUT / (name + ".npz")).open("xb") as f:
        np.savez(f, **values)


def load(name):
    return np.load(pinned(OUT / (name + ".npz"), read(OUT / (name + ".json"))["sha256"]), allow_pickle=False)


def featurize():
    assert len(NAMES) == 217
    smiles = [r[1] for rows in raw_records() for r in rows]
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        identifiers = list(pool.map(keys, smiles, chunksize=256))
        x = np.empty((len(smiles), len(NAMES)+6+1024), np.float32)
        g = np.empty((len(smiles), 6), np.int16)
        for i, (v, tags) in enumerate(pool.map(feature_row, smiles, chunksize=128)):
            x[i], g[i] = v, tags
    scaffold, canonical = zip(*identifiers)
    save("features", idx=np.r_[FIT, SCORE], x=x, g=g, scaffold=np.array(scaffold), canonical=np.array(canonical))
    return {"source_archive": OFFICIAL_ARCHIVE_SHA256, "score_labels_used": False, "rows": len(smiles)}


def shard(num):
    from molgap.pcqm_topology import load_packed_topology_shard
    assert num in (1, 2, 10, 11)
    m = read(pinned(FULL / "manifest.json", "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"))
    rec, = [r for r in m["assets"]["topology"] if r["source_idx_min"] == num*50000]
    assert rec["role"] == "train" and rec["source_idx_max"] == (num+1)*50000-1
    ds = load_packed_topology_shard(pinned(FULL / rec["file"], rec["sha256"]))
    assert np.array_equal(ds._data.source_idx.ravel().numpy(), np.arange(num*50000, (num+1)*50000))
    return ds


def predict_all():
    import torch
    from molgap.experiment_spec import ExperimentSpec
    from molgap.experiment_workflow import accept_workflow

    sys.path.insert(0, str(ROOT / "platforms/local"))
    from pcqm_official_valid_eval import setup_determinism
    from pcqm_tail_holdout_eval import gptrans_scorer
    setup_determinism()
    torch.set_num_threads(2)
    stores = {n: shard(n) for n in (1, 2, 10, 11)}
    result, checks, identities = {}, {}, {}
    for name, folder, prepared, expected in (
            ("full100", ANCHOR / "output-native", ANCHOR / "prepared-native",
             "bd0ebc8d6c15639abf7627b94dea89709c4c12a58ff206a51277b7d27b3ca007"),
            ("reduced", OUT / "output", OUT / "prepared", None)):
        pkg = prepared / "package"
        spec = ExperimentSpec(read(pkg / "experiment_spec.json"))
        sub = folder / ARM
        assert read(folder / "local_training_report.json")["status"] == "MECHANICALLY_VERIFIED"
        mp = sub / "output_manifest.json"
        if expected:
            pinned(mp, expected)
        else:
            assert spec.identity == ExperimentSpec(read(HERE / "spec.json")).identity
            assert read(sub / "training_contract.json") == read(HERE / "recipe.json")
        manifest = read(mp)
        assets = manifest["artifacts"]
        files = {k: pinned(sub / assets[k]["path"], assets[k]["sha256"]) for k in ("contract", "predictions", "selected_model")}
        accepted = accept_workflow(spec, ROOT, {ARM: dict(output_dir=str(sub),
            expected=read(files["contract"])["acceptance_requirements"])}, package_dir=pkg,
            expected_package_identity=read(prepared / "prepared_training.json")["expected_package_identity"],
            receipt_path=folder / "launch_receipt.json")
        assert accepted["status"] == "MECHANICALLY_VERIFIED", accepted
        identities[name] = (files, INIT, 100000, sha256_file(mp))
    for name in ARMS:
        files = basefiles(name)
        provenance = read(ROOT / "experiments/pcqm_gptrans_pairnorm_init_noise_500k" / ("init_" + name) / "initialization_provenance.json")
        identities[name] = (files, pinned(INITS/(name + ".pt"), provenance["state_file_sha256"]), 500000,
                            sha256_file(files["contract"].parent / "output_manifest.json"))
    hashes = {}
    for name, (files, init, dev_first, manifest_hash) in identities.items():
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
        ix = dev["source_idx"].ravel().argsort()
        assert np.array_equal(dev["source_idx"].ravel()[ix].numpy(), np.arange(dev_first, dev_first+50000))
        replay = np.arange(dev_first, dev_first+128)
        assert np.array_equal(dev["target_eV"].ravel()[ix[:128]].numpy(), stores[dev_first//50000]._data.y.ravel()[:128].numpy())
        checks[name] = float(np.max(np.abs(run(replay)-dev["prediction_eV"].ravel()[ix[:128]].numpy())))
        assert checks[name] <= 1e-4, checks[name]
        if name == "reduced":
            result["fit_pred"] = run(FIT)
        result[name] = run(SCORE)
        hashes[name] = dict(manifest=manifest_hash, init=sha256_file(init), **{k: sha256_file(p) for k, p in files.items()})
        del model
    assert all(np.isfinite(v).all() for v in result.values())
    fit_y = stores[1]._data.y.ravel().numpy()[FIT-50000]
    assert np.array_equal(fit_y, np.array([r[2] for r in next(raw_records())], dtype=np.float32))
    save("predict", idx=SCORE, fit_idx=FIT, fit_y=fit_y, **result)
    return {"bindings": hashes, "replay_max_abs_eV": checks, "device": torch.cuda.get_device_name(),
            "score_labels_used": False}


def correct():
    import joblib
    with load("features") as f, load("predict") as p:
        assert np.array_equal(f["idx"], np.r_[p["fit_idx"], p["idx"]]) and np.array_equal(p["idx"], SCORE)
        fit, q, y = p["fit_pred"], p["reduced"], p["fit_y"].astype(np.float64)
        x = np.column_stack([f["x"], np.r_[fit, q].astype(np.float32)])
        sc = f["scaffold"][:len(FIT)]
        val = np.array([bucket("control:"+k, 5) == 0 for k in sc])
        losses = {k: float(np.abs(calibrate(k, fit[~val], y[~val], fit[val])-y[val]).mean())
                  for k in ("identity", "median", "affine", "isotonic", "gbm")}
        control_name = min(losses, key=losses.get)
        scalar = calibrate(control_name, fit, y, q)
        reg = make_gbm().fit(x[:len(FIT)], y-fit)
        trained = OUT / "chemistry.joblib"
        with trained.open("xb") as out:
            joblib.dump(reg, out)
        chemistry = q + reg.predict(x[len(FIT):])
        assert np.isfinite(scalar).all() and np.isfinite(chemistry).all()
        save("correct", idx=SCORE, fit_idx=FIT, scalar=scalar, chemistry=chemistry)
    return {"control": control_name, "inner_rows": int(val.sum()), "fit_validation_mae_eV": losses,
            "model_sha256": sha256_file(trained), "score_labels_used": False}


def score():
    with load("features") as f, load("predict") as p, load("correct") as c:
        assert np.array_equal(f["idx"], np.r_[FIT, SCORE]) and np.array_equal(c["idx"], SCORE) and np.array_equal(p["idx"], SCORE)
        reader = raw_records()
        next(reader)
        y = np.array([r[2] for r in next(reader)], dtype=np.float32)
        assert np.array_equal(y, shard(11)._data.y.ravel().numpy()) and np.isfinite(y).all()
        y = y.astype(np.float64)
        e = {k: (v-y)*1000 for k,v in {"full100":p["full100"], "reduced":p["reduced"],
             "scalar":c["scalar"], "chemistry":c["chemistry"], **{k:p[k] for k in ARMS}}.items()}
        ae = {k: np.abs(v) for k,v in e.items()}
        diffs = {k: paired_bootstrap_mean(ae[a]-ae[b], n_bootstrap=2000, seed=SEED) for k,a,b in (
            ("net","chemistry","full100"), ("rescue","chemistry","reduced"), ("features","chemistry","scalar"),
            ("cost","reduced","full100"))}
        n, r, fdelta, cost = diffs.values()
        assert abs(n["delta"]-r["delta"]-cost["delta"]) < 1e-9
        passed = all(d["delta"] <= -1.5 and d["ci95"][1] < 0 for d in (n,r)) and fdelta["ci95"][1] < 0
        failed = any(d["ci95"][0] > -1.5 for d in (n,r)) or fdelta["ci95"][0] >= 0
        sc, can = f["scaffold"], f["canonical"]
        new = ~np.isin(sc[len(FIT):], np.unique(sc[:len(FIT)]))
        dup = np.isin(can[len(FIT):], np.unique(can[:len(FIT)])) & (can[len(FIT):] != "")
        rad, pi, rings = f["g"][len(FIT):,:3].T
        heavy = f["x"][len(FIT):, NAMES.index("HeavyAtomCount")]
        singles = pi == 1
        same_miss = (e["s43"]*e["s44"] > 0) & (np.minimum(ae["s43"],ae["s44"]) >= 150)
        assert singles.any() and same_miss.any()
        subset = {k: {"mae_meV": float(ae[k][singles].mean()), "ratio": float(ae[k][singles].mean()/ae[k].mean()),
                      "share": float(ae[k][singles].sum()/ae[k].sum())} for k in ARMS}
        rate, base = float(same_miss[singles].mean()), float(same_miss.mean())
        flags = all(v["ratio"] >= 1.5 and v["share"] >= .1 for v in subset.values()) and rate/base >= 1.5
        cell = (np.digitize(heavy,[11,16])*4 + np.minimum(3,rings))*4 + np.digitize(y,[4,6,8])
        count, eligible = 0, 0
        adj = np.zeros(2)
        for i in np.unique(cell[rad>=0]):
            a, b = singles & (cell==i), (pi==0) & (cell==i)
            if min(a.sum(), b.sum()) < 20:
                continue
            count += int(a.sum())
            eligible += 1
            adj += a.sum()*np.array([ae[k][a].mean()-ae[k][b].mean() for k in ARMS])
        return dict(h1_verdict="PASS" if passed else "FAIL" if failed else "INCONCLUSIVE",
            rows=len(y), mae_meV={k:float(v.mean()) for k,v in ae.items()}, paired_meV=diffs,
            unseen_fit_scaffolds=dict(rows=int(new.sum()), mae_meV={k:float(v[new].mean()) for k,v in ae.items()} if new.any() else {}),
            rows_with_fit_canonical=int(dup.sum()), invalid_score=int((rad<0).sum()),
            n4=dict(replicates=flags, rows=int(singles.sum()), members=subset, comiss_group=rate, comiss_all=base,
                    enrichment=rate/base, matched_rows=count, matched_cells=eligible, matched_coverage=count/int(singles.sum()),
                    standardized_excess_meV=(adj/count).tolist() if count else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["features", "predict", "correct", "score"])
    stage = parser.parse_args().stage
    assert (HERE / "rml/trajectory.json").is_file()
    dest = HERE / "results.json" if stage == "score" else OUT / (stage + ".json")
    assert not dest.exists() and not (stage != "score" and (OUT/(stage+".npz")).exists())
    start, cpu = perf_counter(), process_time()
    res = {"features":featurize, "predict":predict_all, "correct":correct, "score":score}[stage]()
    if stage != "score":
        res["sha256"] = sha256_file(OUT/(stage+".npz"))
    else:
        res["receipts"] = {k:read(OUT/(k+".json")) for k in ("features","predict","correct")}
    res.update(wall_seconds=perf_counter()-start, parent_cpu_seconds=process_time()-cpu,
               versions={k:version(k) for k in ("scikit-learn","torch","rdkit","numpy")},
               prospective_sha256=sha256_file(HERE/"rml/trajectory.json"), source_sha256=sha256_file(Path(__file__)))
    write(dest, res)
    print(stage, dest, flush=True)
