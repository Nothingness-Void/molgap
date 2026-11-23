"""C2 GFN2-xTB residual diagnostic: 2K pilot gate, full dev/far xTB features, fit and score."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import io
import json
from importlib.metadata import version
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter, process_time
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / "pcqm_pairnorm_residual_2d_p0"), str(HERE.parent / "pcqm_h1_heldout_budget")]
import probe as p0
from prepare import pinned, read, write
from molgap.pcqm_fixed_datasets import OFFICIAL_ARCHIVE_SHA256
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import sha256_file

OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-c2-xtb-diagnostic")
BT_OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-residual-block-transfer")
FULL = Path("D:/molgap-data/pcqm_full_train_v1")
XTB_EXE = Path("D:/molgap-artifacts/tools/xtb-6.8.0/stage/bin/xtb.exe")
XTB_EXE_SHA256 = "9e825df458f43e2bb4b48a58a269cc6c40d2228ecc127eb23bd7b87373c8c0a1"
DEV = np.arange(500000, 550000)
FAR = np.arange(3200000, 3250000)
SEED = 20261122
PILOT_SEED = 20261123
PILOT_N = 2000
WORKERS = 12

XTB_NAMES = [
    "etkdg_ok", "mmff_ok", "mmff_energy_kcal", "xtb_ok",
    "xtb_gap_eV", "xtb_homo_eV", "xtb_lumo_eV", "xtb_homo_m1_eV", "xtb_lumo_p1_eV", "xtb_fermi_eV",
    "xtb_total_energy_Eh", "xtb_elec_energy_Eh", "xtb_dipole_norm_au",
    "xtb_q_min", "xtb_q_max", "xtb_q_std", "xtb_q_abs_mean",
    "xtb_n_orb", "xtb_n_elec", "xtb_n_unpaired",
]

_SCRATCH = None
_ENV = None
_XTB_STR = None


def _init_worker(xtb_str):
    global _SCRATCH, _ENV, _XTB_STR
    import atexit
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _XTB_STR = xtb_str
    _SCRATCH = Path(tempfile.mkdtemp(prefix="molgap_xtb_"))
    atexit.register(lambda p=_SCRATCH: shutil.rmtree(p, ignore_errors=True))
    _ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_STACKSIZE": "512M"}


def _xtb_one(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    t0 = perf_counter()
    vals = np.full(len(XTB_NAMES), np.nan, np.float32)
    vals[0] = 0.0
    vals[1] = 0.0
    vals[3] = 0.0
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return vals, perf_counter() - t0
    chrg = Chem.GetFormalCharge(m)
    uhf = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    mh = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = False
    rc = AllChem.EmbedMolecule(mh, params)
    if rc == 0:
        vals[0] = 1.0
    else:
        params.useRandomCoords = True
        rc = AllChem.EmbedMolecule(mh, params)
        if rc != 0:
            return vals, perf_counter() - t0
        vals[0] = 2.0
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mh, mmffVariant="MMFF94s")
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mh, mp)
            if ff is not None:
                mrc = ff.Minimize(maxIts=200)
                if mrc in (0, 1):
                    vals[1] = 1.0 if mrc == 0 else 2.0
                    e = float(ff.CalcEnergy())
                    if np.isfinite(e):
                        vals[2] = e
    except Exception:
        pass
    for fname in ("mol.xyz", "xtbout.json", "charges", "wbo", "xtbrestart", "xtbtopo.mol"):
        fp = _SCRATCH / fname
        if fp.exists():
            try:
                fp.unlink()
            except OSError:
                pass
    (_SCRATCH / "mol.xyz").write_text(Chem.MolToXYZBlock(mh), encoding="utf8")
    try:
        proc = subprocess.run(
            [_XTB_STR, "mol.xyz", "--gfn", "2", "--chrg", str(chrg), "--uhf", str(uhf), "--json", "--norestart"],
            cwd=_SCRATCH, env=_ENV, capture_output=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        return vals, perf_counter() - t0
    jpath = _SCRATCH / "xtbout.json"
    if proc.returncode != 0 or not jpath.is_file():
        return vals, perf_counter() - t0
    try:
        raw = jpath.read_text(encoding="utf8", errors="replace")
        # Guard against unescaped Windows paths in xtbout.json if any appear
        lines = [ln for ln in raw.splitlines() if '"program call"' not in ln]
        d = json.loads("\n".join(lines))
        gap = float(d["HOMO-LUMO gap / eV"])
        if not np.isfinite(gap):
            return vals, perf_counter() - t0
        orbs = d.get("orbital energies / eV", [])
        fr = d.get("fractional occupation", [])
        occ = [i for i, f in enumerate(fr) if f >= 0.5]
        unocc = [i for i, f in enumerate(fr) if f < 0.5]
        homo = float(orbs[occ[-1]]) if occ and occ[-1] < len(orbs) else np.nan
        homo_m1 = float(orbs[occ[-2]]) if len(occ) >= 2 and occ[-2] < len(orbs) else np.nan
        lumo = float(orbs[unocc[0]]) if unocc and unocc[0] < len(orbs) else np.nan
        lumo_p1 = float(orbs[unocc[1]]) if len(unocc) >= 2 and unocc[1] < len(orbs) else np.nan
        fermi = 0.5 * (homo + lumo) if (np.isfinite(homo) and np.isfinite(lumo)) else np.nan
        dip = np.asarray(d.get("dipole / a.u.", [np.nan, np.nan, np.nan]), dtype=np.float64)
        dip_norm = float(np.linalg.norm(dip)) if np.isfinite(dip).all() else np.nan
        q = np.asarray(d.get("partial charges", []), dtype=np.float64)
        q_ok = q.size > 0 and np.isfinite(q).all()
        vals[3] = 1.0
        vals[4] = gap
        vals[5] = homo
        vals[6] = lumo
        vals[7] = homo_m1
        vals[8] = lumo_p1
        vals[9] = fermi
        vals[10] = float(d.get("total energy", np.nan))
        vals[11] = float(d.get("electronic energy", np.nan))
        vals[12] = dip_norm
        vals[13] = float(q.min()) if q_ok else np.nan
        vals[14] = float(q.max()) if q_ok else np.nan
        vals[15] = float(q.std()) if q_ok else np.nan
        vals[16] = float(np.abs(q).mean()) if q_ok else np.nan
        vals[17] = float(d.get("number of molecular orbitals", np.nan))
        vals[18] = float(d.get("number of electrons", np.nan))
        vals[19] = float(d.get("number of unpaired electrons", np.nan))
    except Exception:
        vals[3:] = np.nan
        vals[3] = 0.0
    return vals, perf_counter() - t0


def column(rows, field):
    rows = np.asarray(rows)
    assert np.all(np.diff(rows) > 0) and rows[-1] < 3250000
    wanted = iter(rows.tolist())
    target, values = next(wanted), []
    with ZipFile(pinned(p0.ARCHIVE, OFFICIAL_ARCHIVE_SHA256)) as z, z.open("pcqm4m-v2/raw/data.csv.gz") as f:
        with gzip.open(f, "rb") as gz, io.TextIOWrapper(gz, encoding="utf8") as lines:
            assert next(lines).strip() == "idx,smiles,homolumogap"
            for i, line in enumerate(lines):
                if i != target:
                    continue
                rec = next(csv.reader([line]))
                assert int(rec[0]) == i and len(rec) == 3
                values.append(rec[field])
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


def run_xtb_batch(smiles, tag):
    pinned(XTB_EXE, XTB_EXE_SHA256)
    xtb_str = XTB_EXE.as_posix()
    x = np.empty((len(smiles), len(XTB_NAMES)), np.float32)
    worker_sec = 0.0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn"),
                             initializer=_init_worker, initargs=(xtb_str,)) as pool:
        for i, (row, dt) in enumerate(pool.map(_xtb_one, smiles, chunksize=32)):
            x[i] = row
            worker_sec += dt
            if (i + 1) % 5000 == 0 or (i + 1) == len(smiles):
                ok = int((x[:i + 1, 3] == 1.0).sum())
                print(f"{tag}: {i+1}/{len(smiles)} ok={ok} ({100*ok/(i+1):.2f}%) worker_cpu_s={worker_sec:.1f}", flush=True)
    return x, worker_sec


def pilot():
    rows = np.sort(np.random.default_rng(PILOT_SEED).choice(DEV, PILOT_N, replace=False))
    smiles = column(rows, 1)
    x, worker_sec = run_xtb_batch(smiles, "pilot")
    ok = int((x[:, 3] == 1.0).sum())
    fail_rate = 1.0 - ok / len(rows)
    proj_50k_cpu_h = (worker_sec * (50000.0 / len(rows))) / 3600.0
    passed = (fail_rate < 0.01) and (proj_50k_cpu_h <= 50.0)
    save("pilot", idx=rows, x=x)
    return {
        "xtb_exe_sha256": XTB_EXE_SHA256,
        "feature_names": XTB_NAMES,
        "rows": len(rows),
        "ok_rows": ok,
        "fail_rows": len(rows) - ok,
        "fail_rate": float(fail_rate),
        "etkdg_fail_rows": int((x[:, 0] == 0).sum()),
        "mmff_fail_rows": int((x[:, 1] == 0).sum()),
        "worker_seconds": float(worker_sec),
        "projected_50k_cpu_hours": float(proj_50k_cpu_h),
        "pilot_passed": bool(passed),
        "score_labels_used": False,
    }


def featurize_block(name):
    pil = read(OUT / "pilot.json")
    assert pil["pilot_passed"], "Pilot gate failed; Stage 2 is blocked"
    rows = DEV if name == "dev" else FAR
    x = np.empty((len(rows), len(XTB_NAMES)), np.float32)
    worker_sec = 0.0
    reused_pilot = 0
    if name == "dev":
        with load(OUT, "pilot") as pf:
            p_idx, p_x = pf["idx"], pf["x"]
        in_pil = np.isin(rows, p_idx)
        x[in_pil] = p_x
        rem_rows = rows[~in_pil]
        reused_pilot = int(in_pil.sum())
        assert reused_pilot == PILOT_N and len(rem_rows) == 50000 - PILOT_N
        rem_x, worker_sec = run_xtb_batch(column(rem_rows, 1), "dev_rem")
        x[~in_pil] = rem_x
    else:
        x, worker_sec = run_xtb_batch(column(rows, 1), "far")
    ok = int((x[:, 3] == 1.0).sum())
    save("xtb_" + name, idx=rows, x=x)
    return {
        "xtb_exe_sha256": XTB_EXE_SHA256,
        "feature_names": XTB_NAMES,
        "rows": len(rows),
        "reused_pilot_rows": reused_pilot,
        "ok_rows": ok,
        "fail_rows": len(rows) - ok,
        "fail_rate": float(1.0 - ok / len(rows)),
        "etkdg_fail_rows": int((x[:, 0] == 0).sum()),
        "mmff_fail_rows": int((x[:, 1] == 0).sum()),
        "worker_seconds": float(worker_sec),
        "score_labels_used": False,
    }


def correct():
    assert read(OUT / "pilot.json")["pilot_passed"]
    with p0.read("features") as pf, p0.read("replay") as pr, \
            load(OUT, "xtb_dev") as xd, load(OUT, "xtb_far") as xf, \
            load(BT_OUT, "features") as bf, load(BT_OUT, "predict") as bq:
        p0_ids, p0_x, p0_y = pf["idx"], pf["x"], pf["y"]
        assert np.array_equal(p0_ids, pr["idx"])
        assert np.array_equal(xd["idx"], DEV) and np.array_equal(xf["idx"], FAR)
        assert np.array_equal(bf["idx"], FAR) and np.array_equal(bq["idx"], FAR)
        fit_ids, score_ids = pf["devfit"], pf["score"]
        fit_pos = np.searchsorted(p0_ids, fit_ids)
        score_pos = np.searchsorted(p0_ids, score_ids)
        fit_dev_pos = fit_ids - 500000
        score_dev_pos = score_ids - 500000
        assert len(fit_ids) == 30155 and len(score_ids) == 19845
        y_fit = p0_y[fit_pos].astype(np.float64)
        xtb_dev, xtb_far = xd["x"], xf["x"]
        far_2d = bf["x"]
        far_raw = {arm: bq[arm] for arm in p0.ARMS}
        p0_raw = {arm: pr[arm] for arm in p0.ARMS}

    result, info = {}, {}
    for arm in p0.ARMS:
        p = p0_raw[arm]
        q_dev = p[score_pos]
        q_far = far_raw[arm]
        p_fit = p[fit_pos]
        resid = y_fit - p_fit

        # 1. Reproduce P0 2D strong control exactly
        x2d_fit = np.column_stack((p0_x[fit_pos], p_fit.astype(np.float32)))
        x2d_dev = np.column_stack((p0_x[score_pos], q_dev.astype(np.float32)))
        x2d_far = np.column_stack((far_2d, q_far.astype(np.float32)))
        reg_2d = p0.make_gbm().fit(x2d_fit, resid)
        pred_dev_2d = q_dev + reg_2d.predict(x2d_dev)
        with p0.read(arm + "-dev") as saved:
            assert np.array_equal(saved["idx"], score_ids) and np.array_equal(saved["fit_idx"], fit_ids)
            refit_gap = float(np.max(np.abs(pred_dev_2d - saved["stack"])))
            scalar_dev = saved["control"].copy()
        assert refit_gap <= 1e-4, f"P0 refit mismatch for {arm}: {refit_gap}"
        pred_far_2d = q_far + reg_2d.predict(x2d_far)

        # 2. P0_2d + 20 xTB features
        x2d_xtb_fit = np.column_stack((x2d_fit, xtb_dev[fit_dev_pos]))
        x2d_xtb_dev = np.column_stack((x2d_dev, xtb_dev[score_dev_pos]))
        x2d_xtb_far = np.column_stack((x2d_far, xtb_far))
        reg_both = p0.make_gbm().fit(x2d_xtb_fit, resid)
        pred_dev_both = q_dev + reg_both.predict(x2d_xtb_dev)
        pred_far_both = q_far + reg_both.predict(x2d_xtb_far)

        # 3. Low-dim physics-only xTB corrector: yhat + 20 xTB features (NO 1,247 2D descriptors/fingerprints)
        xtb_only_fit = np.column_stack((p_fit.astype(np.float32), xtb_dev[fit_dev_pos]))
        xtb_only_dev = np.column_stack((q_dev.astype(np.float32), xtb_dev[score_dev_pos]))
        xtb_only_far = np.column_stack((q_far.astype(np.float32), xtb_far))
        reg_xtb = p0.make_gbm().fit(xtb_only_fit, resid)
        pred_dev_xtb = q_dev + reg_xtb.predict(xtb_only_dev)
        pred_far_xtb = q_far + reg_xtb.predict(xtb_only_far)

        # Reuse block-transfer's sealed scalar control on far
        with load(BT_OUT, "correct") as btc:
            scalar_far = btc[f"far-{arm}-30k-scalar"].copy()

        result[f"dev-{arm}-raw"] = q_dev
        result[f"dev-{arm}-scalar"] = scalar_dev
        result[f"dev-{arm}-p0_2d"] = pred_dev_2d
        result[f"dev-{arm}-p0_2d_plus_xtb"] = pred_dev_both
        result[f"dev-{arm}-xtb_only"] = pred_dev_xtb

        result[f"far-{arm}-raw"] = q_far
        result[f"far-{arm}-scalar"] = scalar_far
        result[f"far-{arm}-p0_2d"] = pred_far_2d
        result[f"far-{arm}-p0_2d_plus_xtb"] = pred_far_both
        result[f"far-{arm}-xtb_only"] = pred_far_xtb

        info[arm] = {"fit_rows": len(fit_ids), "p0_2d_refit_max_abs_eV": refit_gap}
        print(f"fitted {arm}: refit_gap={refit_gap:.2e}", flush=True)

    assert all(np.isfinite(v).all() for v in result.values())
    save("correct", fit_idx=fit_ids, dev_score_idx=score_ids, far_idx=FAR, **result)
    return {"fits": info, "score_labels_used": False}


def shard(num):
    from molgap.pcqm_topology import load_packed_topology_shard
    assert num in (10, 64)
    m = read(pinned(FULL / "manifest.json", "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"))
    rec, = [r for r in m["assets"]["topology"] if r["source_idx_min"] == num * 50000]
    assert rec["role"] == "train" and rec["source_idx_max"] == (num + 1) * 50000 - 1
    ds = load_packed_topology_shard(pinned(FULL / rec["file"], rec["sha256"]))
    assert np.array_equal(ds._data.source_idx.ravel().numpy(), np.arange(num * 50000, (num + 1) * 50000))
    return ds


def score():
    pil = read(OUT / "pilot.json")
    if not pil["pilot_passed"]:
        return {
            "verdict": "FAIL",
            "killed_at": "pilot",
            "pilot": pil,
        }
    with load(OUT, "correct") as c, load(OUT, "xtb_dev") as xd, load(OUT, "xtb_far") as xf, \
            p0.read("features") as pf, load(BT_OUT, "features") as bf:
        fit_ids, dev_ids = c["fit_idx"], c["dev_score_idx"]
        assert np.array_equal(c["far_idx"], FAR)
        y_dev_all = np.array(column(DEV, 2), dtype=np.float32)
        assert np.array_equal(y_dev_all, shard(10)._data.y.ravel().numpy()) and np.isfinite(y_dev_all).all()
        y_dev = y_dev_all[dev_ids - 500000].astype(np.float64)

        y_far = np.array(column(FAR, 2), dtype=np.float32)
        assert np.array_equal(y_far, shard(64)._data.y.ravel().numpy()) and np.isfinite(y_far).all()
        y_far = y_far.astype(np.float64)

        fit_sc = np.unique(pf["scaffold"][np.searchsorted(pf["idx"], fit_ids)])
        far_unseen = ~np.isin(bf["scaffold"], fit_sc)
        dev_no_pi = pf["groups"][np.searchsorted(pf["idx"], dev_ids), 1] == 1
        far_no_pi = bf["g"][:, 1] == 1

        gap_dev = xd["x"][dev_ids - 500000, 4].astype(np.float64)
        gap_far = xf["x"][:, 4].astype(np.float64)
        ok_dev = xd["x"][dev_ids - 500000, 3] == 1.0
        ok_far = xf["x"][:, 3] == 1.0

        blocks = {}
        for bname, y, gap, ok, unseen_mask, no_pi in (
                ("dev_scaffold", y_dev, gap_dev, ok_dev, None, dev_no_pi),
                ("far", y_far, gap_far, ok_far, far_unseen, far_no_pi)):
            prefix = "dev" if bname == "dev_scaffold" else "far"
            blk = {
                "rows": len(y),
                "xtb_ok_rows": int(ok.sum()),
                "xtb_gap_true_corr": float(np.corrcoef(gap[ok], y[ok])[0, 1]),
                "arms": {},
            }
            for arm in p0.ARMS:
                err = {k: np.abs(c[f"{prefix}-{arm}-{k}"] - y) * 1000.0
                       for k in ("raw", "scalar", "p0_2d", "p0_2d_plus_xtb", "xtb_only")}
                raw_pred = c[f"{prefix}-{arm}-raw"]
                entry = {
                    "mae_meV": {k: float(v.mean()) for k, v in err.items()},
                    "inc_over_p0_2d": paired_bootstrap_mean(err["p0_2d_plus_xtb"] - err["p0_2d"], n_bootstrap=2000, seed=SEED),
                    "xtb_only_vs_raw": paired_bootstrap_mean(err["xtb_only"] - err["raw"], n_bootstrap=2000, seed=SEED),
                    "xtb_only_vs_scalar": paired_bootstrap_mean(err["xtb_only"] - err["scalar"], n_bootstrap=2000, seed=SEED),
                    "both_vs_raw": paired_bootstrap_mean(err["p0_2d_plus_xtb"] - err["raw"], n_bootstrap=2000, seed=SEED),
                    "no_pi_rows": int(no_pi.sum()),
                    "no_pi_mae_meV": {k: float(v[no_pi].mean()) for k, v in err.items()},
                    "xtb_gap_resid_corr": float(np.corrcoef((gap[ok] - raw_pred[ok]), (y[ok] - raw_pred[ok]))[0, 1]),
                }
                if unseen_mask is not None:
                    entry["unseen_fit_scaffolds"] = {
                        "rows": int(unseen_mask.sum()),
                        "mae_meV": {k: float(v[unseen_mask].mean()) for k, v in err.items()},
                    }
                blk["arms"][arm] = entry
            blocks[bname] = blk

    incs = [blocks["dev_scaffold"]["arms"][a]["inc_over_p0_2d"] for a in p0.ARMS]
    passed = all(x["delta"] <= -1.5 and x["ci95"][1] < 0 for x in incs)
    failed = any(x["ci95"][0] > -0.75 for x in incs)
    return {
        "verdict": "PASS" if passed else "FAIL" if failed else "INCONCLUSIVE",
        "killed_at": None,
        "blocks": blocks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "xtb_dev", "xtb_far", "correct", "score"])
    stage = parser.parse_args().stage
    assert (HERE / "rml/trajectory.json").is_file()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = HERE / "results.json" if stage == "score" else OUT / (stage + ".json")
    assert not dest.exists() and not (stage != "score" and (OUT / (stage + ".npz")).exists())
    start, cpu = perf_counter(), process_time()
    fn = {
        "pilot": pilot,
        "xtb_dev": lambda: featurize_block("dev"),
        "xtb_far": lambda: featurize_block("far"),
        "correct": correct,
        "score": score,
    }[stage]
    res = fn()
    if stage != "score":
        res["sha256"] = sha256_file(OUT / (stage + ".npz"))
    else:
        receipt_names = ["pilot"] + (["xtb_dev", "xtb_far", "correct"] if res.get("killed_at") is None else [])
        res["receipts"] = {k: read(OUT / (k + ".json")) for k in receipt_names}
    res.update(wall_seconds=perf_counter() - start, parent_cpu_seconds=process_time() - cpu,
               versions={k: version(k) for k in ("scikit-learn", "rdkit", "numpy")},
               prospective_sha256=sha256_file(HERE / "rml/trajectory.json"),
               source_sha256=sha256_file(Path(__file__)))
    write(dest, res)
    print(stage, dest, flush=True)
