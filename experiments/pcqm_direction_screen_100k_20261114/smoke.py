"""S0 smoke: environment, data identity, one 50-step run per arm class, 1-vs-4 concurrency slowdown, resume, contract.

Writes <work>/smoke.json. Run locally and on IMS with the same command shape:
  python smoke.py --export-root <dir> --features <npz> --work <dir> [--identity row_identity_local.json] [--write-identity]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import common_data as cd  # noqa: E402

SMOKE_ARMS = ["k1_l1_s42", "k1_n12500", "k1_mse", "k1_huber", "k1_binhyb", "k1_laplace", "k1_lds", "k1_oversample",
              "gine_h96", "gine_h192", "gine_l12_h192", "gine_laplace", "fpmlp_n12500"]
IS_WIN = os.name == "nt"


def run_arm(arm, out, a, extra=(), wait=True, log=None):
    cmd = [sys.executable, "-u", str(HERE / "train_arm.py"), "--arm", arm, "--export-root", a.export_root,
           "--features", a.features, "--out", str(out), *extra]
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN else {}
    p = subprocess.Popen(cmd, stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    if not wait:
        return p
    o, _ = p.communicate()
    return p.returncode, o


def check_contract(d: Path):
    done = json.loads((d / "done.json").read_text(encoding="utf-8"))
    miss = [n for n in done["files"] if not (d / n).is_file()]
    bad = [n for n, h in done["files"].items() if (d / n).is_file() and cd.sha256_file(d / n) != h]
    need = {"config.json", "env.json", "curve.json", "dev_pred_final.npz", "dev_pred_best.npz", "train_probe_final.npz",
            "train_pred_final.npz", "timing.json", "model_final.pt"}
    z = np.load(d / "dev_pred_final.npz")
    ok_vals = all(np.isfinite(z[k]).all() for k in z.files)
    probe_rows = len(np.load(d / "train_probe_final.npz")["prediction_eV"])
    env = json.loads((d / "env.json").read_text(encoding="utf-8"))
    return {"missing": miss, "hash_mismatch": bad, "files_complete": need == set(done["files"]), "finite": bool(ok_vals),
            "no_ckpt_left": not (d / "ckpt_last.pt").exists(), "probe_rows": probe_rows,
            "tf32_off": not (env["tf32_matmul"] or env["tf32_cudnn"]), "dev_mae": done["metrics"]["dev_mae_final_eV"],
            "params": done["params"], "status": done["status"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--identity", default=str(HERE / "row_identity_local.json"))
    ap.add_argument("--write-identity", action="store_true")
    ap.add_argument("--conc-steps", type=int, default=300)
    ap.add_argument("--skip-conc", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    res = {"ok": True, "failures": []}

    def fail(msg):
        res["ok"] = False
        res["failures"].append(msg)
        print("FAIL:", msg, flush=True)

    # 1. imports / versions
    import torch
    import torch_geometric
    import ogb  # noqa: F401
    import molgap  # noqa: F401
    res["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "pyg": torch_geometric.__version__,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "molgap_from": str(Path(molgap.__file__).parent)}
    # 2. manifest + shards
    t = time.time()
    man = cd.read_manifest(a.export_root, True)
    res["manifest_sha256"] = man[1]
    res["shard_verify_s"] = time.time() - t
    # 3. row identity
    dev_ = "cpu"
    tr = cd.load_role(a.export_root, "train", None, dev_, _manifest=man)
    dv = cd.load_role(a.export_root, "development", None, dev_, _manifest=man)
    cd.assert_disjoint(tr, dv)
    ident = {"manifest_sha256": man[1], "train": cd.row_identity(tr), "development": cd.row_identity(dv)}
    res["row_identity"] = ident
    idp = Path(a.identity)
    if a.write_identity:
        cd.atomic_json(idp, ident)
        res["identity_check"] = "written"
    elif idp.is_file():
        ref = json.loads(idp.read_text(encoding="utf-8"))
        same = ref == ident
        res["identity_check"] = "match" if same else "MISMATCH"
        if not same:
            fail("row identity differs from row_identity_local.json (STOP: do not train)")
    else:
        res["identity_check"] = "no reference file"
    # 4. K1 params
    sys.path.insert(0, str(HERE))
    import models
    from molgap.qm9_neural_atom import make_encoder
    m = make_encoder("neural_atom_k1")
    res["k1_params"] = models.n_params(m)
    if res["k1_params"] != models.K1_PARAMS:
        fail(f"K1 params {res['k1_params']}")
    del m
    # 5. per-class 50-step runs
    res["arms"] = {}
    for arm in SMOKE_ARMS:
        d = work / "arms"
        shutil_rm(d / arm)
        t = time.time()
        rc, o = run_arm(arm, d, a, ["--max-steps", "50", "--dev-limit", "2000"])
        if rc != 0:
            fail(f"{arm} rc={rc}: {o[-400:]}")
            continue
        c = check_contract(d / arm)
        c["wall_s"] = time.time() - t
        res["arms"][arm] = c
        if c["missing"] or c["hash_mismatch"] or not c["files_complete"] or not c["finite"] or not c["no_ckpt_left"] or not c["tf32_off"]:
            fail(f"{arm} contract: {c}")
        print(arm, c["dev_mae"], f"{c['wall_s']:.0f}s", flush=True)
    # 6. concurrency slowdown (1 vs 4 identical K1 runs)
    if not a.skip_conc:
        n = a.conc_steps
        ex = ["--max-steps", str(n), "--dev-limit", "2000"]
        shutil_rm(work / "conc1")
        rc, o = run_arm("k1_l1_s42", work / "conc1", a, ex)
        t1 = json.loads((work / "conc1/k1_l1_s42/timing.json").read_text(encoding="utf-8"))["train_s"]
        ps = []
        for i in range(4):
            shutil_rm(work / f"conc4_{i}")
            ps.append(run_arm("k1_l1_s42", work / f"conc4_{i}", a, ex, wait=False))
        for p in ps:
            p.communicate()
        t4 = [json.loads((work / f"conc4_{i}/k1_l1_s42/timing.json").read_text(encoding="utf-8"))["train_s"] for i in range(4)]
        res["slowdown"] = {"solo_train_s": t1, "conc4_train_s": t4, "s": float(np.mean(t4) / t1), "steps": n}
        s = max(res["slowdown"]["s"], 1.0)
        res["walltime_suggestion_h"] = {j: round(min(8.0, v["max_slot_U"] * s * 1.25 + 0.3), 2)
                                        for j, v in json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))["jobs"].items()}
        if s > 2.5:
            res["slots_advice"] = "s > 2.5: drop to 3 slots and trim arms"
    # 7. resume: real stop signal after the first checkpoint, then resume
    d = work / "resume"
    shutil_rm(d / "k1_n12500")
    p = run_arm("k1_n12500", d, a, ["--max-steps", "400", "--dev-limit", "2000"], wait=False)
    ck = d / "k1_n12500" / "ckpt_last.pt"
    t = time.time()
    while not ck.exists() and p.poll() is None and time.time() - t < 600:
        time.sleep(0.5)
    if p.poll() is None:
        p.send_signal(signal.CTRL_BREAK_EVENT if IS_WIN else signal.SIGTERM)
    o, _ = p.communicate(timeout=300)
    res["resume"] = {"first_rc": p.returncode, "signal": "CTRL_BREAK_EVENT" if IS_WIN else "SIGTERM"}
    if p.returncode != 75:
        fail(f"resume: first run rc={p.returncode} (wanted 75)")
    rc, o = run_arm("k1_n12500", d, a, ["--max-steps", "400", "--dev-limit", "2000"])
    tj = json.loads((d / "k1_n12500/timing.json").read_text(encoding="utf-8")) if rc == 0 else {}
    res["resume"].update(second_rc=rc, resumes=tj.get("resumes"), steps=tj.get("steps"), tail=o[-300:])
    if rc != 0 or not tj.get("resumes") or tj.get("steps") != 400:
        fail(f"resume failed: {res['resume']}")
    else:
        res["resume"]["contract"] = check_contract(d / "k1_n12500")
    cd.atomic_json(work / "smoke.json", res)
    print("SMOKE", "PASS" if res["ok"] else "FAIL", json.dumps(res["failures"]), flush=True)
    return 0 if res["ok"] else 1


def shutil_rm(p):
    import shutil
    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
