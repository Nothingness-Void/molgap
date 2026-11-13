"""Train one screen arm (pure 2D, strict FP32) with SIGTERM-safe step-boundary checkpointing and resume.

Writes <out>/<arm>/{config,env,curve,timing,done}.json, dev_pred_{final,best}.npz, train_probe_final.npz,
train_pred_final.npz, model_final.pt; ckpt_last.pt exists only while running. done.json is written last and carries
the sha256 of every contract file. Exit codes: 0 done, 75 interrupted (resumable), 3 non-finite loss (recorded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_data as cd  # noqa: E402
import models  # noqa: E402
import objectives as ob  # noqa: E402

EVAL_EVERY = 5
PROBE_ROWS = 5000
CKPT_SECONDS = 600
STOP = {"flag": False, "sig": None}
CONTRACT_FILES = ("config.json", "env.json", "curve.json", "dev_pred_final.npz", "dev_pred_best.npz",
                  "train_probe_final.npz", "train_pred_final.npz", "timing.json", "model_final.pt")


def _on_signal(signum, frame):
    STOP["flag"], STOP["sig"] = True, int(signum)


def install_signals():
    # SIGBREAK = CTRL_BREAK_EVENT on Windows (SIGTERM cannot be delivered to a Windows process by os.kill)
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _on_signal)


def lr_at(cfg, epoch):
    base, mn, E, W = cfg["lr"], cfg["min_lr"], cfg["epochs"], cfg.get("warmup", 0)
    if epoch < W:
        return base * (epoch + 1) / W
    t = (epoch - W) / max(1, E - W)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * t))


def code_sha():
    h = hashlib.sha256()
    for p in sorted(HERE.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@torch.no_grad()
def predict(model, fam, store, obj, mean, std, bs):
    model.eval()
    preds, scales = [], []
    for s in range(0, store.n, bs):
        b = store.batch(torch.arange(s, min(s + bs, store.n)))
        o = models.forward(model, fam, b)
        preds.append(obj["pred"](o).float())
        sc = obj["scale"](o)
        if sc is not None:
            scales.append(sc.float())
    p = (torch.cat(preds) * std + mean).cpu().numpy().astype(np.float32)
    sc = (torch.cat(scales) * std).cpu().numpy().astype(np.float32) if scales else None
    model.train()
    return p, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--arms-json", default=str(HERE / "arms.json"))
    ap.add_argument("--export-root", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--out", required=True, help="runs directory; the arm writes to <out>/<arm>")
    ap.add_argument("--extra-root", default=None, help="reserved (N200K not implemented)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stop-after-steps", type=int, default=None, help="test hook: behave as if SIGTERM arrived at this global step")
    a = ap.parse_args()
    install_signals()
    cfg = dict(json.loads(Path(a.arms_json).read_text(encoding="utf-8"))["arms"][a.arm])
    if a.seed is not None:
        cfg["seed"] = a.seed
    fam = cfg["family"]
    smoke = a.max_steps is not None or a.dev_limit is not None
    out = Path(a.out) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").is_file():
        print(f"[{a.arm}] done.json exists; nothing to do", flush=True)
        return 0
    dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev_.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    t_load = time.perf_counter()
    man = cd.read_manifest(a.export_root, True)
    train = cd.load_role(a.export_root, "train", cfg["N"], dev_, _manifest=man)
    dev = cd.load_role(a.export_root, "development", a.dev_limit, dev_, _manifest=man)
    cd.assert_disjoint(train, dev)
    if fam == "fpmlp" or cfg["sampler"] == "oversample":
        if not a.features:
            raise SystemExit("--features required for this arm")
        need = ("ecfp", "desc", "tags") if fam == "fpmlp" else ("tags",)
        train.attach_features(cd.load_features(a.features, train, train, need=need))
        dev.attach_features(cd.load_features(a.features, dev, train, need=need))
    load_s = time.perf_counter() - t_load

    y_tr = train.y_cpu.double()
    mean, std = float(y_tr.mean()), float(y_tr.std())
    y_norm = ((train.y_cpu - mean) / std).float()
    obj = ob.make_objective(cfg["loss"], y_norm, dev_)
    desc_dim = int(train.desc.shape[1]) if train.desc is not None else 0
    model = models.build(cfg, obj["n_out"], desc_dim).to(dev_)
    model.train()
    nparams = models.n_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lw, sw = ob.sample_weights(cfg["sampler"], y_norm.numpy(), train.tags)
    lw_t = None if lw is None else torch.as_tensor(lw, dtype=torch.float32, device=dev_)
    sampler = cd.EpochSampler(train.n, cfg["batch"], seed, True, sw)
    spe = sampler.steps()
    E = cfg["epochs"]
    bs_eval = 4096 if fam == "fpmlp" else 512

    cfg_full = {**cfg, "arm": a.arm, "target_mean_eV": mean, "target_std_eV": std, "params": nparams,
                "steps_per_epoch": spe, "max_steps": a.max_steps, "dev_limit": a.dev_limit, "smoke": smoke,
                "train_provenance": train.provenance, "dev_provenance": dev.provenance,
                "train_identity": cd.row_identity(train), "dev_identity": cd.row_identity(dev)}
    cfg_hash = cd.sha256_bytes(json.dumps({k: v for k, v in cfg_full.items() if k != "max_steps"}, sort_keys=True).encode())

    st = {"epoch": 0, "step_in_epoch": 0, "gstep": 0, "curve": [], "best": None, "resumes": [], "train_s": 0.0,
          "eval_s": 0.0, "acc": [0.0, 0.0, 0], "ep_elapsed": 0.0}
    ck = out / "ckpt_last.pt"
    if ck.is_file():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        if c["cfg_hash"] != cfg_hash:
            raise SystemExit("ckpt_last.pt was written under a different config; refusing to resume")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        torch.set_rng_state(c["rng_cpu"])
        if dev_.type == "cuda" and c["rng_cuda"] is not None:
            torch.cuda.set_rng_state(c["rng_cuda"])
        st = c["state"]
        st["resumes"].append({"from_gstep": st["gstep"], "epoch": st["epoch"], "step_in_epoch": st["step_in_epoch"]})
        print(f"[{a.arm}] RESUMED at gstep {st['gstep']} (epoch {st['epoch']}, step {st['step_in_epoch']})", flush=True)

    def save_ckpt():
        cd.atomic_torch_save(ck, {"cfg_hash": cfg_hash, "model": model.state_dict(), "opt": opt.state_dict(),
                                  "rng_cpu": torch.get_rng_state(),
                                  "rng_cuda": torch.cuda.get_rng_state() if dev_.type == "cuda" else None, "state": st})

    dev_y = dev.y_cpu.numpy()

    def evaluate():
        t = time.perf_counter()
        p, sc = predict(model, fam, dev, obj, mean, std, bs_eval)
        st["eval_s"] += time.perf_counter() - t
        return p, sc, float(np.mean(np.abs(p - dev_y)))

    last_ck = time.perf_counter()
    stopped = truncated = False
    while st["epoch"] < E and not stopped:
        ep = st["epoch"]
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, ep)
        order = sampler.order(ep)
        t_ep = time.perf_counter()
        loss_acc = torch.zeros((), device=dev_) + st["acc"][0]
        mae_acc = torch.zeros((), device=dev_) + st["acc"][1]
        nacc = st["acc"][2]
        prev_elapsed = st["ep_elapsed"]
        if dev_.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def snap():
            st["acc"] = [float(loss_acc), float(mae_acc), nacc]
            st["ep_elapsed"] = prev_elapsed + time.perf_counter() - t_ep

        for s in range(st["step_in_epoch"], spe):
            if a.max_steps is not None and st["gstep"] >= a.max_steps:
                truncated = True
                break
            if STOP["flag"] or (a.stop_after_steps is not None and st["gstep"] >= a.stop_after_steps):
                stopped = True
                break
            b = train.batch(sampler.ids(order, s))
            y = (b.y - mean) / std
            o = models.forward(model, fam, b)
            w = None if lw_t is None else lw_t.index_select(0, b.ids)
            loss = obj["loss"](o, y, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                loss_acc += loss.detach()
                mae_acc += (obj["pred"](o) - y).abs().mean().detach() * std
            nacc += 1
            st["gstep"] += 1
            st["step_in_epoch"] = s + 1
            if nacc % 100 == 0 and not bool(torch.isfinite(loss_acc)):
                print(f"[{a.arm}] non-finite loss at gstep {st['gstep']}", flush=True)
                cd.atomic_json(out / "curve.json", st["curve"])
                cd.atomic_json(out / "done.json", {"status": "nan", "arm": a.arm, "gstep": st["gstep"], "files": {}})
                return 3
            if time.perf_counter() - last_ck > CKPT_SECONDS:
                snap()
                save_ckpt()
                last_ck = time.perf_counter()
        snap()
        if stopped:
            save_ckpt()
            print(f"[{a.arm}] STOP at gstep {st['gstep']}: checkpoint saved", flush=True)
            return 75
        if dev_.type == "cuda":
            torch.cuda.synchronize()
        ep_s = prev_elapsed + time.perf_counter() - t_ep
        st["train_s"] += ep_s - prev_elapsed
        row = {"epoch": ep + 1, "step": st["gstep"], "lr": lr_at(cfg, ep),
               "train_loss": float(loss_acc) / max(nacc, 1), "train_mae_eV": float(mae_acc) / max(nacc, 1),
               "dev_mae_eV": None, "epoch_s": ep_s,
               "peak_mem_bytes": int(torch.cuda.max_memory_allocated()) if dev_.type == "cuda" else 0,
               "graphs_per_s": nacc * cfg["batch"] / max(ep_s, 1e-9)}
        last = (ep + 1 == E) or truncated
        if (ep + 1) % EVAL_EVERY == 0 or last:
            p, sc, m = evaluate()
            row["dev_mae_eV"] = m
            if st["best"] is None or m < st["best"]["mae"]:
                st["best"] = {"mae": m, "epoch": ep + 1, "pred": p, "scale": sc}
            st["final_pred"], st["final_scale"] = p, sc
        st["curve"].append(row)
        print(f"[{a.arm}] ep {ep+1}/{E} loss {row['train_loss']:.4f} train_mae {row['train_mae_eV']:.4f} "
              f"dev {row['dev_mae_eV']} {ep_s:.1f}s", flush=True)
        if truncated:
            break
        st["epoch"] += 1
        st["step_in_epoch"] = 0
        st["acc"] = [0.0, 0.0, 0]
        st["ep_elapsed"] = 0.0
        if not last:
            save_ckpt()
            last_ck = time.perf_counter()

    # ---- finalisation: all contract files, then done.json last ----
    fp, fsc = st["final_pred"], st["final_scale"]
    sidx = dev.source_idx.astype(np.int64)

    def dev_npz(name, pred, scale):
        kw = dict(source_idx=sidx, target_eV=dev_y.astype(np.float32), prediction_eV=pred)
        if scale is not None:
            kw["scale_eV"] = scale
        cd.atomic_npz(out / name, **kw)

    dev_npz("dev_pred_final.npz", fp, fsc)
    bst = st["best"]
    dev_npz("dev_pred_best.npz", bst["pred"], bst["scale"])
    tp, tsc = predict(model, fam, train, obj, mean, std, bs_eval)
    ty = train.y_cpu.numpy().astype(np.float32)
    kw = dict(source_idx=train.source_idx.astype(np.int64), target_eV=ty, prediction_eV=tp)
    if tsc is not None:
        kw["scale_eV"] = tsc
    cd.atomic_npz(out / "train_pred_final.npz", **kw)
    probe = {k: v[:PROBE_ROWS] for k, v in kw.items()}
    cd.atomic_npz(out / "train_probe_final.npz", **probe)
    cd.atomic_torch_save(out / "model_final.pt", {"model": model.state_dict(), "cfg": cfg_full})
    cd.atomic_json(out / "config.json", cfg_full)
    cd.atomic_json(out / "env.json", {
        "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0) if dev_.type == "cuda" else None, "torch": torch.__version__,
        "cuda": torch.version.cuda, "pyg": __import__("torch_geometric").__version__,
        "code_sha256": code_sha(), "code_tar_sha256": os.environ.get("CODE_TAR_SHA256"),
        "manifest_sha256": man[1], "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32), "dropped_3d_keys": list(cd.DROP_KEYS)})
    cd.atomic_json(out / "curve.json", st["curve"])
    cd.atomic_json(out / "timing.json", {"train_s": st["train_s"], "eval_s": st["eval_s"], "load_s": load_s,
                                         "resumes": st["resumes"], "steps": st["gstep"]})
    files = {n: cd.sha256_file(out / n) for n in CONTRACT_FILES}
    final_mae = float(np.mean(np.abs(fp - dev_y)))
    cd.atomic_json(out / "done.json", {
        "status": "smoke_truncated" if truncated else "complete", "arm": a.arm, "gstep": st["gstep"],
        "epochs_done": len(st["curve"]),
        "metrics": {"dev_mae_final_eV": final_mae, "dev_mae_best_eV": bst["mae"], "best_epoch": bst["epoch"],
                    "train_probe_mae_eV": float(np.mean(np.abs(probe["prediction_eV"] - probe["target_eV"]))),
                    "train_mae_eV": float(np.mean(np.abs(tp - ty)))},
        "params": nparams, "files": files})
    if ck.is_file():
        ck.unlink()
    print(f"[{a.arm}] DONE final dev MAE {final_mae:.4f} eV", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
