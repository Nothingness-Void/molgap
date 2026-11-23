"""C3 checkpoint-space probe: MC-dropout (K=8,16,32) and pass-60 EMA/live weights."""
import argparse
from copy import copy
import csv
import gzip
import io
from importlib.metadata import version
from pathlib import Path
import sys
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

OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-c3-checkpoint-space")
FULL = Path("D:/molgap-data/pcqm_full_train_v1")
BLOCKS = {"dev": (10, np.arange(500000, 550000)), "unseen": (62, np.arange(3100000, 3150000))}
SEED = 20261122
MC_SEED = 20261123
MC_K = 32
BATCH_SIZE = 128


def column(rows, field):
    assert rows[-1] < 3250000
    with ZipFile(pinned(p0.ARCHIVE, OFFICIAL_ARCHIVE_SHA256)) as z, z.open("pcqm4m-v2/raw/data.csv.gz") as f:
        with gzip.open(f, "rb") as gz, io.TextIOWrapper(gz, encoding="utf8") as lines:
            assert next(lines).strip() == "idx,smiles,homolumogap"
            start, stop = int(rows[0]), int(rows[-1]) + 1
            recs = []
            for i, line in enumerate(lines):
                if i < start:
                    continue
                if i >= stop:
                    break
                recs.append(next(csv.reader([line])))
    assert np.array_equal([int(r[0]) for r in recs], rows)
    return [r[field] for r in recs]


def shard(num):
    from molgap.pcqm_topology import load_packed_topology_shard
    assert num in (10, 62)
    m = read(pinned(FULL / "manifest.json", "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"))
    rec, = [r for r in m["assets"]["topology"] if r["source_idx_min"] == num * 50000]
    assert rec["role"] == "train" and rec["source_idx_max"] == (num + 1) * 50000 - 1
    ds = load_packed_topology_shard(pinned(FULL / rec["file"], rec["sha256"]))
    assert np.array_equal(ds._data.source_idx.ravel().numpy(), np.arange(num * 50000, (num + 1) * 50000))
    return ds


def arm_files(arm):
    folder = p0.RUN / ("init_" + arm)
    manifest = read(pinned(folder / "output_manifest.json", p0.MANIFESTS[p0.ARMS.index(arm)]))
    assert manifest["context"]["arm_id"] == "init_" + arm
    return {k: pinned(folder / v["path"], v["sha256"])
            for k, v in manifest["artifacts"].items()
            if k in {"contract", "selected_model", "predictions", "resume", "trace"}}


def save(name, **values):
    with (OUT / (name + ".npz")).open("xb") as f:
        np.savez(f, **values)


def load(name):
    return np.load(pinned(OUT / (name + ".npz"), read(OUT / (name + ".json"))["sha256"]), allow_pickle=False)


def prepare_batches(graphs, policy):
    from torch_geometric.loader import DataLoader
    from molgap.gptrans_execution_backend import prepare_layout
    from molgap.gptrans_execution_hotpath import MODEL_FIELDS
    batches = []
    for batch in DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False):
        b = prepare_layout(copy(batch))
        if policy.prepare_batch is not None:
            b = policy.prepare_batch(b, training=False)
        extra = tuple(f for f in policy.extra_batch_fields if hasattr(b, f))
        batches.append((b, MODEL_FIELDS + extra))
    return batches


def run_det(model, batches, mean, std, device):
    import torch
    from molgap.gptrans_execution_backend import prepared_forward
    model.eval()
    out = []
    with torch.no_grad():
        for b, fields in batches:
            out.append(prepared_forward(model, b.to(device, *fields)).double().cpu().numpy().ravel())
    return np.concatenate(out) * std + mean


def run_mc(model, batches, mean, std, device):
    import torch
    import torch.nn as nn
    from molgap.gptrans_execution_backend import prepared_forward
    model.eval()
    n_drop = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n_drop += 1
    assert n_drop > 0
    s8, s16, s32, sq32 = [], [], [], []
    with torch.no_grad():
        for bi, (b, fields) in enumerate(batches):
            bg = b.to(device, *fields)
            acc = None
            acc_sq = None
            c8 = c16 = None
            for k in range(MC_K):
                torch.manual_seed(MC_SEED + k * 100000 + bi)
                torch.cuda.manual_seed_all(MC_SEED + k * 100000 + bi)
                y = prepared_forward(model, bg).double() * std + mean
                acc = y if acc is None else acc + y
                acc_sq = y * y if acc_sq is None else acc_sq + y * y
                if k == 7:
                    c8 = (acc / 8.0).cpu().numpy().ravel()
                elif k == 15:
                    c16 = (acc / 16.0).cpu().numpy().ravel()
            m32 = acc / MC_K
            var32 = torch.clamp(acc_sq / MC_K - m32 * m32, min=0.0)
            s8.append(c8)
            s16.append(c16)
            s32.append(m32.cpu().numpy().ravel())
            sq32.append(torch.sqrt(var32).cpu().numpy().ravel())
    model.eval()
    res = (np.concatenate(s8), np.concatenate(s16), np.concatenate(s32), np.concatenate(sq32))
    assert float(res[3].mean()) > 1e-4, "MC-dropout produced zero variance"
    return res, n_drop


def predict_all():
    import torch
    from molgap.family_training_kernel import load_family_model, mode_policy
    from molgap.family_training_runner import read_recipe
    sys.path.insert(0, str(ROOT / "platforms/local"))
    from pcqm_official_valid_eval import setup_determinism

    setup_determinism()
    torch.set_num_threads(2)
    device = "cuda"
    result, checks, bindings, trace_pass60, dropout_counts = {}, {}, {}, {}, {}

    # Prepare both 50K blocks once on CPU
    block_batches = {}
    sample_recipe = None
    for arm in p0.ARMS:
        files = arm_files(arm)
        _, recipe = read_recipe(files["contract"])
        sample_recipe = recipe
        break
    policy = mode_policy(sample_recipe.family, sample_recipe.mode)
    for bname, (num, rows) in BLOCKS.items():
        ds = shard(num)
        graphs = [ds[i] for i in range(len(rows))]
        assert np.array_equal([int(g.source_idx) for g in graphs], rows)
        block_batches[bname] = prepare_batches(graphs, policy)
        del ds, graphs
        print("prepared block", bname, len(block_batches[bname]), "batches", flush=True)

    for arm in p0.ARMS:
        files = arm_files(arm)
        provenance = read(p0.PARENT / ("init_" + arm) / "initialization_provenance.json")
        init = pinned(p0.INITS / (arm + ".pt"), provenance["state_file_sha256"])
        _, recipe = read_recipe(files["contract"])
        mean, std = recipe.target_mean_eV, recipe.target_std_eV
        tr = read(files["trace"])
        p60 = [pt for pt in tr["observations"] if pt.get("epoch_or_pass") == 60][-1]
        assert p60["checkpoint_identity"] == sha256_file(files["resume"])
        trace_pass60[arm] = float(p60["ema_dev_metric"])

        model, _ = load_family_model(recipe, init)
        model = model.to(device).eval()

        # Verify selected_model.pt replay against saved first 128 dev predictions
        sel_state = torch.load(files["selected_model"], map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(sel_state, strict=True)
        dev_saved = torch.load(files["predictions"], map_location="cpu", weights_only=True)
        ix = dev_saved["source_idx"].ravel().argsort()
        assert np.array_equal(dev_saved["source_idx"].ravel()[ix].numpy(), np.arange(500000, 550000))
        first128 = run_det(model, block_batches["dev"][:1], mean, std, device)
        checks[arm] = float(np.max(np.abs(first128 - dev_saved["prediction_eV"].ravel()[ix[:BATCH_SIZE]].numpy())))
        assert checks[arm] <= 1e-4, f"{arm} replay mismatch: {checks[arm]}"

        # Save selected_model predictions on both blocks (dev from full det run to check parity)
        for bname in BLOCKS:
            result[f"{bname}-{arm}-sel"] = run_det(model, block_batches[bname], mean, std, device)
        dev_full_diff = float(np.max(np.abs(result[f"dev-{arm}-sel"] - dev_saved["prediction_eV"].ravel()[ix].numpy())))
        assert dev_full_diff <= 1e-4, dev_full_diff

        # Load pass-60 EMA and live weights from last_checkpoint.pt
        ckpt = torch.load(files["resume"], map_location="cpu", weights_only=True)
        assert ckpt["optimizer_step"] == 234360 and ckpt["cursor"]["pass_index"] == 60
        ema_state, live_state = ckpt["ema"], ckpt["model"]
        avg_state = {k: (0.5 * (ema_state[k].float() + live_state[k].float()).to(ema_state[k].dtype)
                         if ema_state[k].is_floating_point() else ema_state[k])
                     for k in ema_state}

        # Deterministic live60 and wavg60
        for tag, state in (("live60", live_state), ("wavg60", avg_state)):
            model.load_state_dict(state, strict=True)
            for bname in BLOCKS:
                result[f"{bname}-{arm}-{tag}"] = run_det(model, block_batches[bname], mean, std, device)

        # Deterministic ema60 + MC-dropout K=8,16,32 on ema60
        model.load_state_dict(ema_state, strict=True)
        for bname in BLOCKS:
            t0 = perf_counter()
            result[f"{bname}-{arm}-ema60"] = run_det(model, block_batches[bname], mean, std, device)
            (mc8, mc16, mc32, mc32_std), n_drop = run_mc(model, block_batches[bname], mean, std, device)
            result[f"{bname}-{arm}-mc8"] = mc8
            result[f"{bname}-{arm}-mc16"] = mc16
            result[f"{bname}-{arm}-mc32"] = mc32
            result[f"{bname}-{arm}-mc32_std"] = mc32_std
            dropout_counts[f"{bname}-{arm}"] = n_drop
            print(f"{arm} {bname} ema60+mc32 done in {perf_counter()-t0:.1f}s, mean_std={mc32_std.mean()*1000:.2f} meV", flush=True)

        bindings[arm] = dict(init=sha256_file(init), **{k: sha256_file(p) for k, p in files.items()})
        del model, ckpt, sel_state, ema_state, live_state, avg_state
        torch.cuda.empty_cache()

    assert all(np.isfinite(v).all() and len(v) == 50000 for v in result.values())
    save("predict", dev_idx=BLOCKS["dev"][1], unseen_idx=BLOCKS["unseen"][1], **result)
    return {"bindings": bindings, "replay_max_abs_eV": checks, "trace_pass60_ema_dev_mae_eV": trace_pass60,
            "dropout_modules_enabled": dropout_counts, "mc_k": MC_K, "batch_size": BATCH_SIZE,
            "device": torch.cuda.get_device_name()}


def score():
    receipt = read(OUT / "predict.json")
    with load("predict") as p:
        blocks = {}
        for bname, (num, rows) in BLOCKS.items():
            assert np.array_equal(p[bname + "_idx"], rows)
            y = np.array(column(rows, 2), dtype=np.float32)
            assert np.array_equal(y, shard(num)._data.y.ravel().numpy()) and np.isfinite(y).all()
            y = y.astype(np.float64)
            blk = {"rows": len(y), "arms": {}, "ensemble": {}}
            for arm in p0.ARMS:
                ema = np.abs(p[f"{bname}-{arm}-ema60"] - y) * 1000.0
                if bname == "dev":
                    expected = receipt["trace_pass60_ema_dev_mae_eV"][arm]
                    assert abs(ema.mean() / 1000.0 - expected) <= 1e-4, (arm, ema.mean() / 1000.0, expected)
                pavg = 0.5 * (p[f"{bname}-{arm}-ema60"] + p[f"{bname}-{arm}-live60"])
                preds = {k: np.abs(p[f"{bname}-{arm}-{k}"] - y) * 1000.0
                         for k in ("ema60", "mc8", "mc16", "mc32", "live60", "wavg60", "sel")}
                preds["pavg60"] = np.abs(pavg - y) * 1000.0
                std_mev = p[f"{bname}-{arm}-mc32_std"] * 1000.0
                blk["arms"][arm] = {
                    "mae_meV": {k: float(v.mean()) for k, v in preds.items()},
                    "vs_ema60_meV": {k: paired_bootstrap_mean(v - ema, n_bootstrap=2000, seed=SEED)
                                     for k, v in preds.items() if k != "ema60"},
                    "mc32_std_mean_meV": float(std_mev.mean()),
                    "mc32_std_abs_err_corr": float(np.corrcoef(std_mev, ema)[0, 1]),
                }
            # 2-init equal-weight ensemble (s43 + s44)
            ens = {k: np.abs(0.5 * (p[f"{bname}-s43-{k}"] + p[f"{bname}-s44-{k}"]) - y) * 1000.0
                   for k in ("ema60", "mc8", "mc16", "mc32", "live60", "wavg60", "sel")}
            blk["ensemble"] = {
                "mae_meV": {k: float(v.mean()) for k, v in ens.items()},
                "vs_ema60_meV": {k: paired_bootstrap_mean(v - ens["ema60"], n_bootstrap=2000, seed=SEED)
                                 for k, v in ens.items() if k != "ema60"},
            }
            blocks[bname] = blk

    d = blocks["dev"]
    singles = [d["arms"][a]["vs_ema60_meV"]["mc32"] for a in p0.ARMS]
    ens_mc = d["ensemble"]["vs_ema60_meV"]["mc32"]
    passed = all(x["delta"] <= -0.5 and x["ci95"][1] < 0 for x in singles) and ens_mc["delta"] <= 0
    failed = any(x["ci95"][0] > -0.5 for x in singles) or ens_mc["ci95"][0] > 0
    return {
        "verdict": "PASS" if passed else "FAIL" if failed else "INCONCLUSIVE",
        "snapshot_averaging_status": "CANCELLED_SNAPSHOTS_NOT_RETAINED",
        "blocks": blocks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["predict", "score"])
    stage = parser.parse_args().stage
    assert (HERE / "rml/trajectory.json").is_file()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = HERE / "results.json" if stage == "score" else OUT / (stage + ".json")
    assert not dest.exists() and not (stage != "score" and (OUT / (stage + ".npz")).exists())
    start, cpu = perf_counter(), process_time()
    res = {"predict": predict_all, "score": score}[stage]()
    if stage != "score":
        res.update(sha256=sha256_file(OUT / (stage + ".npz")), score_labels_used=False)
    else:
        res["receipts"] = {"predict": read(OUT / "predict.json")}
    res.update(wall_seconds=perf_counter() - start, parent_cpu_seconds=process_time() - cpu,
               versions={k: version(k) for k in ("torch", "numpy")},
               prospective_sha256=sha256_file(HERE / "rml/trajectory.json"),
               source_sha256=sha256_file(Path(__file__)))
    write(dest, res)
    print(stage, dest, flush=True)
