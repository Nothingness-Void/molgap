"""Real-flow check of the GPU-resident DeMol-CIN inputs against the CellData-object path.

  equiv  batch equality (raw and after FamilyKernel._gptrans_batch), full train/dev pass, 1/10/50-step loss
  mem    one variant (old|new): host peak (process tree RSS and USS), GPU use, build time, a few steps

Windows: runs in-process with 0 loader workers. Reads only the selected train/dev roles; never tail/valid/test.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from copy import copy
from pathlib import Path

import psutil
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from molgap.demol_cin import _CellDataset, extract_cellular_complex  # noqa: E402
from molgap.demol_cin_resident import ResidentCellDataset  # noqa: E402
from molgap.family_training_data import load_training_data  # noqa: E402
from molgap.family_training_kernel import FamilyKernel  # noqa: E402
from molgap.family_training_loaders import FamilyLoaders  # noqa: E402
from molgap.family_training_runner import read_recipe  # noqa: E402
from molgap.gptrans_execution_backend import prepare_layout  # noqa: E402
from molgap.gptrans_execution_tuning import ExecutionTuning  # noqa: E402
from molgap.training_reproducibility import atomic_torch_save, configure_fp32_determinism  # noqa: E402
from molgap.v4_runtime import state_dict_sha256  # noqa: E402

UNUSED = {"random_walk_pe", "split_code", "smiles_sanitized"}  # carried by the old batch, never read by DeMol


class LazyCells(_CellDataset):
    """Same items as _CellDataset, extracted on demand (no host list) so old and new can share one process."""
    def __init__(self, base):
        self.base, self.transform = base, None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return extract_cellular_complex(self.base[idx])


class Monitor(threading.Thread):
    """Polls the process tree: peak sum of RSS, peak sum of USS, peak single-process RSS/USS, peak handles."""
    def __init__(self, limit_gb=22.0):
        super().__init__(daemon=True)
        self.stop_flag, self.limit = False, limit_gb * 2**30
        self.peak = dict(rss_sum=0, uss_sum=0, rss_max=0, uss_max=0, handles=0)

    def run(self):
        me = psutil.Process()
        while not self.stop_flag:
            procs = [me] + me.children(recursive=True)
            rss = uss = rmax = umax = handles = 0
            for p in procs:
                try:
                    r = p.memory_info().rss
                    u = p.memory_full_info().uss
                    handles += p.num_handles()
                except psutil.Error:
                    continue
                rss, uss, rmax, umax = rss + r, uss + u, max(rmax, r), max(umax, u)
            for k, v in zip(self.peak, (rss, uss, rmax, umax, handles)):
                self.peak[k] = max(self.peak[k], v)
            if rss > self.limit or handles > 20000:
                print("GUARD TRIPPED", self.peak, flush=True)
                psutil.Process().kill()
            time.sleep(0.2)

    def result(self):
        self.stop_flag = True
        return {k: (round(v / 2**20, 1) if k != "handles" else v) for k, v in self.peak.items()}  # MiB


def setup(args):
    record, recipe = read_recipe(args.recipe)
    data = load_training_data(args.root, recipe.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    initial = out / "initial_state_placeholder.pt"
    atomic_torch_save(initial, {"unused_standalone_model_builds_from_seed": torch.zeros(1)})
    tuning = ExecutionTuning(independent_models=1, training_workers=0, total_cpu_budget=4)
    return recipe, data, initial, tuning


def compare(old, new, label, fails):
    ok_keys = set(old.keys()) - UNUSED
    if set(new.keys()) != ok_keys:
        fails.append((label, "keys", sorted(set(new.keys()) ^ ok_keys)))
        return
    for k in sorted(ok_keys):
        a, b = old[k], new[k]
        if isinstance(a, dict) or isinstance(a, tuple):
            continue
        b = b.cpu() if torch.is_tensor(b) else b
        if not (a.dtype == b.dtype and a.shape == b.shape and a.is_contiguous() == b.is_contiguous() and torch.equal(a, b)):
            fails.append((label, k, a.dtype, b.dtype, tuple(a.shape), tuple(b.shape)))
    if int(old.num_graphs) != int(new.num_graphs):
        fails.append((label, "num_graphs"))


def compare_layout(old, new, label, fails):
    a, b = old.execution_layout, new.execution_layout
    if set(a) != set(b) or a["batch_size"] != b["batch_size"] or a["max_nodes"] != b["max_nodes"]:
        fails.append((label, "layout meta"))
        return
    pairs = [("mask", a["mask"], b["mask"]), ("flat_index", a["flat_index"], b["flat_index"])]
    pairs += [("edges%d" % i, x, y) for i, (x, y) in enumerate(zip(a["edges"], b["edges"]))]
    for name, x, y in pairs:
        y = y.cpu()
        if not (x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
            fails.append((label, "layout." + name))


def hexf(x):
    return struct.pack(">f", float(x)).hex()


def equiv(args):
    recipe, data, initial, tuning = setup(args)
    base = data.train
    t0 = time.perf_counter()
    res = ResidentCellDataset(base)
    print("resident build train %d rows: %.1f s, %.1f MiB compact" % (len(res), time.perf_counter() - t0, res.store.nbytes / 2**20), flush=True)
    res.store.upload("cuda")
    ptr = res.store.ptr
    nodes, bonds, rings = (ptr[k][1:] - ptr[k][:-1] for k in ("node", "bond", "ring"))
    special = {"largest": int(nodes.argmax()), "no_ring": int((rings == 0).nonzero()[0]),
               "most_rings": int(rings.argmax()), "no_bond": int((bonds == 0).nonzero()[0]) if bool((bonds == 0).any()) else None}
    print("molecules: max nodes %d, max rings %d, no-ring %d, no-bond %s, ring-cap(>=16) %d" % (
        int(nodes.max()), int(rings.max()), int((rings == 0).sum()),
        int((bonds == 0).sum()), int((rings >= 16).sum())), "special:", special, flush=True)
    gen = torch.Generator().manual_seed(7)
    cases = [("largest bs1", [special["largest"]]), ("no_ring bs1", [special["no_ring"]]),
             ("most_rings bs1", [special["most_rings"]])]
    if special["no_bond"] is not None:
        cases.append(("no_bond bs1", [special["no_bond"]]))
    mix = [v for v in special.values() if v is not None]
    for bs in (1, 7, 128, 1000):
        cases.append(("random bs%d" % bs, torch.randperm(len(res), generator=gen)[:bs].tolist()))
    cases.append(("mixed specials+random", mix + torch.randperm(len(res), generator=gen)[:50].tolist()))
    cases.append(("tail of data", list(range(len(res) - 130, len(res)))))
    fails = []
    for label, idx in cases:
        old = prepare_layout(Batch.from_data_list([extract_cellular_complex(base[i]) for i in idx]))
        new = res.store.batch(idx)
        compare(old, new, label, fails)
        compare_layout(old, new, label, fails)
        print("  %-24s size %-5d nodes %-6d -> %s" % (label, len(idx), int(old.x.shape[0]), "equal" if not any(f[0] == label for f in fails) else "MISMATCH"), flush=True)
    print("RAW BATCH CASES:", "PASS" if not fails else "FAIL %r" % fails[:5], flush=True)

    # Real loaders: old = FamilyLoaders over lazily extracted CellData; new = FamilyLoaders over the resident dataset.
    dev_res = ResidentCellDataset(data.development)
    from dataclasses import replace
    d_old = replace(data, train=LazyCells(data.train), development=LazyCells(data.development))
    d_new = replace(data, train=res, development=dev_res)
    configure_fp32_determinism(recipe.seed)
    kernel = FamilyKernel(recipe, initial, device="cuda")
    lo, ln = FamilyLoaders(recipe, d_old, tuning), FamilyLoaders(recipe, d_new, tuning)
    print("loaders: new device_train on =", ln.device_train is not None, "| old =", lo.device_train is not None, flush=True)
    fails, nb, t0 = [], 0, time.perf_counter()
    src_old, src_new = [], []
    for i, (bo, bn) in enumerate(zip(lo.train(0, 0), ln.train(0, 0))):
        nb += 1
        bo2 = prepare_layout(copy(bo))
        compare(bo2, bn, "train%d" % i, fails)
        compare_layout(bo2, bn, "train%d" % i, fails)
        src_old.append(bo.source_idx.view(-1)); src_new.append(bn.source_idx.view(-1))
        if i < 3 or i % 100 == 0 or i == len(lo.train(0, 0)) - 1:
            mo, mn = kernel._gptrans_batch(bo, training=True), kernel._gptrans_batch(bn, training=True)
            for k in set(mo.keys()) | set(mn.keys()):
                a, b = mo[k] if k in mo else None, mn[k] if k in mn else None
                if k in UNUSED and b is None:
                    continue
                if isinstance(a, dict):
                    a, b = list(a.values()), list(b.values())
                    a = [t for x in a for t in (x if isinstance(x, tuple) else (x,)) if torch.is_tensor(t)]
                    b = [t for x in b for t in (x if isinstance(x, tuple) else (x,)) if torch.is_tensor(t)]
                    ok = len(a) == len(b) and all(torch.equal(x.cpu(), y.cpu()) and x.dtype == y.dtype for x, y in zip(a, b))
                elif torch.is_tensor(a):
                    ok = b is not None and a.device == b.device and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
                else:
                    ok = a == b
                if not ok:
                    fails.append(("model-input train%d" % i, k))
    order_ok = torch.equal(torch.cat(src_old), torch.cat(src_new))
    print("TRAIN PASS 0 (%d batches, %.0f s): fields+layout %s; source_idx order identical: %s; first-pass rows %d" % (
        nb, time.perf_counter() - t0, "PASS" if not fails else "FAIL %r" % fails[:5], order_ok, int(torch.cat(src_new).numel())), flush=True)
    # A second pass index (resume cursor) must also agree on order only
    so, sn = [], []
    for bo, bn in zip(lo.train(1, 5), ln.train(1, 5)):
        so.append(bo.source_idx.view(-1)); sn.append(bn.source_idx.view(-1))
    print("PASS 1 from batch 5: source_idx order identical:", torch.equal(torch.cat(so), torch.cat(sn)), flush=True)
    fails, nb = [], 0
    for i, (bo, bn) in enumerate(zip(lo.dev(), ln.dev())):
        nb += 1
        bo2 = prepare_layout(copy(bo))
        compare(bo2, bn, "dev%d" % i, fails)
        compare_layout(bo2, bn, "dev%d" % i, fails)
    print("DEV PASS (%d batches, last size %d): %s" % (nb, int(bn.num_graphs), "PASS" if not fails else "FAIL %r" % fails[:5]), flush=True)
    del lo, ln

    # Loss on the real FamilyKernel.step: old/old first (determinism), then old/new.
    def run(dataset, steps=50):
        configure_fp32_determinism(recipe.seed)
        k = FamilyKernel(recipe, initial, device="cuda")
        loaders = FamilyLoaders(recipe, dataset, tuning)
        losses = []
        for i, batch in enumerate(loaders.train(0, 0)):
            if i >= steps:
                break
            losses.append(float(k.step(batch, check_finite=False)[0]))
        digest = state_dict_sha256(k.model.state_dict())
        loaders.close()
        return losses, digest
    ref_a, ref_b, new = run(d_old), run(d_old), run(d_new)
    for name, (l, d) in (("old A", ref_a), ("old B", ref_b), ("new  ", new)):
        print("  %s loss step 1/10/50: %s %s %s  state sha %s" % (name, hexf(l[0]), hexf(l[9]), hexf(l[49]), d[:16]), flush=True)
    print("LOSS old/old bitwise:", ref_a == ref_b, "| old/new bitwise all 50 steps:", ref_a == new, flush=True)
    print("loss values old/new step 1,10,50:", [(ref_a[0][i], new[0][i]) for i in (0, 9, 49)], flush=True)


def synthetic(args):
    """Edge cases the 100K data lacks: ring-cap (>16 rings), 1-node graphs, long chains, 51-node molecules."""
    from torch_geometric.data import Data
    from molgap.demol_cin_resident import CompactCells

    def graph(n, edges, i):
        ei = torch.tensor([[a for a, b in edges] + [b for a, b in edges], [b for a, b in edges] + [a for a, b in edges]], dtype=torch.long).reshape(2, -1)
        ea = torch.randint(0, 4, (ei.shape[1] // 2, 3)).repeat(2, 1)
        return Data(x=torch.randint(0, 100, (n, 9)), edge_index=ei, edge_attr=ea, y=torch.tensor([float(i)]), source_idx=torch.tensor([i]))

    def grid(r, c):
        e = [(i * c + j, i * c + j + 1) for i in range(r) for j in range(c - 1)] + [(i * c + j, (i + 1) * c + j) for i in range(r - 1) for j in range(c)]
        return r * c, e

    torch.manual_seed(0)
    gs = [graph(*grid(6, 6), 0), graph(*grid(5, 5), 1), graph(1, [], 2), graph(2, [(0, 1)], 3), graph(51, [(i, i + 1) for i in range(50)], 4),
          graph(8, [(i, (i + 1) % 8) for i in range(8)], 5), graph(*grid(3, 17), 6), graph(5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 7)]
    class Base(list):
        pass
    base = Base(gs)
    store = CompactCells.build(base, chunk=3).upload("cuda")
    rings = (store.ptr["ring"][1:] - store.ptr["ring"][:-1]).tolist()
    print("synthetic ring counts per graph:", rings, "(cap 16)")
    fails = []
    for label, idx in (("all", list(range(8))), ("reverse", list(range(7, -1, -1))), ("single ring-cap", [0]), ("1-node", [2]), ("mix", [2, 0, 3, 4, 2])):
        old = prepare_layout(Batch.from_data_list([extract_cellular_complex(base[i]) for i in idx]))
        new = store.batch(idx)
        compare(old, new, label, fails)
        compare_layout(old, new, label, fails)
    print("SYNTHETIC EDGE CASES:", "PASS" if not fails else "FAIL %r" % fails[:5], flush=True)


def mem(args):
    mon = Monitor()
    mon.start()
    if args.variant == "new":
        import os
        os.environ["MOLGAP_DEMOL_GPU_RESIDENT"] = "1"
    from molgap.demol_cin import attach_inputs
    recipe, data, initial, tuning = setup(args)
    base_peak = dict(mon.peak)
    t0 = time.perf_counter()
    data = attach_inputs(None, None, data)
    build_s = time.perf_counter() - t0
    after_build = dict(mon.peak)
    configure_fp32_determinism(recipe.seed)
    kernel = FamilyKernel(recipe, initial, device="cuda")
    loaders = FamilyLoaders(recipe, data, tuning)
    torch.cuda.synchronize()
    resident_bytes = torch.cuda.memory_allocated()
    t1 = time.perf_counter()
    for i, batch in enumerate(loaders.train(0, 0)):
        if i >= args.steps:
            break
        kernel.step(batch, check_finite=False)
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - t1) / args.steps
    info = dict(variant=args.variant, train_rows=len(data.train), dev_rows=len(data.development),
                build_seconds=round(build_s, 1), seconds_per_step=round(step_s, 4),
                cuda_allocated_after_model_and_store_MiB=round(resident_bytes / 2**20, 1),
                cuda_peak_allocated_MiB=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                host_MiB_after_data_load=base_peak, host_MiB_peak_during_build={k: round(v / 2**20, 1) if k != "handles" else v for k, v in after_build.items()},
                host_peak_whole_run=mon.result())
    if args.variant == "new":
        info["compact_store_MiB"] = round((data.train.store.nbytes + data.development.store.nbytes) / 2**20, 1)
        info["bytes_per_molecule"] = round((data.train.store.nbytes + data.development.store.nbytes) / (len(data.train) + len(data.development)), 1)
    print(json.dumps(info, indent=1), flush=True)
    Path(args.out, "mem_%s.json" % args.variant).write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("equiv", "mem", "synthetic"))
    ap.add_argument("--recipe", type=Path, default=ROOT / "experiments/pcqm_demol_cin_s43_100k_ims_20261120/j1_demol_cin/recipe.json")
    ap.add_argument("--root", type=Path, required=True, help="directory holding the hard-linked dataset mount (manifest + shards)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", choices=("old", "new"))
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    {"equiv": equiv, "mem": mem, "synthetic": synthetic}[args.mode](args)


if __name__ == "__main__":
    main()
