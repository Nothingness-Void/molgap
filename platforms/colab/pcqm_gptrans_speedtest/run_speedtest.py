"""Throughput matrix for GPTrans-T BondLocal + pair-update norm (12L/256/32/8).

Measures training-step speed only: no checkpoint, no weights saved, no
validation/test/protected role. Data are training rows of a fixed dataset
(--data-root) or, without it, synthetic OGB-style graphs (sizes ~ PCQM4Mv2).
The step is built here (forward, backward, clip, AdamW) with a tiny lr; the
model, input layout and loaders are the repository's own.
"""
import argparse, gc, hashlib, json, os, platform, random, sys, time
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from molgap.family_training_data import DatasetSelection, PassBatchSampler, load_training_data
from molgap.family_training_device_inputs import DeviceResidentTrain, device_training_stream
from molgap.family_training_loaders import training_loader_options
from molgap.gptrans_bond_pair_norm import construct_family_model
from molgap.gptrans_execution_backend import bind_model_layout, prepare_layout
from molgap.gptrans_execution_hotpath import MODEL_FIELDS

MEAN, STD = 5.69484785548952, 1.1609073178422544  # frozen GPTrans recipe constants
ATOM_DIMS, BOND_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2), (5, 6, 2)


def synthetic(rows, seed=0):
    """Random connected graphs; atom count ~ N(14, 3.5) clipped to [3, 51]."""
    rng = random.Random(seed)
    out = []
    for _ in range(rows):
        n = min(51, max(3, round(rng.gauss(14, 3.5))))
        pairs = {(rng.randrange(i), i) for i in range(1, n)}
        while len(pairs) < n + 1 and n > 3:  # a ring or so
            a, b = rng.sample(range(n), 2)
            pairs.add((min(a, b), max(a, b)))
        both = [p for a, b in pairs for p in ((a, b), (b, a))]
        out.append(Data(
            x=torch.tensor([[rng.randrange(d) for d in ATOM_DIMS] for _ in range(n)]),
            edge_index=torch.tensor(both).t().contiguous(),
            edge_attr=torch.tensor([[rng.randrange(d) for d in BOND_DIMS] for _ in both]),
            y=torch.tensor([rng.gauss(MEAN, STD)])))
    return out


class Fp32Head(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, value):
        with torch.autocast("cuda", enabled=False):
            return self.inner(value.float())


class FloatOut(torch.nn.Module):
    """BondLocal index_add_ needs a dtype equal to the FP32 node stream."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, value):
        return self.inner(value).float()


def batches(dataset, bs, workers, resident, seed=0):
    """Endless stream of CPU-collated (or device-resident) training batches."""
    rows, epoch = len(dataset), 0
    store = DeviceResidentTrain(dataset, rows).upload("cuda") if resident else None
    loader = None if resident else DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True,
                                              **training_loader_options(workers))
    while True:
        if resident:
            sampler = PassBatchSampler(rows, bs, seed, algorithm="torch-pass-randperm", pass_index=epoch)
            yield from device_training_stream(store, sampler)
        else:
            yield from loader
        epoch += 1


def run_cell(dataset, bs, precision, workers, resident, compiled, warmup, steps, device):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = precision == "tf32"
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = bind_model_layout(construct_family_model(None, None))
    if precision == "bf16":
        model.readout = Fp32Head(model.readout)
        for block in model.blocks:
            block.output = FloatOut(block.output)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05, foreach=True)
    run = torch.compile(model, dynamic=True) if compiled else model
    stream = batches(dataset, bs, workers, resident)
    wait = total = 0.0
    nodes, padding, max_n = [], [], 0
    begin = time.perf_counter()
    for step in range(warmup + steps):
        t0 = time.perf_counter()
        cpu = next(stream)
        t1 = time.perf_counter()
        prepared = prepare_layout(cpu).to(device, *MODEL_FIELDS, "y", non_blocking=True)
        model._execution_layout = prepared.execution_layout
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            out = run(prepared.x, prepared.edge_index, prepared.edge_attr, prepared.batch, None).view(-1)
        loss = torch.nn.functional.smooth_l1_loss(out.float(), (prepared.y.view(-1).float() - MEAN) / STD)
        del model._execution_layout
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss.item()  # one sync per step keeps compute inside the timed span
        t2 = time.perf_counter()
        if step == warmup - 1:
            warm_seconds = t2 - begin
        if step >= warmup:
            wait += t1 - t0; total += t2 - t0
            layout = prepared.execution_layout
            counts = layout["mask"].sum(1)
            max_n = max(max_n, int(layout["max_nodes"]))
            nodes.append(int(counts.sum()))
            padding.append(1 - int(counts.sum()) / (bs * int(layout["max_nodes"])))
    return {"steps_per_s": steps / total, "graphs_per_s": steps * bs / total, "wait_fraction": wait / total,
            "timed_seconds": total, "warmup_seconds": warm_seconds,
            "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "max_nodes_N": max_n, "mean_padding_ratio": sum(padding) / len(padding),
            "mean_real_nodes_per_batch": sum(nodes) / len(nodes), "final_loss": float(loss)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", help="dir holding a fixed-dataset manifest; training role only")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--rows", type=int, default=20480, help="training rows used (or synthetic graphs)")
    p.add_argument("--batches", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "bf16"])
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--no-switches", action="store_true", help="skip resident-input and torch.compile cells")
    p.add_argument("--out", default="speedtest_manifest.json")
    a = p.parse_args()
    device = torch.device("cuda")
    start = time.time()
    if a.data_root:
        manifest = Path(a.data_root) / a.manifest
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        full = load_training_data(Path(a.data_root), DatasetSelection(a.manifest, sha, False)).train
        dataset, source = Subset(full, range(min(a.rows, len(full)))), {"kind": "fixed-dataset-train", "manifest_sha256": sha, "rows": min(a.rows, len(full))}
    else:
        dataset, source = synthetic(a.rows), {"kind": "synthetic", "rows": a.rows, "note": "OGB-style random graphs, N~N(14,3.5)"}
    doc = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": source,
           "gpu": torch.cuda.get_device_name(0), "cpu_cores": os.cpu_count(), "torch": torch.__version__,
           "python": platform.python_version(), "warmup_steps": a.warmup, "timed_steps": a.steps,
           "bf16_note": "autocast; readout runs FP32 (Fp32Head); bond messages are cast to FP32 before the scatter-add; LayerNorm/softmax/loss are FP32 under autocast policy",
           "cells": [], "wall_seconds": None}

    def cell(bs, precision, workers, resident=False, compiled=False):
        spec = {"batch": bs, "precision": precision, "workers": 0 if resident else workers,
                "device_resident_inputs": resident, "torch_compile": compiled}
        try:
            spec.update(status="ok", **run_cell(dataset, bs, precision, workers, resident, compiled, a.warmup, a.steps, device))
        except torch.OutOfMemoryError:
            spec.update(status="oom")
        except Exception as error:  # record and keep going; one bad cell must not end the matrix
            spec.update(status="error", error="%s: %s" % (type(error).__name__, str(error)[:300]))
        gc.collect(); torch.cuda.empty_cache()
        doc["cells"].append(spec)
        doc["wall_seconds"] = time.time() - start
        tmp = Path(a.out + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1)); os.replace(tmp, a.out)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in spec.items()}), flush=True)
        return spec

    for bs in a.batches:
        for precision in a.precisions:
            for workers in a.workers:
                cell(bs, precision, workers)
    if not a.no_switches:
        ok = [c for c in doc["cells"] if c["status"] == "ok" and c["batch"] == 1024 and c["precision"] == "fp32"]
        best = max(ok, key=lambda c: c["graphs_per_s"])["workers"] if ok else a.workers[0]
        cell(1024, "fp32", best, resident=True)
        cell(1024, "fp32", best, compiled=True)
    print("manifest:", a.out)


if __name__ == "__main__":
    sys.exit(main())
