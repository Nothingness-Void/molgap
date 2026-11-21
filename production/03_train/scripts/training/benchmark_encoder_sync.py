"""Time and audit the real repaired-2M trainer using synthetic CPU graphs."""

import argparse
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
import importlib.util
from io import StringIO
import json
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import torch
from torch_geometric.data import Data


EXPERTS = {"gps7": (7, 192), "gps9": (9, 192), "gps11_160": (11, 160)}


def synthetic_graphs(count):
    rng = torch.Generator().manual_seed(42)
    graphs = []
    for i in range(count):
        n = int(torch.randint(12, 41, (), generator=rng))
        a = torch.arange(n - 1)
        edge_index = torch.stack((torch.cat((a, a + 1)), torch.cat((a + 1, a))))
        graphs.append(Data(x=torch.rand(n, 9, generator=rng), edge_index=edge_index,
                           edge_attr=torch.rand(edge_index.size(1), 4, generator=rng),
                           y=torch.rand(1, 3, generator=rng), source_idx=torch.tensor([i])))
    return graphs


def traced_method(original, name, reads, active):
    def traced(self, *args, **kwargs):
        caller = sys._getframe(1)
        file = Path(caller.f_code.co_filename).name
        if active() and file in {"train_encoder.py", "_to_dense_batch.py", "_scatter.py",
                                 "retention.py", "grad_scaler.py"}:
            reads[f"{file}:{caller.f_lineno}:{name}"] += 1
        return original(self, *args, **kwargs)
    return traced


def run(trainer, args, directory, layers, width, profile=False):
    graph_path, model_path, metrics_path = (directory / name for name in ("graphs.pt", "model.pt", "metrics.json"))
    argv = ["train_encoder", "--kind", "gps", "--graphs", str(graph_path),
            "--model-out", str(model_path), "--metrics-out", str(metrics_path),
            "--no-embeddings", "--epochs", "1", "--num-layers", str(layers),
            "--hidden-channels", str(width), "--batch-size", str(args.batch_size),
            "--lr", "0.00008", "--replay-boundary", str(args.batch_size), "--replay-weight", "3"]
    if args.retention is not None:
        argv += ["--retention-teacher", str(directory / "teacher.pt"),
                 "--retention-boundary", str(args.batch_size), "--retention-weight", "0.25"]
        if args.retention == "cached":
            argv += ["--retention-targets-cache", str(directory / "targets.pt")]
    train_start, train_end = None, None
    reads = Counter()
    module_train = trainer.GPSWrapper.train
    forward = trainer._forward
    phase = torch.profiler.record_function("gps_training_epoch") if profile else None

    def train(model, mode=True):
        nonlocal train_start, train_end
        if mode:
            train_start = perf_counter()
            if phase is not None:
                phase.__enter__()
        elif train_start is not None and train_end is None:
            train_end = perf_counter()
            if phase is not None:
                phase.__exit__(None, None, None)
        return module_train(model, mode)

    def counted_forward(kind, model, batch):
        if train_start is None or train_end is not None:
            return forward(kind, model, batch)
        with torch.profiler.record_function("gps_training_forward"):
            return forward(kind, model, batch)

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(trainer.GPSWrapper, "train", train))
        stack.enter_context(redirect_stdout(StringIO()))
        if profile:
            stack.enter_context(patch.object(trainer, "_forward", counted_forward))
            for name in ("item", "cpu", "tolist", "numpy", "__bool__", "__float__", "__int__", "nonzero"):
                method = traced_method(getattr(torch.Tensor, name), name, reads,
                                       lambda: train_start is not None and train_end is None)
                stack.enter_context(patch.object(torch.Tensor, name, method))
            profiler = stack.enter_context(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]))
        trainer.main()
    if profile:
        for event in profiler.events():
            if event.name != "aten::nonzero":
                continue
            parents = []
            parent = event.cpu_parent
            while parent is not None:
                parents.append(parent.name)
                parent = parent.cpu_parent
            if "gps_training_forward" in parents:
                reads["gps_training_forward:aten::nonzero"] += 1
            elif "gps_training_epoch" in parents and not any("DataLoader" in name for name in parents):
                reads["gps_training_step:aten::nonzero"] += 1
    state_hash = sha256()
    for value in torch.load(model_path, map_location="cpu", weights_only=True).values():
        state_hash.update(value.numpy().tobytes())
    report = json.loads(metrics_path.read_text())
    metrics = [{key: value for key, value in row.items() if key != "time_s"} for row in report["log"]]
    return train_end - train_start, state_hash.hexdigest(), metrics, reads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--count-syncs", action="store_true")
    parser.add_argument("--retention", choices=("online", "cached"))
    args = parser.parse_args()
    if torch.cuda.is_available():
        parser.error("Run this CPU benchmark with CUDA_VISIBLE_DEVICES=''")
    if args.source is not None:
        sys.path.insert(0, str(args.source.resolve()))
    from molgap.constants import TRAIN_DIR

    trainer_path = TRAIN_DIR / "scripts" / "training" / "train_encoder.py"
    spec = importlib.util.spec_from_file_location("benchmark_train_encoder", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    torch.set_num_threads(1)
    with TemporaryDirectory() as work:
        directory = Path(work)
        n_train = args.steps * args.batch_size
        torch.save(synthetic_graphs((n_train * 5 + 3) // 4), directory / "graphs.pt")
        for expert, dims in EXPERTS.items():
            if args.retention is not None:
                torch.manual_seed(42)
                teacher = trainer.GPSWrapper(**{
                    **trainer.PARAMS_GPS_2D, "num_layers": dims[0], "hidden_channels": dims[1],
                })
                torch.save(teacher.state_dict(), directory / "teacher.pt")
            timings = [run(trainer, args, directory, *dims) for _ in range(args.repeats)]
            _, checkpoint, metrics, _ = timings[-1]
            result = {"expert": expert, "steps_per_second": args.steps / median(row[0] for row in timings),
                      "repeat_seconds": [row[0] for row in timings],
                      "checkpoint_sha256": checkpoint, "epoch_metrics": metrics}
            if args.count_syncs:
                result["training_reads"] = run(trainer, args, directory, *dims, profile=True)[-1]
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
