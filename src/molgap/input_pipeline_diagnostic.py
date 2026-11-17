"""Bounded train-only sampling diagnostics, not a registered training policy."""
from __future__ import annotations

from copy import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from .family_training_data import PassBatchSampler

STRATEGIES = ("global", "block", "bucket")


class DiagnosticBatchSampler:
    """Preserve the reference membership/tail; vary only order and grouping.

    This deliberately does not register bucket sampling in FamilyRecipe. Batch
    composition affects BN and gradients and still needs quality qualification.
    """
    def __init__(self, counts, batch_size, seed, *, strategy, window_batches=20,
                 pass_index=0, start_batch=0, drop_last=True):
        self.counts = tuple(counts)
        if any(type(n) is not int or n < 1 for n in self.counts):
            raise ValueError("Positive integer node counts required")
        if strategy not in STRATEGIES or type(window_batches) is not int or window_batches < 2:
            raise ValueError("Unsupported diagnostic strategy/window")
        self.reference = PassBatchSampler(len(self.counts), batch_size, seed,
            algorithm="python-epoch-shuffle", pass_index=pass_index,
            start_batch=start_batch, drop_last=drop_last)
        self.strategy, self.window_batches = strategy, window_batches

    def batches(self):
        ref = self.reference
        # Choose/drop the reference tail before regrouping, not once per bucket.
        order = [i for batch in PassBatchSampler(ref.rows, ref.batch_size, ref.seed,
            algorithm=ref.algorithm, pass_index=ref.pass_index,
            drop_last=ref.drop_last) for i in batch]
        rng = random.Random(ref.seed * 1_000_003 + ref.pass_index + 17)
        window = ref.batch_size * self.window_batches
        if self.strategy == "block":
            blocks = {}
            for i in order:
                blocks.setdefault(i // window, []).append(i)
            groups = list(blocks.values())
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
            order = [i for group in groups for i in group]
        elif self.strategy == "bucket":
            # Random windows change companions each pass; equal-size ties keep
            # their randomized order rather than fixing batches for all epochs.
            order = [i for start in range(0, len(order), window)
                     for i in sorted(order[start:start + window], key=self.counts.__getitem__)]
        batches = [order[start:start + ref.batch_size]
                   for start in range(0, len(order), ref.batch_size)]
        if self.strategy == "bucket":
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches()[self.reference.start_batch:])

    def __len__(self):
        return len(self.reference)


def packed_node_counts(dataset):
    """Read packed slices, without materializing graphs or reading dev shards."""
    parts = getattr(dataset, "datasets", (dataset,))
    counts = []
    for part in parts:
        if getattr(part, "_indices", None) is not None:
            raise ValueError("Expected an unfiltered accepted packed shard")
        slices = part.slices["x"]
        sizes = (slices[1:] - slices[:-1]).tolist()
        if len(sizes) != len(part) or any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError("Invalid packed node slices")
        counts.extend(sizes)
    if len(counts) != len(dataset):
        raise ValueError("Packed metadata row count differs")
    return counts


def padding_statistics(counts, batches):
    nodes = slots = squared_nodes = squared_slots = samples = 0
    maxima = []
    for indices in batches:
        sizes = [counts[i] for i in indices]
        maximum = max(sizes)
        nodes += sum(sizes)
        slots += len(sizes) * maximum
        squared_nodes += sum(n * n for n in sizes)
        squared_slots += len(sizes) * maximum * maximum
        samples += len(sizes)
        maxima.append(maximum)
    return {"batches": len(maxima), "samples": samples, "actual_nodes": nodes,
        "padded_node_slots": slots, "node_padding_fraction": 1 - nodes / slots,
        "dense_attention_pair_slots": squared_slots,
        "attention_padding_proxy_fraction": 1 - squared_nodes / squared_slots,
        "mean_max_nodes": statistics.mean(maxima), "largest_batch_max_nodes": max(maxima)}


def index_digest(values):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _shutdown(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def _loader(dataset, sampler, workers, seed):
    import torch
    from torch_geometric.loader import DataLoader
    from .family_training_loaders import training_loader_options
    return DataLoader(dataset, batch_sampler=sampler,
        generator=torch.Generator().manual_seed(seed), **training_loader_options(workers))


def _fresh_kernel(recipe, initial, family, device):
    import torch
    from .family_training_kernel import FamilyKernel, make_family_optimizer
    from .family_training_core import normalized_gap_update
    from .k1_screen_training import _forward, _model_batch_to_device
    from .training_reproducibility import configure_fp32_determinism

    configure_fp32_determinism(recipe.seed)
    if family == "k1":
        return FamilyKernel(recipe, initial, device=device)
    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper
    model = OGBEdgeStateStructuralGPSWrapper(in_channels=9, edge_dim=3,
        hidden_channels=192, num_layers=9, num_heads=4, dropout=.05,
        n_targets=1, pooling="mean", rwse_dim=16, edge_state_channels=64).to(device)
    optimizer = make_family_optimizer(recipe, model.parameters())

    class GPSProbe:
        def __init__(self):
            self.model = model

        def step(self, cpu_batch, *, check_finite):
            batch = _model_batch_to_device(copy(cpu_batch), device, training=True)
            loss = normalized_gap_update(model, optimizer, batch,
                recipe.target_mean_eV, recipe.target_std_eV, forward=_forward,
                clip_norm=recipe.gradient_clip, check_finite=check_finite, cast_target_float=False)
            return loss, loss * batch.num_graphs, int(batch.num_graphs)

        def close(self):
            pass

    return GPSProbe()


def run(root, config_path, output, *, gpu=False):
    """Time existing packed data and update kernels; discard every probe model."""
    import torch
    from torch.utils.data import Subset
    from .family_training_data import DatasetSelection, load_training_data
    from .family_training_recipe import FamilyRecipe, ScheduleSpec
    from .training_reproducibility import (
        atomic_json, canonical_fingerprint, configure_fp32_determinism, sha256_file,
    )
    from .v4_runtime import state_dict_sha256

    root, output = Path(root).resolve(), Path(output).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    trajectory = json.loads((root / config["trajectory_ref"]).read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective diagnostic required")
    if trajectory["state_at_start"]["source_config_identity"] != canonical_fingerprint(config):
        raise ValueError("Config differs from its prospective identity")
    for name, digest in config["source_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Frozen diagnostic source changed: " + name)
    if output.exists():
        raise FileExistsError("Use a fresh diagnostic output")
    output.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + config["total_wall_seconds"]

    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Diagnostic wall budget exhausted")

    configure_fp32_determinism(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU graph admission must precede CUDA initialization")
    selection = DatasetSelection(config["manifest"], config["manifest_sha256"], False)
    t0 = time.perf_counter()
    data = load_training_data(Path(config["input_root"]), selection)
    load_seconds = time.perf_counter() - t0
    if data.train_rows != config["train_rows"] or data.development is not None:
        raise ValueError("Expected the pinned TRAIN-only view")
    counts = packed_node_counts(data.train)
    recipe = FamilyRecipe("neural_atom_k1", "reference", config["seed"],
        config["initialization_sha256"], selection, data.train_rows, 0,
        data.train_rows // 128, 4e-4, 1e-5, ScheduleSpec("cosine_epoch", 40),
        "final-live", "python-epoch-shuffle", 50,
        target_mean_eV=config["target_mean_eV"], target_std_eV=config["target_std_eV"],
        torch_version=str(torch.__version__).split("+", 1)[0])

    result = {"format": "molgap-input-pipeline-diagnostic-v1", "status": "RUNNING",
        "source_commit": config["source_commit"], "config_sha256": sha256_file(Path(config_path)),
        "manifest_sha256": selection.manifest_sha256,
        "train_rows": data.train_rows, "development_rows_consumed": 0,
        "protected_roles_consumed": False, "quality_evaluated": False,
        "training_replay_ready": False, "startup_data_seconds": load_seconds,
        "node_count_sha256": index_digest(counts), "cpu": [], "gpu": [],
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "cpu_threads": torch.get_num_threads()},
        "limits": ["No cold disk-cache claim: SHA verification reads the shard bytes",
            "First versus repeated traversal includes PyG lazy graph cache effects",
            "GPU measures a fixed train subset, not a full100K GPU epoch",
            "CUDA event span includes transfer/update and host submission gaps, not exclusive kernel time",
            "Batch regrouping changes BN/gradients; no precision or MAE equivalence",
            "GPS9 is matched192/9 EdgeState scratch profiling, not production timing"]}
    atomic_json(output / "result.json", result)
    for workers in config["cpu_workers"]:
        for strategy in STRATEGIES:
            guard()
            # Start each policy with a new lazy graph cache; do not let one
            # strategy's object-cache warmup masquerade as another's speedup.
            for part in data.train.datasets:
                part._data_list = None
            sampler = DiagnosticBatchSampler(counts, 128, config["seed"], strategy=strategy,
                window_batches=config["window_batches"])
            batches = sampler.batches()
            stat = padding_statistics(counts, batches)
            stat.update(strategy=strategy, workers=workers,
                order_sha256=index_digest([i for batch in batches for i in batch]),
                membership_sha256=index_digest(sorted(i for batch in batches for i in batch)))
            loader = _loader(data.train, sampler, workers, config["seed"])
            try:
                traversals = []
                for repeat in range(config["cpu_repeats"]):
                    guard()
                    start = time.perf_counter()
                    seen = 0
                    first = None
                    for batch in loader:
                        guard()
                        seen += int(batch.num_graphs)
                        if first is None:
                            first = time.perf_counter() - start
                    elapsed = time.perf_counter() - start
                    if seen != stat["samples"]:
                        raise RuntimeError("Loader exposure differs from its sampling plan")
                    traversals.append({"repeat": repeat, "epoch_seconds": elapsed,
                        "first_batch_seconds": first, "molecules_per_second": seen / elapsed,
                        "samples": seen})
                stat["traversals"] = traversals
            finally:
                _shutdown(loader)
            result["cpu"].append(stat)
            atomic_json(output / "result.json", result)
            print(json.dumps({"phase": "cpu", "strategy": strategy, "workers": workers,
                              "traversals": traversals}), flush=True)
            gc.collect()
    if gpu:
        guard()
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != config["accelerator"]:
            raise ValueError("Expected the declared local RTX5060")
        result["runtime"]["gpu"] = torch.cuda.get_device_name(0)
        initial = Path(config["initial_path"])
        if sha256_file(initial) != config["initial_file_sha256"]:
            raise ValueError("Frozen K1 initialization bytes changed")
        full_order = [i for batch in PassBatchSampler(data.train_rows, 128, config["seed"],
            algorithm="python-epoch-shuffle") for i in batch]
        chosen = sorted(full_order[:config["gpu_rows"]])
        if len(chosen) != config["gpu_rows"] or len(chosen) % 128:
            raise ValueError("Fixed GPU subset must contain complete BS128 batches")
        subset = Subset(data.train, chosen)
        subset_counts = [counts[i] for i in chosen]
        result["gpu_source_idx_sha256"] = index_digest([data.train_source_start + i for i in chosen])
        result["gpu_rows"] = len(chosen)
        gpu_start = time.perf_counter()
        gpu_deadline = min(deadline, gpu_start + config["gpu_wall_seconds"])
        for family in config["gpu_families"]:
            initial_digest = None
            for repeat in range(config["gpu_repeats"]):
                order = list(STRATEGIES)
                random.Random(config["seed"] + repeat).shuffle(order)
                for strategy in order:
                    guard()
                    if time.perf_counter() >= gpu_deadline:
                        raise TimeoutError("GPU diagnostic ceiling exhausted")
                    kernel = _fresh_kernel(recipe, initial, family, "cuda:0")
                    kernel.model.train()
                    sampler = DiagnosticBatchSampler(subset_counts, 128, config["seed"],
                        strategy=strategy, window_batches=config["window_batches"])
                    loader = _loader(subset, sampler, config["gpu_workers"], config["seed"])
                    try:
                        warm = next(iter(loader))
                        for _ in range(config["gpu_warmup_steps"]):
                            guard()
                            kernel.step(warm, check_finite=True)
                        torch.cuda.synchronize()
                        kernel.close()
                        del kernel
                        kernel = None
                        kernel = _fresh_kernel(recipe, initial, family, "cuda:0")
                        kernel.model.train()
                        digest = state_dict_sha256(kernel.model.state_dict())
                        initial_digest = digest if initial_digest is None else initial_digest
                        if digest != initial_digest:
                            raise RuntimeError("Probe initialization changed across policies")
                        torch.cuda.reset_peak_memory_stats()
                        iterator = iter(loader)
                        # Worker/prefetch startup is outside the measured steady
                        # update window. CPU epoch records retain startup costs.
                        torch.cuda.synchronize()
                        stamp = time.perf_counter()
                        waits, events, losses, samples = [], [], [], 0
                        for step in range(len(sampler)):
                            guard()
                            if time.perf_counter() >= gpu_deadline:
                                raise TimeoutError("GPU diagnostic ceiling exhausted")
                            wait_start = time.perf_counter()
                            batch = next(iterator)
                            waits.append(time.perf_counter() - wait_start)
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            begin.record()
                            loss, _, rows = kernel.step(batch, check_finite=(step % 50 == 0))
                            end.record()
                            losses.append(loss)
                            events.append((begin, end))
                            samples += rows
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - stamp
                        if samples != len(chosen) or not bool(torch.isfinite(torch.stack(losses)).all()):
                            raise RuntimeError("Invalid measured exposure or nonfinite probe")
                        observation = {"family": family, "strategy": strategy, "repeat": repeat,
                            "initialization_sha256": digest, "steps": len(events), "samples": samples,
                            "pipeline_seconds": elapsed, "molecules_per_second": samples / elapsed,
                            "loader_wait_mean_ms": 1000 * statistics.mean(waits),
                            "cuda_step_span_mean_ms": statistics.mean(a.elapsed_time(b) for a, b in events),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "padding": padding_statistics(subset_counts, sampler.batches())}
                        result["gpu"].append(observation)
                        atomic_json(output / "result.json", result)
                        print(json.dumps({k: v for k, v in observation.items() if k != "padding"}), flush=True)
                    finally:
                        _shutdown(loader)
                        if kernel is not None:
                            kernel.close()
                        del kernel, loader
                        gc.collect()
                        torch.cuda.empty_cache()
        result["gpu_allocation_worker_seconds"] = time.perf_counter() - gpu_start
    result["status"] = "COMPLETE"
    result["cost"] = {"worker_wall_seconds": time.perf_counter() - started,
        "parent_cpu_seconds": time.process_time() - cpu_started,
        "cpu_child_seconds": None, "gpu_busy_seconds": None}
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", {"status": "COMPLETE",
        "result_sha256": sha256_file(output / "result.json"),
        "quality_evaluated": False, "training_replay_ready": False})
    return result
