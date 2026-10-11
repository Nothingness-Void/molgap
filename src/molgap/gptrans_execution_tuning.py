"""Opt-in resource/cache tuning layered over the qualified v1 backend.

This module does not change the v1 source, launch jobs, or qualify a runtime.
Cached tensors are borrowed read-only on CPU; PyG device transfer uses a fresh
storage wrapper, so .to() cannot move the retained cache to the accelerator.
"""
from contextlib import contextmanager
from copy import copy
from dataclasses import asdict, dataclass
import os
import time

import torch

from .gptrans_execution_backend import (
    EpochBatches, PreparedLoader, assert_static_dataset, backend, buffered_evaluate,
)
from .screen_policy import canonical_fingerprint
from .gptrans_execution_hotpath import (
    BoundForeachEMA, ReusedWeightView, TransferLoader, development_cpu_view, LoadedDatasetCache,
)


@dataclass(frozen=True)
class ExecutionTuning:
    cpu_threads: int = 1
    training_workers: int = 1
    development_workers: int = 0
    independent_models: int = 2
    total_cpu_budget: int = 4
    development_cache_bytes: int = 256 * 1024 * 1024
    cache_development: bool = True
    selective_transfers: bool = True
    cache_ema_plan: bool = True
    reuse_weight_backup: bool = True
    reuse_loaded_datasets: bool = True

    def __post_init__(self):
        for name in ("cpu_threads", "independent_models", "total_cpu_budget", "development_cache_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer required: " + name)
        for name in ("training_workers", "development_workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Nonnegative worker count required: " + name)
        for name in ('cache_development', 'selective_transfers', 'cache_ema_plan',
                     'reuse_weight_backup', 'reuse_loaded_datasets'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be boolean")
        # Workers are single-threaded by PyTorch; both persistent pools can live
        # concurrently. This is a declared process budget, not CPU utilization.
        demand = self.independent_models * (
            self.cpu_threads + self.training_workers + self.development_workers)
        if demand > self.total_cpu_budget:
            raise ValueError("Declared aggregate CPU budget exceeded")

    @property
    def identity(self):
        return canonical_fingerprint(asdict(self))

    def environment(self):
        return {name: str(self.cpu_threads) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}


def _cpu_storage_bytes(batch):
    seen = set()
    def count(value):
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                raise ValueError("Development cache permits CPU tensors only")
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(count(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(count(v) for v in value)
        return 0
    return count(batch.to_dict()) + count(getattr(batch, "_slice_dict", {})) + count(
        getattr(batch, "_inc_dict", {}))


class PreparedDevelopmentCache:
    """Publish only an exhausted, row-validated pass; overflow streams normally."""
    def __init__(self, source, *, rows, batch_size, max_bytes, source_start=100000,
                 retain_model_inputs=False):
        if (type(rows) is not int or rows < 1 or type(max_bytes) is not int or max_bytes < 1
                or batch_size != 128 or type(source_start) is not int or source_start < 0
                or type(retain_model_inputs) is not bool):
            raise ValueError("Invalid cache rows, budget or physical batch")
        self.source = source
        self.rows, self.batch_size, self.max_bytes = rows, batch_size, max_bytes
        self.source_start = source_start
        self.retain_model_inputs = retain_model_inputs
        self.batches = None
        self.busy = False
        self.disabled_reason = None
        self.stats = dict(cold_passes=0, warm_passes=0, cached_tensor_storage_bytes=0)

    def __iter__(self):
        if self.busy:
            raise ValueError("Concurrent development-cache iteration is forbidden")
        self.busy = True
        pending, size, cursor = [], 0, 0
        try:
            if self.batches is not None:
                self.stats["warm_passes"] += 1
                for batch in self.batches:
                    yield copy(batch)
                return
            self.stats["cold_passes"] += 1
            for batch in self.source:
                if self.retain_model_inputs:
                    batch = development_cpu_view(batch)
                count = min(self.batch_size, self.rows - cursor)
                index = getattr(batch, "source_idx", getattr(batch, "row_index", None))
                if (count <= 0 or int(batch.num_graphs) != count or index is None
                        or not torch.equal(index.view(-1), torch.arange(
                            self.source_start + cursor, self.source_start + cursor + count))):
                    raise ValueError("Development cache row count/order changed")
                if self.disabled_reason is None:
                    measured = _cpu_storage_bytes(batch)
                    size += measured
                    if size > self.max_bytes:
                        pending.clear()
                        self.disabled_reason = "tensor_storage_budget_exceeded"
                    else:
                        pending.append(batch)
                cursor += count
                # Data.__copy__ isolates the store mapping; tensors are not
                # duplicated. The owning GPU forward must not mutate CPU input.
                yield copy(batch)
            if cursor != self.rows:
                raise ValueError("Development cache pass is incomplete")
            if self.disabled_reason is None:
                self.batches = pending
                self.stats["cached_tensor_storage_bytes"] = size
        finally:
            self.busy = False

    def clear(self):
        self.batches = None
        self.stats["cached_tensor_storage_bytes"] = 0


@contextmanager
def tuned_backend(owner, tuning: ExecutionTuning, *, timing_root=None):
    """Keep native training/evaluation ownership and expose explicit v3 identity."""
    from torch_geometric.loader import DataLoader
    if not isinstance(tuning, ExecutionTuning) or owner.PHYSICAL_BATCH != 128:
        raise ValueError("Explicit tuning and frozen BS128 required")
    previous_threads = torch.get_num_threads()
    previous_env = {key: os.environ.get(key) for key in tuning.environment()}
    loaders, caches, ema_statistics = {}, {}, []
    weight_view = ReusedWeightView()
    dataset_cache = LoadedDatasetCache(owner._load_datasets) if tuning.reuse_loaded_datasets else None
    report = {"format": "molgap-gptrans-execution-tuning-v3", "tuning": asdict(tuning),
              "tuning_identity": tuning.identity, "previous_cpu_threads": previous_threads,
              "cpu_count_reported_by_os": os.cpu_count(), "remote_qualified": False,
              "cache_budget_scope": "retained tensor storage, not total process RSS"}
    try:
        torch.set_num_threads(tuning.cpu_threads)
        os.environ.update(tuning.environment())
        with backend(owner, True, timing_root) as timing:
            names = ("_training_loader", "_development_loader", "_evaluate", "_scientific_fields",
                     "_make_training_state", "_load_datasets")
            saved = {name: getattr(owner, name) for name in names}
            def make_loader(graphs, workers, **kwargs):
                options = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
                if workers:
                    options["prefetch_factor"] = 2
                return PreparedLoader(DataLoader(graphs, **options, **kwargs))
            def training_loader(graphs, epoch):
                key = ("train", id(graphs))
                if key not in loaders:
                    assert_static_dataset(graphs)
                    sampler, generator = EpochBatches(owner, len(graphs)), torch.Generator()
                    loader = make_loader(graphs, tuning.training_workers, batch_sampler=sampler,
                                         generator=generator)
                    loaders[key] = (graphs, loader, sampler, generator)
                _, loader, sampler, generator = loaders[key]
                sampler.epoch = epoch
                generator.manual_seed(owner.SEED + epoch)
                return TransferLoader(loader, training=True) if tuning.selective_transfers else loader
            def development_loader(graphs):
                # Preserve v1's main CPU RNG draw, even on warm cache hits.
                torch.empty((), dtype=torch.int64).random_()
                key = ("dev", id(graphs))
                if key not in loaders:
                    assert_static_dataset(graphs)
                    loader = make_loader(graphs, tuning.development_workers,
                        batch_size=128, shuffle=False, generator=torch.Generator().manual_seed(314159))
                    loaders[key] = (graphs, loader)
                    if tuning.cache_development:
                        caches[key] = PreparedDevelopmentCache(loader, rows=owner.DEVELOPMENT_ROWS,
                            batch_size=128, max_bytes=tuning.development_cache_bytes,
                            retain_model_inputs=tuning.selective_transfers)
                return caches.get(key, loaders[key][1])
            def evaluation_inputs(graphs):
                source = development_loader(graphs)
                return TransferLoader(source, training=False) if tuning.selective_transfers else source
            def evaluate(model, ema, graphs, mean, std, **kwargs):
                torch.cuda.synchronize()
                began = time.perf_counter()
                try:
                    if not tuning.reuse_weight_backup:
                        return buffered_evaluate(owner, owner._forward, evaluation_inputs,
                            model, ema, graphs, mean, std, **kwargs)
                    options = dict(kwargs)
                    weights = options.pop('weights', 'ema')
                    with weight_view.use(model, ema, weights):
                        return buffered_evaluate(owner, owner._forward, evaluation_inputs,
                            model, ema, graphs, mean, std, weights='live', **options)
                finally:
                    torch.cuda.synchronize()
                    timing["evaluation_seconds"] += time.perf_counter() - began
                    timing["evaluation_passes"] += 1
                    report["development_caches"] = [
                        {**cache.stats, "disabled_reason": cache.disabled_reason}
                        for cache in caches.values()]
                    report['weight_backup_allocations'] = weight_view.allocations
                    report['weight_backup_tensor_bytes'] = weight_view.retained_tensor_bytes
            def training_state(*args, **kwargs):
                model, optimizer, scheduler, ema = saved['_make_training_state'](*args, **kwargs)
                if tuning.cache_ema_plan:
                    bound = BoundForeachEMA(model, ema.decay)
                    bound.load_state_dict(ema.state_dict())
                    ema = bound
                    ema_statistics.append(bound.statistics)
                return model, optimizer, scheduler, ema
            def scientific_fields(*args, **kwargs):
                fields = saved["_scientific_fields"](*args, **kwargs)
                fields["execution_backend"] = "optimized-tuning-v3:" + tuning.identity
                return fields
            for name, fn in zip(names, (training_loader, development_loader, evaluate,
                                       scientific_fields, training_state,
                                       dataset_cache if dataset_cache is not None else saved['_load_datasets'])):
                setattr(owner, name, fn)
            report["phase_timing"] = timing
            try:
                yield report
            finally:
                report['ema_tensor_plans'] = [dict(stats) for stats in ema_statistics]
                report['dataset_cache'] = dict(dataset_cache.statistics) if dataset_cache is not None else None
                for name, value in saved.items():
                    setattr(owner, name, value)
    finally:
        try:
            weight_view.clear()
            if dataset_cache is not None:
                dataset_cache.clear()
            for cache in caches.values():
                cache.clear()
            for item in loaders.values():
                iterator = getattr(item[1].loader, "_iterator", None)
                if iterator is not None:
                    iterator._shutdown_workers()
        finally:
            try:
                torch.set_num_threads(previous_threads)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
