"""Opt-in reuse of the original K1 list-sampler loader; no permutation logic."""
from contextlib import contextmanager


class K1LoaderReuse:
    """Exclusively own a fresh loader with caller-authorized deterministic data.

    ``deterministic_dataset=True`` asserts that the plain list/tuple or exact
    ConcatDataset of shared PackedGraphDataset shards and all their cached
    elements remain immutable and have no stochastic transforms or worker-RNG
    contract. Retained workers continue their RNG state, unlike fresh workers
    seeded by the original per-epoch recipe; stochastic datasets are unsupported.
    The loader generator is not reseeded. This is not fresh-worker RNG parity.
    Packed feature backing supports tensors/scalars only, plus a flat slice
    field dictionary. Python row containers are refused without traversal.
    Normal PyG lazy graph-cache population is not part of the backing signature.
    ``preserve_global_rng=True`` instead requires generator=None and the stock
    PyTorch 2.7.1 Map iterator on a CPU default device. Later persistent resets
    emulate the main CPU base-seed draw of a fresh iterator, not worker reseeding.

    Pass flattened exact-int indices from the existing epoch sampler, already
    sliced for its processed resume cursor and dropped tail. No order is generated
    here. Use ``with pool.stream(indices) as iterator`` and consume all batches.
    The native iterator is valid only inside that context. Never iterate the
    supplied loader directly while this owner is live.
    """

    def __init__(self, loader, *, deterministic_dataset: bool, preserve_global_rng: bool = False):
        import torch
        from torch.utils.data import BatchSampler, DataLoader

        if type(preserve_global_rng) is not bool:
            raise ValueError("preserve_global_rng must be an exact bool")
        if deterministic_dataset is not True:
            raise ValueError("Explicit deterministic immutable dataset authority required")
        if not isinstance(loader, DataLoader):
            raise ValueError("Expected an already constructed DataLoader")
        dataset_state = self._dataset_state(loader.dataset)
        if type(loader.sampler) is not list:
            raise ValueError("Original literal list sampler required")
        if type(loader.batch_sampler) is not BatchSampler or loader.batch_sampler.sampler is not loader.sampler:
            raise ValueError("Batch sampler must alias the original sampler list")
        if preserve_global_rng:
            self._check_global_rng_contract(loader)
        elif not isinstance(loader.generator, torch.Generator) or loader.generator.device.type != "cpu":
            raise ValueError("Explicit CPU loader generator required")
        if loader.num_workers > 0 and loader.persistent_workers is not True:
            raise ValueError("Worker reuse requires persistent workers")
        if loader.worker_init_fn is not None:
            raise ValueError("Worker RNG initialization contracts are unsupported")
        if getattr(loader, "_iterator", None) is not None:
            raise ValueError("Loader already has an iterator; exclusive fresh ownership required")
        self.loader = loader
        self._dataset = loader.dataset
        self._dataset_binding = dataset_state
        self._sampler = loader.sampler
        self._generator = loader.generator
        self._preserve_global_rng = preserve_global_rng
        self._iterator = None
        self._workers = ()
        self._active = False
        self._poisoned = False
        self._closed = False
        self._cleanup_done = False

    @staticmethod
    def _check_global_rng_contract(loader):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data.dataloader import _DatasetKind
        from torch.utils import _device

        # Source-audited 2.7.1 draws one int64 base seed in __init__, none in
        # Map _reset. Restrict versions/iterator overrides rather than repeatedly
        # parsing installed source or guessing semantics for another backend.
        if str(torch.__version__).split("+", 1)[0] != "2.7.1":
            raise ValueError("Global RNG preservation supports stock PyTorch 2.7.1 only")
        if (loader._dataset_kind != _DatasetKind.Map
                or type(loader).__iter__ is not DataLoader.__iter__
                or type(loader)._get_iterator is not DataLoader._get_iterator
                or "__iter__" in loader.__dict__ or "_get_iterator" in loader.__dict__):
            raise ValueError("Global RNG preservation requires native Map iterator semantics")
        if loader.generator is not None:
            raise ValueError("Global RNG preservation requires generator=None")
        # Inspect legacy defaults and factory modes before get_default_device
        # can allocate on an unindexed accelerator. Device contexts need not be
        # reflected in get_default_device (or _device.CURRENT_DEVICE).
        legacy_default = getattr(torch._C, "_get_default_device", None)
        if legacy_default is None or torch.device(legacy_default()).type != "cpu":
            raise ValueError("Global RNG preservation requires a CPU default device")
        modes = torch.overrides._get_current_function_mode_stack()
        if (any(type(mode) is not _device.DeviceContext or mode.device.type != "cpu"
                for mode in modes) or torch.get_default_device().type != "cpu"):
            raise ValueError("Global RNG preservation requires a CPU default device")

    @staticmethod
    def _dataset_state(dataset):
        from torch.utils.data import ConcatDataset

        if type(dataset) in (list, tuple):
            return tuple(id(member) for member in dataset)
        if type(dataset) is not ConcatDataset:
            raise ValueError("Only plain list/tuple or exact packed ConcatDataset supported")
        from .packed_graph_dataset import PackedGraphDataset
        import torch

        def backing(value):
            # Metadata only: no tensor values, graph separation or .data access.
            if type(value) is torch.Tensor:
                if value.device.type != "cpu":
                    raise ValueError("Packed backing tensors must be CPU tensors")
                try:
                    version = value._version
                except RuntimeError as error:
                    raise ValueError("Packed tensor mutation tracking required") from error
                return (id(value), version, tuple(value.shape), value.dtype, value.layout)
            if isinstance(value, (list, tuple)):
                raise ValueError("Packed Python list/tuple backing is unsupported; no row traversal")
            if type(value) in (int, float, bool, str, type(None)):
                return (type(value), repr(value))
            raise ValueError("Unsupported packed backing metadata")

        if type(dataset.datasets) is not list or not dataset.datasets:
            raise ValueError("Exact nonempty packed shard membership required")
        children = []
        cumulative = []
        total = 0
        for child in dataset.datasets:
            if type(child) is not PackedGraphDataset:
                raise ValueError("Only exact shared PackedGraphDataset children supported")
            if (any(getattr(child, name) is not None
                    for name in ("transform", "pre_transform", "pre_filter", "_indices"))):
                raise ValueError("Packed transforms and index selections are unsupported")
            children.append((id(child), id(child._data), id(child.slices),
                             tuple((key, backing(value))
                                   for key, value in child._data.to_dict().items()),
                             tuple((key, backing(value)) for key, value in child.slices.items())
                             if type(child.slices) is dict else backing(child.slices)))
            total += len(child)
            cumulative.append(total)
        if (type(dataset.cumulative_sizes) is not list
                or any(type(n) is not int for n in dataset.cumulative_sizes)
                or dataset.cumulative_sizes != cumulative):
            raise ValueError("Packed concat sizes changed")
        return (tuple(children), tuple(cumulative))

    def _check_loader(self):
        if self._preserve_global_rng:
            self._check_global_rng_contract(self.loader)
        if (self.loader.dataset is not self._dataset
                or self._dataset_state(self._dataset) != self._dataset_binding):
            raise ValueError("Authorized dataset membership changed")
        if (self.loader.sampler is not self._sampler
                or self.loader.batch_sampler.sampler is not self._sampler
                or self.loader.generator is not self._generator):
            raise ValueError("Original sampler alias/generator changed")
        retained = getattr(self.loader, "_iterator", None)
        if retained is not None and retained is not self._iterator:
            raise ValueError("Loader has an externally owned iterator")

    @contextmanager
    def stream(self, indices):
        """Replace the sampler in place; incomplete/error streams require close."""
        if self._closed or self._poisoned:
            raise RuntimeError("Loader reuse is closed or poisoned; close before discarding it")
        if self._active:
            raise RuntimeError("Overlapping loader iterations are forbidden")
        self._check_loader()
        rows = list(indices)
        size = len(self._dataset)
        if (any(type(i) is not int or not 0 <= i < size for i in rows)
                or len(set(rows)) != len(rows)):
            raise ValueError("Indices must be unique exact ints within the dataset")
        self._active = True
        completed = False
        iterator = None
        expected_batches = None
        try:
            self._sampler[:] = rows
            if self._preserve_global_rng and self.loader.num_workers > 0 and self._iterator is not None:
                import torch
                torch.empty((), dtype=torch.int64).random_(generator=None).item()
            iterator = iter(self.loader)
            if self._iterator is not None and self.loader.num_workers > 0 and iterator is not self._iterator:
                raise RuntimeError("Persistent loader replaced its owned iterator")
            self._iterator = iterator
            self._workers = tuple(getattr(iterator, "_workers", ()))
            if type(getattr(iterator, "_num_yielded", None)) is not int or iterator._num_yielded != 0:
                raise RuntimeError("Native iterator yielded-batch counter is unavailable or stale")
            expected_batches = len(self.loader)
            yield iterator
            self._check_loader()
            completed = True
        finally:
            self._active = False
            # All scheduled batches must be returned; no extra timed fetch is
            # needed just to observe StopIteration before the persistent reset.
            count = getattr(iterator, "_num_yielded", None)
            if not completed or type(count) is not int or count != expected_batches:
                self._poisoned = True

    def close(self):
        """Shut down and join only the iterator/workers acquired by this owner."""
        self._closed = True
        if self._cleanup_done:
            return
        if self._active:
            self._poisoned = True
        iterator = self._iterator
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        for worker in self._workers:
            worker.join(timeout=1.0)
        if any(worker.is_alive() for worker in self._workers):
            raise RuntimeError("Owned loader workers did not stop")
        if iterator is not None and getattr(self.loader, "_iterator", None) is iterator:
            self.loader._iterator = None
        self._iterator = None
        self._workers = ()
        self._cleanup_done = True

    def __enter__(self):
        if self._closed or self._poisoned:
            raise RuntimeError("Loader reuse is closed or poisoned")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
