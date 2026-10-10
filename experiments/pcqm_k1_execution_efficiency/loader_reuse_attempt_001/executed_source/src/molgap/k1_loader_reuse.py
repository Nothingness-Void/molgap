"""Opt-in reuse of the original K1 list-sampler loader; no permutation logic."""
from contextlib import contextmanager


class K1LoaderReuse:
    """Exclusively own a fresh loader with caller-authorized deterministic data.

    ``deterministic_dataset=True`` asserts that the plain list/tuple and its
    elements remain immutable and have no stochastic transforms or worker-RNG
    contract. Retained workers continue their RNG state, unlike fresh workers
    seeded by the original per-epoch recipe; stochastic datasets are unsupported.
    The loader generator is not reseeded. This is not fresh-worker RNG parity.

    Pass flattened exact-int indices from the existing epoch sampler, already
    sliced for its processed resume cursor and dropped tail. No order is generated
    here. Use ``with pool.stream(indices) as iterator`` and consume all batches.
    The native iterator is valid only inside that context. Never iterate the
    supplied loader directly while this owner is live.
    """

    def __init__(self, loader, *, deterministic_dataset: bool):
        import torch
        from torch.utils.data import BatchSampler, DataLoader

        if deterministic_dataset is not True:
            raise ValueError("Explicit deterministic immutable dataset authority required")
        if not isinstance(loader, DataLoader):
            raise ValueError("Expected an already constructed DataLoader")
        if type(loader.dataset) not in (list, tuple):
            raise ValueError("Only immutable caller-owned plain list/tuple datasets are supported")
        if type(loader.sampler) is not list:
            raise ValueError("Original literal list sampler required")
        if type(loader.batch_sampler) is not BatchSampler or loader.batch_sampler.sampler is not loader.sampler:
            raise ValueError("Batch sampler must alias the original sampler list")
        if not isinstance(loader.generator, torch.Generator) or loader.generator.device.type != "cpu":
            raise ValueError("Explicit CPU loader generator required")
        if loader.num_workers > 0 and loader.persistent_workers is not True:
            raise ValueError("Worker reuse requires persistent workers")
        if loader.worker_init_fn is not None:
            raise ValueError("Worker RNG initialization contracts are unsupported")
        if getattr(loader, "_iterator", None) is not None:
            raise ValueError("Loader already has an iterator; exclusive fresh ownership required")
        self.loader = loader
        self._dataset = loader.dataset
        self._members = tuple(loader.dataset)
        self._sampler = loader.sampler
        self._generator = loader.generator
        self._iterator = None
        self._workers = ()
        self._active = False
        self._poisoned = False
        self._closed = False
        self._cleanup_done = False

    def _check_loader(self):
        if (self.loader.dataset is not self._dataset
                or len(self._dataset) != len(self._members)
                or any(a is not b for a, b in zip(self._dataset, self._members))):
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
        if (any(type(i) is not int or not 0 <= i < len(self._dataset) for i in rows)
                or len(set(rows)) != len(rows)):
            raise ValueError("Indices must be unique exact ints within the dataset")
        self._active = True
        completed = False
        iterator = None
        expected_batches = None
        try:
            self._sampler[:] = rows
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
