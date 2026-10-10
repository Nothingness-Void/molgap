"""CPU scalar datasets and mocked workers; no graphs, models or accelerators."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from molgap.edge_state_training_core import EpochPermutationBatchSampler
from molgap.k1_loader_reuse import K1LoaderReuse


def loader_for(dataset=None, *, workers=0, drop_last=False):
    return DataLoader(list(range(11)) if dataset is None else dataset,
                      batch_size=3, sampler=[0, 1, 2],
                      generator=torch.Generator().manual_seed(9000),
                      num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=False, drop_last=drop_last)


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "_lazy_init",
                        lambda *a, **k: pytest.fail("CPU reuse must not initialize CUDA"))
    yield
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("dataset_type", [list, tuple])
def test_native_iterator_order_resume_and_original_sampler_drop_tail(dataset_type, monkeypatch):
    loader = loader_for(dataset_type(range(11)))
    original = dict(sampler=loader.sampler, batch_sampler=loader.batch_sampler,
                    collate_fn=loader.collate_fn, generator=loader.generator,
                    batch_size=loader.batch_size, pin_memory=loader.pin_memory,
                    persistent_workers=loader.persistent_workers)
    orders = []
    for epoch, cursor in [(0, 0), (1, 1), (2, 3)]:
        sampler = EpochPermutationBatchSampler(11, 3, seed=42, epoch=epoch, start_batch=cursor)
        orders.append([i for batch in sampler for i in batch])
    monkeypatch.setattr(torch, "randperm", lambda *a, **k: pytest.fail("No new permutation allowed"))
    rng = torch.get_rng_state().clone()
    with K1LoaderReuse(loader, deterministic_dataset=True) as pool:
        for order in orders:
            with pool.stream(order) as iterator:
                assert iterator is pool._iterator
                assert iterator._num_yielded == 0
                batches = list(iterator)
            assert [i for b in batches for i in b.tolist()] == order
            assert all(len(b) == 3 for b in batches)
            assert loader.sampler == order
            assert loader.batch_sampler.sampler is original["sampler"]
        for name, value in original.items():
            assert getattr(loader, name) is value
    assert torch.equal(torch.get_rng_state(), rng)
    assert pool._iterator is None


def test_pyg_collater_and_pinning_options_untouched():
    from torch_geometric.loader import DataLoader as PyGDataLoader

    loader = PyGDataLoader([10, 20, 30, 40], batch_size=2, sampler=[0, 1],
                          generator=torch.Generator().manual_seed(9000), pin_memory=True)
    collater = loader.collate_fn
    with K1LoaderReuse(loader, deterministic_dataset=True) as pool:
        with pool.stream([3, 2, 1, 0]) as iterator:
            assert [b.tolist() for b in iterator] == [[40, 30], [20, 10]]
    assert loader.collate_fn is collater
    assert loader.pin_memory is True


@pytest.mark.parametrize("epoch,cursor", [(0, 0), (1, 0), (1, 1), (3, 1), (4, 2)])
def test_original_subset_sender_batches_match_list_sampler_reuse(epoch, cursor, monkeypatch):
    from torch.utils.data import Subset
    from torch_geometric.data import Data
    from molgap.pcqm_500k_v4_evidence import loader as original_loader
    from molgap.k1_screen_training import _batch_sha256

    monkeypatch.setenv("MOLGAP_V4_LOADER_WORKERS", "0")
    # Synthetic one-node graphs only; no molecular cache, labels or checkpoints.
    graphs = [Data(x=torch.zeros((1, 9), dtype=torch.long),
                   edge_index=torch.zeros((2, 1), dtype=torch.long),
                   edge_attr=torch.zeros((1, 3), dtype=torch.long),
                   random_walk_pe=torch.zeros((1, 16)),
                   y=torch.tensor([float(i)]), source_idx=torch.tensor(i)) for i in range(260)]
    sampler = EpochPermutationBatchSampler(260, 128, seed=42, epoch=epoch, start_batch=cursor)
    indices = sampler.indices[cursor * 128:].tolist()
    reference = [_batch_sha256(b) for b in original_loader(Subset(graphs, indices))]
    with K1LoaderReuse(original_loader(graphs, epoch=0), deterministic_dataset=True) as pool:
        with pool.stream(indices) as batches:
            assert [_batch_sha256(b) for b in batches] == reference


@pytest.mark.parametrize("drop_last,expected", [(False, [[4, 3, 2], [1]]), (True, [[4, 3, 2]])])
def test_original_batch_sampler_owns_partial_tail(drop_last, expected):
    loader = loader_for(drop_last=drop_last)
    with K1LoaderReuse(loader, deterministic_dataset=True) as pool:
        with pool.stream([4, 3, 2, 1]) as iterator:
            assert [b.tolist() for b in iterator] == expected
        # Fetching all batches without a trailing StopIteration also completes.
        with pool.stream([2, 1, 0]) as iterator:
            assert next(iterator).tolist() == [2, 1, 0]
        with pool.stream([]) as iterator:
            assert list(iterator) == []


class IntSubclass(int):
    pass


@pytest.mark.parametrize("indices", [[False], [True], [0.0], [np.int64(0)], [IntSubclass(0)],
                                     [-1], [11], [0, 0], ["0"], [None]])
def test_invalid_indices_rejected_before_sampler_mutation(indices):
    loader = loader_for()
    original = loader.sampler.copy()
    with K1LoaderReuse(loader, deterministic_dataset=True) as pool:
        with pytest.raises(ValueError, match="unique exact ints"):
            with pool.stream(indices):
                pytest.fail("Invalid indices reached iteration")
        assert loader.sampler == original
        assert pool._iterator is None
        with pool.stream([0, 1, 2]) as iterator:
            assert len(list(iterator)) == 1


@pytest.mark.parametrize("authority", [False, 0, 1, None, "true", np.bool_(True)])
def test_explicit_dataset_authority_is_fail_closed(authority):
    loader = loader_for()
    with pytest.raises(ValueError, match="authority"):
        K1LoaderReuse(loader, deterministic_dataset=authority)
    assert loader._iterator is None


@pytest.mark.parametrize("defect", ["dataset", "sampler", "alias", "generator", "persistent", "worker_init", "existing_iterator"])
def test_unsupported_loader_contracts_rejected_without_iteration(defect):
    loader = loader_for(workers=2)
    # Bypass DataLoader's public assignment guard only to make invalid fixtures.
    if defect == "dataset":
        object.__setattr__(loader, "dataset", range(11))
    elif defect == "sampler":
        object.__setattr__(loader, "sampler", tuple(loader.sampler))
    elif defect == "alias":
        loader.batch_sampler.sampler = loader.sampler.copy()
    elif defect == "generator":
        loader.generator = None
    elif defect == "persistent":
        object.__setattr__(loader, "persistent_workers", False)
    elif defect == "worker_init":
        loader.worker_init_fn = lambda worker_id: None
    else:
        loader._iterator = object()
    with pytest.raises(ValueError):
        K1LoaderReuse(loader, deterministic_dataset=True)


def test_overlapping_stream_refused_before_mutation():
    loader = loader_for()
    with K1LoaderReuse(loader, deterministic_dataset=True) as pool:
        with pool.stream([0, 1, 2]) as iterator:
            with pytest.raises(RuntimeError, match="Overlapping"):
                with pool.stream([3, 4, 5]):
                    pytest.fail("Overlapping iteration allowed")
            assert loader.sampler == [0, 1, 2]
            assert next(iterator).tolist() == [0, 1, 2]
        with pool.stream([3, 4, 5]) as iterator:
            assert len(list(iterator)) == 1


@pytest.mark.parametrize("mode", ["early", "body_error", "caught_fetch_error", "startup_error"])
def test_incomplete_or_error_stream_poison_requires_close(monkeypatch, mode):
    loader = loader_for()
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    if mode == "startup_error":
        monkeypatch.setattr(loader, "_get_iterator", lambda: (_ for _ in ()).throw(RuntimeError("startup")))
        with pytest.raises(RuntimeError, match="startup"):
            with pool.stream([0, 1, 2]):
                pass
    elif mode == "body_error":
        with pytest.raises(RuntimeError, match="body"):
            with pool.stream([0, 1, 2]) as iterator:
                list(iterator)
                raise RuntimeError("body")
    else:
        with pool.stream([0, 1, 2, 3, 4, 5]) as iterator:
            if mode == "caught_fetch_error":
                monkeypatch.setattr(iterator, "_next_data", lambda: (_ for _ in ()).throw(RuntimeError("fetch")))
                with pytest.raises(RuntimeError, match="fetch"):
                    next(iterator)
            else:
                next(iterator)
    before = loader.sampler.copy()
    with pytest.raises(RuntimeError, match="poisoned"):
        with pool.stream([6, 7, 8]):
            pass
    assert loader.sampler == before
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        with pool.stream([0, 1, 2]):
            pass


def test_dataset_or_alias_mutation_refused_before_sampler_change():
    loader = loader_for()
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    before = loader.sampler.copy()
    loader.dataset.reverse()
    with pytest.raises(ValueError, match="dataset membership changed"):
        with pool.stream([3, 4, 5]):
            pass
    assert loader.sampler == before
    loader.dataset.reverse()
    loader.batch_sampler.sampler = []
    with pytest.raises(ValueError, match="alias/generator changed"):
        with pool.stream([3, 4, 5]):
            pass
    assert loader.sampler == before
    pool.close()


class Worker:
    def __init__(self, events, *, stubborn=False):
        self.events, self.stubborn = events, stubborn
        self.alive = True

    def join(self, timeout):
        assert timeout == 1.0
        self.events.append("join")
        self.alive = self.stubborn

    def is_alive(self):
        self.events.append("alive")
        return self.alive


class NativeIterator:
    def __init__(self, loader, events):
        self.events = events
        self._workers = [Worker(events), Worker(events)]
        self._reset(loader)

    def _reset(self, loader):
        self.events.append("reset")
        self._num_yielded = 0
        self.rows = iter(list(loader.batch_sampler))

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.rows)
        self._num_yielded += 1
        return value

    def _shutdown_workers(self):
        self.events.append("shutdown")


def test_persistent_native_iterator_reuse_and_only_owned_cleanup(monkeypatch):
    loader = loader_for(workers=2)
    events = []
    owned = NativeIterator(loader, events)
    foreign_events = []
    foreign = NativeIterator(loader, foreign_events)
    monkeypatch.setattr(loader, "_get_iterator", lambda: owned)
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    for order in ([0, 1, 2], [5, 4, 3]):
        with pool.stream(order) as iterator:
            assert iterator is owned is loader._iterator
            assert list(iterator) == [order]
    loader._iterator = foreign
    with pytest.raises(ValueError, match="externally owned"):
        with pool.stream([6, 7, 8]):
            pass
    pool.close()
    assert loader._iterator is foreign
    assert events.count("shutdown") == 1
    assert events[-5:] == ["shutdown", "join", "join", "alive", "alive"]
    assert foreign_events == ["reset"]
    pool.close()
    assert events.count("shutdown") == 1


def test_early_exit_closes_owned_persistent_workers_and_detaches_iterator(monkeypatch):
    loader = loader_for(workers=2)
    events = []
    owned = NativeIterator(loader, events)
    monkeypatch.setattr(loader, "_get_iterator", lambda: owned)
    with K1LoaderReuse(loader, deterministic_dataset=True) as pool:
        with pool.stream([0, 1, 2, 3, 4, 5]) as iterator:
            next(iterator)
        with pytest.raises(RuntimeError, match="poisoned"):
            with pool.stream([6, 7, 8]):
                pass
    assert loader._iterator is None
    assert events.count("shutdown") == 1
    assert not any(w.alive for w in owned._workers)


def test_close_failure_retains_owned_handles_for_cleanup_retry(monkeypatch):
    loader = loader_for(workers=2)
    events = []
    owned = NativeIterator(loader, events)
    owned._workers[0].stubborn = True
    monkeypatch.setattr(loader, "_get_iterator", lambda: owned)
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    with pool.stream([0, 1, 2]) as iterator:
        list(iterator)
    with pytest.raises(RuntimeError, match="did not stop"):
        pool.close()
    assert loader._iterator is owned
    owned._workers[0].stubborn = False
    pool.close()
    assert loader._iterator is None


def test_close_before_first_stream_starts_no_iterator(monkeypatch):
    loader = loader_for(workers=2)
    monkeypatch.setattr(loader, "_get_iterator", lambda: pytest.fail("Unexpected worker startup"))
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        with pool.stream([0, 1, 2]):
            pass


@pytest.mark.parametrize("counter", [None, True, 1])
def test_missing_or_stale_native_counter_poison_and_owned_cleanup(monkeypatch, counter):
    loader = loader_for(workers=2)
    events = []
    owned = NativeIterator(loader, events)
    owned._num_yielded = counter
    monkeypatch.setattr(loader, "_get_iterator", lambda: owned)
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    with pytest.raises(RuntimeError, match="counter"):
        with pool.stream([0, 1, 2]):
            pass
    with pytest.raises(RuntimeError, match="poisoned"):
        with pool.stream([3, 4, 5]):
            pass
    pool.close()
    assert events.count("shutdown") == 1
    assert loader._iterator is None


def test_close_during_active_stream_prevents_reuse_and_cleans_once(monkeypatch):
    loader = loader_for(workers=2)
    events = []
    owned = NativeIterator(loader, events)
    monkeypatch.setattr(loader, "_get_iterator", lambda: owned)
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    with pool.stream([0, 1, 2, 3, 4, 5]) as iterator:
        next(iterator)
        pool.close()
    assert loader._iterator is None
    with pytest.raises(RuntimeError, match="closed"):
        with pool.stream([6, 7, 8]):
            pass
    pool.close()
    assert events.count("shutdown") == 1


def test_dataset_mutation_at_context_exit_poisons_even_after_all_batches():
    loader = loader_for()
    pool = K1LoaderReuse(loader, deterministic_dataset=True)
    with pytest.raises(ValueError, match="dataset membership changed"):
        with pool.stream([0, 1, 2]) as iterator:
            list(iterator)
            loader.dataset.reverse()
    with pytest.raises(RuntimeError, match="poisoned"):
        with pool.stream([3, 4, 5]):
            pass
    pool.close()
