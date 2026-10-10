"""Bounded synthetic CPU checks for main-process RNG, not worker RNG parity."""
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from molgap.k1_loader_reuse import K1LoaderReuse


@pytest.fixture(autouse=True)
def cpu_rng_only(monkeypatch):
    state = torch.get_rng_state()
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "_lazy_init",
                        lambda *a, **k: pytest.fail("No CUDA initialization allowed"))
    yield
    assert torch.cuda.is_initialized() == initialized
    torch.set_rng_state(state)


@pytest.fixture
def packed(tmp_path):
    from torch_geometric.data import Data, InMemoryDataset
    from molgap.packed_graph_dataset import PackedGraphDataset

    children = []
    for start, stop in [(0, 4), (4, 9)]:
        path = tmp_path / f"scalar_{start}.pt"
        torch.save(InMemoryDataset.collate([
            Data(x=torch.tensor([[i]]), source_idx=torch.tensor(i))
            for i in range(start, stop)]), path)
        children.append(PackedGraphDataset(path))
    return ConcatDataset(children)


def loader(dataset, *, rows=None, workers=0, retained=False, generator=None):
    if type(dataset) is ConcatDataset or (type(dataset) is Subset
                                         and type(dataset.dataset) is ConcatDataset):
        from torch_geometric.loader import DataLoader as factory
    else:
        factory = DataLoader
    kwargs = {"multiprocessing_context": "spawn", "timeout": 30} if workers else {}
    return factory(dataset, batch_size=3, sampler=rows, shuffle=False,
                   num_workers=workers, persistent_workers=retained and workers > 0,
                   generator=generator, pin_memory=False, **kwargs)


def values(iterator, is_packed):
    return [(batch.source_idx if is_packed else batch).tolist() for batch in iterator]


def compare_fresh_and_retained(dataset, *, workers, orders):
    is_packed = type(dataset) is ConcatDataset
    initial = torch.Generator().manual_seed(713).get_state()
    torch.set_rng_state(initial)
    expected = []
    for rows in orders:
        fresh = loader(Subset(dataset, rows), workers=workers)
        iterator = iter(fresh)
        try:
            entered_state = torch.get_rng_state().clone()
            batches = values(iterator, is_packed)
            ended_state = torch.get_rng_state().clone()
            draw = torch.rand(4)
            expected.append((batches, entered_state, ended_state, draw, torch.get_rng_state().clone()))
        finally:
            if workers:
                # Empty fresh loaders can finish before Windows spawn exits;
                # join only these fixture-owned handles before asserting cleanup.
                iterator._shutdown_workers()
                for worker in iterator._workers:
                    worker.join(timeout=30)
                assert all(not worker.is_alive() for worker in iterator._workers)

    torch.set_rng_state(initial)
    retained = loader(dataset, rows=[], workers=workers, retained=True)
    pool = K1LoaderReuse(retained, deterministic_dataset=True, preserve_global_rng=True)
    owned = ()
    previous = None
    try:
        for rows, (batches, entered_state, ended_state, draw, after_draw) in zip(orders, expected):
            before_invalid = torch.get_rng_state().clone()
            with pytest.raises(ValueError, match="unique exact ints"):
                with pool.stream([len(dataset)]):
                    pytest.fail("Invalid indices must not draw a base seed")
            assert torch.equal(torch.get_rng_state(), before_invalid)
            with pool.stream(rows) as iterator:
                assert torch.equal(torch.get_rng_state(), entered_state)
                assert values(iterator, is_packed) == batches
                assert torch.equal(torch.get_rng_state(), ended_state)
                if workers:
                    if previous is not None:
                        assert iterator is previous
                        assert pool._workers == owned
                    owned = pool._workers
                elif previous is not None:
                    assert iterator is not previous
                previous = iterator
            assert torch.equal(torch.rand(4), draw)
            assert torch.equal(torch.get_rng_state(), after_draw)
            assert not pool._poisoned
    finally:
        state = torch.get_rng_state().clone()
        pool.close()
        pool.close()
        assert torch.equal(torch.get_rng_state(), state)
    assert all(not worker.is_alive() for worker in owned)
    assert retained._iterator is None


@pytest.mark.parametrize("kind", ["list", "tuple", "packed"])
def test_worker_zero_order_resume_terminal_and_global_rng(kind, packed):
    dataset = packed if kind == "packed" else (list if kind == "list" else tuple)(range(9))
    compare_fresh_and_retained(dataset, workers=0,
                               orders=[[8, 0, 7, 1, 6, 2, 5, 3, 4], [6, 2, 5, 3, 4], []])


def test_real_two_worker_spawn_packed_order_resume_and_global_rng(packed):
    compare_fresh_and_retained(packed, workers=2,
                               orders=[[8, 0, 7, 1, 6, 2, 5, 3, 4], [2, 4, 0, 8, 1, 7], []])


def test_existing_screen_epoch_order_and_resume_worker_zero(monkeypatch):
    from molgap import k1_screen_training as owner

    monkeypatch.setattr(owner, "TRAIN_ROWS", 9)
    monkeypatch.setattr(owner, "ROWS_PER_EPOCH", 9)
    compare_fresh_and_retained(list(range(9)), workers=0,
                               orders=[owner.epoch_order(0), owner.epoch_order(1)[3:]])


@pytest.mark.parametrize("flag", [0, 1, None, "true"])
def test_flag_requires_exact_bool(flag):
    with pytest.raises(ValueError, match="exact bool"):
        K1LoaderReuse(loader([0], rows=[0]), deterministic_dataset=True, preserve_global_rng=flag)


def test_none_generator_still_rejected_by_default():
    with pytest.raises(ValueError, match="Explicit CPU loader generator"):
        K1LoaderReuse(loader([0], rows=[0]), deterministic_dataset=True)


def test_optin_rejects_explicit_generator_without_rng_draw():
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="generator=None"):
        K1LoaderReuse(loader([0], rows=[0], generator=torch.Generator()),
                      deterministic_dataset=True, preserve_global_rng=True)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("rows", [[True], [1.0], [-1], [3], [0, 0]])
def test_invalid_indices_no_rng_draw_or_sampler_mutation(rows):
    native = loader([0, 1, 2], rows=[1])
    with K1LoaderReuse(native, deterministic_dataset=True, preserve_global_rng=True) as pool:
        for attempted in range(2):
            state = torch.get_rng_state().clone()
            previous = list(native.sampler)
            with pytest.raises(ValueError, match="unique exact ints"):
                with pool.stream(rows):
                    pytest.fail("Invalid stream accepted")
            assert torch.equal(torch.get_rng_state(), state)
            assert native.sampler == previous
            assert not pool._poisoned
            if not attempted:
                with pool.stream([2, 0]) as iterator:
                    assert values(iterator, False) == [[2, 0]]


@pytest.mark.parametrize("device", ["meta", "cuda:0"])
def test_non_cpu_default_context_rejected_without_allocation(device):
    native = loader([0], rows=[0])
    state = torch.get_rng_state().clone()
    with torch.device(device):
        with pytest.raises(ValueError, match="CPU default device"):
            K1LoaderReuse(native, deterministic_dataset=True, preserve_global_rng=True)
    assert torch.equal(torch.get_rng_state(), state)


def test_default_device_change_before_stream_rejected(monkeypatch):
    native = loader([0], rows=[0])
    with K1LoaderReuse(native, deterministic_dataset=True, preserve_global_rng=True) as pool:
        state = torch.get_rng_state().clone()
        monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
        with pytest.raises(ValueError, match="CPU default device"):
            with pool.stream([0]):
                pytest.fail("Non CPU stream accepted")
        assert torch.equal(torch.get_rng_state(), state)
        assert pool._iterator is None


def test_legacy_default_device_rejected_before_default_device_query(monkeypatch):
    native = loader([0], rows=[0])
    state = torch.get_rng_state().clone()
    monkeypatch.setattr(torch._C, "_get_default_device", lambda: "cuda")
    monkeypatch.setattr(torch, "get_default_device",
                        lambda: pytest.fail("Legacy accelerator default must be rejected before allocation"))
    with pytest.raises(ValueError, match="CPU default device"):
        K1LoaderReuse(native, deterministic_dataset=True, preserve_global_rng=True)
    assert torch.equal(torch.get_rng_state(), state)


def test_explicit_generator_default_does_not_gain_global_rng_restrictions(monkeypatch):
    native = loader([0, 1], rows=[0], generator=torch.Generator().manual_seed(42))
    monkeypatch.setattr(torch, "__version__", "unsupported-version")
    monkeypatch.setattr(torch._C, "_get_default_device", lambda: "cuda")
    state = torch.get_rng_state().clone()
    with K1LoaderReuse(native, deterministic_dataset=True) as pool:
        with pool.stream([1, 0]) as iterator:
            assert values(iterator, False) == [[1, 0]]
    assert torch.equal(torch.get_rng_state(), state)


def test_unsupported_version_rejected(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.8.0")
    with pytest.raises(ValueError, match="2.7.1"):
        K1LoaderReuse(loader([0], rows=[0]), deterministic_dataset=True, preserve_global_rng=True)


def test_custom_iterator_semantics_rejected():
    class Custom(DataLoader):
        def __iter__(self):
            return super().__iter__()

    native = Custom([0], sampler=[0])
    with pytest.raises(ValueError, match="native Map iterator"):
        K1LoaderReuse(native, deterministic_dataset=True, preserve_global_rng=True)


def test_instance_iterator_override_rejected_without_rng_draw(monkeypatch):
    native = loader([0], rows=[0])
    state = torch.get_rng_state().clone()
    monkeypatch.setattr(native, "_get_iterator", lambda: pytest.fail("Override must not execute"))
    with pytest.raises(ValueError, match="native Map iterator"):
        K1LoaderReuse(native, deterministic_dataset=True, preserve_global_rng=True)
    assert torch.equal(torch.get_rng_state(), state)


def test_unknown_factory_mode_rejected_without_rng_draw():
    class Passthrough(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    native = loader([0], rows=[0])
    state = torch.get_rng_state().clone()
    with Passthrough():
        with pytest.raises(ValueError, match="CPU default device"):
            K1LoaderReuse(native, deterministic_dataset=True, preserve_global_rng=True)
    assert torch.equal(torch.get_rng_state(), state)


def test_deterministic_supported_map_authority_still_required():
    for dataset, authority in [([0], False), (range(1), True)]:
        with pytest.raises(ValueError):
            K1LoaderReuse(loader(dataset, rows=[0]), deterministic_dataset=authority,
                          preserve_global_rng=True)
