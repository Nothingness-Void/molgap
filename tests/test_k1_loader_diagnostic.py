"""CPU synthetic loader/control-flow checks, not molecular or native timing."""
import pytest
from types import SimpleNamespace

from molgap.k1_loader_diagnostic import profile_loader_fetches, profile_epoch_loader_reuse


class Iterator:
    def __init__(self, events, name, *, fail=False):
        self.events, self.name, self.fail = events, name, fail
        self.index = 0
        self._workers = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail:
            raise RuntimeError("synthetic fetch failure")
        if self.index == 3:
            raise StopIteration
        self.index += 1
        return self.index

    def _shutdown_workers(self):
        self.events.append(("shutdown", self.name))


def test_balanced_order_complete_fetches_and_owned_cleanup():
    events = []

    def factory(name):
        def create():
            events.append(("create", name))
            return Iterator(events, name)
        return create

    report, batches = profile_loader_fetches({"a": factory("a"), "b": factory("b")})
    assert [n for event, n in events if event == "create"] == ["a", "b", "b", "a", "a", "b", "b", "a"]
    assert len([e for e in events if e[0] == "shutdown"]) == 8
    assert batches == {"a": [[1, 2, 3]] * 4, "b": [[1, 2, 3]] * 4}
    assert report["worker_process_cpu_seconds"] == 0.0
    for pair in report["windows"]:
        for case in pair.values():
            assert case["batch_count"] == 3 and case["workers_stopped"]
            assert len(case["fetch_samples_seconds"]) == 3


def test_fetch_failure_cleans_up_only_owned_iterator():
    events = []
    with pytest.raises(RuntimeError, match="synthetic fetch failure"):
        profile_loader_fetches({"a": lambda: Iterator(events, "a", fail=True),
                               "b": lambda: Iterator(events, "b")})
    assert events == [("shutdown", "a")]


def test_shutdown_joins_owned_handles_before_testing_liveness(monkeypatch):
    import psutil
    events = []
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(
        cpu_times=lambda: SimpleNamespace(user=0.0, system=0.0)))

    class Worker:
        pid = 123
        joined = False

        def join(self, timeout):
            assert timeout == 1.0
            self.joined = True
            events.append(("join", self.pid))

        def is_alive(self):
            return not self.joined

    def create():
        iterator = Iterator(events, "fixture")
        iterator._workers = [Worker()]
        return iterator

    report, _ = profile_loader_fetches({"a": create, "b": create}, repeats=2)
    assert len([event for event in events if event[0] == "join"]) == 4
    assert all(c["workers_stopped"] for pair in report["windows"] for c in pair.values())


@pytest.mark.parametrize("kwargs", [dict(repeats=1), dict(repeats=3), dict(repeats=True),
                                   dict(repeats=10), dict(wall_seconds=0), dict(wall_seconds=float("nan")),
                                   dict(wall_seconds=True), dict(wall_seconds=301)])
def test_invalid_bounds_do_not_launch_loader(kwargs):
    def never():
        pytest.fail("Invalid arguments must not construct a loader")
    with pytest.raises(ValueError):
        profile_loader_fetches({"a": never, "b": never}, **kwargs)


@pytest.mark.parametrize("factories", [None, {}, {"a": lambda: []}, {"a": None, "b": lambda: []}])
def test_invalid_factories_rejected(factories):
    with pytest.raises(ValueError):
        profile_loader_fetches(factories)


def test_reuse_profile_balances_windows_preserves_orders_and_closes_all(monkeypatch):
    from contextlib import contextmanager
    from molgap import k1_loader_reuse
    pools = []

    class Pool:
        def __init__(self, loader, *, deterministic_dataset):
            assert deterministic_dataset is True
            self.closed = False
            self._workers = [SimpleNamespace(pid=len(pools))]
            self.orders = []
            pools.append(self)

        @contextmanager
        def stream(self, rows):
            assert not self.closed
            self.orders.append(rows)
            yield iter(rows)

        def close(self):
            self.closed = True

    monkeypatch.setattr(k1_loader_reuse, "K1LoaderReuse", Pool)
    orders = [[e, e + 1, e + 2] for e in range(4)]
    report, batches = profile_epoch_loader_reuse(lambda: object(), orders)
    assert len(pools) == 5 and all(p.closed for p in pools)
    assert [p.orders for p in pools].count(orders) == 1
    assert batches == {"cold": orders, "retained": orders}
    assert [list(w) for w in report["windows"]] == [["cold", "retained"],
        ["retained", "cold"], ["cold", "retained"], ["retained", "cold"]]
    assert len({tuple(w["retained"]["worker_pids"]) for w in report["windows"]}) == 1
    assert report["worker_process_cpu_seconds"] is None
    assert all(c["workers_stopped"] for w in report["windows"] for c in w.values())


def test_reuse_profile_cleans_up_after_fetch_error(monkeypatch):
    from contextlib import contextmanager
    from molgap import k1_loader_reuse
    closed = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        @contextmanager
        def stream(self, rows):
            def broken():
                yield rows[0]
                raise RuntimeError("fetch failed")
            yield broken()

        def close(self):
            closed.append(self)

    monkeypatch.setattr(k1_loader_reuse, "K1LoaderReuse", Pool)
    with pytest.raises(RuntimeError, match="fetch failed"):
        profile_epoch_loader_reuse(lambda: object(), [[1, 2]] * 4)
    assert closed


@pytest.mark.parametrize("orders,budget", [(None, 100), ([[1, 2]], 100),
    ([[1, 2]] * 4, True), ([[1, 2]] * 4, float("nan")), ([[1, 2]] * 4, 301)])
def test_reuse_profile_rejects_bounds_before_factory(orders, budget):
    with pytest.raises(ValueError):
        profile_epoch_loader_reuse(lambda: pytest.fail("invalid plan must not launch"), orders, wall_seconds=budget)
