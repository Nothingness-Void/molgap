"""Measure caller-owned CPU loader fetches; no data discovery or model execution."""
from __future__ import annotations

import math
import statistics
import time


def profile_loader_fetches(factories, *, repeats=4, wall_seconds=180):
    """Balanced cold-loader windows; no GPU consumer, pinning or training claim.

    Factories are trusted caller code. Membership and full batch-byte acceptance
    stay with the caller. Workers are only the ones spawned by these iterators.
    Worker CPU is observed before explicit cleanup; unavailable cost stays None.
    """
    if type(repeats) is not int or repeats < 2 or repeats % 2 or repeats > 8:
        raise ValueError("Use2--8 balanced windows")
    if not isinstance(factories, dict) or len(factories) != 2 or not all(callable(f) for f in factories.values()):
        raise ValueError("Supply two explicit loader factories")
    if type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 300:
        raise ValueError("Invalid CPU diagnostic budget")
    started = time.perf_counter()
    observed, batches_by_case = [], {}
    worker_total = 0.0
    all_worker_costs_known = True
    for window in range(repeats):
        pair = {}
        names = list(factories) if window % 2 == 0 else list(reversed(factories))
        for name in names:
            if time.perf_counter() - started > wall_seconds:
                raise TimeoutError("CPU loader diagnostic budget exhausted")
            tick = time.perf_counter()
            loader, iterator = None, None
            samples, batches = [], []
            try:
                loader = factories[name]()
                iterator = iter(loader)
                startup = time.perf_counter() - tick
                while True:
                    if time.perf_counter() - started > wall_seconds:
                        raise TimeoutError("CPU loader diagnostic budget exhausted")
                    tick = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    samples.append(time.perf_counter() - tick)
                    batches.append(batch)
                if len(samples) < 2:
                    raise ValueError("Need at least two complete batches")
                workers = list(getattr(iterator, "_workers", ()))
                child_cpu = 0.0 if not workers else None
                if workers:
                    try:
                        import psutil
                    except ImportError:
                        pass
                    else:
                        try:
                            times = [psutil.Process(w.pid).cpu_times() for w in workers]
                            child_cpu = sum(c.user + c.system for c in times)
                        except psutil.Error:
                            pass
                if child_cpu is None:
                    all_worker_costs_known = False
                else:
                    worker_total += child_cpu
                pair[name] = dict(startup_seconds=startup, first_fetch_seconds=samples[0],
                                  steady_median_seconds=statistics.median(samples[1:]),
                                  all_fetch_seconds=sum(samples), fetch_samples_seconds=samples,
                                  batch_count=len(batches), worker_process_cpu_seconds=child_cpu,
                                  worker_pids=[w.pid for w in workers])
                batches_by_case.setdefault(name, []).append(batches)
            finally:
                if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                    tick = time.perf_counter()
                    iterator._shutdown_workers()
                    if name in pair:
                        for worker in workers:
                            worker.join(timeout=1.0)
                        pair[name]["shutdown_seconds"] = time.perf_counter() - tick
                        pair[name]["workers_stopped"] = not any(w.is_alive() for w in workers)
                elif name in pair:
                    pair[name].update(shutdown_seconds=0.0, workers_stopped=True)
        observed.append(pair)
    return dict(windows=observed, worker_process_cpu_seconds=worker_total if all_worker_costs_known else None,
                medians={name: {field: statistics.median(w[name][field] for w in observed)
                               for field in ("startup_seconds", "first_fetch_seconds", "steady_median_seconds", "all_fetch_seconds")}
                         for name in factories}), batches_by_case


def profile_epoch_loader_reuse(factory, index_orders, *, wall_seconds=250):
    """Cold versus retained original loader; caller owns exact epoch/resume orders.

    No model or permutation generation. Whole iteration startup includes sampler
    validation. Child CPU is unmeasured, not inferred from parent process time.
    """
    from .k1_loader_reuse import K1LoaderReuse

    if (not callable(factory) or type(index_orders) is not list
            or len(index_orders) != 4 or any(type(rows) is not list for rows in index_orders)):
        raise ValueError("Require factory and four explicit index orders")
    if type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 300:
        raise ValueError("Invalid CPU diagnostic budget")
    started = time.perf_counter()
    retained = None
    pools, windows, batches_by_case = [], [], {}
    try:
        for window, rows in enumerate(index_orders):
            pair = {}
            names = ["cold", "retained"] if window % 2 == 0 else ["retained", "cold"]
            for name in names:
                if time.perf_counter() - started > wall_seconds:
                    raise TimeoutError("CPU loader diagnostic budget exhausted")
                tick = time.perf_counter()
                if name == "retained" and retained is not None:
                    pool = retained
                else:
                    pool = K1LoaderReuse(factory(), deterministic_dataset=True)
                    pools.append(pool)
                    if name == "retained":
                        retained = pool
                samples, values = [], []
                try:
                    with pool.stream(rows) as iterator:
                        startup = time.perf_counter() - tick
                        while True:
                            if time.perf_counter() - started > wall_seconds:
                                raise TimeoutError("CPU loader diagnostic budget exhausted")
                            tick = time.perf_counter()
                            try:
                                batch = next(iterator)
                            except StopIteration:
                                break
                            samples.append(time.perf_counter() - tick)
                            values.append(batch)
                        pids = [w.pid for w in pool._workers]
                    if len(samples) < 2:
                        raise ValueError("Require at least two complete batches")
                    pair[name] = dict(startup_seconds=startup, first_fetch_seconds=samples[0],
                        steady_median_seconds=statistics.median(samples[1:]),
                        all_fetch_seconds=sum(samples), batch_count=len(values), worker_pids=pids,
                        fetch_samples_seconds=samples)
                    batches_by_case.setdefault(name, []).append(values)
                finally:
                    if name == "cold":
                        pool.close()
            windows.append(pair)
    finally:
        # A failing close must not prevent another owned pool from being closed.
        failures = []
        for pool in pools:
            try:
                pool.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError("Owned loader cleanup failed") from failures[0]
    for pair in windows:
        for case in pair.values():
            case["workers_stopped"] = True  # close() verified owned handles above.
    return dict(windows=windows, worker_process_cpu_seconds=None,
        medians={name: {field: statistics.median(w[name][field] for w in windows)
            for field in ("startup_seconds", "first_fetch_seconds", "steady_median_seconds", "all_fetch_seconds")}
            for name in ("cold", "retained")}), batches_by_case
