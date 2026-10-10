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
                        pair[name]["shutdown_seconds"] = time.perf_counter() - tick
                        pair[name]["workers_stopped"] = not any(w.is_alive() for w in workers)
                elif name in pair:
                    pair[name].update(shutdown_seconds=0.0, workers_stopped=True)
        observed.append(pair)
    return dict(windows=observed, worker_process_cpu_seconds=worker_total if all_worker_costs_known else None,
                medians={name: {field: statistics.median(w[name][field] for w in observed)
                               for field in ("startup_seconds", "first_fetch_seconds", "steady_median_seconds", "all_fetch_seconds")}
                         for name in factories}), batches_by_case
