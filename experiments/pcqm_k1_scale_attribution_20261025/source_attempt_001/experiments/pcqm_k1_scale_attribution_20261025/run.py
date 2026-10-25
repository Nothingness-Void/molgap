"""One bounded worker; the parent kills it at the user or diagnostic deadline."""
import multiprocessing
from pathlib import Path
import time
from molgap.k1_retained_tail_diagnostic import run

if __name__ == '__main__':
    from molgap.training_reproducibility import atomic_json
    here = Path(__file__).resolve().parent
    started = time.perf_counter()
    budget = min(1200, max(0, 1792914095 - time.time()))
    if budget <= 0:
        raise SystemExit('User deadline already elapsed')
    worker = multiprocessing.get_context('spawn').Process(target=run, args=(here, here / 'results'))
    worker.start()
    worker.join(budget)
    timeout = worker.is_alive()
    if timeout:
        worker.terminate()
        worker.join()
    atomic_json(here / 'execution_receipt.json', {'worker_exitcode': worker.exitcode, 'timeout': timeout,
        'parent_observed_wall_seconds': time.perf_counter() - started, 'status': 'complete' if worker.exitcode == 0 and not timeout else 'failed'})
    raise SystemExit(0 if worker.exitcode == 0 and not timeout else 1)
