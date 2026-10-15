"""One bounded numerical worker; preserve partial output on failure."""
import multiprocessing
from pathlib import Path
import time

from molgap.k1_session_state_diagnostic import run
from molgap.training_reproducibility import atomic_json


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    started = time.perf_counter()
    worker = multiprocessing.get_context("spawn").Process(target=run, args=(here, here / "results"))
    worker.start()
    worker.join(max(0, 1200 - (time.perf_counter() - started)))
    timeout = worker.is_alive()
    if timeout:
        worker.terminate()
        worker.join()
    atomic_json(here / "execution_receipt.json", {"worker_exitcode": worker.exitcode, "timeout": timeout,
        "parent_observed_wall_seconds": time.perf_counter() - started, "status": "complete" if worker.exitcode == 0 and not timeout else "failed"})
    raise SystemExit(0 if worker.exitcode == 0 and not timeout else 1)
