"""Isolated T4 workers for the frozen K1 edge-memory experiment."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from molgap.k1_edge_memory import MODES

ROOT = Path("/kaggle/working/pcqm_k1_edge_memory")


def child(context, mode):
    import torch
    from molgap.pcqm_k1_variants_runner import train_arm
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Each worker requires exactly one T4")
    resume = os.environ.get("MOLGAP_K1_RESUME_ROOT")
    train_arm(mode, ROOT / mode, source_commit=context["source_commit"],
              source_archive_sha256=context["source_archive_sha256"],
              resume_from=Path(resume) / mode if resume else None)


def worker(mode, device, context, deadline):
    ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / f"{mode}.log"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(device), MOLGAP_EDGE_MODE=mode,
               MOLGAP_EDGE_CONTEXT=json.dumps(context), PYTHONUNBUFFERED="1",
               PYTHONPATH=context["python_root"])
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "molgap.k1_edge_kaggle_runtime"],
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        last = None
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                raise TimeoutError(f"{mode}: ten-hour wall limit; retain checkpoints")
            try:
                process.wait(timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                pass
            lines = log_path.read_text(errors="replace").splitlines()
            if lines and lines[-1] != last:
                last = lines[-1]
                print(f"{mode}: {last}", flush=True)
    if process.returncode:
        print(log_path.read_text(errors="replace")[-14000:], flush=True)
        raise RuntimeError(f"{mode}: exit {process.returncode}")
    return {"mode": mode, "complete": True}


def main(context=None):
    if os.environ.get("MOLGAP_EDGE_MODE"):
        mode = os.environ["MOLGAP_EDGE_MODE"]
        if mode not in MODES:
            raise ValueError(mode)
        child(json.loads(os.environ["MOLGAP_EDGE_CONTEXT"]), mode)
        return
    started = time.monotonic()
    if not context:
        raise RuntimeError("Use verified thin launcher")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6"])
    subprocess.check_call([sys.executable, "-c", "import torch; assert torch.cuda.device_count()==2; assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))"])
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        pending = {executor.submit(worker, mode, device, context, started + 36000): mode
                   for device, mode in enumerate(MODES)}
        for future in concurrent.futures.as_completed(pending):
            try:
                outcomes.append(future.result())
            except Exception as error:
                outcomes.append({"mode": pending[future], "complete": False, "error": str(error)})
    temporary = ROOT / "job_summary.json.tmp"
    temporary.write_text(json.dumps({"outcomes": outcomes, "elapsed_seconds": time.monotonic()-started,
        "source_commit": context["source_commit"], "source_archive_sha256": context["source_archive_sha256"]}, indent=2))
    os.replace(temporary, ROOT / "job_summary.json")
    if not all(item["complete"] for item in outcomes):
        raise RuntimeError("Worker failure: preserve independent outcomes, do not duplicate completed arm")


if __name__ == "__main__":
    main()
