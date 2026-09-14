"""Shared Kaggle T4x2 orchestration with isolated fresh-process stages."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODES = tuple(json.loads(os.environ.get("MOLGAP_SCREEN_MODES", '["memory_value", "memory_message"]')))
ROOT = Path(os.environ.get("MOLGAP_SCREEN_ROOT", "/kaggle/working/gptrans_memory_readback"))
MAX_WALL_SECONDS = 10 * 3600


def find_one(pattern):
    paths = list(Path("/kaggle/input").rglob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {paths}")
    return paths[0]


def discover():
    archive = find_one("source_payload.bin")
    modules = list(Path("/kaggle/input").rglob("src/molgap/gptrans_variants.py"))
    if len(modules) == 1:
        python_root = modules[0].parents[1]
    else:
        expanded = Path("/kaggle/working/_relation_source")
        shutil.unpack_archive(archive, expanded, format="gztar")
        python_root = expanded / "src"
    inventory = json.loads(find_one("SOURCE_FILES.json").read_text())
    for item in inventory["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "src":
            raise RuntimeError("Invalid source inventory path")
        installed = python_root.joinpath(*relative.parts[1:])
        if hashlib.sha256(installed.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {relative}")
    manifests = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            value = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if value.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1" and value.get("identity", {}).get("name") == "ogb-train-100k":
            manifests.append(path)
    if len(manifests) != 1:
        raise RuntimeError(f"Fixed cache manifest ambiguity: {manifests}")
    return {"python_root": str(python_root), "dataset_root": str(manifests[0].parent),
            "manifest_path": str(manifests[0]), "source_archive": str(archive),
            "source_archive_sha256": find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip(),
            "source_commit": find_one("SOURCE_COMMIT.txt").read_text().strip(),
            "initial_state_path": str(find_one("initial_state.pt"))}


def child(stage, variant, context):
    sys.path.insert(0, context.pop("python_root"))
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training
    from molgap.gptrans_variant_checks import run_checks
    from molgap.training_reproducibility import atomic_json
    root = ROOT / variant
    for key in ("dataset_root", "manifest_path", "source_archive", "initial_state_path"):
        context[key] = Path(context[key])
    common = {**context, "variant": variant, "platform_id": "kaggle2-t4"}
    import torch
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Each candidate must see exactly one T4")
    if stage == "checks":
        atomic_json(root / "remote_checks.json", run_checks(common["initial_state_path"], variant))
    elif stage == "preflight":
        run_preflight(output=root / "preflight", **common)
    elif stage == "training":
        run_training(output=root / "training", preflight_path=root / "preflight/preflight.json", **common)
    else:
        raise ValueError(stage)


def worker(variant, device, context, deadline):
    root = ROOT / variant
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(device), MOLGAP_VARIANT=variant,
               MOLGAP_CONTEXT=json.dumps(context), PYTHONUNBUFFERED="1")
    for stage in ("checks", "preflight", "training"):
        env["MOLGAP_STAGE"] = stage
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Frozen notebook wall budget exhausted")
        with (root / f"{stage}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, __file__], env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            last_line = None
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise TimeoutError(f"{variant} reached frozen wall budget")
                try:
                    process.wait(timeout=min(60, remaining))
                except subprocess.TimeoutExpired:
                    pass
                lines = (root / f"{stage}.log").read_text(errors="replace").splitlines()
                if lines and lines[-1] != last_line:
                    last_line = lines[-1]
                    print(f"{variant}/{stage}: {last_line}", flush=True)
        print(f"{variant}/{stage}: exit={process.returncode}", flush=True)
        if process.returncode:
            print((root / f"{stage}.log").read_text()[-12000:], flush=True)
            raise RuntimeError(f"{variant} failed {stage}")
    return {"variant": variant, "complete": True}


def main():
    if os.environ.get("MOLGAP_STAGE"):
        mode = os.environ["MOLGAP_VARIANT"]
        if mode not in MODES:
            raise ValueError(mode)
        child(os.environ["MOLGAP_STAGE"], mode, json.loads(os.environ["MOLGAP_CONTEXT"]))
        return
    started = time.monotonic()
    # Pin a supported runtime; never inherit the rolling Kaggle torch ABI.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                           "--index-url", "https://download.pytorch.org/whl/cu121"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "torch-geometric==2.6.1", "ogb==1.3.6"])
    subprocess.check_call([sys.executable, "-c", "import torch; assert torch.cuda.device_count()==2; assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))"])
    context = discover()
    ROOT.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(worker, mode, device, context, started + MAX_WALL_SECONDS): mode
                   for device, mode in enumerate(MODES)}
        for future in concurrent.futures.as_completed(futures):
            try:
                outcomes.append(future.result())
            except Exception as error:
                outcomes.append({"variant": futures[future], "complete": False, "error": str(error)})
    temporary = ROOT / "job_summary.json.tmp"
    temporary.write_text(json.dumps({"outcomes": outcomes, "elapsed_seconds": time.monotonic() - started}, indent=2))
    os.replace(temporary, ROOT / "job_summary.json")
    if not all(item["complete"] for item in outcomes):
        raise RuntimeError(f"Candidate failure; retain all independent outputs: {outcomes}")


if __name__ == "__main__":
    main()
