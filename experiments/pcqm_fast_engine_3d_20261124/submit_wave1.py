"""Submit Wave-1 experiments across all 3 Kaggle T4x2 accounts safely using existing credential adapters."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from molgap.kaggle_accelerator_push import push_kernel_with_accelerator
from molgap.kaggle_submission import _poll_source_ready, connect_kaggle
from molgap.common.kaggle import dataset_status as _dataset_status, dataset_details as _dataset_details


FROZEN_INIT = Path(r"D:\molgap-artifacts\initializations\structural-bias-norm-100k\gptrans.pt")

JOBS = [
    {
        "label": "kaggle1",
        "account": "nothingnessvoid",
        "credentials": Path(r"D:\下载\Key\.kaggle1_default\kaggle.json"),
        "source_slug": "molgap-fast-engine-w1-src",
        "kernel_slug": "molgap-fast-w1-speed-parity-100k",
        "kernel_title": "MolGap Fast W1 Speed Parity 100K",
        "mode": "wave1_kaggle1",
        "extra_datasets": ["nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1"],
    },
    {
        "label": "kaggle2",
        "account": "kaseichou",
        "credentials": Path(r"D:\下载\Key\.kaggle2_cli\kaggle.json"),
        "source_slug": "molgap-fast-engine-w1-src",
        "kernel_slug": "molgap-fast-w1-200ep-demolreg-100k",
        "kernel_title": "MolGap Fast W1 200ep DeMolReg 100K",
        "mode": "wave1_kaggle2",
        "extra_datasets": ["kaseichou/pcqm4mv2-ogb-fixed-100k-v1"],
    },
    {
        "label": "kaggle3",
        "account": "nvoid912",
        "credentials": Path(r"D:\下载\Key\.kaggle3_cli\kaggle.json"),
        "source_slug": "molgap-fast-engine-w1-src",
        "kernel_slug": "molgap-fast-w1-etkdg3d-100k",
        "kernel_title": "MolGap Fast W1 ETKDG3D 100K",
        "mode": "wave1_kaggle3",
        "extra_datasets": [
            "nvoid912/pcqm4mv2-ogb-fixed-100k-v1",
            "nvoid912/pcqm4mv2-ogb-fixed-100k-pos-v1",
        ],
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_source_dataset(job: dict, dry_run: bool) -> dict:
    account = job["account"]
    ds_ref = f"{account}/{job['source_slug']}"
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "src_ds"
        staged.mkdir()
        files = {
            "fast_gptrans.py": EXP_DIR / "fast_gptrans.py",
            "fast_demol_cin.py": EXP_DIR / "fast_demol_cin.py",
            "run_kaggle_fast.py": EXP_DIR / "run_kaggle_fast.py",
            "gptrans.pt": FROZEN_INIT,
        }
        hashes = {}
        for name, src in files.items():
            dst = staged / name
            shutil.copyfile(src, dst)
            hashes[name] = _sha256(dst)
        meta = {
            "title": "MolGap Fast Engine W1 Source",
            "id": ds_ref,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staged / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if dry_run:
            return {"dataset": ds_ref, "status": "DRY_RUN", "hashes": hashes}

        api = connect_kaggle(job["credentials"])
        existing = _dataset_status(api, ds_ref)
        if existing is None:
            resp = api.dataset_create_new(str(staged), public=False, quiet=True, convert_to_csv=False, dir_mode="zip")
            action = "created"
        else:
            resp = api.dataset_create_version(
                str(staged),
                version_notes=f"Wave1 sync {datetime.now(timezone.utc).isoformat()}",
                quiet=True,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="zip",
            )
            action = "updated"
        if getattr(resp, "error", None):
            raise RuntimeError(f"Dataset upload error for {ds_ref}: {resp.error}")
        status, forbidden = _poll_source_ready(api, ds_ref)
        if forbidden or not status or status.get("status") != "ready":
            raise RuntimeError(f"Source dataset not ready for {ds_ref}: status={status}, forbidden={forbidden}")
        details = _dataset_details(api, ds_ref)
        return {
            "dataset": ds_ref,
            "action": action,
            "version": int(details.current_version_number),
            "hashes": hashes,
        }


def make_entry_script(mode: str, expected_init_sha: str) -> str:
    return f'''"""Auto-generated Kaggle Wave-1 bootstrap entrypoint."""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
INPUT_ROOT = Path("/kaggle/input")
WORK_OUT = Path("/kaggle/working")

def ensure_deps():
    need = []
    for mod, pkg in (("torch_geometric", "torch-geometric==2.6.1"), ("ogb", "ogb==1.3.6")):
        try:
            __import__(mod)
        except Exception:
            need.append(pkg)
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2", *need], check=True)

ensure_deps()
runners = list(INPUT_ROOT.rglob("run_kaggle_fast.py"))
if len(runners) != 1:
    raise RuntimeError(f"Expected 1 run_kaggle_fast.py under /kaggle/input, found {{len(runners)}}: {{runners}}")
src_dir = str(runners[0].parent)
sys.path.insert(0, src_dir)
print("Bootstrapped from:", src_dir, "| init_sha_expected:", "{expected_init_sha}", flush=True)

import run_kaggle_fast
raise SystemExit(run_kaggle_fast.main(["--mode", "{mode}", "--input-root", "/kaggle/input", "--output-dir", "/kaggle/working"]))
'''


def submit_job(job: dict, ds_info: dict, dry_run: bool) -> dict:
    account = job["account"]
    kernel_ref = f"{account}/{job['kernel_slug']}"
    datasets = [ds_info["dataset"], *job["extra_datasets"]]
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "kernel_pkg"
        pkg.mkdir()
        meta = {
            "id": kernel_ref,
            "title": job["kernel_title"],
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": datasets,
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
            "keywords": [],
        }
        (pkg / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        init_sha = ds_info["hashes"]["gptrans.pt"]
        (pkg / "run.py").write_text(make_entry_script(job["mode"], init_sha), encoding="utf-8")
        if dry_run:
            return {"label": job["label"], "kernel": kernel_ref, "status": "DRY_RUN", "datasets": datasets}
        res = push_kernel_with_accelerator(
            package_dir=pkg,
            credential_path=job["credentials"],
            accelerator="NvidiaTeslaT4",
        )
        return {"label": job["label"], "account": account, "mode": job["mode"], "datasets": datasets, "source_dataset": ds_info, "push_result": res}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)

    selected = [j for j in JOBS if not args.only or j["label"] in args.only]
    receipts = {"submitted_at_utc": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run, "jobs": []}
    for job in selected:
        print(f"=== Preparing {job['label']} ({job['account']}) ===", flush=True)
        ds_info = ensure_source_dataset(job, dry_run=args.dry_run)
        rec = submit_job(job, ds_info, dry_run=args.dry_run)
        receipts["jobs"].append(rec)
        print(f"  -> {json.dumps(rec.get('push_result', rec), indent=2)}", flush=True)

    sub_dir = EXP_DIR / "submissions"
    sub_dir.mkdir(exist_ok=True)
    if not args.dry_run:
        (sub_dir / "wave1_receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
