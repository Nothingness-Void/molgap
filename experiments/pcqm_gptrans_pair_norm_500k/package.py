"""Package an immutable Kaggle3 source dataset and the formal 500K kernel."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from molgap.constants import REPO_ROOT


EXPERIMENT = "experiments/pcqm_gptrans_pair_norm_500k"
GRAPH_DATASET = "nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1"
KERNEL_ID = "nvoid912/molgap-gptrans-pair-update-norm-500k-v1"


ENTRY = r'''"""Formal Kaggle3 entry for the frozen pair-update-norm 500K bridge."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path


SOURCE_SHA256 = __SOURCE_SHA256__


def main() -> None:
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    print("GPU allocation:", names, flush=True)
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected Kaggle T4x2, received {names}")

    archives = list(Path("/kaggle/input").rglob("evidence_source.bin"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one immutable source archive, found {archives}")
    if hashlib.sha256(archives[0].read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Immutable source archive hash changed")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    runtime = Path("/kaggle/working/runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(runtime)

    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": str(runtime / "src"),
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MOLGAP_V4_LOADER_WORKERS": "2",
    })
    output = Path("/kaggle/working/gptrans_pair_update_norm")
    command = [
        sys.executable,
        "-u",
        "-m",
        "molgap.pcqm_500k_v4_evidence",
        "--arm",
        "gptrans_pair_update_norm",
        "--output",
        str(output),
        "--source-sha",
        SOURCE_SHA256,
        "--stage-epochs",
        "60",
        "--max-stage-seconds",
        "41400",
        "--platform-id",
        "kaggle3-t4x2",
    ]
    code = subprocess.call(command, env=environment)
    if code:
        raise RuntimeError(f"Training worker failed with exit code {code}")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", EXPERIMENT],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Packaging requires committed source and experiment files")

    staging = (
        REPO_ROOT
        / "platforms/_records/kaggle/staging/pcqm_gptrans_pair_norm_500k"
    )
    source = staging / "source"
    kernel = staging / "kernel"
    source.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)

    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT
    )
    paths = sorted(
        (
            REPO_ROOT / item.decode("utf-8")
            for item in raw.split(b"\0")
            if item
        ),
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    archive_path = source / "evidence_source.bin"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                relative = path.relative_to(REPO_ROOT).as_posix()
                item = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, path.read_bytes().replace(b"\r\n", b"\n"))
    source_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    source_slug = f"nvoid912/molgap-gptrans-pair-norm-500k-source-{source_sha[:10]}"
    (source / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": f"MolGap GPTrans PairNorm 500K Source {source_sha[:10]}",
                "id": source_slug,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    entry = ENTRY.replace("__SOURCE_SHA256__", repr(source_sha))
    (kernel / "run.py").write_text(entry, encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KERNEL_ID,
                "title": "MolGap GPTrans Pair Update Norm 500K V1",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [source_slug, GRAPH_DATASET],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_directory": str(source),
                "source_dataset": source_slug,
                "source_sha256": source_sha,
                "kernel_directory": str(kernel),
                "kernel": KERNEL_ID,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

