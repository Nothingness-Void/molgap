"""Packager for the self-contained Kaggle GPTrans Noisy Nodes + Pair Update Norm 100K training kernel."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

from molgap.constants import REPO_ROOT


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "archive"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def tracked_source_files() -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT
    )
    paths = [REPO_ROOT / value.decode("utf-8") for value in raw.split(b"\0") if value]
    for extra in (
        "src/molgap/noisy_nodes.py",
        "src/molgap/gptrans_variants.py",
        "src/molgap/pcqm_wedge.py",
        "src/molgap/pcqm_500k_v4_evidence.py",
    ):
        p = REPO_ROOT / extra
        if p.is_file() and p not in paths:
            paths.append(p)
    selected = [
        path
        for path in paths
        if not EXCLUDED_PARTS.intersection(path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    return sorted(selected, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def build_reproducible_zip(files: list[Path]) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            relative = source.relative_to(REPO_ROOT).as_posix()
            data = source.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    return payload, digest


def build_kaggle_package(
    output_dir: Path,
    account: str = "nothingnessvoid",
) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = tracked_source_files()
    payload, source_sha256 = build_reproducible_zip(files)
    payload_b64 = base64.b64encode(payload).decode("ascii")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()

    # Write kernel-metadata.json
    dataset_ref = f"{account}/pcqm4mv2-ogb-fixed-100k-v1"
    metadata = {
        "id": f"{account}/molgap-pcqm-gptrans-noisy-pair-norm-s42",
        "title": "MolGap PCQM GPTrans Noisy Pair Norm S42",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            dataset_ref,
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Write standalone run.py
    run_script = f'''"""Kaggle Tesla T4 runner for GPTrans Noisy Nodes + Pair Update Norm 100K V4."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

SOURCE_COMMIT = "{commit}"
EXPECTED_SOURCE_SHA256 = "{source_sha256}"
SOURCE_PAYLOAD_B64 = """{payload_b64}"""
EXPECTED_MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"


def find_fixed_dataset() -> tuple[Path, Path]:
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
            and payload.get("identity", {{}}).get("name") == "ogb-train-100k"
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fixed 100K manifest, found {{matches}}")
    manifest_path = matches[0]
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Manifest hash mismatch: {{manifest_hash}} vs {{EXPECTED_MANIFEST_SHA256}}")
    return manifest_path.parent, manifest_path


def main() -> None:
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    print("GPU allocation:", names, flush=True)

    # Verify and extract source archive
    source_bytes = base64.b64decode(SOURCE_PAYLOAD_B64)
    actual_sha = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Source SHA mismatch: {{actual_sha}} != {{EXPECTED_SOURCE_SHA256}}")

    runtime_dir = Path("/kaggle/working/runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        archive.extractall(runtime_dir)

    # Install PyG and OGB dependencies
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])

    dataset_root, manifest_path = find_fixed_dataset()
    print("Using 100K dataset root:", dataset_root, flush=True)

    output_dir = Path("/kaggle/working/pcqm_gptrans_noisy_pair_norm_100k")
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update({{
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": str(runtime_dir / "src"),
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }})

    command = [
        sys.executable,
        "-u",
        "-m",
        "molgap.noisy_nodes",
        "--dataset-root", str(dataset_root),
        "--manifest-path", str(manifest_path),
        "--output", str(output_dir),
        "--platform-id", "kaggle1-t4",
        "--noise-std", "0.15",
        "--loss-weight", "0.1",
        "--pair-update-norm",
    ]
    print("Launching GPTrans Noisy Nodes + Pair Update Norm command:", " ".join(command), flush=True)
    code = subprocess.call(command, env=environment)
    if code != 0:
        raise RuntimeError(f"Training worker failed with exit code {{code}}")
    print("Training finished successfully!", flush=True)


if __name__ == "__main__":
    main()
'''
    (output_dir / "run.py").write_text(run_script, encoding="utf-8")
    (output_dir / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output_dir / "SOURCE_ARCHIVE_SHA256.txt").write_text(source_sha256 + "\n", encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "source_commit": commit,
        "source_archive_sha256": source_sha256,
        "account": account,
        "metadata_path": str(output_dir / "kernel-metadata.json"),
        "run_script_path": str(output_dir / "run.py"),
        "package_bytes": len(run_script),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="nothingnessvoid")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = REPO_ROOT / f"platforms/_records/kaggle/staging/pcqm_gptrans_noisy_pair_norm_100k"
    res = build_kaggle_package(args.output, account=args.account)
    print(json.dumps(res, indent=2))
