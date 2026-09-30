"""Bind the frozen V4 source bundle to one Kaggle1 T4x2 kernel."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from molgap.v4_bundle import build_v4_source_bundle


ARMS = ("gptrans_distance_only", "k1_distance_angle")
DATASET = "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1"
DATASET_MANIFEST_SHA256 = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
KERNEL = "nothingnessvoid/molgap-geometry-v4-500k-paired-s42-v1"

ENTRY = '''"""Frozen Kaggle1 two-GPU entrypoint for the geometry V4 bridge."""
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

SOURCE_COMMIT = __SOURCE_COMMIT__
SOURCE_SHA256 = __SOURCE_SHA256__
SOURCE_PAYLOAD = __SOURCE_PAYLOAD__
DATASET_MANIFEST_SHA256 = __DATASET_MANIFEST_SHA256__
RESUME_CONFIG = __RESUME_CONFIG__
ARMS = ("gptrans_distance_only", "k1_distance_angle")

def unpack_source():
    payload = base64.b64decode(SOURCE_PAYLOAD)
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Frozen source archive hash mismatch")
    root = Path("/kaggle/temp/molgap-source")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute()
                    or any(part in {"", ".", ".."} for part in name.parts)):
                raise RuntimeError("Unsafe source archive member")
            destination = root.joinpath(*name.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError("Missing source member")
            with stream, destination.open("xb") as output:
                shutil.copyfileobj(stream, output)
    return root

def check_dataset():
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        if hashlib.sha256(path.read_bytes()).hexdigest() == DATASET_MANIFEST_SHA256:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one accepted fixed-500K manifest")
    manifest = json.loads(matches[0].read_text(encoding="utf-8"))
    if (manifest.get("identity", {}).get("name") != "ogb-train-500k-scnet-v1"
            or manifest.get("official_validation_role_read") is not False
            or manifest.get("test_dev_role_read") is not False
            or manifest.get("test_challenge_role_read") is not False):
        raise RuntimeError("Fixed-500K role identity changed")
    return matches[0]

def prepare_resume():
    if RESUME_CONFIG is None:
        return None
    input_root = Path("/kaggle/input")
    matches = list(input_root.rglob("gptrans_distance_only__stage_manifest.json"))
    if len(matches) != 1:
        visible = sorted(path.name for path in input_root.iterdir())
        raise RuntimeError("Expected one checkpoint manifest under /kaggle/input; visible=" + repr(visible))
    mount = matches[0].parent
    destination = Path("/kaggle/temp/molgap-resume")
    for arm in ARMS:
        arm_dir = destination / arm
        arm_dir.mkdir(parents=True, exist_ok=False)
        for name in ("stage_manifest.json", "last_checkpoint.pt", "best_model.pt",
                     "best_predictions.pt", "initial_state.pt"):
            flat_name = arm + "__" + name
            source = mount / flat_name
            if not source.is_file():
                raise RuntimeError("Missing checkpoint dataset file: " + flat_name)
            if hashlib.sha256(source.read_bytes()).hexdigest() != RESUME_CONFIG["files"][flat_name]:
                raise RuntimeError("Checkpoint dataset hash mismatch: " + flat_name)
            shutil.copy2(source, arm_dir / name)
        stage = json.loads((arm_dir / "stage_manifest.json").read_text(encoding="utf-8"))
        expected_epoch = RESUME_CONFIG.get("arm_next_epochs", {}).get(
            arm, RESUME_CONFIG.get("resume_next_epoch")
        )
        if (stage["arm"] != arm or stage["status"] != RESUME_CONFIG["resume_status"]
                or stage["next_epoch"] != expected_epoch
                or stage["source_sha256"] != RESUME_CONFIG["resume_source_sha256"]):
            raise RuntimeError("Checkpoint stage identity mismatch: " + arm)
    return destination

def launch(root, mode, resume_root=None):
    processes = []
    for gpu, arm in enumerate(ARMS):
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(root / "src"),
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "MOLGAP_V4_LOADER_WORKERS": "2",
            "MOLGAP_SOURCE_COMMIT": SOURCE_COMMIT,
        })
        output = Path("/kaggle/working") / mode / arm
        command = [sys.executable, "-u", "-m", "molgap.pcqm_500k_v4_evidence",
                   "--arm", arm, "--output", str(output),
                   "--source-sha", SOURCE_SHA256,
                   "--stage-epochs", "60", "--max-stage-seconds", "39600",
                   "--platform-id", "kaggle1-t4"]
        if mode == "preflight":
            command.append("--preflight-only")
        elif resume_root is not None:
            command.extend(("--resume", str(resume_root / arm),
                            "--resume-source-sha", RESUME_CONFIG["resume_source_sha256"],
                            "--device-second-cap",
                            str(RESUME_CONFIG["arm_device_second_caps"][arm])))
        processes.append(subprocess.Popen(command, env=env))
    return {arm: process.wait() for arm, process in zip(ARMS, processes)}

def main():
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if len(devices) != 2 or not all("T4" in name for name in devices):
        raise RuntimeError("Expected two isolated Tesla T4 devices")
    check_dataset()
    resume_root = prepare_resume()
    root = unpack_source()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "torch==2.4.1", "--index-url",
                           "https://download.pytorch.org/whl/cu121"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"])
    preflight_codes = launch(root, "preflight")
    if any(preflight_codes.values()):
        raise RuntimeError("Both-arm GPU preflight failed: " + str(preflight_codes))
    training_codes = launch(root, "evidence", resume_root)
    if any(training_codes.values()):
        raise RuntimeError("Geometry training worker failed: " + str(training_codes))
    terminal = {
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "resume_config": RESUME_CONFIG,
        "arms": {},
    }
    for arm in ARMS:
        result = json.loads((Path("/kaggle/working/evidence") / arm /
                             "stage_manifest.json").read_text(encoding="utf-8"))
        terminal["arms"][arm] = {
            "status": result["status"], "next_epoch": result["next_epoch"],
            "cumulative_device_seconds": result.get("cumulative_device_seconds"),
        }
    Path("/kaggle/working/kernel_status.json").write_text(
        json.dumps(terminal, indent=2) + "\\n", encoding="utf-8"
    )
    print("Two-arm stage published", json.dumps(terminal), flush=True)

if __name__ == "__main__":
    main()
'''


def build_package(repo_root: Path, output: Path, resume_config: Path | None = None,
                  source_bundle: Path | None = None) -> dict:
    repo_root = repo_root.resolve()
    output = output.resolve()
    resume = None if resume_config is None else json.loads(resume_config.read_text(encoding="utf-8"))
    if resume is not None:
        expected = {f"{arm}__{name}" for arm in ARMS for name in (
            "stage_manifest.json", "last_checkpoint.pt", "best_model.pt",
            "best_predictions.pt", "initial_state.pt")}
        if (set(resume["files"]) != expected or set(resume["arm_device_second_caps"]) != set(ARMS)
                or any(not isinstance(value, int) or value <= 0 for value in resume["arm_device_second_caps"].values())
                or not all(isinstance(value, str) and len(value) == 64 for value in resume["files"].values())):
            raise RuntimeError("Invalid pinned two-arm resume configuration")
        if not resume["dataset"].startswith("nothingnessvoid/") or not resume["kernel"].startswith("nothingnessvoid/"):
            raise RuntimeError("Resume dataset and kernel must belong to Kaggle1")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src/molgap"], cwd=repo_root
    )
    source_paths = sorted(
        path.decode("utf-8") for path in tracked.split(b"\0")
        if path and path.decode("utf-8").endswith(".py")
    )
    bundle_dir = output / "source_bundle"
    if source_bundle is None:
        commit = head
        bundle = build_v4_source_bundle(
            repo_root=repo_root, relative_paths=source_paths,
            output_dir=bundle_dir, source_commit=commit,
        )
    else:
        source_bundle = source_bundle.resolve()
        commit = (source_bundle / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
        inventory = json.loads((source_bundle / "SOURCE_FILES.json").read_text(encoding="utf-8"))
        if inventory["source_commit"] != commit or sorted(item["path"] for item in inventory["files"]) != source_paths:
            raise RuntimeError("Frozen source inventory differs from the package source list")
        subprocess.run(["git", "diff", "--quiet", commit, head, "--", "src/molgap"],
                       cwd=repo_root, check=True)
        dirty = subprocess.check_output(["git", "status", "--porcelain=v1", "--", "src/molgap"],
                                        cwd=repo_root, text=True)
        if dirty.strip():
            raise RuntimeError("Frozen source files changed in the working tree")
        shutil.copytree(source_bundle, bundle_dir)
        bundle = {
            "archive": str(bundle_dir / "source.tar.gz"),
            "archive_sha256": (bundle_dir / "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip(),
        }
    archive = Path(bundle["archive"]).read_bytes()
    if hashlib.sha256(archive).hexdigest() != bundle["archive_sha256"]:
        raise RuntimeError("Source bundle changed after packaging")
    kernel_dir = output / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    script = (ENTRY.replace("__SOURCE_COMMIT__", repr(commit))
              .replace("__SOURCE_SHA256__", repr(bundle["archive_sha256"]))
              .replace("__SOURCE_PAYLOAD__", repr(base64.b64encode(archive).decode("ascii")))
              .replace("__DATASET_MANIFEST_SHA256__", repr(DATASET_MANIFEST_SHA256))
              .replace("__RESUME_CONFIG__", repr(resume)))
    (kernel_dir / "run.py").write_text(script, encoding="utf-8", newline="\n")
    metadata = {
        "id": KERNEL if resume is None else resume["kernel"],
        "title": "MolGap Geometry V4 500K GPTrans K1 S42" if resume is None else resume["title"],
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET] + ([] if resume is None else [resume["dataset"]]),
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    if len(metadata["title"]) > 50:
        raise RuntimeError("Kaggle title exceeds 50 characters")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "source_commit": commit, "source_archive_sha256": bundle["archive_sha256"],
        "source_file_count": len(source_paths), "kernel": metadata["id"],
        "kernel_code_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "kernel_dir": str(kernel_dir), "dataset": DATASET,
        "resume_dataset": None if resume is None else resume["dataset"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-config", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_package(args.repo_root, args.output, args.resume_config,
                                   args.source_bundle), indent=2))


if __name__ == "__main__":
    main()
