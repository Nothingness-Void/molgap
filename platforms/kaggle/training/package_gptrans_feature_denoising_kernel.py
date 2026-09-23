"""Package the replay-ready GPTrans feature-denoising dual-arm Kaggle run."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from molgap.constants import REPO_ROOT
from molgap.v4_bundle import build_v4_source_bundle


SOURCE_COMMIT = "fa564238e33c70bd377edf4d64f3182238a3ee49"
KERNEL_SLUG = "molgap-gptrans-feature-denoising-s42"
DATASET_REF = "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1"
REFERENCE_REF = "nothingnessvoid/molgap-gptrans-noisy-pair-norm-reference-s42-v1"
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
REFERENCE_MODEL_SHA256 = "c841cdee799daa7a874e0f112ce6dea2932fe0f640f434812bac15b83684b092"


def _assert_frozen_contract() -> None:
    contract_path = (
        Path(REPO_ROOT)
        / "experiments/pcqm_gptrans_feature_denoising_100k/training_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:src/molgap/pcqm_gptrans_v4.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names = {
        "EPOCHS", "WARMUP_EPOCHS", "LEARNING_RATE", "MIN_LEARNING_RATE",
        "WEIGHT_DECAY", "PHYSICAL_BATCH", "TRAIN_ROWS",
    }
    constants = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    if names - constants.keys():
        raise RuntimeError(f"Cannot verify frozen runtime constants: {sorted(names - constants.keys())}")
    warmup_names = {4: "four", 5: "five"}
    warmup = warmup_names.get(constants["WARMUP_EPOCHS"])
    observed = {
        "source_commit": SOURCE_COMMIT,
        "epochs": constants["EPOCHS"],
        "physical_batch_per_device": constants["PHYSICAL_BATCH"],
        "optimizer_steps_per_epoch": constants["TRAIN_ROWS"] // constants["PHYSICAL_BATCH"],
        "sample_presentations": (
            constants["TRAIN_ROWS"] // constants["PHYSICAL_BATCH"]
            * constants["PHYSICAL_BATCH"] * constants["EPOCHS"]
        ),
        "optimizer": (
            f"AdamW-lr{constants['LEARNING_RATE']:.4f}"
            f"-weight_decay{constants['WEIGHT_DECAY']:.2f}-fused_false-foreach_false"
        ),
        "lr_schedule": (
            f"{warmup}_epoch_linear_warmup_then_cosine_to_"
            f"{constants['MIN_LEARNING_RATE']:.6f}"
        ),
    }
    mismatches = {
        field: {"frozen": contract.get(field), "source": value}
        for field, value in observed.items()
        if contract.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"Frozen contract differs from packaged source: {mismatches}")


def _source_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", SOURCE_COMMIT, "--", "src"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if "src/molgap/gptrans_feature_denoising.py" not in paths:
        raise RuntimeError("Frozen feature-denoising source is absent")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_import_check(archive: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="molgap-feature-denoise-") as temporary:
        root = Path(temporary)
        shutil.unpack_archive(str(archive), str(root), format="gztar")
        environment = {
            **dict(__import__("os").environ),
            "PYTHONPATH": str(root / "src"),
        }
        subprocess.run(
            [
                str(Path(__import__("sys").executable)),
                "-c",
                (
                    "from molgap.gptrans_feature_denoising import "
                    "ARMS, EXPECTED_PARAMETERS; "
                    "assert ARMS == ('full_atom', 'full_atom_bond'); "
                    "assert EXPECTED_PARAMETERS['full_atom_bond'] == 5291964"
                ),
            ],
            env=environment,
            check=True,
        )


def build_package(output: Path, account: str) -> dict:
    _assert_frozen_contract()
    output = output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    bundle_dir = output / "_bundle"
    bundle = build_v4_source_bundle(
        repo_root=Path(REPO_ROOT),
        relative_paths=_source_paths(),
        output_dir=bundle_dir,
        source_commit=SOURCE_COMMIT,
    )
    archive = bundle_dir / "source.tar.gz"
    _clean_import_check(archive)
    source_payload = base64.b64encode(archive.read_bytes()).decode("ascii")
    inventory_payload = base64.b64encode(
        (bundle_dir / "SOURCE_FILES.json").read_bytes()
    ).decode("ascii")

    metadata = {
        "id": f"{account}/{KERNEL_SLUG}",
        "title": "MolGap GPTrans Feature Denoising S42",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_REF, REFERENCE_REF],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    run_script = f'''"""Frozen Kaggle T4x2 runner for GPTrans feature denoising."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

SOURCE_COMMIT = "{SOURCE_COMMIT}"
SOURCE_ARCHIVE_SHA256 = "{bundle['archive_sha256']}"
SOURCE_PAYLOAD = """{source_payload}"""
SOURCE_INVENTORY = """{inventory_payload}"""
MANIFEST_SHA256 = "{MANIFEST_SHA256}"
REFERENCE_MODEL_SHA256 = "{REFERENCE_MODEL_SHA256}"
ARMS = ("full_atom", "full_atom_bond")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_inputs() -> tuple[Path, Path, Path]:
    manifests = []
    references = []
    for path in Path("/kaggle/input").rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
            and payload.get("identity", {{}}).get("name") == "ogb-train-100k"
        ):
            manifests.append(path)
        if payload.get("format") == "molgap-gptrans-noisy-pair-norm-reference-v1":
            references.append(path)
    if len(manifests) != 1 or len(references) != 1:
        raise RuntimeError(
            f"Expected one fixed manifest and one reference manifest: "
            f"{{manifests}}, {{references}}"
        )
    manifest = manifests[0]
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("Fixed 100K manifest hash changed")
    reference_payload = json.loads(references[0].read_text(encoding="utf-8"))
    reference_model = references[0].parent / reference_payload["artifact"]
    if sha256(reference_model) != REFERENCE_MODEL_SHA256:
        raise RuntimeError("Accepted reference checkpoint hash changed")
    return manifest.parent, manifest, reference_model


def main() -> None:
    names = [
        line.strip()
        for line in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).splitlines()
        if line.strip()
    ]
    print("GPU allocation:", names, flush=True)
    if len(names) != 2 or not all("T4" in name for name in names):
        raise RuntimeError(f"Expected Kaggle T4x2, received {{names}}")

    runtime = Path("/kaggle/working/runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    archive = runtime / "source.tar.gz"
    archive.write_bytes(base64.b64decode(SOURCE_PAYLOAD))
    if sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Embedded source archive hash changed")
    (runtime / "SOURCE_COMMIT.txt").write_text(SOURCE_COMMIT + "\\n", encoding="utf-8")
    (runtime / "SOURCE_ARCHIVE_SHA256.txt").write_text(
        SOURCE_ARCHIVE_SHA256 + "\\n", encoding="utf-8"
    )
    (runtime / "SOURCE_FILES.json").write_bytes(base64.b64decode(SOURCE_INVENTORY))
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(runtime)

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )
    dataset_root, manifest, reference_model = find_inputs()
    output = Path("/kaggle/working/gptrans_feature_denoising_100k")
    output.mkdir(parents=True, exist_ok=True)

    workers = []
    for gpu, arm in enumerate(ARMS):
        arm_output = output / arm
        environment = os.environ.copy()
        environment.update(
            {{
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(runtime / "src"),
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            }}
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "molgap.gptrans_feature_denoising",
            "--arm",
            arm,
            "--dataset-root",
            str(dataset_root),
            "--manifest-path",
            str(manifest),
            "--source-archive",
            str(archive),
            "--source-archive-sha256",
            SOURCE_ARCHIVE_SHA256,
            "--source-commit",
            SOURCE_COMMIT,
            "--output",
            str(arm_output),
            "--platform-id",
            f"kaggle1-t4x2-gpu{{gpu}}",
        ]
        print("Launching", arm, "on", names[gpu], flush=True)
        workers.append((arm, subprocess.Popen(command, env=environment)))

    failures = []
    for arm, worker in workers:
        code = worker.wait()
        if code:
            failures.append((arm, code))
    if failures:
        raise RuntimeError(f"Dual-arm worker failures: {{failures}}")

    reference_environment = os.environ.copy()
    reference_environment.update(
        {{
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(runtime / "src"),
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }}
    )
    subprocess.check_call(
        [
            sys.executable,
            "-u",
            "-m",
            "molgap.gptrans_feature_denoising",
            "--reference-model",
            str(reference_model),
            "--dataset-root",
            str(dataset_root),
            "--manifest-path",
            str(manifest),
            "--source-archive",
            str(archive),
            "--source-archive-sha256",
            SOURCE_ARCHIVE_SHA256,
            "--source-commit",
            SOURCE_COMMIT,
            "--output",
            str(output / "reference"),
        ],
        env=reference_environment,
    )

    artifacts = {{}}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            artifacts[path.relative_to(output).as_posix()] = sha256(path)
    summary = {{
        "format": "molgap-gptrans-feature-denoising-dual-arm-v1",
        "complete": True,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "reference_model_sha256": REFERENCE_MODEL_SHA256,
        "gpu_allocation": names,
        "arms": list(ARMS),
        "artifacts": artifacts,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }}
    (output / "job_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''
    (output / "run.py").write_text(run_script, encoding="utf-8", newline="\n")
    for path in bundle_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    shutil.rmtree(bundle_dir)
    package_manifest = {
        "format": "molgap-kaggle-feature-denoising-package-v1",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": bundle["archive_sha256"],
        "source_files": bundle["file_count"],
        "run_py_sha256": _sha256(output / "run.py"),
        "metadata_sha256": _sha256(output / "kernel-metadata.json"),
        "arms": ["full_atom", "full_atom_bond"],
        "reference_model_sha256": REFERENCE_MODEL_SHA256,
    }
    (output / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="nothingnessvoid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_package(args.output, args.account), indent=2))


if __name__ == "__main__":
    main()
