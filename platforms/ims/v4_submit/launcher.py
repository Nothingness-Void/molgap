"""Fail-closed IMS V4 staging, submission, and worker launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


BOUNDARY = Path("/lustre/home/users/sm2/chou")
INFRA_ROOT = BOUNDARY / "molgap-v4-submit"
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
WALLTIME = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
PLACEHOLDERS = {
    "code_root",
    "dataset_root",
    "dataset_manifest",
    "output_root",
    "preflight_output",
    "training_output",
    "source_archive",
    "source_archive_sha256",
    "source_commit",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def require_below(path: Path, root: Path, *, resolve: bool = True) -> Path:
    candidate = path.resolve() if resolve else Path(os.path.abspath(path))
    root = root.resolve()
    if candidate == root or root not in candidate.parents:
        raise RuntimeError(f"Path escapes authorized root: {candidate}")
    return candidate


def walltime_seconds(value: str) -> int:
    match = WALLTIME.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid walltime: {value}")
    hours, minutes, seconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid walltime: {value}")
    return hours * 3600 + minutes * 60 + seconds


def maintenance_guard(train_seconds: int, now: datetime | None = None) -> None:
    now = now or datetime.now(ZoneInfo("Asia/Tokyo"))
    first = now.replace(day=1, hour=9, minute=0, second=0, microsecond=0)
    first_monday = first + timedelta(days=(7 - first.weekday()) % 7)
    if first_monday < now:
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        first_monday = next_month + timedelta(days=(7 - next_month.weekday()) % 7)
    protected_start = first_monday - timedelta(days=1)
    projected = now + timedelta(seconds=int(train_seconds * 1.25))
    if now >= protected_start or projected >= first_monday:
        raise RuntimeError(
            "IMS maintenance guard: submission would enter the Sunday/first-Monday window"
        )


def _registry_paths(base: Path) -> tuple[Path, Path]:
    return base / "dataset_registry.json", base / "runtime_registry.json"


def validate_registries(base: Path) -> tuple[dict, dict]:
    dataset_path, runtime_path = _registry_paths(base)
    datasets = load_json(dataset_path)
    runtimes = load_json(runtime_path)
    if datasets.get("boundary") != str(BOUNDARY):
        raise RuntimeError("Dataset registry boundary changed")
    fixed_root = require_below(Path(datasets["fixed_root"]), BOUNDARY)
    acceptance = fixed_root / datasets["acceptance"]["path"]
    if sha256_file(acceptance) != datasets["acceptance"]["sha256"]:
        raise RuntimeError("Fixed dataset acceptance identity changed")
    accepted = load_json(acceptance)
    if accepted.get("status") != "accepted" or not accepted.get("verify_content"):
        raise RuntimeError("Fixed dataset root is not content-accepted")
    for identity, item in datasets["datasets"].items():
        manifest = fixed_root / item["manifest"]
        if sha256_file(manifest) != item["manifest_sha256"]:
            raise RuntimeError(f"Dataset manifest identity changed: {identity}")
        payload = load_json(manifest)
        if payload.get("status") != "complete":
            raise RuntimeError(f"Dataset manifest is incomplete: {identity}")
        if payload.get("identity", {}).get("name") != identity:
            raise RuntimeError(f"Dataset identity mismatch: {identity}")
        for role in (
            "official_validation_role_read",
            "test_dev_role_read",
            "test_challenge_role_read",
        ):
            if payload.get(role) is not False:
                raise RuntimeError(f"Sealed-role flag changed in {identity}: {role}")
    return datasets, runtimes


def validate_spec(spec: dict, datasets: dict, runtimes: dict) -> None:
    if spec.get("format") != "molgap-ims-v4-run-v1":
        raise RuntimeError("Unsupported run spec")
    if not RUN_ID.fullmatch(str(spec.get("run_id", ""))):
        raise RuntimeError("Unsafe run_id")
    if spec.get("dataset_id") not in datasets["datasets"]:
        raise RuntimeError("Unknown fixed dataset identity")
    if spec.get("runtime_id") not in runtimes["runtimes"]:
        raise RuntimeError("Unknown runtime identity")
    source = spec.get("source", {})
    if not COMMIT.fullmatch(str(source.get("commit", ""))):
        raise RuntimeError("Source commit must be a full SHA1")
    if not SHA256.fullmatch(str(source.get("archive_sha256", ""))):
        raise RuntimeError("Source archive SHA256 is invalid")
    contract = spec.get("contract", {})
    if contract.get("physical_batch_per_device") != 128:
        raise RuntimeError("V4 requires physical batch 128")
    if contract.get("precision") != "fp32" or contract.get("tf32") is not False:
        raise RuntimeError("V4 requires strict FP32 with TF32 disabled")
    if contract.get("tail_batch_policy") != "drop_last":
        raise RuntimeError("V4 requires drop_last")
    if not SHA256.fullmatch(str(contract.get("sha256", ""))):
        raise RuntimeError("Contract SHA256 is invalid")
    sealed = spec.get("sealed_roles", {})
    required_sealed = {
        "official_validation_read",
        "test_dev_read",
        "test_challenge_read",
    }
    if set(sealed) != required_sealed or any(value is not False for value in sealed.values()):
        raise RuntimeError("Sealed evaluation roles must remain unread")
    resources = spec.get("resources", {})
    runtime = runtimes["runtimes"][spec["runtime_id"]]
    if resources.get("queue") != runtime["queue"] or resources.get("queue") != "H":
        raise RuntimeError("Only the frozen H-queue runtime is supported")
    if resources.get("ngpus") != 1 or resources.get("ncpus") not in (8, 16):
        raise RuntimeError("Unsupported IMS resource request")
    if resources.get("ompthreads") not in (4, 8):
        raise RuntimeError("Unsupported OMP thread count")
    if resources["ompthreads"] > resources["ncpus"]:
        raise RuntimeError("OMP threads exceed allocated CPUs")
    if walltime_seconds(resources["preflight_walltime"]) > 3600:
        raise RuntimeError("Preflight walltime exceeds one hour")
    if walltime_seconds(resources["train_walltime"]) > 24 * 3600:
        raise RuntimeError("Training walltime exceeds 24 hours")
    for stage_name in ("preflight", "train"):
        stage = spec.get("stages", {}).get(stage_name, {})
        entrypoint = Path(stage.get("entrypoint", ""))
        if entrypoint.is_absolute() or ".." in entrypoint.parts:
            raise RuntimeError(f"Unsafe {stage_name} entrypoint")
        if not isinstance(stage.get("args"), list) or not all(
            isinstance(item, str) for item in stage["args"]
        ):
            raise RuntimeError(f"Invalid {stage_name} arguments")
        if not isinstance(stage.get("required_outputs"), list):
            raise RuntimeError(f"Missing {stage_name} required outputs")


def safe_extract(archive_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = Path(os.path.abspath(output / member.name))
            if output not in target.parents:
                raise RuntimeError(f"Unsafe source archive member: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"Unsupported source archive member: {member.name}")
        archive.extractall(output)


def validate_source_inventory(archive_path: Path, inventory_path: Path) -> None:
    inventory = load_json(inventory_path)
    if inventory.get("format") != "molgap-source-files-v1":
        raise RuntimeError("Unsupported source inventory")
    expected = {item["path"]: item["sha256"] for item in inventory.get("files", [])}
    observed = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Cannot read source member: {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            observed[member.name] = digest.hexdigest()
    if not expected or observed != expected:
        raise RuntimeError("Source archive inventory changed")


def render_pbs(run_root: Path, stage: str, spec: dict, runtime: dict) -> str:
    resources = spec["resources"]
    walltime = resources[f"{stage}_walltime"]
    launcher = run_root / "infrastructure" / "launcher.py"
    return f"""#!/bin/bash
#PBS -l select=1:ncpus={resources['ncpus']}:mpiprocs=1:ompthreads={resources['ompthreads']}:ngpus=1
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {run_root}/logs/{stage}.stdout.log
#PBS -e {run_root}/logs/{stage}.stderr.log
set -euo pipefail
test \"$(realpath {run_root})\" = \"{run_root}\"
cd {run_root}
exec {runtime['python']} -u {launcher} worker --run-root {run_root} --stage {stage}
"""


def stage_run(inbox: Path, infrastructure: Path) -> Path:
    infrastructure = require_below(infrastructure, BOUNDARY)
    inbox = require_below(inbox, infrastructure)
    datasets, runtimes = validate_registries(infrastructure)
    spec = load_json(inbox / "run_spec.json")
    validate_spec(spec, datasets, runtimes)
    run_root = infrastructure / "runs" / spec["run_id"]
    if run_root.exists():
        raise FileExistsError(f"Run identity already exists: {run_root}")
    archive = inbox / "source.tar.gz"
    if sha256_file(archive) != spec["source"]["archive_sha256"]:
        raise RuntimeError("Source archive identity changed")
    validate_source_inventory(archive, inbox / "source_files.json")
    staging_parent = infrastructure / "runs"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / f".{spec['run_id']}.staging-{os.getpid()}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        code = staging / "code"
        safe_extract(archive, code)
        contract = require_below(code / spec["contract"]["path"], code)
        if sha256_file(contract) != spec["contract"]["sha256"]:
            raise RuntimeError("Scientific contract identity changed")
        for stage in spec["stages"].values():
            entrypoint = require_below(code / stage["entrypoint"], code)
            if not entrypoint.is_file():
                raise FileNotFoundError(entrypoint)
        (staging / "logs").mkdir()
        (staging / "output" / "preflight").mkdir(parents=True)
        (staging / "output" / "training").mkdir(parents=True)
        (staging / "infrastructure").mkdir()
        shutil.copy2(Path(__file__), staging / "infrastructure" / "launcher.py")
        for name in ("dataset_registry.json", "runtime_registry.json"):
            shutil.copy2(infrastructure / name, staging / "infrastructure" / name)
        shutil.copy2(archive, staging / "source.tar.gz")
        shutil.copy2(inbox / "source_files.json", staging / "source_files.json")
        atomic_json(staging / "run_spec.json", spec)
        runtime = runtimes["runtimes"][spec["runtime_id"]]
        for stage_name in ("preflight", "train"):
            (staging / f"{stage_name}.pbs").write_text(
                render_pbs(run_root, stage_name, spec, runtime), encoding="utf-8"
            )
        atomic_json(
            staging / "staging_manifest.json",
            {
                "format": "molgap-ims-v4-staging-v1",
                "status": "complete",
                "run_id": spec["run_id"],
                "source_commit": spec["source"]["commit"],
                "source_archive_sha256": spec["source"]["archive_sha256"],
                "contract_sha256": spec["contract"]["sha256"],
                "dataset_id": spec["dataset_id"],
                "dataset_manifest_sha256": datasets["datasets"][spec["dataset_id"]]["manifest_sha256"],
                "launcher_sha256": sha256_file(Path(__file__)),
            },
        )
        os.replace(staging, run_root)
    except BaseException as error:
        atomic_json(
            staging / "staging_error.json",
            {"status": "error", "error": f"{type(error).__name__}: {error}"},
        )
        raise
    return run_root


def parse_job_id(output: str) -> str:
    matches = re.findall(r"\b\d+(?:\.[A-Za-z0-9_-]+)?\b", output)
    if not matches:
        raise RuntimeError(f"Could not parse jsub output: {output!r}")
    return matches[-1]


def submit_run(inbox: Path, infrastructure: Path) -> dict:
    run_root = stage_run(inbox, infrastructure)
    spec = load_json(run_root / "run_spec.json")
    maintenance_guard(walltime_seconds(spec["resources"]["train_walltime"]))
    queue = spec["resources"]["queue"]
    name = spec["run_id"][:15]
    preflight = subprocess.run(
        ["jsub", "-q", queue, "-N", f"{name}-pre", str(run_root / "preflight.pbs")],
        check=True,
        capture_output=True,
        text=True,
    )
    preflight_job = parse_job_id(preflight.stdout + preflight.stderr)
    submission = {
        "format": "molgap-ims-v4-submission-v1",
        "status": "preflight_submitted",
        "run_id": spec["run_id"],
        "preflight_job": preflight_job,
    }
    atomic_json(run_root / "submission.json", submission)
    training = subprocess.run(
        [
            "jsub",
            "-q",
            queue,
            "-N",
            name,
            "-W",
            f"depend=afterok:{preflight_job}",
            str(run_root / "train.pbs"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    submission.update(
        status="submitted",
        training_job=parse_job_id(training.stdout + training.stderr),
        dependency=f"afterok:{preflight_job}",
    )
    atomic_json(run_root / "submission.json", submission)
    return submission


def expand_args(values: list[str], context: dict[str, str]) -> list[str]:
    expanded = []
    for value in values:
        names = set(re.findall(r"\{([a-z_]+)\}", value))
        if not names.issubset(PLACEHOLDERS):
            raise RuntimeError(f"Unknown command placeholder: {sorted(names)}")
        expanded.append(value.format(**context))
    return expanded


def worker(run_root: Path, stage_name: str) -> None:
    run_root = require_below(run_root, INFRA_ROOT)
    spec = load_json(run_root / "run_spec.json")
    datasets, runtimes = validate_registries(run_root / "infrastructure")
    validate_spec(spec, datasets, runtimes)
    dataset_root = Path(datasets["fixed_root"])
    dataset_manifest = dataset_root / datasets["datasets"][spec["dataset_id"]]["manifest"]
    context = {
        "code_root": str(run_root / "code"),
        "dataset_root": str(dataset_root),
        "dataset_manifest": str(dataset_manifest),
        "output_root": str(run_root / "output"),
        "preflight_output": str(run_root / "output" / "preflight"),
        "training_output": str(run_root / "output" / "training"),
        "source_archive": str(run_root / "source.tar.gz"),
        "source_archive_sha256": spec["source"]["archive_sha256"],
        "source_commit": spec["source"]["commit"],
    }
    runtime = runtimes["runtimes"][spec["runtime_id"]]
    pythonpath = [item.format(**context) for item in runtime["pythonpath"]]
    environment = os.environ.copy()
    environment.update(
        PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join(pythonpath),
        OMP_NUM_THREADS=str(spec["resources"]["ompthreads"]),
        MKL_NUM_THREADS=str(spec["resources"]["ompthreads"]),
        OPENBLAS_NUM_THREADS="1",
    )
    stage = spec["stages"][stage_name]
    entrypoint = require_below(run_root / "code" / stage["entrypoint"], run_root / "code")
    command = [runtime["python"], "-u", str(entrypoint)] + expand_args(stage["args"], context)
    status_path = run_root / "output" / stage_name / "stage_status.json"
    atomic_json(status_path, {"status": "running", "stage": stage_name, "command": command})
    try:
        subprocess.run(command, check=True, cwd=run_root, env=environment)
        for relative in stage["required_outputs"]:
            output = require_below(run_root / "output" / relative, run_root / "output")
            if not output.is_file():
                raise FileNotFoundError(f"Required output missing: {output}")
        atomic_json(status_path, {"status": "complete", "stage": stage_name})
    except BaseException as error:
        atomic_json(
            status_path,
            {"status": "error", "stage": stage_name, "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--infrastructure", type=Path, default=INFRA_ROOT)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--inbox", type=Path, required=True)
    submit.add_argument("--infrastructure", type=Path, default=INFRA_ROOT)
    work = subparsers.add_parser("worker")
    work.add_argument("--run-root", type=Path, required=True)
    work.add_argument("--stage", choices=("preflight", "train"), required=True)
    args = parser.parse_args()
    if args.command == "audit":
        datasets, runtimes = validate_registries(args.infrastructure)
        print(json.dumps({"status": "accepted", "datasets": sorted(datasets["datasets"]), "runtimes": sorted(runtimes["runtimes"])}, indent=2))
    elif args.command == "submit":
        print(json.dumps(submit_run(args.inbox, args.infrastructure), indent=2))
    else:
        worker(args.run_root, args.stage)


if __name__ == "__main__":
    main()
