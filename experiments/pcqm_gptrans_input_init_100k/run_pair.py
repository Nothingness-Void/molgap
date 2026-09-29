"""Run the frozen GPTrans-T initialization control and candidate on Kaggle1 T4x2."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile


EXPERIMENT = "pcqm_gptrans_input_init_100k"
ARMS = (("reference", "reference", 0),
        ("input-embedding-normal002", "input_embedding_normal002", 1))
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
INITIAL_STATE_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"
REFERENCE_MODEL_SHA256 = "8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(name: str) -> Path:
    paths = list(Path("/kaggle/input").rglob(name))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {len(paths)}")
    return paths[0]


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def extract_source(archive: Path, arm: str, phase: str) -> Path:
    root = Path("/kaggle/temp") / f"molgap-input-init-{arm}-{phase}"
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute()
                    or any(part in {"", ".", ".."} for part in name.parts)):
                raise RuntimeError("Unsafe source archive member")
            destination = root.joinpath(*name.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError("Missing source archive member")
            with stream, destination.open("xb") as target:
                shutil.copyfileobj(stream, target)
    return root


def verify_inputs() -> tuple[Path, Path, Path, dict]:
    archive = find_one("source_payload.bin")
    source_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    source_commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    package = json.loads(find_one("package_manifest.json").read_bytes())
    spec_raw = find_one("experiment_spec.json").read_bytes()
    spec = json.loads(spec_raw)
    stable = {key: value for key, value in package.items() if key != "package_identity"}
    if hashlib.sha256(canonical(stable)).hexdigest() != package["package_identity"]:
        raise RuntimeError("Source package identity changed")
    if (hashlib.sha256(spec_raw).hexdigest() != package["spec_sha256"]
            or sha256_file(archive) != source_sha
            or source_sha != package["archive_sha256"]
            or source_commit != package["source_commit"]):
        raise RuntimeError("Source/spec/package binding changed")
    expected = {"reference": ("reference", REFERENCE_MODEL_SHA256),
                "input-embedding-normal002": ("candidate", None)}
    if (spec["platform"]["accelerator"] != "NvidiaTeslaT4"
            or spec["platform"]["device_count"] != 2
            or spec["prospective"]["same_run_replay"] != {
                "reference_arm_id": "reference", "candidate_arm_ids": ["input-embedding-normal002"]}
            or {arm["arm_id"] for arm in spec["arms"]} != set(expected)):
        raise RuntimeError("Frozen two-arm T4 specification changed")
    for arm in spec["arms"]:
        role, initial_hash = expected[arm["arm_id"]]
        if (arm["scientific_role"] != role or arm["addons"]
                or arm["initialization"]["seed"] != 42
                or arm["initialization"]["kind"] != "frozen_state"
                or (initial_hash is not None
                    and arm["initialization"]["state_sha256"] != initial_hash)):
            raise RuntimeError(f"Frozen arm changed: {arm['arm_id']}")
    manifest = find_one("manifest.json")
    initial_state = find_one("initial_state.pt")
    if (sha256_file(manifest) != MANIFEST_SHA256
            or sha256_file(initial_state) != INITIAL_STATE_SHA256):
        raise RuntimeError("Accepted graph or initial-state identity changed")
    output = Path("/kaggle/working") / EXPERIMENT
    output.mkdir(parents=True, exist_ok=True)
    (output / "submission_binding.json").write_bytes(canonical({
        "spec_identity": package["spec_identity"],
        "package_identity": package["package_identity"],
        "source_commit": source_commit,
        "source_archive_sha256": source_sha,
        "manifest_sha256": MANIFEST_SHA256,
        "initial_state_sha256": INITIAL_STATE_SHA256,
        "arms": [arm["arm_id"] for arm in spec["arms"]],
    }))
    return archive, manifest, initial_state, spec


def worker(arm_id: str, phase: str, archive: Path, manifest: Path,
           initial_state: Path, spec: dict) -> None:
    if phase not in {"preflight", "train"}:
        raise RuntimeError("Unauthorized phase")
    variants = {arm: variant for arm, variant, _ in ARMS}
    root = extract_source(archive, arm_id, phase)
    embedded = root / "experiments" / EXPERIMENT / "run_pair.py"
    if sha256_file(Path(__file__)) != sha256_file(embedded):
        raise RuntimeError("Kaggle entry differs from committed source archive")
    sys.path.insert(0, str(root / "src"))
    import torch
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training

    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Each paired worker requires exactly one visible T4")
    candidate_hash = next(arm["initialization"]["state_sha256"]
                          for arm in spec["arms"] if arm["arm_id"] == arm_id)
    output = Path("/kaggle/working") / EXPERIMENT / arm_id
    common = {
        "dataset_root": manifest.parent,
        "manifest_path": manifest,
        "source_archive": archive,
        "source_archive_sha256": sha256_file(archive),
        "source_commit": find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip(),
        "output": output,
        "platform_id": "kaggle1-gptrans-input-init-pair-100k",
        "initial_state_path": initial_state,
        "variant": variants[arm_id],
        "candidate_initial_sha256": candidate_hash if arm_id != "reference" else None,
    }
    if phase == "preflight":
        result = run_preflight(**common)
        if result.get("accepted") is not True:
            raise RuntimeError(f"GPU preflight rejected {arm_id}")
    else:
        run_training(**common, preflight_path=output / "preflight.json")


def main() -> None:
    archive, manifest, initial_state, spec = verify_inputs()
    arm_id = os.environ.get("MOLGAP_INPUT_INIT_ARM")
    phase = os.environ.get("MOLGAP_INPUT_INIT_PHASE")
    if arm_id is not None:
        if arm_id not in {arm for arm, _, _ in ARMS}:
            raise RuntimeError("Unauthorized arm")
        worker(arm_id, phase, archive, manifest, initial_state, spec)
        return
    if phase is not None:
        raise RuntimeError("Parent phase must be unset")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "torch-geometric==2.6.1", "ogb==1.3.6"])
    import torch
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected Kaggle T4x2, found {names}")
    for stage in ("preflight", "train"):
        processes = []
        for arm, _, device in ARMS:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["MOLGAP_INPUT_INIT_ARM"] = arm
            environment["MOLGAP_INPUT_INIT_PHASE"] = stage
            processes.append(subprocess.Popen([sys.executable, __file__], env=environment))
        codes = [process.wait() for process in processes]
        if codes != [0, 0]:
            raise RuntimeError(f"Pair {stage} workers failed: {codes}")
    print("Paired training reached terminal output; desktop acceptance remains separate", flush=True)


if __name__ == "__main__":
    main()
