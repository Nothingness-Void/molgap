"""Run one frozen GPTrans input-initialization arm on Kaggle."""
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
VARIANT = "input_embedding_normal002"
ARM_ID = "input-embedding-normal002"
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
REFERENCE_INITIAL_STATE_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}, found {len(matches)}")
    return matches[0]


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def extract_source(archive: Path, phase: str) -> Path:
    root = Path(f"/kaggle/temp/molgap-input-init-source-{phase}")
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


def main() -> None:
    # The V4 trainer certifies exactly one visible accelerator.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    archive = find_one("source_payload.bin")
    source_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    source_commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    package_manifest = json.loads(find_one("package_manifest.json").read_bytes())
    spec_raw = find_one("experiment_spec.json").read_bytes()
    spec = json.loads(spec_raw)
    stable_manifest = {key: value for key, value in package_manifest.items()
                       if key != "package_identity"}
    if hashlib.sha256(canonical(stable_manifest)).hexdigest() != package_manifest["package_identity"]:
        raise RuntimeError("Source package identity changed")
    if (hashlib.sha256(spec_raw).hexdigest() != package_manifest["spec_sha256"]
            or sha256_file(archive) != source_sha
            or source_sha != package_manifest["archive_sha256"]
            or source_commit != package_manifest["source_commit"]):
        raise RuntimeError("Source/spec/package binding changed")
    if len(spec["arms"]) != 1:
        raise RuntimeError("Expected one candidate arm")
    arm = spec["arms"][0]
    if (arm["arm_id"] != ARM_ID or arm["scientific_role"] != "candidate"
            or arm["addons"] or arm["addon_semantics"] != "baseline"
            or arm["initialization"]["seed"] != 42
            or arm["initialization"]["kind"] != "frozen_state"):
        raise RuntimeError("Frozen input-initialization arm changed")
    candidate_initial_sha256 = arm["initialization"]["state_sha256"]
    dataset_manifest = find_one("manifest.json")
    if sha256_file(dataset_manifest) != MANIFEST_SHA256:
        raise RuntimeError("Accepted graph manifest changed")
    initial_state = find_one("initial_state.pt")
    if sha256_file(initial_state) != REFERENCE_INITIAL_STATE_SHA256:
        raise RuntimeError("Frozen reference initial state changed")

    output = Path("/kaggle/working") / EXPERIMENT
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "spec_identity": package_manifest["spec_identity"],
        "package_identity": package_manifest["package_identity"],
        "source_commit": source_commit,
        "source_archive_sha256": source_sha,
        "manifest_sha256": MANIFEST_SHA256,
        "reference_initial_state_sha256": REFERENCE_INITIAL_STATE_SHA256,
        "candidate_initial_state_sha256": candidate_initial_sha256,
        "variant": VARIANT,
    }
    (output / "submission_binding.json").write_bytes(canonical(binding))
    phase = os.environ.get("MOLGAP_INPUT_INIT_PHASE")
    if phase is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "torch-geometric==2.6.1", "ogb==1.3.6",
        ])
        # V4 runtime fingerprints require a fresh interpreter for each phase.
        for child_phase in ("preflight", "train"):
            environment = os.environ.copy()
            environment["MOLGAP_INPUT_INIT_PHASE"] = child_phase
            subprocess.check_call([sys.executable, __file__], env=environment)
        print("Training reached terminal output; desktop acceptance remains separate", flush=True)
        return
    if phase not in {"preflight", "train"}:
        raise RuntimeError("Unauthorized candidate phase")
    root = extract_source(archive, phase)
    embedded_entry = root / "experiments" / EXPERIMENT / "run_candidate.py"
    if sha256_file(Path(__file__)) != sha256_file(embedded_entry):
        raise RuntimeError("Kaggle entry differs from committed source archive")
    sys.path.insert(0, str(root / "src"))
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training
    import torch
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("T4x2 candidate requires one visible T4 in each worker")

    common = {
        "dataset_root": dataset_manifest.parent,
        "manifest_path": dataset_manifest,
        "source_archive": archive,
        "source_archive_sha256": source_sha,
        "source_commit": source_commit,
        "output": output,
        "platform_id": "kaggle-gptrans-input-init-100k",
        "initial_state_path": initial_state,
        "variant": VARIANT,
        "candidate_initial_sha256": candidate_initial_sha256,
    }
    if phase == "preflight":
        preflight = run_preflight(**common)
        if preflight.get("accepted") is not True:
            raise RuntimeError("Frozen GPU runtime preflight rejected candidate")
        print(f"Runtime certificate: {preflight['runtime_certificate_id']}", flush=True)
    else:
        run_training(**common, preflight_path=output / "preflight.json")


if __name__ == "__main__":
    main()
