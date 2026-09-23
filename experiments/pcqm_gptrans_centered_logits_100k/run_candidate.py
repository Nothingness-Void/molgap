"""Kaggle entry point for one frozen GPTrans centered-logits 100K arm."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


VARIANT = "centered_logits"
EXPERIMENT = "pcqm_gptrans_centered_logits_100k"
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
INITIAL_STATE_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def main() -> None:
    # Keep only one device visible even if Kaggle assigns a two-T4 machine.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    source_module = find_one("src/molgap/pcqm_gptrans_v4.py")
    source_root = source_module.parents[2]
    archive = find_one("source_payload.bin")
    source_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    source_commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    spec_path = find_one("experiment_spec.json")
    package_manifest = json.loads(find_one("package_manifest.json").read_bytes())
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    stable = {key: value for key, value in package_manifest.items() if key != "package_identity"}
    if hashlib.sha256(canonical(stable)).hexdigest() != package_manifest["package_identity"]:
        raise RuntimeError("Source package identity changed")
    if (hashlib.sha256(spec_raw).hexdigest() != package_manifest["spec_sha256"]
            or sha256_file(archive) != source_sha
            or source_sha != package_manifest["archive_sha256"]
            or source_commit != package_manifest["source_commit"]):
        raise RuntimeError("Source/spec/package binding changed")
    if (len(spec["arms"]) != 1
            or spec["arms"][0]["addons"][0]["name"] != VARIANT
            or spec["arms"][0]["initialization"]["seed"] != 42):
        raise RuntimeError("Frozen candidate binding changed")
    embedded_entry = source_root / "experiments" / EXPERIMENT / "run_candidate.py"
    if sha256_file(Path(__file__)) != sha256_file(embedded_entry):
        raise RuntimeError("Kaggle entry differs from committed source archive")

    dataset_manifest = find_one("manifest.json")
    if sha256_file(dataset_manifest) != MANIFEST_SHA256:
        raise RuntimeError("Accepted graph manifest changed")
    initial_state = find_one("initial_state.pt")
    if sha256_file(initial_state) != INITIAL_STATE_SHA256:
        raise RuntimeError("Frozen initial state changed")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    sys.path.insert(0, str(source_root / "src"))
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training

    output = Path("/kaggle/working") / EXPERIMENT
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "spec_identity": package_manifest["spec_identity"],
        "package_identity": package_manifest["package_identity"],
        "source_commit": source_commit,
        "source_archive_sha256": source_sha,
        "manifest_sha256": MANIFEST_SHA256,
        "initial_state_sha256": INITIAL_STATE_SHA256,
        "variant": VARIANT,
    }
    (output / "submission_binding.json").write_bytes(canonical(binding))
    print(f"Verified {EXPERIMENT} source/data/initial-state binding", flush=True)
    result = run_preflight(
        dataset_root=dataset_manifest.parent,
        manifest_path=dataset_manifest,
        source_archive=archive,
        source_archive_sha256=source_sha,
        source_commit=source_commit,
        output=output,
        platform_id="kaggle3-gptrans-centered-logits-100k",
        initial_state_path=initial_state,
        variant=VARIANT,
    )
    if result.get("accepted") is not True:
        raise RuntimeError("Frozen GPU runtime preflight rejected candidate")
    print(f"Runtime certificate: {result['runtime_certificate_id']}", flush=True)
    run_training(
        dataset_root=dataset_manifest.parent,
        manifest_path=dataset_manifest,
        preflight_path=output / "preflight.json",
        source_archive=archive,
        source_archive_sha256=source_sha,
        source_commit=source_commit,
        output=output,
        platform_id="kaggle3-gptrans-centered-logits-100k",
        initial_state_path=initial_state,
        variant=VARIANT,
    )
    print("Training reached terminal output; desktop acceptance remains separate", flush=True)


if __name__ == "__main__":
    main()
