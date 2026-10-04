"""Narrow source/input qualification for NO_TRAIN jobs outside training Spec."""
from __future__ import annotations

import ast
import hashlib
import json
import tarfile
from pathlib import Path

from .training_reproducibility import sha256_file

FORMAT = "molgap-frozen-inference-release-v1"


def check_frozen_inference_release(input_root: Path, entry: Path, metadata: Path):
    """Recheck retained executable/input bytes, not scientific outcome or GPU runtime."""
    root = input_root.resolve()
    release = json.loads((root / "audit_release.json").read_text())
    from .training_reproducibility import canonical_fingerprint
    if release.get("identity") != canonical_fingerprint({k: v for k, v in release.items() if k != "identity"}):
        raise ValueError("Release identity changed")
    if release["format"] != FORMAT or release["experiment_purpose"] != "NO_TRAIN":
        raise ValueError("Unsupported release contract")
    required = {"source.tar.gz", "SOURCE_FILES.json", "contract.json", "target_transform.json",
                "ema999_model.pt", "ema999_predictions.pt", "ema9999_model.pt", "ema9999_predictions.pt"}
    if set(release["files"]) != required:
        raise ValueError("Frozen inference input allowlist changed")
    for name, digest in release["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen inference input changed: {name}")
    if release["archive_sha256"] != release["files"]["source.tar.gz"] or release["contract_sha256"] != release["files"]["contract.json"]:
        raise ValueError("Source/contract release binding differs")
    if sha256_file(entry) != release["entry_sha256"] or sha256_file(metadata) != release["metadata_sha256"]:
        raise ValueError("Entrypoint/platform metadata changed")
    ast.parse(entry.read_text(encoding="utf-8"))
    contract = json.loads((root / "contract.json").read_text())
    if (contract["training_executed"] is not False or contract["physical_batch"] != 128
            or contract["precision"] != "fp32" or contract["tf32_enabled"] is not False
            or contract["allocation_cap_seconds"] != 5400
            or contract["roles"] != {"original_100k": [100000, 150000], "unseen_500k": [500000, 550000]}):
        raise ValueError("Audit execution scope changed")
    meta = json.loads(metadata.read_text())
    if (meta["dataset_sources"] != release["dataset_sources"] or meta["id"] != release["kernel"]
            or meta["enable_gpu"] is not True or meta["is_private"] is not True
            or meta.get("competition_sources") or meta.get("kernel_sources") or meta.get("model_sources")):
        raise ValueError("Mount/account/accelerator scope changed")
    inventory = json.loads((root / "SOURCE_FILES.json").read_text())
    if inventory["source_commit"] != release["source_commit"]:
        raise ValueError("Source commit identity changed")
    files = {row["path"]: row for row in inventory["files"]}
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError("Source inventory mismatch")
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or member.linkname:
                raise ValueError("Unsafe packaged source")
            payload = archive.extractfile(member).read()
            if len(payload) != files[member.name]["bytes"] or hashlib.sha256(payload).hexdigest() != files[member.name]["sha256"]:
                raise ValueError("Source member bytes changed")
            if member.name.endswith(".py"):
                ast.parse(payload)
    from .gptrans_portability import ARMS, MODEL_SOURCE, TRANSFORM_ASSET
    for arm, spec in ARMS.items():
        if release["files"][f"{arm}_model.pt"] != spec["model_sha256"] or release["files"][f"{arm}_predictions.pt"] != spec["payload_sha256"]:
            raise ValueError("Model/prediction identity differs from frozen adapter")
    if files["src/molgap/gptrans.py"]["sha256"] != MODEL_SOURCE:
        raise ValueError("Frozen architecture implementation changed")
    transform = json.loads((root / "target_transform.json").read_text())
    from .comparison_readiness import validate_target_transform_asset
    validate_target_transform_asset(transform)
    if transform["asset_sha256"] != TRANSFORM_ASSET:
        raise ValueError("Portable target transform identity changed")
    return dict(format=FORMAT, status="LOCAL_RELEASE_INPUTS_VERIFIED", release=release,
                inputs=dict(input_root=str(root), entry_script=str(entry.resolve()), metadata=str(metadata.resolve())),
                runtime_qualification="REMOTE_REPRODUCTION_REQUIRED", training_recipe="NOT_APPLICABLE")
