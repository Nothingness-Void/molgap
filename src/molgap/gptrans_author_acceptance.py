"""Saved-artifact verification of CPU author inputs, never GPU admission."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .gptrans_author_inputs import (
    DEGREE_SCALE, INITIAL_MODEL_SHA256, INITIAL_STATE_SHA256,
    INITIAL_SCALED_FORMAT, state_digest, validate_sidecar,
)
from .gptrans_author_variants import DEGREE_INITIAL_SHA256
from .training_reproducibility import sha256_file

NO_READ = ("training_executed", "model_inference_executed", "labels_read",
           "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")


def accept_prepared_inputs(output: Path, staged_package: Path) -> dict:
    """Check retrieved source, complete paths, exact tensor intervention and cost."""
    import torch

    output, staged_package = Path(output), Path(staged_package)
    release = json.loads((staged_package / "release.json").read_text())
    if release["status"] != "LOCAL_RELEASE_INPUTS_VERIFIED" or release["errors"]:
        raise ValueError("CPU source package was not qualified")
    startup = json.loads((output / "startup.json").read_text())
    result = json.loads((output / "preparation_result.json").read_text())
    cost = json.loads((output / "native_cost.json").read_text())
    if (startup["source_archive_sha256"] != release["source_archive_sha256"]
            or startup["source_commit"] != release["source_commit"]
            or startup["manifest_sha256"] != "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
            or startup["initial_state_file_sha256"] != INITIAL_STATE_SHA256):
        raise ValueError("CPU startup does not bind the released input/source identities")
    for record in (startup, result):
        if any(record.get(field) is not False for field in NO_READ):
            raise ValueError("CPU preparation used a forbidden model/label/evaluation role")
    if (cost.get("status") != "COMPLETE" or cost.get("allocated_gpu_count") != 0
            or (output / "failure.json").exists()):
        raise ValueError("CPU preparation was not completely successful")
    for key in ("wall_seconds", "process_cpu_seconds"):
        value = cost.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing/nonfinite CPU cost")
    if cost["wall_seconds"] > 3 * 3600:
        raise ValueError("CPU wall budget exceeded")
    inventory = json.loads((staged_package / "source/SOURCE_FILES.json").read_text())
    source = (output / "verified_source").resolve()
    for item in inventory["files"]:
        member = (source / item["path"]).resolve()
        if not member.is_relative_to(source) or sha256_file(member) != item["sha256"]:
            raise ValueError("Retrieved CPU implementation differs from the release")

    paths = validate_sidecar(output / "paths")
    expected = result["path_acceptance"]
    if expected.get("source_rederived") is not True or paths["runtime_compact_cache"]["within_cap"] is not True:
        raise ValueError("Independent CPU path content/cap verification is absent")
    for key, value in paths.items():
        if key != "source_rederived" and expected.get(key) != value:
            raise ValueError(f"Saved path acceptance changed: {key}")

    base_path = staged_package / "inputs/initial_state.pt"
    if sha256_file(base_path) != INITIAL_STATE_SHA256:
        raise ValueError("Retained base initialization changed")
    base = torch.load(base_path, map_location="cpu", weights_only=True)["model_state"]
    prepared_path = output / "degree_initial_state.pt"
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=True)
    state = prepared["model_state"]
    if (prepared["format"] != INITIAL_SCALED_FORMAT or set(state) != set(base)
            or state_digest(base) != INITIAL_MODEL_SHA256
            or state_digest(state) != DEGREE_INITIAL_SHA256
            or prepared["state_sha256"] != DEGREE_INITIAL_SHA256):
        raise ValueError("Prepared degree initialization has wrong tensor identity")
    for name, tensor in base.items():
        wanted = tensor * DEGREE_SCALE if name in ("in_degree_encoder.weight", "out_degree_encoder.weight") else tensor
        actual = state[name]
        if (actual.dtype != wanted.dtype or actual.shape != wanted.shape
                or not bool(torch.isfinite(actual).all()) or not torch.equal(actual, wanted)):
            raise ValueError(f"Unexpected initial tensor modification: {name}")
    if sha256_file(prepared_path) != result["degree_initialization"]["output_sha256"]:
        raise ValueError("Prepared initialization file changed")
    return {
        "format": "molgap-gptrans-author-inputs-acceptance-v1", "accepted": True,
        "source_commit": release["source_commit"], "source_archive_sha256": release["source_archive_sha256"],
        "package_identity": release["package_identity"], "path_acceptance": paths,
        "cpu_source_rederivation_verified": True,
        "degree_initial_state_sha256": DEGREE_INITIAL_SHA256,
        "degree_initial_file_sha256": sha256_file(prepared_path),
        "native_cost": cost, "gpu_compute_released": False,
        **{field: False for field in NO_READ},
    }
