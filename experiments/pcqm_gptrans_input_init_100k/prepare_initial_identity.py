"""Verify the frozen tensor asset and derive the candidate initialization hash.

This is an asset identity check. It does not construct or execute a model and
does not read graph targets or development rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.gptrans_input_init import (
    INPUT_EMBEDDING_KEYS,
    normal002_input_embedding_state,
)
from molgap.v4_runtime import sha256_file, state_dict_sha256, torch_load_compat


REFERENCE_ARTIFACT_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"
REFERENCE_STATE_SHA256 = "8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6"
REFERENCE_FORMAT = "molgap-gptrans-t-seed42-initial-state-v1"


def derive(reference_path: Path) -> dict:
    artifact_sha256 = sha256_file(reference_path)
    if artifact_sha256 != REFERENCE_ARTIFACT_SHA256:
        raise RuntimeError("Frozen reference initial-state artifact changed")
    payload = torch_load_compat(reference_path, map_location="cpu", weights_only=False)
    if payload.get("format") != REFERENCE_FORMAT:
        raise RuntimeError("Frozen reference initial-state format changed")
    reference = payload["model_state"]
    reference_sha256 = state_dict_sha256(reference)
    if payload.get("state_sha256") != reference_sha256 or reference_sha256 != REFERENCE_STATE_SHA256:
        raise RuntimeError("Frozen reference tensor state changed")
    candidate = normal002_input_embedding_state(
        reference, expected_reference_sha256=REFERENCE_STATE_SHA256
    )
    return {
        "format": "molgap-gptrans-input-initialization-identity-v1",
        "reference_artifact_sha256": artifact_sha256,
        "reference_state_sha256": reference_sha256,
        "candidate_state_sha256": state_dict_sha256(candidate),
        "seed": 42,
        "normal_std": 0.02,
        "changed_tensor_keys": list(INPUT_EMBEDDING_KEYS),
        "changed_tensor_count": len(INPUT_EMBEDDING_KEYS),
        "parameter_count": sum(value.numel() for value in candidate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-state", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.reference_state), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
