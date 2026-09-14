"""Saved-artifact acceptance for K1 edge/slot pairwise interactions."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.k1_edge_slot_interaction import MODES, PARAMETERS
from molgap.training_reproducibility import atomic_json, sha256_file


SOURCE_COMMIT = "0c2d086e9d9568228a4f5604e2cb6bc07d33132b"
SOURCE_ARCHIVE_SHA256 = "7dd55503f91dd44ff9a956300303cc502a02f7d9e07c502649593078ae94693b"
EXPECTED_SHARED_INITIAL_STATE_SHA256 = {
    # Removing the redundant slot-attention module also removes its tensors.
    # This is the accepted shared-K1 hash with those tensor names omitted.
    MODES[0]: "325c5c7c65b3dcf5072fb074f20e1d58a19128cdf6145c57b3db870f7cc78805",
    # Uniform return retains the complete K1 parameter set.
    MODES[1]: "ce878c7a6e71519e4bd25ef10e45e6c9f693687f7f8c76dbf37c66f3c5708dc3",
}


def accept(reference_root, candidate_root):
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_frozen_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    result = shared.accept(
        reference_root,
        candidate_root,
        modes=MODES,
        expected_parameters=PARAMETERS,
        initialization_policy="identical-tensors-altered-edge-dataflow",
        expected_shared_initial_state_sha256=EXPECTED_SHARED_INITIAL_STATE_SHA256,
    )
    for mode in MODES:
        root = candidate_root / mode
        record = result["candidates"][mode]["record"]
        checks = record["preflight"]["mechanism_checks"]
        expected = "no-slot-attention" if mode == MODES[0] else "uniform-return"
        if checks.get("slot_mode") != expected or len(checks.get("slot_layers", [])) != 3:
            raise RuntimeError(f"Slot interaction checks changed: {mode}")
        if not all(row.get("invariant") is True for row in checks["slot_layers"]):
            raise RuntimeError(f"Slot invariant failed: {mode}")
        if record.get("source_commit") != SOURCE_COMMIT:
            raise RuntimeError(f"Source commit mismatch: {mode}")
        checkpoint = torch.load(
            root / "last_checkpoint.pt", map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("source_commit") != SOURCE_COMMIT
            or checkpoint.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        ):
            raise RuntimeError(f"Checkpoint source mismatch: {mode}")
        required = (
            "optimizer", "scheduler", "rng_state", "best", "best_epoch",
            "best_artifact_sha256",
        )
        if checkpoint.get("epoch") != 39 or not all(key in checkpoint for key in required):
            raise RuntimeError(f"Incomplete recoverable checkpoint: {mode}")
        if not all(
            key in checkpoint["rng_state"] for key in ("python", "numpy", "torch", "cuda")
        ):
            raise RuntimeError(f"RNG recovery incomplete: {mode}")
        for name, digest in checkpoint["best_artifact_sha256"].items():
            if sha256_file(root / name) != digest:
                raise RuntimeError(f"Selected artifact mismatch: {mode}/{name}")
        for name in ("best_model.pt", "last_checkpoint.pt"):
            saved = torch.load(root / name, map_location="cpu", weights_only=False)
            state = saved["model"] if name == "last_checkpoint.pt" else saved
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise RuntimeError(f"Nonfinite saved model tensor: {mode}/{name}")
        trace = json.loads((root / "trace.json").read_text(encoding="utf-8"))["epochs"]
        if [row["optimizer_steps"] for row in trace] != [
            781 * (epoch + 1) for epoch in range(40)
        ]:
            raise RuntimeError(f"Optimizer trace mismatch: {mode}")
        if not all(
            (root / f"recovery_epoch_{epoch:02d}.tar").is_file()
            for epoch in (10, 20, 30, 40)
        ):
            raise RuntimeError(f"Recovery chunk missing: {mode}")
    result["format"] = "molgap-k1-edge-slot-interaction-acceptance-v1"
    result["scientific_round"] = 2
    result["source_commit"] = SOURCE_COMMIT
    result["source_archive_sha256"] = SOURCE_ARCHIVE_SHA256
    result["expected_shared_initial_state_sha256"] = EXPECTED_SHARED_INITIAL_STATE_SHA256
    result["model_inference_executed"] = False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(args.reference_root, args.candidate_root))
