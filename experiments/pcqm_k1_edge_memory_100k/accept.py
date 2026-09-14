"""Frozen saved-artifact acceptance; never construct or execute a model."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from molgap.constants import REPO_ROOT
from molgap.k1_edge_memory import MODES, PARAMETERS
from molgap.training_reproducibility import atomic_json, sha256_file


def accept(reference_root, candidate_root, source_commit, archive_sha256):
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_frozen_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    result = shared.accept(reference_root, candidate_root, modes=MODES,
        expected_parameters={mode: PARAMETERS for mode in MODES},
        initialization_policy="identical-tensors-altered-edge-dataflow")
    for mode in MODES:
        root = candidate_root / mode
        record = result["candidates"][mode]["record"]
        if record["source_commit"] != source_commit:
            raise RuntimeError("Source commit mismatch")
        checkpoint = torch.load(root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["source_archive_sha256"] != archive_sha256 or checkpoint["source_commit"] != source_commit:
            raise RuntimeError("Checkpoint source mismatch")
        if checkpoint["epoch"] != 39 or not all(key in checkpoint for key in
            ("optimizer", "scheduler", "rng_state", "best", "best_epoch", "best_artifact_sha256")):
            raise RuntimeError("Incomplete recoverable checkpoint")
        if not all(key in checkpoint["rng_state"] for key in ("python", "numpy", "torch", "cuda")):
            raise RuntimeError("RNG recovery incomplete")
        for name, digest in checkpoint["best_artifact_sha256"].items():
            if sha256_file(root / name) != digest:
                raise RuntimeError("Selected artifact mismatch")
        for name in ("best_model.pt", "last_checkpoint.pt"):
            state = torch.load(root / name, map_location="cpu", weights_only=False)
            state = state["model"] if name == "last_checkpoint.pt" else state
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise RuntimeError("Nonfinite saved model tensor")
        trace = json.loads((root / "trace.json").read_text())["epochs"]
        if [row["optimizer_steps"] for row in trace] != [781 * (epoch + 1) for epoch in range(40)]:
            raise RuntimeError("Optimizer trace mismatch")
        if not all((root / f"recovery_epoch_{epoch:02d}.tar").is_file() for epoch in (10,20,30,40)):
            raise RuntimeError("Recovery chunk missing")
    result["format"] = "molgap-k1-edge-memory-acceptance-v1"
    result["source_commit"] = source_commit
    result["source_archive_sha256"] = archive_sha256
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(args.reference_root, args.candidate_root, args.source_commit, args.archive_sha256))
