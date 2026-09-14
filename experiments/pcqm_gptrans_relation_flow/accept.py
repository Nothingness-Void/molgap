"""No-model-inference acceptance against the immutable GPTrans 100K reference."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from molgap.constants import REPO_ROOT
from molgap.screen_policy import validate_reference_screen_contract
from molgap.training_reproducibility import atomic_json, sha256_file

MODES = ("pair_prenorm", "centered_logits")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def accept(root, reference_root, submission_path, output, modes=MODES):
    spec = importlib.util.spec_from_file_location("gptrans_reference_accept", REPO_ROOT / "experiments/pcqm_gptrans_t_100k_v4/accept_result.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    submission = read(submission_path)
    base = read(reference_root / "frozen_reference.json")
    base_preflight = read(reference_root.parent / "preflight/preflight.json")
    require(sha256_file(reference_root / "best_model.pt") == "f4da386ae1e32f6953b645c0bdb8e208aaba1f1d7b8ebec132776bd63c22d6ab", "Frozen reference model changed")
    require(sha256_file(reference_root / "development_predictions.pt") == "4fa3d32f83b183503bfafeee33b7b44dc7ee5f396bef5b1b546c956669ce6d29", "Frozen reference predictions changed")
    require(sha256_file(reference_root / "frozen_reference.json") == "c523e000925ce3bfab35e01c3e8ca6e7ce27a7876d4cdf7c012d6ccd398436aa", "Frozen reference contract changed")
    base_payload = torch.load(reference_root / "development_predictions.pt", map_location="cpu", weights_only=False)
    base_error = (base_payload["prediction_eV"].view(-1).double() - base_payload["target_eV"].view(-1).double()).abs()
    arms = []
    for mode in modes:
        arm = root / mode
        training = arm / "training"
        mechanical = helper.accept(training, arm / "mechanical_acceptance.json")
        completion = read(training / "completion_manifest.json")
        preflight = read(arm / "preflight/preflight.json")
        checks = read(arm / "remote_checks.json")
        contract = read(training / "frozen_reference.json")
        require(completion.get("variant") == preflight.get("variant") == checks.get("variant") == mode, "Variant mismatch")
        require(checks.get("accepted") is True and checks.get("initial_state_matches_reference") is True, "Remote model checks")
        require(checks.get("parameters") == completion.get("parameters") == 5246817, "Parameter identity")
        require(completion.get("source_commit") == preflight.get("source_commit") == submission["source_commit"], "Source commit")
        require(completion.get("source_archive_sha256") == preflight.get("source_archive_sha256") == submission["archive_sha256"], "Source archive")
        require(completion.get("variant_source_sha256") == preflight.get("variant_source_sha256") == submission["variant_source_sha256"], "Variant source")
        require(sha256_file(training / "last_checkpoint.pt") == completion["checkpoint_sha256"], "Checkpoint hash")
        require(set(completion["checkpoint_chunks"]) == {f"checkpoint_epoch_{i:02d}.pt" for i in (9,19,29,39,49,59)}, "Checkpoint chunks")
        for name, sha in completion["checkpoint_chunks"].items():
            require(sha256_file(training / name) == sha, f"Chunk hash {name}")
        checkpoint = torch.load(training / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        require(checkpoint.get("variant") == mode and checkpoint.get("epoch") == 59, "Checkpoint cursor/variant")
        require(all(key in checkpoint for key in ("optimizer", "scheduler", "ema", "rng_state")), "Resume completeness")
        require(all(bool(torch.isfinite(t).all()) for t in checkpoint["model"].values()), "Finite checkpoint")
        best = torch.load(training / "best_model.pt", map_location="cpu", weights_only=False)
        require(best["model_config"].get("variant") == mode, "Self-contained model config")
        require(all(bool(torch.isfinite(t).all()) for t in best["model"].values()), "Finite model")
        comparison = validate_reference_screen_contract(reference=base, candidate=contract, runtime_certificates={base["runtime_certificate_id"]: base_preflight["runtime_certificate"], contract["runtime_certificate_id"]: preflight["runtime_certificate"]})
        payload = torch.load(training / "development_predictions.pt", map_location="cpu", weights_only=False)
        require(torch.equal(payload["source_idx"].view(-1), base_payload["source_idx"].view(-1)), "Paired source IDs")
        require(torch.equal(payload["target_eV"].view(-1), base_payload["target_eV"].view(-1)), "Paired labels")
        delta = ((payload["prediction_eV"].view(-1).double() - payload["target_eV"].view(-1).double()).abs() - base_error).numpy()
        rng = np.random.default_rng(42)
        means = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(2000)]
        upper = float(np.quantile(means, 0.975))
        gain = float(-delta.mean())
        arms.append({"variant": mode, **mechanical, "gain_eV": gain, "bootstrap_upper95_eV": upper, "mechanism_shortlist": gain >= 0.003 and upper < 0, "comparison": comparison, "parameters": 5246817, "preflight_cost": preflight["optimizer_step_calibration"]})
    result = {"accepted": True, "model_inference_executed": False, "arms": arms,
              "source_commit": submission["source_commit"], "official_validation_role_read": False,
              "test_dev_role_read": False, "test_challenge_role_read": False,
              "k1_superiority_established": False, "full_scale_authorized": False}
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(accept(args.root, args.reference, args.submission, args.output))
