"""Inference-free V5 acceptance for the matched GPTrans 500K comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import assess_strict_comparison


EXPECTED_MANIFEST = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
EXPECTED_PARAMETERS = 5_246_817
EXPECTED_PRESENTATIONS = 29_998_080
MINIMUM_GAIN_EV = 0.003


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arm(root: Path, expected_mode: str) -> dict:
    import torch

    completion = json.loads((root / "completion_manifest.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    contract = json.loads((root / "scientific_contract.json").read_text())
    trace = json.loads((root / "trace.json").read_text())["epochs"]
    preflight = json.loads((root / "preflight.json").read_text())
    certificate = json.loads((root / "runtime_certificate.json").read_text())
    payload = torch.load(
        root / "best_predictions.pt", map_location="cpu", weights_only=False
    )
    if completion.get("complete") is not True or metrics.get("complete") is not True:
        raise RuntimeError(f"{expected_mode} is not complete")
    if metrics.get("mode") != expected_mode or contract.get("mode") != expected_mode:
        raise RuntimeError(f"{expected_mode} mode identity changed")
    for key, expected in {
        "data_role_fingerprint": EXPECTED_MANIFEST,
        "physical_batch_per_device": 128,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "precision": "fp32",
        "epochs": 60,
        "steps_per_epoch": 3906,
        "sample_exposure": EXPECTED_PRESENTATIONS,
    }.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"{expected_mode} contract changed: {key}")
    if metrics.get("parameter_count") != EXPECTED_PARAMETERS:
        raise RuntimeError(f"{expected_mode} parameter count changed")
    if preflight.get("memory_reserve_fraction", 0.0) < 0.15:
        raise RuntimeError(f"{expected_mode} memory preflight failed")
    if len(trace) != 60 or [row["epoch"] for row in trace] != list(range(60)):
        raise RuntimeError(f"{expected_mode} trace is incomplete")
    if trace[-1]["sample_presentations"] != EXPECTED_PRESENTATIONS:
        raise RuntimeError(f"{expected_mode} exposure changed")
    if not all(math.isfinite(row["development_mae_eV"]) for row in trace):
        raise RuntimeError(f"{expected_mode} trace is non-finite")
    expected_idx = torch.arange(500_000, 550_000, dtype=torch.long)
    prediction = payload["prediction"].view(-1)
    target = payload["target"].view(-1)
    source_idx = payload["source_idx"].view(-1).long()
    if not torch.equal(source_idx, expected_idx):
        raise RuntimeError(f"{expected_mode} source indices changed")
    if prediction.numel() != 50_000 or target.numel() != 50_000:
        raise RuntimeError(f"{expected_mode} payload rows changed")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise RuntimeError(f"{expected_mode} payload is non-finite")
    recomputed = float((prediction - target).abs().mean())
    if abs(recomputed - metrics["development_gap_mae_eV"]) > 1e-7:
        raise RuntimeError(f"{expected_mode} metric does not reproduce")
    for key in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if metrics.get(key) is not False:
            raise RuntimeError(f"{expected_mode} sealed role changed: {key}")
    for name, digest in completion["artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"{expected_mode} artifact hash mismatch: {name}")
    arm_contract = {
        **contract,
        "run_id": f"gptrans-500k-v5-{expected_mode}-seed42",
        "model_id": f"gptrans_t_12x256_pair32/{expected_mode}",
        "architecture_fingerprint": canonical_fingerprint(
            {"core": "gptrans_t_12x256_pair32", "mode": expected_mode}
        ),
        "source_archive_sha256": metrics["source_archive_sha256"],
        "result_artifact_sha256": sha256_file(root / "completion_manifest.json"),
        "platform_id": "scnet-kunshan",
        "accelerator": certificate["accelerator"],
        "runtime_certificate_id": metrics["runtime_certificate_id"],
        "execution_status": "COMPLETE",
        "scale_rows": 500_000,
    }
    return {
        "root": root,
        "metrics": metrics,
        "contract": arm_contract,
        "certificate": certificate,
        "prediction": prediction,
        "target": target,
        "source_idx": source_idx,
        "mae": recomputed,
        "initial_state_sha256": preflight["initial_state_sha256"],
        "artifact_hashes": completion["artifact_sha256"],
    }


def accept(
    reference_root: Path,
    candidate_root: Path,
    *,
    reference_elapsed_s: float,
    candidate_elapsed_s: float,
) -> dict:
    reference = _load_arm(reference_root, "reference")
    candidate = _load_arm(candidate_root, "pair_prenorm")
    if reference["initial_state_sha256"] != candidate["initial_state_sha256"]:
        raise RuntimeError("Reference and candidate initialization differ")
    if not np.array_equal(reference["source_idx"].numpy(), candidate["source_idx"].numpy()):
        raise RuntimeError("Paired source indices differ")
    if not np.array_equal(reference["target"].numpy(), candidate["target"].numpy()):
        raise RuntimeError("Paired targets differ")

    ref_error = np.abs(reference["prediction"].numpy() - reference["target"].numpy())
    cand_error = np.abs(candidate["prediction"].numpy() - candidate["target"].numpy())
    delta = cand_error - ref_error
    rng = np.random.default_rng(20260918)
    means = np.empty(5000, dtype=np.float64)
    for index in range(means.size):
        sample = rng.integers(0, delta.size, size=delta.size)
        means[index] = float(delta[sample].mean())
    lower, upper = np.quantile(means, [0.025, 0.975]).tolist()
    gain = reference["mae"] - candidate["mae"]
    positive = gain >= MINIMUM_GAIN_EV and upper < 0.0
    scientific_status = "POSITIVE" if positive else "NEGATIVE_UNDER_CONTRACT"
    paired = {
        "rows": 50_000,
        "reference_mae_eV": reference["mae"],
        "candidate_mae_eV": candidate["mae"],
        "candidate_minus_reference_eV": float(delta.mean()),
        "gain_eV": gain,
        "minimum_gain_eV": MINIMUM_GAIN_EV,
        "material_gate_passed": gain >= MINIMUM_GAIN_EV,
    }
    bootstrap = {
        "method": "paired-row-bootstrap",
        "seed": 20260918,
        "replicates": 5000,
        "lower95_eV": lower,
        "upper95_eV": upper,
        "favorable": upper < 0.0,
        "does_not_measure_training_randomness": True,
    }
    certificates = {
        reference["contract"]["runtime_certificate_id"]: reference["certificate"],
        candidate["contract"]["runtime_certificate_id"]: candidate["certificate"],
    }
    artifacts = {
        f"reference/{name}": digest
        for name, digest in reference["artifact_hashes"].items()
    }
    artifacts.update(
        {
            f"pair_prenorm/{name}": digest
            for name, digest in candidate["artifact_hashes"].items()
        }
    )
    native_cost = {
        "unit": "DCU-hour",
        "reference": reference_elapsed_s / 3600.0,
        "pair_prenorm": candidate_elapsed_s / 3600.0,
        "total": (reference_elapsed_s + candidate_elapsed_s) / 3600.0,
        "source": "Slurm allocated elapsed seconds",
    }
    budget_decision = (
        "WITHIN_SERVER_BUDGET"
        if positive and native_cost["total"] <= 32.0
        else "STOP_FOR_COST"
    )
    outcome = assess_strict_comparison(
        reference_contract=reference["contract"],
        candidate_contract=candidate["contract"],
        runtime_certificates=certificates,
        reference_bundle={
            "prediction": reference["prediction"],
            "target": reference["target"],
            "source_idx": reference["source_idx"],
        },
        candidate_bundle={
            "prediction": candidate["prediction"],
            "target": candidate["target"],
            "source_idx": candidate["source_idx"],
        },
        paired_analysis=paired,
        bootstrap=bootstrap,
        artifact_hashes=artifacts,
        resume_cursor={"reference_next_epoch": 60, "candidate_next_epoch": 60},
        role_history={
            "train": True,
            "development": True,
            "official_validation": False,
            "test_dev": False,
            "test_challenge": False,
        },
        actual_native_cost=native_cost,
        scientific_status=scientific_status,
        budget_decision=budget_decision,
    )
    return {
        "format": "molgap-gptrans-prenorm-500k-v5-acceptance-v1",
        "model_inference_executed": False,
        "v5_outcome": outcome,
        "paired_analysis": paired,
        "paired_bootstrap": bootstrap,
        "native_cost": native_cost,
        "initial_state_sha256": reference["initial_state_sha256"],
        "reference_model_sha256": reference["metrics"]["best_model_sha256"],
        "candidate_model_sha256": candidate["metrics"]["best_model_sha256"],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--reference-elapsed-s", type=float, required=True)
    parser.add_argument("--candidate-elapsed-s", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(
        args.reference_root.resolve(),
        args.candidate_root.resolve(),
        reference_elapsed_s=args.reference_elapsed_s,
        candidate_elapsed_s=args.candidate_elapsed_s,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
