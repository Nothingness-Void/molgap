"""Mechanical artifact acceptance for a completed full GPTrans-T run."""
from __future__ import annotations

import json
from pathlib import Path

from .pcqm_gptrans_full_runner import (
    EXPECTED_PARAMETERS,
    EXPECTED_INITIAL_SHA256,
    MINIMUM_LEARNING_RATE,
    MODEL_ID,
    SAMPLE_PRESENTATIONS,
    SEED,
    TOTAL_STEPS,
    TRAINING_CONTRACT,
    TRAINING_CONTRACT_SHA256,
)
from .pcqm_k1_full_runner import (
    BATCHES_PER_PASS,
    FULL_TOPOLOGY_AGGREGATE_SHA256,
    PHYSICAL_BATCH,
)
from .training_reproducibility import (
    CHECKPOINT_FORMAT,
    MODEL_BUNDLE_FORMAT,
    assert_finite_state_dict,
    atomic_json,
    canonical_fingerprint,
    sha256_file,
)


def _state_dict_sha256(state_dict) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def accept(root: Path) -> dict:
    import torch

    root = root.resolve()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads((root / "training_contract.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "runtime_manifest.json").read_text(encoding="utf-8"))
    certificate = json.loads(
        (root / "runtime_certificate.json").read_text(encoding="utf-8")
    )
    preflight = json.loads(
        (root.parent / "preflight" / "preflight.json").read_text(encoding="utf-8")
    )

    if contract != TRAINING_CONTRACT or canonical_fingerprint(contract) != TRAINING_CONTRACT_SHA256:
        raise RuntimeError("GPTrans-T full training contract changed")
    if summary != completion:
        raise RuntimeError("Run summary and completion manifest differ")
    for payload_name, payload in (("summary", summary), ("completion", completion)):
        required = {
            "format": "molgap-pcqm-gptrans-t-full-result-v1",
            "complete": True,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "global_step": TOTAL_STEPS,
            "sample_presentations": SAMPLE_PRESENTATIONS,
            "train_rows": TRAINING_CONTRACT["train_rows"],
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"{payload_name} completion identity changed")
    if not preflight.get("accepted") or preflight.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256:
        raise RuntimeError("Accepted matching preflight is missing")
    certificate_id = canonical_fingerprint(certificate)
    if certificate_id != preflight.get("runtime_certificate_id"):
        raise RuntimeError("Preflight/runtime certificate identity changed")
    if certificate.get("status") != "accepted" or certificate.get("accelerator", "").find("A100") < 0:
        raise RuntimeError("The accepted A100 preflight is missing")
    if runtime.get("runtime_fingerprint") != canonical_fingerprint(
        {key: value for key, value in runtime.items() if key != "runtime_fingerprint"}
    ):
        raise RuntimeError("Runtime manifest fingerprint changed")
    if runtime.get("runtime_fingerprint") != summary.get("runtime_fingerprint"):
        raise RuntimeError("Training runtime differs from preflight runtime")
    external_ogb = runtime.get("external_runtime_artifacts", {}).get("ogb", {})
    if external_ogb != TRAINING_CONTRACT["ogb_runtime"]:
        raise RuntimeError("Vendored OGB runtime identity changed")
    if certificate.get("runtime_fingerprint") != runtime.get("runtime_fingerprint"):
        raise RuntimeError("Accepted certificate and runtime differ")
    if certificate.get("initial_model_sha256") != EXPECTED_INITIAL_SHA256:
        raise RuntimeError("Accepted initial model hash changed")
    if summary.get("runtime_certificate_id") != certificate_id:
        raise RuntimeError("Training certificate identity changed")

    checkpoint_path = root / "last_checkpoint.pt"
    bundle_path = root / "model_bundle.pt"
    if sha256_file(checkpoint_path) != summary.get("checkpoint_sha256"):
        raise RuntimeError("Final checkpoint SHA changed")
    if sha256_file(bundle_path) != summary.get("model_bundle_sha256"):
        raise RuntimeError("Final model bundle SHA changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    for key, expected in {
        "format": "molgap-pcqm-gptrans-t-full-checkpoint-v1",
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "global_step": TOTAL_STEPS,
        "source_commit": summary.get("source_commit"),
        "source_archive_sha256": summary.get("source_archive_sha256"),
        "initial_model_sha256": EXPECTED_INITIAL_SHA256,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"Final checkpoint changed: {key}")
    if (checkpoint.get("pass_index"), checkpoint.get("next_batch_in_pass")) != divmod(
        TOTAL_STEPS, BATCHES_PER_PASS
    ):
        raise RuntimeError("Final checkpoint cursor changed")
    optimizer_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
    if len(optimizer_groups) != 1 or abs(
        float(optimizer_groups[0]["lr"]) - MINIMUM_LEARNING_RATE
    ) > 1e-12:
        raise RuntimeError("Final learning rate changed")
    assert_finite_state_dict(checkpoint["model"], label="final model state")
    assert_finite_state_dict(checkpoint["ema"], label="final EMA state")
    assert_finite_state_dict(bundle["state_dict"], label="final model bundle")
    if bundle.get("format") != MODEL_BUNDLE_FORMAT or bundle.get("model_id") != MODEL_ID:
        raise RuntimeError("Model bundle identity changed")
    if bundle.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256:
        raise RuntimeError("Model bundle contract changed")
    if bundle.get("training_global_step") != TOTAL_STEPS:
        raise RuntimeError("Model bundle step count changed")
    if bundle.get("target_stats") != summary.get("target_stats"):
        raise RuntimeError("Model bundle target statistics changed")
    parameter_count = sum(value.numel() for value in bundle["state_dict"].values())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError("Model bundle parameter count changed")
    if _state_dict_sha256(bundle["state_dict"]) != summary.get("final_state_sha256"):
        raise RuntimeError("Final model-state SHA changed")
    if completion.get("source_archive_sha256") != summary.get("source_archive_sha256"):
        raise RuntimeError("Source archive identity changed")
    if completion.get("cache_aggregate_sha256") != FULL_TOPOLOGY_AGGREGATE_SHA256:
        raise RuntimeError("Completion is missing the cache identity")

    acceptance = {
        "format": "molgap-pcqm-gptrans-t-full-acceptance-v1",
        "accepted": True,
        "model_inference_executed": False,
        "model_id": MODEL_ID,
        "parameter_count": EXPECTED_PARAMETERS,
        "initial_state_sha256": EXPECTED_INITIAL_SHA256,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "source_commit": summary["source_commit"],
        "source_archive_sha256": summary["source_archive_sha256"],
        "global_step": TOTAL_STEPS,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "model_bundle_sha256": summary["model_bundle_sha256"],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(root / "acceptance.json", acceptance)
    return acceptance


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(accept(args.root), flush=True)


if __name__ == "__main__":
    main()
