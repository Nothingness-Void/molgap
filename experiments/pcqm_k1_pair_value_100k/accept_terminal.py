"""Verify retained PairValue v2 artifacts without running model inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from molgap.research_memory.trace import canonicalize_trace, json_bytes


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/pcqm_k1_pair_value_100k"
RAW = ROOT / (
    "platforms/_records/kaggle/training/k1_pair_value_s42_v2/"
    "pcqm_k1_pair_value_100k/neural_atom_k1_pair_token_value_decoupled"
)
SOURCE = ROOT / "platforms/_records/kaggle/staging/pair_value_source_50a6db5_v2/source.tar.gz"
RUN_ID = "nothingnessvoid/molgap-pcqm-k1-pair-value-s42:v2"
TRAJECTORY_ID = "TC-k1-pair-value-decoupled-100k-s42"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    contract = read_json(EXPERIMENT / "training_contract.json")
    submission = read_json(EXPERIMENT / "results/submission_v2.json")
    completion = read_json(RAW / "completion_manifest.json")
    arm = read_json(RAW / "arm_record.json")
    preflight = read_json(RAW / "preflight.json")
    certificate = read_json(RAW / "runtime_certificate.json")
    source_config = read_json(EXPERIMENT / "source_config.json")
    trace_rows = read_json(RAW / "trace.json")["epochs"]
    checks: dict[str, bool] = {}

    def check(name: str, passed: bool) -> None:
        checks[name] = bool(passed)

    check("complete", completion.get("complete") is True and arm.get("complete") is True)
    required_artifacts = (
        "arm_record.json", "best_development_payload.pt", "best_model.pt",
        "last_checkpoint.pt", "preflight.json", "runtime_certificate.json",
        "runtime_manifest.json", "trace.json",
    )
    for name in required_artifacts:
        check(f"sha256:{name}", sha256(RAW / name) == completion["artifact_sha256"][name])
    check("source_archive_sha256", sha256(SOURCE) == submission["source_archive_sha256"])
    check("source_commit", arm["source_commit"] == source_config["source_commit"] == submission["source_commit"])
    check("source_archive_bound", arm["contract"]["source_archive_sha256"] == submission["source_archive_sha256"])
    check("dataset_manifest", arm["fixed_manifest_sha256"] == contract["manifest_sha256"] == arm["contract"]["data_role_fingerprint"])
    check("geometry_manifest", arm["fixed_geometry_sha256"] == contract["geometry_aggregate_sha256"])
    check("row_order", arm["contract"]["row_order_fingerprint"] == contract["row_order_sha256"])
    check("seed_precision_batch", arm["contract"]["seed"] == 42 and certificate["precision"] == "fp32" and certificate["tf32_enabled"] is False and certificate["deterministic_algorithms"] is True and certificate["physical_batch_per_device"] == 128)
    check("runtime_certificate", certificate["status"] == "accepted" and certificate["calibration_checks_passed"] is True and certificate["platform_id"] == "kaggle1")
    check("initial_function_and_trainability", preflight["exact_k1_function_at_initialization"] is True and preflight["candidate_mechanism_trainable_after_two_steps"] is True and preflight["mechanism_checks"]["value_projection_identity_initialized"] is True)
    check("memory_reserve", preflight["preflight_memory_reserve_fraction"] >= 0.15)
    check("parameter_count", preflight["parameter_count"] == arm["training"]["parameter_count"] == contract["architecture"]["parameters"])
    check("exposure", len(trace_rows) == contract["epochs"] == arm["training"]["epochs_completed"] and arm["training"]["optimizer_steps"] == contract["total_optimizer_steps"] and arm["training"]["sample_presentations"] == contract["total_sample_presentations"])
    check("trace_axes", all(row["epoch"] == index and row["optimizer_steps"] == (index + 1) * contract["optimizer_steps_per_epoch"] and row["sample_presentations"] == (index + 1) * contract["sample_presentations_per_epoch"] for index, row in enumerate(trace_rows)))
    check("trace_finite", all(all(math.isfinite(row[field]) for field in ("train_normalized_mae", "development_gap_mae_eV", "learning_rate", "seconds")) for row in trace_rows))
    best_epoch = min(range(len(trace_rows)), key=lambda index: trace_rows[index]["development_gap_mae_eV"])
    check("best_epoch", best_epoch == arm["training"]["best_epoch"])
    check("best_score", math.isclose(trace_rows[best_epoch]["development_gap_mae_eV"], arm["training"]["development_gap_mae_eV"], abs_tol=1e-9))
    check("protected_roles", all(arm[role] is False and completion[role] is False for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")))

    payload = torch.load(RAW / "best_development_payload.pt", map_location="cpu", weights_only=True)
    target, prediction, source_idx = (payload[name] for name in ("target_eV", "prediction_eV", "source_idx"))
    check("prediction_shapes", tuple(target.shape) == tuple(prediction.shape) == tuple(source_idx.shape) == (50000,))
    check("prediction_finite", bool(torch.isfinite(target).all() and torch.isfinite(prediction).all()))
    check("source_index_alignment", bool(torch.equal(source_idx, torch.arange(100000, 150000))))
    recomputed_mae = float(torch.mean(torch.abs(prediction.double() - target.double())))
    check("prediction_mae", math.isclose(recomputed_mae, arm["training"]["development_gap_mae_eV"], abs_tol=1e-6))

    model = torch.load(RAW / "best_model.pt", map_location="cpu", weights_only=True)
    check("best_model_parameter_count", sum(t.numel() for name, t in model.items() if not name.endswith(("running_mean", "running_var", "num_batches_tracked"))) == contract["architecture"]["parameters"])
    checkpoint = torch.load(RAW / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    check("checkpoint_identity", checkpoint["source_commit"] == submission["source_commit"] and checkpoint["source_archive_sha256"] == submission["source_archive_sha256"] and checkpoint["fixed_manifest_sha256"] == contract["manifest_sha256"] and checkpoint["row_order_fingerprint"] == contract["row_order_sha256"])
    check("checkpoint_exposure", checkpoint["epoch"] == 39 and checkpoint["best_epoch"] == best_epoch and checkpoint["trace"] == trace_rows)
    check("checkpoint_best_artifacts", all(checkpoint["best_artifact_sha256"][name] == sha256(RAW / name) for name in ("best_model.pt", "best_development_payload.pt")))
    check("optimizer_weight_decay", all(math.isclose(group["weight_decay"], contract["optimizer"]["weight_decay"]) for group in checkpoint["optimizer"]["param_groups"]))
    check("schedule_endpoint", math.isclose(checkpoint["scheduler"]["eta_min"], contract["schedule"]["eta_min"]))

    canonical = canonicalize_trace({
        "trajectory_id": TRAJECTORY_ID,
        "run_id": RUN_ID,
        "metric_semantics": {
            "live_train_metric": {"metric": "MAE", "unit": "normalized", "target": "PCQM4Mv2 Gap", "role_identity": "fixed100k-train", "weights": "live", "direction": "minimize"},
            "live_dev_metric": {"metric": "MAE", "unit": "eV", "target": "PCQM4Mv2 Gap", "role_identity": "fixed50k-development", "weights": "live", "direction": "minimize"},
            "ema_dev_metric": None,
        },
        "observations": [
            {"epoch_or_pass": row["epoch"], "optimizer_step": row["optimizer_steps"], "sample_presentations": row["sample_presentations"], "live_train_metric": row["train_normalized_mae"], "live_dev_metric": row["development_gap_mae_eV"], "learning_rate": row["learning_rate"], "wall_time_seconds": row["seconds"]}
            for row in trace_rows
        ],
        "provenance": {"sources": [{"path": str((RAW / "trace.json").relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(RAW / "trace.json")}]},
    })
    trace_path = EXPERIMENT / "results/canonical_trace_v2.json"
    trace_path.write_bytes(json_bytes(canonical))
    k1_ref = contract["supporting_controls"]["k1_v4_development_gap_mae_eV"]
    pair_ref = contract["supporting_controls"]["original_pair_token_development_gap_mae_eV"]
    result = {
        "format": "molgap-k1-pair-value-terminal-acceptance-v1",
        "trajectory_id": TRAJECTORY_ID,
        "run_id": RUN_ID,
        "kernel_status_at_retrieval": "COMPLETE",
        "kernel_version": 2,
        "checks": checks,
        "artifact_status": "verified" if all(checks.values()) else "failed",
        "mechanical_status": "resume_replay_unverified" if all(checks.values()) else "failed",
        "replay_limitations": ["frozen trajectory has no reference_ids", "no locally accepted aligned K1-v4 prediction payload", "deterministic checkpoint resume was not independently replayed"],
        "canonical_trace_ref": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
        "canonical_trace_sha256": sha256(trace_path),
        "best_epoch": best_epoch,
        "prediction_mae_eV": recomputed_mae,
        "reference_k1_mae_eV": k1_ref,
        "reference_pair_token_mae_eV": pair_ref,
        "gain_vs_k1_eV": k1_ref - recomputed_mae,
        "gain_vs_pair_token_eV": pair_ref - recomputed_mae,
        "point_gate_passed": k1_ref - recomputed_mae >= contract["materiality_rule"]["gain_vs_k1_eV"] and pair_ref - recomputed_mae >= contract["materiality_rule"]["gain_vs_original_pair_token_eV"],
        "paired_interval_status": "unavailable_reference_predictions",
        "protected_role_status": "untouched",
        "artifact_sha256": {name: sha256(RAW / name) for name in ("completion_manifest.json", *required_artifacts)},
        "replay_ready_files": ["completion_manifest.json", *required_artifacts, "source.tar.gz"],
        "excluded_redundant_outputs": ["recovery_epoch_10.tar", "recovery_epoch_20.tar", "recovery_epoch_30.tar", "recovery_epoch_40.tar"],
        "source_archive_sha256": sha256(SOURCE),
    }
    out = EXPERIMENT / "results/acceptance_v2.json"
    out.write_bytes(json_bytes(result))
    print(json.dumps({"artifact_status": result["artifact_status"], "mechanical_status": result["mechanical_status"], "failed_checks": [k for k, v in checks.items() if not v], "prediction_mae_eV": recomputed_mae, "point_gate_passed": result["point_gate_passed"]}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
