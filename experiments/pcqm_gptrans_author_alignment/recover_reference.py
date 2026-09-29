"""Recover the retained GPTrans V4 run as a V5 reference without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from molgap.comparison_readiness import target_transform_asset_digest
from molgap.training_reproducibility import atomic_json, sha256_file


HERE = Path("experiments/pcqm_gptrans_author_alignment")
OUT = HERE / "recovered_reference"
DATA = Path("experiments/v5_legacy_evidence_migration/k1_v4_100k_reference")
HIST = Path("experiments/pcqm_gptrans_t_100k_v4")
RUN = "kaseichou/molgap-pcqm-gptrans-t-v4-s42:v4"
TRAJECTORY = "TC-gptrans-t-v4-100k-reference-s42"
REFERENCE = "pcqm-gptrans-t-v4-100k-reference-s42"
BUNDLE = "reference-gptrans-t-v4-100k-s42-v5-recovered"
MANIFEST_SHA = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
RECORD_REL = Path("platforms/_records/kaggle/training/pcqm_gptrans_t_v4_s42_v4/pcqm_gptrans_t_100k_v4")


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def must(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def recover(root: Path) -> None:
    out = root / OUT
    record = root / RECORD_REL
    train = record / "training"
    pre = record / "preflight"
    contract = read(root / HIST / "training_contract.json")
    accepted = read(root / HIST / "results/kaggle_acceptance.json")
    completion = read(train / "completion_manifest.json")
    frozen = read(train / "frozen_reference.json")
    preflight = read(pre / "preflight.json")
    trace = read(train / "trace.json")
    row = read(root / DATA / "row_manifest.json")
    target = read(root / DATA / "target_manifest.json")
    fixed_manifest = root / "platforms/_records/kaggle/gptrans_real_path_preflight/fixed_manifest/manifest.json"
    p0_acceptance = read(root / HERE / "results/acceptance.json")

    must(sha256_file(fixed_manifest) == MANIFEST_SHA, "fixed manifest identity")
    must(p0_acceptance.get("accepted") is True, "P0 acceptance")
    must(row["dataset_identity"].endswith(MANIFEST_SHA), "row manifest identity")
    must(target["train_rows"] == 100_000 and target["development_rows"] == 50_000, "target rows")
    must(contract["manifest_sha256"] == completion["manifest_sha256"] == MANIFEST_SHA, "run dataset identity")
    must(completion["complete"] is True and accepted["accepted"] is True, "historical acceptance")
    must(completion["epochs"] == contract["epochs"] == 60, "epochs")
    must(completion["optimizer_steps"] == 46_860, "optimizer steps")
    must(completion["sample_presentations"] == contract["sample_exposure"] == 5_998_080, "exposure")
    must(completion["best_epoch"] == frozen["best_epoch"] == 59, "best epoch")
    must(all(completion[k] is False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")), "protected role")
    artifacts = {
        "best_model.pt": completion["best_model_sha256"],
        "development_predictions.pt": completion["development_predictions_sha256"],
        "frozen_reference.json": completion["frozen_reference_sha256"],
    }
    for name, digest in artifacts.items():
        must(sha256_file(train / name) == digest == accepted[{"best_model.pt": "best_model_sha256", "development_predictions.pt": "development_predictions_sha256", "frozen_reference.json": "frozen_reference_sha256"}[name]], f"{name} SHA")
    checkpoint_sha = sha256_file(train / "last_checkpoint.pt")
    must(frozen["result_artifact_sha256"] == artifacts["best_model.pt"], "frozen model")
    must(frozen["source_archive_sha256"] == completion["source_archive_sha256"], "source archive")
    must(frozen["architecture_fingerprint"] == contract["architecture_sha256"], "architecture")
    must(frozen["physical_batch_per_device"] == contract["physical_batch_per_device"] == 128, "physical batch")
    must(frozen["gradient_accumulation_steps"] == contract["gradient_accumulation_steps"] == 1, "accumulation")
    must(frozen["tail_batch_policy"] == contract["tail_batch_policy"] == "drop_last", "tail policy")
    must(preflight["accepted"] is True and preflight["runtime_certificate_id"] == completion["runtime_certificate_id"], "runtime preflight")
    runtime = preflight["runtime_certificate"]
    must(runtime["status"] == "accepted" and runtime["precision"] == "fp32" and runtime["tf32_enabled"] is False, "runtime precision")
    must(runtime["physical_batch_per_device"] == 128 and runtime["calibration_checks_passed"] is True, "runtime calibration")

    best = torch.load(train / "best_model.pt", map_location="cpu", weights_only=False)
    predictions = torch.load(train / "development_predictions.pt", map_location="cpu", weights_only=False)
    pred = predictions["prediction_eV"].view(-1).float()
    truth = predictions["target_eV"].view(-1).float()
    indices = predictions["source_idx"].view(-1).long()
    must(pred.numel() == truth.numel() == indices.numel() == 50_000, "prediction count")
    must(torch.equal(indices, torch.arange(100_000, 150_000)), "prediction source order")
    must(bool(torch.isfinite(pred).all() and torch.isfinite(truth).all()), "finite predictions")
    must(tensor_hash(truth) == target["development_target_sha256"], "development target identity")
    must(tensor_hash(indices) == row["development_source_idx_sha256"], "development row identity")
    mae = float((pred.double() - truth.double()).abs().mean())
    must(abs(mae - float(accepted["best_development_mae_eV"])) < 1e-10, "accepted MAE")
    must(abs(mae - float(best["development_mae_eV"])) < 1e-7, "checkpoint MAE")
    must(best["epoch"] == 59 and best["manifest_sha256"] == MANIFEST_SHA, "best model metadata")
    must(best["target_stats"] == preflight["target_stats"], "training target statistics")
    rows = trace["rows"]
    must(len(rows) == 60 and [r["epoch"] for r in rows] == list(range(60)), "trace epochs")
    must(all(r["optimizer_steps"] == 781 and r["sample_presentations"] == 99_968 for r in rows), "trace exposure")
    must(abs(rows[-1]["development_mae_eV"] - float(best["development_mae_eV"])) < 1e-12, "last trace MAE")

    def ref(name: str) -> str:
        return (OUT / name).as_posix()

    output: dict[str, dict] = {}
    output["runtime_certificate.json"] = runtime
    stats = preflight["target_stats"]
    transform = {
        "format": "molgap-target-transform-asset-v1",
        "asset_id": contract["target_transform_fingerprint"],
        "target_identity": target["target_identity"],
        "mean": stats["mean_eV"],
        "std": stats["sample_std_eV"],
        "ddof": 1,
        "variance_convention": "sample-standard-deviation-over-fixed-100k-train-role-double-reduction-then-float32-model",
        "source_row_manifest_sha256": sha256_file(root / DATA / "row_manifest.json"),
        "target_sha256": target["train_target_sha256"],
    }
    transform["asset_sha256"] = target_transform_asset_digest(transform)
    output["target_transform.json"] = transform
    prediction_manifest = {
        "format": "molgap-aligned-prediction-manifest-v1",
        "artifact_locator": f"external://local-platform-record/{RECORD_REL.as_posix()}/training/development_predictions.pt",
        "artifact_sha256": artifacts["development_predictions.pt"],
        "prediction_sha256": tensor_hash(pred),
        "source_idx_sha256": tensor_hash(indices),
        "target_sha256": tensor_hash(truth),
        "ordering_semantics": "source_idx ascending",
        "evaluation_role_identity": "pcqm4mv2-ogb-fixed-100k-v1:internal-development-100000-150000",
        "row_count": 50_000,
        "unique_source_idx": int(indices.unique().numel()),
        "development_gap_mae_eV": mae,
        "tensor_hash_semantics": "sha256-contiguous-c-order-bytes",
    }
    output["prediction_manifest.json"] = prediction_manifest
    run_id = RUN
    evidence_ref = ref("v5_evidence.json")
    role_names = {
        "train": ("official_train_prefix_0_100000", "training_membership", False),
        "prediction-input": ("internal_development_100000_150000", "prediction_input", False),
        "labels-read": ("internal_development_100000_150000", "labels_read", False),
        "metric-computed": ("internal_development_100000_150000", "metric_computed", False),
        "selection-used": ("internal_development_100000_150000", "selection_used", True),
    }
    for name, (role_name, access_kind, selected) in role_names.items():
        output[f"roles/{name}.json"] = {
            "schema": "molgap-role-event-v1", "role_event_id": f"role-{TRAJECTORY}-{name}",
            "trajectory_id": TRAJECTORY, "action_id": "A001", "run_id": run_id,
            "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1", "row_manifest_hash": MANIFEST_SHA,
            "role_name": role_name, "access_kind": access_kind,
            "selection_used": selected, "evidence_ref": evidence_ref,
        }
    cost_id = f"cost-{TRAJECTORY}"
    output[f"costs/{cost_id}.json"] = {
        "schema": "molgap-cost-event-v1", "cost_event_id": cost_id,
        "trajectory_id": TRAJECTORY, "action_id": "A001", "run_id": run_id,
        "attempt_id": "reference-v4", "category": "training", "platform": "kaggle2",
        "hardware": "Tesla_P100_PCIE_16GB",
        "measurement": {key: {"value": None, "status": "measurement_missing"} for key in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")},
        "evidence_ref": evidence_ref,
    }
    identity = {
        "benchmark_identity": "pcqm4mv2-ogb-fixed-100k-gap-v4",
        "dataset_identity": row["dataset_identity"],
        "data_role_identity": contract["data_role_fingerprint"],
        "row_membership_identity": MANIFEST_SHA,
        "row_order_identity": contract["row_order_fingerprint"],
        "feature_identity": contract["feature_fingerprint"],
        "target_identity": target["target_identity"],
        "seed": 42, "precision": "fp32", "tf32_enabled": False,
        "deterministic_algorithms": True, "physical_batch_per_device": 128,
        "gradient_accumulation_steps": 1, "tail_batch_policy": "drop_last",
        "optimizer_identity": contract["optimizer_fingerprint"],
        "optimizer_mode": "adamw-single-parameter-group", "optimizer_fused": False,
        "schedule_identity": contract["schedule_fingerprint"],
        "loss_identity": contract["loss_fingerprint"],
        "target_transform_identity": transform["asset_id"],
        "target_transform_asset_sha256": transform["asset_sha256"],
        "sample_presentations": 5_998_080, "optimizer_steps": 46_860,
        "checkpoint_selection_identity": contract["selection_fingerprint"],
        "evaluation_role_identity": prediction_manifest["evaluation_role_identity"],
        "selection_role_identity": prediction_manifest["evaluation_role_identity"],
        "architecture_config_identity": contract["architecture_sha256"],
        "runtime_certificate_scope": "single-device-fp32-no-tf32-bs128-optimizer-inclusive",
        "weight_semantics": "ema", "ema_enabled": True, "ema_decay": 0.9999,
        "ema_update_frequency": "each_optimizer_step", "evaluation_weight_source": "ema",
    }
    output["trace_manifest.json"] = {
        "schema": "molgap-trace-manifest-v1", "trajectory_id": TRAJECTORY,
        "run_id": run_id, "contract_ref": (HIST / "training_contract.json").as_posix(),
        "model_identity": contract["architecture_sha256"], "reference_id": REFERENCE,
        "comparison_role": "reference", "x_axis": "optimizer_steps",
        "presentation_semantics_ref": (HIST / "training_contract.json").as_posix(),
        "weight_semantics": "ema", "metric_semantics": "internal_development_ema_gap_mae_eV",
        "evaluation_role_identity": prediction_manifest["evaluation_role_identity"],
        "selection_semantics": "best_internal_development_ema_over_60_epochs",
        "trace_artifact_ref": f"external://local-platform-record/{RECORD_REL.as_posix()}/training/trace.json",
        "terminal_evidence_ref": evidence_ref,
        "comparability_identity": {
            "scientific_contract": "pcqm4mv2-ogb-fixed-100k-gptrans-t-v4-s42-fp32-bs128-60epochs",
            "dataset_identity": identity["dataset_identity"],
            "row_split_identity": f"train-0-100000-development-100000-150000@{identity['row_order_identity']}",
            "architecture_identity": contract["model_id"],
            "optimizer_identity": identity["optimizer_identity"],
            "lr_schedule_identity": identity["schedule_identity"],
            "target_transform_identity": transform["asset_id"],
            "precision_identity": "fp32-no-tf32-deterministic-bs128-drop-last",
            "ema_semantics": "ema9999-each-step-selection",
            "evaluation_role_identity": prediction_manifest["evaluation_role_identity"],
            "selection_role_identity": prediction_manifest["evaluation_role_identity"],
            "x_axis_semantics": "optimizer_steps", "terminal_endpoint_identity": "46860-optimizer-steps",
            "matched_architecture_required": False,
        },
        "exposure": {"optimizer_steps": 46_860, "sample_presentations": 5_998_080},
        "backtest_eligibility": {"eligible": False,
                                 "exclusion_reasons": ["Historical trace has EMA development MAE but no independently evaluated live-model development MAE at each epoch"]},
    }
    output["v5_evidence.json"] = {
        "format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": REFERENCE, "track": "C", "scope": "historical_100k_immutable_reference",
        "legacy_contract": contract["format"],
        "outcome": {
            "execution_status": "complete", "artifact_status": "complete_local_verified",
            "comparison_status": "reference_bundle_structurally_accepted_trace_limited", "scientific_status": "frozen_reference",
            "transfer_status": "paired_endpoint_or_context_only_until_trace_gate", "budget_decision": "closed",
            "full_handoff_status": "not_applicable",
        },
        "authority": {"pointers": [
            (HIST / "training_contract.json").as_posix(),
            (HIST / "decision.md").as_posix(), ref("reference_bundle.json"),
            ref("reference_acceptance.json"),
        ]},
        "artifacts": [
            {"name": name, "locator": f"external://local-platform-record/{RECORD_REL.as_posix()}/training/{name}",
             "sha256": digest, "availability": "local_record_verified"}
            for name, digest in (*artifacts.items(), ("last_checkpoint.pt", checkpoint_sha), ("trace.json", sha256_file(train / "trace.json")))
        ],
        "role_use": {"internal_development": "consumed", "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"},
        "migration": {"migrated_at": "2026-09-29", "verification_scope": "Retained V4 accepted artifact SHA, prediction alignment and MAE, P0 fixed manifest, reused K1 data manifests, runtime, EMA trace and roles; no new graph scan",
                      "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False},
    }
    output["trajectory.json"] = {
        "schema": "molgap-trajectory-v1", "trajectory_id": TRAJECTORY,
        "record_mode": "retrospective_partial", "track": "C", "owner": "historical",
        "family_id": "gptrans-t-v4-reference",
        "question": "Which immutable GPTrans-T V4 run anchors same-contract PCQM-100K comparisons?",
        "hypothesis": {"hypothesis_id": f"H-{TRAJECTORY}", "supporting_evidence_ids": [], "alternative_explanations": [], "related_closed_family_ids": [],
                       "historical_unknowns": ["native device cost was not captured in the retained V4 record"]},
        "state_at_start": {"source_commit": completion["source_commit"],
                           "contract_refs": [(HIST / "training_contract.json").as_posix()],
                           "reference_ids": [], "parent_trajectory_ids": [], "prior_trajectory_ids": [],
                           "prior_evidence_ids": [], "role_snapshot_refs": [(HIST / "results/kaggle_acceptance.json").as_posix()],
                           "budget_snapshot_ref": None},
        "actions": [{"action_id": "A001", "type": "historical_reference_artifact_verification",
                     "run_ids": [run_id], "attempt_ids": ["reference-v4"],
                     "source_commit": completion["source_commit"],
                     "evidence_refs": [(HIST / "decision.md").as_posix(), ref("reference_acceptance.json")],
                     "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [REFERENCE], "evidence_refs": [evidence_ref]},
        "decision": {"decision_ref": (HIST / "decision.md").as_posix(),
                     "outcome": "INFRASTRUCTURE_ONLY", "next_allowed_actions": ["reuse only for exact GPTrans-T 100K V4 scientific-contract matches"],
                     "reopen_conditions": []},
        "comparison_class": "NO_COMPARISON", "comparison_readiness_ref": ref("reference_acceptance.json"),
        "comparison_blockers": [], "reference_bundle_id": BUNDLE,
    }
    # Write dependencies first, then their hash indexes and the bundle.
    for name, value in output.items():
        atomic_json(out / name, value)
    atomic_json(out / "role_history.json", {
        "format": "molgap-role-history-index-v1", "trajectory_id": TRAJECTORY,
        "events": [{"ref": ref(f"roles/{name}.json"), "sha256": sha256_file(out / f"roles/{name}.json")} for name in role_names],
        "protected_roles_read": False,
    })
    atomic_json(out / "cost_records.json", {
        "format": "molgap-cost-record-index-v1", "trajectory_id": TRAJECTORY,
        "records": [{"ref": ref(f"costs/{cost_id}.json"), "sha256": sha256_file(out / f"costs/{cost_id}.json")}],
        "historical_measurement_status": "measurement_missing",
    })
    acceptance = {
        "format": "molgap-v5-reference-recovery-acceptance-v1", "accepted": True,
        "reference_id": REFERENCE, "training_executed": False, "model_inference_executed": False,
        "protected_role_read": False, "reused_fixed_data_manifests_ref": DATA.as_posix(),
        "verified_external_artifacts": {**artifacts, "last_checkpoint.pt": checkpoint_sha,
                                         "trace.json": sha256_file(train / "trace.json")},
        "prediction_mae_eV": mae,
        "strict_causal_trace_ready": False,
        "strict_trace_blocker": "No per-epoch live-model development metric in retained historical trace; only EMA development metric is available",
        "compact_manifest_sha256": {name: sha256_file(out / name) for name in (
            "runtime_certificate.json", "target_transform.json", "prediction_manifest.json",
            "trace_manifest.json", "role_history.json", "cost_records.json")},
    }
    atomic_json(out / "reference_acceptance.json", acceptance)
    bundle = {
        "format": "molgap-reference-bundle-v1", "reference_bundle_id": BUNDLE,
        "reference_id": REFERENCE, "contract_ref": (HIST / "training_contract.json").as_posix(),
        "architecture_config_identity": identity["architecture_config_identity"],
        "source_commit_or_archive": completion["source_commit"],
        "checkpoint_identity": artifacts["best_model.pt"],
        "runtime_certificate_ref": ref("runtime_certificate.json"),
        "prediction_manifest": {key: prediction_manifest[key] for key in (
            "prediction_sha256", "source_idx_sha256", "target_sha256", "ordering_semantics",
            "evaluation_role_identity", "row_count", "unique_source_idx")},
        "row_manifest_ref": (DATA / "row_manifest.json").as_posix(),
        "target_manifest_ref": (DATA / "target_manifest.json").as_posix(),
        "trace_manifest_ref": ref("trace_manifest.json"),
        "role_history_ref": ref("role_history.json"),
        "target_transform_asset_ref": ref("target_transform.json"),
        "cost_records_ref": ref("cost_records.json"),
        "acceptance_ref": ref("reference_acceptance.json"),
        "decision_ref": (HIST / "decision.md").as_posix(),
        "comparison_identity": identity,
        "stochasticity": {"row_bootstrap_uncertainty": {"status": "not_requested"},
                          "training_stochasticity": {"status": "unavailable", "estimation_method": "unavailable",
                                                     "source": "unavailable", "same_contract_repeat_ids": [],
                                                     "n_repeats": 0, "stochasticity_floor_eV": None}},
    }
    atomic_json(out / "reference_bundle.json", bundle)
    print(json.dumps({"accepted": True, "reference_bundle_id": BUNDLE,
                      "prediction_mae_eV": mae, "checkpoint_sha256": artifacts["best_model.pt"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    recover(args.repo_root.resolve())


if __name__ == "__main__":
    main()
