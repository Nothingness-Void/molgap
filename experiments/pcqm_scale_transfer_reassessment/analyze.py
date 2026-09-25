"""Reassess accepted PCQM scale-transfer evidence without prediction access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TRACE_DIRS = {
    "reference": "pcqm_500k_v4_evidence",
    "pair_norm": "pcqm_gptrans_pair_norm_500k",
    "noisy_nodes": "pcqm_gptrans_noisy_nodes_500k",
    "joint": "pcqm_gptrans_noisy_pair_norm_500k",
}
MATCH_FIELDS = (
    "scientific_contract",
    "dataset_identity",
    "row_split_identity",
    "optimizer_identity",
    "lr_schedule_identity",
    "precision_identity",
    "ema_semantics",
    "target_transform_identity",
    "evaluation_role_identity",
    "selection_role_identity",
    "terminal_endpoint_identity",
    "x_axis_semantics",
)
SNAPSHOT_EPOCHS = (9, 19, 29, 39, 49, 59)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root: Path) -> dict:
    records = {}
    for name, dirname in TRACE_DIRS.items():
        base = root / "experiments" / dirname
        manifest = load_json(base / "trace_manifest.json")
        trace_path = root / manifest["trace_artifact_ref"]
        actual_sha = sha256(trace_path)
        if actual_sha != manifest["trace_artifact_sha256"]:
            raise ValueError(f"trace SHA mismatch: {name}")
        trace = load_json(trace_path)
        observations = trace["observations"]
        if len(observations) != 60:
            raise ValueError(f"expected 60 complete observations: {name}")
        by_step = {item["optimizer_step"]: item for item in observations}
        if len(by_step) != 60 or any(item["live_dev_metric"] is None for item in observations):
            raise ValueError(f"incomplete or duplicate development trace: {name}")
        records[name] = {
            "manifest": manifest,
            "trace_ref": manifest["trace_artifact_ref"],
            "trace_sha256": actual_sha,
            "by_step": by_step,
            "observations": observations,
        }

    reference = records["reference"]
    identity = reference["manifest"]["comparability_identity"]
    for name, record in records.items():
        other = record["manifest"]["comparability_identity"]
        mismatched = [field for field in MATCH_FIELDS if other[field] != identity[field]]
        if mismatched:
            raise ValueError(f"unmatched 500K contract for {name}: {mismatched}")
        if set(record["by_step"]) != set(reference["by_step"]):
            raise ValueError(f"optimizer-step mismatch for {name}")
        if [item["optimizer_step"] for item in record["observations"]] != [
            item["optimizer_step"] for item in reference["observations"]
        ]:
            raise ValueError(f"optimizer-step order mismatch for {name}")

    baseline = reference["observations"]
    arms = {}
    for name, record in records.items():
        if name == "reference":
            continue
        observations = record["observations"]
        gains = [base["live_dev_metric"] - arm["live_dev_metric"] for base, arm in zip(baseline, observations)]
        train_differences = [base["live_train_metric"] - arm["live_train_metric"] for base, arm in zip(baseline, observations)]
        arms[name] = {
            "trace_ref": record["trace_ref"],
            "trace_sha256": record["trace_sha256"],
            "best_selected_gain_eV": min(base["live_dev_metric"] for base in baseline)
            - min(arm["live_dev_metric"] for arm in observations),
            "peak_same_step_gain_eV": max(gains),
            "peak_same_step_epoch": gains.index(max(gains)),
            "first_10_mean_gain_eV": sum(gains[:10]) / 10,
            "last_10_mean_gain_eV": sum(gains[-10:]) / 10,
            "terminal_gain_eV": gains[-1],
            "terminal_online_train_difference_eV": train_differences[-1],
            "snapshots": [
                {
                    "epoch_zero_based": epoch,
                    "optimizer_step": baseline[epoch]["optimizer_step"],
                    "sample_presentations": baseline[epoch]["sample_presentations"],
                    "same_step_development_gain_eV": gains[epoch],
                    "online_train_difference_eV": train_differences[epoch],
                }
                for epoch in SNAPSHOT_EPOCHS
            ],
        }

    metrics_100k = {
        name: load_json(root / "experiments" / dirname / "v5_evidence.json")["metrics"]
        for name, dirname in {
            "noisy_nodes": "pcqm_gptrans_noisy_nodes_100k",
            "joint": "pcqm_gptrans_noisy_pair_norm_100k",
        }.items()
    }
    transfer = {}
    for name, metric in metrics_100k.items():
        gain_100k = metric["material_gain_eV"]
        gain_500k = arms[name]["best_selected_gain_eV"]
        transfer[name] = {
            "accepted_100k_gain_eV": gain_100k,
            "accepted_500k_gain_eV": gain_500k,
            "retained_fraction": gain_500k / gain_100k,
            "note": "Contextual cross-scale ratio; EMA, development cohort, and horizon differ.",
        }

    gptrans_contract = load_json(root / "experiments/pcqm_k1_gptrans_full_fusion/training_contract.json")
    k1_contract = load_json(root / "experiments/pcqm_k1_gptrans_full_fusion/results/accepted_k1_gptrans_fusion_r3/k1/training_contract.json")
    edge_metrics = load_json(root / "experiments/pcqm_edge_state_full/results/convergence_40/remote_metrics.json")
    gptrans_cont = load_json(root / "experiments/pcqm_gptrans_full_convergence/results/canonical_trace.json")
    k1_cont = load_json(root / "experiments/pcqm_k1_full_convergence/results/canonical_trace.json")
    return {
        "schema": "molgap-pcqm-scale-transfer-reassessment-v1",
        "method": "source-only accepted RML trace and contract audit; no predictions or protected roles opened",
        "matched_500k": {
            "reference_trace_ref": reference["trace_ref"],
            "reference_trace_sha256": reference["trace_sha256"],
            "comparison_identity": {field: identity[field] for field in MATCH_FIELDS},
            "reference_best_development_eV": min(item["live_dev_metric"] for item in baseline),
            "arms": arms,
            "online_train_semantics": "epoch-online training metric, not fixed-subset training evaluation",
        },
        "cross_scale_context": {
            "transfer": transfer,
            "historical_joint_500k_projection_lower_bound_eV": 0.0078,
            "historical_projection_source": "experiments/pcqm_scale_transfer_diagnostic/decision.md",
            "joint_projection_shortfall_eV": 0.0078 - arms["joint"]["best_selected_gain_eV"],
            "100k_weight_semantics": "EMA decay 0.9999",
            "500k_weight_semantics": "live/raw",
            "100k_and_500k_development_cohorts_identical": False,
        },
        "full_context": {
            "gptrans_initial_presentations": gptrans_contract["sample_presentations"],
            "k1_initial_presentations": k1_contract["sample_presentations"],
            "gptrans_selected_continuation_presentations": gptrans_cont["observations"][-1]["sample_presentations"],
            "k1_completed_continuation_presentations": k1_cont["observations"][-1]["sample_presentations"],
            "gptrans_selected_total_presentations": gptrans_contract["sample_presentations"] + gptrans_cont["observations"][-1]["sample_presentations"],
            "k1_completed_total_presentations": k1_contract["sample_presentations"] + k1_cont["observations"][-1]["sample_presentations"],
            "edge_state_selected_epoch_zero_based": edge_metrics["best_epoch"],
            "edge_state_train_rows_per_pass": edge_metrics["train_log"][0]["train_rows"],
            "edge_state_selected_passes_if_zero_based": edge_metrics["best_epoch"] + 1,
            "edge_state_nominal_selected_presentations_if_complete_passes": (edge_metrics["best_epoch"] + 1) * edge_metrics["train_log"][0]["train_rows"],
            "gptrans_continuation_ema_development_eV": [item["ema_dev_metric"] for item in gptrans_cont["observations"]],
            "k1_continuation_live_development_eV": [item["live_dev_metric"] for item in k1_cont["observations"]],
            "source_sha256": {
                "gptrans_initial_contract": sha256(root / "experiments/pcqm_k1_gptrans_full_fusion/training_contract.json"),
                "k1_initial_contract": sha256(root / "experiments/pcqm_k1_gptrans_full_fusion/results/accepted_k1_gptrans_fusion_r3/k1/training_contract.json"),
                "edge_state_continuation_metrics": sha256(root / "experiments/pcqm_edge_state_full/results/convergence_40/remote_metrics.json"),
                "gptrans_continuation_trace": sha256(root / "experiments/pcqm_gptrans_full_convergence/results/canonical_trace.json"),
                "k1_continuation_trace": sha256(root / "experiments/pcqm_k1_full_convergence/results/canonical_trace.json"),
            },
            "full_training_contracts": {
                "gptrans": "experiments/pcqm_k1_gptrans_full_fusion/training_contract.json",
                "k1": "experiments/pcqm_k1_gptrans_full_fusion/results/accepted_k1_gptrans_fusion_r3/k1/training_contract.json",
                "edgestate": "experiments/pcqm_edge_state_full/results/convergence_40/remote_metrics.json",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(analyze(args.repo_root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
