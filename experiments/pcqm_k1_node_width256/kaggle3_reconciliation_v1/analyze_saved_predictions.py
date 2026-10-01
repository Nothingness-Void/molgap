"""Compare retained internal-development predictions; no model execution."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASES = {
    "reference": ROOT / "platforms/_records/kaggle/training/pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference",
    "candidate": ROOT / "platforms/_records/kaggle/training/pcqm_k1_node_width256_kaggle3_v1/experiment/width256",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    helper_path = ROOT / "experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py"
    spec = importlib.util.spec_from_file_location("retained_pair_analysis", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    predictions, traces, manifests, runtimes, artifacts = {}, {}, {}, {}, {}
    for name, base in BASES.items():
        manifests[name] = read_json(base / "output_manifest.json")
        traces[name] = read_json(base / "canonical_trace.json")
        runtimes[name] = read_json(base / "runtime_manifest.json")
        artifacts[name] = {}
        for key in ("predictions", "trace", "selected_model", "resume"):
            item = manifests[name]["artifacts"][key]
            digest = sha256(base / item["path"])
            if digest != item["sha256"]:
                raise RuntimeError(f"{name} {key} digest differs")
            artifacts[name][key] = {"path": (base / item["path"]).relative_to(ROOT).as_posix(), "sha256": digest}
        predictions[name] = torch.load(base / "development_predictions.pt", map_location="cpu", weights_only=False)
        if not all(predictions[name][key].shape == (50000,) for key in ("prediction_eV", "target_eV", "source_idx")):
            raise RuntimeError(f"{name} tensor shape differs")
    endpoint = helper.paired_metrics(predictions["reference"], predictions["candidate"], 100000, 150000)
    endpoint.update({"gain_definition": "reference absolute error minus candidate absolute error", "gate_eV": 0.003,
                     "row_bootstrap_replicates": 1000, "row_bootstrap_rng_seed": 42,
                     "material_gate_ci_lower_pass": endpoint["row_bootstrap_95pct_eV"][0] >= 0.003,
                     "training_seed_variability": "unknown_single_seed42"})
    curves = {}
    for name in BASES:
        observations = traces[name]["observations"]
        if len(observations) != 40:
            raise RuntimeError("Expected 40 complete observations")
        curves[name] = {row["optimizer_step"]: row for row in observations}
        for epoch, row in enumerate(observations, 1):
            if (row["epoch_or_pass"], row["optimizer_step"], row["sample_presentations"]) != (epoch, epoch * 781, epoch * 99968):
                raise RuntimeError("Exposure axes differ from contract")
        if observations[-1]["checkpoint_identity"] != "sha256:" + artifacts[name]["resume"]["sha256"]:
            raise RuntimeError("Final trace checkpoint identity differs")
    if traces["reference"]["metric_semantics"] != traces["candidate"]["metric_semantics"]:
        raise RuntimeError("Trace metric semantics differ")
    curve = []
    for step, ref in curves["reference"].items():
        candidate = curves["candidate"][step]
        if ref["learning_rate"] != candidate["learning_rate"]:
            raise RuntimeError("LR axes differ")
        curve.append({"epoch": ref["epoch_or_pass"], "optimizer_steps": step,
                      "sample_presentations": ref["sample_presentations"], "learning_rate": ref["learning_rate"],
                      "reference_online_train_mae_eV": ref["live_train_metric"],
                      "candidate_online_train_mae_eV": candidate["live_train_metric"],
                      "reference_development_mae_eV": ref["live_dev_metric"],
                      "candidate_development_mae_eV": candidate["live_dev_metric"],
                      "development_gain_eV": ref["live_dev_metric"] - candidate["live_dev_metric"]})
    selection = {}
    for name in BASES:
        selected = min(traces[name]["observations"], key=lambda row: row["live_dev_metric"])
        selection[name] = {"minimum_development_trace_epoch": selected["epoch_or_pass"],
                           "minimum_development_trace_mae_eV": selected["live_dev_metric"],
                           "selected_model_sha256": artifacts[name]["selected_model"]["sha256"],
                           "final_resume_sha256": artifacts[name]["resume"]["sha256"],
                           "epochs": 40, "optimizer_steps": 31240, "sample_presentations": 3998720}
    signed_errors = {}
    for name in BASES:
        record = predictions[name]
        residual = record["prediction_eV"].double().numpy() - record["target_eV"].double().numpy()
        signed_errors[name] = {"mean_prediction_minus_target_eV": float(residual.mean()),
                               "residual_quantiles_eV": dict(zip(("q05", "q50", "q95"), map(float, np.quantile(residual, [0.05, 0.5, 0.95]))))}
    runtime_fields = ("python", "platform", "torch", "torch_cuda", "cudnn", "accelerator", "determinism", "runtime_fingerprint", "installed_distributions_sha256")
    runtime_summary = {name: {key: runtime[key] for key in runtime_fields} for name, runtime in runtimes.items()}
    package_differences = {"reference_only": sorted(set(runtimes["reference"]["installed_distributions"]) - set(runtimes["candidate"]["installed_distributions"])),
                           "candidate_only": sorted(set(runtimes["candidate"]["installed_distributions"]) - set(runtimes["reference"]["installed_distributions"]))}
    result = {"format": "molgap-k1-width256-saved-prediction-analysis-v1", "analysis_scope": "retained_internal_development_predictions_and_traces_only",
              "helper": {"path": helper_path.relative_to(ROOT).as_posix(), "callable": "paired_metrics", "sha256": sha256(helper_path)},
              "alignment": {"source_idx": [100000, 150000], "stop_exclusive": True, "rows": 50000, "exact_order": True,
                            "identical_targets": True, "finite_predictions_and_targets": True,
                            "target_float32_sha256": sha256_bytes(predictions["reference"]["target_eV"].numpy().tobytes())},
              "endpoint": endpoint, "selection_and_exposure": selection, "trace_metric_semantics": traces["reference"]["metric_semantics"],
              "paired_curve": curve, "posthoc_descriptive_signed_errors_no_fit": signed_errors,
              "runtime_summary": runtime_summary, "installed_distribution_differences": package_differences,
              "native_costs": {name: manifests[name]["costs"] for name in BASES},
              "candidate_allocation_scope": read_json(BASES["candidate"] / "allocation_cost.json"),
              "cost_comparison_status": "descriptive_only_cross_job_runtime_and_cost_scopes_not_controlled_no_architecture_speedup_claim",
              "artifacts": artifacts,
              "limitations": ["single training seed; row bootstrap excludes training stochasticity", "online pre-update dropout training metric is not fixed-cohort development evaluation",
                              "no controlled architecture throughput comparison", "CPU and queue allocation are missing; training timing excludes bootstrap/queue",
                              "strict runtime, role, source and qualification acceptance is owned by the reconciliation decision"]}
    (OUT / "scientific_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"endpoint": endpoint, "selection": selection, "runtime_package_differences": package_differences}, indent=2))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()
