"""Stdlib-only hash/CSV acceptance; never import or execute a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_vector_state_triangle_edge_state_graph_state9"
CACHE = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
EXPECTED_CONTRACT = {
    "seed": 42, "batch_size": 48, "precision": "fp32", "max_epochs": 40,
    "patience": 8, "learning_rate": 1.6e-4, "weight_decay": 1e-6,
    "optimizer": "AdamW", "loss": "train_standardized_L1",
    "scheduler": "CosineAnnealingLR", "eta_min": 1e-6,
    "loader_workers": 0, "pin_memory": False, "target": "gap",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accept_screen(
    root: Path,
    source: str,
    *,
    completion_format: str,
    candidate: str,
    candidate_parameter_count: int,
    baseline_delta: dict,
    candidate_delta: dict,
    report_format: str,
) -> dict:
    errors = []
    rows_by_model = {}
    summaries = []

    def require(condition, label):
        if not condition:
            errors.append(label)

    def child(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Artifact path escapes result root")
        return path

    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    require(completion.get("format") == completion_format, "format")
    require(completion.get("complete") is True, "incomplete screen")
    require(completion.get("source_commit") == source, "source")
    require(completion.get("candidates") == [BASELINE, candidate], "candidates")
    require(completion.get("geometry_cache_aggregate_sha256") == CACHE, "cache")
    require(completion.get("contract") == EXPECTED_CONTRACT, "scientific contract")
    require(completion.get("platform") == "SCNet Kunshan", "platform")
    require(completion.get("device_count") == 1, "one DCU")
    require(completion.get("train_graphs") == 100_000 and completion.get("validation_graphs") == 10_000, "data role counts")
    require(completion.get("official_validation_role_read") is False, "official validation")
    require(completion.get("test_dev_role_read") is False, "test dev")
    preflight_path = root / "preflight.json"
    require(sha256(preflight_path) == completion.get("preflight_sha256"), "preflight SHA")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight.get("accepted") is True, "preflight")
    runs = completion.get("runs", [])
    require(len(runs) == 2, "run count")
    for candidate_name in (BASELINE, candidate):
        metrics = json.loads((root / "results" / candidate_name / "metrics.json").read_text(encoding="utf-8"))
        require(metrics in runs, f"completion/metrics {candidate_name}")
        require(metrics.get("candidate") == candidate_name and metrics.get("complete") is True, f"identity/complete {candidate_name}")
        require(metrics.get("source_commit") == source, f"source {candidate_name}")
        require(metrics.get("input_cache_aggregate_sha256") == CACHE, f"cache {candidate_name}")
        require(metrics.get("seed") == 42, f"seed {candidate_name}")
        require(metrics.get("platform_contract") == EXPECTED_CONTRACT, f"contract {candidate_name}")
        architecture_delta = metrics.get("architecture_delta")
        if candidate_name == BASELINE:
            require(architecture_delta == baseline_delta, "baseline architecture delta")
        else:
            require(architecture_delta == candidate_delta, "candidate architecture delta")
        require(metrics.get("official_validation_role_read") is False and metrics.get("test_dev_role_read") is False, f"sealed roles {candidate_name}")
        count = metrics.get("parameter_count", 0)
        require(0 < count <= 4_000_000 and count == preflight.get("parameter_counts", {}).get(candidate_name), f"parameters {candidate_name}")
        if candidate_name == BASELINE:
            require(count == 3_665_809, "baseline parameter count")
        else:
            require(count == candidate_parameter_count, "candidate parameter count")
        artifacts = metrics["artifacts"]
        for key in ("best_model", "checkpoint", "validation_payload", "validation_csv", "trace"):
            path = child(artifacts[key])
            require(path.is_file() and path.stat().st_size > 0, f"artifact {candidate_name} {key}")
            require(sha256(path) == artifacts[key + "_sha256"], f"hash {candidate_name} {key}")
        with child(artifacts["validation_csv"]).open(newline="", encoding="utf-8") as handle:
            values = [(int(r["row_index"]), float(r["target_eV"]), float(r["prediction_eV"])) for r in csv.DictReader(handle)]
        require(len(values) == 10_000 and len({r[0] for r in values}) == 10_000, f"validation identities {candidate_name}")
        require(all(0 <= row < 3_378_606 and math.isfinite(y) and math.isfinite(p) for row, y, p in values), f"roles/finite {candidate_name}")
        mae = sum(abs(y - p) for _, y, p in values) / max(1, len(values))
        require(abs(mae - metrics["validation_gap_mae_eV"]) < 1e-6, f"recomputed MAE {candidate_name}")
        rows_by_model[candidate_name] = [(row, y) for row, y, _ in values]
        trace = json.loads(child(artifacts["trace"]).read_text(encoding="utf-8"))["epochs"]
        require([r["epoch"] for r in trace] == list(range(len(trace))), f"epoch continuity {candidate_name}")
        require(0 < len(trace) <= 40 and len(trace) == metrics.get("epochs_completed"), f"epoch count {candidate_name}")
        if trace:
            best_epoch = min(range(len(trace)), key=lambda i: trace[i]["validation_mae_eV"])
            require(metrics.get("best_epoch") == best_epoch, f"best epoch {candidate_name}")
            require(abs(trace[best_epoch]["validation_mae_eV"] - mae) < 1e-6, f"best MAE {candidate_name}")
            require(len(trace) == 40 or len(trace) - best_epoch - 1 == 8, f"incomplete schedule {candidate_name}")
        require(all(math.isfinite(r[k]) and r["elapsed_s"] > 0 for r in trace for k in ("train_mae_eV", "validation_mae_eV", "elapsed_s", "graphs_per_s", "learning_rate")), f"trace finite {candidate_name}")
        summaries.append({"candidate": candidate_name, "mae_eV": mae, "parameters": count, "epochs": len(trace), "best_epoch": metrics.get("best_epoch"), "throughput": metrics.get("mean_throughput_graphs_per_s"), "peak_reserved_bytes": metrics.get("peak_memory_reserved_bytes"), "device_total_bytes": metrics.get("device_total_memory_bytes")})
    require(rows_by_model[BASELINE] == rows_by_model[candidate], "paired validation row/target mismatch")
    return {"format": report_format, "accepted": not errors, "errors": errors, "source_commit": source, "model_inference_executed": False, "official_validation_role_read": False, "test_dev_role_read": False, "runs": summaries, "candidate_minus_control_eV": summaries[1]["mae_eV"] - summaries[0]["mae_eV"]}


def accept(root: Path, source: str) -> dict:
    return accept_screen(
        root,
        source,
        completion_format="molgap-kunshan-vector-screen-v1",
        candidate=CANDIDATE,
        candidate_parameter_count=3_696_209,
        baseline_delta={"vector_state": "none"},
        candidate_delta={
            "vector_state": "persistent_polar_order1_channels16",
            "vector_update_blocks": [2, 4, 6, 8],
            "relation": "directed_real_bond_displacement",
            "scalar_return": "norm_norm_dot_linear192_bias_free_zero_init",
        },
        report_format="molgap-kunshan-vector-acceptance-v1",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = accept(args.root, args.source_commit)
    except Exception as error:
        result = {"accepted": False, "errors": [f"{type(error).__name__}: {error}"], "model_inference_executed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
