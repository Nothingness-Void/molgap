"""No-model acceptance for the relative-value GraphState seed-42 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = (
    "ogb_distance_angle_relative_value_triangle_edge_state_graph_state9"
)
CANDIDATES = (BASELINE, CANDIDATE)
PARAMETERS = {BASELINE: 3_665_809, CANDIDATE: 3_734_977}
GEOMETRY_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
T4_MEMORY_BYTES = 16 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str, input_sha: str) -> dict:
    import numpy as np
    import torch

    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    required = {
        "complete": True,
        "run_mode": "relative_value_graphstate",
        "source_commit": expected_source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_SHA,
        "input_cache_aggregate_sha256": input_sha,
        "hop_path_cache_aggregate_sha256": input_sha,
        "seed": 42,
        "candidates": list(CANDIDATES),
        "execution": "dual_t4_candidate_parallel",
        "device_assignments": {"0": [BASELINE], "1": [CANDIDATE]},
        "search_budget_s": 14_400,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
        "seed43_44_submitted": False,
        "full_data_authorized": False,
    }
    for key, value in required.items():
        require(selection.get(key) == value, key)
    gpu_names = selection.get("gpu_names", [])
    require(len(gpu_names) == 2 and all("T4" in name for name in gpu_names), "dual T4")

    preflight = selection.get("preflight", [])
    require(len(preflight) == 2, "preflight count")
    peak_memory: dict[str, int] = {}
    for row in preflight:
        identity = row.get("candidate")
        require(identity in CANDIDATES, f"preflight identity {identity}")
        require(row.get("parameter_count") == PARAMETERS.get(identity), f"parameters {identity}")
        require(row.get("global_attention_blocks") == [], f"dense attention {identity}")
        require(row.get("graph_state_present") is True, f"GraphState {identity}")
        require(row.get("hop_path_present") is (identity == CANDIDATE), f"relative path {identity}")
        require(row.get("hop_path_injection_zero") is True, f"zero return {identity}")
        require(row.get("hop_path_return_gradient_nonzero") is True, f"gradient {identity}")
        require(row.get("initial_function_structurally_equal_to_baseline") is True, f"initial function {identity}")
        require(row.get("shared_parameter_mismatches") == [], f"shared init {identity}")
        require(
            all(row.get(key) is True for key in ("finite_prediction", "finite_loss", "finite_gradients")),
            f"finite {identity}",
        )
        memory = row.get("peak_memory_bytes")
        require(isinstance(memory, int) and 0 < memory <= 0.85 * T4_MEMORY_BYTES, f"memory {identity}")
        if isinstance(memory, int):
            peak_memory[identity] = memory

    runs = selection.get("runs", [])
    require(len(runs) == 2, "run count")
    by_name = {row.get("candidate"): row for row in runs}
    payloads = {}
    epoch_time: dict[str, float] = {}
    for identity in CANDIDATES:
        metrics_path = root / "results" / identity / "metrics.json"
        require(metrics_path.is_file(), f"metrics {identity}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics == by_name.get(identity), f"selection metrics {identity}")
        require(metrics.get("complete") is True, f"complete {identity}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {identity}")
        require(metrics.get("input_cache_aggregate_sha256") == input_sha, f"cache {identity}")
        require(metrics.get("parameter_count") == PARAMETERS[identity], f"parameters {identity}")
        require(metrics.get("validation_rows") == 10_000, f"rows {identity}")
        contract = metrics.get("contract", {})
        require(contract.get("global_mechanism") == "gated_graph_state", f"global {identity}")
        require(contract.get("global_attention_blocks") == [], f"attention {identity}")
        expected_path = identity == CANDIDATE
        path_contract = contract.get("hop_path_mixer", "none")
        require(("path-key-value-attention" in path_contract) is expected_path, f"value path {identity}")
        require(contract.get("max_epochs") == 40, f"epochs contract {identity}")
        require(contract.get("precision") == "fp32", f"precision {identity}")
        completed = metrics.get("epochs_completed")
        elapsed = metrics.get("training_elapsed_s")
        require(isinstance(completed, int) and 0 < completed <= 40, f"epochs {identity}")
        require(isinstance(elapsed, (int, float)) and elapsed > 0, f"elapsed {identity}")
        if isinstance(completed, int) and completed and isinstance(elapsed, (int, float)):
            epoch_time[identity] = elapsed / completed
        for role_key in ("official_validation_role_read", "test_dev_role_read"):
            require(metrics.get(role_key) is False, f"{role_key} {identity}")

        artifacts = metrics.get("artifacts", {})
        artifact_paths = {}
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            relative = artifacts.get(name)
            path = root / relative if isinstance(relative, str) else root / "__missing__"
            artifact_paths[name] = path
            require(path.is_file(), f"missing {name} {identity}")
            if path.is_file():
                require(sha256_file(path) == artifacts.get(f"{name}_sha256"), f"hash {name} {identity}")
        if not all(path.is_file() for path in artifact_paths.values()):
            continue

        payload = torch.load(artifact_paths["validation_payload"], map_location="cpu", weights_only=False)
        require(payload.get("candidate") == identity, f"payload identity {identity}")
        require(payload.get("source_commit") == expected_source_commit, f"payload source {identity}")
        require(payload.get("seed") == 42, f"payload seed {identity}")
        require(payload.get("official_validation_role_read") is False, f"payload validation role {identity}")
        require(payload.get("test_dev_role_read") is False, f"payload test role {identity}")
        rows = payload.get("row_index")
        targets = payload.get("target_eV")
        predictions = payload.get("prediction_eV")
        require(isinstance(rows, torch.Tensor) and rows.numel() == 10_000, f"payload rows {identity}")
        require(isinstance(targets, torch.Tensor) and targets.numel() == 10_000, f"payload targets {identity}")
        require(isinstance(predictions, torch.Tensor) and predictions.numel() == 10_000, f"payload predictions {identity}")
        if all(isinstance(value, torch.Tensor) for value in (rows, targets, predictions)):
            rows = rows.reshape(-1).cpu()
            targets = targets.reshape(-1).float().cpu()
            predictions = predictions.reshape(-1).float().cpu()
            require(torch.unique(rows).numel() == 10_000, f"unique rows {identity}")
            require(bool(torch.isfinite(targets).all()), f"finite targets {identity}")
            require(bool(torch.isfinite(predictions).all()), f"finite predictions {identity}")
            recomputed = float(torch.mean(torch.abs(predictions - targets)))
            reported = metrics.get("validation_gap_mae_eV")
            require(isinstance(reported, (int, float)) and math.isclose(recomputed, reported, rel_tol=0, abs_tol=5e-8), f"recomputed MAE {identity}")
            payloads[identity] = (rows, targets, predictions, recomputed)

        best = torch.load(artifact_paths["best_model"], map_location="cpu", weights_only=False)
        checkpoint = torch.load(artifact_paths["checkpoint"], map_location="cpu", weights_only=False)
        for label, artifact in (("best", best), ("checkpoint", checkpoint)):
            require(artifact.get("candidate") == identity, f"{label} identity {identity}")
            require(artifact.get("source_commit") == expected_source_commit, f"{label} source {identity}")
            require(artifact.get("input_cache_aggregate_sha256") == input_sha, f"{label} cache {identity}")
            require(artifact.get("seed") == 42, f"{label} seed {identity}")
            require(isinstance(artifact.get("model"), dict) and bool(artifact["model"]), f"{label} model state {identity}")
        require(best.get("best_epoch") == metrics.get("best_epoch"), f"best epoch {identity}")
        require(math.isclose(float(best.get("best_mae", math.inf)), float(metrics.get("validation_gap_mae_eV", -math.inf)), rel_tol=0, abs_tol=5e-8), f"best MAE {identity}")
        checkpoint_epoch = checkpoint.get("epoch")
        require(
            isinstance(checkpoint_epoch, int)
            and checkpoint_epoch + 1 == metrics.get("epochs_completed"),
            f"checkpoint epoch {identity}",
        )
        require(len(checkpoint.get("trace", [])) == metrics.get("epochs_completed"), f"checkpoint trace {identity}")
        state_keys = set(best.get("model", {}))
        has_relative_values = any(key.endswith("hop_path_mixer.path_value.weight") for key in state_keys)
        require(has_relative_values is expected_path, f"checkpoint relative-value state {identity}")

    comparable = set(payloads) == set(CANDIDATES)
    require(comparable, "paired payloads")
    baseline_mae = candidate_mae = delta = epoch_ratio = None
    ci95 = None
    if comparable:
        rows0, targets0, predictions0, baseline_mae = payloads[BASELINE]
        rows1, targets1, predictions1, candidate_mae = payloads[CANDIDATE]
        require(torch.equal(rows0, rows1), "paired row identity")
        require(torch.equal(targets0, targets1), "paired target identity")
        paired_delta = torch.abs(predictions1 - targets1) - torch.abs(predictions0 - targets0)
        delta = float(paired_delta.mean())
        rng = np.random.default_rng(42)
        values = paired_delta.numpy()
        bootstrap = np.empty(1000, dtype=np.float64)
        for index in range(bootstrap.size):
            sample = rng.integers(0, values.size, values.size)
            bootstrap[index] = values[sample].mean()
        ci95 = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    if set(epoch_time) == set(CANDIDATES):
        epoch_ratio = epoch_time[CANDIDATE] / epoch_time[BASELINE]

    improves = comparable and candidate_mae < baseline_mae
    material = comparable and baseline_mae - candidate_mae >= 0.001
    resource_gate = epoch_ratio is not None and epoch_ratio <= 1.25
    selected = CANDIDATE if improves else BASELINE
    require(selection.get("selected_candidate") == selected, "selection")
    require(selection.get("selected_strictly_improves_baseline") is improves, "selection gate")
    result = {
        "format": "molgap-pcqm-gap100k-relative-value-graphstate-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "input_cache_aggregate_sha256": input_sha,
        "seed": 42,
        "baseline_validation_gap_mae_eV": baseline_mae,
        "candidate_validation_gap_mae_eV": candidate_mae,
        "candidate_minus_baseline_eV": delta,
        "paired_delta_bootstrap_ci95_eV": ci95,
        "selected_candidate": selected,
        "selected_strictly_improves_baseline": improves,
        "material_gain_at_least_0_001_eV": material,
        "candidate_to_baseline_epoch_time_ratio": epoch_ratio,
        "epoch_time_gate_at_most_1_25x": resource_gate,
        "preflight_peak_memory_bytes": peak_memory,
        "memory_reserve_gate_at_least_15_percent": len(peak_memory) == 2,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--input-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit, args.input_sha)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
