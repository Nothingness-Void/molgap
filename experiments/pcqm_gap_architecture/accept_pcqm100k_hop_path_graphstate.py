"""No-model acceptance for the paired hop-path GraphState seed-42 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_hop_path_triangle_edge_state_graph_state9"
CANDIDATES = (BASELINE, CANDIDATE)
PARAMETERS = {BASELINE: 3_665_809, CANDIDATE: 3_697_537}
GEOMETRY_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
T4_MEMORY_BYTES = 16 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str, input_sha: str) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    required = {
        "complete": True,
        "run_mode": "hop_path_graphstate",
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
        require(row.get("global_attention_blocks") == [], f"attention {identity}")
        require(row.get("graph_state_present") is True, f"GraphState {identity}")
        require(row.get("hop_path_present") is (identity == CANDIDATE), f"mechanism {identity}")
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
    row_hashes, target_hashes = set(), set()
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
        mae = metrics.get("validation_gap_mae_eV")
        require(isinstance(mae, (int, float)) and math.isfinite(mae), f"MAE {identity}")
        contract = metrics.get("contract", {})
        require(contract.get("global_mechanism") == "gated_graph_state", f"global {identity}")
        require(contract.get("global_attention_blocks") == [], f"attention {identity}")
        require((contract.get("hop_path_mixer") != "none") is (identity == CANDIDATE), f"contract {identity}")
        row_hashes.add(metrics.get("validation_row_index_sha256"))
        target_hashes.add(metrics.get("validation_target_sha256"))
        completed = metrics.get("epochs_completed")
        elapsed = metrics.get("training_elapsed_s")
        require(isinstance(completed, int) and completed > 0, f"epochs {identity}")
        require(isinstance(elapsed, (int, float)) and elapsed > 0, f"elapsed {identity}")
        if isinstance(completed, int) and completed > 0 and isinstance(elapsed, (int, float)):
            epoch_time[identity] = elapsed / completed
        for role_key in ("official_validation_role_read", "test_dev_role_read"):
            require(metrics.get(role_key) is False, f"{role_key} {identity}")
        artifacts = metrics.get("artifacts", {})
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            relative = artifacts.get(name)
            path = root / relative if isinstance(relative, str) else root / "__missing__"
            require(path.is_file(), f"missing {name} {identity}")
            if path.is_file():
                require(sha256_file(path) == artifacts.get(f"{name}_sha256"), f"hash {name} {identity}")

    require(len(row_hashes) == 1, "row identity")
    require(len(target_hashes) == 1, "target identity")
    baseline_mae = by_name.get(BASELINE, {}).get("validation_gap_mae_eV")
    candidate_mae = by_name.get(CANDIDATE, {}).get("validation_gap_mae_eV")
    comparable = isinstance(baseline_mae, (int, float)) and isinstance(candidate_mae, (int, float))
    delta = candidate_mae - baseline_mae if comparable else None
    improves = comparable and candidate_mae < baseline_mae
    material_gain = comparable and baseline_mae - candidate_mae >= 0.001
    selected = CANDIDATE if improves else BASELINE
    require(selection.get("selected_candidate") == selected, "selection")
    require(selection.get("selected_strictly_improves_baseline") is improves, "selection gate")
    epoch_ratio = (
        epoch_time[CANDIDATE] / epoch_time[BASELINE]
        if set(epoch_time) == set(CANDIDATES)
        else None
    )
    resource_gate = epoch_ratio is not None and epoch_ratio <= 1.5
    result = {
        "format": "molgap-pcqm-gap100k-hop-path-graphstate-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "input_cache_aggregate_sha256": input_sha,
        "seed": 42,
        "baseline_validation_gap_mae_eV": baseline_mae,
        "candidate_validation_gap_mae_eV": candidate_mae,
        "candidate_minus_baseline_eV": delta,
        "selected_candidate": selected,
        "selected_strictly_improves_baseline": improves,
        "material_gain_at_least_0_001_eV": material_gain,
        "candidate_to_baseline_epoch_time_ratio": epoch_ratio,
        "epoch_time_gate_at_most_1_5x": resource_gate,
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
