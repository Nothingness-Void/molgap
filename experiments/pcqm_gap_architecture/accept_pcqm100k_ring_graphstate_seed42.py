"""No-model acceptance for the paired Ring-GraphState seed-42 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = (
    "ogb_distance_angle_ring_hierarchy_triangle_edge_state_graph_state9"
)
CANDIDATES = (BASELINE, CANDIDATE)
EXPECTED_PARAMETERS = {BASELINE: 3_665_809, CANDIDATE: 3_723_849}
EXPECTED_RING_SHA256 = (
    "3f8b271571b8d1026e96fc1dae51d9479489ddd13b73df95740288e6f630779f"
)
EXPECTED_GEOMETRY_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(selection.get("complete") is True, "complete")
    require(selection.get("run_mode") == "ring_graphstate", "run mode")
    require(selection.get("source_commit") == expected_source_commit, "source")
    require(selection.get("seed") == 42, "seed")
    require(selection.get("candidates") == list(CANDIDATES), "candidates")
    require(selection.get("search_budget_s") == 14_400, "budget")
    require(selection.get("geometry_cache_aggregate_sha256") == EXPECTED_GEOMETRY_SHA256, "geometry SHA")
    require(selection.get("ring_cache_aggregate_sha256") == EXPECTED_RING_SHA256, "ring SHA")
    require(selection.get("input_cache_aggregate_sha256") == EXPECTED_RING_SHA256, "input SHA")
    require(selection.get("execution") == "dual_t4_candidate_parallel", "execution")
    require(selection.get("device_assignments") == {"0": [BASELINE], "1": [CANDIDATE]}, "devices")
    require(
        len(selection.get("gpu_names", [])) == 2
        and all("T4" in name for name in selection["gpu_names"]),
        "dual T4",
    )
    for key in (
        "official_validation_role_read",
        "test_dev_role_read",
        "molecular_research_server_accessed",
    ):
        require(selection.get(key) is False, key)

    preflight = selection.get("preflight", [])
    require(len(preflight) == 2, "preflight count")
    for row in preflight:
        identity = row.get("candidate")
        require(identity in CANDIDATES, f"preflight identity {identity}")
        require(row.get("parameter_count") == EXPECTED_PARAMETERS.get(identity), f"parameters {identity}")
        require(row.get("global_attention_blocks") == [], f"attention {identity}")
        require(row.get("graph_state_present") is True, f"GraphState {identity}")
        require(row.get("ring_hierarchy_present") is (identity == CANDIDATE), f"ring {identity}")
        require(row.get("ring_injection_zero") is True, f"ring zero {identity}")
        require(row.get("ring_return_gradient_nonzero") is True, f"ring gradient {identity}")
        require(row.get("shared_parameter_mismatches") == [], f"shared init {identity}")
        require(
            all(row.get(key) is True for key in ("finite_prediction", "finite_loss", "finite_gradients")),
            f"finite {identity}",
        )

    runs = selection.get("runs", [])
    require(len(runs) == 2, "run count")
    by_name = {row.get("candidate"): row for row in runs}
    row_hashes = set()
    target_hashes = set()
    for identity in CANDIDATES:
        metrics_path = root / "results" / identity / "metrics.json"
        require(metrics_path.is_file(), f"metrics {identity}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics == by_name.get(identity), f"selection metrics {identity}")
        require(metrics.get("complete") is True, f"complete {identity}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {identity}")
        require(metrics.get("seed") == 42, f"seed {identity}")
        require(metrics.get("parameter_count") == EXPECTED_PARAMETERS[identity], f"parameters {identity}")
        require(metrics.get("input_cache_aggregate_sha256") == EXPECTED_RING_SHA256, f"cache {identity}")
        value = metrics.get("validation_gap_mae_eV")
        require(isinstance(value, (int, float)) and math.isfinite(value), f"MAE {identity}")
        require(metrics.get("validation_rows") == 10_000, f"rows {identity}")
        contract = metrics.get("contract", {})
        require(contract.get("global_mechanism") == "gated_graph_state", f"global {identity}")
        require(contract.get("global_attention_blocks") == [], f"attention {identity}")
        require(contract.get("ring_update_layers") == ([2, 4, 6, 8] if identity == CANDIDATE else []), f"ring layers {identity}")
        require(contract.get("ring_hierarchy") != "none" if identity == CANDIDATE else contract.get("ring_hierarchy") == "none", f"ring contract {identity}")
        row_hashes.add(metrics.get("validation_row_index_sha256"))
        target_hashes.add(metrics.get("validation_target_sha256"))
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
    improves = (
        isinstance(baseline_mae, (int, float))
        and isinstance(candidate_mae, (int, float))
        and candidate_mae < baseline_mae
    )
    selected = CANDIDATE if improves else BASELINE
    require(selection.get("selected_candidate") == selected, "selection")
    require(selection.get("selected_strictly_improves_baseline") is improves, "gate")
    result = {
        "format": "molgap-pcqm-gap100k-ring-graphstate-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "seed": 42,
        "ring_cache_aggregate_sha256": EXPECTED_RING_SHA256,
        "baseline_validation_gap_mae_eV": baseline_mae,
        "candidate_validation_gap_mae_eV": candidate_mae,
        "candidate_minus_baseline_eV": (
            candidate_mae - baseline_mae
            if isinstance(candidate_mae, (int, float))
            and isinstance(baseline_mae, (int, float))
            else None
        ),
        "selected_candidate": selected,
        "selected_strictly_improves_baseline": improves,
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
