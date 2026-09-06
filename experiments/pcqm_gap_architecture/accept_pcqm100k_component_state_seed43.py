"""No-model acceptance for the paired ComponentState seed-43 confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CONTROL = "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_descriptor"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_component"
PARAMETERS = {CONTROL: 3_672_257, CANDIDATE: 3_694_033}
GEOMETRY_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, source_commit: str, cache_sha: str) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    expected = {
        "complete": True,
        "run_mode": "conjugated_component_confirmation",
        "source_commit": source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_SHA,
        "input_cache_aggregate_sha256": cache_sha,
        "seed": 43,
        "candidates": [CONTROL, CANDIDATE],
        "execution": "dual_t4_candidate_parallel",
        "device_assignments": {"0": [CONTROL], "1": [CANDIDATE]},
        "search_budget_s": 14_400,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
        "seed43_44_submitted": False,
        "full_data_authorized": False,
    }
    for key, value in expected.items():
        require(selection.get(key) == value, key)
    require(
        len(selection.get("gpu_names", [])) == 2
        and all("T4" in name for name in selection["gpu_names"]),
        "dual T4",
    )
    preflight = selection.get("preflight", [])
    require(len(preflight) == 2, "preflight count")
    for row in preflight:
        identity = row.get("candidate")
        require(identity in PARAMETERS, f"preflight identity {identity}")
        require(row.get("parameter_count") == PARAMETERS.get(identity), f"parameters {identity}")
        require(row.get("global_attention_blocks") == [], f"attention {identity}")
        require(row.get("graph_state_present") is True, f"GraphState {identity}")
        require(row.get("conjugated_descriptor_present") is True, f"descriptor {identity}")
        require(row.get("component_state_present") is (identity == CANDIDATE), f"component {identity}")
        require(row.get("component_return_zero") is True, f"zero return {identity}")
        require(row.get("component_return_gradient_nonzero") is True, f"gradient {identity}")
        require(row.get("initial_function_structurally_equal_to_baseline") is True, f"initial function {identity}")
        require(row.get("shared_parameter_mismatches") == [], f"shared init {identity}")
        require(all(row.get(key) is True for key in ("finite_prediction", "finite_loss", "finite_gradients")), f"finite {identity}")

    runs = selection.get("runs", [])
    require(len(runs) == 2, "run count")
    by_name = {row.get("candidate"): row for row in runs}
    row_hashes, target_hashes = set(), set()
    for identity in PARAMETERS:
        metrics_path = root / "results" / identity / "metrics.json"
        require(metrics_path.is_file(), f"metrics {identity}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics == by_name.get(identity), f"selection metrics {identity}")
        require(metrics.get("complete") is True, f"complete {identity}")
        require(metrics.get("source_commit") == source_commit, f"source {identity}")
        require(metrics.get("input_cache_aggregate_sha256") == cache_sha, f"cache {identity}")
        require(metrics.get("parameter_count") == PARAMETERS[identity], f"parameters {identity}")
        require(metrics.get("validation_rows") == 10_000, f"rows {identity}")
        require(math.isfinite(float(metrics.get("validation_gap_mae_eV", math.nan))), f"MAE {identity}")
        contract = metrics.get("contract", {})
        require(contract.get("global_mechanism") == "gated_graph_state", f"global {identity}")
        require(contract.get("global_attention_blocks") == [], f"attention {identity}")
        require((contract.get("conjugated_component_state") != "none") is (identity == CANDIDATE), f"contract {identity}")
        row_hashes.add(metrics.get("validation_row_index_sha256"))
        target_hashes.add(metrics.get("validation_target_sha256"))
        require(metrics.get("official_validation_role_read") is False, f"official role {identity}")
        require(metrics.get("test_dev_role_read") is False, f"test-dev role {identity}")
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            relative = metrics.get("artifacts", {}).get(name)
            path = root / relative if isinstance(relative, str) else root / "__missing__"
            require(path.is_file(), f"missing {name} {identity}")
            if path.is_file():
                require(sha256_file(path) == metrics["artifacts"].get(f"{name}_sha256"), f"hash {name} {identity}")
    require(len(row_hashes) == 1, "row identity")
    require(len(target_hashes) == 1, "target identity")
    control_mae = by_name.get(CONTROL, {}).get("validation_gap_mae_eV")
    candidate_mae = by_name.get(CANDIDATE, {}).get("validation_gap_mae_eV")
    comparable = isinstance(control_mae, (int, float)) and isinstance(candidate_mae, (int, float))
    improves = comparable and candidate_mae < control_mae
    require(selection.get("selected_candidate") == (CANDIDATE if improves else CONTROL), "selection")
    result = {
        "format": "molgap-pcqm-gap100k-component-state-seed43-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": source_commit,
        "cache_sha256": cache_sha,
        "seed": 43,
        "control_validation_gap_mae_eV": control_mae,
        "candidate_validation_gap_mae_eV": candidate_mae,
        "candidate_minus_control_eV": candidate_mae - control_mae if comparable else None,
        "strictly_improves_control": improves,
        "material_gain_at_least_0_001_eV": comparable and candidate_mae <= control_mae - 0.001,
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
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cache-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.source_commit, args.cache_sha)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
