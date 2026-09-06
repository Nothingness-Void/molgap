"""No-model acceptance for paired directed-bond or SignNet-LapPE screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
GEOMETRY_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
MODES = {
    "directed_bond_graphstate": {
        "candidate": "ogb_distance_angle_directed_bond_triangle_edge_state_graph_state9",
        "parameters": 3_741_265,
        "presence": "directed_bond_present",
        "gradient": "directed_bond_return_gradient_nonzero",
        "contract": "directed_bond_memory",
    },
    "signnet_lappe_graphstate": {
        "candidate": "ogb_distance_angle_signnet_lappe_triangle_edge_state_graph_state9",
        "parameters": 3_673_109,
        "presence": "signnet_lappe_present",
        "gradient": "signnet_lappe_return_gradient_nonzero",
        "contract": "spectral_position_encoding",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str, mode: str, input_sha: str) -> dict:
    spec = MODES[mode]
    candidate = spec["candidate"]
    candidates = (BASELINE, candidate)
    parameters = {BASELINE: 3_665_809, candidate: spec["parameters"]}
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    required = {
        "complete": True,
        "run_mode": mode,
        "source_commit": expected_source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_SHA,
        "input_cache_aggregate_sha256": input_sha,
        "seed": 42,
        "candidates": list(candidates),
        "execution": "dual_t4_candidate_parallel",
        "device_assignments": {"0": [BASELINE], "1": [candidate]},
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
    for row in preflight:
        identity = row.get("candidate")
        require(identity in candidates, f"preflight identity {identity}")
        require(row.get("parameter_count") == parameters.get(identity), f"parameters {identity}")
        require(row.get("global_attention_blocks") == [], f"attention {identity}")
        require(row.get("graph_state_present") is True, f"GraphState {identity}")
        require(row.get(spec["presence"]) is (identity == candidate), f"mechanism {identity}")
        require(row.get(spec["gradient"]) is True, f"gradient {identity}")
        require(row.get("initial_function_structurally_equal_to_baseline") is True, f"initial function {identity}")
        require(row.get("shared_parameter_mismatches") == [], f"shared init {identity}")
        require(all(row.get(key) is True for key in ("finite_prediction", "finite_loss", "finite_gradients")), f"finite {identity}")

    runs = selection.get("runs", [])
    require(len(runs) == 2, "run count")
    by_name = {row.get("candidate"): row for row in runs}
    row_hashes, target_hashes = set(), set()
    for identity in candidates:
        metrics_path = root / "results" / identity / "metrics.json"
        require(metrics_path.is_file(), f"metrics {identity}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics == by_name.get(identity), f"selection metrics {identity}")
        require(metrics.get("complete") is True, f"complete {identity}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {identity}")
        require(metrics.get("input_cache_aggregate_sha256") == input_sha, f"input cache {identity}")
        require(metrics.get("parameter_count") == parameters[identity], f"parameters {identity}")
        require(metrics.get("validation_rows") == 10_000, f"rows {identity}")
        mae = metrics.get("validation_gap_mae_eV")
        require(isinstance(mae, (int, float)) and math.isfinite(mae), f"MAE {identity}")
        contract = metrics.get("contract", {})
        require(contract.get("global_mechanism") == "gated_graph_state", f"global {identity}")
        require(contract.get("global_attention_blocks") == [], f"attention {identity}")
        require((contract.get(spec["contract"]) != "none") is (identity == candidate), f"contract {identity}")
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
    candidate_mae = by_name.get(candidate, {}).get("validation_gap_mae_eV")
    comparable = isinstance(baseline_mae, (int, float)) and isinstance(candidate_mae, (int, float))
    improves = comparable and candidate_mae < baseline_mae
    selected = candidate if improves else BASELINE
    require(selection.get("selected_candidate") == selected, "selection")
    require(selection.get("selected_strictly_improves_baseline") is improves, "gate")
    result = {
        "format": "molgap-pcqm-gap100k-directed-spectral-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "mode": mode,
        "source_commit": expected_source_commit,
        "input_cache_aggregate_sha256": input_sha,
        "seed": 42,
        "baseline_validation_gap_mae_eV": baseline_mae,
        "candidate_validation_gap_mae_eV": candidate_mae,
        "candidate_minus_baseline_eV": candidate_mae - baseline_mae if comparable else None,
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
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--input-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit, args.mode, args.input_sha)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
