"""Mechanical acceptance for the GraphState64/128 seed-42 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9_w128"
CANDIDATES = (BASELINE, CANDIDATE)
PARAMETERS = {BASELINE: 3_665_809, CANDIDATE: 3_803_985}
WIDTHS = {BASELINE: 64, CANDIDATE: 128}
GEOMETRY_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
T4_MEMORY_BYTES = 16 * 1024**3
DUAL_T4_EXECUTION = "dual_t4_candidate_parallel"
SPLIT_EXECUTION = "two_isolated_single_gpu_parallel_kernels"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str) -> dict:
    import numpy as np
    import torch

    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    required = {
        "complete": True,
        "run_mode": "graph_state_width",
        "source_commit": expected_source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_SHA,
        "input_cache_aggregate_sha256": GEOMETRY_SHA,
        "seed": 42,
        "candidates": list(CANDIDATES),
        "device_assignments": {"0": [BASELINE], "1": [CANDIDATE]},
        "search_budget_s": 14_400,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
        "seed43_44_submitted": False,
        "full_data_authorized": False,
    }
    for key, expected in required.items():
        require(selection.get(key) == expected, key)
    execution = selection.get("execution")
    require(execution in (DUAL_T4_EXECUTION, SPLIT_EXECUTION), "execution")
    gpu_names = selection.get("gpu_names", [])
    if execution == DUAL_T4_EXECUTION:
        require(
            len(gpu_names) == 2 and all("T4" in name for name in gpu_names),
            "dual T4 allocation",
        )
    elif execution == SPLIT_EXECUTION:
        require(
            len(gpu_names) == 2
            and all(isinstance(name, str) and name for name in gpu_names)
            and len(set(gpu_names)) == 1,
            "split matching GPU allocation",
        )
        source_versions = selection.get("source_kernel_versions", {})
        require(set(source_versions) == set(CANDIDATES), "split source kernel identities")
        require(
            all(isinstance(value, str) and ":v" in value for value in source_versions.values()),
            "split source kernel versions",
        )

    preflight = selection.get("preflight", [])
    require(len(preflight) == 2, "preflight count")
    peak_memory: dict[str, int] = {}
    for row in preflight:
        identity = row.get("candidate")
        require(identity in CANDIDATES, f"preflight identity {identity}")
        require(row.get("parameter_count") == PARAMETERS.get(identity), f"parameters {identity}")
        require(row.get("global_attention_blocks") == [], f"attention {identity}")
        require(row.get("graph_state_present") is True, f"GraphState {identity}")
        require(row.get("graph_state_channels") == WIDTHS.get(identity), f"width {identity}")
        require(row.get("shared_parameter_mismatches") == [], f"shared init {identity}")
        shape_changes = row.get("shared_parameter_shape_differences", [])
        if identity == BASELINE:
            require(shape_changes == [], "baseline shape changes")
        else:
            require(bool(shape_changes), "candidate width shape changes")
            require(
                all(str(name).startswith("graph_context.") for name in shape_changes),
                "candidate non-GraphState shape changes",
            )
        require(
            all(row.get(key) is True for key in ("finite_prediction", "finite_loss", "finite_gradients")),
            f"finite {identity}",
        )
        memory = row.get("peak_memory_bytes")
        require(
            isinstance(memory, int) and 0 < memory <= 0.85 * T4_MEMORY_BYTES,
            f"memory {identity}",
        )
        if isinstance(memory, int):
            peak_memory[identity] = memory

    runs = selection.get("runs", [])
    require(len(runs) == 2, "run count")
    by_name = {row.get("candidate"): row for row in runs}
    payloads = {}
    for identity in CANDIDATES:
        path = root / "results" / identity / "metrics.json"
        require(path.is_file(), f"metrics {identity}")
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        require(metrics == by_name.get(identity), f"selection metrics {identity}")
        require(metrics.get("complete") is True, f"complete {identity}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {identity}")
        require(metrics.get("input_cache_aggregate_sha256") == GEOMETRY_SHA, f"cache {identity}")
        require(metrics.get("parameter_count") == PARAMETERS[identity], f"parameters {identity}")
        require(metrics.get("gpu") in gpu_names, f"GPU identity {identity}")
        require(metrics.get("validation_rows") == 10_000, f"rows {identity}")
        contract = metrics.get("contract", {})
        require(contract.get("graph_state_channels") == WIDTHS[identity], f"contract width {identity}")
        require(contract.get("global_mechanism") == "gated_graph_state", f"mechanism {identity}")
        require(contract.get("global_attention_blocks") == [], f"contract attention {identity}")
        require(contract.get("max_epochs") == 40, f"epochs {identity}")
        require(contract.get("precision") == "fp32", f"precision {identity}")
        require(metrics.get("official_validation_role_read") is False, f"official valid {identity}")
        require(metrics.get("test_dev_role_read") is False, f"test-dev {identity}")

        artifacts = metrics.get("artifacts", {})
        artifact_paths = {}
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            relative = artifacts.get(name)
            artifact = root / relative if isinstance(relative, str) else root / "__missing__"
            artifact_paths[name] = artifact
            require(artifact.is_file(), f"missing {name} {identity}")
            if artifact.is_file():
                require(
                    sha256_file(artifact) == artifacts.get(f"{name}_sha256"),
                    f"hash {name} {identity}",
                )
        if not all(path.is_file() for path in artifact_paths.values()):
            continue

        payload = torch.load(
            artifact_paths["validation_payload"], map_location="cpu", weights_only=False
        )
        rows = payload.get("row_index")
        targets = payload.get("target_eV")
        predictions = payload.get("prediction_eV")
        require(payload.get("candidate") == identity, f"payload identity {identity}")
        require(payload.get("source_commit") == expected_source_commit, f"payload source {identity}")
        require(payload.get("seed") == 42, f"payload seed {identity}")
        require(all(isinstance(value, torch.Tensor) for value in (rows, targets, predictions)), f"payload tensors {identity}")
        if all(isinstance(value, torch.Tensor) for value in (rows, targets, predictions)):
            rows = rows.reshape(-1).cpu()
            targets = targets.reshape(-1).float().cpu()
            predictions = predictions.reshape(-1).float().cpu()
            require(rows.numel() == 10_000 and torch.unique(rows).numel() == 10_000, f"payload rows {identity}")
            require(bool(torch.isfinite(targets).all()), f"finite targets {identity}")
            require(bool(torch.isfinite(predictions).all()), f"finite predictions {identity}")
            recomputed = float(torch.mean(torch.abs(predictions - targets)))
            require(
                math.isclose(recomputed, metrics.get("validation_gap_mae_eV", math.inf), rel_tol=0, abs_tol=5e-8),
                f"recomputed MAE {identity}",
            )
            payloads[identity] = (rows, targets, predictions, recomputed)

        best = torch.load(artifact_paths["best_model"], map_location="cpu", weights_only=False)
        checkpoint = torch.load(artifact_paths["checkpoint"], map_location="cpu", weights_only=False)
        for label, artifact in (("best", best), ("checkpoint", checkpoint)):
            require(artifact.get("candidate") == identity, f"{label} identity {identity}")
            require(artifact.get("source_commit") == expected_source_commit, f"{label} source {identity}")
            require(artifact.get("seed") == 42, f"{label} seed {identity}")
            require(isinstance(artifact.get("model"), dict) and bool(artifact["model"]), f"{label} state {identity}")

    comparable = set(payloads) == set(CANDIDATES)
    require(comparable, "paired payloads")
    baseline_mae = candidate_mae = delta = throughput_ratio = None
    ci95 = None
    if comparable:
        rows0, targets0, predictions0, baseline_mae = payloads[BASELINE]
        rows1, targets1, predictions1, candidate_mae = payloads[CANDIDATE]
        require(torch.equal(rows0, rows1), "paired rows")
        require(torch.equal(targets0, targets1), "paired targets")
        paired_delta = torch.abs(predictions1 - targets1) - torch.abs(predictions0 - targets0)
        delta = float(paired_delta.mean())
        values = paired_delta.numpy()
        rng = np.random.default_rng(42)
        bootstrap = np.empty(1000, dtype=np.float64)
        for index in range(bootstrap.size):
            sample = rng.integers(0, values.size, values.size)
            bootstrap[index] = values[sample].mean()
        ci95 = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
        throughput_ratio = (
            by_name[CANDIDATE]["mean_throughput_graphs_per_s"]
            / by_name[BASELINE]["mean_throughput_graphs_per_s"]
        )

    improves = comparable and candidate_mae < baseline_mae
    material = comparable and baseline_mae - candidate_mae >= 0.001
    throughput_gate = throughput_ratio is not None and throughput_ratio >= 0.70
    selected = CANDIDATE if improves else BASELINE
    require(selection.get("selected_candidate") == selected, "selection")
    require(selection.get("selected_strictly_improves_baseline") is improves, "selection gate")
    result = {
        "format": "molgap-pcqm-gap100k-graph-state-width-acceptance-v2",
        "accepted": not errors,
        "errors": errors,
        "execution": execution,
        "gpu_names": gpu_names,
        "source_commit": expected_source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_SHA,
        "seed": 42,
        "baseline_validation_gap_mae_eV": baseline_mae,
        "candidate_validation_gap_mae_eV": candidate_mae,
        "candidate_minus_baseline_eV": delta,
        "paired_delta_bootstrap_ci95_eV": ci95,
        "material_gain_at_least_0_001_eV": material,
        "candidate_to_baseline_throughput_ratio": throughput_ratio,
        "throughput_gate_at_least_0_70x": throughput_gate,
        "scientific_gate_passed": bool(material and throughput_gate),
        "preflight_peak_memory_bytes": peak_memory,
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
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
