"""Strict acceptance and paired comparison for architecture screens."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable

import torch

from .artifact_acceptance import sha256_file
from .distillation import atomic_json_write


TARGETS = ("HOMO", "LUMO", "Gap")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _artifact_path(run_dir: Path, artifact_name: str) -> Path:
    names = {
        "model": "model.pt",
        "last_checkpoint": "training_state.pt",
        "metrics": "metrics.json",
        "predictions": "test_predictions.pt",
    }
    return run_dir / names[artifact_name]


def _mae(targets: torch.Tensor, predictions: torch.Tensor) -> list[float]:
    values = (targets - predictions).abs().mean(dim=0).tolist()
    return [float(value) for value in values]


def _accept_run(
    run_dir: Path,
    *,
    kind: str,
    seed: int,
    graph_sha256: str,
    split_sha256: str,
    target_mode: str = "all",
) -> tuple[dict, dict]:
    manifest = _read_json(run_dir / "completion_manifest.json")
    metrics = _read_json(run_dir / "metrics.json")
    checkpoint = torch.load(
        run_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    predictions = torch.load(
        run_dir / "test_predictions.pt", map_location="cpu", weights_only=False
    )

    _require(manifest.get("status") == "accepted", f"{run_dir}: rejected manifest")
    for payload_name, payload in (
        ("manifest", manifest),
        ("metrics", metrics),
        ("checkpoint", checkpoint),
        ("predictions", predictions),
    ):
        _require(payload.get("kind") == kind, f"{run_dir}: {payload_name} kind differs")
        _require(int(payload.get("seed", -1)) == seed, f"{run_dir}: {payload_name} seed differs")
        _require(
            payload.get("target_mode", "all") == target_mode,
            f"{run_dir}: {payload_name} target mode differs",
        )

    _require(manifest.get("graph_sha256") == graph_sha256, f"{run_dir}: graph hash differs")
    _require(manifest.get("split_sha256") == split_sha256, f"{run_dir}: split hash differs")
    for payload_name, payload in (
        ("metrics", metrics),
        ("checkpoint", checkpoint),
        ("predictions", predictions),
    ):
        _require(
            payload.get("split_contract", {}).get("sha256") == split_sha256,
            f"{run_dir}: {payload_name} split hash differs",
        )

    for artifact_name, declared in manifest["artifacts"].items():
        path = _artifact_path(run_dir, artifact_name)
        _require(path.is_file(), f"{run_dir}: missing {artifact_name}")
        _require(path.stat().st_size == int(declared["bytes"]), f"{path}: byte count differs")
        _require(sha256_file(path) == declared["sha256"], f"{path}: SHA256 differs")

    if kind in {
        "structural_gps",
        "normalized_structural_gps",
        "gated_structural_gps",
        "edge_state_structural_gps",
    }:
        graph_contract = metrics.get("graph_contract", {})
        _require(graph_contract.get("sha256") == graph_sha256, f"{run_dir}: RWSE hash differs")
        _require(int(graph_contract.get("rwse_dim", -1)) == 16, f"{run_dir}: RWSE dimension differs")

    source_idx = predictions.get("source_idx")
    targets = predictions.get("targets")
    predicted = predictions.get("predictions")
    _require(
        all(isinstance(value, torch.Tensor) for value in (source_idx, targets, predicted)),
        f"{run_dir}: prediction tensors are missing",
    )
    expected_rows = int(metrics["split_contract"]["rows"]["test"])
    target_names = ("Gap",) if target_mode == "gap" else TARGETS
    expected_targets = len(target_names)
    _require(tuple(source_idx.shape) == (expected_rows,), f"{run_dir}: source_idx shape differs")
    _require(
        tuple(targets.shape) == (expected_rows, expected_targets),
        f"{run_dir}: target shape differs",
    )
    _require(predicted.shape == targets.shape, f"{run_dir}: prediction shape differs")
    _require(source_idx.unique().numel() == expected_rows, f"{run_dir}: duplicate source_idx")
    _require(torch.isfinite(targets).all().item(), f"{run_dir}: non-finite targets")
    _require(torch.isfinite(predicted).all().item(), f"{run_dir}: non-finite predictions")
    _require(int(metrics.get("best_epoch", -1)) >= 0, f"{run_dir}: no best epoch")
    _require(int(checkpoint.get("next_epoch", 0)) > 0, f"{run_dir}: empty checkpoint")

    recomputed = _mae(targets, predicted)
    recorded = [float(metrics["test_metrics"][target]["mae"]) for target in target_names]
    for target, actual, expected in zip(target_names, recomputed, recorded):
        _require(math.isclose(actual, expected, abs_tol=1e-7), f"{run_dir}: {target} MAE differs")
    average = fmean(recomputed)
    _require(
        math.isclose(average, float(metrics["test_metrics"]["average"]["mae"]), abs_tol=1e-7),
        f"{run_dir}: average MAE differs",
    )

    accepted = {
        "seed": seed,
        "kind": kind,
        "parameters": int(metrics["n_params"]),
        "training_time_s": float(metrics["training_time_s"]),
        "best_epoch": int(metrics["best_epoch"]),
        "validation_average_mae_eV": float(metrics["best_val_mae"]),
        "target_mode": target_mode,
        "test_mae_eV": dict(zip((*target_names, "average"), (*recomputed, average))),
        "test_rows": expected_rows,
        "artifacts": {
            name: {
                "bytes": int(declared["bytes"]),
                "sha256": declared["sha256"],
            }
            for name, declared in manifest["artifacts"].items()
        },
    }
    return accepted, predictions


def _summary(values: Iterable[float]) -> dict:
    values = [float(value) for value in values]
    return {
        "mean": fmean(values),
        "population_sd": pstdev(values),
        "values": values,
    }


def _ensemble_metrics(
    predictions: list[dict], target_mode: str = "all"
) -> dict[str, float]:
    stacked = torch.stack([payload["predictions"] for payload in predictions])
    values = _mae(predictions[0]["targets"], stacked.mean(dim=0))
    target_names = ("Gap",) if target_mode == "gap" else TARGETS
    return dict(zip((*target_names, "average"), (*values, fmean(values))))


def accept_paired_architecture_screen(
    *,
    control_dir: Path,
    structural_dir: Path,
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    gate_eV: float = 0.001,
) -> dict:
    """Accept two Kaggle runs and compare paired seeds on one immutable split."""
    base_hash = sha256_file(base_graph)
    rwse_hash = sha256_file(rwse_graph)
    split_hash = sha256_file(split_csv)
    expected_hashes = {
        "base_graph": base_hash,
        "rwse_graph": rwse_hash,
        "split_csv": split_hash,
    }

    variant_specs = {
        "control": (control_dir, "gps", base_hash),
        "structural": (structural_dir, "structural_gps", rwse_hash),
    }
    accepted_runs: dict[str, list[dict]] = {}
    prediction_payloads: dict[str, list[dict]] = {}
    for variant, (root, kind, graph_hash) in variant_specs.items():
        kernel = _read_json(root / "kernel_completion_manifest.json")
        preflight = _read_json(root / "preflight.json")
        _require(kernel.get("status") == "accepted", f"{root}: kernel not accepted")
        _require(tuple(kernel.get("seeds", ())) == seeds, f"{root}: kernel seeds differ")
        _require(kernel.get("input_sha256") == expected_hashes, f"{root}: input hashes differ")
        _require(preflight.get("status") == "accepted", f"{root}: preflight not accepted")
        _require(int(preflight.get("rows", -1)) == 120_000, f"{root}: preflight rows differ")
        _require(preflight.get("shared_initialization_exact") is True, f"{root}: initialization differs")
        _require(preflight.get("forward_backward_finite") is True, f"{root}: non-finite preflight")
        runs, payloads = [], []
        for seed in seeds:
            accepted, payload = _accept_run(
                root / f"seed{seed}",
                kind=kind,
                seed=seed,
                graph_sha256=graph_hash,
                split_sha256=split_hash,
            )
            runs.append(accepted)
            payloads.append(payload)
        accepted_runs[variant] = runs
        prediction_payloads[variant] = payloads

    reference = prediction_payloads["control"][0]
    for variant, payloads in prediction_payloads.items():
        for seed, payload in zip(seeds, payloads):
            _require(torch.equal(payload["source_idx"], reference["source_idx"]), f"{variant}/{seed}: test identity differs")
            _require(torch.equal(payload["targets"], reference["targets"]), f"{variant}/{seed}: targets differ")

    comparisons = []
    for seed, control, structural in zip(
        seeds, accepted_runs["control"], accepted_runs["structural"]
    ):
        comparisons.append(
            {
                "seed": seed,
                "validation_delta_eV": structural["validation_average_mae_eV"]
                - control["validation_average_mae_eV"],
                "test_delta_eV": {
                    target: structural["test_mae_eV"][target]
                    - control["test_mae_eV"][target]
                    for target in (*TARGETS, "average")
                },
            }
        )

    validation_delta = _summary(item["validation_delta_eV"] for item in comparisons)
    test_deltas = {
        target: _summary(item["test_delta_eV"][target] for item in comparisons)
        for target in (*TARGETS, "average")
    }
    direction_consistent = all(item["validation_delta_eV"] < 0 for item in comparisons)
    gate_passed = direction_consistent and validation_delta["mean"] <= -gate_eV
    report = {
        "format": "molgap-paired-architecture-screen-acceptance-v1",
        "status": "accepted",
        "input_sha256": expected_hashes,
        "seeds": list(seeds),
        "test_alignment": {
            "rows": int(reference["source_idx"].numel()),
            "source_idx_exact": True,
            "targets_exact": True,
        },
        "runs": accepted_runs,
        "paired_comparison": comparisons,
        "aggregate": {
            "validation_delta_eV": validation_delta,
            "test_delta_eV": test_deltas,
            "control_equal_seed_ensemble_test_mae_eV": _ensemble_metrics(
                prediction_payloads["control"]
            ),
            "structural_equal_seed_ensemble_test_mae_eV": _ensemble_metrics(
                prediction_payloads["structural"]
            ),
        },
        "selection_gate": {
            "required_improvement_eV": gate_eV,
            "direction_consistent": direction_consistent,
            "passed": gate_passed,
        },
    }
    atomic_json_write(report, output_path)
    return report


def accept_gap_rwse_screen(
    *,
    baseline_dir: Path,
    gap_only_dir: Path,
    normalized_dir: Path,
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    gate_eV: float = 0.001,
) -> dict:
    """Compare three-output Structural GPS, Gap-only, and normalized Gap-only RWSE."""
    hashes = {
        "base_graph": sha256_file(base_graph),
        "rwse_graph": sha256_file(rwse_graph),
        "split_csv": sha256_file(split_csv),
    }
    specs = {
        "three_output_baseline": (
            baseline_dir,
            "structural_gps",
            "all",
        ),
        "gap_only": (gap_only_dir, "structural_gps", "gap"),
        "normalized_gap": (
            normalized_dir,
            "normalized_structural_gps",
            "gap",
        ),
    }
    accepted_runs: dict[str, list[dict]] = {}
    prediction_payloads: dict[str, list[dict]] = {}
    for variant, (root, kind, target_mode) in specs.items():
        kernel = _read_json(root / "kernel_completion_manifest.json")
        preflight = _read_json(root / "preflight.json")
        _require(kernel.get("status") == "accepted", f"{root}: kernel not accepted")
        _require(tuple(kernel.get("seeds", ())) == seeds, f"{root}: kernel seeds differ")
        _require(kernel.get("input_sha256") == hashes, f"{root}: input hashes differ")
        _require(preflight.get("status") == "accepted", f"{root}: preflight not accepted")
        _require(int(preflight.get("rows", -1)) == 120_000, f"{root}: row count differs")
        if variant == "normalized_gap":
            _require(
                preflight.get("normalized_shared_initialization_exact") is True,
                f"{root}: normalized initialization differs",
            )
        runs, payloads = [], []
        for seed in seeds:
            accepted, payload = _accept_run(
                root / f"seed{seed}",
                kind=kind,
                seed=seed,
                graph_sha256=hashes["rwse_graph"],
                split_sha256=hashes["split_csv"],
                target_mode=target_mode,
            )
            runs.append(accepted)
            payloads.append(payload)
        accepted_runs[variant] = runs
        prediction_payloads[variant] = payloads

    baseline_reference = prediction_payloads["three_output_baseline"][0]
    reference_gap = baseline_reference["targets"][:, 2:3]
    for variant, payloads in prediction_payloads.items():
        for seed, payload in zip(seeds, payloads):
            _require(
                torch.equal(payload["source_idx"], baseline_reference["source_idx"]),
                f"{variant}/{seed}: test identity differs",
            )
            candidate_target = (
                payload["targets"][:, 2:3]
                if variant == "three_output_baseline"
                else payload["targets"]
            )
            _require(
                torch.equal(candidate_target, reference_gap),
                f"{variant}/{seed}: Gap targets differ",
            )

    comparisons = []
    for position, seed in enumerate(seeds):
        baseline = accepted_runs["three_output_baseline"][position]
        gap_only = accepted_runs["gap_only"][position]
        normalized = accepted_runs["normalized_gap"][position]
        comparisons.append(
            {
                "seed": seed,
                "gap_only_minus_three_output_test_gap_eV": (
                    gap_only["test_mae_eV"]["Gap"]
                    - baseline["test_mae_eV"]["Gap"]
                ),
                "normalized_minus_gap_only_validation_eV": (
                    normalized["validation_average_mae_eV"]
                    - gap_only["validation_average_mae_eV"]
                ),
                "normalized_minus_gap_only_test_gap_eV": (
                    normalized["test_mae_eV"]["Gap"]
                    - gap_only["test_mae_eV"]["Gap"]
                ),
                "normalized_minus_three_output_test_gap_eV": (
                    normalized["test_mae_eV"]["Gap"]
                    - baseline["test_mae_eV"]["Gap"]
                ),
            }
        )
    validation_delta = _summary(
        item["normalized_minus_gap_only_validation_eV"] for item in comparisons
    )
    normalized_direction_consistent = all(
        item["normalized_minus_gap_only_validation_eV"] < 0
        for item in comparisons
    )
    report = {
        "format": "molgap-gap-rwse-screen-acceptance-v1",
        "status": "accepted",
        "input_sha256": hashes,
        "seeds": list(seeds),
        "runs": accepted_runs,
        "paired_comparison": comparisons,
        "aggregate": {
            key: _summary(item[key] for item in comparisons)
            for key in comparisons[0]
            if key != "seed"
        },
        "equal_seed_ensemble_test_gap_mae_eV": {
            "three_output_baseline": _ensemble_metrics(
                [
                    {
                        **payload,
                        "targets": payload["targets"][:, 2:3],
                        "predictions": payload["predictions"][:, 2:3],
                    }
                    for payload in prediction_payloads["three_output_baseline"]
                ],
                "gap",
            )["Gap"],
            "gap_only": _ensemble_metrics(
                prediction_payloads["gap_only"], "gap"
            )["Gap"],
            "normalized_gap": _ensemble_metrics(
                prediction_payloads["normalized_gap"], "gap"
            )["Gap"],
        },
        "normalized_rwse_gate": {
            "required_validation_improvement_eV": gate_eV,
            "direction_consistent": normalized_direction_consistent,
            "passed": (
                normalized_direction_consistent
                and validation_delta["mean"] <= -gate_eV
            ),
        },
    }
    atomic_json_write(report, output_path)
    return report


def _accept_structural_feasibility(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    candidate_kind: str,
    output_shape_key: str,
    report_format: str,
    seed: int = 42,
    gate_eV: float = 0.0,
    max_training_time_s: float = 4_500.0,
) -> dict:
    """Accept one Structural GPS derivative and apply its precision/runtime gate."""
    hashes = {
        "base_graph": sha256_file(base_graph),
        "rwse_graph": sha256_file(rwse_graph),
        "split_csv": sha256_file(split_csv),
    }
    baseline_kernel = _read_json(baseline_dir / "kernel_completion_manifest.json")
    candidate_kernel = _read_json(candidate_dir / "kernel_completion_manifest.json")
    candidate_preflight = _read_json(candidate_dir / "preflight.json")
    _require(baseline_kernel.get("status") == "accepted", "baseline kernel not accepted")
    _require(candidate_kernel.get("status") == "accepted", "candidate kernel not accepted")
    _require(candidate_kernel.get("seeds") == [seed], "candidate seed contract differs")
    _require(candidate_kernel.get("input_sha256") == hashes, "candidate input hashes differ")
    _require(candidate_preflight.get("status") == "accepted", "candidate preflight rejected")
    _require(int(candidate_preflight.get("rows", -1)) == 120_000, "candidate rows differ")
    _require(
        candidate_preflight.get("forward_backward_finite") is True,
        "candidate preflight is non-finite",
    )
    _require(
        candidate_preflight.get(output_shape_key) == [8, 3],
        "candidate output shape differs",
    )

    baseline, baseline_predictions = _accept_run(
        baseline_dir / f"seed{seed}",
        kind="structural_gps",
        seed=seed,
        graph_sha256=hashes["rwse_graph"],
        split_sha256=hashes["split_csv"],
    )
    candidate, candidate_predictions = _accept_run(
        candidate_dir / f"seed{seed}",
        kind=candidate_kind,
        seed=seed,
        graph_sha256=hashes["rwse_graph"],
        split_sha256=hashes["split_csv"],
    )
    _require(
        torch.equal(candidate_predictions["source_idx"], baseline_predictions["source_idx"]),
        "candidate test identity differs",
    )
    _require(
        torch.equal(candidate_predictions["targets"], baseline_predictions["targets"]),
        "candidate test targets differ",
    )

    validation_delta = (
        candidate["validation_average_mae_eV"]
        - baseline["validation_average_mae_eV"]
    )
    report = {
        "format": report_format,
        "status": "accepted",
        "input_sha256": hashes,
        "seed": seed,
        "runs": {"baseline": baseline, "candidate": candidate},
        "paired_delta_eV": {
            "validation_average": validation_delta,
            "test": {
                target: candidate["test_mae_eV"][target]
                - baseline["test_mae_eV"][target]
                for target in (*TARGETS, "average")
            },
        },
        "feasibility_gate": {
            "validation_improved": validation_delta < 0.0,
            "required_validation_improvement_eV": float(gate_eV),
            "max_training_time_s": float(max_training_time_s),
            "training_time_passed": (
                candidate["training_time_s"] <= max_training_time_s
            ),
            "passed": (
                validation_delta <= -gate_eV
                and candidate["training_time_s"] <= max_training_time_s
            ),
        },
    }
    atomic_json_write(report, output_path)
    return report


def accept_gated_structural_feasibility(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    seed: int = 42,
    max_training_time_s: float = 4_500.0,
) -> dict:
    """Accept one gated-local seed and apply its precision/runtime gate."""
    return _accept_structural_feasibility(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        base_graph=base_graph,
        rwse_graph=rwse_graph,
        split_csv=split_csv,
        output_path=output_path,
        candidate_kind="gated_structural_gps",
        output_shape_key="gated_output_shape",
        report_format="molgap-gated-structural-feasibility-acceptance-v1",
        seed=seed,
        max_training_time_s=max_training_time_s,
    )


def accept_edge_state_structural_feasibility(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    seed: int = 42,
    gate_eV: float = 0.001,
    max_training_time_s: float = 4_500.0,
) -> dict:
    """Accept one persistent-edge seed against the paired Structural GPS seed."""
    return _accept_structural_feasibility(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        base_graph=base_graph,
        rwse_graph=rwse_graph,
        split_csv=split_csv,
        output_path=output_path,
        candidate_kind="edge_state_structural_gps",
        output_shape_key="edge_state_output_shape",
        report_format="molgap-edge-state-structural-feasibility-acceptance-v1",
        seed=seed,
        gate_eV=gate_eV,
        max_training_time_s=max_training_time_s,
    )


def accept_gated_structural_multiseed(
    *,
    baseline_dir: Path,
    candidate_dirs: dict[int, Path],
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    gate_eV: float = 0.001,
    max_training_time_s: float = 4_500.0,
) -> dict:
    """Accept independently packaged gated seeds and apply the final 100K gate."""
    return _accept_structural_multiseed(
        baseline_dir=baseline_dir,
        candidate_dirs=candidate_dirs,
        base_graph=base_graph,
        rwse_graph=rwse_graph,
        split_csv=split_csv,
        output_path=output_path,
        seeds=seeds,
        gate_eV=gate_eV,
        max_training_time_s=max_training_time_s,
        candidate_kind="gated_structural_gps",
        output_shape_key="gated_output_shape",
        report_format="molgap-gated-structural-multiseed-acceptance-v1",
    )


def accept_edge_state_structural_multiseed(
    *,
    baseline_dir: Path,
    candidate_dirs: dict[int, Path],
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    gate_eV: float = 0.001,
    max_training_time_s: float = 4_500.0,
) -> dict:
    """Accept persistent-edge confirmation seeds and apply the final 100K gate."""
    return _accept_structural_multiseed(
        baseline_dir=baseline_dir,
        candidate_dirs=candidate_dirs,
        base_graph=base_graph,
        rwse_graph=rwse_graph,
        split_csv=split_csv,
        output_path=output_path,
        seeds=seeds,
        gate_eV=gate_eV,
        max_training_time_s=max_training_time_s,
        candidate_kind="edge_state_structural_gps",
        output_shape_key="edge_state_output_shape",
        report_format="molgap-edge-state-structural-multiseed-acceptance-v1",
    )


def _accept_structural_multiseed(
    *,
    baseline_dir: Path,
    candidate_dirs: dict[int, Path],
    base_graph: Path,
    rwse_graph: Path,
    split_csv: Path,
    output_path: Path,
    seeds: tuple[int, ...],
    gate_eV: float,
    max_training_time_s: float,
    candidate_kind: str,
    output_shape_key: str,
    report_format: str,
) -> dict:
    _require(set(candidate_dirs) == set(seeds), "candidate seed directories differ")
    hashes = {
        "base_graph": sha256_file(base_graph),
        "rwse_graph": sha256_file(rwse_graph),
        "split_csv": sha256_file(split_csv),
    }
    baseline_kernel = _read_json(baseline_dir / "kernel_completion_manifest.json")
    _require(baseline_kernel.get("status") == "accepted", "baseline kernel not accepted")
    _require(baseline_kernel.get("input_sha256") == hashes, "baseline input hashes differ")

    baseline_runs, candidate_runs = [], []
    baseline_predictions, candidate_predictions = [], []
    for seed in seeds:
        root = candidate_dirs[seed]
        kernel = _read_json(root / "kernel_completion_manifest.json")
        preflight = _read_json(root / "preflight.json")
        _require(kernel.get("status") == "accepted", f"seed {seed}: kernel not accepted")
        _require(kernel.get("seeds") == [seed], f"seed {seed}: seed contract differs")
        _require(kernel.get("input_sha256") == hashes, f"seed {seed}: input hashes differ")
        _require(preflight.get("status") == "accepted", f"seed {seed}: preflight rejected")
        _require(int(preflight.get("rows", -1)) == 120_000, f"seed {seed}: rows differ")
        _require(
            preflight.get("forward_backward_finite") is True,
            f"seed {seed}: non-finite preflight",
        )
        _require(
            preflight.get(output_shape_key) == [8, 3],
            f"seed {seed}: output shape differs",
        )
        baseline, baseline_payload = _accept_run(
            baseline_dir / f"seed{seed}",
            kind="structural_gps",
            seed=seed,
            graph_sha256=hashes["rwse_graph"],
            split_sha256=hashes["split_csv"],
        )
        candidate, candidate_payload = _accept_run(
            root / f"seed{seed}",
            kind=candidate_kind,
            seed=seed,
            graph_sha256=hashes["rwse_graph"],
            split_sha256=hashes["split_csv"],
        )
        _require(
            torch.equal(candidate_payload["source_idx"], baseline_payload["source_idx"]),
            f"seed {seed}: test identity differs",
        )
        _require(
            torch.equal(candidate_payload["targets"], baseline_payload["targets"]),
            f"seed {seed}: targets differ",
        )
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)
        baseline_predictions.append(baseline_payload)
        candidate_predictions.append(candidate_payload)

    comparisons = []
    for seed, baseline, candidate in zip(seeds, baseline_runs, candidate_runs):
        comparisons.append(
            {
                "seed": seed,
                "validation_delta_eV": (
                    candidate["validation_average_mae_eV"]
                    - baseline["validation_average_mae_eV"]
                ),
                "test_delta_eV": {
                    target: candidate["test_mae_eV"][target]
                    - baseline["test_mae_eV"][target]
                    for target in (*TARGETS, "average")
                },
                "training_time_s": candidate["training_time_s"],
            }
        )
    validation = _summary(item["validation_delta_eV"] for item in comparisons)
    direction_consistent = all(item["validation_delta_eV"] < 0.0 for item in comparisons)
    time_passed = all(
        item["training_time_s"] <= max_training_time_s for item in comparisons
    )
    report = {
        "format": report_format,
        "status": "accepted",
        "input_sha256": hashes,
        "seeds": list(seeds),
        "runs": {"baseline": baseline_runs, "candidate": candidate_runs},
        "paired_comparison": comparisons,
        "aggregate": {
            "validation_delta_eV": validation,
            "test_delta_eV": {
                target: _summary(
                    item["test_delta_eV"][target] for item in comparisons
                )
                for target in (*TARGETS, "average")
            },
            "baseline_equal_seed_ensemble_test_mae_eV": _ensemble_metrics(
                baseline_predictions
            ),
            "candidate_equal_seed_ensemble_test_mae_eV": _ensemble_metrics(
                candidate_predictions
            ),
        },
        "selection_gate": {
            "required_validation_improvement_eV": float(gate_eV),
            "direction_consistent": direction_consistent,
            "max_training_time_s": float(max_training_time_s),
            "training_time_passed": time_passed,
            "passed": (
                direction_consistent
                and validation["mean"] <= -gate_eV
                and time_passed
            ),
        },
    }
    atomic_json_write(report, output_path)
    return report
