"""No-model acceptance for the three seed-42 geometry candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CANDIDATES = (
    "ogb_distance_triangle_edge_state_gps9",
    "ogb_angle_triangle_edge_state_gps9",
    "ogb_distance_angle_triangle_edge_state_gps9",
)
COMPARATOR = "ogb_sparse_triangle_edge_state_gps9"
COMPARATOR_MAE = 0.13790177369117737
CONTRACT = {
    "batch_size": 48,
    "learning_rate": 1.6e-4,
    "weight_decay": 1.0e-6,
    "max_epochs": 40,
    "patience": 8,
    "precision": "fp32",
    "target": "gap",
    "geometry": "ETKDGv3+MMFF94s-single-conformer-bottom-fusion",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(
    root: Path,
    expected_source_commit: str,
    expected_geometry_sha256: str,
) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        selection.get("format")
        == "molgap-pcqm-gap100k-geometry-bottom-fusion-screen-v1",
        "format",
    )
    require(selection.get("complete") is True, "complete")
    require(selection.get("source_commit") == expected_source_commit, "source")
    require(
        selection.get("geometry_cache_aggregate_sha256")
        == expected_geometry_sha256,
        "geometry cache SHA",
    )
    require(selection.get("seed") == 42, "seed")
    require(selection.get("candidates") == list(CANDIDATES), "candidates")
    require(selection.get("search_budget_s") == 34_200, "budget")
    require(selection.get("official_validation_role_read") is False, "official valid")
    require(selection.get("test_dev_role_read") is False, "test-dev")
    preflight = selection.get("preflight", [])
    require(isinstance(preflight, list) and len(preflight) == 3, "preflight")
    for row in preflight if isinstance(preflight, list) else []:
        require(row.get("candidate") in CANDIDATES, "preflight candidate")
        require(
            all(
                row.get(key) is True
                for key in ("finite_prediction", "finite_loss", "finite_gradients")
            ),
            f"preflight finite {row.get('candidate')}",
        )
        require(
            isinstance(row.get("parameter_count"), int)
            and 0 < row["parameter_count"] <= 5_200_000,
            f"preflight parameters {row.get('candidate')}",
        )

    runs = selection.get("runs", [])
    require(isinstance(runs, list) and len(runs) == 3, "run count")
    accepted_rows = []
    row_hashes = set()
    target_hashes = set()
    for candidate in CANDIDATES:
        metrics_path = root / "results" / candidate / "metrics.json"
        require(metrics_path.is_file(), f"metrics {candidate}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics in runs, f"selection metrics {candidate}")
        require(
            metrics.get("format") == "molgap-pcqm-gap100k-geometry-candidate-v1",
            f"format {candidate}",
        )
        require(metrics.get("complete") is True, f"complete {candidate}")
        require(metrics.get("candidate") == candidate, f"identity {candidate}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {candidate}")
        require(metrics.get("seed") == 42, f"seed {candidate}")
        require(metrics.get("contract") == CONTRACT, f"contract {candidate}")
        require(metrics.get("validation_rows") == 10_000, f"rows {candidate}")
        value = metrics.get("validation_gap_mae_eV")
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"MAE {candidate}",
        )
        require(
            isinstance(metrics.get("parameter_count"), int)
            and 0 < metrics["parameter_count"] <= 5_200_000,
            f"parameters {candidate}",
        )
        row_hashes.add(metrics.get("validation_row_index_sha256"))
        target_hashes.add(metrics.get("validation_target_sha256"))
        artifacts = metrics.get("artifacts", {})
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            relative = artifacts.get(name)
            path = root / relative if isinstance(relative, str) else root / "__missing__"
            require(path.is_file(), f"missing {name} {candidate}")
            if path.is_file():
                require(
                    sha256_file(path) == artifacts.get(f"{name}_sha256"),
                    f"hash {name} {candidate}",
                )
        require(metrics.get("official_validation_role_read") is False, f"official {candidate}")
        require(metrics.get("test_dev_role_read") is False, f"test-dev {candidate}")
        accepted_rows.append(metrics)
    require(len(row_hashes) == 1, "row identity")
    require(len(target_hashes) == 1, "target identity")
    positive = [
        row for row in accepted_rows if row["validation_gap_mae_eV"] < COMPARATOR_MAE
    ]
    if positive:
        expected_selected = min(
            positive, key=lambda row: row["validation_gap_mae_eV"]
        )["candidate"]
        expected_improves = True
    else:
        expected_selected = COMPARATOR
        expected_improves = False
    require(selection.get("selected_candidate") == expected_selected, "selection")
    require(
        selection.get("selected_strictly_improves") is expected_improves,
        "positive gate",
    )
    result = {
        "format": "molgap-pcqm-gap100k-geometry-bottom-fusion-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "geometry_cache_aggregate_sha256": expected_geometry_sha256,
        "comparator": COMPARATOR,
        "comparator_validation_gap_mae_eV": COMPARATOR_MAE,
        "candidates": [
            {
                "candidate": row["candidate"],
                "parameter_count": row["parameter_count"],
                "best_epoch": row["best_epoch"],
                "validation_gap_mae_eV": row["validation_gap_mae_eV"],
                "mean_throughput_graphs_per_s": row[
                    "mean_throughput_graphs_per_s"
                ],
            }
            for row in accepted_rows
        ],
        "selected_candidate": expected_selected,
        "selected_strictly_improves": expected_improves,
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
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(
        args.root,
        args.expected_source_commit,
        args.expected_geometry_sha256,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
