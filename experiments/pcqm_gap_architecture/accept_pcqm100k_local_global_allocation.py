"""No-model acceptance for the seed-42 local/global allocation screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CANDIDATES = (
    "ogb_distance_angle_triangle_edge_state_gps9",
    "ogb_distance_angle_triangle_edge_state_sparse_gps369",
    "ogb_distance_angle_triangle_edge_state_graph_state9",
)
COMPARATOR = CANDIDATES[0]
FROZEN_COMPARATOR_MAE = 0.1353926807641983
EXPECTED_GLOBAL_BLOCKS = {
    CANDIDATES[0]: list(range(1, 10)),
    CANDIDATES[1]: [3, 6, 9],
    CANDIDATES[2]: [],
}
COMMON_CONTRACT = {
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
        == "molgap-pcqm-gap100k-local-global-allocation-screen-v1",
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
    require(selection.get("search_budget_s") == 39_600, "budget")
    require(selection.get("official_validation_role_read") is False, "official valid")
    require(selection.get("test_dev_role_read") is False, "test-dev")
    require(
        selection.get("molecular_research_server_accessed") is False,
        "molecular server",
    )
    preflight = selection.get("preflight", [])
    require(isinstance(preflight, list) and len(preflight) == 3, "preflight")
    for row in preflight if isinstance(preflight, list) else []:
        require(row.get("candidate") in CANDIDATES, "preflight candidate")
        require(
            row.get("global_attention_blocks")
            == EXPECTED_GLOBAL_BLOCKS.get(row.get("candidate")),
            f"preflight global blocks {row.get('candidate')}",
        )
        require(
            row.get("shared_parameter_mismatches") == [],
            f"shared initialization {row.get('candidate')}",
        )
        require(
            row.get("graph_state_present")
            == (row.get("candidate") == CANDIDATES[2]),
            f"graph state {row.get('candidate')}",
        )
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
            metrics.get("format") == "molgap-pcqm-gap100k-local-global-candidate-v1",
            f"format {candidate}",
        )
        require(metrics.get("complete") is True, f"complete {candidate}")
        require(metrics.get("candidate") == candidate, f"identity {candidate}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {candidate}")
        require(metrics.get("seed") == 42, f"seed {candidate}")
        contract = metrics.get("contract", {})
        for key, expected in COMMON_CONTRACT.items():
            require(contract.get(key) == expected, f"contract {key} {candidate}")
        require(
            contract.get("global_attention_blocks")
            == EXPECTED_GLOBAL_BLOCKS[candidate],
            f"global blocks {candidate}",
        )
        require(
            contract.get("global_mechanism")
            == ("gated_graph_state" if candidate == CANDIDATES[2] else "multihead_attention"),
            f"global mechanism {candidate}",
        )
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
    by_name = {row["candidate"]: row for row in accepted_rows}
    fresh_comparator_mae = by_name[COMPARATOR]["validation_gap_mae_eV"]
    winner = min(accepted_rows, key=lambda row: row["validation_gap_mae_eV"])
    expected_selected = winner["candidate"]
    expected_improves = (
        expected_selected != COMPARATOR
        and winner["validation_gap_mae_eV"] < fresh_comparator_mae
    )
    require(selection.get("selected_candidate") == expected_selected, "selection")
    require(
        selection.get("selected_strictly_improves_full_gps") is expected_improves,
        "positive gate",
    )
    comparisons = selection.get("paired_against_fresh_full_gps", [])
    require(isinstance(comparisons, list) and len(comparisons) == 2, "comparisons")
    for comparison in comparisons if isinstance(comparisons, list) else []:
        candidate = comparison.get("candidate")
        require(candidate in CANDIDATES[1:], "comparison candidate")
        if candidate in by_name:
            expected_delta = (
                by_name[candidate]["validation_gap_mae_eV"]
                - fresh_comparator_mae
            )
            require(
                math.isclose(
                    comparison.get("candidate_minus_full_gps_eV", float("nan")),
                    expected_delta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"comparison delta {candidate}",
            )
    result = {
        "format": "molgap-pcqm-gap100k-local-global-allocation-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "geometry_cache_aggregate_sha256": expected_geometry_sha256,
        "comparator": COMPARATOR,
        "frozen_comparator_validation_gap_mae_eV": FROZEN_COMPARATOR_MAE,
        "fresh_comparator_validation_gap_mae_eV": fresh_comparator_mae,
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
        "selected_strictly_improves_full_gps": expected_improves,
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
