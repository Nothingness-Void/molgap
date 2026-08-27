"""No-inference acceptance for the PCQM recurrent graph-state screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CANDIDATE = "ogb_recurrent_graph_state_gps9"
COMPARATOR = "ogb_edge_state_structural_gps9"
COMPARATOR_MAE = 0.13798263211250306


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str, expected_cache_sha256: str) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        selection.get("format") == "molgap-pcqm-gap100k-recurrent-graph-state-v1",
        "format",
    )
    require(selection.get("complete") is True, "complete")
    require(selection.get("source_commit") == expected_source_commit, "source commit")
    require(selection.get("cache_aggregate_sha256") == expected_cache_sha256, "cache SHA")
    require(selection.get("official_validation_role_read") is False, "official validation")
    require(selection.get("test_dev_role_read") is False, "test-dev")
    require(selection.get("search_budget_s") == 14_400, "search budget")

    comparator = selection.get("frozen_comparator", {})
    require(comparator.get("candidate") == COMPARATOR, "comparator identity")
    require(comparator.get("validation_gap_mae_eV") == COMPARATOR_MAE, "comparator MAE")

    row = selection.get("candidate", {})
    rows = [row] if isinstance(row, dict) and row else []
    names = [row.get("candidate") for row in rows]
    require(names == [CANDIDATE], "candidate identity/completeness")

    preflight = selection.get("preflight", {})
    preflight_rows = preflight.get("candidates", [])
    require(preflight.get("source_commit") == expected_source_commit, "preflight source")
    require(preflight.get("official_validation_role_read") is False, "preflight validation")
    require(preflight.get("test_dev_role_read") is False, "preflight test-dev")
    require(len(preflight_rows) == 1, "preflight candidate count")
    if len(preflight_rows) == 1:
        preflight_row = preflight_rows[0]
        require(preflight_row.get("candidate") == CANDIDATE, "preflight candidate")
        require(preflight_row.get("finite_prediction") is True, "preflight prediction")
        require(preflight_row.get("finite_loss") is True, "preflight loss")
        require(preflight_row.get("finite_gradients") is True, "preflight gradients")

    accepted_rows = []
    for row in rows:
        candidate = row.get("candidate", "__missing__")
        metrics_path = root / "results" / candidate / "metrics.json"
        require(metrics_path.is_file(), f"missing metrics {candidate}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics == row, f"selection/metrics mismatch {candidate}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {candidate}")
        require(metrics.get("seed") == 42, f"seed {candidate}")
        require(
            0 < int(metrics.get("parameter_count", 0)) <= 5_200_000,
            f"parameters {candidate}",
        )
        value = metrics.get("validation_gap_mae_eV")
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"MAE {candidate}",
        )
        elapsed = metrics.get("training_elapsed_s")
        require(
            isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0,
            f"elapsed {candidate}",
        )
        require(
            metrics.get("contract")
            == {
                "batch_size": 48,
                "learning_rate": 1.6e-4,
                "weight_decay": 1.0e-6,
                "max_epochs": 40,
                "patience": 8,
                "precision": "fp32",
                "target": "homolumogap",
            },
            f"contract {candidate}",
        )
        require(
            metrics.get("official_validation_role_read") is False,
            f"official validation {candidate}",
        )
        require(metrics.get("test_dev_role_read") is False, f"test-dev {candidate}")
        artifacts = metrics.get("artifacts", {})
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            path = root / artifacts.get(name, "__missing__")
            require(path.is_file(), f"missing {name} {candidate}")
            if path.is_file():
                require(
                    sha256_file(path) == artifacts.get(f"{name}_sha256"),
                    f"hash {name} {candidate}",
                )
        accepted_rows.append(
            {
                "candidate": candidate,
                "parameter_count": metrics.get("parameter_count"),
                "best_epoch": metrics.get("best_epoch"),
                "validation_gap_mae_eV": value,
                "training_elapsed_s": elapsed,
                "strictly_improves": (
                    isinstance(value, (int, float)) and value < COMPARATOR_MAE
                ),
            }
        )

    finite_rows = [
        row
        for row in accepted_rows
        if isinstance(row["validation_gap_mae_eV"], (int, float))
        and math.isfinite(row["validation_gap_mae_eV"])
    ]
    winner = finite_rows[0] if len(finite_rows) == 1 else None
    require(winner is not None, "winner")
    if winner is not None:
        require(selection.get("selected_candidate") == winner["candidate"], "selection")
        require(
            selection.get("selected_strictly_improves")
            is (winner["validation_gap_mae_eV"] < COMPARATOR_MAE),
            "improvement arithmetic",
        )

    result = {
        "format": "molgap-pcqm-gap100k-recurrent-graph-state-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "cache_aggregate_sha256": expected_cache_sha256,
        "completed_candidates": accepted_rows,
        "selected_candidate": winner["candidate"] if winner else None,
        "selected_validation_gap_mae_eV": (
            winner["validation_gap_mae_eV"] if winner else None
        ),
        "comparator_validation_gap_mae_eV": COMPARATOR_MAE,
        "selected_strictly_improves": (
            winner is not None and winner["validation_gap_mae_eV"] < COMPARATOR_MAE
        ),
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
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(
        root=args.root,
        expected_source_commit=args.expected_source_commit,
        expected_cache_sha256=args.expected_cache_sha256,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
