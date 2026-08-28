"""No-inference acceptance for the sparse-triangle PCQM seed-42 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CANDIDATE = "ogb_sparse_triangle_edge_state_gps9"
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
        selection.get("format")
        == "molgap-pcqm-gap100k-sparse-triangle-edge-state-v1",
        "format",
    )
    require(selection.get("complete") is True, "complete")
    require(selection.get("source_commit") == expected_source_commit, "source commit")
    require(
        selection.get("cache_aggregate_sha256") == expected_cache_sha256,
        "cache SHA",
    )
    require(selection.get("official_validation_role_read") is False, "official validation")
    require(selection.get("test_dev_role_read") is False, "test-dev")
    require(selection.get("search_budget_s") == 14_400, "search budget")

    comparator = selection.get("frozen_comparator", {})
    require(comparator.get("candidate") == COMPARATOR, "comparator identity")
    require(comparator.get("validation_gap_mae_eV") == COMPARATOR_MAE, "comparator MAE")

    row = selection.get("candidate", {})
    require(row.get("candidate") == CANDIDATE, "candidate identity")
    preflight = selection.get("preflight", {})
    require(preflight.get("candidate") == CANDIDATE, "preflight identity")
    require(preflight.get("source_commit") == expected_source_commit, "preflight source")
    for key in ("finite_prediction", "finite_loss", "finite_gradients"):
        require(preflight.get(key) is True, f"preflight {key}")
    require(
        isinstance(preflight.get("parameter_count"), int)
        and 0 < preflight["parameter_count"] <= 5_200_000,
        "preflight parameters",
    )

    metrics_path = root / "results" / CANDIDATE / "metrics.json"
    require(metrics_path.is_file(), "missing metrics")
    metrics = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics == row, "selection/metrics mismatch")
        require(metrics.get("candidate") == CANDIDATE, "metrics candidate")
        require(metrics.get("source_commit") == expected_source_commit, "metrics source")
        require(metrics.get("seed") == 42, "seed")
        parameter_count = metrics.get("parameter_count")
        require(
            isinstance(parameter_count, int) and 0 < parameter_count <= 5_200_000,
            "parameters",
        )
        value = metrics.get("validation_gap_mae_eV")
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            "validation MAE",
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
            "contract",
        )
        require(metrics.get("official_validation_role_read") is False, "metrics official validation")
        require(metrics.get("test_dev_role_read") is False, "metrics test-dev")
        artifacts = metrics.get("artifacts", {})
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            path = root / artifacts.get(name, "__missing__")
            require(path.is_file(), f"missing {name}")
            if path.is_file():
                require(sha256_file(path) == artifacts.get(f"{name}_sha256"), f"hash {name}")

    value = metrics.get("validation_gap_mae_eV")
    strictly_improves = isinstance(value, (int, float)) and value < COMPARATOR_MAE
    require(selection.get("selected_candidate") == CANDIDATE, "selection candidate")
    require(selection.get("selected_strictly_improves") is strictly_improves, "selection arithmetic")

    result = {
        "format": "molgap-pcqm-gap100k-sparse-triangle-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "cache_aggregate_sha256": expected_cache_sha256,
        "candidate": CANDIDATE,
        "parameter_count": metrics.get("parameter_count"),
        "best_epoch": metrics.get("best_epoch"),
        "validation_gap_mae_eV": value,
        "comparator_validation_gap_mae_eV": COMPARATOR_MAE,
        "selected_strictly_improves": strictly_improves,
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
    result = accept(args.root, args.expected_source_commit, args.expected_cache_sha256)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
