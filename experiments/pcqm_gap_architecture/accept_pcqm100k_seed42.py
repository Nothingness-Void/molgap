"""No-inference acceptance for the matched PCQM Gap100K seed-42 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CANDIDATES = ("ogb_structural_gps9", "ogb_edge_state_structural_gps9")


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

    require(selection.get("format") == "molgap-pcqm-gap100k-seed42-screen-v1", "format")
    require(selection.get("complete") is True, "complete")
    require(selection.get("source_commit") == expected_source_commit, "source commit")
    require(selection.get("cache_aggregate_sha256") == expected_cache_sha256, "cache SHA")
    require(selection.get("official_validation_role_read") is False, "official validation read")
    require(selection.get("test_dev_role_read") is False, "test-dev read")
    rows = selection.get("candidates", [])
    require([row.get("candidate") for row in rows] == list(CANDIDATES), "candidate order")
    values = {}
    for row in rows:
        candidate = row.get("candidate")
        metrics_path = root / "results" / str(candidate) / "metrics.json"
        require(metrics_path.is_file(), f"missing metrics {candidate}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(metrics == row, f"selection/metrics mismatch {candidate}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {candidate}")
        require(metrics.get("seed") == 42, f"seed {candidate}")
        require(0 < int(metrics.get("parameter_count", 0)) <= 5_200_000, f"params {candidate}")
        value = metrics.get("validation_gap_mae_eV")
        require(isinstance(value, (int, float)) and math.isfinite(value), f"MAE {candidate}")
        values[candidate] = float(value) if isinstance(value, (int, float)) else math.inf
        contract = metrics.get("contract", {})
        require(
            contract
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
        require(metrics.get("official_validation_role_read") is False, f"official validation {candidate}")
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
    if len(values) == 2:
        expected_winner = min(values, key=values.get)
        require(selection.get("selected_candidate") == expected_winner, "winner arithmetic")
        expected_improvement = values[CANDIDATES[1]] < values[CANDIDATES[0]]
        require(
            selection.get("edge_state_strictly_improves") is expected_improvement,
            "improvement arithmetic",
        )
    result = {
        "format": "molgap-pcqm-gap100k-seed42-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "cache_aggregate_sha256": expected_cache_sha256,
        "selected_candidate": selection.get("selected_candidate"),
        "edge_state_strictly_improves": selection.get("edge_state_strictly_improves"),
        "validation_gap_mae_eV": values,
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
        args.root,
        args.expected_source_commit,
        args.expected_cache_sha256,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

