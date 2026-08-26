"""Accept downloaded directed EdgeState R10 artifacts without inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from accept_edge_state_jk_readout_r5 import (
    load_unique_json,
    resolve_remote_artifact,
    sha256,
    validate_metric_block,
)

CANDIDATE = "directed_edge_state_structural_gps"
R3_WINNER = "edge_state_structural_gps"
R3_MODEL_SHA256 = (
    "c99883ba5efb247121cce6d83f64d95f5d493e82862ec040eed4a5206c86e186"
)
EXPECTED_SOURCE_COMMIT = "06bf8f439783cced552760b873e1702a0098c802"
EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"
EXPECTED_RWSE_SHA256 = (
    "09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5"
)
REFERENCE_AVERAGE = 0.10527653247117996
REFERENCE_GAP = 0.1261376142501831
EXPECTED_PARAMETER_COUNT = 4_776_515
PARAMETER_BUDGET = 4_800_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    selection_path, selection = load_unique_json(root, "selection.json")
    preflight_path, preflight = load_unique_json(root, "preflight.json")
    metrics_path, metrics = load_unique_json(root, "metrics.json")
    for record in (selection, preflight):
        if record.get("source_commit") != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("Source commit mismatch")
        if record.get("test_role_read") is not False:
            raise RuntimeError("A remote record reports a test-role read")
    expected_preflight = {
        "format": "molgap-directed-edgestate-r10-preflight-v1",
        "complete": True,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "rwse_output_sha256": EXPECTED_RWSE_SHA256,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_budget": PARAMETER_BUDGET,
        "finite_prediction": True,
        "finite_loss": True,
        "finite_gradients": True,
        "local_edge_feature_dim": 4,
        "reverse_edge_coverage": True,
        "validation_role_read": False,
    }
    mismatches = {
        key: (preflight.get(key), value)
        for key, value in expected_preflight.items()
        if preflight.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Preflight contract mismatch: {mismatches}")
    anchor = selection.get("r3_anchor", {})
    if anchor.get("selected_candidate") != R3_WINNER:
        raise RuntimeError("R3 anchor winner changed")
    if anchor.get("selected_model_sha256") != R3_MODEL_SHA256:
        raise RuntimeError("R3 anchor model hash changed")
    if anchor.get("test_role_read") is not False:
        raise RuntimeError("R3 anchor reports a test-role read")
    expected_metrics = {
        "candidate": CANDIDATE,
        "geometry": "topology",
        "seed": 42,
        "split_seed": 42,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "test_role_evaluated": False,
        "split_rows": {"train": 30_000, "validation": 3_000},
        "requested_rows": {
            "train": 30_000,
            "validation": 3_000,
            "test": 3_000,
        },
        "n_params": EXPECTED_PARAMETER_COUNT,
    }
    mismatches = {
        key: (metrics.get(key), value)
        for key, value in expected_metrics.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Metrics contract mismatch: {mismatches}")
    if set(metrics.get("metrics", {})) != {"train", "validation"}:
        raise RuntimeError("Metrics contain a role other than train/validation")
    validate_metric_block(metrics["metrics"]["train"])
    validate_metric_block(metrics["metrics"]["validation"])
    geometry_report = metrics.get("geometry_report", {})
    if geometry_report.get("test_role_read_during_selection") is not False:
        raise RuntimeError("Selection reports a test-role read")
    if geometry_report.get("test_role_read_after_selection") is not False:
        raise RuntimeError("Post-selection evaluation reports a test-role read")
    log = metrics.get("log", [])
    epochs = [int(row["epoch"]) for row in log]
    if not 1 <= len(log) <= 20 or epochs != list(range(len(log))):
        raise RuntimeError("Training epoch trail is invalid")
    best_epoch = int(metrics["best_epoch"])
    selected_epochs = [int(row["epoch"]) for row in log if row["selected"]]
    if best_epoch not in epochs or not selected_epochs or selected_epochs[-1] != best_epoch:
        raise RuntimeError("Best-epoch trail is inconsistent")
    validation = metrics["metrics"]["validation"]
    eligible = (
        float(validation["average"]["mae"]) < REFERENCE_AVERAGE
        and float(validation["Gap"]["mae"]) < REFERENCE_GAP
    )
    expected_selection = {
        "candidate": CANDIDATE,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "best_epoch": best_epoch,
        "validation": validation,
        "eligible": eligible,
        "selected_candidate": CANDIDATE if eligible else R3_WINNER,
        "test_role_read": False,
    }
    mismatches = {
        key: (selection.get(key), value)
        for key, value in expected_selection.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Selection mismatch: {mismatches}")
    artifacts = []
    for kind, remote_path in metrics.get("artifacts", {}).items():
        artifact = resolve_remote_artifact(root, remote_path)
        artifacts.append(
            {
                "kind": kind,
                "path": artifact.relative_to(root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
        )
    if {row["kind"] for row in artifacts} != {"embeddings", "model", "checkpoint"}:
        raise RuntimeError("Required model artifacts are incomplete")
    artifacts.append(
        {
            "kind": "metrics",
            "path": metrics_path.relative_to(root).as_posix(),
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256(metrics_path),
        }
    )
    report = {
        "format": "molgap-directed-edgestate-r10-local-acceptance-v1",
        "accepted": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "rwse_output_sha256": EXPECTED_RWSE_SHA256,
        "test_role_read": False,
        "candidate": CANDIDATE,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "eligible": eligible,
        "selected_candidate": expected_selection["selected_candidate"],
        "model_inference_executed": False,
        "selection_json": {
            "path": selection_path.relative_to(root).as_posix(),
            "sha256": sha256(selection_path),
        },
        "preflight_json": {
            "path": preflight_path.relative_to(root).as_posix(),
            "sha256": sha256(preflight_path),
        },
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"accepted": True, "selected_candidate": report["selected_candidate"]}))


if __name__ == "__main__":
    main()
