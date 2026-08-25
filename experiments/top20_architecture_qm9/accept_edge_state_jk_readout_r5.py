"""Accept downloaded EdgeState-JK R5 validation artifacts without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


CANDIDATE = "edge_state_structural_jk_readout"
R3_WINNER = "edge_state_structural_gps"
R3_MODEL_SHA256 = (
    "c99883ba5efb247121cce6d83f64d95f5d493e82862ec040eed4a5206c86e186"
)
EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"
EXPECTED_RWSE_SHA256 = (
    "09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5"
)
REFERENCE_AVERAGE = 0.10527653247117996
REFERENCE_GAP = 0.1261376142501831
EXPECTED_PARAMETER_COUNT = 4_767_779
PARAMETER_BUDGET = 4_800_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unique_json(root: Path, name: str) -> tuple[Path, dict]:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}, found {matches}")
    return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))


def resolve_remote_artifact(root: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path).relative_to("/kaggle/working")
    suffix = relative.as_posix()
    matches = [
        path
        for path in root.rglob(relative.name)
        if path.as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected artifact suffix {suffix}, found {matches}")
    if not matches[0].is_file() or matches[0].stat().st_size <= 0:
        raise RuntimeError(f"Artifact is empty: {matches[0]}")
    return matches[0]


def close(left: float, right: float, tolerance: float = 2e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def validate_metric_block(block: dict) -> None:
    if set(block) != {"HOMO", "LUMO", "Gap", "average"}:
        raise RuntimeError(f"Unexpected metric keys: {sorted(block)}")
    target_values = []
    for target in ("HOMO", "LUMO", "Gap"):
        value = float(block[target]["mae"])
        if not 0.0 <= value < 10.0:
            raise RuntimeError(f"Invalid {target} MAE: {value}")
        target_values.append(value)
    average = float(block["average"]["mae"])
    if not close(average, sum(target_values) / len(target_values)):
        raise RuntimeError("Average MAE does not match the three target MAEs")


def validate_metrics(metrics: dict) -> None:
    expected_fields = {
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
        for key, value in expected_fields.items()
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
    if not 1 <= len(log) <= 20:
        raise RuntimeError(f"Unexpected epoch count: {len(log)}")
    epochs = [int(row["epoch"]) for row in log]
    if epochs != list(range(len(log))):
        raise RuntimeError(f"Non-contiguous epochs: {epochs}")
    best_epoch = int(metrics["best_epoch"])
    selected_epochs = [int(row["epoch"]) for row in log if row["selected"]]
    if best_epoch not in epochs or not selected_epochs or selected_epochs[-1] != best_epoch:
        raise RuntimeError("Best-epoch trail is inconsistent")
    if not close(
        log[best_epoch]["validation_average_mae_eV"],
        metrics["best_validation_average_mae_eV"],
    ):
        raise RuntimeError("Best validation MAE is inconsistent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    selection_path, selection = load_unique_json(root, "selection.json")
    preflight_path, preflight = load_unique_json(root, "preflight.json")
    metrics_path, metrics = load_unique_json(root, "metrics.json")
    if selection.get("source_commit") != args.source_commit:
        raise RuntimeError("Selection source commit mismatch")
    if preflight.get("source_commit") != args.source_commit:
        raise RuntimeError("Preflight source commit mismatch")
    if selection.get("test_role_read") is not False:
        raise RuntimeError("Selection reports a test-role read")
    expected_preflight = {
        "format": "molgap-edge-state-jk-readout-r5-preflight-v1",
        "complete": True,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "rwse_output_sha256": EXPECTED_RWSE_SHA256,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_budget": PARAMETER_BUDGET,
        "finite_prediction": True,
        "finite_loss": True,
        "finite_gradients": True,
        "validation_role_read": False,
        "test_role_read": False,
    }
    preflight_mismatches = {
        key: (preflight.get(key), value)
        for key, value in expected_preflight.items()
        if preflight.get(key) != value
    }
    if preflight_mismatches:
        raise RuntimeError(f"Preflight contract mismatch: {preflight_mismatches}")
    anchor = selection.get("r3_anchor", {})
    if anchor.get("selected_candidate") != R3_WINNER:
        raise RuntimeError("R3 anchor winner changed")
    if anchor.get("selected_model_sha256") != R3_MODEL_SHA256:
        raise RuntimeError("R3 anchor model hash changed")
    if anchor.get("test_role_read") is not False:
        raise RuntimeError("R3 anchor reports a test-role read")

    validate_metrics(metrics)
    if selection.get("candidate") != CANDIDATE:
        raise RuntimeError("Selection candidate mismatch")
    if int(selection.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Selection parameter count mismatch")
    if int(selection.get("best_epoch", -1)) != int(metrics["best_epoch"]):
        raise RuntimeError("Selection best epoch mismatch")
    validation = metrics["metrics"]["validation"]
    if selection.get("validation") != validation:
        raise RuntimeError("Selection and metrics validation blocks differ")
    eligible = (
        float(validation["average"]["mae"]) < REFERENCE_AVERAGE
        and float(validation["Gap"]["mae"]) < REFERENCE_GAP
    )
    if selection.get("eligible") is not eligible:
        raise RuntimeError("Selection eligibility is inconsistent")
    expected_winner = CANDIDATE if eligible else R3_WINNER
    if selection.get("selected_candidate") != expected_winner:
        raise RuntimeError("Selection winner is inconsistent")

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
        "format": "molgap-edge-state-jk-readout-r5-local-acceptance-v1",
        "accepted": True,
        "source_commit": args.source_commit,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "rwse_output_sha256": EXPECTED_RWSE_SHA256,
        "test_role_read": False,
        "candidate": CANDIDATE,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "eligible": eligible,
        "selected_candidate": expected_winner,
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
    print(json.dumps({"accepted": True, "selected_candidate": expected_winner}))


if __name__ == "__main__":
    main()
