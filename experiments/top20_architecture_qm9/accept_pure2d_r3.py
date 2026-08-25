"""Accept downloaded pure-2D R3 validation artifacts without model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

CANDIDATES = (
    "edge_state_structural_gps",
    "edge_state_structural_orbital",
    "pair_gps_2d_r3_orbital",
    "pair_gps_2d_r3_triplet",
    "pair_gps_2d_r3_combined",
)
EXPECTED_SOURCE_COMMIT = "b56205967f12f517c7eea4428c0dfe8571c54996"
EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"
EXPECTED_RWSE_SHA256 = (
    "09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5"
)
REFERENCE_AVERAGE = 0.11006919294595718
REFERENCE_GAP = 0.1318935602903366
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


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
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
    expected_average = sum(target_values) / len(target_values)
    if not close(average, expected_average, tolerance=2e-8):
        raise RuntimeError(
            f"Average MAE mismatch: recorded={average} expected={expected_average}"
        )


def validate_metrics(metrics: dict, candidate: str) -> None:
    if metrics.get("experiment") != "qm9_architecture_screen":
        raise RuntimeError(f"Unexpected experiment for {candidate}")
    if metrics.get("candidate") != candidate:
        raise RuntimeError(f"Candidate mismatch: {metrics.get('candidate')}")
    if metrics.get("split_fingerprint") != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(f"Split mismatch for {candidate}")
    if metrics.get("seed") != 42 or metrics.get("split_seed") != 42:
        raise RuntimeError(f"Seed mismatch for {candidate}")
    if metrics.get("geometry") != "topology":
        raise RuntimeError(f"Geometry mismatch for {candidate}")
    if metrics.get("test_role_evaluated") is not False:
        raise RuntimeError(f"Test role was evaluated for {candidate}")
    if metrics.get("split_rows") != {"train": 30_000, "validation": 3_000}:
        raise RuntimeError(f"Unexpected split rows for {candidate}")
    if metrics.get("requested_rows") != {
        "train": 30_000,
        "validation": 3_000,
        "test": 3_000,
    }:
        raise RuntimeError(f"Unexpected requested rows for {candidate}")
    if set(metrics.get("metrics", {})) != {"train", "validation"}:
        raise RuntimeError(f"Unexpected evaluated roles for {candidate}")
    validate_metric_block(metrics["metrics"]["train"])
    validate_metric_block(metrics["metrics"]["validation"])
    geometry_report = metrics.get("geometry_report", {})
    if geometry_report.get("test_role_read_during_selection") is not False:
        raise RuntimeError(f"Selection test-role flag changed for {candidate}")
    if geometry_report.get("test_role_read_after_selection") is not False:
        raise RuntimeError(f"Post-selection test-role flag changed for {candidate}")
    log = metrics.get("log", [])
    if not 1 <= len(log) <= 20:
        raise RuntimeError(f"Unexpected epoch count for {candidate}: {len(log)}")
    epochs = [int(row["epoch"]) for row in log]
    if epochs != list(range(len(log))):
        raise RuntimeError(f"Non-contiguous epochs for {candidate}: {epochs}")
    best_epoch = int(metrics["best_epoch"])
    if best_epoch not in epochs:
        raise RuntimeError(f"Missing best epoch for {candidate}: {best_epoch}")
    selected_epochs = [int(row["epoch"]) for row in log if row["selected"]]
    if not selected_epochs or selected_epochs[-1] != best_epoch:
        raise RuntimeError(f"Best epoch trail mismatch for {candidate}")
    best_logged = float(log[best_epoch]["validation_average_mae_eV"])
    if not close(best_logged, metrics["best_validation_average_mae_eV"]):
        raise RuntimeError(f"Best validation mismatch for {candidate}")


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
    if selection.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("Selection source commit mismatch")
    if selection.get("test_role_read") is not False:
        raise RuntimeError("Selection reports a test read")
    if selection.get("parameter_budget") != PARAMETER_BUDGET:
        raise RuntimeError("Selection parameter budget mismatch")
    if preflight.get("complete") is not True:
        raise RuntimeError("Preflight is incomplete")
    if preflight.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("Preflight source commit mismatch")
    if preflight.get("split_fingerprint") != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError("Preflight split mismatch")
    if preflight.get("rwse_output_sha256") != EXPECTED_RWSE_SHA256:
        raise RuntimeError("Preflight RWSE mismatch")
    if preflight.get("test_role_read") is not False:
        raise RuntimeError("Preflight reports a test read")
    preflight_rows = preflight.get("candidates", [])
    if tuple(row.get("candidate") for row in preflight_rows) != CANDIDATES:
        raise RuntimeError("Preflight candidate order mismatch")

    completed = selection.get("completed", [])
    if tuple(row.get("candidate") for row in completed) != CANDIDATES:
        raise RuntimeError("Selection candidate order mismatch")
    metric_paths = list(root.rglob("metrics.json"))
    metrics_by_candidate = {}
    for path in metric_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        candidate = value.get("candidate")
        if candidate in CANDIDATES:
            if candidate in metrics_by_candidate:
                raise RuntimeError(f"Duplicate metrics for {candidate}")
            metrics_by_candidate[candidate] = (path, value)
    if set(metrics_by_candidate) != set(CANDIDATES):
        raise RuntimeError(f"Missing candidate metrics: {set(CANDIDATES) - set(metrics_by_candidate)}")

    artifacts = []
    eligible = []
    for preflight_row, selected_row in zip(preflight_rows, completed):
        candidate = selected_row["candidate"]
        for key in ("finite_prediction", "finite_loss", "finite_gradients"):
            if preflight_row.get(key) is not True:
                raise RuntimeError(f"Failed {key} preflight for {candidate}")
        parameter_count = int(preflight_row["parameter_count"])
        if parameter_count > PARAMETER_BUDGET:
            raise RuntimeError(f"Parameter budget exceeded for {candidate}")
        metrics_path, metrics = metrics_by_candidate[candidate]
        validate_metrics(metrics, candidate)
        if int(metrics["n_params"]) != parameter_count:
            raise RuntimeError(f"Parameter count mismatch for {candidate}")
        if int(selected_row["parameter_count"]) != parameter_count:
            raise RuntimeError(f"Selection parameter mismatch for {candidate}")
        if int(selected_row["best_epoch"]) != int(metrics["best_epoch"]):
            raise RuntimeError(f"Selection epoch mismatch for {candidate}")
        if selected_row["validation"] != metrics["metrics"]["validation"]:
            raise RuntimeError(f"Selection metrics mismatch for {candidate}")
        validation = metrics["metrics"]["validation"]
        row_eligible = (
            validation["average"]["mae"] < REFERENCE_AVERAGE
            and validation["Gap"]["mae"] < REFERENCE_GAP
        )
        if selected_row.get("eligible") is not row_eligible:
            raise RuntimeError(f"Eligibility mismatch for {candidate}")
        if row_eligible:
            eligible.append((validation["average"]["mae"], candidate))
        for artifact_name, remote_path in metrics["artifacts"].items():
            artifact_path = resolve_remote_artifact(root, remote_path)
            artifacts.append(
                {
                    "candidate": candidate,
                    "kind": artifact_name,
                    "path": artifact_path.relative_to(root).as_posix(),
                    "bytes": artifact_path.stat().st_size,
                    "sha256": sha256(artifact_path),
                }
            )
        artifacts.append(
            {
                "candidate": candidate,
                "kind": "metrics",
                "path": metrics_path.relative_to(root).as_posix(),
                "bytes": metrics_path.stat().st_size,
                "sha256": sha256(metrics_path),
            }
        )

    expected_winner = min(eligible)[1] if eligible else None
    if selection.get("selected_candidate") != expected_winner:
        raise RuntimeError(
            f"Winner mismatch: {selection.get('selected_candidate')} vs {expected_winner}"
        )
    report = {
        "format": "molgap-pure2d-r3-local-acceptance-v1",
        "accepted": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "rwse_output_sha256": EXPECTED_RWSE_SHA256,
        "test_role_read": False,
        "candidate_count": len(completed),
        "eligible_candidates": [candidate for _, candidate in sorted(eligible)],
        "selected_candidate": expected_winner,
        "tensor_payload_recomputed": False,
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
