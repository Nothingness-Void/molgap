"""No-model acceptance for a completed Kunshan GraphState runtime gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9"
EXPECTED_MODEL_SOURCE = "9068ddb82e6bdf16b841570abbff023b90c07f07"
EXPECTED_CACHE = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
EXPECTED_PARAMS = 3_665_809


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def accept(root: Path) -> dict:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    require(metrics == completion, "completion manifest differs from metrics")
    for payload, name in ((metrics, "metrics"), (run, "run")):
        require(
            payload.get("candidate") == EXPECTED_CANDIDATE,
            f"{name} candidate",
        )
        require(
            payload.get("model_source_commit") == EXPECTED_MODEL_SOURCE,
            f"{name} model source",
        )
        require(
            payload.get("cache_aggregate_sha256") == EXPECTED_CACHE,
            f"{name} cache aggregate",
        )
        require(payload.get("seed") == 42, f"{name} seed")
        require(payload.get("precision") == "fp32", f"{name} precision")
        require(payload.get("batch_size") == 48, f"{name} batch")
        require(payload.get("target") == "gap", f"{name} target")
        require(
            payload.get("official_validation_role_read") is False,
            f"{name} official validation role",
        )
        require(payload.get("test_dev_role_read") is False, f"{name} test-dev role")
    require(metrics.get("complete") is True, "complete")
    require(metrics.get("parameter_count") == EXPECTED_PARAMS, "parameter count")
    require(metrics.get("device_count") == 1, "one visible device")
    require(metrics.get("model_inference_executed") is True, "training inference")
    rows = metrics.get("epochs")
    require(isinstance(rows, list) and len(rows) == 3, "three epoch rows")
    if isinstance(rows, list):
        for expected_epoch, row in enumerate(rows, start=1):
            require(row.get("epoch") == expected_epoch, f"epoch {expected_epoch}")
            require(row.get("train_graphs") == 100_000, f"train count {expected_epoch}")
            require(
                row.get("validation_graphs") == 10_000,
                f"validation count {expected_epoch}",
            )
            for key in (
                "train_mae_eV",
                "validation_mae_eV",
                "epoch_s",
                "train_graphs_per_s",
                "peak_memory_bytes",
            ):
                require(finite(row.get(key)), f"finite {key} {expected_epoch}")
    for filename in ("best_model.pt", "last_checkpoint.pt"):
        path = root / filename
        require(path.is_file() and path.stat().st_size > 0, f"checkpoint {filename}")
    accelerator = metrics.get("accelerator_snapshot", {})
    require(bool(accelerator.get("commands")), "accelerator telemetry")
    result = {
        "format": "molgap-pcqm-kunshan-graphstate-runtime-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "candidate": metrics.get("candidate"),
        "model_source_commit": metrics.get("model_source_commit"),
        "cache_aggregate_sha256": metrics.get("cache_aggregate_sha256"),
        "parameter_count": metrics.get("parameter_count"),
        "device": metrics.get("device"),
        "best_epoch": metrics.get("best_epoch"),
        "best_validation_mae_eV": metrics.get("best_validation_mae_eV"),
        "peak_memory_bytes": metrics.get("peak_memory_bytes"),
        "model_inference_executed": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "checkpoint_sha256": {
            filename: sha256_file(root / filename)
            for filename in ("best_model.pt", "last_checkpoint.pt")
            if (root / filename).is_file()
        },
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
