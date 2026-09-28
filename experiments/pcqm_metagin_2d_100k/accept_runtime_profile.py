"""Accept the bounded train-role-only profile without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
RUN_ID = "kaseichou/molgap-metagin-2d-runtime-profile:v2"
SIDECAR = json.loads((ROOT / "results/cpu_sidecar_acceptance.json").read_text(encoding="utf-8"))
TRANSFORM = json.loads((ROOT.parent / "v5_legacy_evidence_migration/k1_v4_100k_reference/target_transform.json").read_text(encoding="utf-8"))
CONTRACT = json.loads((ROOT / "training_contract.json").read_text(encoding="utf-8"))
SOURCE_RECEIPT = json.loads((ROOT / "results/profile_v2_source_receipt.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbers(values: object, count: int, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{label} requires {count} observations")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{label} contains invalid timing")
    return result


def _same(actual: object, expected: float, label: str) -> None:
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"{label} differs from the measured observations")


def accept(record: Path, source: Path) -> dict:
    """Fail closed on identity, roles, arithmetic and native cost; return a receipt."""
    record, source = record.resolve(), source.resolve()
    if (record / "failure.json").exists():
        raise ValueError("Profile output also contains a failure marker")
    profile_path = record / "runtime_profile.json"
    cost_path = record / "native_cost.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    commit = (source / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    archive_sha = (source / "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    if (
        commit != SOURCE_RECEIPT["source_commit"]
        or archive_sha != SOURCE_RECEIPT["source_archive_sha256"]
        or _sha(source / "source_payload.bin") != archive_sha
        or SOURCE_RECEIPT["producer_sidecar_commit"] != SIDECAR["source_commit"]
        or SOURCE_RECEIPT["sidecar_aggregate_sha256"] != SIDECAR["aggregate_sha256"]
        or SIDECAR.get("accepted") is not True
    ):
        raise ValueError("Profile executable package changed")
    if (
        profile.get("format") != "molgap-metagin-2d-train-role-runtime-profile-v1"
        or profile.get("complete") is not True
        or profile.get("training_screen_executed") is not False
        or profile.get("run_id") != RUN_ID
        or profile.get("source_commit") != commit
        or profile.get("sidecar_producer_commit") != SIDECAR["source_commit"]
        or profile.get("sidecar_aggregate_sha256") != SIDECAR["aggregate_sha256"]
        or profile.get("sidecar_accepted") is not True
        or profile.get("target_transform_asset_id") != TRANSFORM["asset_id"]
        or profile.get("precision") != CONTRACT["precision"]
        or profile.get("tf32_enabled") is not False
        or profile.get("physical_batch_per_device") != CONTRACT["physical_batch_per_device"]
        or profile.get("train_role_read") is not True
        or profile.get("internal_development_graphs_loaded") is not True
        or profile.get("internal_development_labels_used") is not False
        or any(profile.get(role) is not False for role in (
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"
        ))
    ):
        raise ValueError("Profile identity, contract or role boundary changed")
    observation = profile.get("target_transform_observation", {})
    if (
        observation.get("train_target_sha256") != TRANSFORM["target_sha256"]
        or observation.get("train_source_indices_verified") is not True
        or observation.get("transform_source") != "immutable_asset_exact_values"
    ):
        raise ValueError("Profile did not verify frozen training targets")
    _numbers(profile.get("warmup_step_seconds"), 8, "warm-up")
    train = _numbers(profile.get("measured_step_seconds"), 72, "steady optimizer")
    evaluation = _numbers(profile.get("evaluation_forward_seconds"), 32, "forward")
    train_mean, eval_mean = statistics.mean(train), statistics.mean(evaluation)
    _same(profile.get("steady_train_step_mean_seconds"), train_mean, "optimizer mean")
    _same(profile.get("steady_train_step_median_seconds"), statistics.median(train), "optimizer median")
    _same(profile.get("steady_train_step_max_seconds"), max(train), "optimizer max")
    _same(profile.get("evaluation_step_mean_seconds"), eval_mean, "forward mean")
    projected = CONTRACT["epochs"] * (
        CONTRACT["steps_per_epoch"] * train_mean
        + math.ceil(50_000 / CONTRACT["physical_batch_per_device"]) * eval_mean
    )
    _same(profile.get("projected_epoch_seconds"), projected / CONTRACT["epochs"], "epoch projection")
    _same(profile.get("projected_40_epoch_seconds"), projected, "total projection")
    _same(profile.get("projected_with_20_percent_reserve_seconds"), 1.2 * projected, "reserve projection")
    peak, total = float(profile["peak_reserved_mib"]), float(profile["total_memory_mib"])
    if not all(math.isfinite(value) for value in (peak, total)) or not 0 < peak < total:
        raise ValueError("Profile GPU memory measurements invalid")
    hardware, count = profile.get("hardware"), profile.get("allocated_device_count")
    if not isinstance(hardware, str) or not hardware or type(count) is not int or count < 1:
        raise ValueError("Profile GPU allocation identity missing")
    wall = float(profile["profile_wall_seconds"])
    cost_wall = float(cost["wall_seconds"])
    if (
        not all(math.isfinite(value) and value > 0 for value in (wall, cost_wall))
        or wall > 1200 or cost_wall < wall
        or cost.get("format") != "molgap-metagin-runtime-profile-cost-v1"
        or cost.get("run_id") != RUN_ID
        or cost.get("source_commit") != commit
        or cost.get("training_screen_executed") is not False
    ):
        raise ValueError("Profile exceeded its bounded run or native cost is invalid")
    memory_reserve = 1.0 - peak / total
    within_worker_budget = 1.2 * projected < CONTRACT["max_worker_wall_hours"] * 3600
    memory_accepted = memory_reserve >= CONTRACT["minimum_memory_reserve_fraction"]
    return {
        "format": "molgap-metagin-2d-runtime-profile-acceptance-v1",
        "accepted": True,
        "training_screen_executed": False,
        "model_inference_executed_locally": False,
        "run_id": RUN_ID,
        "source_commit": commit,
        "source_archive_sha256": archive_sha,
        "sidecar_aggregate_sha256": SIDECAR["aggregate_sha256"],
        "runtime_profile_sha256": _sha(profile_path),
        "native_cost_sha256": _sha(cost_path),
        "hardware": hardware,
        "allocated_device_count": count,
        "profile_wall_seconds": wall,
        "measured_optimizer_step_mean_seconds": train_mean,
        "measured_forward_step_mean_seconds": eval_mean,
        "projected_40_epoch_wall_seconds": projected,
        "projected_wall_with_20_percent_reserve_seconds": 1.2 * projected,
        "projected_allocated_device_hours_with_reserve": count * 1.2 * projected / 3600,
        "peak_reserved_mib": peak,
        "total_memory_mib": total,
        "memory_reserve_fraction": memory_reserve,
        "worker_budget_gate_passed": within_worker_budget,
        "memory_gate_passed": memory_accepted,
        "same_size_training_admissible_on_measured_hardware": within_worker_budget and memory_accepted,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = accept(args.record, args.source)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
