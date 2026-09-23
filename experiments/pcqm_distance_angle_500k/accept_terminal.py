"""Replay the retained distance-angle 500K terminal audit without model inference."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

from molgap.pcqm_distance_angle_scale import (
    ARMS, BASELINE, CANDIDATE, DistanceAngleScaleConfig, accept_pair, make_model,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/pcqm_distance_angle_500k"
RETAINED = ROOT / "platforms/_records/scnet/pcqm_distance_angle_500k_20260923/remote"


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    launch = read_json(EXPERIMENT / "remote_launch.json")
    audit = read_json(EXPERIMENT / "results/artifact_reconciliation_20260923.json")
    scheduler = read_json(EXPERIMENT / "results/scheduler_20260923.json")
    cache = read_json(RETAINED / "input/cache/manifest.json")
    cache_acceptance = read_json(RETAINED / "input/cache/acceptance.json")
    preflight = read_json(RETAINED / "output/preflight/preflight.json")
    remote_pair_path = RETAINED / "output/paired_acceptance.json"
    remote_pair_bytes = remote_pair_path.read_bytes()
    remote_pair = json.loads(remote_pair_bytes)
    checks: dict[str, bool] = {}

    def check(name: str, passed: bool) -> None:
        checks[name] = bool(passed)

    check("scheduler", all(
        scheduler["jobs"][job]["state"] == "COMPLETED"
        and scheduler["jobs"][job]["exit_code"] == "0:0"
        and scheduler["jobs"][job]["allocated_dcu"] == 1
        for job in launch["jobs"].values()
    ))
    check("retained_artifact_hashes", all(
        (RETAINED / name).is_file()
        and (RETAINED / name).stat().st_size == bound["bytes"]
        and digest(RETAINED / name) == bound["sha256"]
        for name, bound in audit["selected_artifacts"].items()
    ))
    check("source_snapshot", all(
        (RETAINED / "code" / name).is_file()
        and digest(RETAINED / "code" / name) == bound["remote_sha256"]
        and bound["remote_line_endings_normalized_equal_to_git"] is True
        for name, bound in audit["source_checks"].items()
    ))
    source_bytes_equal = True
    for name in audit["source_checks"]:
        git_file = subprocess.run(
            ["git", "show", f"{launch['source_commit']}:{name}"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
        source_bytes_equal &= (
            (RETAINED / "code" / name).read_bytes().replace(b"\r\n", b"\n")
            == git_file
        )
    check("source_commit_contents", source_bytes_equal)
    check("cache_identity", cache["complete"] is True
          and cache_acceptance["accepted"] is True
          and cache["aggregate_sha256"] == cache_acceptance["aggregate_sha256"]
          == launch["cache_aggregate_sha256"])
    check("preflight", preflight["accepted"] is True
          and preflight["batch_size"] == 128)
    check("protected_roles", all(
        record["official_validation_role_read"] is False
        and record["test_dev_role_read"] is False
        for record in (cache, cache_acceptance, preflight, remote_pair)
    ))

    try:
        local_pair = accept_pair(RETAINED / "output")
    finally:
        remote_pair_path.write_bytes(remote_pair_bytes)
    check("paired_acceptance_replay", local_pair == remote_pair
          and local_pair["accepted"] is True)

    development = {}
    for arm in ARMS:
        folder = RETAINED / "output" / arm
        completion = read_json(folder / "completion_manifest.json")
        trace = read_json(folder / "direct_gap_trace.json")["epochs"]
        gap = completion["gap"]
        best = torch.load(folder / "direct_gap_best.pt", map_location="cpu", weights_only=False)
        last = torch.load(folder / "direct_gap_last.pt", map_location="cpu", weights_only=False)
        model = make_model(arm, DistanceAngleScaleConfig(**completion["config"]))
        model.load_state_dict(best["model"], strict=True)
        development[arm] = torch.load(
            folder / "direct_gap_development.pt", map_location="cpu", weights_only=True
        )
        check(f"{arm}:development_roles",
              development[arm]["official_validation_role_read"] is False
              and development[arm]["test_dev_role_read"] is False)
        check(f"{arm}:identity", completion["source_commit"] == launch["source_commit"]
              and completion["cache_aggregate_sha256"] == cache["aggregate_sha256"]
              and best["cache_aggregate_sha256"] == last["cache_aggregate_sha256"]
              == cache["aggregate_sha256"]
              and best["role"] == last["role"] == arm
              and best["config"] == last["config"] == completion["config"])
        check(f"{arm}:trace", len(trace) == 60
              and trace == gap["trace"] == last["trace"]
              and [row["epoch"] for row in trace] == list(range(60))
              and all(math.isfinite(row["train_mae_eV"])
                      and math.isfinite(row["development_mae_eV"])
                      and math.isfinite(row["learning_rate"]) for row in trace))
        selected = min(range(len(trace)), key=lambda i: trace[i]["development_mae_eV"])
        check(f"{arm}:selection", selected == gap["best_epoch"] == best["epoch"]
              and math.isclose(trace[selected]["development_mae_eV"],
                               gap["best_development_mae_eV"], abs_tol=1e-7)
              and last["epoch"] == 59)
        check(f"{arm}:model_and_optimizer", sum(value.numel() for value in model.parameters())
              == completion["parameter_count"]
              and all(bool(torch.isfinite(value).all()) for value in best["model"].values())
              and bool(last["optimizer"]["state"])
              and all(math.isclose(group["weight_decay"], completion["config"]["weight_decay"])
                      for group in last["optimizer"]["param_groups"])
              and all(math.isclose(group["lr"], completion["config"]["minimum_learning_rate"])
                      for group in last["optimizer"]["param_groups"]))
        check(f"{arm}:checkpoint_resume_state", all(
            key in last for key in ("python_rng_state", "numpy_rng_state",
                                   "torch_rng_state", "accelerator_rng_state")
        ))

    check("paired_row_identity", torch.equal(
        development[BASELINE]["source_idx"], development[CANDIDATE]["source_idx"]
    ) and torch.equal(
        development[BASELINE]["target_eV"], development[CANDIDATE]["target_eV"]
    ))
    receipt = {
        "format": "molgap-pcqm-distance-angle-terminal-replay-v1",
        "audited_at": "2026-09-23",
        "checks": checks,
        "artifact_status": "verified" if all(checks.values()) else "failed",
        "inference_executed": False,
        "deterministic_resume_executed": False,
        "paired_acceptance_ref": "results/local_acceptance_20260923.json",
        "candidate_minus_baseline_mae_eV": local_pair["candidate_minus_baseline_mae_eV"],
        "nominated_under_frozen_point_gate": local_pair["nominated"],
        "protected_roles": "untouched",
        "selected_artifact_count": len(audit["selected_artifacts"]),
    }
    path = EXPERIMENT / "results/terminal_replay_20260923.json"
    path.write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"artifact_status": receipt["artifact_status"],
                      "failed_checks": [name for name, passed in checks.items() if not passed],
                      "candidate_minus_baseline_mae_eV": receipt["candidate_minus_baseline_mae_eV"]},
                     indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
