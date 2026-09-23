"""Replay the retained Xi'an full-model determinism decision from JSON evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/pcqm_xian_determinism"
PROBE = EXPERIMENT / "results/initial_probe"
PAYLOAD = PROBE / "model-replay"


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def main() -> None:
    submission = read_json(PROBE / "submission.json")
    scheduler = read_json(PROBE / "scheduler_20260923.json")["job"]
    acceptance = read_json(PAYLOAD / "acceptance.json")
    first = read_json(PAYLOAD / "first.json")
    second = read_json(PAYLOAD / "second.json")
    checks: dict[str, bool] = {}

    def check(name: str, passed: bool) -> None:
        checks[name] = bool(passed)

    expected = {}
    for line in (PAYLOAD / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            expected[Path(path).name] = digest
    check("remote_sha256_sums", set(expected) == {"acceptance.json", "first.json", "second.json"}
          and all(hashlib.sha256((PAYLOAD / name).read_bytes()).hexdigest() == digest
                  for name, digest in expected.items()))
    check("scheduler", scheduler["job_id"] == submission["job_id"]
          and scheduler["state"] == "COMPLETED" and scheduler["exit_code"] == "0:0"
          and scheduler["allocated_dcu"] == 1)
    check("same_frozen_contract", acceptance["same_contract"] is True
          and first["cache_aggregate_sha256"] == second["cache_aggregate_sha256"]
          and first["model_source_hashes"] == second["model_source_hashes"]
          and first["settings"] == second["settings"]
          and first["parameter_count"] == second["parameter_count"]
          and first["batch_size"] == second["batch_size"] == submission["batch_size"])
    check("complete_three_replays", acceptance["complete"] is True
          and first["complete"] is True and second["complete"] is True
          and len(first["repetitions"]) == len(second["repetitions"])
          == submission["replays_per_process"])
    check("within_process_identity", all(
        record["within_process_initial_model_identity"] is True
        and record["within_process_prediction_identity"] is True
        and record["within_process_gradient_identity"] is True
        for record in (first, second)
    ))
    first_replay = first["repetitions"][0]
    second_replay = second["repetitions"][0]
    different = sorted(
        name for name in first_replay["gradient_fingerprints"]
        if first_replay["gradient_fingerprints"][name]
        != second_replay["gradient_fingerprints"][name]
    )
    check("cross_process_initial_and_prediction", all(
        first_replay[name] == second_replay[name]
        for name in ("initial_model_sha256", "prediction_sha256", "loss")
    ) and acceptance["initial_model_bitwise_replay"] is True
          and acceptance["prediction_bitwise_replay"] is True)
    check("cross_process_gradient_failure", different == sorted(
        acceptance["differing_gradient_parameters"]
    ) and len(different) == acceptance["differing_gradient_parameter_count"] == 269
          and first_replay["gradient_sha256"] != second_replay["gradient_sha256"]
          and acceptance["gradient_bitwise_replay"] is False
          and acceptance["accepted"] is False
          and acceptance["short_training_retest_authorized"] is False)
    check("protected_roles", all(
        record["official_validation_role_read"] is False
        and record["test_dev_role_read"] is False
        for record in (first, second, acceptance, submission)
    ))
    receipt = {
        "format": "molgap-xian-determinism-terminal-replay-v1",
        "audited_at": "2026-09-23",
        "checks": checks,
        "artifact_status": "verified" if all(checks.values()) else "failed",
        "full_model_replay_gate": "failed_cross_process_gradients" if all(checks.values()) else "unverified",
        "differing_gradient_parameter_count": len(different),
        "short_training_retest_authorized": False,
        "model_inference_executed_locally": False,
        "protected_roles": "untouched",
    }
    path = PROBE / "terminal_replay_20260923.json"
    path.write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"artifact_status": receipt["artifact_status"],
                      "failed_checks": [name for name, passed in checks.items() if not passed],
                      "full_model_replay_gate": receipt["full_model_replay_gate"]}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
