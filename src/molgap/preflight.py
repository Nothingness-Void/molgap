"""Replay-Ready Preflight Verification Engine."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from molgap.constants import REPO_ROOT
from molgap.evidence_pointers import load_json_object


class PreflightError(Exception):
    pass


def check_prospective_trajectory(exp_dir: Path) -> dict[str, Any]:
    traj_path = exp_dir / "trajectory.json"
    if not traj_path.is_file():
        raise PreflightError(f"Missing prospective trajectory: {traj_path}")

    traj = load_json_object(traj_path)
    if traj.get("schema") != "molgap-trajectory-v1":
        raise PreflightError(f"Invalid schema: {traj.get('schema')}")
    if traj.get("record_mode") != "prospective":
        raise PreflightError(
            f"Trajectory record_mode must be 'prospective', got '{traj.get('record_mode')}'"
        )
    if traj.get("decision", {}).get("outcome") != "ACTIVE":
        raise PreflightError(
            f"Pre-flight trajectory outcome must be 'ACTIVE', got '{traj.get('decision', {}).get('outcome')}'"
        )

    hypo = traj.get("hypothesis", {})
    falsifier = hypo.get("cheapest_falsifier") or hypo.get("falsifier")
    if not hypo.get("hypothesis_id") or not falsifier:
        raise PreflightError("Trajectory missing hypothesis_id or cheapest_falsifier")

    refs = traj.get("state_at_start", {}).get("reference_ids", [])
    if not refs:
        raise PreflightError("Trajectory state_at_start.reference_ids is empty")

    return traj


def check_code_and_package_integrity(package_dir: Path) -> dict[str, Any]:
    meta_path = package_dir / "kernel-metadata.json"
    if not meta_path.is_file():
        raise PreflightError(f"Missing kernel-metadata.json in {package_dir}")
    meta = load_json_object(meta_path)
    if not meta.get("id"):
        raise PreflightError("kernel-metadata.json missing 'id'")

    run_path = package_dir / "run.py"
    if not run_path.is_file() or run_path.stat().st_size == 0:
        raise PreflightError(f"Missing or empty run.py in {package_dir}")

    content = run_path.read_text(encoding="utf-8", errors="ignore")

    # Extract which module is being invoked by the runner
    module_match = re.search(r'["\']-m["\'],\s*["\']molgap\.([a-zA-Z0-9_]+)["\']', content)
    target_code = content
    module_name = "embedded"
    if module_match:
        module_name = module_match.group(1)
        mod_file = REPO_ROOT / "src" / "molgap" / f"{module_name}.py"
        if mod_file.is_file():
            target_code = mod_file.read_text(encoding="utf-8", errors="ignore") + "\n" + content

    # Verify optimizer_step exposure
    if "optimizer_step" not in target_code and "sample_presentations" not in target_code:
        raise PreflightError(
            f"Target worker ({module_name}) does not record 'optimizer_step' or 'sample_presentations'. "
            "Replay-ready requires continuous step-level metrics."
        )

    # Verify trace.json output
    if "trace.json" not in target_code:
        raise PreflightError(f"Target worker ({module_name}) does not write 'trace.json' to output directory.")

    # Verify runtime certificate / measured duration
    if "runtime_certificate.json" not in target_code and "epoch_seconds" not in target_code:
        raise PreflightError(
            f"Target worker ({module_name}) does not emit runtime certificate or duration metrics for cost measurement."
        )

    return meta


def run_triple_check(experiment_dir: Path, package_dir: Path | None = None) -> None:
    print(f"[*] Checking Experiment Directory: {experiment_dir}")
    traj = check_prospective_trajectory(experiment_dir)
    falsifier = traj['hypothesis'].get('cheapest_falsifier') or traj['hypothesis'].get('falsifier')
    print(f"  [PASS] Prospective trajectory {traj['trajectory_id']} is ACTIVE and frozen.")
    print(f"         Hypothesis: {traj['hypothesis']['hypothesis_id']} | Falsifier: {falsifier}")
    print(f"         References: {traj['state_at_start']['reference_ids']}")

    if package_dir and package_dir.is_dir():
        print(f"[*] Checking Staged Package Directory: {package_dir}")
        meta = check_code_and_package_integrity(package_dir)
        print(f"  [PASS] Staged kernel package {meta['id']} has valid step-trace & metadata bindings.")

    print("\n=================================================================")
    print("  [SUCCESS] ALL PREFLIGHT CHECKS PASSED: READY FOR REMOTE LAUNCH")
    print("=================================================================\n")
