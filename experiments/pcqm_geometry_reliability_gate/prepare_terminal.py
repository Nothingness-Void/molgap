"""Translate the accepted local screen into the existing RML terminal input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPERIMENT = Path("experiments/pcqm_geometry_reliability_gate")
SOURCE = Path("experiments/pcqm_distance_angle_500k")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--finalized-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    acceptance = json.loads((repo / EXPERIMENT / "acceptance.json").read_text(encoding="utf-8"))
    analysis = json.loads((repo / EXPERIMENT / "analysis.json").read_text(encoding="utf-8"))
    if not acceptance["accepted"] or not analysis["frozen_screen_pass"]:
        raise ValueError("the local screen has not passed acceptance")
    if acceptance["evidence_id"] != "pcqm-geometry-reliability-screen-positive-20260928":
        raise ValueError("acceptance evidence identity changed")
    files = [
        EXPERIMENT / "protocol.md",
        EXPERIMENT / "analysis.json",
        EXPERIMENT / "decision.md",
        EXPERIMENT / "acceptance.json",
        SOURCE / "v5_evidence.json",
        SOURCE / "results/local_acceptance_20260923.json",
    ]
    hashes = {path.as_posix(): _sha256(repo / path) for path in files}
    if hashes[(SOURCE / "v5_evidence.json").as_posix()] != analysis["source_envelope_sha256"]:
        raise ValueError("source envelope hash changed")
    if hashes[(SOURCE / "results/local_acceptance_20260923.json").as_posix()] != analysis["source_acceptance_sha256"]:
        raise ValueError("source acceptance hash changed")
    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": acceptance["evidence_id"],
        "track": "B",
        "scope": "accepted_existing_prediction_geometry_fusion_screen",
        "legacy_contract": "molgap-pcqm-geometry-fusion-screen-v1",
        "outcome": acceptance["outcome"],
        "authority": {"pointers": [
            (EXPERIMENT / name).as_posix()
            for name in ("protocol.md", "analysis.json", "decision.md", "acceptance.json")
        ]},
        "artifacts": [
            {
                "name": "fusion_screen_analysis",
                "locator": (EXPERIMENT / "analysis.json").as_posix(),
                "sha256": hashes[(EXPERIMENT / "analysis.json").as_posix()],
                "availability": "committed_metadata_verified",
            },
            {
                "name": "screen_decision",
                "locator": (EXPERIMENT / "decision.md").as_posix(),
                "sha256": hashes[(EXPERIMENT / "decision.md").as_posix()],
                "availability": "committed_metadata_verified",
            },
        ],
        "role_use": acceptance["role_use"],
        "migration": {
            "migrated_at": "2026-09-28",
            "verification_scope": "Hash-gated existing-prediction cross-fitted screen; no model execution",
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
        },
    }
    terminal = {
        "format": "molgap-rml-terminal-package-v1",
        "trajectory_id": "TB-pcqm-geometry-reliability-screen-20260928",
        "run_id": acceptance["run_id"],
        "action_id": "A001",
        "finalized_at": args.finalized_at,
        "acceptance_ref": (EXPERIMENT / "acceptance.json").as_posix(),
        "artifact_hashes": hashes,
        "evidence": evidence,
        "decision": acceptance["trajectory_decision"],
        "costs": acceptance["costs"],
        "roles": acceptance["roles"],
        "role_use": acceptance["role_use"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
