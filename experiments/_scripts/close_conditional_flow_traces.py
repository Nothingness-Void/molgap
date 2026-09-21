"""Close the two retained conditional-flow traces without rewriting the joint run.

The source experiment froze one prospective joint trajectory for two training
arms.  This adapter therefore publishes trace metadata for retrospective arm
views only; it does not mint independent prospective finalizations or split the
joint cost/role evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.schemas import validate_trace_manifest
from molgap.research_memory.terminal_wiring import (
    build_default_trace_manifest,
    resolve_trace_for_terminal_arm,
)
from molgap.research_memory.trace import (
    atomic_write,
    file_digest,
    json_bytes,
    load_canonical_trace,
    validate_manifest_trace,
)


HISTORY = Path("experiments/pcqm_gptrans_conditional_flow_history")
ARMS = ("conditional_pair_readback", "conditional_pair_recurrence")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(REPO_ROOT)
    history = root / HISTORY
    joint_evidence = _load(history / "sources/joint_evidence_snapshot.json")
    joint_receipt = _load(history / "sources/joint_finalization_receipt.json")
    joint_snapshot = _load(history / "sources/joint_prospective_snapshot.json")
    joint_contract = _load(history / "sources/joint_training_contract.json")

    run_id = "nothingnessvoid/molgap-gptrans-conditional-flow-s42:v2"
    reference_ids = list(joint_snapshot["state_at_start"]["reference_ids"])
    if len(reference_ids) != 1:
        raise ValueError("historical joint snapshot must identify exactly one reference")

    terminal_view = {"run_id": run_id, "evidence": joint_evidence}
    outputs = []
    for arm in ARMS:
        arm_dir = history / arm
        trajectory_path = arm_dir / "trajectory.json"
        trace_path = arm_dir / "canonical_trace.json"
        trajectory = _load(trajectory_path)
        trace = load_canonical_trace(trace_path)

        required, resolved = resolve_trace_for_terminal_arm(
            root,
            trajectory,
            terminal_view,
            arm_identifier=arm,
            canonical_trace=trace_path,
        )
        if not required or resolved.resolve() != trace_path.resolve():
            raise ValueError(f"arm trace did not resolve to its retained canonical file: {arm}")

        manifest_trajectory = copy.deepcopy(trajectory)
        manifest_trajectory["state_at_start"]["reference_ids"] = reference_ids
        manifest = build_default_trace_manifest(
            root, manifest_trajectory, terminal_view, trace
        )
        accepted_arm = trajectory["result"]["arm_acceptance"]
        model_identity = (
            f"{joint_contract['model_id']}:{arm}:"
            f"params_{accepted_arm['parameters']}"
        )
        prefix = (HISTORY / arm).as_posix()
        manifest.update(
            reference_id=reference_ids[0],
            model_identity=model_identity,
            trace_artifact_ref=f"{prefix}/canonical_trace.json",
            trace_artifact_sha256=file_digest(trace_path),
            terminal_evidence_ref=(
                f"{HISTORY.as_posix()}/sources/joint_evidence_snapshot.json"
            ),
            backtest_eligibility={
                "eligible": False,
                "exclusion_reasons": [
                    "retrospective_arm_view_not_independently_frozen",
                    "per_epoch_step_and_sample_counts_not_cumulative_axes",
                ],
            },
        )
        manifest["comparability_identity"]["architecture_identity"] = model_identity
        manifest = validate_trace_manifest(manifest)
        validate_manifest_trace(manifest, trace)
        manifest_bytes = json_bytes(manifest)

        raw_pointer = trajectory["result"]["provenance"]["source"]
        raw_digest = trajectory["result"]["provenance"]["sha256"]
        if joint_receipt["input_artifact_hashes"].get(raw_pointer) != raw_digest:
            raise ValueError(f"joint receipt no longer binds the raw trace: {arm}")

        receipt = {
            "format": "molgap-rml-retrospective-trace-closure-v1",
            "trajectory_id": trajectory["trajectory_id"],
            "run_id": run_id,
            "arm": arm,
            "trace_status": "available",
            "record_mode": "retrospective_partial",
            "independent_finalization": False,
            "joint_trajectory_id": joint_snapshot["trajectory_id"],
            "joint_finalization_id": joint_receipt["finalization_id"],
            "raw_trace_ref": raw_pointer,
            "raw_trace_sha256": raw_digest,
            "canonical_trace_ref": f"{prefix}/canonical_trace.json",
            "canonical_trace_sha256": file_digest(trace_path),
            "trace_manifest_ref": f"{prefix}/trace_manifest.json",
            "trace_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "replay_ready": False,
            "replay_blockers": manifest["backtest_eligibility"]["exclusion_reasons"],
        }

        manifest_ref = f"{prefix}/trace_manifest.json"
        receipt_ref = f"{prefix}/trace_closure_receipt.json"
        refs = trajectory["result"]["evidence_refs"]
        trajectory["result"]["evidence_refs"] = list(
            dict.fromkeys([*refs, manifest_ref, receipt_ref])
        )

        atomic_write(arm_dir / "trace_manifest.json", manifest_bytes)
        atomic_write(arm_dir / "trace_closure_receipt.json", json_bytes(receipt))
        atomic_write(trajectory_path, json_bytes(trajectory))
        outputs.append(
            {
                "arm": arm,
                "trajectory_id": trajectory["trajectory_id"],
                "trace_status": "available",
                "manifest": manifest_ref,
            }
        )

    print(json.dumps({"status": "closed", "arms": outputs}, indent=2))


if __name__ == "__main__":
    main()
