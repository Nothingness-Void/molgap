"""Prepare two separate RML terminals from accepted saved artifacts, without inference."""
from __future__ import annotations

import importlib.util

from molgap.constants import REPO_ROOT


EXPERIMENT_REL = "experiments/pcqm_k1_conjugated_hyperedge_100k"
EXPERIMENT = REPO_ROOT / EXPERIMENT_REL
RECORD_REL = "platforms/_records/kaggle/training/k1_conjugated_dual_s42_v1"
RECORD = REPO_ROOT / RECORD_REL
SIDECAR_ACCEPTANCE = (
    REPO_ROOT
    / "platforms/_records/kaggle/training/k1_conjugated_sidecar_v1"
    / "molgap-k1-conjugated-sidecar-v1/acceptance.json"
)
FROZEN_SOURCE = "b2340edd93bff46172959bdc2f9771da51fad3d6"
FROZEN_ARCHIVE = "b2cc400565d02f26b448bd2e749352143a7ce875c08c8a2c2af8059c7ca0f30d"
FROZEN_SIDECAR = "d3e34ccba8c35184c25fd9c00d5de5989b4c2cc943fb76020d33ba9a87614cfe"
MODES = {
    "oneshot": ("neural_atom_k1_conjugated_oneshot", "NEGATIVE_UNDER_CONTRACT", "negative"),
    "persistent": ("neural_atom_k1_conjugated_persistent", "POSITIVE_BELOW_GATE", "positive_below_gate"),
}


def shared_adapter():
    path = (
        REPO_ROOT
        / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py"
    )
    spec = importlib.util.spec_from_file_location("shared_rml_terminal_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Shared RML terminal adapter unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    raw_acceptance = shared_adapter().load(RECORD / "acceptance.json")
    if (
        raw_acceptance.get("accepted") is not True
        or raw_acceptance.get("model_inference_executed") is not False
        or raw_acceptance.get("source_commit") != FROZEN_SOURCE
        or raw_acceptance.get("source_archive_sha256") != FROZEN_ARCHIVE
        or raw_acceptance.get("conjugated_sidecar_aggregate_sha256") != FROZEN_SIDECAR
        or raw_acceptance.get("selected_candidate") is not None
    ):
        raise RuntimeError("Accepted dual-arm identity or gate changed")
    sidecar_acceptance = shared_adapter().load(SIDECAR_ACCEPTANCE)
    if (
        sidecar_acceptance.get("accepted") is not True
        or sidecar_acceptance.get("aggregate_sha256") != FROZEN_SIDECAR
        or sidecar_acceptance.get("model_inference_executed") is not False
    ):
        raise RuntimeError("Sidecar acceptance changed")
    shared = shared_adapter()
    shared.write(EXPERIMENT / "results/sidecar_acceptance.json", sidecar_acceptance)

    for arm_name, (mode, outcome, scientific_status) in MODES.items():
        arm_rel = f"{EXPERIMENT_REL}/arms/{arm_name}"
        arm = REPO_ROOT / arm_rel
        trajectory = shared.load(arm / "trajectory.json")
        action = trajectory["actions"][0]
        if (
            trajectory["trajectory_id"] != f"TC-k1-conjugated-{arm_name}-100k-s42"
            or action["action_id"] != "A001"
            or action["run_ids"] != [
                "kaseichou/molgap-k1-conjugated-dual-s42:v1-planned"
            ]
            or action["source_commit"] != FROZEN_SOURCE
            or mode not in raw_acceptance["candidates"]
            or raw_acceptance["candidates"][mode]["gate"]["passed"] is not False
        ):
            raise RuntimeError(f"Prospective trajectory or terminal gate changed: {mode}")
        adapter = shared_adapter()
        adapter.ROOT = arm
        adapter.ROOT_REL = arm_rel
        adapter.SHARED_ROOT = EXPERIMENT
        adapter.SHARED_REL = EXPERIMENT_REL
        adapter.RESULTS = arm / "results"
        adapter.RECORD_ROOT = RECORD
        adapter.CANDIDATE_ROOT = RECORD / "pcqm_k1_conjugated_dual" / mode
        adapter.RAW_ACCEPTANCE = adapter.RESULTS / "acceptance_v1.json"
        adapter.SOURCE_ACCEPTANCE_REL = "results/acceptance_v1.json"
        adapter.write(adapter.RAW_ACCEPTANCE, raw_acceptance)
        adapter.TRAJECTORY_ID = trajectory["trajectory_id"]
        adapter.RUN_ID = action["run_ids"][0]
        adapter.MODE = mode
        adapter.ACTION_ID = action["action_id"]
        adapter.EVIDENCE_ID = trajectory["result"]["evidence_ids"][0]
        adapter.LOCAL_RECORD_URI = (
            f"external://local-platform-record/{RECORD_REL}/"
            f"pcqm_k1_conjugated_dual/{mode}"
        )
        adapter.DECISION_OUTCOME = outcome
        adapter.SCIENTIFIC_STATUS = scientific_status
        adapter.NEXT_ALLOWED_ACTIONS = [
            "close this arm under the frozen 100K material gate; retain its evidence for replay"
        ]
        adapter.REOPEN_CONDITIONS = [
            "a distinct mechanism and compute budget require new prospective authority"
        ]
        adapter.FINALIZED_AT = "2026-09-24T05:58:14+09:00"
        adapter.MIGRATED_AT = "2026-09-24"
        adapter.PLATFORM = "kaggle2"
        adapter.HARDWARE = "Tesla_T4_16GB"
        adapter.EXTRA_ARTIFACTS = (
            ("sidecar_acceptance", adapter.shared_rel("results/sidecar_acceptance.json")),
            ("saved_error_attribution", adapter.shared_rel("results/saved_error_attribution.json")),
            ("submission_receipt", adapter.shared_rel("results/submission_receipt.json")),
        )
        adapter.main()
        print(f"Prepared no-inference RML terminal: {arm_name}", flush=True)


if __name__ == "__main__":
    main()
