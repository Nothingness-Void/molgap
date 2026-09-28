"""Close the accepted MetaGIN v2 run through the existing V5/RML adapter.

Only retained accepted artifacts are translated. No model forward, training,
new role read, or platform submission occurs here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.terminal_wiring import close_terminal_arm
from molgap.research_memory.trace import file_digest


ROOT = REPO_ROOT / "experiments/pcqm_metagin_2d_100k"
ATTEMPT = ROOT / "attempt_v2"
RESULTS = ATTEMPT / "results"
CANDIDATE = REPO_ROOT / "platforms/_records/kaggle/training/metagin_2d_s42_v2/molgap-metagin-2d-s42-v2"
ADAPTER = REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py"
REL = "experiments/pcqm_metagin_2d_100k"
ATTEMPT_REL = REL + "/attempt_v2"
MODE = "metagin_2d_3hop_4x256"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(finalized_at: str) -> dict[str, str]:
    accepted = read(RESULTS / "acceptance.json")
    record = read(CANDIDATE / "arm_record.json")
    native_cost = read(CANDIDATE / "native_cost.json")
    native_roles = read(CANDIDATE / "observed_role_history.json")
    frozen = read(ATTEMPT / "rml_plan/trajectory.json")
    if (not accepted["accepted"] or accepted["nomination_gate_passed"]
            or accepted["run_id"] != record["run_id"]
            or accepted["trajectory_id"] != frozen["trajectory_id"]
            or frozen["actions"][0]["run_ids"] != [record["run_id"]]
            or native_cost["allocated_device_count"] != 2
            or native_cost["training_completed"] is not True
            or native_roles["official_validation_role_read"] is not False
            or native_roles["test_dev_role_read"] is not False
            or native_roles["test_challenge_role_read"] is not False):
        raise ValueError("MetaGIN terminal is not the accepted negative v2 run")

    spec = importlib.util.spec_from_file_location("metagin_shared_terminal_adapter", ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(adapter)
    adapter.ROOT, adapter.ROOT_REL = ATTEMPT, ATTEMPT_REL
    adapter.SHARED_ROOT, adapter.SHARED_REL = ROOT, REL
    adapter.RESULTS = RESULTS
    adapter.RECORD_ROOT, adapter.CANDIDATE_ROOT = CANDIDATE.parent, CANDIDATE
    adapter.RAW_ACCEPTANCE = RESULTS / "terminal_adapter_input.json"
    adapter.SOURCE_ACCEPTANCE_REL = "results/terminal_adapter_input.json"
    adapter.TRAJECTORY_ID, adapter.RUN_ID = frozen["trajectory_id"], record["run_id"]
    adapter.MODE, adapter.ACTION_ID = MODE, frozen["actions"][0]["action_id"]
    adapter.EVIDENCE_ID = "pcqm-metagin-2d-3hop-100k-s42-v2"
    adapter.COST_EVENT_ID = frozen["actions"][0]["cost_event_ids"][0]
    adapter.LOCAL_RECORD_URI = "external://local-platform-record/" + CANDIDATE.relative_to(REPO_ROOT).as_posix()
    adapter.DECISION_OUTCOME = "NEGATIVE_UNDER_CONTRACT"
    adapter.SCIENTIFIC_STATUS = "negative"
    adapter.NEXT_ALLOWED_ACTIONS = []
    adapter.REOPEN_CONDITIONS = ["distinct evidence-backed 2D hypothesis and explicit new compute decision"]
    adapter.FINALIZED_AT, adapter.MIGRATED_AT = finalized_at, finalized_at[:10]
    adapter.PLATFORM, adapter.HARDWARE = "kaggle2", "Tesla_T4_16GB"
    adapter.rel = lambda name: (f"{ATTEMPT_REL}/rml_plan/trajectory.json" if name == "trajectory.json"
                                else f"{ATTEMPT_REL}/{name}")
    adapter.shared_rel = lambda name: (f"{ATTEMPT_REL}/source_config.json" if name == "source_config.json"
                                       else f"{REL}/{name}")
    adapter.make_trace = lambda raw, checkpoint: read(CANDIDATE / "canonical_trace.json")

    normalized_record = dict(record)
    normalized_record["contract"] = {"architecture_fingerprint": record["architecture_config_identity"]}
    adapter.write(RESULTS / "normalized_arm_record.json", normalized_record)
    adapter.write(adapter.RAW_ACCEPTANCE, {
        "accepted": True,
        "model_inference_executed": False,
        "source_commit": accepted["source_commit"],
        "source_archive_sha256": accepted["source_archive_sha256"],
        "reference_development_gap_mae_eV": accepted["reference_dev_mae_eV"],
        "candidates": {MODE: {
            "record": normalized_record,
            "recomputed_development_gap_mae_eV": accepted["candidate_dev_mae_eV"],
            "gate": {"gain_eV": accepted["gain_eV"], "required_gain_eV": 0.003,
                     "passed": False, "training_stochasticity_accounted": False,
                     "row_bootstrap_is_sufficient_alone": False},
            "paired_error_delta_bootstrap_95_eV": accepted["paired"]["paired_row_bootstrap"]["ci95"],
        }},
    })
    original_load = adapter.load

    def load(path):
        path = Path(path)
        if path == CANDIDATE / "arm_record.json":
            return original_load(RESULTS / "normalized_arm_record.json")
        if path == ATTEMPT / "role_plan.json":
            return original_load(ROOT / "role_plan.json")
        return original_load(path)

    adapter.load = load
    for name in ("native_cost.json", "observed_role_history.json", "arm_record.json",
                 "completion_manifest.json", "sidecar_acceptance_binding.json"):
        adapter.write(RESULTS / name, read(CANDIDATE / name))
    adapter.EXTRA_ARTIFACTS = tuple(
        (Path(name).stem, f"{ATTEMPT_REL}/results/{name}") for name in
        ("acceptance.json", "normalized_arm_record.json", "native_cost.json",
         "observed_role_history.json", "arm_record.json", "completion_manifest.json",
         "sidecar_acceptance_binding.json", "terminal_adapter_input.json")
    ) + (("submission_receipt", f"{ATTEMPT_REL}/submission_receipt.json"),)
    original_write = adapter.write

    def write(path, value):
        if Path(path) == RESULTS / "cost_records.json":
            event = value["costs"][0]
            event["attempt_id"] = "v2"
            event["measurement"]["device_hours"] = {
                "status": "measured", "value": native_cost["allocated_device_seconds"] / 3600}
            event["measurement"]["wall_hours"] = {
                "status": "measured", "value": native_cost["wall_seconds"] / 3600}
        original_write(path, value)

    adapter.write = write
    adapter.main()
    if file_digest(RESULTS / "canonical_trace.json") != file_digest(CANDIDATE / "canonical_trace.json"):
        raise ValueError("Native canonical trace bytes changed during retention")
    return {"trajectory": f"{ATTEMPT_REL}/rml_plan/trajectory.json",
            "terminal": f"{ATTEMPT_REL}/results/terminal.json",
            "trace": f"{ATTEMPT_REL}/results/canonical_trace.json"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-at", required=True)
    args = parser.parse_args()
    inputs = prepare(args.finalized_at)
    print(json.dumps(close_terminal_arm(REPO_ROOT, **inputs), default=str))
