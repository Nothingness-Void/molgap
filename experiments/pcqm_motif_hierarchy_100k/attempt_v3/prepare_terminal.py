"""Accept retained v3 tensors and reuse the K1 V5/RML terminal adapter."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.k1_terminal_analysis import accepted_native_trace
from molgap.research_memory.trace import file_digest


REL = "experiments/pcqm_motif_hierarchy_100k"
ATTEMPT_REL = f"{REL}/attempt_v3"
ATTEMPT = REPO_ROOT / ATTEMPT_REL
RECORDS = REPO_ROOT / "platforms/_records/kaggle/training/motif_hierarchy_gpu_v3"
RUN_ROOT = RECORDS / "pcqm_k1_motif_hierarchy"
MODE = "neural_atom_k1_motif_hierarchy"
ARM = RUN_ROOT / MODE
REFERENCE = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference"
RUN_ID = "kaseichou/molgap-k1-motif-hierarchy-s42:v3"
TRAJECTORY_ID = "TC-k1-motif-hierarchy-100k-s42-v3"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def prepare(finalized_at: str) -> dict[str, str]:
    """No training or model inference; exact physical run and evidence only."""
    receipt = load(ATTEMPT / "submission_receipt.json")
    source = load(ATTEMPT / "source_config.json")
    frozen = load(ATTEMPT / "gpu_rml_plan/trajectory.json")
    launch = load(RUN_ROOT / "launch_identity.json")
    execution = load(RUN_ROOT / "execution_summary.json")
    native = load(ARM / "native_cost.json")
    roles = load(ARM / "observed_role_history.json")
    record = load(ARM / "arm_record.json")
    if (receipt["kernel_id"] != 136356323 or receipt["version"] != 3
        or receipt["run_id"] != launch["run_id"] != RUN_ID
        or execution["run_id"] != native["run_id"] != roles["run_id"] != RUN_ID
        or frozen["trajectory_id"] != roles["trajectory_id"] != TRAJECTORY_ID
        or frozen["actions"][0]["run_ids"] != [RUN_ID]
        or source["source_commit"] != receipt["source_commit"] != record["source_commit"] != launch["source_commit"]
        or receipt["source_archive_sha256"] != launch["source_archive_sha256"] != record["contract"]["source_archive_sha256"]):
        raise ValueError("Frozen source, trajectory or physical version changed")
    if (execution["complete"] is not True or native["training_completed"] is not True
        or native["allocated_device_count"] != len(execution["allocated_device_names"])
        or native["allocated_device_count"] != 2 or native["used_device_count"] != 1
        or any("T4" not in name for name in execution["allocated_device_names"])
        or abs(native["wall_seconds"] - execution["total_job_wall_seconds"]) > 1e-6):
        raise ValueError("Allocated T4 execution or native cost incomplete")
    for key in ("training_labels_read", "development_labels_read", "development_metric_computed", "development_selection_used"):
        if roles.get(key) is not True:
            raise ValueError(f"Missing observed role use: {key}")
    for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if roles.get(key) is not False:
            raise ValueError(f"Protected role was read or unknown: {key}")

    native_trace = accepted_native_trace(
        ARM / "canonical_trace.json", load(ARM / "trace.json"), file_digest(ARM / "last_checkpoint.pt"))
    if (native_trace["run_id"] != RUN_ID or native_trace["trajectory_id"] != TRAJECTORY_ID
        or len(native_trace["observations"]) != 40
        or native_trace["observations"][-1]["optimizer_step"] != 31240
        or native_trace["observations"][-1]["sample_presentations"] != 3998720):
        raise ValueError("Canonical training exposure or checkpoint changed")

    shared = module(REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py", "motif_k1_saved_acceptance")
    accepted = shared.accept(REFERENCE, RUN_ROOT, modes=(MODE,),
        expected_parameters={MODE: 3_693_505})
    row = accepted["candidates"][MODE]
    if (accepted["selected_candidate"] is not None or row["gate"]["passed"] is not False
        or row["gate"]["gain_eV"] >= 0):
        raise ValueError("Negative terminal adapter cannot close a winning candidate")
    accepted.update(run_id=RUN_ID, source_commit=source["source_commit"],
        source_archive_sha256=receipt["source_archive_sha256"])

    adapter = module(REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py",
                     "motif_existing_k1_terminal_adapter")
    results = ATTEMPT / "results"
    adapter.ROOT, adapter.ROOT_REL = ATTEMPT, ATTEMPT_REL
    adapter.SHARED_ROOT, adapter.SHARED_REL = REPO_ROOT / REL, REL
    adapter.RESULTS, adapter.RECORD_ROOT, adapter.CANDIDATE_ROOT = results, RECORDS, ARM
    adapter.RAW_ACCEPTANCE = results / "raw_acceptance.json"
    adapter.SOURCE_ACCEPTANCE_REL = "results/raw_acceptance.json"
    adapter.TRAJECTORY_ID, adapter.RUN_ID, adapter.MODE = TRAJECTORY_ID, RUN_ID, MODE
    adapter.ACTION_ID = frozen["actions"][0]["action_id"]
    adapter.COST_EVENT_ID = frozen["actions"][0]["cost_event_ids"][0]
    adapter.EVIDENCE_ID = "pcqm-k1-motif-hierarchy-100k-s42-v3"
    adapter.LOCAL_RECORD_URI = "external://local-platform-record/" + ARM.relative_to(REPO_ROOT).as_posix()
    adapter.DECISION_OUTCOME = "NEGATIVE_UNDER_CONTRACT"
    adapter.SCIENTIFIC_STATUS = "negative_under_contract"
    adapter.NEXT_ALLOWED_ACTIONS = []
    adapter.REOPEN_CONDITIONS = ["new mechanism and explicit prospective compute authority"]
    adapter.FINALIZED_AT, adapter.MIGRATED_AT = finalized_at, finalized_at[:10]
    adapter.PLATFORM, adapter.HARDWARE = "kaggle2", "Tesla_T4_16GB"
    adapter.rel = lambda name: (
        f"{ATTEMPT_REL}/gpu_rml_plan/trajectory.json" if name == "trajectory.json"
        else f"{ATTEMPT_REL}/{name}")
    shared_paths = {
        "protocol.md": f"{REL}/gpu_protocol.md",
        "training_contract.json": f"{REL}/gpu_training_contract.json",
        "source_config.json": f"{ATTEMPT_REL}/source_config.json",
    }
    adapter.shared_rel = lambda name: shared_paths[name]
    adapter.make_trace = lambda raw, checkpoint: accepted_native_trace(
        ARM / "canonical_trace.json", raw, checkpoint)
    compact = ("native_cost.json", "observed_role_history.json", "arm_record.json",
               "completion_manifest.json")
    for name in compact:
        adapter.write(results / name, load(ARM / name))
    for name in ("execution_summary.json", "launch_identity.json", "runtime_probe.json"):
        adapter.write(results / name, load(RUN_ROOT / name))
    adapter.EXTRA_ARTIFACTS = tuple(
        (Path(name).stem, f"{ATTEMPT_REL}/results/{name}")
        for name in (*compact, "execution_summary.json", "launch_identity.json", "runtime_probe.json")) + (
        ("submission_receipt", f"{ATTEMPT_REL}/submission_receipt.json"),
        ("attempt_protocol", f"{ATTEMPT_REL}/protocol.md"),)
    write = adapter.write

    def write_with_native_cost(path: Path, value):
        if Path(path) == results / "cost_records.json":
            event = value["costs"][0]
            event["attempt_id"] = "v3"
            event["measurement"]["device_hours"] = {
                "value": native["allocated_device_seconds"] / 3600, "status": "measured"}
            event["measurement"]["wall_hours"] = {
                "value": native["wall_seconds"] / 3600, "status": "measured"}
            event["measurement"]["cpu_hours"] = {
                "value": native["cpu_process_seconds"] / 3600, "status": "measured"}
        write(path, value)

    adapter.write = write_with_native_cost
    adapter.write(adapter.RAW_ACCEPTANCE, accepted)
    adapter.main()
    if file_digest(results / "canonical_trace.json") != file_digest(ARM / "canonical_trace.json"):
        raise ValueError("Native canonical trace bytes changed during terminal preparation")
    return {
        "trajectory": f"{ATTEMPT_REL}/gpu_rml_plan/trajectory.json",
        "terminal": f"{ATTEMPT_REL}/results/terminal.json",
        "trace": f"{ATTEMPT_REL}/results/canonical_trace.json",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-at", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.finalized_at), indent=2))
