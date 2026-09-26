"""Prepare saved-artifact relation-study terminals using the retained K1 adapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .constants import REPO_ROOT
from .k1_relation_study_records import REL, ROOT, load, save
from .k1_relation_study_runtime import SLOTS, RUNS, TRAJECTORIES
from .k1_terminal_analysis import accepted_native_trace
from .research_memory.trace import file_digest


def allocated_arm_cost(execution, mode):
    """Allocate actual notebook occupancy once, including unused assigned GPUs.

    Each worker owns its observed wall/device interval. Bootstrap and idle
    occupancy are attributed to the first worker for ledger bookkeeping only;
    native per-epoch timing is unchanged and remains the throughput authority.
    """
    workers = execution["workers"]
    worker = next(row for row in workers if row["mode"] == mode)
    allocated = execution["total_allocated_device_seconds"]
    active = sum(row["allocated_device_seconds"] for row in workers)
    if allocated < active or not execution["complete"] or not worker["complete"]:
        raise ValueError("Incomplete or inconsistent observed notebook cost")
    overhead = allocated - active if workers[0]["mode"] == mode else 0
    return {"device_seconds": worker["allocated_device_seconds"] + overhead,
        "wall_seconds": execution["total_job_wall_seconds"] if len(workers) == 1 else worker["worker_wall_seconds"],
        "allocated_overhead_seconds": overhead,
        "allocation_semantics": "all observed notebook bootstrap/idle occupancy assigned once to first worker; not per-arm model throughput"}


def prepare(mode, finalized_at):
    slot = next(slot for slot, modes in SLOTS.items() if mode in modes)
    arm_rel = f"{REL}/arms/{mode.removeprefix('neural_atom_k1_')}"
    arm = REPO_ROOT / arm_rel
    records = REPO_ROOT / f"platforms/_records/kaggle/training/k1_relation_resolution_{slot}_v1"
    candidate = records / "pcqm_k1_relation_resolution" / mode
    accepted = load(records / "acceptance.json")
    frozen = load(arm / "rml_plan/trajectory.json")
    evidence_id = "pcqm-" + mode.removeprefix("neural_atom_").replace("_", "-") + "-100k-s42"
    if (accepted.get("accepted") is not True or accepted.get("model_inference_executed") is not False
        or frozen["actions"][0]["run_ids"] != [RUNS[slot]]):
        raise ValueError("Verified saved acceptance and frozen run identity required")
    gate = accepted["candidates"][mode]["gate"]
    if gate["passed"]:
        raise ValueError("Material winner needs an explicit controller decision; do not auto-close it")
    positive = gate["gain_eV"] > 0 and gate["paired_row_bootstrap_upper_eV"] < 0
    receipt = load(ROOT / "submission_receipt_v1.json")
    binding = next(row for row in receipt["jobs"] if row["slot"] == slot)
    record = accepted["candidates"][mode]["record"]
    if (binding["embedded_trace_run_id"] != RUNS[slot]
        or binding["source_commit"] != record["source_commit"]
        or binding["source_archive_sha256"] != record["contract"]["source_archive_sha256"]):
        raise ValueError("Physical submission receipt does not bind this native trace")
    results = arm / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name in ("native_cost.json", "observed_role_history.json", "arm_record.json", "completion_manifest.json"):
        save(results / name, load(candidate / name))
    execution = load(candidate.parent / "execution_summary.json")
    save(results / "execution_summary.json", execution)
    allocation = allocated_arm_cost(execution, mode)
    save(results / "cost_allocation.json", allocation)
    adapter_path = REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py"
    spec = importlib.util.spec_from_file_location("relation_existing_k1_terminal_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.ROOT, adapter.ROOT_REL = arm, arm_rel
    adapter.SHARED_ROOT, adapter.SHARED_REL = ROOT, REL
    adapter.RESULTS, adapter.RECORD_ROOT, adapter.CANDIDATE_ROOT = results, records, candidate
    adapter.RAW_ACCEPTANCE = results / "raw_acceptance.json"
    adapter.SOURCE_ACCEPTANCE_REL = "results/raw_acceptance.json"
    adapter.TRAJECTORY_ID, adapter.RUN_ID, adapter.MODE = TRAJECTORIES[mode], RUNS[slot], mode
    adapter.ACTION_ID, adapter.EVIDENCE_ID = "A001", evidence_id
    adapter.COST_EVENT_ID = frozen["actions"][0]["cost_event_ids"][0]
    adapter.LOCAL_RECORD_URI = "external://local-platform-record/" + candidate.relative_to(REPO_ROOT).as_posix()
    adapter.DECISION_OUTCOME = "POSITIVE_BELOW_GATE" if positive else "NEGATIVE_UNDER_CONTRACT"
    adapter.SCIENTIFIC_STATUS = "positive_below_gate" if positive else "negative_under_contract"
    adapter.NEXT_ALLOWED_ACTIONS = ["separate predeclared accepted-checkpoint NO_TRAIN audit after all training arms are accepted"]
    adapter.REOPEN_CONDITIONS = ["new scientific mechanism and explicit prospective compute authority"]
    adapter.FINALIZED_AT, adapter.MIGRATED_AT = finalized_at, finalized_at[:10]
    adapter.PLATFORM, adapter.HARDWARE = "kaggle2", record["runtime_certificate"]["accelerator"]
    adapter.rel = lambda name: f"{arm_rel}/rml_plan/trajectory.json" if name == "trajectory.json" else f"{arm_rel}/{name}"
    adapter.make_trace = lambda raw, checkpoint: accepted_native_trace(candidate / "canonical_trace.json", raw, checkpoint)
    adapter.EXTRA_ARTIFACTS = tuple((name.removesuffix(".json"), f"{arm_rel}/results/{name}")
        for name in ("native_cost.json", "observed_role_history.json", "arm_record.json",
                     "completion_manifest.json", "execution_summary.json", "cost_allocation.json")) + (
        ("submission_receipt", f"{REL}/submission_receipt_v1.json"),
        ("acceptance_interface_diagnosis", f"{REL}/acceptance_interface_diagnosis.md"),)
    original_write = adapter.write

    def write_with_observed_cost(path, value):
        if Path(path) == results / "cost_records.json":
            # The existing adapter reuses this same object in all bound outputs.
            event = value["costs"][0]
            event["attempt_id"] = "v1"
            event["measurement"]["device_hours"] = {"value": allocation["device_seconds"] / 3600, "status": "measured"}
            event["measurement"]["wall_hours"] = {"value": allocation["wall_seconds"] / 3600, "status": "measured"}
        original_write(path, value)

    adapter.write = write_with_observed_cost
    adapter.write(adapter.RAW_ACCEPTANCE, accepted)
    adapter.main()
    if file_digest(results / "canonical_trace.json") != file_digest(candidate / "canonical_trace.json"):
        raise ValueError("Native canonical bytes changed during terminal preparation")
    return {"trajectory": f"{arm_rel}/rml_plan/trajectory.json",
            "terminal": f"{arm_rel}/results/terminal.json", "trace": f"{arm_rel}/results/canonical_trace.json"}
