"""Map accepted chemistry-local artifacts into existing K1/V5 terminal APIs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .constants import REPO_ROOT
from .k1_chem_local import MODES
from .k1_relation_terminal import allocated_arm_cost
from .k1_terminal_analysis import accepted_native_trace
from .research_memory.trace import atomic_write, file_digest, json_bytes


REL = "experiments/pcqm_k1_chem_local_100k"
ROOT = REPO_ROOT / REL
RECORDS = REPO_ROOT / "platforms/_records/kaggle/training/k1_chem_local_s42_v1"
CANDIDATE = RECORDS / "pcqm_k1_chem_local"
RUN_ID = "kaseichou/molgap-k1-chem-local-s42:v1"
TRAJECTORIES = {
    MODES[0]: "TC-k1-atom-pair-local-100k-s42",
    MODES[1]: "TC-k1-bond-type-local-100k-s42",
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, value):
    atomic_write(path, json_bytes(value))


def prepare(mode: str, finalized_at: str) -> dict[str, str]:
    """Translate one accepted arm, without loading a model or rerunning inference."""
    if mode not in MODES:
        raise ValueError(mode)
    accepted = _load(ROOT / "results/raw_acceptance.json")
    receipt = _load(ROOT / "submission_receipt_v1.json")
    execution = _load(CANDIDATE / "execution_summary.json")
    if (accepted.get("accepted") is not True
        or accepted.get("model_inference_executed") is not False
        or accepted.get("selected_candidate") is not None
        or accepted.get("run_id") != RUN_ID
        or receipt.get("physical_run_id") != RUN_ID
        or receipt.get("source_commit") != accepted.get("source_commit")
        or receipt.get("source_archive_sha256") != accepted.get("source_archive_sha256")
        or execution.get("run_id") != RUN_ID or execution.get("complete") is not True):
        raise ValueError("Accepted terminal and submission receipt required")
    row = accepted["candidates"][mode]
    if row["gate"]["passed"] is not False or row["gate"]["gain_eV"] >= 0:
        raise ValueError("This negative closure cannot process a winning arm")

    sub = "atom_pair_local" if mode == MODES[0] else "bond_type_local"
    arm_rel = f"{REL}/arms/{sub}"
    arm = REPO_ROOT / arm_rel
    candidate = CANDIDATE / mode
    frozen = _load(arm / "rml_plan/trajectory.json")
    if (frozen["trajectory_id"] != TRAJECTORIES[mode]
        or frozen["actions"][0]["run_ids"] != [RUN_ID]):
        raise ValueError("Prospective arm did not bind physical run")
    results = arm / "results"
    results.mkdir(parents=True, exist_ok=True)
    native_names = ("native_cost.json", "observed_role_history.json",
                    "arm_record.json", "completion_manifest.json")
    for name in native_names:
        _save(results / name, _load(candidate / name))
    _save(results / "execution_summary.json", execution)
    allocation = allocated_arm_cost(execution, mode)
    _save(results / "cost_allocation.json", allocation)

    path = REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py"
    spec = importlib.util.spec_from_file_location("k1_chem_local_existing_terminal", path)
    adapter = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(adapter)
    adapter.ROOT, adapter.ROOT_REL = arm, arm_rel
    adapter.SHARED_ROOT, adapter.SHARED_REL = ROOT, REL
    adapter.RESULTS, adapter.RECORD_ROOT, adapter.CANDIDATE_ROOT = results, RECORDS, candidate
    adapter.RAW_ACCEPTANCE = results / "raw_acceptance.json"
    adapter.SOURCE_ACCEPTANCE_REL = "results/raw_acceptance.json"
    adapter.TRAJECTORY_ID, adapter.RUN_ID, adapter.MODE = TRAJECTORIES[mode], RUN_ID, mode
    adapter.ACTION_ID = frozen["actions"][0]["action_id"]
    adapter.EVIDENCE_ID = f"pcqm-k1-{sub.replace('_', '-')}-100k-s42"
    adapter.COST_EVENT_ID = frozen["actions"][0]["cost_event_ids"][0]
    adapter.LOCAL_RECORD_URI = "external://local-platform-record/" + candidate.relative_to(REPO_ROOT).as_posix()
    adapter.DECISION_OUTCOME = "NEGATIVE_UNDER_CONTRACT"
    adapter.SCIENTIFIC_STATUS = "negative_under_contract"
    adapter.NEXT_ALLOWED_ACTIONS = []
    adapter.REOPEN_CONDITIONS = ["new mechanism and explicit prospective compute authority"]
    adapter.FINALIZED_AT, adapter.MIGRATED_AT = finalized_at, finalized_at[:10]
    adapter.PLATFORM, adapter.HARDWARE = "kaggle2", "Tesla_T4_16GB"
    adapter.rel = lambda name: (f"{arm_rel}/rml_plan/trajectory.json"
                                if name == "trajectory.json" else f"{arm_rel}/{name}")
    adapter.make_trace = lambda raw, checkpoint: accepted_native_trace(
        candidate / "canonical_trace.json", raw, checkpoint)
    extra_names = (*native_names, "execution_summary.json", "cost_allocation.json")
    adapter.EXTRA_ARTIFACTS = tuple((Path(name).stem, f"{arm_rel}/results/{name}")
                                    for name in extra_names) + (
        ("submission_receipt", f"{REL}/submission_receipt_v1.json"),)
    original_write = adapter.write

    def write_with_native_allocation(path, value):
        if Path(path) == results / "cost_records.json":
            event = value["costs"][0]
            event["attempt_id"] = "v1"
            event["measurement"]["device_hours"] = {
                "value": allocation["device_seconds"] / 3600, "status": "measured"}
            event["measurement"]["wall_hours"] = {
                "value": allocation["wall_seconds"] / 3600, "status": "measured"}
        original_write(path, value)

    adapter.write = write_with_native_allocation
    adapter.write(adapter.RAW_ACCEPTANCE, accepted)
    adapter.main()
    if file_digest(results / "canonical_trace.json") != file_digest(candidate / "canonical_trace.json"):
        raise ValueError("Native trace bytes changed during terminal preparation")
    return {"trajectory": f"{arm_rel}/rml_plan/trajectory.json",
            "terminal": f"{arm_rel}/results/terminal.json",
            "trace": f"{arm_rel}/results/canonical_trace.json"}
