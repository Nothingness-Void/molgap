"""Bind native training telemetry; preserve distinct failed/recovered audit runs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.k1_terminal_analysis import accepted_native_trace, analyze_saved_portability
from molgap.k1_audit_records import prepare_audit_records
from molgap.research_memory.trace import file_digest
from molgap.training_reproducibility import atomic_json

REL = "experiments/pcqm_k1_linear_attention_100k"
ROOT = REPO_ROOT / REL
MODE = "neural_atom_k1_linear_attention"
TRAIN_RECORD = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_linear_attention_s42_v1"
CANDIDATE = TRAIN_RECORD / "pcqm_k1_linear_attention" / MODE
AUDIT_RECORD = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_linear_attention_audit_s42_v1"
AUDIT = AUDIT_RECORD / "pcqm_k1_linear_attention_audit/post100k_audit"
REFERENCE = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference/neural_atom_k1_v4"
FINALIZED_AT = "2026-09-26T07:03:42+00:00"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_training():
    path = REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py"
    spec = importlib.util.spec_from_file_location("k1_terminal_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    accepted = load(AUDIT_RECORD / "controller_acceptance.json")
    frozen = load(ROOT / "rml_plan/trajectory.json")
    if not accepted["accepted"] or accepted["model_inference_executed_by_acceptance"]:
        raise RuntimeError("No-inference acceptance required")
    if accepted["training"]["candidates"][MODE]["gate"]["passed"]:
        raise RuntimeError("This terminal adapter freezes a non-promotion decision only")
    native_roles = load(CANDIDATE / "observed_role_history.json")
    for key in ("training_labels_read", "development_labels_read", "development_metric_computed", "development_selection_used"):
        if native_roles[key] is not True:
            raise RuntimeError("Missing native observed-role truth")
    if native_roles["run_id"] != frozen["actions"][0]["run_ids"][0]:
        raise RuntimeError("Native run differs from prospective identity")
    adapter.ROOT, adapter.ROOT_REL = ROOT, REL
    adapter.RESULTS = ROOT / "results"
    adapter.CANDIDATE_ROOT = CANDIDATE
    adapter.RECORD_ROOT = TRAIN_RECORD
    adapter.RAW_ACCEPTANCE = adapter.RESULTS / "training_acceptance.json"
    adapter.SOURCE_ACCEPTANCE_REL = "results/training_acceptance.json"
    adapter.TRAJECTORY_ID = frozen["trajectory_id"]
    adapter.RUN_ID = frozen["actions"][0]["run_ids"][0]
    adapter.MODE = MODE
    adapter.ACTION_ID = "A001"
    adapter.EVIDENCE_ID = "pcqm-k1-linear-attention-100k-s42"
    adapter.COST_EVENT_ID = frozen["actions"][0]["cost_event_ids"][0]
    adapter.LOCAL_RECORD_URI = "external://local-platform-record/" + CANDIDATE.relative_to(REPO_ROOT).as_posix()
    adapter.DECISION_OUTCOME = "NEGATIVE_UNDER_CONTRACT"
    adapter.SCIENTIFIC_STATUS = "NEGATIVE_UNDER_CONTRACT"
    adapter.NEXT_ALLOWED_ACTIONS = []
    adapter.REOPEN_CONDITIONS = ["new mechanism and explicit new prospective budget required"]
    adapter.FINALIZED_AT = FINALIZED_AT
    adapter.MIGRATED_AT = "2026-09-26"
    adapter.PLATFORM = "kaggle1"
    adapter.HARDWARE = "Tesla_T4_16GB"
    adapter.rel = lambda name: f"{REL}/rml_plan/trajectory.json" if name == "trajectory.json" else f"{REL}/{name}"
    adapter.make_trace = lambda raw, checkpoint: accepted_native_trace(CANDIDATE / "canonical_trace.json", raw, checkpoint)
    atomic_json(ROOT / "results/observed_role_history.json", native_roles)
    adapter.EXTRA_ARTIFACTS = (
        ("native_role_truth", f"{REL}/results/observed_role_history.json"),
        ("training_submission", f"{REL}/results/submission_receipt.json"),
        ("audit_failure", f"{REL}/results/audit_failure_v1.md"),
        ("saved_portability_analysis", f"{REL}/results/portability_analysis.json"),
    )
    adapter.write(adapter.RAW_ACCEPTANCE, {**accepted["training"],
        "source_commit": accepted["source_commit"],
        "source_archive_sha256": accepted["source_archive_sha256"]})
    adapter.main()
    if file_digest(ROOT / "results/canonical_trace.json") != file_digest(CANDIDATE / "canonical_trace.json"):
        raise RuntimeError("Native canonical trace bytes must be retained exactly, not reconstructed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("analysis", "training", "audit"))
    args = parser.parse_args()
    if args.stage == "analysis":
        result = analyze_saved_portability(REFERENCE, CANDIDATE, AUDIT, MODE)
        result["artifact_sha256"] = {Path(k).relative_to(REPO_ROOT).as_posix(): v
                                    for k, v in result["artifact_sha256"].items()}
        atomic_json(ROOT / "results/portability_analysis.json", result)
    elif args.stage == "training":
        prepare_training()
    else:
        prepare_audit_records(REPO_ROOT, REL, AUDIT_RECORD, finalized_at=FINALIZED_AT)


if __name__ == "__main__":
    main()
