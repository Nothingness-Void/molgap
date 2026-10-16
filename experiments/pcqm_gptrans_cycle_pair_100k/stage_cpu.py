"""Thin prospective CPU planning and reproducible packaging, no submission."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import subprocess

from molgap.constants import REPO_ROOT
from molgap.research_memory.plan import plan
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_json
from molgap.v4_bundle import build_v4_source_bundle

BASE = "experiments/pcqm_gptrans_cycle_pair_100k"
RUN = "kaseichou/molgap-gptrans-cycle-inputs-v1"
TRAJECTORY = "TC-gptrans-cycle-input-preparation"
CONTEXT = "pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--recovery-attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    root, output = Path(REPO_ROOT), args.output.absolute()
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    trajectory = TRAJECTORY + "-recovery" if args.recovery else TRAJECTORY
    run = "kaseichou/molgap-gptrans-cycle-inputs-recovery-v1" if args.recovery else RUN
    source_id = "kaseichou/molgap-gptrans-cycle-source-recovery-v1" if args.recovery else "kaseichou/molgap-gptrans-cycle-source-v1"
    plan_ref = f"{BASE}/cpu/recovery_rml_plan" if args.recovery else f"{BASE}/cpu/rml_plan"
    if args.recovery and args.recovery_attempt == 2:
        trajectory += "-v2"
        run = run.replace("-v1", "-v2")
        source_id = source_id.replace("-v1", "-v2")
        plan_ref += "_v2"
    budget, roles, protocol = (f"{BASE}/{name}" for name in ("cpu_budget.json", "cpu_roles.json", "protocol.md"))
    if args.recovery:
        budget, roles = f"{BASE}/cpu_recovery_budget.json", f"{BASE}/cpu_recovery_roles.json"
    cost = "cost-" + trajectory
    proposal = {"trajectory": {
        "trajectory_id": trajectory, "record_mode": "prospective", "owner": "server", "track": "C",
        "family_id": "gptrans-cycle-input-preparation", "question": "Are common-chordless-cycle inputs exact on the immutable PCQM roles?",
        "hypothesis": {"hypothesis_id": "H-" + trajectory,
            "observed_deficiency": "No accepted common-cycle sidecar exists for the authorized ring-pair screen",
            "supporting_evidence_ids": [CONTEXT], "alternative_explanations": ["Enumeration, source-row or indexing error"],
            "changed_mechanism": "CPU-only exact cycle enumeration from unchanged accepted topology",
            "cheapest_falsifier": "Independent NetworkX rederivation for every row without model execution",
            "related_closed_family_ids": ["gptrans-shortest-path-execution-profile"], "expected_native_cost_ref": cost,
            "decision_changed_if_positive": "permit separate GPU release checks, not promotion",
            "decision_changed_if_negative": "block training and retain partial evidence", "historical_unknowns": []},
        "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint({
            "cap": 18, "roles": json.loads((root / roles).read_bytes())}), "contract_refs": [protocol],
            "reference_ids": [CONTEXT], "parent_trajectory_ids": [], "prior_trajectory_ids":
            ([TRAJECTORY, TRAJECTORY + "-recovery"] if args.recovery_attempt == 2 else [TRAJECTORY]) if args.recovery else [],
            "prior_evidence_ids": [CONTEXT],
            "role_snapshot_refs": [roles], "budget_snapshot_ref": budget},
        "actions": [{"action_id": "A001", "type": "cpu_input_preparation", "source_commit": commit,
            "run_ids": [run], "attempt_ids": ["v1"], "evidence_refs": [protocol], "cost_event_ids": [cost]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"decision_ref": protocol, "outcome": "ACTIVE",
            "next_allowed_actions": ["one CPU preparation and independent acceptance"], "reopen_conditions": ["terminal CPU evidence"]}},
        "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
            "available_actions": ["PREPARE_CYCLE_INPUTS", "DEFER"], "chosen_action": "PREPARE_CYCLE_INPUTS",
            "policy_id": "gptrans-cycle-input-preparation", "policy_version": "v1", "budget_snapshot_ref": budget,
            "role_snapshot_refs": [roles], "state_timestamp": datetime.now(timezone.utc).isoformat(), "source_commit": commit},
        "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cost, "trajectory_id": trajectory,
            "action_id": "A001", "run_id": run, "attempt_id": "v1", "category": "cache_build", "platform": "kaggle2",
            "hardware": "CPU-only", "evidence_ref": budget, "measurement": {
                "device_hours": {"status": "not_applicable", "value": None},
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "wall_hours": {"status": "estimated", "value": 7140/3600 if args.recovery else 2}, "queue_hours": {"status": "measurement_missing", "value": None}}}]}
    plan(root, proposal, plan_ref)
    sources = ["src/molgap/" + name for name in ("__init__.py", "constants.py", "training_reproducibility.py",
        "v4_runtime.py", "packed_graph_dataset.py", "experiment_preflight.py", "gptrans_cycle_inputs.py",
        "pcqm_fixed_records.py", "pcqm_wedge.py", "data/__init__.py",
        "data/packed_graph.py", "data/fixed_records.py", "compat/__init__.py",
        "compat/legacy_constants.py")]
    sources += [f"{BASE}/cpu_run.py", protocol]
    built = build_v4_source_bundle(repo_root=root, relative_paths=sources, output_dir=output / "source",
                                   source_commit=commit)
    shutil.copyfile(output / "source/source.tar.gz", output / "source/source_payload.bin")
    (output / "source/source.tar.gz").unlink()
    atomic_json(output / "source/dataset-metadata.json", {"title": "MolGap GPTrans Cycle Source V1",
        "id": source_id, "licenses": [{"name": "other"}]})
    kernel = output / "kernel"
    kernel.mkdir()
    entry = (root / BASE / "cpu_run.py").read_text(encoding="utf-8").replace("__SOURCE_ARCHIVE_SHA256__",
        built["archive_sha256"]).replace("__SOURCE_COMMIT__", commit)
    if args.recovery:
        entry = entry.replace("7200 - (time.monotonic() - began)",
                              f"{7080 if args.recovery_attempt == 2 else 7140} - (time.monotonic() - began)")
    (kernel / "run.py").write_text(entry, encoding="utf-8", newline="\n")
    atomic_json(kernel / "kernel-metadata.json", {"id": run, "title":
        f"MolGap GPTrans Cycle Inputs Recovery V{args.recovery_attempt}" if args.recovery else "MolGap GPTrans Cycle Inputs V1",
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_internet": True, "dataset_sources": [source_id,
        "kaseichou/pcqm4mv2-ogb-fixed-100k-v1", "kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []})
    atomic_json(output / "preparation.json", {**built, "run": run, "trajectory_id": trajectory,
        "rml_plan": plan_ref, "cpu_only": True, "gpu_released": False})
    print(json.dumps({"output": str(output), "archive_sha256": built["archive_sha256"], "run": run}))


if __name__ == "__main__":
    main()
