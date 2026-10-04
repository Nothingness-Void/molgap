"""Freeze CPU qualification and bind its executable; no platform submission."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from molgap.frozen_inference_release import check_frozen_inference_release
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

BASE = Path("experiments/pcqm_gptrans_ema_portability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if args.output.exists():
        raise ValueError("Never overwrite a CPU qualification package")
    report = check_frozen_inference_release(args.inputs, BASE/"run.py", BASE/"kernel-metadata.json")
    snapshot = copy.deepcopy(json.loads((BASE/"rml_plan/trajectory.json").read_text()))
    snapshot.pop("decision_state", None)
    tid = "TC-gptrans-ema-runtime-qualification-v1"
    run = "kaseichou/molgap-gptrans-runtime-qualification:v1"
    cost_id = "cost-"+tid
    snapshot.update(trajectory_id=tid, family_id="runtime-qualification",
        question="Can the unchanged frozen PyTorch workload import under isolated Python3.12 on Kaggle?")
    state = snapshot["state_at_start"]
    state.update(contract_refs=[(BASE/"environment_protocol.md").as_posix()],
        source_config_identity=report["release"]["identity"],
        prior_evidence_ids=["pcqm-gptrans-ema-portability-infrastructure-v1"],
        parent_trajectory_ids=["TC-gptrans-g1-ema-portability-frozen-s42"],
        role_snapshot_refs=[(BASE/"environment_protocol.md").as_posix()])
    snapshot["hypothesis"].update(hypothesis_id="H-"+tid,
        observed_deficiency="Kaggle system Python3.13 cannot install the frozen torch2.4.1 runtime",
        changed_mechanism="Interpreter isolation only; package import qualification, no models/data",
        supporting_evidence_ids=["pcqm-gptrans-ema-portability-infrastructure-v1"],
        cheapest_falsifier="CPU-only import probe", expected_native_cost_ref=cost_id,
        decision_changed_if_positive="Controller may release separately planned frozen GPU audit v2",
        decision_changed_if_negative="No GPU release; diagnose retained environment error",
        historical_unknowns=["CPU import qualification does not establish GPU numerical reproduction"])
    snapshot["actions"] = [dict(action_id="A001", type="NO_TRAIN_environment_qualification",
        run_ids=[run], attempt_ids=["v1"], source_commit=report["release"]["source_commit"],
        evidence_refs=[(BASE/"environment_protocol.md").as_posix()], cost_event_ids=[cost_id])]
    snapshot["decision"] = dict(outcome="ACTIVE", decision_ref=(BASE/"environment_protocol.md").as_posix(),
        next_allowed_actions=["CPU imports only"], reopen_conditions=["Terminal handoff to controller"])
    missing = dict(status="measurement_missing", value=None)
    cost = dict(schema="molgap-cost-event-v1", cost_event_id=cost_id, trajectory_id=tid, action_id="A001",
        run_id=run, attempt_id="v1", category="preflight", platform="kaggle2", hardware="CPU-only",
        evidence_ref=(BASE/"environment_protocol.md").as_posix(), measurement=dict(
            device_hours=dict(status="not_applicable", value=None),
            wall_hours=dict(status="estimated", value=0.5), cpu_hours=missing, queue_hours=missing))
    planned = plan(root, dict(trajectory=snapshot, costs=[cost], decision_state=dict(
        available_actions=["RUN_CPU_ENVIRONMENT_QUALIFICATION", "DEFER"],
        chosen_action="RUN_CPU_ENVIRONMENT_QUALIFICATION", policy_id="gptrans-ema-portability-audit",
        policy_version="v1", state_timestamp=datetime.now(timezone.utc).isoformat())), BASE/"environment_rml_plan")
    args.output.mkdir(parents=True)
    shutil.copyfile(BASE/"environment.py", args.output/"environment.py")
    shutil.copyfile(BASE/"environment-kernel-metadata.json", args.output/"kernel-metadata.json")
    atomic_json(BASE/"environment_release.json", dict(prospective=planned, source_release=report["release"],
        entry_sha256=sha256_file(args.output/"environment.py"),
        metadata_sha256=sha256_file(args.output/"kernel-metadata.json"),
        prospective_sha256=sha256_file(BASE/"environment_rml_plan/trajectory.json"),
        training_executed=False, model_inference_executed=False, graph_datasets_mounted=False,
        gpu_enabled=False, gpu_successor_released=False))
    print(json.dumps(planned))


if __name__ == "__main__":
    main()
