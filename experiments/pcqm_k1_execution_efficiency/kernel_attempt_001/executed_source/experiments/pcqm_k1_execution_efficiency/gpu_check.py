"""Bind a fresh GPU sub-attempt to existing execution and RML owners."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.research_memory.finalize import finalize
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "gpu_attempt_001"
REL = ATTEMPT.relative_to(ROOT).as_posix()
TID = "TB-k1-execution-efficiency-gpu-20261010"
RUN = "k1-execution-efficiency-rtx5060-20261010"
EID = "pcqm-k1-execution-efficiency-gpu-20261010"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (ATTEMPT / "rml").exists():
        raise FileExistsError("Prospective exists; reconcile, do not overwrite")
    spec, inputs = deepcopy(read(HERE / "plan_input.json")), deepcopy(read(HERE / "inputs.json"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [*inputs["files"], Path(__file__).relative_to(ROOT).as_posix(),
             f"{REL}/README.md", f"{REL}/decision.md"]
    inputs.update(source_commit=head, worker_wall_limit_seconds=600,
                  files={name: sha256_file(ROOT / name) for name in files})
    for entry in inputs["bindings"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Retained training input changed")
    atomic_json(ATTEMPT / "inputs.json", inputs)
    atomic_json(ATTEMPT / "role_plan.json", read(HERE / "role_plan.json"))
    cost_id = "cost-k1-execution-efficiency-gpu-estimate"
    prior = ["pcqm-k1-execution-efficiency-20261010", "pcqm-k1-execution-checkpoint-recovery-20261010"]
    trajectory = spec["trajectory"]
    trajectory.update(trajectory_id=TID, question="Do CPU-qualified K1 execution changes preserve exact CUDA state and save local step time?",
                      comparison_readiness_ref=f"{REL}/README.md")
    trajectory["hypothesis"].update(hypothesis_id="H-k1-execution-efficiency-gpu-20261010",
        supporting_evidence_ids=prior, expected_native_cost_ref=cost_id,
        observed_deficiency="CPU layout is exact but CPU paired timings have mixed signs; CUDA qualification is absent.",
        alternative_explanations=["Implicit finite/loss/RNG synchronization remains", "GPU reductions alter low bits", "Local operators dominate"],
        decision_changed_if_positive="Nominate only exact, consistently faster local execution paths; native T4 remains separate",
        decision_changed_if_negative="Keep defaults and attribute failed exactness or absent speed benefit")
    state = trajectory["state_at_start"]
    state.update(source_commit=head, source_config_identity=sha256_file(ATTEMPT / "inputs.json"),
        contract_refs=[f"{REL}/README.md", f"{REL}/inputs.json"],
        parent_trajectory_ids=["TB-k1-execution-efficiency-20261010"], prior_evidence_ids=prior,
        role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/README.md")
    trajectory["actions"] = [dict(action_id="A001", type="execution_profile", run_ids=[RUN],
        attempt_ids=["gpu-attempt-001"], source_commit=head, evidence_refs=[f"{REL}/inputs.json"], cost_event_ids=[cost_id])]
    trajectory["decision"].update(decision_ref=f"{REL}/decision.md", next_allowed_actions=["Bounded idle-only RTX5060 execution diagnostic"])
    spec["decision_state"].update(source_commit=head, state_timestamp=datetime.now(timezone.utc).isoformat(),
        budget_snapshot_ref=f"{REL}/README.md", role_snapshot_refs=[f"{REL}/role_plan.json"])
    cost = spec["costs"][0]
    cost.update(cost_event_id=cost_id, trajectory_id=TID, run_id=RUN, attempt_id="gpu-attempt-001", evidence_ref=f"{REL}/README.md")
    for unit in ("wall_hours", "device_hours"):
        cost["measurement"][unit] = dict(value=600 / 3600, status="estimated")
    atomic_json(ATTEMPT / "plan_input.json", spec)
    atomic_json(ATTEMPT / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print(TID + " prospective frozen")


def execute():
    if (ATTEMPT / "rml/rml_finalized").exists():
        raise ValueError("Attempt already finalized")
    # Rebind only artifact custody; reuse all loading, numerical, timing and
    # checkpoint logic in the corrected wrapper, without copying its loop.
    import run as owner
    owner.HERE = ATTEMPT
    owner.run("cuda", ATTEMPT / "results")


def close():
    result_dir = ATTEMPT / "results"
    inputs, runtime = read(ATTEMPT / "inputs.json"), read(result_dir / "runtime.json")
    result, completion = read(result_dir / "result.json"), read(result_dir / "completion.json")
    observation, resume = read(result_dir / "cost_observation.json"), read(result_dir / "checkpoint_resume.json")
    if (completion["status"] != "complete" or result["status"] != "complete"
            or runtime["execution_device"] != "cuda" or result["contract_promotion"]
            or runtime["source_commit"] != inputs["source_commit"]
            or runtime["input_sha256"] != sha256_file(ATTEMPT / "inputs.json")
            or runtime["prospective_sha256"] != sha256_file(ATTEMPT / "rml/trajectory.json")
            or runtime["observed_train_source_idx"] != inputs["source_idx"]
            or runtime["development_rows"] or not all(x["idle"] for x in runtime["idle_observations"])):
        raise ValueError("GPU result/input/runtime/role/completion identity differs")
    for name, digest in completion["artifacts"].items():
        if sha256_file(result_dir / name) != digest:
            raise ValueError("GPU output bytes differ")
    for name, digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Frozen GPU source changed")
    for name in ("terminal_decision.md", "attribution.md"):
        if not (ATTEMPT / name).exists():
            raise ValueError("Parent interpretation required before closure")
    acceptance = deepcopy(read(HERE / "acceptance.json"))
    outcome = acceptance["outcome"]
    outcome.update(execution_status="complete_gpu_scratch_diagnostic", comparison_status="matched_local_cuda_scratch_only",
                   budget_decision="bounded_local_gpu_diagnostic_complete")
    acceptance.update(evidence_id=EID, run_id=RUN, outcome=outcome, gpu_qualification="local_scratch_only",
        checkpoint_resume_exact=resume["comparison"]["exact"] if resume["comparison"] is not None else None,
        case_qualification={k: dict(exact=v.get("equivalence", {}).get("exact"),
            execution_only_admissible=v["execution_only_admissible"], status=v["status"]) for k,v in result["cases"].items()},
        execution_scope="Discarded CUDA scratch updates, not scientific fitting or native T4 qualification",
        cost_scope=observation["cost_scope"], checks=dict(prospective=True, artifact_hashes=True, train_only=True, complete_worker=True))
    acceptance.pop("infrastructure_failure_ref", None)
    decision = acceptance["trajectory_decision"]
    decision.update(decision_ref=f"{REL}/terminal_decision.md", reopen_conditions=["Separately prospective native runtime qualification"])
    cost = acceptance["costs"][0]
    cost.update(cost_event_id="cost-k1-execution-efficiency-gpu-observed", trajectory_id=TID, run_id=RUN,
                attempt_id="gpu-attempt-001", hardware="RTX5060; local CUDA scratch diagnostic", evidence_ref=f"{REL}/acceptance.json")
    for unit, key in (("wall_hours", "worker_wall_seconds"), ("cpu_hours", "process_cpu_seconds")):
        cost["measurement"][unit] = dict(value=observation[key] / 3600, status="measured")
    cost["measurement"]["device_hours"] = dict(value=None, status="measurement_missing")
    for role in acceptance["roles"]:
        role.update(role_event_id=role["role_event_id"].replace("execution-efficiency-", "execution-efficiency-gpu-"),
                    trajectory_id=TID, run_id=RUN, evidence_ref=f"{REL}/acceptance.json")
    atomic_json(ATTEMPT / "acceptance.json", acceptance)
    files = [p for p in ATTEMPT.rglob("*") if p.is_file() and "rml" not in p.relative_to(ATTEMPT).parts
             and p.name not in ("terminal.json", "closure_receipt.json")]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}
    evidence = deepcopy(read(HERE / "rml/rml_finalized/v5_evidence.json"))
    evidence.update(evidence_id=EID, scope="desktop_k1_gpu_execution_efficiency", outcome=outcome,
        authority=dict(pointers=[f"{REL}/{name}" for name in ("README.md", "terminal_decision.md", "attribution.md", "acceptance.json")]),
        artifacts=[dict(name=p.removeprefix(REL + "/"), locator=p, sha256=h, availability="locally_retained_hash_verified")
                   for p,h in hashes.items()])
    evidence["observed_execution"]["execution_ref"] = f"{REL}/results/result.json"
    evidence["migration"].update(verification_scope="Metadata closure of separately executed prospective GPU scratch diagnostic")
    atomic_json(ATTEMPT / "terminal.json", dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID,
        run_id=RUN, action_id="A001", finalized_at=datetime.now(timezone.utc).isoformat(),
        acceptance_ref=f"{REL}/acceptance.json", artifact_hashes=hashes, evidence=evidence, decision=decision,
        costs=acceptance["costs"], roles=acceptance["roles"]))
    atomic_json(ATTEMPT / "closure_receipt.json", finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json"))
    print(EID + " finalized NO_TRAIN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "execute", "close"))
    {"prepare": prepare, "execute": execute, "close": close}[parser.parse_args().operation]()
