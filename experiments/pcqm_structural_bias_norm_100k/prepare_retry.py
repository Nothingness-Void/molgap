"""Declare the user-authorized bootstrap retry; reuse unchanged training inputs."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.common.declarations import ExecutionProfile, ExecutionQuestion, build_execution_proposal
from molgap.experiment_spec import ExperimentSpec
from molgap.research_memory.trace import atomic_write
from molgap.training_reproducibility import sha256_file


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def generate():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    timestamp = datetime.now(timezone.utc).isoformat()

    def write(path, record):
        if path.exists():
            raise ValueError("Retry declarations already exist; reconcile rather than overwrite")
        atomic_write(path, (json.dumps(record, indent=2) + "\n").encode("utf-8"))

    for family in ("k1", "gptrans"):
        original = HERE / family
        destination = original / "attempt_002"
        relative = destination.relative_to(ROOT).as_posix()
        authority = relative + "/authority.json"
        decision = relative + "/decision.md"
        old = json.loads((original / "experiment_spec.json").read_bytes())
        workflow = json.loads((original / "workflow_plan.json").read_bytes())
        if not (destination / "authority.json").is_file():
            raise ValueError("Explicit retry authority must be committed first")
        run = old["logical_run_id"] + "-r2"
        declaration = deepcopy(old)
        declaration.update(experiment_id=run, logical_run_id=run)
        bindings = []
        for arm in declaration["arms"]:
            name = arm["arm_id"]
            previous = json.loads((original / ("plan_" + name + ".json")).read_bytes())
            trajectory, hypothesis = previous["trajectory"], previous["trajectory"]["hypothesis"]
            question = ExecutionQuestion(trajectory["family_id"], "RUN_PAIRED_100K", trajectory["question"],
                hypothesis["observed_deficiency"], hypothesis["changed_mechanism"],
                tuple(hypothesis["supporting_evidence_ids"]), tuple(hypothesis["alternative_explanations"]),
                tuple(hypothesis["related_closed_family_ids"]), hypothesis["decision_changed_if_positive"],
                hypothesis["decision_changed_if_negative"], prior_trajectories=(trajectory["trajectory_id"],))
            recipe = next(job["recipe"] for job in workflow["arms"] if job["arm_id"] == name)
            tid = trajectory["trajectory_id"] + "-r2"
            plan = build_execution_proposal(question=question, arm=arm, trajectory_id=tid,
                source_commit=commit, run=run, version="v1", contract_ref=recipe,
                contract_refs=[*trajectory["state_at_start"]["contract_refs"],
                    HERE.relative_to(ROOT).as_posix() + "/failure_v1.md"],
                authority_ref=authority, role_ref=trajectory["state_at_start"]["role_snapshot_refs"][0],
                decision_ref=decision, timestamp=timestamp,
                profile=ExecutionProfile("desktop", "B", "kaggle", "Tesla T4",
                    "Repaired bootstrap real-guard regression, then native all-arm preflight before training.",
                    None, None), action_type="training")
            plan["decision_state"]["policy_version"] = "1"
            plan["decision_state"]["known_trajectory_ids"] = [trajectory["trajectory_id"]]
            plan["trajectory"]["actions"][0]["attempt_ids"] = [run + "-v1"]
            plan["costs"][0]["attempt_id"] = run + "-v1"
            plan_path = destination / ("plan_" + name + ".json")
            write(plan_path, plan)
            bindings.append({"arm_id": name, "trajectory_id": tid,
                "plan_spec_ref": plan_path.relative_to(ROOT).as_posix(),
                "plan_spec_sha256": sha256_file(plan_path), "output": relative + "/rml/" + name})
        declaration["prospective"]["arms"] = bindings
        spec = ExperimentSpec(declaration)
        write(destination / "experiment_spec.json", spec.to_dict())
        acceptance = json.loads((original / "acceptance_plan.json").read_bytes())
        acceptance["spec_identity"] = spec.identity
        write(destination / "acceptance_plan.json", acceptance)
        workflow["spec_identity"] = spec.identity
        workflow["acceptance_plan"] = relative + "/acceptance_plan.json"
        workflow["source_records"] = [authority if path.endswith("/authority.json") else
            decision if path.endswith("/decision.md") else path for path in workflow["source_records"]]
        workflow["source_records"].append(HERE.relative_to(ROOT).as_posix() + "/failure_v1.md")
        platform = workflow["kaggle"]
        old_source = platform["source_dataset"]
        platform.update(kernel=platform["kernel"] + "-r2", title=platform["title"] + " R2",
                        source_dataset=old_source + "-r2")
        platform["datasets"] = [platform["source_dataset"] if path == old_source else path
                                for path in platform["datasets"]]
        write(destination / "workflow_plan.json", workflow)
        editable = json.loads((original / "input.json").read_bytes())
        editable.update(spec_ref=relative + "/experiment_spec.json",
            workflow_ref=relative + "/workflow_plan.json", authority_ref=authority)
        write(destination / "input.json", editable)
        print(json.dumps({"family": family, "spec_identity": spec.identity, "source_commit": commit,
                          "status": "DRAFT_RETRY_ONLY", "kernel": platform["kernel"]}))


if __name__ == "__main__":
    generate()
