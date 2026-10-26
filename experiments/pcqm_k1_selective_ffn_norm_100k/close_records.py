"""Close retained local arms through shared owners; never grant strict replay."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys

for variable in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[variable] = "-1"
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molgap.common.io import publish_immutable_bytes
from molgap.common.paths import MachinePaths
from molgap.experiment_family_workflow import (
    RunContext, TargetIdentityBinding, build_verified_terminal_descriptor,
    inspect_terminal_output,
)
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_terminal import translate_terminal_descriptor
from molgap.experiment_terminal_staging import stage_verified_terminal_packages
from molgap.research_memory.paths import artifact_locator
from molgap.research_memory.terminal_wiring import close_terminal_multi_arm
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.training_reproducibility import atomic_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prior_attempts(config, paths, decision_ref):
    """Translate observed no-training dispositions, without a fabricated trace."""
    decision_ref = (HERE / "v1_closure_decision.md").relative_to(ROOT).as_posix()
    closure_inputs = []
    original = paths.artifact_root / "research" / config["experiment_id"]
    invocations = sorted((original / "bn_reference" / "invocations").glob("*.json"))
    if len(invocations) != 1:
        raise ValueError("Expected the retained original failed invocation")
    failed = read(invocations[0])
    if (failed["status"] != "failed" or failed["error"]["message"] !=
            "Executed MolGap source differs from package: src/molgap/local_training.py"):
        raise ValueError("Original attempt requires separate incident review")
    for item in config["arms"]:
        arm = item["arm_id"]
        owner = HERE / arm
        trajectory_path = owner / "prospective" / "trajectory.json"
        trajectory = read(trajectory_path)
        action, = trajectory["actions"]
        run_id, = action["run_ids"]
        outcome = "INFRASTRUCTURE_ONLY" if arm == "bn_reference" else "NO_TRAIN"
        evidence_id = "ev-terminal-" + trajectory["trajectory_id"]
        dimensions = {"execution_status": "preflight_failed" if arm == "bn_reference" else "not_launched",
            "artifact_status": "incident_retained", "comparison_status": "NOT_EVALUATED",
            "scientific_status": outcome, "transfer_status": "NOT_EVALUATED",
            "budget_decision": "NOT_EVALUATED", "full_handoff_status": "NOT_EVALUATED"}
        decision = {"outcome": outcome, "final": True, "decision_ref": decision_ref,
            "next_allowed_actions": [], "reopen_conditions": []}
        directory = owner / "terminal_closure"
        acceptance_path = directory / "acceptance.json"
        acceptance_ref = acceptance_path.relative_to(ROOT).as_posix()
        # No role events or successful qualification are invented for prepared arms.
        role_use = {role: "unavailable" for role in ("train", "development", "official_validation", "test_dev", "test_challenge")}
        # The failed registered preflight had only the frozen official-train
        # cache; the other two arms never launched. No protected input was used.
        role_use.update({role: "untouched" for role in ("official_validation", "test_dev", "test_challenge")})
        acceptance = {"run_id": run_id, "evidence_id": evidence_id,
            "outcome": dimensions, "trajectory_decision": decision, "role_use": role_use}
        publish_immutable_bytes(acceptance_path, json_bytes(acceptance))
        refs = [ROOT / decision_ref, acceptance_path, HERE / "failure_v1.md",
            owner / "experiment_spec.json"]
        if arm == "bn_reference":
            refs.extend([invocations[0], original / "bn_reference" / "launch_receipt.json"])
        bindings = {artifact_locator(ROOT, path): file_digest(path) for path in refs}
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "evidence_id": evidence_id, "track": "B", "scope": "local_preparation_incident",
            "legacy_contract": trajectory["state_at_start"]["contract_refs"][0],
            "outcome": dimensions, "authority": {"pointers": [decision_ref]},
            "artifacts": [{"name": "incident_disposition", "locator": decision_ref,
                "sha256": bindings[decision_ref], "availability": "durable_local_verified"}], "role_use": role_use,
            "metrics": {"source_ref": decision_ref, "unit": "eV", "missing_reason": "No formal training or scientific endpoint"},
            "migration": {"migrated_at": "terminal_closure", "verification_scope": "retained failure and explicit no-launch disposition",
                "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
        terminal_path = directory / "terminal.json"
        terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": trajectory["trajectory_id"],
            "run_id": run_id, "action_id": action["action_id"], "acceptance_ref": acceptance_ref,
            "decision": decision, "evidence": evidence, "costs": [], "roles": [],
            "role_use": role_use, "artifact_hashes": bindings,
            "finalized_at": read(terminal_path)["finalized_at"] if terminal_path.exists() else datetime.now(timezone.utc).isoformat()}
        publish_immutable_bytes(terminal_path, json_bytes(terminal))
        closure_inputs.append({"trajectory": trajectory_path.relative_to(ROOT).as_posix(),
            "terminal": terminal_path.relative_to(ROOT).as_posix(), "arm_identifier": arm})
    return closure_inputs


def main():
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Terminal closure is CPU-only")
    torch.set_num_threads(2)
    config = read(HERE / "input.json")
    paths = MachinePaths.from_file(config["machine_profile"]).outside_checkout(ROOT)
    outputs = paths.artifact_root / "research" / config["experiment_id"] / "attempt_2"
    decision_ref = (HERE / "closure_decision.md").relative_to(ROOT).as_posix()
    closures = prior_attempts(config, paths, decision_ref)
    staged_arms = {}
    for item in config["arms"]:
        arm = item["arm_id"]
        owner = HERE / "attempt_2" / arm
        spec = ExperimentSpec(read(owner / "experiment_spec.json"))
        logical_run = spec.to_dict()["logical_run_id"]
        prepared = paths.run_directory(logical_run, "v2") / "prepared"
        package = prepared / "package"
        frozen = read(prepared / "prepared_training.json")
        execution = outputs / arm
        folder = execution / arm
        context = RunContext.from_launch(spec, execution / "launch_receipt.json", package,
            expected_package_identity=frozen["expected_package_identity"], arm_id=arm)
        job, = frozen["jobs"]
        expected = read(ROOT / job["recipe"])["acceptance_requirements"]
        target = HERE / "terminal_checks" / (arm + "_target_encoding.json")
        binding = TargetIdentityBinding(ROOT, target.relative_to(ROOT).as_posix(), file_digest(target))
        output = {"context": context, "output_dir": folder, "expected": expected,
            "target_identity": binding, "package_dir": package}
        report = inspect_terminal_output(output)
        if report["status"] != "MECHANICALLY_VERIFIED":
            raise ValueError(str(report["blockers"]))
        native = read(HERE / "terminal_checks" / (arm + "_native.json"))
        if native["status"] != "VERIFIED":
            raise ValueError("Retained native verification is not complete")
        trajectory_path = owner / "prospective" / "trajectory.json"
        trajectory = read(trajectory_path)
        trajectory_id = trajectory["trajectory_id"]
        run_id = logical_run + ":" + arm + ":downstream"
        action, = trajectory["actions"]
        audit_path = owner / "observed_roles.json"
        audit_ref = audit_path.relative_to(ROOT).as_posix()
        arm_data = spec.to_dict()["arms"][0]["data"]
        members = {role["role"]: role for role in arm_data["roles"]}
        roles = []
        for role, accesses in (("train", ("training_membership", "labels_read")),
                ("development", ("prediction_input", "labels_read", "metric_computed", "selection_used"))):
            for access in accesses:
                roles.append({"schema": "molgap-role-event-v1",
                    "role_event_id": f"role-{trajectory_id}-{role}-{access}",
                    "trajectory_id": trajectory_id, "action_id": action["action_id"], "run_id": run_id,
                    "dataset_identity": arm_data["dataset"]["sha256"],
                    "row_manifest_hash": members[role]["row_order_sha256"],
                    "role_name": role, "access_kind": access,
                    "selection_used": access == "selection_used", "evidence_ref": audit_ref})
        audit_sources = [execution / "launch_receipt.json", folder / "output_manifest.json",
            folder / "training_contract.json", folder / "canonical_trace.json",
            HERE / "terminal_checks" / (arm + "_native.json"), target]
        audit = {"context": context.to_dict(), "roles": roles,
            "role_use": {"train": "used", "development": "selection_used",
                "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"},
            "scope": "Observed registered local execution and retained outputs; not a machine-wide file-access audit",
            "observations": "40 completed training/development passes and best-development Live selection; only fixed official-train members were supplied",
            "artifact_hashes": {artifact_locator(ROOT, source): file_digest(source) for source in audit_sources}}
        publish_immutable_bytes(audit_path, json_bytes(audit))
        where = {"trajectory_id": trajectory_id, "run_id": run_id,
            "trajectory": trajectory_path.relative_to(ROOT).as_posix(),
            "terminal": (owner / "terminal.json").relative_to(ROOT).as_posix(),
            "trace": artifact_locator(ROOT, folder / "canonical_trace.json")}
        resolved = {"spec": spec, "outputs": {arm: output}, "contexts": {arm: context},
            "locations": {arm: where}, "run_directory": execution,
            "options": {"package_dir": package, "expected_package_identity": frozen["expected_package_identity"]}}
        staged = stage_verified_terminal_packages(None, ROOT, logical_run, "v2", resolved=resolved,
            reports={arm: report}, outcome="CLOSED" if arm == "bn_reference" else "INCONCLUSIVE",
            decision_ref=decision_ref, role_audit={arm: audit_ref})
        if staged["status"] != "STAGED_CLOSURE_READY":
            raise ValueError(str(staged["blockers"]))
        descriptor = build_verified_terminal_descriptor(ROOT, spec, outputs={arm: output},
            locations=staged["locations"], _reports={arm: report})
        closures.extend(translate_terminal_descriptor(ROOT, spec, descriptor))
        staged_arms[arm] = staged["paths"][arm]
    results = close_terminal_multi_arm(ROOT, closures)
    # Global policy reports already have one owner in research_memory/derived.
    # Do not replicate that full catalog into each arm's closure summary.
    compact = [{key: value for key, value in row.items() if key != "candidate_policy_deltas"}
        for row in results]
    result = {"training_or_inference_executed": False, "arms": staged_arms,
        "finalizations": compact, "policy_report_ref": "research_memory/derived/policy_backtest.json",
        "strict_comparison": "NOT_QUALIFIED",
        "replay_ready": False,
        "irrecoverable_prelaunch_gap": "No frozen reference_ids, same_run_replay or trace-manifest plan; post-run edits cannot supply them"}
    atomic_json(HERE / "terminal_checks" / "rml_closure.json", result)
    print(json.dumps(result, indent=2))
    if any(row["pipeline_status"] != "COMPLETE" for row in results):
        raise RuntimeError("Canonical closure or derived update failed; see retained result")


if __name__ == "__main__":
    main()
