"""Publish accepted representation metadata; never load a model or graph."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from molgap.constants import REPO_ROOT

from molgap.research_memory.terminal_wiring import close_terminal_arm
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes


def main() -> None:
    root = REPO_ROOT
    directory = Path(__file__).resolve().parent
    results = directory / "results"
    relative = lambda p: p.relative_to(root).as_posix()
    load = lambda p: json.loads(p.read_bytes())
    save = lambda p, obj: atomic_write(p, json_bytes(obj))
    metadata = load(results / "terminal_metadata.json")
    evidence = copy.deepcopy(load(results / "evidence_envelope.json"))
    plan = directory / "rml_plan/trajectory.json"
    assert file_digest(plan) == metadata["original_prospective_sha256"]
    raw = root / "platforms/_records/scnet/k1_representation_audit/attempt-123315282/raw/started.json"
    atomic_write(results / "started.json", raw.read_bytes())
    binding = {
        "format": "molgap-no-train-postlaunch-binding-v1",
        "trajectory_id": metadata["trajectory_id"], "action_id": metadata["action_id"],
        "run_id": metadata["run_id"], "original_trajectory_sha256": file_digest(plan),
    }
    hashes = dict(metadata["artifact_hashes"])
    for name, path in (("submission", directory / "submission.json"),
                       ("scheduler", results / "scheduler.json"), ("started", results / "started.json")):
        pointer = relative(path)
        binding[name + "_ref"] = pointer
        binding[name + "_sha256"] = file_digest(path)
        hashes[pointer] = file_digest(path)
    save(results / "postlaunch_binding.json", binding)
    hashes[relative(results / "postlaunch_binding.json")] = file_digest(results / "postlaunch_binding.json")
    decision = {
        "outcome": "NO_TRAIN", "final": True,
        "decision_ref": relative(results / "terminal_decision.md"),
        "next_allowed_actions": ["CLOSE"], "reopen_conditions": [],
    }
    hashes[decision["decision_ref"]] = file_digest(root / decision["decision_ref"])
    evidence["authority"]["pointers"].append(decision["decision_ref"])
    for pointer in evidence["authority"]["pointers"]:
        hashes[pointer] = file_digest(root / pointer)
    evidence["migration"]["verification_scope"] = (
        "Saved JSON/SHA acceptance and receipt-bound diagnostic terminal publication; "
        "no training trace, no causal claim, no local inference"
    )
    for path in (results / "started.json", results / "postlaunch_binding.json", results / "terminal_metadata.json"):
        pointer = relative(path)
        hashes[pointer] = file_digest(path)
        evidence["artifacts"].append({"name": path.name, "locator": pointer,
                                      "sha256": hashes[pointer], "availability": "repository_retained"})
    acceptance = {
        "evidence_id": evidence["evidence_id"], "run_id": metadata["run_id"],
        "outcome": evidence["outcome"], "trajectory_decision": decision,
        "role_use": metadata["role_use"], "accepted": True,
        "scope": "NO_TRAIN context diagnostic; original empty-run plan retained byte-for-byte",
    }
    save(results / "terminal_acceptance.json", acceptance)
    acceptance_ref = relative(results / "terminal_acceptance.json")
    hashes[acceptance_ref] = file_digest(root / acceptance_ref)
    terminal_path = results / "terminal.json"
    finalized_at = (load(terminal_path)["finalized_at"] if terminal_path.exists()
                    else datetime.now(timezone.utc).isoformat())
    package = {
        "format": "molgap-rml-terminal-package-v1", "trajectory_id": metadata["trajectory_id"],
        "action_id": metadata["action_id"], "run_id": metadata["run_id"],
        "finalized_at": finalized_at, "evidence": evidence,
        "decision": decision, "acceptance_ref": acceptance_ref, "artifact_hashes": hashes,
        "costs": metadata["costs"], "roles": metadata["roles"], "role_use": metadata["role_use"],
        "postlaunch_run_binding_ref": relative(results / "postlaunch_binding.json"),
    }
    save(results / "terminal.json", package)
    print(json.dumps(close_terminal_arm(repo_root=root, trajectory=plan,
                                       terminal=results / "terminal.json"), indent=2))


if __name__ == "__main__":
    main()
