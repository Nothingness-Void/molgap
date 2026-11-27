"""One-off old-style closure helper (user authorisation 2026-11-28); not merged into infrastructure.

Verbatim copy of molgap.research_memory.finalize.legacy_terminal (desktop source eecae4b30) with ONE change:
role and cost event ids carry an item tag, because five trajectories of this chain freeze the same run_id and
the stock ids (role-{run_id}-{role}-{access}, cost-{run_id}-{i}) repeat across them. The staging check rejects
repeated ids (finalize.py line 248). run_id itself is never altered.

Run with the desktop source first on PYTHONPATH, e.g.
  $env:PYTHONPATH = "<desktop worktree>\\src"; python legacy_terminal_tagged.py --repo-root <worktree> --tag j5 <legacy-terminal args>
then run the usual `terminal-pipeline`.
"""
import argparse
import json
from pathlib import Path

from molgap.research_memory import finalize as F
from molgap.research_memory.schemas import (CLOSURE_FACTS_FORMAT, TRAJECTORY_OUTCOMES, validate_closure_facts,
                                            validate_cost_event, validate_role_event)


def legacy_terminal_tagged(root, trajectory, action_id, run_id, decision_file, evidence_dirs, roles_file, out,
                           costs_file, tag):
    load_json_object, repo_local_path, atomic_write = F.load_json_object, F.repo_local_path, F.atomic_write
    json_bytes, file_digest, validate_observed_role_truth = F.json_bytes, F.file_digest, F.validate_observed_role_truth
    root = Path(root).resolve()
    traj = load_json_object(repo_local_path(root, trajectory))
    if not any(a["action_id"] == action_id and run_id in a["run_ids"] for a in traj["actions"]):
        raise ValueError("action_id/run_id is not frozen in the trajectory")
    dec, declared = load_json_object(Path(decision_file)), load_json_object(Path(roles_file))
    absent = [k for k in ("evidence_id", "scope", "v5_outcome", "finalized_at", "outcome", "facts", "next_allowed_actions",
              "reopen_conditions", "decision_ref") if k not in dec]
    if absent or not {"effect_meV", "ci_meV", "failure_mode", "search_terms", "source"} <= set(dec["facts"]):
        raise ValueError(f"decision file is missing fields: {absent or 'facts'}")
    if dec["outcome"] not in TRAJECTORY_OUTCOMES - {"ACTIVE"}:
        raise ValueError(f"unsupported terminal outcome: {dec['outcome']}")
    facts = validate_closure_facts({**dec["facts"], "format": CLOSURE_FACTS_FORMAT})
    effect, ci = facts["effect_meV"], facts["ci_meV"]
    if (effect is None) != (ci is None) or ci and not ci[0] <= effect <= ci[1]:
        raise ValueError("effect_meV must lie inside ci_meV (both set or both null)")
    role_use = declared.get("role_use", {})
    if missing := {"official_validation", "test_dev", "test_challenge"} - set(role_use):
        raise ValueError(f"roles declaration must state every protected role: {sorted(missing)}")
    files = [p for d in evidence_dirs for p in sorted(repo_local_path(root, d).rglob("*")) if p.is_file() and p.resolve().is_relative_to(root)]
    pointer = lambda p: Path(p).relative_to(root).as_posix()
    out_dir = repo_local_path(root, out)
    refs = [repo_local_path(root, x) for x in (dec["decision_ref"], facts["source"]) if x]
    if not files or not all(r.is_file() for r in refs) or Path(pointer(out_dir)).parts[0] == "research_memory" \
            or any((out_dir / n).exists() for n in ("acceptance.json", "terminal.json")):
        raise ValueError("need evidence files and existing decision_ref/facts.source; --out must be a new directory outside research_memory")
    acceptance_ref = pointer(out_dir / "acceptance.json")
    ident = {"trajectory_id": traj["trajectory_id"], "action_id": action_id, "run_id": run_id, "evidence_ref": acceptance_ref}
    # ONLY CHANGE: item tag in the two id formats.
    roles = [{**e, "schema": "molgap-role-event-v1", "role_event_id": f"role-{run_id}-{tag}-{e['role_name']}-{e['access_kind']}", **ident}
             for e in declared.get("events", [])]
    costs = [{**c, "schema": "molgap-cost-event-v1", "cost_event_id": f"cost-{run_id}-{tag}-{i}", **ident}
             for i, c in enumerate(load_json_object(Path(costs_file)).get("events", []) if costs_file else [])]
    for event in roles + costs:
        (validate_role_event if "role_event_id" in event else validate_cost_event)(event)
    validate_observed_role_truth([role_use], roles)
    decision = {k: dec[k] for k in ("decision_ref", "outcome", "next_allowed_actions", "reopen_conditions")} | {"facts": facts}
    acceptance = {"evidence_id": dec["evidence_id"], "run_id": run_id, "outcome": dec["v5_outcome"], "role_use": role_use,
                  "roles": roles, "costs": costs, "comparison_class": "CONTEXT_ONLY", "trajectory_decision": decision}
    atomic_write(out_dir / "acceptance.json", json_bytes(acceptance))
    hashes = {pointer(p): file_digest(p) for p in [*files, *refs, out_dir / "acceptance.json"]}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": traj["trajectory_id"], "run_id": run_id,
                "action_id": action_id, "finalized_at": dec["finalized_at"], "acceptance_ref": acceptance_ref,
                "artifact_hashes": hashes, "decision": decision, "costs": costs, "roles": roles,
                "evidence": {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
                             "evidence_id": dec["evidence_id"], "track": traj["track"], "scope": dec["scope"],
                             "legacy_contract": dec.get("legacy_contract", dec["scope"]), "outcome": dec["v5_outcome"],
                             "role_use": role_use, "authority": {"pointers": [dec["decision_ref"], acceptance_ref]},
                             "artifacts": [{"name": Path(k).name, "locator": k, "sha256": v,
                                            "availability": "locally_retained_hash_verified"} for k, v in hashes.items()],
                             "migration": {"migrated_at": dec["finalized_at"][:10], "training_executed": False,
                                           "inference_executed": False, "scientific_reinterpretation": False,
                                           "verification_scope": "Legacy retrieval closure; metadata only."}}}
    atomic_write(out_dir / "terminal.json", json_bytes(terminal))
    return {"acceptance": acceptance_ref, "terminal": pointer(out_dir / "terminal.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("--repo-root", "--tag", "--trajectory", "--action-id", "--run-id", "--decision-file", "--roles-file", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--evidence", action="append", required=True)
    p.add_argument("--costs-file")
    a = p.parse_args()
    print(json.dumps(legacy_terminal_tagged(a.repo_root, a.trajectory, a.action_id, a.run_id, a.decision_file, a.evidence,
                                            a.roles_file, a.out, a.costs_file, a.tag), indent=2))
