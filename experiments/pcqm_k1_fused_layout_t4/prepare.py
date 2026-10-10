"""Thin draft/binding entrypoint over the existing K1 pair preparation owner."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from molgap.k1_pair_preparation import build_inputs
from molgap.research_memory.policy import validate_policy
from molgap.research_memory.schemas import validate_trajectory
from molgap.training_reproducibility import canonical_fingerprint
from molgap.experiment_spec import ExperimentSpec

REL = "experiments/pcqm_k1_fused_layout_t4"
RUN = "molgap-k1-fused-layout-100k-s42-v1"


def prepare(root, initial, *, draft):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    folder = root / REL
    original = json.loads((root / "experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json").read_text())
    # Identical objective identity: only execution policy, not supervision, changes.
    records = build_inputs(root, source_commit=commit, initial_state=initial,
        state_timestamp=datetime.now(timezone.utc).isoformat(), relative_dir=REL,
        logical_run_id=RUN, reference_arm_id="reference", reference_addon=None,
        candidate_arm_id="fused_layout", candidate_addon="k1_fused_layout",
        candidate_objective={"base_loss": "normalized-gap-l1", "forward_passes": 1,
            "optimizer_updates_per_batch": 1, "consistency_penalty": False,
            "batchnorm_updates_per_batch": 1},
        allocation_wall_limit_seconds=None, family_id="k1-native-fused-layout",
        policy_id="k1-native-fused-layout-100k", account="nothingnessvoid",
        source_dataset="nothingnessvoid/k1-fused-layout-100k-s42-source-v1",
        dataset="nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1", title=RUN,
        question="Does fused AdamW plus CPU layout reduce native T4 cost without degrading complete100K K1 quality?",
        changed_mechanism="Single-forward K1 with fused AdamW plus CPU layout; no loader/architecture/objective changes.",
        supporting_evidence_ids=["pcqm-k1-local-speed-10ep-fused_layout-20261011",
                                 "pcqm-k1-colab-execution-profile-20261007"],
        hypothesis_overrides={
            "observed_deficiency": "RTX5060 TRAIN-only10ep cannot qualify native-default T4 speed or40ep quality.",
            "alternative_explanations": ["Native foreach already closes the speed gap", "Fused rounding degrades quality", "Development/IO/shared CPU hides step gain"],
            "cheapest_falsifier": "Native all-arm qualification then one authorized fixed100K/40ep pair, no protected roles",
            "related_closed_family_ids": ["k1-local-fused-layout-speed"],
            "decision_changed_if_positive": "Nominate runtime for separately reviewed deployment; no automatic500K/full/promotion",
            "decision_changed_if_negative": "Close with native cost/quality attribution; no automatic retry"})
    for arm in records["experiment_spec.json"]["arms"]:
        arm["training"]["objective"] = original["training"]["objective"].copy()
    if draft:
        for arm in ("reference", "fused_layout"):
            write(folder / f"training_recipe_{arm}.json", records[f"training_recipe_{arm}.json"])
        print("Recipes drafted; commit source/protocol/recipes before final binding.")
        return
    authority = REL + "/user_release.json"
    policy = records["policy.json"]
    policy.update(status="approved", action_rule={"field": "user_release_validated", "operator": "eq",
                                                 "threshold": 1, "action": "TRAIN_PAIR_100K"})
    policy["required_observable_fields"] = ["user_release_validated"]
    policy["approval"] = {"approved_by": "user", "approved_at": "2026-10-11",
                          "authority_ref": authority}
    validate_policy(policy)
    for arm in ("reference", "fused_layout"):
        plan = records[f"training_plan_{arm}.json"]
        trajectory = plan["trajectory"]
        declaration = next(a for a in records["experiment_spec.json"]["arms"] if a["arm_id"] == arm)
        trajectory["state_at_start"]["source_config_identity"] = canonical_fingerprint(declaration)
        trajectory["state_at_start"]["contract_refs"].append(authority)
        trajectory["actions"][0].update(type="user_authorized_native_t4_pair_100k",
                                      evidence_refs=[REL + "/protocol.md", authority])
        trajectory["decision"].update(decision_ref=REL + "/protocol.md", next_allowed_actions=["A001"])
        plan["decision_state"].update(available_actions=["NO_TRAIN", "TRAIN_PAIR_100K"],
                                      chosen_action="TRAIN_PAIR_100K")
        validate_trajectory(trajectory)
    for binding in records["experiment_spec.json"]["prospective"]["arms"]:
        binding["plan_spec_sha256"] = canonical_fingerprint(records[f"training_plan_{binding['arm_id']}.json"])
    spec = ExperimentSpec(records["experiment_spec.json"])
    for name in ("family_acceptance_plan.json", "workflow_plan.json"):
        records[name]["spec_identity"] = spec.identity
    records["preparation_report.json"].update(status="USER_RELEASE_BOUND_UNPUBLISHED",
        user_release_ref=authority, training_authorized=True, prospective_published=False,
        blockers=["Exact package/release and real-input CPU checks pending", "Native all-arm T4 qualification pending"])
    for name, value in records.items():
        write(folder / name, value)
    # Policy registration remains at the RML owner's existing search location.
    write(root / "research_memory/policies/k1-native-fused-layout-100k-v1.json", policy)
    print(json.dumps({"spec_identity": spec.identity, "source_commit": commit, "submitted": False}))


def write(path, value):
    # Canonical bytes match the package recipe hash and LF-normalized snapshot.
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    prepare(args.repo_root.resolve(), args.initial_state.resolve(), draft=args.draft)
