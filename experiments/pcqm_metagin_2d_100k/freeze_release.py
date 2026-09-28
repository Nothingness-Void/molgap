"""Freeze a real K1-reference prelaunch and one prospective MetaGIN RML plan."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.constants import REPO_ROOT
from molgap.pcqm_metagin_screen import ARCHITECTURE, MODEL_ID, TRAJECTORY_ID
from molgap.research_memory.plan import plan
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import write_server_comparison_prelaunch


REL = "experiments/pcqm_metagin_2d_100k"
ROOT = REPO_ROOT / REL
TEMPLATE = REPO_ROOT / "experiments/pcqm_k1_sparse_triplet_100k"
REFERENCE = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
RUN_ID = "kaseichou/molgap-metagin-2d-s42:v1"
POLICY_ID = "metagin-2d-independent-backbone-screen"
COST_ID = f"cost-{TRAJECTORY_ID}-training"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    if path.exists():
        if load(path) != value:
            raise RuntimeError(f"Partial release evidence changed: {path}")
        return
    atomic_write(path, json_bytes(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--attempt", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    target = ROOT if args.attempt == "v1" else ROOT / "attempt_v2"
    target_rel = REL if args.attempt == "v1" else f"{REL}/attempt_v2"
    run_id = RUN_ID if args.attempt == "v1" else "kaseichou/molgap-metagin-2d-s42:v2"
    trajectory_id = TRAJECTORY_ID if args.attempt == "v1" else f"{TRAJECTORY_ID}-v2"
    cost_id = COST_ID if args.attempt == "v1" else f"cost-{trajectory_id}-training"
    package = args.package.resolve()
    commit = (package / "SOURCE_COMMIT.txt").read_text().strip()
    digest = (package / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    if len(commit) != 40 or file_digest(package / "source_payload.bin") != digest:
        raise RuntimeError("Frozen source package bytes changed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT, check=True,
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", commit, "HEAD", "--", "src",
         f"{REL}/package_source.py", f"{REL}/kaggle_cpu", f"{REL}/kaggle_gpu"],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if changed:
        raise RuntimeError("Executable package source changed since frozen commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src"],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Commit architecture and runner before freezing")
    if (target / "rml_plan").exists():
        raise FileExistsError("Frozen RML plan must not be overwritten")
    bundle = load(REFERENCE)
    role = load(TEMPLATE / "role_plan.json")
    trace = load(TEMPLATE / "trace_plan.json")
    config_id = canonical_fingerprint(ARCHITECTURE)
    now = datetime.now(timezone.utc).isoformat()
    save(target / "source_config.json", {
        "format": f"molgap-metagin-2d-source-config-{args.attempt}",
        "source_commit": commit, "source_archive_sha256": digest,
        "candidate_id": MODEL_ID, "architecture_config_identity": config_id,
        "model_parameters_expected": 5_268_481,
        "training_contract_ref": f"{REL}/training_contract.json",
        "training_contract_sha256": file_digest(ROOT / "training_contract.json"),
        "reference_bundle_ref": str(REFERENCE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cpu_run_id": "kaseichou/molgap-metagin-2d-sidecar-v1:v1",
        "gpu_run_id": run_id,
        "sidecar_required_before_gpu": True,
    })
    save(target / "role_plan.json", role)
    save(target / "trace_plan.json", trace)
    save(target / "budget_snapshot.json", {
        "authorization": "user-requested-one-distinct-model-family-seed42-screen",
        "platform": "kaggle2", "accelerator_allocation": (
            "one-requested-P100" if args.attempt == "v1"
            else "P100-request-with-observed-T4x2-allocation-cost-counted"
        ),
        "available_device_hours": None,
        "gpu_estimated_device_hours": (
            6.0 if args.attempt == "v1" else 3.669210724325182
        ),
        "cpu_sidecar_estimated_wall_hours": 1.0,
        "max_worker_wall_hours": 6.0,
        "preflight_cost_gate": True,
        "measured_profile_ref": (
            None if args.attempt == "v1"
            else f"{REL}/results/profile_v2_acceptance.json"
        ),
        "automatic_second_seed_or_scale_authorized": False,
    })
    save(REPO_ROOT / f"research_memory/policies/{POLICY_ID}.v1.json", {
        "schema": "molgap-policy-v1", "policy_id": POLICY_ID, "version": "v1",
        "policy_type": "research_action", "status": "candidate",
        "comparability_selector": {
            "scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-40epochs",
        },
        "required_observable_fields": ["independent_multihop_backbone_frozen"],
        "created_from_source_digest": file_digest(ROOT / "protocol.md"),
        "approval": {"authority_ref": f"{REL}/protocol.md",
                     "activation": "controller-only-bounded-user-authorization"},
        "cost_model": {"kind": "measured_only", "assumptions": [
            "no cross-hardware scalar conversion", "CPU sidecar cost separately retained",
        ]},
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "action_rule": {
            "field": "independent_multihop_backbone_frozen", "operator": "eq",
            "threshold": True, "action": "RUN_BOUNDED_METAGIN2D_SCREEN",
        },
        "borderline_action": "DEFER",
    })
    identity = dict(bundle["comparison_identity"], architecture_config_identity=config_id)
    prelaunch = assess_comparison_prelaunch(
        candidate_id=MODEL_ID, reference_id=bundle["reference_id"],
        reference_bundle=bundle,
        candidate_plan={"comparison_identity": identity,
                        "source_config_status": "frozen",
                        "source_commit_or_archive": commit},
        experiment_purpose="architecture_comparison",
        intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
        role_applicability_plan={key: role[key] for key in (
            "training_membership", "prediction_input", "labels_read",
            "metric_computed", "selection_used", "external_submission",
        )},
        trace_plan={key: trace[key] for key in (
            "optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate",
            "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity",
        )},
        runtime_qualification_plan={
            "status": "declared", "runtime_certificate_required": True,
            "qualification_scope": "fp32-no-tf32-bs128-optimizer-inclusive",
        },
    )
    write_server_comparison_prelaunch(
        target / "comparison_readiness_prelaunch.json",
        comparison_prelaunch=prelaunch,
        experiment_purpose="architecture_comparison", reference_bundle=bundle,
        repo_root=REPO_ROOT, reference_bundle_path=REFERENCE,
    )
    if prelaunch["prelaunch_ready"] is not True:
        raise RuntimeError(f"MetaGIN prelaunch not ready: {prelaunch['blocker_codes']}")
    trajectory = copy.deepcopy(load(TEMPLATE / "trajectory.json"))
    trajectory.update(
        trajectory_id=trajectory_id, family_id="metagin-2d-multihop",
        question="Does an independent pure-2D multihop virtual-state backbone outperform immutable K1-v4 under matched PCQM-100K V5 exposure?",
        comparison_readiness_ref=f"{target_rel}/comparison_readiness_prelaunch.json",
        comparison_class="NO_COMPARISON", comparison_blockers=[],
        reference_bundle_id=bundle["reference_bundle_id"],
    )
    trajectory["hypothesis"] = {
        "hypothesis_id": f"H-{trajectory_id}",
        "observed_deficiency": "K1 local-capacity additions regress and its full-scale advantages erode; a distinct 2D multi-hop information flow remains untested.",
        "supporting_evidence_ids": [bundle["reference_id"]],
        "alternative_explanations": [
            "path counts duplicate K1 local depth", "virtual-state bandwidth is too weak",
            "larger independent encoder overfits 100K", "reused-development selection optimism",
        ],
        "related_closed_family_ids": [
            "k1-chemistry-separated-local", "shortest-path-attention", "k1-sparse-triplet",
        ],
        "changed_mechanism": "replace K1 EdgeState-plus-slot backbone with four sequential bond/2-hop/3-hop gated MetaFormer blocks and persistent virtual molecular state",
        "cheapest_falsifier": "one accepted CPU topology sidecar then one seed42 fixed100K screen against immutable K1-v4",
        "decision_changed_if_positive": "nominate for separately authorized frozen500K NO_TRAIN transfer audit",
        "decision_changed_if_negative": "close this exact adaptation without a seed or scale successor",
        "expected_native_cost_ref": cost_id,
        "historical_unknowns": ["training stochasticity is not inferred from a single new run"],
    }
    trajectory["state_at_start"] = {
        "source_commit": commit, "source_config_identity": config_id,
        "contract_refs": [f"{REL}/training_contract.json"],
        "reference_ids": [bundle["reference_id"]],
        "parent_trajectory_ids": ["TC-k1-v4-100k-reference-s42"],
        "prior_trajectory_ids": (
            [] if args.attempt == "v1" else [TRAJECTORY_ID]
        ), "prior_evidence_ids": [bundle["reference_id"]],
        "role_snapshot_refs": [f"{target_rel}/role_plan.json"],
        "budget_snapshot_ref": f"{target_rel}/budget_snapshot.json",
    }
    trajectory["actions"] = [{
        "action_id": "A001", "type": "bounded_cpu_sidecar_then_100k_training",
        "run_ids": [run_id], "attempt_ids": [args.attempt],
        "source_commit": commit,
        "evidence_refs": [f"{target_rel}/source_config.json",
                          f"{target_rel}/comparison_readiness_prelaunch.json"],
        "cost_event_ids": [cost_id],
    }]
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
    trajectory["decision"] = {
        "decision_ref": f"{REL}/protocol.md", "outcome": "ACTIVE",
        "next_allowed_actions": ["CPU sidecar acceptance, then one preflight-gated seed42 GPU screen"],
        "reopen_conditions": ["terminal attribution and explicit new compute decision"],
    }
    cost = copy.deepcopy(load(TEMPLATE / "costs/expected_training.json"))
    cost.update(
        cost_event_id=cost_id, trajectory_id=trajectory_id,
        run_id=run_id, attempt_id=args.attempt, platform="kaggle2",
        hardware="actual_GPU_pending", category="training",
        evidence_ref=f"{target_rel}/budget_snapshot.json",
    )
    for unit in ("device_hours", "wall_hours"):
        estimate = (
            6.0 if args.attempt == "v1" else
            (3.669210724325182 if unit == "device_hours" else 1.834605362162591)
        )
        cost["measurement"][unit] = {"value": estimate, "status": "estimated"}
    created = plan(REPO_ROOT, {
        "trajectory": trajectory, "costs": [cost], "decision_state": {
            "available_actions": ["RUN_BOUNDED_METAGIN2D_SCREEN", "DEFER"],
            "chosen_action": "RUN_BOUNDED_METAGIN2D_SCREEN",
            "policy_id": POLICY_ID, "policy_version": "v1", "state_timestamp": now,
        },
    }, f"{target_rel}/rml_plan")
    print(json.dumps({
        "source_commit": commit, "source_archive_sha256": digest,
        "prelaunch_status": prelaunch["planned_status"],
        "trajectory_id": trajectory_id, "plan": str(created),
    }, indent=2))


if __name__ == "__main__":
    main()
