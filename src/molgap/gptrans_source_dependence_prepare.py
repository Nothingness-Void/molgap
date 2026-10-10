"""Retained-artifact identity plus shared NO_TRAIN plan/package preparation."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from .gptrans_source_dependence import BASE, PROFILE, KERNEL, DATASET, panel_indices
from .gptrans_portability import verify_file
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file

ASSETS = "platforms/_records/kaggle/source_dependence_assets_v1"
PREVIOUS = "experiments/pcqm_gptrans_triplet_portability"
PARENT = "experiments/pcqm_gptrans_bottleneck_audit"


def freeze_assets(root):
    """Read saved CPU tensors only. Never construct or execute a model."""
    import torch
    root = Path(root)
    if (root / BASE / "contract.json").exists():
        raise ValueError("Never refreeze an existing contract")
    previous = json.loads((root/PREVIOUS/"contract.json").read_text())
    parent = json.loads((root/PARENT/"contract.json").read_text())
    destination = root / ASSETS
    destination.mkdir(parents=True, exist_ok=False)
    copies = {
        "parent_best.pt": (root/"platforms/_records/kaggle/bottleneck_assets_v1/local_best.pt", parent["model_assets"]["local_best.pt"]["sha256"]),
        "triplet_best.pt": (root/"platforms/_records/kaggle/triplet_portability_assets_v1/local_best.pt", previous["model_assets"]["local_best.pt"]["sha256"]),
        "parent_original.pt": (root/"platforms/_records/kaggle/triplet_portability_assets_v1/reference_predictions.pt", previous["reference_payloads"]["reference_predictions.pt"]["sha256"]),
        "parent_later.pt": (root/"platforms/_records/kaggle/triplet_portability_assets_v1/reference_later.pt", previous["reference_payloads"]["reference_later.pt"]["sha256"]),
        "triplet_original.pt": (root/"platforms/_records/kaggle/triplet_portability_assets_v1/local_predictions.pt", previous["reference_payloads"]["local_predictions.pt"]["sha256"]),
    }
    for name, (path, digest) in copies.items():
        verify_file(path, digest)
        shutil.copyfile(path, destination/name)
    prior_output = root/"platforms/_records/kaggle/training/gptrans_triplet_portability_v1/gptrans_triplet_portability"
    evidence = json.loads((root/PREVIOUS/"results/terminal_evidence.json").read_text())
    manifest = next(a for a in evidence["artifacts"] if a["name"]=="output_manifest")
    verify_file(prior_output/"output_manifest.json", manifest["sha256"])
    inventory = json.loads((prior_output/"output_manifest.json").read_text())["files"]
    relative = "portability/unseen_500k/"
    verify_file(prior_output/relative/"progress.json", inventory[relative+"progress.json"])
    progress = json.loads((prior_output/relative/"progress.json").read_text())
    if progress["model_sha256"] != previous["model_assets"]["local_best.pt"]["sha256"] or len(progress["chunks"]) != 2:
        raise ValueError("Triplet later predictions not bound to accepted checkpoint")
    parts = []
    for n,row in enumerate(progress["chunks"]):
        if row["file"] != f"chunk_{n:02d}.pt" or row["rows"] != 5000:
            raise ValueError("Triplet later chunk identity differs")
        path = prior_output/relative/row["file"]
        verify_file(path,row["sha256"])
        verify_file(path,inventory[relative+row["file"]])
        parts.append(torch.load(path,map_location="cpu",weights_only=False))
    atomic_torch_save(destination/"triplet_later.pt", {k:torch.cat([p[k] for p in parts]) for k in ("source_idx","prediction_eV","target_eV")})
    shutil.copyfile(root/"platforms/_records/kaggle/triplet_portability_assets_v1/target_transform.json",destination/"target_transform.json")
    panels = {role:(panel_indices(role)+(100000 if role=="original_100k" else 500000)).tolist() for role in ("original_100k","unseen_500k")}
    atomic_json(destination/"panels.json", panels)
    contract = dict(format="molgap-gptrans-source-dependence-contract-v1", release_profile=PROFILE,
        experiment_purpose="NO_TRAIN", comparison_class="CONTEXT_ONLY", training_executed=False,
        optimizer_steps=0, physical_batch=128, precision="fp32", tf32_enabled=False,
        allocation_cap_seconds=1800, workers=["sources"], protected_roles_read=False,
        roles={"original_100k":[100000,150000],"unseen_500k":[500000,550000]},
        diagnostic_rows_per_role=512, panel_selection="RandomState42-choice-without-replacement-sorted-input-ID-only;later-within-retained10000",
        cohort_is_reused_development=True, weight_mode="eval-EMA-frozen-no-update",
        observer_policy="read-only-shadow-projections;bitwise-no-observer-prediction-check;unchanged-state-and-RNG",
        automatic_training_released=False, model_assets={
            "parent_best.pt":deepcopy(parent["model_assets"]["local_best.pt"]),
            "triplet_best.pt":deepcopy(previous["model_assets"]["local_best.pt"])},
        reference_payloads={name:dict(sha256=sha256_file(destination/name)) for name in
            ("parent_original.pt","parent_later.pt","triplet_original.pt","triplet_later.pt","panels.json")},
        source_identities=previous["source_identities"],
        target_transform_asset=json.loads((destination/"target_transform.json").read_text())["asset_sha256"],
        prior_later_manifest=manifest)
    atomic_json(root/BASE/"contract.json",contract)
    return contract


def prepare(root, output):
    from .frozen_audit_prepare import prepare_package
    root = Path(root).resolve()
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    t = deepcopy(json.loads((root/PREVIOUS/"rml_plan/trajectory.json").read_text()))
    t.pop("decision_state",None)
    tid,run = "TC-gptrans-source-dependence-s42-v1","gptrans-source-dependence:v1"
    refs = ["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42","pcqm-gptrans-g1-degree-local-triplet-aggregate-ema999-100k-s42","pcqm-gptrans-triplet-portability-s42-v1"]
    t.update(trajectory_id=tid,family_id="gptrans-source-dependence",question="Do input-only matched cohorts expose attention-source concentration or triplet gate-mass shift in frozen parent and aggregation?")
    t["state_at_start"].update(source_commit=commit,source_config_identity=sha256_file(root/BASE/"contract.json"),
        contract_refs=[BASE+"/contract.json",BASE+"/protocol.md"],reference_ids=refs,prior_evidence_ids=refs,
        prior_trajectory_ids=["TC-gptrans-g1-local-triplet-aggregate-100k-s42","TC-gptrans-triplet-portability-s42-v1"],
        parent_trajectory_ids=[],role_snapshot_refs=[BASE+"/role_plan.json"],budget_snapshot_ref=BASE+"/budget.json")
    t["hypothesis"].update(hypothesis_id="H-"+tid,observed_deficiency=t["question"],supporting_evidence_ids=refs,
        alternative_explanations=["source dependence shifts by cohort","gate mass shifts by cohort","sampling or other unmeasured training factors"],
        changed_mechanism="none; read-only observations of frozen checkpoints",cheapest_falsifier="two models;512 input-only rows per role;reproduce retained predictions",
        expected_native_cost_ref="cost-"+tid,decision_changed_if_positive="prioritize separately authorized source-regularization hypothesis",
        decision_changed_if_negative="do not train a source/gate fix based on this explanation",
        historical_unknowns=["concentration is not causal failure","not retraining or full-cohort audit","reused development;single seed"])
    t["actions"]=[dict(action_id="A001",type="NO_TRAIN_frozen_inference",source_commit=commit,run_ids=[run],attempt_ids=["v1"],evidence_refs=[BASE+"/contract.json"],cost_event_ids=["cost-"+tid])]
    t["result"]=dict(evidence_ids=[],evidence_refs=[])
    t["decision"]=dict(decision_ref=BASE+"/decision.md",outcome="ACTIVE",next_allowed_actions=["bounded read-only diagnostic and terminal acceptance"],reopen_conditions=["terminal state or actionable execution fault"])
    missing=dict(status="measurement_missing",value=None)
    spec=dict(trajectory=t,decision_state=dict(known_trajectory_ids=[],known_evidence_ids=[],active_reference_ids=refs,
        available_actions=["RUN_BOTTLENECK_AUDIT","DEFER"],chosen_action="RUN_BOTTLENECK_AUDIT",policy_id="gptrans-bottleneck-audit",policy_version="v1",
        budget_snapshot_ref=BASE+"/budget.json",role_snapshot_refs=[BASE+"/role_plan.json"],state_timestamp=datetime.now(timezone.utc).isoformat(),source_commit=commit),
        costs=[dict(schema="molgap-cost-event-v1",cost_event_id="cost-"+tid,trajectory_id=tid,action_id="A001",run_id=run,attempt_id="v1",
            category="audit",platform="kaggle2",hardware="Tesla_T4_up_to2",evidence_ref=BASE+"/budget.json",
            measurement=dict(device_hours=dict(status="estimated",value=.3),wall_hours=dict(status="estimated",value=.15),cpu_hours=missing,queue_hours=missing))])
    return prepare_package(root,output,base=BASE,assets=ASSETS,kernel=KERNEL,dataset=DATASET,plan_spec=spec)
