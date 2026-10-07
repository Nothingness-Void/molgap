"""Scientific declarations; source/release/prospective plumbing is shared."""
from pathlib import Path
import argparse, json, subprocess
from molgap.pcqm_500k_preparation import read, write, ref, pinned, prepare_payload, prepare_continuation
from molgap.experiment_spec import ExperimentSpec, FAMILIES
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256
ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
REL = EXP.relative_to(ROOT).as_posix()
RUN = "molgap-k1-consistency-500k-pair-s42-v1"
POLICY = "pcqm-k1-consistency-ablation-500k-launch"
ARMS = ("k1_pretrained_mean2", "k1_pretrained_consistency")
MANIFEST = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
STAGING = ROOT / "platforms/_records/kaggle/staging/pcqm_k1_consistency_ablation_500k"
TRUSTED_PICKLE = Path("D:/文档/molgap-exp/molgap-500k-v4-evidence/data/cache/pcqm4mv2_500k_v4/train/train_shard_0000.pt")
REQUIRED_MODULES = ["molgap.pcqm_500k_v4_evidence", "molgap.pcqm_composed_500k", "molgap.qm9_neural_atom", "molgap.k1_pretrained_combo", "molgap.k1_screen_training", "molgap.pcqm_wedge"]
EXTRA_SOURCE = ("platforms/kaggle/run_legacy_500k_pair.py", "src/molgap/pcqm_500k_v4_evidence.py", "src/molgap/pcqm_composed_500k.py", "src/molgap/pcqm_k1_scale_runner.py", "src/molgap/pcqm_k1_scale.py", "src/molgap/futility_gate.py", "src/molgap/k1_pretrained_combo.py", "src/molgap/k1_bn_calibration.py")
EVIDENCE = ["pcqm-matched-500k-v4-three-arm", "pcqm-k1-dropout-consistency2-kaggle3-100k-s42-v1-terminal", "pcqm-k1-500k-bn-calibration-20261007", "pcqm-k1-500k-bottleneck-diagnostic-20261007"]
PARENTS = ["TB-matched-500k-v4-three-arm", "TB-k1-dropout-consistency2-kaggle3-100k-s42-v1", "TB-k1-500k-bn-calibration-20261007", "TB-k1-500k-bottleneck-diagnostic-20261007"]

def declare(source_commit, inputs):
    from molgap.pcqm_composed_500k import scientific_contract
    pins = read(inputs / "initial_pins.json")
    initial = {aid: pinned(inputs, pins["arms"][aid]) for aid in ARMS}
    transform = pinned(inputs, pins["target_transform"])
    write(EXP / "role_plan.json", {
        "train": "source_idx[0,500000): supervised training, train-only GPU qualification; no teacher",
        "development": "source_idx[500000,550000): selection, matched penalty ablation and fixed BN calibration; existing development history",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
        "manifest_sha256": MANIFEST,
        "authority": "Desktop user authorized the K1 penalty ablation on2026-10-07; see protocol.md and plan.md",
    })
    write(EXP / "budget.json", {"training_allocated_T4_hours_cap": 52, "max_stage_seconds": 32400,
        "stage_epochs": 60, "qualification_cost": "measured separately; unknown before execution",
        "resume": "Only the same frozen question with verified independently retained checkpoints; no automatic successor",
        "estimates": {ARMS[0]: 21.61, ARMS[1]: 21.61}, "estimate_unit": "T4 hours; rough scaling, not measured"})
    policy = read(ROOT / "research_memory/policies/pcqm-k1-dropout-consistency-kaggle3-100k-launch.1.json")
    policy.update(policy_id=POLICY, created_from_source_digest=sha256_file(EXP / "plan.md"),
        comparability_selector={"scientific_contract": "pcqm-k1-consistency-ablation-500k-v1"},
        action_rule={"action": "QUALIFY_THEN_TRAIN_PAIR_500K", "field": "evidence_review_complete", "operator": "eq", "threshold": 1})
    write(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    arms, bindings = [], []
    for device, aid in enumerate(ARMS):
        family = ("neural_atom_k1", "3")
        owner = FAMILIES[family]
        recipe = scientific_contract(aid)
        recipe.update(initial_state_sha256=pins["arms"][aid]["sha256"],
            target_transform_artifact_sha256=None,
            source_module_sha256=normalized_source_sha256(ROOT / "src/molgap/pcqm_composed_500k.py"))
        recipe_path = f"{REL}/training_recipe_{aid}.json"
        write(ROOT / recipe_path, recipe)
        roles = []
        for role, start, stop, usage in (("train", 0, 500000, "supervision-and-train-only-qualification"), ("development", 500000, 550000, "selection-and-penalty-ablation-and-fixed-bn-calibration")):
            interval = {"source_idx_start": start, "source_idx_stop": stop, "manifest_sha256": MANIFEST}
            roles.append({"role": role, "membership_sha256": canonical_fingerprint(interval),
                "row_order_sha256": canonical_fingerprint({**interval, "order": "ascending-source_idx"}),
                "usage_sha256": canonical_fingerprint({"role": role, "usage": usage, "role_plan_sha256": sha256_file(EXP / "role_plan.json")})})
        arm = {"arm_id": aid, "scientific_role": "candidate", "family": {"name": family[0], "version": family[1]},
            "base": ref("composed-500k-source", {"source_module_sha256": recipe["source_module_sha256"], "arm": aid}),
            "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": pins["arms"][aid]["state_sha256"]},
            "data": {"dataset": {"name": "pcqm4mv2", "version": "1", "sha256": MANIFEST},
                "split": ref("accepted-fixed-500k-train-development", {"train": [0, 500000], "development": [500000, 550000], "manifest_sha256": MANIFEST}),
                "roles": roles, "feature_schema": owner.feature_schema, "feature_sha256": MANIFEST, "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": {"name": owner.recipe, "version": "1", "sha256": sha256_file(ROOT / recipe_path)}, "overrides": {},
                "objective": ref("normalized-gap-l1-dropout-consistency", {"arm": aid, "contract": recipe}),
                "sampler": ref(owner.sampler, {"seed": 42, "epochs": 60, "batch": 128, "drop_last": 32, "steps": 234360, "presentations": 29998080}),
                "transform": ref(owner.transform, {"manifest": MANIFEST})},
            "addons": [], "addon_semantics": "baseline"}
        arms.append(arm)
        tid = "TB-k1-consistency-ablation-500k-" + aid + "-s42-v1"
        cid, run = "cost-" + tid + "-expected-training", RUN + ":" + aid
        action_path = f"{REL}/action_inputs_{aid}.json"
        write(ROOT / action_path, {"trajectory_id": tid, "state_timestamp": "2026-10-07", "evidence_review_complete": 1, "evidence_ids": EVIDENCE})
        state = {"source_commit": source_commit, "source_config_identity": canonical_fingerprint(arm),
            "prior_evidence_ids": EVIDENCE, "parent_trajectory_ids": PARENTS, "reference_ids": [],
            "budget_snapshot_ref": f"{REL}/budget.json", "role_snapshot_refs": [f"{REL}/role_plan.json"],
            "contract_refs": [f"{REL}/protocol.md", f"{REL}/plan.md", recipe_path, f"{REL}/role_plan.json", f"{REL}/budget.json"]}
        plan = {"action_inputs_ref": action_path, "decision_state": {
            "known_trajectory_ids": PARENTS, "known_evidence_ids": EVIDENCE, "active_reference_ids": [],
            "available_actions": ["QUALIFY_THEN_TRAIN_PAIR_500K", "NO_TRAIN"], "chosen_action": "QUALIFY_THEN_TRAIN_PAIR_500K",
            "policy_id": POLICY, "policy_version": "1", "budget_snapshot_ref": state["budget_snapshot_ref"],
            "role_snapshot_refs": state["role_snapshot_refs"], "source_commit": source_commit, "state_timestamp": "2026-10-07"},
            "trajectory": {"schema": "molgap-trajectory-v1", "record_mode": "prospective", "owner": "desktop", "track": "B",
                "trajectory_id": tid, "family_id": "k1-consistency-ablation-500k", "question": "Does weight0 versus weight0.1 disagreement improve the identical pretrained two-forward K1 at500K/60passes? Arm " + aid + " records its matched endpoint.",
                "state_at_start": state, "hypothesis": {"hypothesis_id": "H-" + tid,
                    "observed_deficiency": "The teacher-free500K enhancement package did not improve the historical K1 endpoint; the consistency contribution has not been isolated at500K.",
                    "changed_mechanism": "Only disagreement coefficient0 versus0.1; identical pretraining, two forwards, normalization, BN updates, native optimizer and exposure.",
                    "alternative_explanations": ["BN-state sensitivity", "Development selection reuse", "Single-seed training variation", "Limited retained100K pretraining coverage"],
                    "cheapest_falsifier": "CPU release and all-arm training-only T4 deterministic optimizer qualification before any training arm.",
                    "decision_changed_if_positive": "Nominate the better coefficient only for gain>=1meV with favorable paired-row95% bounds; compare both states after fixed BN calibration; no automatic full training.",
                    "decision_changed_if_negative": "Close with retained exposure/objective/curve/cost attribution; no automatic sweep.",
                    "supporting_evidence_ids": EVIDENCE, "related_closed_family_ids": ["k1-dropout-consistency", "k1-500k-bn-calibration", "k1-500k-bottleneck-diagnostic"],
                    "historical_unknowns": ["Training stochasticity", "Strict cross-contract historical qualification", "Historical pretraining native costs; preserved rather than invented"],
                    "expected_native_cost_ref": cid},
                "actions": [{"action_id": "A001", "type": "authorized_consistency_ablation_pair_500k", "attempt_ids": ["kaggle1-consistency-ablation-001"],
                    "run_ids": [run], "cost_event_ids": [cid], "evidence_refs": [f"{REL}/protocol.md", f"{REL}/plan.md"], "source_commit": source_commit}],
                "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/protocol.md", "next_allowed_actions": ["A001"], "reopen_conditions": []},
                "result": {"evidence_ids": [], "evidence_refs": []}},
            "costs": [{"schema": "molgap-cost-event-v1", "trajectory_id": tid, "cost_event_id": cid, "action_id": "A001",
                "attempt_id": "kaggle1-consistency-ablation-001", "run_id": run, "category": "training", "platform": "kaggle",
                "hardware": f"Tesla T4; assignedGPU{device} in2T4 pair", "evidence_ref": f"{REL}/budget.json",
                "measurement": {"device_hours": {"status": "estimated", "value": 21.61},
                    "wall_hours": {"status": "estimated", "value": 21.61},
                    "cpu_hours": {"status": "measurement_missing", "value": None}, "queue_hours": {"status": "measurement_missing", "value": None}}}]}
        plan_path = f"{REL}/training_plan_{aid}.json"
        write(ROOT / plan_path, plan)
        bindings.append({"arm_id": aid, "trajectory_id": tid, "plan_spec_ref": plan_path, "plan_spec_sha256": sha256_file(ROOT / plan_path), "output": f"{REL}/kaggle1_v1/{aid}"})
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": "pcqm-k1-consistency-ablation-500k", "logical_run_id": RUN,
        "arms": arms, "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2, "cpu_cores": 4, "memory_gib": 32, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings}, "evidence": {"policy": ref("molgap-v5", {"common": sha256_file(ROOT / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")}),
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": "molgap-experiment-terminal-descriptor-v1"})
    write(EXP / "experiment_spec.json", spec.to_dict())
    return spec, pins, initial, transform


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--declare-only",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--resume-prepared",action="store_true")
    p.add_argument("--continuation-from",type=Path)
    p.add_argument("--resume-root",type=Path)
    p.add_argument("--checkpoint-dataset")
    a=p.parse_args()
    if a.continuation_from:
        if not all((a.resume_root,a.output,a.checkpoint_dataset)):
            p.error("Continuation needs resume-root, output and fresh checkpoint-dataset")
        prepare_continuation(a.continuation_from,a.resume_root,a.output,TRUSTED_PICKLE,a.checkpoint_dataset,repo_root=ROOT,experiment_dir=EXP)
        return
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    inputs=STAGING/"inputs"
    spec,pins,initial,transform=declare(commit,inputs)
    if a.declare_only:
        print(json.dumps({"status":"DECLARATIONS_PREPARED","spec_identity":spec.identity}))
        return
    if not a.output:
        p.error("Provide a fresh output directory")
    result=prepare_payload(ROOT,EXP,spec,pins,initial,transform,a.output,
        source_files=EXTRA_SOURCE,required_modules=REQUIRED_MODULES,run_id=RUN,
        source_dataset="nothingnessvoid/molgap-k1-consistency-500k-source-s42-v1",
        title="MolGap K1 Consistency 500K Pair S42 V1",
        source_title="MolGap K1 Consistency 500K Source S42 V1",
        inputs=inputs,pickle_input=TRUSTED_PICKLE,resume_prepared=a.resume_prepared)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
