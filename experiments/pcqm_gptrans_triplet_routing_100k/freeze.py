"""Small study declaration; reuse native prospective/release and audit assets."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup
from molgap.gptrans_triplet_routing import MODES
from molgap.experiment_staging import UploadArtifact
from molgap.training_reproducibility import atomic_json

BASE = "experiments/pcqm_gptrans_triplet_routing_100k"
STUDY = {
    "experiment_id": "gptrans-triplet-routing",
    "kernel": "kaseichou/molgap-gptrans-triplet-routing-dual-s42",
    "dataset": "kaseichou/molgap-gptrans-triplet-routing-source",
    "dataset_title": "MolGap GPTrans Triplet Routing Source",
    "output_subdirectory": "gptrans_triplet_routing",
    "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 8, "estimated_wall_hours": 5, "training_estimate_cap_hours": 5.5,
    "material_gate_eV": .001,
    "platform_id": "kaggle2-t4-gptrans-triplet-routing",
    "all_arms_preflight_barrier": True,
    "frozen_reference_bundle_ref": "experiments/pcqm_gptrans_triplet_followup_100k/reference/reference_bundle.json",
    "reference_parent_trajectory_id": "TC-gptrans-g1-local-triplet-aggregate-100k-s42",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-local-triplet-aggregate-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-local-triplet-drop-ema999-100k-s42"],
    "prior_trajectory_ids": ["TC-gptrans-g1-local-triplet-aggregate-100k-s42",
        "TC-gptrans-g1-local-triplet-drop-100k-s42", "TC-gptrans-g1-local-triplet-product-100k-s42"],
    "arms": {
        MODES[0]: {"suffix": "local-triplet-product-recovery",
            "question": "Does the previously unevaluated exact channelwise product improve aggregate routing?",
            "alternative_explanations": ["Product capacity may overfit; failed prior preflight was not scientific evidence."]},
        MODES[1]: {"suffix": "local-triplet-bond-route",
            "question": "Does bond-supported junction routing improve generalization without deleting entire triplet returns?",
            "alternative_explanations": ["Nonbonded values may be useful; conditional normalization may amplify noise."]},
    },
}

if __name__ == "__main__":
    result = freeze_followup(REPO_ROOT, base=BASE, modes=MODES,
        run="gptrans-triplet-routing-dual-s42", terminal_reference=True, study=STUDY)
    old = REPO_ROOT / "experiments/pcqm_gptrans_triplet_followup_100k/gpu/post100k_audit_action.json"
    audit = json.loads(old.read_bytes())
    audit.update(training_spec_identity=result["spec_identity"], candidate_modes=list(MODES),
        parent_training_trajectories=["TC-gptrans-g1-" + STUDY["arms"][m]["suffix"] + "-100k-s42" for m in MODES],
        audit_trajectory_ids=["TC-gptrans-g1-" + STUDY["arms"][m]["suffix"] + "-full50k-audit-s42" for m in MODES],
        reference_bundle=STUDY["frozen_reference_bundle_ref"])
    audit_ref = BASE + "/gpu/post100k_audit_action.json"
    atomic_json(REPO_ROOT / audit_ref, audit)
    workflow_path = REPO_ROOT / BASE / "gpu/release_workflow.json"
    workflow = json.loads(workflow_path.read_bytes())
    workflow["source_paths"].append(audit_ref)
    workflow["artifacts"]["post100k_audit_action.json"] = UploadArtifact.from_file(REPO_ROOT / audit_ref).to_workflow()
    atomic_json(workflow_path, workflow)
    print(json.dumps(result))
