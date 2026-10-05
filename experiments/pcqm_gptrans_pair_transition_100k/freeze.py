"""Thin declaration over the retained GPTrans release owner; no new runner."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_pair_transition_100k"
STUDY = {
    "experiment_id": "gptrans-g1-pair-transition", "kernel": "kaseichou/molgap-gptrans-pair-transition-s42",
    "dataset": "kaseichou/molgap-gptrans-pair-transition-source", "dataset_title": "MolGap GPTrans Pair Transition Source",
    "output_subdirectory": "gptrans_pair_transition", "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 7, "estimated_wall_hours": 4, "training_estimate_cap_hours": 6,
    "platform_id": "kaggle2-t4-gptrans-pair-transition",
    "single_arm_reason": "One authorized relation-transition hypothesis; accepted G1 EMA999 reference is reused, not retrained to fill the second allocated T4.",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-scale-ema999-100k-s42", "pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42"],
    "prior_trajectory_ids": ["TC-gptrans-g1-ema999-100k-s42", "TC-gptrans-g1-bond-local-100k-s42"],
    "arms": {"degree_pair_transition_ema999": {
        "suffix": "pair-transition",
        "question": "Does an independent nonlinear32to64to32 real-pair transition before GPA3/6/9/12 improve correctedG1 at fixed exposure?",
        "alternative_explanations": ["Original GPA relation flow may already suffice; extra pair freedom may overfit and quadratic operations may fail cost."]}},
}

if __name__ == "__main__":
    result = freeze_followup(REPO_ROOT, base=BASE, modes=tuple(STUDY["arms"]),
        run="gptrans-pair-transition-s42", terminal_reference=True, study=STUDY)
    from molgap.training_reproducibility import atomic_json
    from molgap.evidence_pointers import load_json_object
    workflow_path = REPO_ROOT / BASE / "gpu/release_workflow.json"
    workflow = load_json_object(workflow_path)
    workflow["source_paths"].append(BASE + "/audit_action.json")
    atomic_json(workflow_path, workflow)
    print(json.dumps(result))
