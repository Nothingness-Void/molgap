"""Thin declaration over the existing native release and terminal reference."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_local_control_100k"
MODE = "degree_bond_local_cap_ema999"
STUDY = {
    "experiment_id": "gptrans-local-control", "kernel": "kaseichou/molgap-gptrans-local-control-s42",
    "dataset": "kaseichou/molgap-gptrans-local-control-source", "dataset_title": "MolGap GPTrans Local Control Source",
    "output_subdirectory": "gptrans_local_control", "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 7, "estimated_wall_hours": 4, "training_estimate_cap_hours": 6,
    "platform_id": "kaggle2-t4-gptrans-local-control",
    "single_arm_reason": "One amplitude-control hypothesis against the accepted local-bond reference; no duplicate baseline or unapproved second candidate.",
    "terminal_reference": {"experiment_ref": "experiments/pcqm_gptrans_capacity_relations_100k",
        "arm": "degree_bond_local_ema999", "acceptance_suffix": "/gpu/results/local_acceptance.json"},
    "reference_parent_trajectory_id": "TC-gptrans-g1-bond-local-100k-s42",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42", "pcqm-gptrans-bottleneck-frozen-v1"],
    "prior_trajectory_ids": ["TC-gptrans-g1-bond-local-100k-s42", "TC-gptrans-bottleneck-frozen-v1"],
    "arms": {MODE: {"suffix": "local-control",
        "question": "Does a per-molecule0.25 update/input norm cap at local layers4-12 improve the accepted local-bond GPTrans at identical exposure and frozen initialization?",
        "alternative_explanations": ["Amplitude correlation may not be causal; saturation may remove useful signal; baseline catch-up may explain late erosion."]}},
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=(MODE,),
        run="gptrans-local-control-s42", terminal_reference=True, study=STUDY)))
