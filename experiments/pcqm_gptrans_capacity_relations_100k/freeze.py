"""One local100K addon plus one500K shared-live study, no extra experiments."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_capacity_relations_100k"
STUDY = {"experiment_id":"gptrans-g1-local-scale", "kernel":"kaseichou/molgap-gptrans-g1-local-scale-s42",
    "dataset":"kaseichou/molgap-gptrans-g1-local-scale-source", "dataset_title":"MolGap GPTrans G1 Local Scale",
    "output_subdirectory":"gptrans_local_scale", "entry_template":"platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours":7, "estimated_wall_hours":4, "training_estimate_cap_hours":6,
    "platform_id":"kaggle2-t4-gptrans-g1-capacity",
    "supporting_evidence_ids":["pcqm-gptrans-g1-degree-scale-ema999-100k-s42","pcqm-gptrans-g1-scale500k-qualification-s42"],
    "prior_trajectory_ids":["TC-gptrans-g1-ema999-100k-s42","TC-gptrans-g1-scale500k-qualification-s42"],
    "arms": {
        "degree_bond_local_ema999":{"suffix":"bond-local", "question":"Does a real-bond interior message bypass supply local information missing from corrected dense GPA?",
            "alternative_explanations":["The dense pair state may already contain this information; extra local fitting can worsen dev generalization."]},
        "scale_ema":{"suffix":"scale-ema-equal-updates", "question":"Does the EMA999 selected-prediction benefit persist during real500K training on one shared live trajectory?",
            "alternative_explanations":["More molecule diversity or later convergence can erase the filter advantage; equal steps are not60full passes."]}}}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=tuple(STUDY["arms"]),
        run="gptrans-g1-local-scale-s42", terminal_reference=True, study=STUDY)))
