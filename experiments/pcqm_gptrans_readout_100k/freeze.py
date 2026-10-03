"""Scientific declarations over the qualified prospective/release builder."""
from pathlib import Path
from molgap.constants import REPO_ROOT
import json
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_readout_100k"
STUDY = {
    "experiment_id": "gptrans-g1-final-readout",
    "kernel": "nvoid912/molgap-gptrans-g1-readout-dual-s42",
    "dataset": "nvoid912/molgap-gptrans-g1-readout-source",
    "dataset_title": "MolGap GPTrans G1 Final Readout Source",
    "output_subdirectory": "gptrans_readout_screen",
    "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 5, "estimated_wall_hours": 3.5,
    "training_estimate_cap_hours": 3.5,
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-scale-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-group-decay-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-pair-depth-scale-ema999-100k-s42"],
    "prior_trajectory_ids": ["TC-gptrans-g1-ema999-100k-s42",
        "TC-gptrans-g1-group-decay-recovery-100k-s42", "TC-gptrans-g1-pair-depth-scale-100k-s42"],
    "arms": {
        "degree_node_mean_readout_ema999": {
            "suffix": "node-mean-readout",
            "question": "Does final real-atom masked-mean readout improve G1 EMA999 while retaining its virtual pair readout?",
            "alternative_explanations": ["Virtual pooling may already retain the relevant signal; mean pooling may dilute rare chemical sites."]},
        "degree_bond_mean_readout_ema999": {
            "suffix": "bond-mean-readout",
            "question": "Does final real-bond relation pooling improve G1 EMA999 while retaining its virtual node readout?",
            "alternative_explanations": ["Global virtual relation state may be sufficient; local bond pooling may omit useful nonbonded/global information."]},
    },
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE,
        modes=tuple(STUDY["arms"]), run="gptrans-g1-readout-dual-s42", terminal_reference=True, study=STUDY)))
