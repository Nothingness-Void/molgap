"""One optimizer coefficient question using the qualified G1 release builder."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_decay_clock_100k"
STUDY = {
    "experiment_id": "gptrans-g1-decay-clock", "kernel": "nvoid912/molgap-gptrans-g1-decay-clock-s42",
    "dataset": "nvoid912/molgap-gptrans-g1-decay-clock-source",
    "dataset_title": "MolGap GPTrans G1 Decay Clock Source",
    "output_subdirectory": "gptrans_decay_clock", "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 5, "estimated_wall_hours": 3.5, "training_estimate_cap_hours": 3.5,
    "single_arm_reason": "Only wd0.01 is justified; reuse the accepted wd0.05 G1 EMA999 reference, not a duplicate baseline or extra coefficient grid. Accounted allocation is two T4s.",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-scale-ema999-100k-s42", "pcqm-gptrans-g1-degree-group-decay-ema999-100k-s42"],
    "prior_trajectory_ids": ["TC-gptrans-g1-ema999-100k-s42", "TC-gptrans-g1-group-decay-recovery-100k-s42"],
    "arms": {"degree_decay001_ema999": {"suffix": "decay-clock-wd001",
        "question": "Does reducing the all-parameter AdamW coefficient from0.05 to0.01 improve G1 EMA999 at unchanged 100K exposure?",
        "alternative_explanations": ["Weaker regularization can worsen generalization; cumulative decay mismatch is a scale hypothesis, not established causation."]}},
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=tuple(STUDY["arms"]),
        run="gptrans-g1-decay-clock-s42", terminal_reference=True, study=STUDY)))
