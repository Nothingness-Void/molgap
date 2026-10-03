"""Question declarations only; reuse the qualified follow-up release builder."""
from pathlib import Path
from molgap.constants import REPO_ROOT
import json
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_pair_scale_100k"
STUDY = {
    "experiment_id": "gptrans-g1-pair-scale-recovery",
    "kernel": "nvoid912/molgap-gptrans-g1-pair-scale-dual-s42",
    "dataset": "nvoid912/molgap-gptrans-g1-pair-scale-source",
    "dataset_title": "MolGap GPTrans G1 Pair Scale Source",
    "output_subdirectory": "gptrans_pair_scale_screen",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-scale-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-path-endpoints-ema999-100k-s42",
        "pcqm-gptrans-g1-group-decay-infrastructure-failure-100k-s42-v1"],
    "prior_trajectory_ids": ["TC-gptrans-g1-path-endpoints-100k-s42", "TC-gptrans-g1-group-decay-100k-s42"],
    "arms": {
        "degree_group_decay_ema999": {
            "suffix": "group-decay-recovery",
            "question": "Does bias/1D decay exemption improve G1 EMA999 after the infrastructure-only failed attempt?",
            "alternative_explanations": ["Grouping may change convergence but worsen generalization."]},
        "degree_pair_depth_scale_ema999": {
            "suffix": "pair-depth-scale",
            "question": "Does scaling each persistent pair update by 1/sqrt(12) improve G1 EMA999 without normalizing away relation magnitude?",
            "alternative_explanations": ["Pair growth may be useful signal; lower update scale may delay convergence rather than improve generalization."]},
    },
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE,
        modes=tuple(STUDY["arms"]), run="gptrans-g1-pair-scale-dual-s42", terminal_reference=True, study=STUDY)))
