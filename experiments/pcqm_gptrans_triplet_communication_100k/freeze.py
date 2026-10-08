"""Study declaration only; native owner freezes contracts/initials/release."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_triplet_communication_100k"
STUDY = {
    "experiment_id": "gptrans-triplet-communication",
    "kernel": "kaseichou/molgap-gptrans-triplet-communication-dual-s42",
    "dataset": "kaseichou/molgap-gptrans-triplet-communication-source",
    "dataset_title": "MolGap GPTrans Triplet Communication Source",
    "output_subdirectory": "gptrans_triplet_communication",
    "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 7, "estimated_wall_hours": 5, "training_estimate_cap_hours": 5,
    "platform_id": "kaggle2-t4-gptrans-triplet-communication",
    "frozen_reference_bundle_ref": "experiments/pcqm_gptrans_local_control_100k/reference/reference_bundle.json",
    "reference_parent_trajectory_id": "TC-gptrans-g1-bond-local-100k-s42",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-local-connected-pair-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-local-bond-return-ema999-100k-s42"],
    "prior_trajectory_ids": ["TC-gptrans-g1-bond-local-100k-s42",
        "TC-gptrans-g1-local-connected-pair-100k-s42", "TC-gptrans-g1-local-bond-return-100k-s42"],
    "arms": {
        "degree_local_triplet_aggregate_ema999": {"suffix": "local-triplet-aggregate",
            "question": "Does third-pair-gated inward/outward aggregation improve frozen local-bond GPTrans at matched100K exposure?",
            "alternative_explanations": ["Averaged pair context can be redundant or oversmooth; prior nonlinear gain did not prove a communication bottleneck."]},
        "degree_local_triplet_attention_ema999": {"suffix": "local-triplet-attention",
            "question": "Does query-conditioned triplet selection improve the same frozen parent sufficiently to justify its cubic compute?",
            "alternative_explanations": ["Query-dependent relation selection may overfit; TGT's geometric training could be essential for its reported benefit."]},
    },
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=tuple(STUDY["arms"]),
        run="gptrans-triplet-communication-dual-s42", terminal_reference=True, study=STUDY)))
