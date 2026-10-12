"""Study declaration over native freeze/release; no local model execution."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_triplet_followup_100k"
STUDY = {
    "experiment_id": "gptrans-triplet-followup",
    "kernel": "kaseichou/molgap-gptrans-triplet-followup-dual-s42",
    "dataset": "kaseichou/molgap-gptrans-triplet-followup-source",
    "dataset_title": "MolGap GPTrans Triplet Followup Source",
    "output_subdirectory": "gptrans_triplet_followup",
    "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 8, "estimated_wall_hours": 5, "training_estimate_cap_hours": 5.5,
    "material_gate_eV": .001,
    "platform_id": "kaggle2-t4-gptrans-triplet-followup",
    "terminal_reference": {"experiment_ref": "experiments/pcqm_gptrans_triplet_communication_100k",
        "arm": "degree_local_triplet_aggregate_ema999"},
    "reference_parent_trajectory_id": "TC-gptrans-g1-local-triplet-aggregate-100k-s42",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-local-triplet-aggregate-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-local-triplet-attention-ema999-100k-s42"],
    "prior_trajectory_ids": ["TC-gptrans-g1-local-triplet-aggregate-100k-s42", "TC-gptrans-g1-local-triplet-attention-100k-s42"],
    "arms": {
        "degree_local_triplet_drop_ema999": {"suffix": "local-triplet-drop",
            "question": "Does isolated per-molecule triplet-branch regularization improve the aggregate selected endpoint and late generalization?",
            "alternative_explanations": ["Branch dropout may slow useful learning without improving generalization; one-seed selection noise remains."]},
        "degree_local_triplet_product_ema999": {"suffix": "local-triplet-product",
            "question": "Does channelwise two-relation composition improve aggregate routing without query matching or geometric inputs?",
            "alternative_explanations": ["Extra relational capacity may overfit the reused development role; cubic arithmetic may increase native cost."]},
    },
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=tuple(STUDY["arms"]),
        run="gptrans-triplet-followup-dual-s42", terminal_reference=True, study=STUDY)))
