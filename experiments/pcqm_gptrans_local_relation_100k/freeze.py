"""Thin two-hypothesis declaration over the native qualified release builder."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_local_relation_100k"
STUDY = {
    "experiment_id": "gptrans-local-relation", "kernel": "kaseichou/molgap-gptrans-local-relation-dual-s42",
    "dataset": "kaseichou/molgap-gptrans-local-relation-source", "dataset_title": "MolGap GPTrans Local Relation Source",
    "output_subdirectory": "gptrans_local_relation", "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 7, "estimated_wall_hours": 5, "training_estimate_cap_hours": 5,
    "platform_id": "kaggle2-t4-gptrans-local-relation",
    "frozen_reference_bundle_ref": "experiments/pcqm_gptrans_local_control_100k/reference/reference_bundle.json",
    "reference_parent_trajectory_id": "TC-gptrans-g1-bond-local-100k-s42",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42", "pcqm-gptrans-bottleneck-frozen-v1"],
    "prior_trajectory_ids": ["TC-gptrans-g1-bond-local-100k-s42", "TC-gptrans-bottleneck-frozen-v1"],
    "arms": {
        "degree_local_connected_pair_ema999": {"suffix": "local-connected-pair",
            "question": "Does virtual-connected valid-pair nonlinearity improve the retained local-bond GPTrans at equal100K exposure?",
            "alternative_explanations": ["Core GPA may already supply sufficient relation processing; connected dense pair freedom can overfit or cost more than it helps."]},
        "degree_local_bond_return_ema999": {"suffix": "local-bond-return",
            "question": "Does same-block sparse true-bond-message-to-pair-to-node feedback improve the retained local parent without widening nodes?",
            "alternative_explanations": ["GPA may already retain this chemistry; double local-message evaluation can add cost without generalization."]},
    },
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=tuple(STUDY["arms"]),
        run="gptrans-local-relation-dual-s42", terminal_reference=True, study=STUDY)))
