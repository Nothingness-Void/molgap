"""Thin declaration: retained G1 control plus one new local-stream optimization."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup
from molgap.gptrans_scale_reference import enroll_scale_reference

BASE = "experiments/pcqm_gptrans_local_transfer_500k"
STUDY = {
    "experiment_id": "gptrans-local-transfer500k", "kernel": "kaseichou/molgap-gptrans-local-transfer500k-s42",
    "dataset": "kaseichou/molgap-gptrans-local-transfer500k-source", "dataset_title": "MolGap GPTrans Local Transfer500K Source",
    "output_subdirectory": "gptrans_local_transfer500k", "entry_template": "platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours": 7, "estimated_wall_hours": 5, "training_estimate_cap_hours": 6,
    "platform_id": "kaggle2-t4-gptrans-local-transfer500k",
    "single_arm_reason": "The accepted equal-update500K G1 control is retained and qualified; train only the local candidate, account for both allocated T4s without inventing another experiment.",
    "scale_model_variant": "degree_bond_local_ema999",
    "retained_initial_facts": {"parameters": 5871201,
        "file_sha256": "d471d9f94ff2c10436261b7f70f3e2d9111978f1ac8aadfabea2da32a5fe9a1d",
        "state_sha256": "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b",
        "architecture_identity": "a79ec716d8ce41d744d9ea05d2f160d54ffac1acc24c3ba1ac276afe800897b3"},
    "reference_parent_trajectory_id": "TC-gptrans-g1-scale-ema-equal-updates-500k-s42",
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42", "pcqm-gptrans-g1-scale-ema-equal-updates-500k-s42", "pcqm-gptrans-bottleneck-frozen-v1"],
    "prior_trajectory_ids": ["TC-gptrans-g1-bond-local-100k-s42", "TC-gptrans-g1-scale-ema-equal-updates-500k-s42", "TC-gptrans-bottleneck-frozen-v1"],
    "arms": {"scale_ema": {"suffix": "local-transfer-equal-updates",
        "question": "Does the uncapped real-bond local stream retain its gain after genuine500K optimization at identical steps and unchanged normalized Gap recipe?",
        "alternative_explanations": ["Early acceleration can disappear at the common endpoint; the local stream can improve fit but not generalization; six million presentations do not establish full convergence."]}},
}

if __name__ == "__main__":
    reference = enroll_scale_reference(REPO_ROOT)
    STUDY["frozen_reference_bundle_ref"] = reference["reference_bundle_ref"]
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=("scale_ema",),
        run="gptrans-local-transfer500k-s42", terminal_reference=True, study=STUDY)))
