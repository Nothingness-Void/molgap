"""Thin declarations; reuse the GPTrans prospective/release builder."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_capacity_nodes_100k"
STUDY = {"experiment_id":"gptrans-g1-capacity-nodes", "kernel":"kaseichou/molgap-gptrans-g1-node-capacity-s42",
    "dataset":"kaseichou/molgap-gptrans-g1-node-capacity-source", "dataset_title":"MolGap GPTrans G1 Node Capacity",
    "output_subdirectory":"gptrans_node_capacity", "entry_template":"platforms/kaggle/bootstrap_gptrans_author.py",
    "maximum_wall_hours":7, "estimated_wall_hours":4.5, "training_estimate_cap_hours":6,
    "platform_id":"kaggle2-t4-gptrans-g1-capacity",
    "supporting_evidence_ids":["pcqm-gptrans-g1-degree-scale-ema999-100k-s42"],
    "prior_trajectory_ids":["TC-gptrans-g1-ema999-100k-s42"],
    "arms": {
        "degree_node352_ema999":{"suffix":"node352", "question":"Does widening the corrected GPTrans node stream256to352 improve its100K endpoint at unchanged recipe?",
            "alternative_explanations":["Extra capacity can overfit or require more exposure; different shapes consume different constructor RNG."]},
        "degree_ffn2_ema999":{"suffix":"ffn2", "question":"Does doubling only node FFN hidden width improve corrected GPTrans without widening GPA?",
            "alternative_explanations":["Added hidden units may be redundant; enlarged dropout changes training RNG even with shared core tensors."]}}}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT, base=BASE, modes=tuple(STUDY["arms"]),
        run="gptrans-g1-node-capacity-s42", terminal_reference=True, study=STUDY)))
