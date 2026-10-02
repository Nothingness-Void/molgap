"""Only combination declarations; reuse prospective/release owners."""
from pathlib import Path
import json
from molgap.gptrans_followup_release import freeze_followup

BASE = "experiments/pcqm_gptrans_path_ema_combination_100k"
MODE = "degree_path_bond_mean_ema999"
STUDY = {
    "experiment_id": "gptrans-g1-path-ema-combination",
    "kernel": "nvoid912/molgap-gptrans-g1-path-ema-combined-s42",
    "dataset": "nvoid912/molgap-gptrans-g1-path-ema-combined-source",
    "dataset_title": "MolGap GPTrans G1 Path EMA Combination Source",
    "output_subdirectory": "gptrans_path_ema_combination_screen",
    "single_arm_reason": "One authorized interaction candidate; immutable G1 EMA999 reference reused without retraining. Count both allocated T4s even when GPU1 is idle.",
    # Independent in-progress generic workflow edits are not native trainer inputs.
    "source_exclusions": ["src/molgap/experiment_execution.py", "src/molgap/experiment_source_inventory.py",
        "src/molgap/experiment_preflight.py"],
    "supporting_evidence_ids": ["pcqm-gptrans-g1-degree-scale-ema999-100k-s42",
        "pcqm-gptrans-g1-degree-path-bond-mean-100k-s42"],
    "prior_trajectory_ids": ["TC-gptrans-g1-ema999-100k-s42", "TC-gptrans-g1-path-mean-100k-s42"],
    "arms": {MODE: {"suffix": "path-mean-ema999-combination",
        "question": "Does chemical path mean improve corrected-EMA G1, rather than a lagging EMA9999 comparator?",
        "alternative_explanations": ["Path input may duplicate existing pair information or impair generalization; correcting EMA alone does not imply an interaction gain."]}},
}

if __name__ == "__main__":
    print(json.dumps(freeze_followup(Path(__file__).resolve().parents[2], base=BASE,
        modes=(MODE,), run="gptrans-g1-path-ema-combined-s42", terminal_reference=True, study=STUDY)))
