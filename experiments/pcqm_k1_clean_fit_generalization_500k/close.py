"""Use the existing local diagnostic closure with observed clean-fit roles."""
from pathlib import Path
import json
from molgap.constants import REPO_ROOT
from molgap.frozen_diagnostic_closure import close_local_diagnostic
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    inputs = json.loads((here / "inputs.json").read_text())
    train, dev = inputs["train_source_idx"], inputs["development_source_idx"]
    receipt = close_local_diagnostic(REPO_ROOT, here,
        evidence_id="pcqm-k1-clean-fit-generalization-500k-20261009",
        policy_id="pcqm-k1-clean-fit-generalization-500k",
        role_rows={"train_decoded": {"labels_read": list(range(500000))},
                   "train_descriptive": {"prediction_input": train, "metric_computed": train},
                   "internal_development": {a:dev for a in ("labels_read", "prediction_input", "metric_computed", "selection_used")}})
    atomic_json(here / "closure_receipt.json", receipt)
    print(json.dumps(receipt))
