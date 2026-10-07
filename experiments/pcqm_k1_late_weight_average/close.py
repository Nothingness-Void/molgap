"""Bind measured local evidence through shared diagnostic closure and RML."""
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.frozen_diagnostic_closure import close_local_diagnostic

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    inputs = json.loads((here / "inputs.json").read_text())
    train, dev = inputs["train_source_idx"], inputs["development_source_idx"]
    print(close_local_diagnostic(REPO_ROOT, here,
        evidence_id="pcqm-k1-late-weight-average-20261008", policy_id="pcqm-k1-late-weight-average",
        role_rows={"train_features": {"labels_read": train, "prediction_input": train},
                   "internal_development": {a:dev for a in ("labels_read", "prediction_input", "metric_computed", "selection_used")}}))
