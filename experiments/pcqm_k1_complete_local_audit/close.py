"""Bind observed component-probe memberships using the existing RML owners."""
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.frozen_diagnostic_closure import close_local_diagnostic

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    inputs = json.loads((here / "inputs.json").read_text())
    result = json.loads((here / "results/result.json").read_text())
    dev = [i for i in result["sample_source_idx"] if i >= 500000]
    train = [i for i in result["sample_source_idx"] if i < 500000]
    print(close_local_diagnostic(REPO_ROOT, here,
        evidence_id="pcqm-k1-complete-module-audit-20261008", policy_id="pcqm-k1-complete-module-audit",
        role_rows={"train_decoded": {"labels_read": inputs["train_source_idx"]},
                   "train_descriptive": {"prediction_input": train, "metric_computed": train},
                   "internal_development_decoded": {"labels_read": inputs["development_source_idx"]},
                   "internal_development": {a:dev for a in ("prediction_input", "metric_computed", "selection_used")}}))
