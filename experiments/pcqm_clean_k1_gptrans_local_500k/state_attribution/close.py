"""Use existing metadata-only diagnostic acceptance and RML finalization."""
import json
from pathlib import Path

from molgap.frozen_diagnostic_closure import close_local_diagnostic
from molgap.training_reproducibility import atomic_json


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    root = next(p for p in here.parents if (p / "AGENTS.md").is_file())
    inputs = json.loads((here / "inputs.json").read_text(encoding="utf-8"))
    train, dev = inputs["train_source_idx"], inputs["development_source_idx"]
    calibration = sorted({i for rows in inputs["calibration_source_idx"].values() for i in rows})
    roles = {"train_decoded": {"labels_read": list(range(500000))},
             "train_feature_calibration": {"prediction_input": calibration},
             "train_descriptive": {a: train for a in ("prediction_input", "metric_computed")},
             "internal_development": {a: dev for a in ("labels_read", "prediction_input", "metric_computed")}}
    result = close_local_diagnostic(root, here, evidence_id="pcqm-k1-session-state-attribution-20261015",
        policy_id="pcqm-k1-session-state-attribution", role_rows=roles)
    atomic_json(here / "finalization_receipt.json", result)
    print(json.dumps(result))
