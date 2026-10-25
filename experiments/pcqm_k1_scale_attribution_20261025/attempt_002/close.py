"""Close the retained CPU matrix through the existing diagnostic/RML owner."""
from pathlib import Path
import json

from molgap.frozen_diagnostic_closure import close_local_diagnostic


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    inputs = json.loads((here / 'inputs.json').read_text(encoding='utf-8'))
    rows = {
        'train_descriptive': {key: inputs['train_source_idx'] for key in ('labels_read', 'prediction_input', 'metric_computed')},
        'internal_development': {key: inputs['development_source_idx'] for key in ('labels_read', 'prediction_input', 'metric_computed', 'selection_used')},
    }
    print(json.dumps(close_local_diagnostic(root, here,
        evidence_id='pcqm-k1-scale-attribution-20261025-attempt002',
        policy_id='pcqm-k1-scale-attribution-20261025-attempt-002', role_rows=rows)))
