# Parent review receipt - 2026-10-09

Final focused synthetic test command, from D:/w/k1-clean-fit with PYTHONPATH=src
and CUDA_VISIBLE_DEVICES=-1:

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' -m pytest tests/test_k1_clean_fit_diagnostic.py tests/test_k1_bn_calibration.py tests/test_k1_frozen_intervention.py -q
```

Observed: 34 passed, one existing PyG deprecation warning. The new metadata
tests cover clean-fit role membership, source identity and native-cost rejection,
plus NO_TRAIN closure. Their finalizer is explicitly mocked in a synthetic
temporary repository; actual closure uses the normal repository finalizer.
An earlier new test fixture omitted required helper/policy files and failed;
the fixture was corrected before the final passing run. No numerical retry.

Actual closure_receipt.json status is FINALIZED, scientific outcome NO_TRAIN.
The initial metadata custody failure remains in closure_attempt001.json.
The prospective executed helper hash is unchanged. Saved-analysis cost remains
separately measured in analysis.json, distinct from the worker cost event.

Global validation passes. Normal RML rebuild and check --frozen pass.
Portable custody inspection reports zero missing locally claimed artifacts.
Two baseline checkpoints and 133 further unique inherited assets were copied
from the parent checkout with confined paths and exact source/destination SHA
verification. The two parent_asset_custody receipts retain these transfers;
no retained model or protected-role data was loaded for this custody repair.

check --frozen --portable is not a global pass: new canonical records/policy
are absent from Git HEAD, and reviewed source/generated indexes differ from
HEAD. The explicit no-commit instruction is retained. Prediction/buffer copies
are ignored local files but must remain under the retained root at their exact
bound paths; copying the code alone is not evidence portability. External
numerical input paths in inputs.json still bind the prior owner/cache and must
remain accessible for any separately authorized reproduction. No fake strict
reference, training trace, replay authority or independent holdout is claimed.

Changed source/test files:
- src/molgap/k1_frozen_inference.py (reviewed arm-aware owner loader)
- src/molgap/k1_bn_diagnostic.py (reviewed driver/import audit reuse)
- src/molgap/k1_clean_fit_diagnostic.py (new bounded diagnostic/analysis)
- src/molgap/frozen_diagnostic_closure.py (clean-fit role format support)
- tests/test_k1_clean_fit_diagnostic.py

New canonical policy: research_memory/policies/pcqm-k1-clean-fit-generalization-500k.1.json.
Experiment wrappers, protocol, inputs, receipts, analysis, interpretation and RML
records reside in experiments/pcqm_k1_clean_fit_generalization_500k/.
Derived files changed only through the normal RML rebuild: completeness_report,
cost_ledger, policy_backtest, reference_reuse_index, research_summary (JSON/MD),
role_reuse_index, trajectory_graph and trajectory_index. No owner historical
record, main checkout, commit or push was changed.
