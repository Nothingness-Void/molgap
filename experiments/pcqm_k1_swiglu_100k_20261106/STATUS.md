# Status — 2026-11-06

## Terminal inspection

Luna observed job142921674/version1 ERROR at13:46:57-13:47:01 JST, latest and
receipt versions both1. Preflight exited0. Formal training completed10passes,
7810updates/999680presentations, best retained development0.17512261867523193eV.
The next development evaluation failed its count/order/finite guard; see
decision.md for INCONCLUSIVE disposition, actual cost and attribution.

Kernel/train logs, partial trace, execution/allocation records, role provenance
and pass10 resume metadata were selectively retained under
`D:/molgap-artifacts/reconciliation/k1-swiglu-100k-20261106-failure/`.
No tensor payloads or unselected outputs were downloaded for this failed run.
The pass10 checkpoint remains bound by its exact-version remote resume bundle.
Owner branch stays retained; no recovery training or new run was launched.

Existing build_incomplete_terminal_descriptor generated
closure_failure_v1/terminal_descriptor.json with the actual10/7810/999680
counters. Canonical RML finalization remains pending the reported infrastructure
gap: failed partial outputs lack the output_manifest required by existing
terminal-package and role-audit staging. The descriptor itself does not finalize
RML. The infrastructure owner received exact paths and observed failures; no
hand-authored V5/role/cost framework was introduced here.

## Submission history

User authorized execution of the reviewed design. Exactly two scientific arms:
retained qualified original K1 reference A and new SwiGLU candidate B. Only B is
newly trained. Kaggle3 (`nvoid912`) selected; no reference retraining. Physical
T4x2 allocation retains the platform's resource shape; this is not two new workers.

Necessary local qualification and one fixed100K40-pass run are released within
the8-device-hour/4h allocation bound in experiment.json. Prospective and source
freeze completed; the corrected existing sandbox passed. No scientific benefit
or replay-readiness claim is made.

Kaggle3 job142921674/version1:
`nvoid912/molgap-k1-swiglu-100k-20261106-s42`.
Submission bound successfully; Luna's one-time latest-version check at
2026-11-06 02:18:39–02:18:43 JST found RUNNING/version1 matching the receipt.
This is scheduler state only: native T4 qualification and training entry remain
unconfirmed. No persistent monitor or queue trigger was installed.

Run `pcqm-k1-swiglu-100k-20261106-s42`, attempt `v1`;
frozen scientific source `b43b7fc70320caf4ca588620d76b7f0ea2a10f70`;
Spec `af3b8b526cd10056375afc0e59b878c780599c709b06bdedb64cf7ce62b7f2fe`;
package `acc69d83e9d1b15f7c0149506d4c8c881f77833ae7ef4bd38c0c8cf4488fb96d`.
Local verifier repair `c4da3a75` preserves those frozen materials.

Durable receipts:
`D:/molgap-artifacts/runs/pcqm-k1-swiglu-100k-20261106-s42/v1/operations/launch_receipt.json`
and `submission_state.json` beside it. Prepared package, release reports and
immutable inputs are under that run's `prepared/` and `inputs/`.
See [execution qualification](execution_qualification.md) for actual checks and
the original sandbox error. Reconcile this exact job before any subsequent action.
