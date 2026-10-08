# Authorized account migration - 2026-10-09

The desktop user explicitly requested submission to Kaggle3 after Kaggle1
exhausted its weekly allocation. Continue the same two-arm question from the
[accepted version2 stage](submission_v2/terminal_inspection_20261009/stage_acceptance.md):
46/60 completed epochs, 179,676 optimizer updates and 22,998,528 sample
presentations per arm. Do not restart, reset a schedule or retrain completed work.
Desktop ownership and the original prospective trajectories remain unchanged.

This additive authorization changes account/publication plumbing only. The
original Spec `65fc71b0a622a8661df33b061915e0630dfb57abf5e01546cfc1803489f23c12`,
recipes, initialization, roles, optimizer, permutation, FP32/noTF32 and
60-epoch learning-rate schedule remain frozen. Both coefficient0 and0.1 arms
still perform two dropout forwards. Preserve prior stage manifests and
prospective records byte-for-byte, and bind the new executable package and
physical attempt separately. Never rewrite the historical Kaggle1 attempts.

## Release boundaries

- Account: Kaggle3, `nvoid912`; requested kernel
  `nvoid912/molgap-k1-consistency-500k-pair-s42-v1`.
- Private graph mirror: `nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1`;
  [accepted mirror](../../platforms/_records/kaggle/pcqm_fixed_datasets_kaggle3_v1/500k/acceptance.json)
  retains manifest SHA256
  `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
- Fresh private source: `nvoid912/molgap-k1-consistency-500k-source-s42-k3-v1`.
- Fresh private recovery: `nvoid912/molgap-k1-consistency-500k-resume-s42-stage46-v1`.
- Explicit T4x2, one arm per device; both CPU and native optimizer preflights
  must pass before training. Retain the accepted Torch2.4.1/cu121 runtime and
  installed-distribution identity. A mismatch stops release; it never grants
  permission to change the recipe, drop resume state or reset initialization.

Reuse `pcqm_500k_preparation.prepare_continuation`, the existing bootstrap,
family trainer, release checks and explicit accelerator submitter. The smallest
extension is explicit account/mirror/platform identity in the existing
continuation adapter, not a second training or submission framework.

## Budget and expected duration

Version1+2 measured training allocation is34.84185189143666 T4 device-hours;
17.15814810856334 remains below the unchanged52-hour ceiling. Bound this
invocation to30,000seconds: even a full paired allocation is16.666666667 T4
device-hours. Bootstrap/setup and native qualification remain separately
accounted; do not add overlapping allocation windows. Unknown queue or billing
cost remains unknown. Native measured costs belong to the actual attempt.

Fourteen epochs remain. Extrapolating version2 measured epoch durations gives
about5.3hours paired wall time and10.6 T4 device-hours for the remaining training.
This is an estimate, not measured migration cost or a completion guarantee.
Any partial stage requires another authoritative reconciliation and budget
review; no automatic retry, continuous monitoring or account fallback is added.

At completion, perform the original raw and identical selected-state clean-BN
comparisons and independent per-arm terminal/RML acceptance under
[protocol](protocol.md). This submission does not release full-data training,
protected roles, model promotion or a server takeover. Physical response,
publication, source, scheduler and startup records belong in
`submission_kaggle3_v1/`; missing IDs and qualifications stay pending.
