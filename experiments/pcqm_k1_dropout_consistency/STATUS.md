# K1 dropout consistency desktop handoff

## Latest reconciliation

Kaggle3 ID136942701/version3 is authenticated COMPLETE on2026-10-04.
Both arms now meet40epochs/31240updates/3998720presentations. Control epoch40
did not refresh its best epoch37 checkpoint. Exact retained-prediction MAE:
mean2=0.1402572855657339eV; consistency2=0.13872850305497647eV.
Primary gain1.5287825meV, paired-row95%interval[0.6161362,2.3489377]meV,
falls below the frozen3meV material gate. Historical clean-K1 gain is2.5659578meV.
Read [result check and attribution](recovery_reconciliation_v3/analysis.md).
The target digest encoding mismatch still blocks mechanical acceptance; resumed
control total native T4 cost is missing. Scientific acceptance is not finalized;
dual replay readiness remains unqualified. No scale-up or successor was submitted.
The original records below retain the dated sequence, not current queue state.

## Previous attempts and frozen submission

Owner codex/exp/k1-dropout-consistency; verified desktop base a779e8cf.
[Kernel](https://www.kaggle.com/code/nvoid912/molgap-k1-dropout-consistency-100k-s42-v1)
actual ID136942701/version1 is ERROR on the authenticated 2026-10-04 recheck.
Submission identity remains in launch_receipt.json; retained failure evidence is
in failure_reconciliation_v1/remote_observation.json and pair_state.json.

Both preflights passed. consistency2 finished40epochs/31240updates;
mean2 retained39epochs/30459updates before orchestration termination. The
family inspector rejected the historical float32 target digest because it
computes a float64 digest. Both retained ordered target tensors reproduce the
original historical digest exactly. See target_encoding_reconciliation.json
and dropout_consistency2/mechanical_inspection.json in failure_reconciliation_v1.
This is an acceptance encoding defect, not an observed numerical/model crash.

Retained prediction MAE: consistency2=0.13872850305497647eV;
mean2=0.1402572855657339eV. Difference1.5287825meV is descriptive and below the
3meV gate; the matched40epoch contract and mechanical acceptance are incomplete.
The user authorized recovery of only mean2 epoch40; see recovery_authority.md
and resume_binding_v2.json. No candidate retraining or scale-up is released.
The frozen validator remains BLOCKED; no dual replay-ready claim is made.

Kaggle3 ID136942701/version2 failed before formal training at the runtime identity
gate. The recovery adapter incorrectly ran qualification and training in one
process; it now reuses separate standard workers for both phases. Preserve the
failed attempt in recovery_reconciliation_v2; no added epoch was acknowledged.
ID136942701/version3 is authenticated RUNNING; only mean2 resumes and consistency
outputs are custody-only inputs. Read recovery_submit_response_v3.json and
recovery_remote_observation_v3.json for actual returned identity and pulled entry.
The original source archive is unchanged.
Seventeen required published source/resume files were independently downloaded
and hash-verified; four focused synthetic recovery cases passed. See
recovery_source_verification_v2.json, recovery_verification_v2.json and the
focused isolation-repair checks in recovery_verification_v3.json.

Two NEW100K40epoch arms: dropout_mean2 (matched two-pass control) and
 dropout_consistency2 (same passes plus0.1 normalized output disagreement).
Original clean K1 is retained for contextual comparison, never retrained.
[Protocol](protocol.md) owns3meV/sign gates, roles, exposure and resource limits.
Frozen source84288398cc45c54eee80493e74c669fdd9418a65;
Spec8edc7cc11717eb00b6bb961e9164db0c081fb01b16e6ccd3862a4a5e0d0b8a02;
package110d86caf22f9e60e9bc6b6426357b1149734d870961d0eae7cd284021f2fc4a.

Standard prepare-workflow and submission release checks passed. Private source
publication was independently downloaded and all11 required file hashes/mount
paths verified. Synthetic focused verification:240 tests, then1 targeted
train-only-loader test after that narrow change; no local molecular diagnostic.
The existing runtime requires BOTH assigned T4 preflights to pass before training.
Runtime qualification and retained phase/MAE evidence are now observed; scientific
acceptance and replay readiness remain pending. Consistency's measured assigned
T4 training window was9180.524511809seconds (2.5501457T4h), excluding bootstrap
and queue; incomplete-control total costs remain unresolved.

No heartbeat,server takeover,automatic retry,scale-up or protected evaluation.
User requested shutdown after verified submission. When desktop returns, query
this exact attempt, verify frozen identities, retrieve only per-arm manifest-bound
replay-required outputs and minimal diagnostics. Reuse accept-workflow/finalizer;
accept and attribute independently, then classify same-run mechanism contribution.
Require complete strict V5/runtime/cost/role/trace evidence before replay claims.
