# Accepted terminal state

Kaggle3 `nvoid912/molgap-k1-fusion-distill-100k-s42-v1`, ID137029945/version2,
is COMPLETE: [authoritative observation](terminal_remote_observation_v2.json).
Weak retains the version1 completed arm; strong continued only epoch40 from the
original epoch39 checkpoint. Both pass40-epoch mechanical acceptance at31,240
steps and3,998,720 presentations each. [Continuation verification](terminal_continuation_verification.json)
uses the existing validator and preserves the exact39-epoch prefix and source identity.

Both are NEGATIVE_UNDER_CONTRACT under the frozen compression nomination gate.
Weak MAE0.141406777eV is2.678meV worse than consistency2. Strong MAE0.138295709eV
improves by0.433meV, with paired-row95% interval[-0.454,+1.390]meV, and remains
3.630meV worse than the fixed50:50 teacher. See [decision](terminal_acceptance/decision.md),
[scientific metrics](terminal_acceptance/scientific_metrics.json), and [attribution](terminal_acceptance/attribution.md).
No adoption, new cohort execution, scale-up or blind seed/schedule retry is released.

Each original prospective is retained unchanged; two independent terminal/RML
publications and full canonical traces are verified in [closure receipt](terminal_acceptance/closure_receipt.json).
Weak: [finalization](kaggle3_v1/distill_weak/rml_finalized/finalization.json).
Strong: [finalization](kaggle3_v1/distill_strong/rml_finalized/finalization.json).
The teacher-cache prerequisite is separately NO_TRAIN in the full owning history.

Strict replay/READY is excluded. Historical strict reference/runtime equivalence
is not established; weak's raw runtime manifest was not retained. Strong's
original1-39epoch cumulative allocated-device ledger is missing. Version2
qualification and137.823T4-second recovery invocation are measured separately;
that increment is not a40-epoch cost. Both retained certificates and calibration
hashes verify; strong's full preflight metadata passes its owning validator.
Official validation/test roles remain untouched.

Full rejected history routes to archive; only accepted student canonical evidence
and its directly required supporting records enter desktop. Rejected implementation
and the complete teacher-cache generation history stay in archive. Source commit
932d107831f22617181a0139b597252512f4ff75, original package, Spec, teacher and recipe
remain unchanged. Shared infrastructure repair26a1bab2 was already separately
integrated on desktop as749a91c2; no new model code is adopted at closure.

Archive custody is now verified and pushed: owner97d667f4 is reachable from
archiveaac3f34c. Desktop accepted evidence is bound in[custody](terminal_custody_decision.md).
