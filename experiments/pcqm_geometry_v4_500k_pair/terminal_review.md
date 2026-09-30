# Terminal acceptance review, 2026-09-30

Subsequent reconciliation: `fixed_blend_scale_decision.md` resolves the
missing-K1-prediction observation below and owns the direct blend-versus-blend
scale review. The original immutable transactions remain preserved.

The authoritative Kaggle1 scheduler reports COMPLETE for
`nothingnessvoid/molgap-geometry-v4-500k-resume-s42-v1`. Both independently
rerun `accept_stage.py` checks passed at epoch cursor 60. Each observed
234,360 optimizer steps and 29,998,080 sample presentations. Selected model,
prediction, initial-state, checkpoint, runtime, calibration and trace hashes
passed the owning adapter. This review executes no model inference or training.
The immutable terminal transactions remain in the continuation directories;
the dated terminal attribution is `continuation_terminal_decision.md`.

## Scientific result

The frozen single-arm point gates are decidable: GPTrans distance-only improves
0.857 meV over its pure-2D reference, below the required 3 meV. K1 distance-angle
is 0.0033 meV worse than its reference point MAE and also misses the gate.
Neither single-arm geometry implementation qualifies for adoption under this
contract. Missing K1 reference predictions prevents its paired uncertainty
analysis; it does not turn the known failed point threshold into an unknown.

The accepted GPTrans reference prediction bytes were independently checked
against SHA256 `0608084eb2a7a90923a652235572ee69578c4e138d5df3c457e5170f7da613bf`.
Its 50,000 indices and targets align exactly with both candidates. Reusing
`analyze_local_ablation.py` gives a GPTrans improvement interval of
[0.341, 1.377] meV from 10,000 paired row-bootstrap draws. The smaller positive
effect is supported; the materiality gate fails.

The predeclared fixed 50:50 blend has MAE 0.100290090 eV and improves
4.573 meV over the better candidate. An independent rerun using the owning
bootstrap helper gives an improvement interval [4.221, 4.936] meV. Its minor
numerical difference from `continuation_pair_analysis.json` reflects a
different deterministic sampling implementation, not different prediction
bytes. This supports the protocol's blend nomination only. No protected role,
full-scale release, or automatic adoption follows from this result.

## Attribution

Both runs completed the exact exposure, ruling out an interrupted or
five-epoch result as the reason for the final scores. K1 selected epoch 48;
from epoch 51 to 60 its live training-stream MAE declined from 0.076234 to
0.074407 eV while development MAE changed from 0.104866 to 0.105432 eV.
This is compatible with diminishing development returns under the frozen
schedule. Training-stream metrics are not fixed-cohort endpoint evaluations,
so these numbers do not prove overfitting or that geometry itself is harmful.
GPTrans selected epoch 53; its final development MAE is 0.106162 eV versus
selected 0.106010 eV. More training is not released by these curves.

The candidates' signed residual correlation is 0.8485; GPTrans has lower
absolute error on 49.078% of rows. Their fixed blend improves over both,
supporting complementary errors despite similar individual endpoint quality.
Attribution of that complementarity specifically to geometry needs the
corresponding pure-2D pair comparison; architecture and single-seed variation
remain compatible explanations. Oracle and fitted-crossfit quantities in
`terminal_review.json` are post-hoc diagnostics and are non-promotional.

## Record reconciliation and exact blockers

Measured cumulative T4 costs are 12.751797 h for GPTrans and 20.289507 h for
K1, below the separately authorized 16 h and 26 h ceilings. The initial
GPTrans manifest measures 3,954.324647338 seconds. Subtracting this from the
terminal 45,906.469968691 seconds gives 11.653373700376 continuation hours.
The published continuation cost event records 11.653643489299 hours, an
overstatement of 0.971240122 seconds. The correct measurement is retained in
`terminal_review.json`; the immutable finalization is not rewritten. The
existing lifecycle exposes no amendment transaction, so its compiler ledger
still contains the original value and needs a supported correction path.

The accepted K1 reference predictions (SHA256
`68fba785a0b028a8445fe8d94d348df2c3a8d7d8caab708acb574d13cac9831b`)
are absent from the primary checkout and the local Codex worktree inventory.
The latest output of its original physical kernel contains only the later
EdgeState arm. The installed Kaggle output SDK has no version selector; this
latest-output observation does not establish loss of historical version 6.
Recover those exact accepted bytes from its durable historical output or
backup. Retraining a reference is not an authorized substitute.

Both RML terminal records are present and schema validation and frozen-derived
checking pass. Both replay manifests currently contain explicit exclusions;
their missing strict qualification cannot be repaired by manually setting
eligibility. The blanket exclusions also conflate pair review with independent
arm qualification: blend nomination or a failed scientific point gate alone
is not a reason to reject an otherwise valid replay record. Independent
reference/readiness binding requires review through the supported lifecycle.
The global portable check additionally fails for unrelated historical missing
artifacts; it is not reported as passing.

Retain this unresolved mixed-result experiment on its owning branch. The
verified result is two complete mechanically accepted runs, two terminal RML
records, failed individual materiality gates, and positive fixed-blend
nomination. Two complete replay-pool entries have not been achieved.
