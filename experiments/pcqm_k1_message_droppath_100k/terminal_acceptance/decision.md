# K1 Message DropPath 100K terminal decision

Recorded 2026-10-27 JST for Kaggle1 kernel 140239597, version 1, source
commit 927783bab325dab85f311790c96b630011c21e21, seed 42.

## Contract result

The baseline reference arm is CLOSED for this paired run. The
message_droppath candidate is NEGATIVE_UNDER_CONTRACT. Both arms completed all
40 frozen passes, 31,240 optimizer steps, and 3,998,720 sample presentations.
Both selected best-development-Live weights at pass 40 / step 31,240.

On the aligned 50,000-row development cohort, baseline Gap MAE was
0.1412944608205557 eV and message_droppath MAE was 0.14366748434484006 eV.
The paired candidate-minus-baseline delta was +0.0023730235242843626 eV
(+2.373023524 meV); the 2,000-resample, seed-42 paired row-bootstrap 95% CI
was [+0.0014132885742187502, +0.0032684505624771118] eV
([+1.413288574, +3.268450562] meV). The frozen gate required a delta at or
below -0.003 eV and a 95% upper bound below zero. This result meets neither
condition.

Development was used for metric computation and best-Live selection, so this
is not an independent test estimate. The row bootstrap is conditional on this
single training seed and does not measure seed-to-seed variability. No causal
conclusion about underfitting, capacity, or a BatchNorm mechanism follows from
this endpoint. Native qualification and strict replay readiness are recorded
separately in the canonical finalization and qualification records; this file
owns the scientific disposition.

## Measured execution cost

The retained physical entry observation measured 6,571.668752029 seconds with
two Tesla T4 devices assigned, or 13,143.337504058 allocated device-seconds
(3.650927084 device-hours). Its scope runs from entry through bootstrap
completion and includes setup, package installation, extraction, preflight,
training, evaluation, and assigned idle GPU time. Queue, pre-entry allocation,
post-entry release, and device busy time were not measured.

Per-arm process wall observations were 6,044.656910304 seconds for baseline
and 6,160.241828717 seconds for message_droppath. These cover entry through
the acknowledged training endpoint and exclude terminal inspection and
publication. They are separate from the physical allocation window.

No adoption, retry, scale-up, or protected-role evaluation is released by this
decision. Physical input and observed role evidence are recorded in
physical_inputs.md and role_audits_complete.json.
