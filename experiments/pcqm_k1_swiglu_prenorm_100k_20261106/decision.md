# Stabilized SwiGLU100K decision — 2026-11-06 JST

Scientific outcome: NEGATIVE_UNDER_CONTRACT; do not adopt or scale this fixed
configuration. Kaggle3 job143019649/version1 completed40passes,31240updates and
3998720presentations. Best and final Live development checkpoint is pass40.

Aligned finite50000-row prediction MAE is0.1420404841130972eV versus the frozen
original K1 reference0.1412944608205557eV. Candidate-minus-reference is
+0.7460232925meV; paired2000-row-bootstrap95percent interval
[-0.3264164256,+1.7831816932]meV. The prospective improvement threshold was
3meV with upper interval below zero; it fails. The interval crosses zero,
so this is not evidence of statistically established endpoint harm or seed
variation. Internal development was selection-used.

This stabilized run completed the evaluation path that interrupted the older
unnormalized SwiGLU run. The older run remains INCONCLUSIVE; it has no comparable
40-pass endpoint,so this is not an isolated causal estimate of LayerNorm's gain.
The completed package did not improve the original K1's terminal accuracy.

The existing diagnosis shows early gains fading: delta at passes10/20/25 is
-12.606/-7.861/-1.003meV; final delta+0.746meV,last-five mean+0.662meV.
Both complete traces select pass40. These observations do not establish that
more exposure would recover a material gain; no additional training is released.
Failure-mode interpretation and limitations are in attribution.md.

Measured training process wall4926.980339015seconds; allocation wall
5234.597692252seconds and10469.195384504allocated T4-device-seconds
(2.908109829T4-device-hours),including the second idle allocated device.
Queue/provisioning-before-Python time is unknown. No causal reference speed
ratio is asserted. Frozen source/Spec/package remain those in STATUS.md.

Authority: protocol.md; retrieved manifest,canonical trace,predictions and native
proofs; existing diagnostic report and closure_v1 observed-role audit. Strict
qualification and RML finalization outcomes are retained by the standard closure
owner and summarized in STATUS.md; this decision does not substitute for them.
No new seed,width,optimizer,500K/full successor or protected-role access follows.
