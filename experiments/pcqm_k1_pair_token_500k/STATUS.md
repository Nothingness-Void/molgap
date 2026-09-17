# Status

Kunshan training job `122312462` completed all 60 epochs and received V5
execution/artifact acceptance. Its separate job `122311513`
optimizer-inclusive preflight passed with exact
three-step deterministic replay and 96.6% measured memory reserve. The formal
job repeated that preflight successfully on node `e10r4n11` before beginning
epoch 0. Source commit: `9a56b65ab373e8f52af3911ce57bd5ec5d09963b`;
source archive SHA-256:
`91fb93ad949b6d75690e71dd69c0f0e50cdfcc2576f4f0b29cd8086dd69190f5`.

No official validation, test-dev, test-challenge, or shadow role was read.

The candidate improved the frozen scalar by only `0.0005561373 eV`, below the
`0.003 eV` material gate. It was scientifically closed as
`NEGATIVE_UNDER_CONTRACT`; strict paired comparison remained pending because
the exact matched60-v4 K1 reference prediction bundle was unavailable. See
`results/decision_122312462.md`.
