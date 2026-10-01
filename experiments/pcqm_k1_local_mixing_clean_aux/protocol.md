# Frozen preparation and release protocol, 2026-10-01

## Scientific pair

- ssma: original K1 plus one zero-initialized d64/kappa4 incoming-message
  residual at layer6. Original sums and persistent edge state are preserved.
  Degree0 adds zero; degree>4 uses only the complete original sum.
- clean_fingerprint: original K1, clean inputs, graph fingerprint BCE weight0.1,
  training-only 192->32->512 head. Gap inference removes the auxiliary head.

Both use seed42 frozen original K1 state 8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd,
FP32/noTF32, physical batch128, AdamW4e-4/weight-decay1e-5/clip1,
cosine40/eta-min1e-6, noEMA, drop_last, best live development selection.
Preserve exact Python V4 epoch order e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34:
40 epochs,31240 acknowledged optimizer steps,3998720 sample presentations.
100K training and next50K internal development only; geometry is stripped.
Official validation, test-dev and challenge are sealed.

## Preparation authority

Publish one prospective record per arm before synthetic execution. The selected
action is bounded local qualification: focused synthetic regressions and existing
reference metadata checks. No new molecular inference, training or accelerator
allocation occurs in this action. Train-only label cache is reused as bytes;
no label regeneration or protected role consumption is required.

## Release gates

No Kaggle1 push until reference, runtime and executable capability gates pass.
The planned job uses two separate visible T4 devices, two candidates and exact
logical-arm/downstream IDs prospectively bound in each trajectory, with the
actual platform ID/version in the returned receipt. Never assume an unreturned
version. Package, initial-state tensor identity, real loader, actual mounted
input layout and executable recipes require existing check-release/preflight.

New T4 optimizer-inclusive repeat calibration and restore qualification are
required. Latency gate: SSMA <=25% and fingerprint <=5% synchronized step
overhead vs the same frozen backbone; record peak memory and whole measured
native allocation separately from process wall. Qualification steps are not
training exposure and must not modify the frozen starting checkpoint.
The current trainer extraction has not yet implemented this qualification
binding or measured-native-cost hook; static code is not a release pass.

Material nomination requires >=3meV same-role gain, positive paired row interval
(10000 bootstrap draws,seed42), strict prospective comparison, complete role,
artifact, native-cost, runtime, trace and resume evidence. Row bootstrap is not
training variance. No automatic scale-up, retry, coefficient sweep or baseline
retraining. Two independent capability-complete replay entries are required
before calling the pair replay-ready. No monitor or server fallback is created.

If the reference gate fails, record NO_TRAIN for this release. That is no
scientific rejection. Recover existing authoritative artifacts/access before
reopening; do not invent a comparator or launch known ineligible evidence.

Preparation budget: <=0.05 estimated CPU hours per arm and <=0.1 estimated wall
hours per arm for the focused execution check; no GPU or queue cost applies.
Remote screening budget, if separately released after qualification: estimated
<=6T4hours per arm, <=6wallhours concurrent; estimates are not measurements.
