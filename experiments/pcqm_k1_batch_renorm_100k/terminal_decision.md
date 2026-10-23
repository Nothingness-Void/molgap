# Terminal training decision — 2026-10-24 JST

Kaggle1 kernel140432543/version1 completed the frozen40pass pair. Source888ff7cc,
package6b56369539f267c25bf93ba6fa47c11c3e4281c6d9e9f1e11ff1a97a37f2c1ab.
The retained exact-version source, mounts, native repeated/serialized-resume
proofs and mechanical outputs were independently verified before this decision.

| Arm | Selected Live MAE (eV) | Selected pass/step | Terminal exposure | Disposition |
|---|---:|---|---|---|
| brn_scale |0.14169638845860957|39 /30459|40passes,31240updates,3998720presentations|CLOSED|
| brn_affine |0.14334189840257167|39 /30459|40passes,31240updates,3998720presentations|NEGATIVE_UNDER_CONTRACT|

Candidate-minus-reference MAE is+1.645510meV; paired row bootstrap95% interval
[+0.906720,+2.400196]meV (50000 aligned finite rows,2000draws,seed42).
The predeclared requirement of at least3meV improvement and negative upper
confidence bound fails. This closes additional mean correction under this
recipe. It does not compare either arm causally against unchanged BN, measure
training-seed variance, reject every BatchRenorm recipe or authorize new training.

Measured native allocation window:5577.032911376seconds on two TeslaT4 devices,
11154.065822752device-seconds total. Queue and pre-Python provisioning are unknown.
Both arms completed without a physical continuation; startup resume proof is
not evidence that a correction-active remote restart actually occurred.

The separately predeclared selected-Live Clean-BN diagnostic remains required
for attribution. It cannot change this training decision or select a posthoc
promotion endpoint. No adoption, retry, continuation,500K or full release.

Authority: protocol.md; exact artifacts and shared attribution report at
D:/molgap-artifacts/reconciliation/k1-batch-renorm-100k/diagnostic.json.
Strict terminal RML publication is a separate operation and must be verified.
