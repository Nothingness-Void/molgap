# K1 / SSMA 100K accuracy attempt, 2026-10-01

## Authority and unresolved question

The user explicitly authorized 100K training after reviewing the prior measured
28.563243% overhead and zero formal training. This attempt asks whether the
additional cost buys Gap accuracy. The original [25% cost-gated attempt](training_protocol_kaggle3_v1.md)
and its [NO_TRAIN attribution](attribution_kaggle3_v1.md) remain closed unchanged.
Existing RML lists both prior arms as NO_TRAIN; no MAE answers this question.

## Frozen comparison and runtime policy

Original K1 V4 reference and the identical bounded SSMA candidate each receive
100K train / disjoint 50K internal development, seed42, FP32 without TF32 or EMA,
physical batch128/drop_last, AdamW4e-4/weight_decay1e-5, clip1, cosine40/eta_min1e-6,
and 40 epochs: 31240 optimizer steps and 3998720 sample presentations per arm.
Both load the same retained initial tensors and original Python epoch sampler.
SSMA stays at layer6, latent64, degree1..4, zero residual initialization.
All official validation and test roles remain sealed. No chemical auxiliary labels.

The recipe pins `runtime_overhead_policy=report_only`: measure synchronized
optimizer-inclusive overhead and record whether the historical25% gate passes,
but it cannot veto this specifically authorized accuracy attempt. Source/data/
initialization, zero-output equivalence, determinism, finite execution,
checkpoint/RNG/resume and selected-state roundtrips still fail closed for both
arms before either starts formal training. Calibration exposure remains separate.
This permission does not relax any other experiment's cost gate or promote SSMA.

## Budget, decision and durability

One Kaggle3 T4x2 session, at most12 wall hours /24 allocated T4 device hours.
Per-arm planning estimate6 device hours/6 wall hours; CPU/queue remain unknown.
Record measured native costs independently; do not label lower bounds total cost.
The empirical step increase is not a full-job cost prediction.

Reference and candidate each need independent V5/RML acceptance and replay
qualification. Compare aligned finite50K development predictions; report raw
MAE difference, paired row-bootstrap95% interval (10000 draws, seed42), selected
epoch and full trace/exposure. >=3meV with entirely positive interval retains
the accuracy advancement criterion. Report overhead beside gain; exceeding the
historical25% threshold remains a deployment cost limitation. One seed and row
bootstrap do not establish training stochasticity; extra capacity remains an
alternative explanation for an SSMA gain.

Existing K1 trainer, FamilyOutputSession, experiment CLI and Kaggle adapter own
execution, atomic selected/resume state, immutable source, receipts and acceptance.
Actual source/job/version must be reconciled. No automatic successor,500K/full
scale-up or adoption. Write terminal attribution before selecting another module.
