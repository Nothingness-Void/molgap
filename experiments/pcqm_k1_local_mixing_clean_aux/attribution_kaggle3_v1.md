# Kaggle3 runtime qualification attribution, 2026-10-01

Outcome: NO_TRAIN, runtime cost gate failed. Both arms have zero formal epochs,
optimizer steps and sample presentations. Calibration optimizer steps are distinct.

On the same candidate-assigned Tesla T4 and same128-graph train fixture,
original K1 median optimizer-inclusive step was0.113287198s and SSMA0.145645696s:
28.563243% overhead, above frozen25%. Five measured samples follow two warmup
steps. Additional capacity/joint computation accompanies this overhead; no
operator-level profiler isolates which operation contributes how much.

Both modes passed zero-added output equivalence, deterministic repeated optimizer
steps, model/optimizer/scheduler/RNG resume and selected Gap state roundtrips.
The reference runtime certificate was accepted. Candidate cost qualification
failed; the pair-wide barrier correctly prevented both formal training workers.

This is not an accuracy negative. There are no development MAE endpoints or
learning curves, so underfitting, overfitting, insufficient exposure and module
harm are insufficient_evidence. The missing discriminator is a trained accuracy
comparison under an admissible runtime contract. No threshold relaxation,
architecture substitution, successor or scale-up is released by this record.

Only minimal logs/runtime/cost JSON were retrieved; diagnostic checkpoint binaries
remain remotely retained. Worker diagnostic allocations measure10.363722364s
(reference) and11.977338187s (SSMA); their22.341060551 allocated T4 seconds are
a lower-bound diagnostic window, excluding bootstrap/data-load/queue. Pair
orchestration wall was54.03020802s. CPU and queue allocations remain missing.
