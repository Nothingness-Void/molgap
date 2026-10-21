# Frozen extended-exposure normalization protocol

User authorization on 2026-10-21 JST covers Kaggle1 two slots/four arms to test
whether LN benefits from more exposure, with complete retained training curves.
This question does not reopen or revise the accepted 40-pass negative decision.

## Four arms

| Pair | Arm | Local norm | Peak LR | Schedule | Passes |
|---|---|---|---:|---|---:|
| norm | bn_long | Original Post-BN | 0.0004 | cosine160 | 160 |
| norm | ln_long | Post-LN | 0.0004 | cosine160 | 160 |
| optimization | ln_low_lr | Post-LN | 0.0002 | cosine160 | 160 |
| optimization | ln_low_lr_warmup | Post-LN | 0.0002 | warmup8 + cosine160 | 160 |

The norm pair changes only the 18 local post-residual normalization operators.
The optimization pair changes only warmup/schedule under the same lower peak LR.
Its reference is an explicitly new LN recipe, not a repeated historical baseline.
Between-notebook LR comparisons are exploratory, with job and training-seed
limits; do not silently treat them as same-job randomized causal contrasts.

All arms bind the same transported seed42 base state, fixed100K train membership,
next50K internal development membership, FP32/no TF32, batch128/drop-last,
single-forward supervised normalized Gap L1, AdamW WD0.00001, clipping1.0,
dropout0.05, unchanged K1 architecture/pooling and EMA0.999 tracking. Select
best-development Live; no clean-BN or EMA selection intervention is included.
Each arm performs 124960 updates and 15994880 gradient sample presentations.
No new data diversity is introduced by repeated exposure.

## Endpoints and curves

Retain every completed epoch's online training MAE, selected-state development
MAE, actual last-update LR, optimizer steps, sample presentations, durable
checkpoint SHA and measured process wall. Use standard epoch-only TRAIN_LOG.
Online train and eval-mode development metrics are not a matched-state
generalization gap and cannot prove overfitting on their own.

Predeclare curve summaries at40/80/120/160 passes (3998720/7997440/11996160/
15994880 presentations). Report both contemporaneous development differences
and best-through-boundary endpoints. Late catch-up means the LN-minus-BN gap
shrinks after40 under their common long schedule; an isolated crossing is not
material superiority. Report the whole curve even when the final gate fails.
Do not invent early stopping from these curves or stop an arm on a bad endpoint.

The primary final gate remains candidate gain >=0.003 eV against its same-job
reference, with aligned finite50K selected predictions and paired row interval.
Warmup comparison is a separate LN recipe question. Row uncertainty is not
training-seed variance. This single-seed screening releases no automatic500K,
full training, model adoption or protected-role access.

Both exposure and cosine horizon differ from the historical40-pass recipe.
Historical40-pass points are context, not a pure exposure-only intervention.
The same-job long-horizon pair answers relative late behavior under its frozen
schedule; it cannot independently identify exposure versus annealing causality.

## Resources, durability and acceptance

Two private Kaggle1 T4x2 notebooks; one arm per T4. Expected per notebook about
6-7 measured-window wall hours /12-14 T4-device hours, estimated from the accepted
40-pass run. Platform maximum12 wall hours /24 T4-device hours per notebook;
the authorized native upper bound is24 T4-device hours per pair,48 total.
GPU-account quota reservations are separate from measured native device cost.
No added local watchdog, automatic retry, monitor, successor or shutdown.

Reuse all-arm native preflight, exact source/config/init/input qualification,
atomic best/last checkpoints and existing10-pass resumable chunks. Interrupted
work remains partial and must be reconciled before any authorized continuation.
Do not overwrite a frozen release or move custody to server.

Official validation/test-dev/test-challenge remain sealed. Per arm retain native
certificate, complete160-row trace, selected model, finite aligned predictions,
last resumable state, exact role/cost records and independent terminal RML.
Strict replay qualification is decided by existing validators after completion,
not promised by submission. Retrieve only an explicit final acceptance allowlist
and minimal diagnostics to D:, with size estimates; never whole output trees,
epoch chunks, graph caches or large formal profiler traces by default.
