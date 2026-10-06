# Frozen bottleneck interpretation

On 2026-10-06 Kaggle2 kernel `kaseichou/molgap-gptrans-bottleneck-audit`,
ID137289711/v1, completed. The controller claimed event
`evt-391295519aee368f28db43ee`, pinned the exact-version output manifest,
retrieved its45 hash-bound files, and independently accepted saved tensors.
The directory-listing transport encountered429; version-specific exact-file
transport recovered the existing outputs without a new remote job.

## Evidence and limits

The source/release, six retained EMA states, strict loaders, fixed role inputs,
native runtime, unchanged-state proofs and complete output inventory passed.
Local-best reproduced50000 original-development predictions with maximum
absolute difference9.5367431640625e-7eV and MAE difference4.816055332312885e-10eV.
There were no optimizer steps, local model execution or protected-role reads.

Native observation covered117.04911984 wall seconds and234.09823968 allocated
T4-device seconds (approximately0.065T4-hour), including setup and idle devices.
Provisioning before Python, queue and platform teardown/billing were unobserved;
this is not a complete billed-cost claim.

Full numerical results live in [acceptance_summary.json](results/acceptance_summary.json)
and [mechanism_interpretation.json](results/mechanism_interpretation.json).

## Reachability

All12 local-bond output branches received nonzero final Gap-L1 parameter and
return gradients in all4 batches at both retained epochs19 and59. Local-bond
failure was therefore not explained by a disconnected branch on this panel.

Pair Transition branches3/6/9 received gradients in all4 batches; branch12
received none. Its real-pair update ratio was also zero. These observations
support the independently documented structural final-readout disconnection.
Unwrapped reference blocks had no addon: their zero addon statistics do not
mean that the GPTrans core had no gradient. The experiment did not repair or
retrain the transition model, and did not establish that fixing only its final
branch would improve MAE.

## Local-update strength

For local-bond layers4–12, mean per-molecule update/input RMS ratio grew from
0.1075882469 to0.1562725104 (about45%). Layer6 grew0.091548→0.154959;
layer12 grew0.150504→0.194405. The shallowest update was already large:
layer1 grew1.135040→1.212191. All inspected values were finite.

This supports studying local/global amplitude balance, not a claim of numerical
explosion or proven overfitting causality. The frozen input-only degree bins did
not show uniformly larger updates for higher-degree molecules across layers.
Maximum-degree<=2 had only7 of512 rows;<=10 atoms had60,11–20 atoms452, and
>20 atoms zero. There was no evidence here for a simple high-degree or large-
molecule diagnosis. These are eval-mode EMA derivatives, not historical
train-mode gradients, clipping frequencies or a causal strength intervention.

## Frozen cohort portability

| Retained role | Rows | Reference MAE | Local MAE | Local minus reference |
|---|---:|---:|---:|---:|
| Original100K internal development | 50000 | 0.1442326291 | 0.1423582280 | -0.0018744012 |
| Fixed500K internal development, frozen random subset | 10000 | 0.1497222179 | 0.1470576105 | -0.0026646073 |

The later paired-row95% interval was[-0.0047395022,-0.0006282118]eV; the
local model won on51.31% of later rows. Its advantage did not disappear on
this cohort before any500K optimization. The alternative explanation of an
immediate cohort-only reversal was not supported for this local addon.

Both roles were reused internal development. Row bootstrap does not measure
training-seed variation. Post-hoc reference-error quintiles were descriptive,
target-derived and subject to regression-to-the-mean; they cannot establish a
deployable hard-molecule router. This result did not change the original
100K promotion gate or qualify500K/full training.

## Decision

The diagnostic closed as **NO_TRAIN / PAIRED_ENDPOINT**, with mechanism
observations **CONTEXT_ONLY**. It did not create a training Replay pair.
The local-bond mechanism remained a plausible bounded research lead, rather
than a cohort-failed mechanism. A separately authorized single-mechanism
local-update strength/control comparison was more informative than another
width/FFN/readout expansion. Its benefit remained unproven; no new experiment,
extra seed, corrected Pair Transition run or automatic successor was released.
