# K1 500K frozen verification — 2026-10-07

## Outcome

NO_TRAIN: accepted completed local diagnostic, no model promotion. This records
measured slot dependency and clean selected/final behavior; capacity saturation
and expanded-pretraining benefit remain insufficient_evidence.

## What was tested

The selected epoch49 and final epoch60 checkpoints were strictly loaded using
the exact accepted packaged model source. The4096 fixed sample includes1024
original100K training-prefix rows,1024 extension-training rows,2048 previously
selected development rows. Parameters/buffers stayed unchanged. Selected saved
predictions reconstruct within1.90734863e-06 eV.
No optimization, GPU, official validation/test or new geometry was used.

## Frozen dependency result

| Development condition | Sample MAE (eV) | Increase over selected (meV) |
|---|---:|---:|
| Original selected predictor |0.109739250|0|
| Zero layer9 slot return |0.223893490|114.154|
| Zero layer3/6/9 slot returns |1.281661228|1171.922|

Layer9 deletion's paired-row95% interval for increased error is104.822–124.072
meV. These effects describe deletion from a co-adapted frozen model; they are
not training gains attributable to the slot, or proof that extra slots help.
The inactive/negligible-slot explanation is contradicted on these rows.
The2048-row sample MAE differs from the accepted full50K MAE0.104904085 eV;
never substitute the sample endpoint for its original selection result.

## Clean late-training behavior

| Fixed cohort | Rows | Epoch49 clean MAE | Epoch60 clean MAE | Final minus selected (meV),95% row interval |
|---|---:|---:|---:|---|
| Original100K train prefix |1024|0.067256420|0.069232365|+1.976 [0.674,3.331]|
| Extension100K:500K train |1024|0.056165132|0.055438370|-0.727 [-1.745,0.306]|
| Consumed development sample |2048|0.109739250|0.110107738|+0.368 [-0.345,1.089]|

The combined raw training sample is50/50 stratified and not representative of
the20/80 member proportions. Its simple pooled MAE is not a500K population
estimate. Weighting the two strata20/80 gives selected0.058383390 and final
0.058197169 eV, a-0.186 meV descriptive point change;
no population confidence interval was computed. There is no uniformly useful
clean-fit improvement in the tail. The accepted full development trace also
retains11 epochs without a new best. This does not identify underfitting,
overfitting, forgetting, BatchNorm drift or learning-rate sufficiency.

## Pretraining coverage

The retained provenance links the accepted100K Stage10 backbone and reset
original head to the exact500K initialization bytes. Original pretraining
members[0,100000) are20% of downstream[0,500000). Its7810 updates/999680 sample
presentations are3.3325% of downstream exposure. Identity and coverage are
confirmed; isolated pretraining benefit is not. The extension training sample
has lower clean error than the original prefix, so a simple account that the
unpretrained extension is uniformly harder to fit is unsupported here.
Different positional cohorts have different molecule/target distributions;
this is not a controlled pretraining exposure effect.

## Decision consequence and missing discriminator

Preserve the existing selected checkpoint and fixed-fusion result. No longer
training or new model is released by this diagnostic. A future capacity test
must change independent relation channels while holding the existing node/edge/
slot widths and training recipe fixed. A future pretraining test must isolate
its membership/exposure budget. Neither question is answered by this deletion
probe, and neither is automatically submitted.

Worker wall22.469s, process CPU
62.609s; accelerator not applicable.
Preparation/hashing/tests/publication/Git overhead is outside these timers.
Previously consumed internal-development results are exploratory; row intervals
are not seed variance or independent validation. No training trace applies.
