# Frozen K1 slot/readout diagnostic

## Outcome

NO_TRAIN. Retain the accepted width192 recommendation. The simple explanation
that K1's final global return is numerically negligible after mean pooling is
not supported by the measured states. No new training or official-role access.

## Accepted measurements

The two strict-loaded selected checkpoints reproduce retained T4 predictions on
2048 uniform development rows within 2.3842e-6 eV (192) and 1.4305e-6 eV (256).
Read-only hooks leave the forward residual unchanged. The measured single-slot
mean-return identity has maximum component error below 9.54e-7.

| Mixer layer | 192 median update / preceding atom mean norm | 256 |
| --- | ---: | ---: |
| 3 | 57.76% | 72.76% |
| 6 | 78.88% | 92.65% |
| 9 | 61.54% | 58.57% |

These are vector-norm ratios, not prediction attribution percentages. Direction,
head sensitivity, and conditional information can matter even at small norm.
The final ratio's p10-p90 ranges are 38.38%-94.86% and 39.33%-94.97%.
The final global direction therefore remains substantial in these checkpoints.

The graph sum-return norm u has log-log size slope 0.0256/0.0275 for192/256;
it does not compensate the explicit 1/N with proportional growth. But the ratio
to the preceding pooled representation varies only modestly with size:
Spearman -0.1397/-0.1352. Relative magnitude and harmful attenuation are distinct.
The final ratio vs paired error gain has Pearson0.0170/0.0119. This diagnostic
does not tie weaker slot magnitude to the width256 regression.

Final attention entropy corresponds to median effective atom counts4.325/3.567,
or31.36%/25.48% of each graph's atoms. Effective count is exp(entropy), not a
literal number of represented atoms. More concentration can be useful selection
or lost coverage; the observation cannot distinguish them.

## Full50K structure joins

All graph source indices and labels match both retained prediction payloads.
AE arithmetic here uses float64 subtraction; tiny last-bit differences from
the historical float32 paired calculation do not change its accepted verdict.

| Structure cohort | Rows | 256 minus192 MAE (meV) |
| --- | ---: | ---: |
| <=15 atoms | 34837 | +1.324 |
| 16-25 atoms | 15163 | +1.917 |
| Cycle rank0 | 4669 | +5.305 |
| Cycle rank1 | 18275 | +0.333 |
| Cycle rank2 | 23464 | +1.273 |
| Cycle rank>=3 | 3592 | +4.027 |
| Highest aromatic fraction quartile (>0.46154) | 10679 | +3.100 |
| Highest conjugated bond fraction quartile (>0.6875) | 12435 | +4.092 |

Cycle rank E-N+components counts independent graph cycles, not perceived SSSR
rings. The conjugation top quartile's paired-row interval for regression is
[2.440,5.781]meV. Its lower three groups are not consistently improved.
Overlapping posthoc groups and unadjusted intervals are exploratory. Maximum
absolute univariate Pearson correlation among the five structural summaries and
paired gain is only0.00763; size correlation is0.00376. Coarse structure is a weak
global explanation, consistent with earlier K1 residual attribution.
No rows occur in the planned26-35 or>=36 bins. This retained prefix gives no
evidence about large molecules or transfer to full training.

## Next unresolved discriminator

Before assigning extra parameters, a separately frozen last-mixer intervention
on the existing checkpoints can measure its actual output utility. If permitted,
zero only the final return, preserve layers3/6, and measure exact paired errors
and prediction shifts. This costs inference only and needs no new training.
No such intervention was executed or released in this observational contract.

For a later capacity question, preserve atom width192 and isolate slot latent
width64->96. The existing three mixer formulas imply194976 added parameters
(about5.33%, excluding any newly designed head), far below the atom256 increase
of2376384 (64.95%). This is an algebraic estimate, not a constructed/qualified
candidate or evidence that64 is the limiting dimension. It changes summary
capacity while leaving one active slot and edge state unchanged. Evidence for a
useful final slot would motivate this narrower question; no training follows here.

## Limits and cost

One seed per retained model; different learned coordinate systems and runtime
distributions. Norms cannot establish underfitting, overfitting, bandwidth
saturation, information quality or a causal reason for the accepted negative.
All observations use previously consumed internal development, no external holdout.

Launcher wall51.317s includes subprocess startup/import. Worker measured wall
10.422s inference +16.185s structure; measured worker process CPU24.297s+16.047s.
Process CPU excludes import intervals before worker timers; planning/Git/analysis
overhead outside these windows unknown. Accelerator and queue not applicable.
Raw rows, summaries, source/contract freezes and strict acceptance are retained
beside this decision; original checkpoint/negative authorities stay in their owner.
