# Phase 8 Scale-up Failure Analysis

## Decision

The statement "every model larger than 500K is worse" is too broad.
Several larger candidates improve common, OOD, or P8-hard MAE. However, no
candidate dominates the routed dual-GPS v4 baseline across common, OOD,
P8-hard, PCQM4Mv2, and inference cost. Therefore, none is a justified global
replacement.

The current evidence supports a **distribution-retention and representation-
complementarity failure**, not a simple parameter-count or optimization
failure:

1. Appended data progressively diluted the targeted 500K distribution.
2. Planned old:new replay was not used in the completed encoder runs.
3. Internal random-split validation improved while fixed external retention
   regressed, so internal validation became a misleading promotion signal.
4. Larger 2D encoders changed their residual relationship with the fixed 3D
   encoder. Reusing the 500K route and late-fusion topology did not preserve the
   original complementarity.
5. Wider ensembles recover some common/OOD accuracy, but lose PCQM accuracy and
   multiply inference cost.

All new training and remote submissions are paused pending review of this
analysis. The prepared PCQM Gap-specialist workload has not been submitted.

## Baseline And Protocol

- Baseline: `phase8_routed_dualgps_hybrid`, trained from the targeted 500K
  corpus.
- Common comparison: 1,977 aligned rows, including 999 OOD rows and 978
  P8-targeted-hard rows.
- PCQM comparison: 4,981 paired ETKDG-valid rows where a direct paired artifact
  exists.
- Negative delta means lower MAE than routed v4.
- Candidate promotion requires a useful accuracy/cost trade-off and no material
  regression on an external domain. A single favorable aggregate is
  insufficient.

## Direct Candidate Comparison

| Candidate | Passes | Common delta | OOD delta | P8-hard delta | PCQM Gap delta | Disposition |
|---|---:|---:|---:|---:|---:|---|
| 1M 2D+3D fusion | 3 | -0.00177 | +0.00153 | -0.00513 | +0.01300 | Better common/hard; worse OOD/PCQM |
| Repair-v2 1M 2D | 2 | -0.00098 | +0.00053 | -0.00252 | +0.02494 | No external-domain replacement |
| Additive 1.5M 2D | 2 | +0.00075 | -0.00089 | +0.00243 | unavailable | Mixed and not deployable |
| Broad 1.098M 2D | 2 | -0.00114 | +0.00261 | -0.00496 | +0.02115 | Hard specialist only |
| Coverage 2M 2D | 2 | -0.00010 | -0.00271 | +0.00256 | worse in separate 5K report | OOD specialist only |
| Two-expert 1M 2D ensemble | 4 | -0.00344 | -0.00042 | -0.00652 | +0.02016 | Common/hard gain; PCQM and cost fail |
| Three-expert 2M 2D ensemble | 6 | -0.00394 | -0.00282 | -0.00509 | worse in separate report | Best common aggregate; excessive cost and PCQM loss |
| Distilled 2M 2D | 1 | +0.00136 | -0.00253 | +0.00534 | +0.01682 | Compression failed |

The pure-2D ensembles prove that the added data contain useful signal. Their
failure is not "no learning"; it is failure to convert domain-specific gains
into one efficient, robust deployment model.

## Cause 1: Targeted-Data Dilution

The original 500K corpus is not a random subset. It is a curated 300K corpus
plus 200K targeted molecules, with substantial representation of low-gap,
large, aromatic, and flexible chemistry.

The original general 500K top-up moved sharply away from that distribution:

| Statistic | Targeted base 500K | Original general top-up 500K |
|---|---:|---:|
| Mean molecular weight | 425.21 | 331.19 |
| Gap < 3 eV | 3.261% | 0.541% |
| Gap < 4 eV | 25.357% | 10.047% |
| MW > 700 | 6.099% | 0.385% |
| Aromatic rings >= 4 | 23.78% | 4.79% |
| Rotatable bonds >= 10 | 17.44% | 3.16% |

Later targeted top-ups improved individual dimensions but did not reproduce the
joint 500K distribution. For example, repair-v2 restored molecular weight and
aromaticity but retained fewer highly flexible molecules and almost eliminated
Gap > 6 eV rows. The exact-2M mixed top-up remained less aromatic and less
flexible than the targeted base.

As the corpus grew, the targeted base represented only 50% of 1M, 33.3% of
1.5M, and 25% of 2M. This changed the effective training objective.

## Cause 2: Replay Was Specified But Not Executed

The repair-v2 sampling specification proposed old:new replay at 2:1. Completed
GPS7/GPS9 metrics record `replay_sampling: null`. Exact-2M training also records
no replay.

Consequently, the targeted 500K gradient share declined with corpus size. This
is a protocol mismatch, not merely a hyperparameter choice, and explains why
adding valid molecules could still reduce retention on the original hard
regions.

## Cause 3: Internal Validation Changed Meaning

Training dynamics do not show ordinary overfitting:

| Run | Train MAE | Validation MAE | Gap |
|---|---:|---:|---:|
| 500K GPS7 | 0.09009 | 0.11069 | 0.02060 |
| Repair 1M GPS7 | 0.09808 | 0.10419 | 0.00611 |
| Repair 1M GPS9 | 0.09650 | 0.10282 | 0.00632 |
| Exact 2M GPS7 | 0.09666 | 0.10095 | 0.00429 |
| Exact 2M GPS9 | 0.09548 | 0.10040 | 0.00492 |

Validation improved and the train-validation gap shrank as data increased.
Because each run used a random split of a different mixture, the validation
set became increasingly dominated by the new distribution. It measured fit to
the enlarged mixture, not retention of the targeted 500K capability.

Therefore, lower internal validation MAE cannot be interpreted as evidence that
the resulting model is a better replacement for routed v4.

## Cause 4: Fusion Complementarity Was Lost

Data scaling did not materially increase each GPS expert's parameter count.
GPS7 remained approximately 2.92M parameters and GPS9 approximately 3.74M.
The wider multi-2D fusion head added only 73,728 parameters, while still
compressing the combined representation to a 192-dimensional hidden state.
Parameter count alone is therefore not the primary explanation.

Three controls isolate the complementarity problem:

1. Transplanting exact-2M GPS encoders into the fixed 500K routed-v4 topology
   regressed average MAE by 0.00546 eV and Gap MAE by 0.00748 eV across three
   paired seeds. The route count barely changed.
2. On PCQM, the 1M GPS7 and GPS9 components were approximately tied with their
   500K counterparts, but the 1M dual fusion regressed by 0.01300 eV. The
   regression is concentrated in fusion calibration, not universal encoder
   degradation.
3. Combining 2M 2D embeddings with the frozen 1M 3D encoder produced worse
   validation and test MAE for all coverage, hard20k, and multi2D controls.
   The encoders were trained on different distributions, and a late head could
   not restore their former residual relationship.

The 500K model's advantage comes from its complete conditional system:
GPS7+SchNet form the broad base, while GPS9 is invoked only in the low-gap
region where it was empirically complementary. That route is representation-
specific and cannot be copied unchanged after retraining the encoders.

## Residual Attribution

Coverage-2M versus the anchor expert has opposing domain behavior:

- common average: +0.00129 eV;
- OOD average: -0.00579 eV;
- P8-hard average: +0.00849 eV.

Adding it to the existing anchor+repair ensemble improves common and OOD but
regresses P8-hard. Residual correlation remains approximately 0.991, so most
errors are shared. The largest regressions concentrate in:

- two-fragment molecules;
- MW > 700;
- rotatable bonds > 10;
- heavy atoms 35-50;
- true Gap between 2 and 4 eV;
- the highest base-expert disagreement quartile.

This pattern is consistent with loss of targeted-region retention and unstable
expert selection, not random label noise.

## Causes Ruled Out

- **Gap-label inconsistency:** all inspected datasets satisfy
  `Gap = LUMO - HOMO` to floating-point precision.
- **Simple duplicate contamination:** assembly reports show no CID or canonical
  SMILES overlap in appended sets.
- **Basic artifact misalignment:** accepted graph, prediction, embedding,
  source-index, row-count, finite-value, and checksum checks passed.
- **Ordinary overfitting:** larger runs improved internal validation and reduced
  the train-validation gap.
- **Parameter count by itself:** the encoders did not grow with dataset size,
  and the main regression can be reproduced by swapping representations under
  a fixed topology.
- **Larger data containing no useful signal:** multi-expert ensembles produce
  statistically meaningful common/P8-hard gains, but not an acceptable global
  accuracy/cost profile.

## Evidence Strength

### High confidence

- Direct paired common/OOD/P8-hard and PCQM comparisons.
- Exact dataset composition and label-identity checks.
- Completed-run metadata showing replay was absent.
- Fixed-topology encoder transplant and frozen 2D+3D fusion controls.

### Medium confidence

- Descriptor-level attribution to aromaticity, flexibility, size, and
  fragmentation. These are strong associations but do not identify a unique
  molecular mechanism.
- The 192-dimensional fusion bottleneck as a contributing factor. It is
  consistent with all controls but has not been isolated by a capacity-only
  ablation.

### Not supported

- "More parameters caused the regression."
- "All larger models are worse on every metric."
- "The 1M/2M datasets are corrupt."

## Decision Boundary

Do not launch another scale-up, specialist, router, distillation, or fusion run
until a future proposal specifies all of the following:

1. a fixed external promotion matrix before training;
2. explicit retention sampling for the targeted 500K corpus;
3. a route or fusion calibration learned for the new representations;
4. an inference-cost budget;
5. a causal ablation that changes one of data mixture, encoder, route, or
   fusion at a time.

The analysis artifacts in this directory are the authoritative comparison
record. Existing production registration remains unchanged.
