# Frozen retained-prediction attribution - 2026-10-08

User authorized both proposed local analyses and continuous execution. Owner:
codex/exp/k1-saved-attribution-20261008 from desktop ae0a9738. This question is
descriptive, not a new BN intervention or repetition of endpoint averaging.

## Question and identities

Accepted selected49 raw/clean and final60 raw/clean full50K predictions exist;
accepted aggregate BN gains are1.245/1.278meV, clean late deterioration0.787meV.
Ask whether selected BN gain is broad or concentrated in raw-error tails, and
whether the same molecules improve with BN yet worsen with late weights.
Alternatives include broad small shifts, high-error concentration and purely
mechanical correlation due to shared selected-clean absolute errors.

Freeze exact original prediction SHA256 before prospective planning; verify
again before weights_only CPU tensor loading. Include the contextual GPTrans
prediction for identity/aggregate checks only, not another fusion selection.
Require exactly source_idx[500000,550000) in original order, finite Gap eV
predictions/labels, identical target bytes, prior same16K train-only/dropout-off
BN configuration. Preserve the historical selection-used development role.
No graph/checkpoint access, new calibration, inference, training or remote work.

## Fixed calculations

Per-row absolute-error quantities: G49=raw49-clean49, G60=raw60-clean60,
Dclean=clean60-clean49, Draw=raw60-raw49. Positive G is improvement; positive D
is deterioration. Verify exact Draw=Dclean+G60-G49 within numeric tolerance.
Report whole-cohort means, paired-row95% intervals with the existing bootstrap
(1000 draws,seed20261008), positive/negative/tied fractions and gross gains/harms.

Fixed baseline raw49-error strata:0-50,50-90,90-95,95-99,99-100 percentiles;
ties break by source_idx. Report top1/5/10% concentration, net contribution,
gross benefit share and remaining-cohort gain. These are descriptive error-based
strata, not deployable inputs. Also describe equal-count target and G49 quartiles;
all slices are exploratory with no multiplicity-adjusted significance claim.
Record sign overlap, conditional late-harm fractions and Pearson correlations,
but do not interpret coupling as mechanism: G49 and Dclean both subtract clean49.
Use an exact covariance-term decomposition to display that shared-term coupling.
True target-based slicing never authorizes a deployment router.

No weight fitting, tuning, reselecting epochs, threshold optimization, router,
independent validation or official/protected-role access. Row bootstrap does not
estimate training stochasticity. Same BN estimation settings do not mean equal
buffers across different weights. Selected49 has same-role selection optimism.

## Resources and acceptance

One local CPU worker,600s ceiling, Torch/native thread environment4, recorded
wall/process CPU. Accelerator/queue N/A. Preparation/tests/RML/Git outside timers.
Publish prospective before tensor/metric analysis. Retain row deltas locally
with exact hash, aggregate arithmetic and immutable input metadata in Git.
NO_TRAIN closure, no training replay-ready, promotion, adoption or successor.
