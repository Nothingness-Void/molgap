# G1 follow-up: EMA correction passed; path additivity unsupported

On 2026-10-01, both isolated Kaggle3 arms completed 60 epochs and independently
passed saved-output acceptance. Each had 5,246,817 parameters, FP32, BS128,
46,860 optimizer steps and 5,998,080 sample presentations. The frozen G1
comparator was reused without retraining. Internal-development predictions
aligned on all 50,000 rows and exact targets. No protected role was consumed.

| Arm | Selected MAE (eV) | Gain versus G1 (eV) | Best epoch, zero-based | Material gate |
|---|---:|---:|---:|---|
| Frozen G1, EMA 0.9999 | 0.1508848917 | — | 59 | Comparator |
| G1 plus chemical path mean | 0.1511319991 | -0.0002471074 | 59 | Failed |
| G1, EMA 0.999 | 0.1442326291 | +0.0066522625 | 58 | Passed |

Authority: [saved-output acceptance](acceptance.json), [retained-trace analysis](analysis.json)
and the [prospective protocol](../../protocol.md). The material threshold was
strictly greater than 0.003 eV with a favorable paired interval; this did not
measure training-seed variability.

## Causal interpretation

The EMA arm's live training loss, live development MAE, learning rate, optimizer
steps and sample presentations matched every one of G1's 60 observations
exactly. Thus the measured endpoint gain came from the declared EMA/selected
weight intervention, not different live optimization or a new architecture.
The paired candidate-minus-reference 95% row-bootstrap interval was
[-0.0070457259, -0.0062624332] eV. This interval describes aligned molecules,
not independent training seeds or transfer to another dataset role.

The old EMA window lagged the converging live model: its development MAE fell
about 0.01246 eV in the last ten epochs, while the corrected arm fell only
0.00056085 eV. At 781 updates per epoch the analytically derived EMA half-life
is about 8.87 epochs for 0.9999 and 0.887 epochs for 0.999. With initialization
included in this uncorrected EMA recurrence, its theoretical remaining initial
state coefficient after 46,860 updates is about 0.922% versus effectively zero.
These are derived filter properties, not newly observed trace fields. Lag and
finite-horizon initialization retention were not separately intervened on.

The path arm's paired interval [-0.0007235882, +0.0011776906] eV crossed zero.
It neither established improvement nor a statistically stable negative effect.
Its terminal live MAE was also slightly worse (0.1449944675 versus 0.1442922503),
and its normalized training loss was higher, not lower. The specific additive
hypothesis therefore failed its release gate; these observations do not prove
chemical path encoding universally useless. Earlier separate G1/G2 positives
against a weaker original control did not establish complementary gains.

## Budget and evidence

The notebook used 3.63436 wall-hours and 7.26872 allocated T4-hours, within its
six-hour/twelve-card-hour cap. Mean epoch time was 212.52 s for the path arm and
185.95 s for the EMA arm. Two empty optional preflight text logs were preserved;
mandatory preflight JSON, runtime certificates, completion manifests, atomic
checkpoints and independent ten-epoch chunks passed validation.

Both observed comparisons are STRICT_CAUSAL and both actual candidate/reference
pairs entered the rebuilt Replay pool with complete 60-observation capability.
The path arm uses matched-recipe grouping. The EMA arm uses an explicit
EMA-intervention world retaining its different observed decay, not a fabricated
matched-EMA identity. Reused reference views do not create new runs or incurred
cost events. Original prospective and terminal receipts remain immutable.

## Bounded decision

Retain the EMA correction as a positive 100K training/selection finding; close
the tested path-additivity hypothesis under this contract. No extra seed,
500K/full training, protected-role evaluation or successor was authorized.
An separately authorized fixed500K frozen-weight audit would be a lower-cost
transfer check than immediate larger-scale training, but cannot prove a matched
500K training gain. Do not silently change the published historical contracts
or desktop recipes to EMA 0.999 based on this one screen.
