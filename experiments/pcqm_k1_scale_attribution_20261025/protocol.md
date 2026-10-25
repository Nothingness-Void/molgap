# K1 scale-transfer and retained tail attribution — 2026-10-25

User authority: two hours from05:41:35UTC, ending07:41:35UTC. Desktop custody.
This diagnostic starts from verified desktop221d55a6 and interprets prior
accepted100K/500K evidence plus completed tail run140639445/version1. No model
promotion, new optimizer, training, official validation or test consumption.

## Question and evidence review

Explain which measured effects account for the apparent failure of100K gains
to persist at500K, and distinguish late parameter/function changes from BN
output-state effects in the just-completed tail pair. Prior complete local audit,
PairToken frozen portability, consistency500K, clean-fit500K and the tail owner
decision are evidence, not templates for another training run.

The best100K teacher package was not transferred intact:500K omitted teacher.
PairToken's selected frozen100K gain already shrank on500K development members.
Historical scalar and exact matched references differ. LIVE/EMA/clean-BN are
distinct states. These established confounds prevent a universal capacity claim.
Prior clean-fit showed improved sampled train fit but worse development at late
consistency weights. New tail WD0.001/0.01 both regress versus their initialization.
Its original protocol already requested fixed-TRAIN and BN attribution but the
accepted closure lacked locally located graph inputs. An existing user-downloaded
archive now provides the exact original16384TRAIN and50000development probes.

## Frozen diagnostic

Use the exact hash-bound Kaggle source package and family factory; no model
implementation rewrite. Use initial epoch49 LIVE state plus each tail selected
pass15 LIVE and finalpass20 LIVE. Predict each with original BN and independently
TRAIN-calibrated BN, Dropout disabled. No EMA reselection, head fitting, weight
averaging or outcome-dependent case addition. Preserve and verify all parameters,
buffers and eval mode after each calibration context. CPU FP32, four threads,
batch128, noTF32, deterministic seed42. Different CPU/GPU runtimes must pass
retained original prediction reproduction before interpreting interventions:
max absolute delta<=1e-4eV and MAE delta<=1e-6eV for initial and selected states.
Final states must reproduce their retained final trace MAE to1e-6eV.

Calibrate on exactly the original NumPyseed20261008 uniform16384 of TRAIN0:500000
in original order. Evaluate the same TRAIN sample and complete development
500000:550000. Calibration uses features, not labels; loading graphs reads labels,
train metrics use that same in-sample cohort. Development is selection-consumed.
No official validation/test/common/OOD inputs. No substitute prefix or new cohort.

Contrasts: each tail endpoint versus initial in each BN state/role; BN intervention
within each fixed state; selected-to-final within each arm; WD1e-2 versus1e-3.
Existing paired-row bootstrap1000/seed42; positive error delta means worse.
Intervals describe rows conditional on selected weights, not seed variability,
independent generalization or multiple-comparison corrected nomination.

The outer worker has a1200s wall limit and per-batch deadline. No automatic retry
or remote fallback. Save partial failure honestly. Exact required source/probes
may be extracted from retained local archives toD:, no network download. Report
actual worker CPU/wall costs separately from preparation/analysis/Git. Unknown
costs remain unknown. Stop all newly launched work before07:41:35UTC.

## Interpretation and closure

If BN repair removes tail regression, identify output-state sensitivity, not a
training-time cause. If regression persists while clean TRAIN fit improves,
identify a conditional generalization tradeoff; optimizer reset, loss and LR
changes are bundled versus initialization, so their individual causes remain
unidentified. If train and dev worsen, label whole-recipe forgetting/optimization
compatibility only, not a unique mechanism. No threshold tuning or next training
is justified just by a favorable post-hoc point.

Close through existing local frozen-diagnostic RML finalizer asNO_TRAIN. This is
new inference evidence, not retroactive strict training replay. Keep actual
strict-target/missing-reference blockers of the trained pair unchanged. Audit
only reproduced infrastructure defects and verify affected paths once.
