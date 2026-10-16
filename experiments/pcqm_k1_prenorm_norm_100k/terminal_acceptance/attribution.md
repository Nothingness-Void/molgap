# Pre-norm100K terminal attribution

Dated2026-10-16; saved artifacts only. The [frozen protocol](../protocol.md)
owns the gate; [analysis JSON](analysis_summary.json) owns computed values and
[curves](development_curves.png) show observed optimizer-step trajectories.
No new inference, training, protected-role access or post-result gate change.

## Observed result

The same-job selected Live delta is Ada-minus-BN +2.181843meV
(paired-row95%[+1.196495,+3.161183]meV). Both noninferiority and material-gain
gates fail.40 observations per arm end at31240 steps /3998720 presentations.
Completion, aligned finite50000-row predictions, selected model and full resume
state pass the existing family-session inspector. Native repeat/serialized
resume checks have their separate per-arm reports; scheduler completion alone
is not their substitute. The original Post-Norm decision remains unchanged.

## Exposure, optimization and state

- Unequal exposure, interrupted training and accidentally selecting EMA are
  contradicted by the retained counters and Live checkpoint bindings.
- From pass30 to40, Live MAE improves2.929meV for Pre-BN and3.619meV for
  Pre-Ada. At common pass40 the Ada deficit is2.055meV; at separately selected
  Live endpoints it is2.182meV. Late improvement remains, so these curves do
  not establish either adequate absolute training or a need for more exposure.
  They do not support blaming this outcome on an incomplete scheduled run.
- Secondary best EMA is0.144870311eV/pass36 for Pre-BN and
  0.147190109/pass40 for Pre-Ada. This observed state choice does not rescue
  the candidate. No EMA predictions/paired uncertainty or EMA promotion is claimed.
- Final online train MAE, converted from normalized Gap, is0.102354eV for
  BN and0.102014eV for Ada. These are changing-weight training measurements
  with dropout0.05, not fixed-checkpoint training-cohort evaluation. Their
  difference from development is not a rigorous generalization gap. Underfit,
  overfit, a unique harmful module, and the cause of the development deficit
  remain `insufficient_evidence` without the appropriate discriminator.

## Historical placement context

| Retained selected Live cell | MAE(eV) |
|---|---:|
| Post-BN | 0.141294461 |
| Post-Ada | 0.145184176 |
| Pre-BN | 0.144917520 |
| Pre-Ada | 0.147099363 |

All four retained predictions have identical rows/targets and matched terminal
update/sample counts. Pre-BN minus Post-BN is +3.623meV; Pre-Ada minus Post-Ada
is +1.915meV. Source/job changes and the added final node normalization confound
placement; these are descriptive cells, not a causal2x2 factorial estimate.
The measured pre-norm package did not rescue the adaptive candidate. This
local adaptation is not a full PPGT implementation and cannot establish a
universal ranking of modern normalization methods.

## Cost and evidence limits

Candidate training invocation wall is6761.064s versus6338.445s for BN,
6.67% longer in this job. Whole-entry assigned cost is3.935686585 T4-device
hours, including setup, qualification, evaluation and idle allocated slots.
This is not device-busy time or an account-quota/billing conversion. Native
worker time, shared entry wall and unknown queue/platform-release costs stay
separate. The raw output manifests retain their missing allocated-cost field;
accepted slot accounting is sourced separately from the measured entry record.

The frozen mode policy did not enable `process_timing`, so the shared writer
used the observation path without checkpoint identities or timing. Separately,
`pass_finished` did not forward learning rates into canonical fields. The
standard human-readable epoch logs contain learning rates/cursors, but a cursor
is not a checkpoint hash. Preserve
the raw manifests/traces, and mark strict comparison/replay unavailable.
Correcting this publication coupling for a future run is infrastructure work;
it is not justification to replay training merely to repair bookkeeping.

## Disposition

Close the exact candidate negative; do not advance this package to500K or
repeat the unchanged40-pass recipe. Preserve control/candidate evidence in RML
with truthful exclusions and archive rejected implementation under branch
routing. The missing mechanism discriminator is fixed-cohort, same-checkpoint
fit/activation or gradient evidence under a separately scoped diagnostic; none
is fabricated or executed during this acceptance. Query this result before
selecting another normalization proposal.
