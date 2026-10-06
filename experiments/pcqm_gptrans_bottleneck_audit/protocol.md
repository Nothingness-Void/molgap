# Frozen GPTrans bottleneck audit

On 2026-10-06 the user authorized the cheapest follow-up to the accepted capacity
and pair-transition interpretations. This is one server-owned NO_TRAIN diagnostic,
not another architecture screen, weight repair, or scale-up release.

## Questions and alternatives

1. Does final Gap loss reach each local/pair branch? The pair-transition terminal
   audit found a disconnected final real-pair branch; an isolated module test
   did not establish end-to-end reachability.
2. Do real-bond updates become larger relative to their input late in training,
   especially on higher-degree molecules? Better fitting with stalled development
   could alternatively reflect ordinary finite-data fitting, not amplitude growth.
3. Does the selected local checkpoint preserve its paired advantage on a fixed
   later cohort without optimization? Cohort dependence is different from an
   optimizer/exposure problem.

Authorities: [local/scale interpretation](../pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md),
[capacity falsifier](../pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md),
[pair-transition audit](../pcqm_gptrans_pair_transition_100k/gpu/results/interpretation.md),
and [accepted EMA portability](../pcqm_gptrans_ema_portability/attempt_v4/decision.md).

## Frozen scope

- Accepted reference/local EMA snapshots at zero-based epochs19 and59 are
  inspected on the same first512 original-development molecules, BS128.
- Selected pair-transition weights receive the same final normalized Gap-L1
  derivative probe. Models remain in eval mode: no dropout, optimizer, clipping,
  EMA update, scheduler, train-mode buffer update, or checkpoint reselection.
- Per-molecule/per-layer real-node input, update and output RMS, input gradients,
  branch-return gradients, parameter gradient norms, atom count and degree are
  retained. A zero observed gradient is not generally structural proof; static
  reachability explains the final pair branch. Gradient probes are not measured
  training gradients or clipping frequencies.
- The local selected checkpoint must first reproduce all50000 accepted original
  development predictions within max-absolute0.001eV and MAE0.0001eV.
- Only after reproduction, infer10000 rows selected without replacement by
  NumPy RandomState42 from fixed500K internal development[500000,550000), sorted
  by source index. Reuse hash-bound reference predictions from the accepted
  frozen EMA audit on exactly those rows. No reference retraining/inference for
  this endpoint. This cohort has already been used for research; not a sealed
  confirmation or evidence of500K optimization.
- FP32/noTF32, deterministic seed42, physical inference batch128 (evaluation
  may retain its last partial batch). No geometry enters the model.
- Two isolated T4 workers separate derivative diagnostics from reproduction/
  portability. Both are parts of one physical diagnostic, not candidate training.
  Allocation cap1800seconds/1allocatedT4-hour; estimated10–20wall minutes. Setup,
  idle time and failures count. Queue/teardown outside observation are unavailable.
- Atomic diagnostic batches,5000-row prediction chunks, model/source/transform/
  cohort hashes, role events, unchanged-state proofs and native allocation ledger
  are mandatory. Retained checkpoints are repackaged mechanically as EMA-only
  assets with source-checkpoint SHA and EMA-state SHA; no local model execution.

## Decisions

Saved endpoint comparison is PAIRED_ENDPOINT, not STRICT_CAUSAL for this new
noncausal run. Gradient/strength observations are CONTEXT_ONLY. RML terminal
outcome is NO_TRAIN; no fabricated training trace or training Replay pair.
Subgroups are frozen input-only degree bins(maxdegree<=2,=3,>=4) and atom-count
bins(<=10,11–20,>20). No target-derived expert selection, threshold tuning,
Router, geometry, extra seed,500K/full training, or automatic successor.

Positive later-cohort paired gain with a favorable row interval permits only
controller consideration of a separately frozen mechanism experiment. An
adverse/uncertain result closes unchanged local-addon scale-up. Amplitude
correlation alone does not authorize a gate or establish causality.

## Release and acceptance

Reuse immutable graph loading, target-transform validation, source packaging,
frozen release checks, Kaggle platform submission, atomic IO, saved-error
analysis, allocation accounting, and NO_TRAIN RML finalization. New code is
limited to derivative/activation capture and this scoped adapter. A release must
bind real retained assets and a prospective plan. The existing Luna B monitors
the actual returned job/version silently and wakes the existing A on terminal
state; no new conversation or heartbeat is created.
