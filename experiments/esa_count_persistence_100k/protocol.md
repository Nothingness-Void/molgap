# Frozen ESA count-persistence experiment

Authorization: the human's2026-10-25 request to research historical RML and
submit a justified successor. Scope: one Kaggle2 T4x2 notebook, two isolated
seed42 ESA/4 arms, at most4 wall/session hours and8 allocated T4-hours.

## Single intervention

- Reference: existing ESA native backbone with no addon.
- Candidate: the same backbone plus existing `esa_count_readout/1`.
- Both: frozen accepted train100K plus Common-Dev50K input, identical initial
  base weights, BS128/drop-last, FP32/TF32off, LR3e-4, WD1e-5, normalized
  direct-Gap L1, AdamW, four-pass warmup/cosine,60 complete passes.
- Endpoint per arm:46,860 optimizer steps and5,998,080 sample presentations;
  no change of layer count, width, data, dropout, precision or loss.
- `prepared-v2` execution is shared and gated by fresh native numerical,
  exact-repeat/serialized-resume and accepted-backend checks for each arm.
  A CPU fixture cannot replace this native admission.

The same-run reference, target manifest, architecture-only intervention,
applicable roles, required trace axes and runtime qualification are frozen by
the existing compiler before upload. Arm processes have independent RNG,
optimizer and checkpoint/output paths, each with one visible device.

## Questions and stopping semantics

Primary: reference-minus-candidate selected-endpoint Common-Dev MAE. The
material triage gate for this experiment is0.003eV; it is explicitly frozen in
the input, not a universal V5 constant. A strict positive below that gate is
directional evidence only. Report final-live and selected endpoints separately,
the last10 complete-pass aligned differences, relative throughput and cost,
and paired fixed-row uncertainty. A row interval does not estimate seed variance.

Report target strata descriptively, especially the previously regressing<4eV
tail; target labels must never supply an inference-time router or calibration.
Without retained aligned size descriptors, do not claim size-caused subgroups.
Common-Dev was reused for discovery and is not an independent confirmation set.
No claim that this beats K1 or GPTrans follows from another benchmark contract.

Budget guard includes native startup, profiling and a600-second checkpoint
reserve. Estimate roughly2–3h from prepared ESA complete-pass timings; this is
not a new throughput certificate. If60 passes cannot finish, save a durable
STOP_FOR_COST result and do not extrapolate or shorten the original endpoint.
No automatic successor, continuation, extra seed,500K or protected evaluation.

## Evidence and closure

Reuse Session atomic selected/latest checkpoints, independently retrievable
chunks, predictions, canonical trace/context, observed loader/metric/selection
role audit, native certificates and full-allocation ledgers. Keep billed and
measured device cost separate; absent external billing stays missing.

Use existing saved-artifact acceptance without inference. Bind the actual
platform ID/version separately from the frozen logical attempt; close the
reference first, register its accepted reference bundle, then close candidate
and check actual replay-pool admission. A complete negative result is eligible.
Missing evidence blocks admission rather than being inferred or relabeled.
Only terminal acceptance can establish Replay-Ready, not this submitted plan.

The existing Luna B monitors the exact returned run every30 minutes. Healthy
QUEUED/RUNNING is silent; uncertainty is UNKNOWN; failure/completion creates
one idempotent handoff to A with no model or reasoning overrides.
