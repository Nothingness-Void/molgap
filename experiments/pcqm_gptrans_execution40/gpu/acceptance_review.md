# Execution40 independent acceptance review — 2026-10-11

While physical version3 was pending, the local saved-output adapter was reviewed
against the frozen execution protocol. The original adapter checked runtime
certificates against self-reported completion fields without independently
binding those fields to the released contract provider.

The tightened adapter derives metadata only from the existing native scientific
field helper under the declared original/optimized backend. It first compares
the local providers' normalized source SHA with their immutable version3 archive
members. It then rejects any mismatching field, including seed, precision,
optimizer, schedule, target and execution backend. No model is constructed or
executed by this metadata derivation. The native60-horizon metadata remains
unchanged; actual40 completion/exposure is separately verified, never relabeled
as a completed60-pass experiment.

Worker binding now independently matches arm, study and package. Preflight and
native partial manifests must match source and fixed-data identities, with the
frozen architecture and EMA999. The retained runtime certificate must equal its
embedded copy. Epoch timing must be finite and positive. The API and CLI require
an explicit receipt, eliminating an accidental default to version1.

The focused backend/acceptance tests passed34 checks, including rehashed runtime
certificates with wrong batch or enabled TF32 and self-consistent but unfrozen
scientific fields. Both local contract providers matched the version3 released
source. No remote payload, training recipe, threshold, model, dataset or source
dataset version changed. No training or inference occurred locally. Real outputs
and terminal RML closure were still unavailable; these tests did not constitute
training acceptance or delivery qualification.

## Timing-report preparation

The saved-output report additionally separates first-pass timing from the39
post-first-pass observations, with inclusive quartiles, extrema and mean. Full
stage throughput uses the actual3,998,720 presentations divided by recorded
stage wall; it includes evaluation/loading/publication, not GPU-only training.
Synchronized evaluation and wrapped resume-checkpoint publication retain their
own scopes. CPU optimizer enqueue time is not relabeled GPU compute or used
to infer an unmeasured bottleneck. The report checks31,240 training batches and
80 live/EMA evaluation passes against the frozen40-pass scope.

Preflight allocated/reserved memory peaks are separately labeled. Training-stage
memory peaks were not emitted by the released worker and remain explicitly
measurement_missing; preflight observations do not fill that gap. Best live
metrics are trace-derived, while selected EMA metrics remain independently
recomputed from retained predictions. Final live and EMA endpoints remain
visible regardless of best-checkpoint selection.

The backend-focused synthetic suite passed29 tests after these additions. The
version3 remote payload was unchanged; final data-dependent qualification and
applicable RML closure still required terminal outputs.

## Terminal acceptance correction

After exact version3 output retrieval, independent Windows recomputation of
the epoch18 cosine LR differed from the Linux record by one floating-point
ULP. The adapter retained exact raw/canonical LR equality and exact exposure
counters, permitting at most2ULP only for local formula recomputation. Finite
values, changed schedules and raw/canonical disagreement remain fail-closed.
The focused suite passed35 tests; real saved-output acceptance then passed.
No remote source or scientific recipe changed. The terminal decision records
the separate unresolved canonical RML reference-binding limitation.
