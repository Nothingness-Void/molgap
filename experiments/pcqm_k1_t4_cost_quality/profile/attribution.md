# Native T4 attribution - 2026-10-09

Disposition: NO_TRAIN execution-cost diagnostic; parent byte acceptance and
canonical closure remain separate. [Decision](terminal_decision.md) owns the
timings, cost scope, identities and exclusions; [result](../../../platforms/_records/kaggle/training/k1_native_t4_profile_s42_v1/profile/result.json)
and [runtime](../../../platforms/_records/kaggle/training/k1_native_t4_profile_s42_v1/profile/runtime.json)
are the retained observations, not an accuracy comparison.

Matched selected-state single versus coefficient0 mean2 scratch execution
supports a37.630898% median step-time reduction in this bounded native T4
sample. Six synchronized mean2 phase means (seconds): loader0.000660297,
H2D0.000842069, forward+loss0.119542521, backward0.119010891,
clip0.007527729, optimizer0.038463208. Forward/loss and backward dominate
these instrumented samples; loader wait is small here. Instrumentation
synchronizes phases and includes an unused disagreement computation, so it is
not an independent throughput estimate or a proof of whole-run savings.

Four train-member eval batches total0.523306494s; localFS atomic scratch write
0.057648949s,14805295bytes. Neither establishes full50K development time or
remote upload time. Entry allocation includes setup; worker/phase/eval windows
are nested and must not be summed again into device-hours.

No infrastructure failure appears in complete output artifacts. Retained
scheduler COMPLETE metadata binds the exact kernel/version; parent reports
retrieval, and this adapter makes no remote query. Artifact verification is
mechanical acceptance, not scientific quality qualification.

Quality causality is `insufficient_evidence`: single/mean2 differ in stochastic
dropout and BN exposure. No matched trained endpoint, development cohort,
quality trace or full-run counterfactual was measured. Underfitting,
overfitting, exposure shortage and module harm cannot be attributed here.
Gradient relation/norm was not measured. Scratch updates are real but not
scientific training; only4096 prebuilt training members were decoded in this
attempt, and no development/protected roles were accessed.

Missing discriminator: a separately authorized prospective matched cost-quality
comparison, with native cumulative costs and quality acceptance. This diagnostic
does not release it, require a reference retrain, or justify global extrapolation.
