# GPTrans execution backend: paired forty-pass assay

Authorized on 2026-10-11: repair the audited execution overhead and submit a
Kaggle T4x2 original/optimized comparison, each for40 epochs. This is a V5
delivery/execution experiment, not a new architecture, protected evaluation,
500K/full nomination, or a rerun of the historical sixty-pass scientific claim.

## Frozen common conditions

- Accepted byte-identical fixed PCQM100K train and internal50K development;
  no graph construction, official validation/test-dev/challenge access.
- Retained local-bond GPTrans initialization and model,5,871,201 parameters.
- Seed42, physical BS128, FP32/noTF32, direct normalized Gap L1, unchanged
  features, dropout, gradient clipping, AdamW hyperparameters and EMA999.
- Both run the original60-epoch LR path through epochs0..39:31,240 updates,
  3,998,720 presentations. No rescaled cosine, early stop or extra seed.
- One isolated visible T4 per arm. Maximum wall7 hours /14 allocated T4 hours;
  no automatic extension, successor or resubmission after unknown POST state.

## Execution intervention

Original retains native dense preparation, loader lifetime, synchronous
development copies, per-tensor EMA and AdamW foreachFalse. Optimized derives
layout on CPU once per batch, packs embedded nodes once, shares local-edge
indices, persists deterministic loaders, buffers development predictions on
device and performs the unchanged CPU metric reduction once per pass.
The original main CPU RNG draw per development iterator is preserved. Optimized
uses tensor-list EMA and AdamW foreachTrue, explicitly recorded as different
execution backend identities; no claim of pre-established trajectory equality.

Both retain live and EMA development metrics every epoch, atomic native resume
state including RNG/optimizer/scheduler/EMA, selected predictions, canonical
trace and independently retrievable ten-epoch checkpoint chunks. The native
trainer legitimately emits a40-epoch partial manifest of its60-epoch recipe;
the execution-study completion separately declares the requested40-epoch assay
complete, never retrofits it as a60-epoch terminal result.

## Measurement and qualification

Both arms complete optimizer-inclusive deterministic preflight before either
formal training starts. Stage wall includes dataset loading and native artifact
publication. Per-epoch timings separately retain CPU optimizer enqueue,
synchronized development passes and checkpoint publication; raw enqueue time
is not GPU compute time. Native job allocation counts both T4s including idle
time, installation, preflight and collection. Preserve timing scopes.

Report complete stage wall ratio, first/steady epoch distributions, synchronized
evaluation and save cost, selected and final live/EMA MAE, prediction deltas and
paired-row intervals, target/row hashes, counters, RNG/resume bindings and
memory peaks. Speed qualification is >=1.10x complete stage throughput, with
optimized selected EMA MAE regression<=0.0005eV and paired-row95% upper bound
<=0.001eV. These prospectively chosen delivery tolerances do not replace any
historical strict numerical or architecture-promotion gate. Also report final
epoch accuracy; do not hide a bad final endpoint behind best-epoch selection.

Record per-arm prospective RML, trace, role, native cost, source/receipt,
mechanical acceptance and terminal delivery decision. Bind a fresh forty-pass
control at closure; the historical sixty-pass endpoint is context only.
Do not claim STRICT_CAUSAL architecture improvement or automatic Replay
admission for a noncausal delivery study. Verify actual admission capabilities
at closure rather than manufacture a qualification.
